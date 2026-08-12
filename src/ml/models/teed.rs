//! Edge Detection — DexiNed (and TEED-compatible)
//!
//! # Model Details
//!
//! - **Current model**: DexiNed (Dense Extreme Inception Network for Edge Detection)
//!   - Source: `SStahan/ONNX-exports-dpt-seg-edge-face` on HuggingFace
//!   - Size: ~134 MB
//!   - Input: RGB float32 NCHW, ImageNet-normalized, fixed 512×512
//!   - Output: `[1, 1, H, W]` — edge probability map, values in [0, 1]
//!
//! - **Also compatible**: TEED (Tiny and Efficient Edge Detector)
//!   - Size: <1 MB, dynamic input shape
//!   - The module auto-detects whether the loaded model has fixed or dynamic
//!     input dimensions and handles both cases.
//!
//! # Why a perceptual edge detector instead of depth-derived edges
//!
//! Depth discontinuities only capture silhouette edges. A perceptual edge
//! detector is trained on edge ground truth and detects within-object
//! boundaries: eyebrow lines, eyelids, fabric folds, hair strands. These are
//! exactly the contours a pixel artist would hand-draw and the ones most
//! damaged by downsampling.
//!
//! # Fixed vs. dynamic input
//!
//! DexiNed is exported with a fixed 512×512 input shape. When the input image
//! is not 512×512, we resize to 512×512 for inference, then bilinearly
//! upsample the edge map back to the original resolution.
//!
//! TEED (if used) supports dynamic input and runs at the exact original
//! resolution — no rescaling needed.
//!
//! The module queries the ONNX model's input dimensions at load time to
//! determine which mode to use. A dimension value of `-1` indicates dynamic.

use crate::ml::preprocess::{
    normalize_min_max, upsample_map_to_original, PreprocessConfig, preprocess,
};
use crate::ml::session::{SessionManager, ModelType};
use anyhow::{anyhow, Result};
use image::{DynamicImage, GenericImageView};
use std::path::PathBuf;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Edge detector using an ONNX edge detection model (DexiNed or TEED).
///
/// On a dynamic-axes export (TEED) the model runs at the exact input resolution.
/// On a fixed export (DexiNed, 512×512) it resizes to the fixed size and
/// bilinearly upsamples the result.
///
/// The struct is named `TeedEdgeDetector` for historical reasons — it was
/// originally written for TEED. It now supports any ONNX edge detector with
/// a `[1, 3, H, W]` input and `[1, 1, H, W]` / `[1, H, W]` / `[H, W]` output.
pub struct TeedEdgeDetector {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
    input_name: String,
    output_name: String,
    /// `None` → model accepts dynamic input (TEED).
    /// `Some((w, h))` → model has fixed input (DexiNed: 512×512); we resize and upsample.
    fixed_dims: Option<(u32, u32)>,
}

impl TeedEdgeDetector {
    /// Load the ONNX model using SessionManager for GPU support.
    ///
    /// Queries the model's input dimensions to determine whether it uses
    /// fixed or dynamic input shapes.
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session_manager = crate::ml::session::global_session_manager();
        let model_session = session_manager.get_or_load(&model_path.to_path_buf(), ModelType::EdgeDetection)?;

        let session = model_session.session.lock().unwrap();
        let input_name  = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        // ── Detect fixed vs. dynamic input dimensions ───────────────────────
        // The input shape is [N, C, H, W]. If H and W are both >= 0, the model
        // has fixed input. If either is -1, the model accepts dynamic input.
        let fixed_dims = Self::detect_fixed_dims(&session);

        drop(session);

        match fixed_dims {
            Some((w, h)) => log::info!(
                "Edge detector loaded (fixed input {}×{}, backend: {:?})",
                w, h, session_manager.backend()
            ),
            None => log::info!(
                "Edge detector loaded (dynamic input, backend: {:?})",
                session_manager.backend()
            ),
        }

        Ok(Self {
            session_manager,
            model_path: model_path.to_path_buf(),
            input_name,
            output_name,
            fixed_dims,
        })
    }

    /// Query the model's input dimensions from the ORT session.
    ///
    /// Returns `Some((width, height))` if both spatial dimensions are fixed,
    /// or `None` if either is dynamic (-1).
    fn detect_fixed_dims(session: &ort::session::Session) -> Option<(u32, u32)> {
        use ort::value::ValueType;

        let input = session.inputs().first()?;
        let dtype = input.dtype();

        if let ValueType::Tensor { shape, .. } = dtype {
            // Shape is [N, C, H, W] — dims[2] = H, dims[3] = W
            if shape.len() >= 4 {
                let h = shape[2];
                let w = shape[3];
                if h > 0 && w > 0 {
                    return Some((w as u32, h as u32));
                }
            }
        }
        None
    }

    /// Detect edges in `image`.
    ///
    /// Returns a flat `Vec<f32>` in row-major order at the *original* image's resolution.
    /// Values are normalized to [0.0, 1.0] — 1.0 = strong edge.
    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (orig_w, orig_h) = image.dimensions();
        let (target_w, target_h) = self.fixed_dims.unwrap_or((orig_w, orig_h));

        let cfg = PreprocessConfig::imagenet(target_w, target_h);
        let (tensor, scale) = preprocess(image, &cfg)?;

        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, target_h as i64, target_w as i64],
            tensor,
        ))?;

        let model_session = self.session_manager.get_or_load(&self.model_path, ModelType::EdgeDetection)?;
        let mut session = model_session.session.lock().unwrap();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow!("Edge detector produced no output"))?;

        let (shape, raw) = output.try_extract_tensor::<f32>()?;

        let (map_h, map_w) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            _ => return Err(anyhow!("Unexpected edge detector output shape: {:?}", shape)),
        };

        let map_size = map_h * map_w;
        if raw.len() < map_size {
            return Err(anyhow!("Edge detector output too small: expected {}, got {}", map_size, raw.len()));
        }
        let map_slice = &raw[raw.len() - map_size..];

        // If the model ran at original resolution, skip upsampling
        let mut result = if map_w == orig_w as usize && map_h == orig_h as usize {
            map_slice.to_vec()
        } else {
            upsample_map_to_original(map_slice, map_w, map_h, &scale)
        };

        normalize_min_max(&mut result);
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder edge detector — Sobel gradient on luminance.
/// Produces reasonable-looking edges for testing without a real model.
pub struct PlaceholderEdgeDetector;

impl PlaceholderEdgeDetector {
    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let gray = image.to_luma8();
        let (w, h) = gray.dimensions();
        let (wi, hi) = (w as usize, h as usize);

        let lum: Vec<f32> = gray.pixels().map(|p| p[0] as f32 / 255.0).collect();

        let mut edges = vec![0.0f32; wi * hi];

        for y in 1..hi - 1 {
            for x in 1..wi - 1 {
                let gx =
                    -lum[(y-1)*wi+(x-1)] - 2.0*lum[y*wi+(x-1)] - lum[(y+1)*wi+(x-1)]
                    +lum[(y-1)*wi+(x+1)] + 2.0*lum[y*wi+(x+1)] + lum[(y+1)*wi+(x+1)];
                let gy =
                    -lum[(y-1)*wi+(x-1)] - 2.0*lum[(y-1)*wi+x] - lum[(y-1)*wi+(x+1)]
                    +lum[(y+1)*wi+(x-1)] + 2.0*lum[(y+1)*wi+x] + lum[(y+1)*wi+(x+1)];
                edges[y*wi+x] = (gx*gx + gy*gy).sqrt();
            }
        }

        normalize_min_max(&mut edges);
        Ok(edges)
    }
}

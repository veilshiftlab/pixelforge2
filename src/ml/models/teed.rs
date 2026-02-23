//! TEED: Tiny and Efficient Edge Detector
//!
//! # Model Details
//!
//! - **Source**: `xavysp/teed` on HuggingFace — export to ONNX with dynamic axes
//! - **Size**: <1 MB (≈58 K parameters)
//! - **Architecture**: Lightweight fully-convolutional CNN — natively handles any H×W
//! - **Input**: RGB float32 NCHW, ImageNet-normalized
//! - **Output**: `[1, 1, H, W]` — edge probability map, values in [0, 1]
//!
//! # Why TEED instead of depth-derived edges
//!
//! Depth discontinuities only capture silhouette edges. TEED is trained on perceptual
//! edge ground truth and detects within-object boundaries: eyebrow lines, eyelids,
//! fabric folds, hair strands. These are exactly the contours a pixel artist would
//! hand-draw and the ones most damaged by downsampling.
//!
//! # Dynamic input
//!
//! With a dynamic-axes ONNX export, TEED runs at the exact original resolution — no
//! rescaling, no bilinear upsampling, no spatial distortion in the edge map.
//!
//! # Export recipe (one-time Python, run locally or on Colab)
//!
//! ```python
//! import torch
//! # pip install git+https://github.com/xavysp/TEED
//! from teed import TED
//!
//! model = TED()
//! model.load_state_dict(torch.load("ted.pth", map_location="cpu"))
//! model.eval()
//!
//! dummy = torch.zeros(1, 3, 512, 512)
//! torch.onnx.export(
//!     model, dummy, "teed.onnx",
//!     input_names=["input"],
//!     output_names=["edges"],
//!     dynamic_axes={"input":  {2: "height", 3: "width"},
//!                   "edges":  {2: "height", 3: "width"}},
//!     opset_version=17,
//! )
//! ```

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

/// Edge detector using TEED ONNX model.
///
/// On a dynamic-axes export (recommended) the model runs at the exact input resolution.
/// On a fixed export it falls back to the export size and bilinearly upsamples the result.
pub struct TeedEdgeDetector {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
    input_name: String,
    output_name: String,
    /// `None` → model accepts dynamic input (preferred).
    /// `Some((w, h))` → model has fixed input; we'll resize and upsample.
    fixed_dims: Option<(u32, u32)>,
}

impl TeedEdgeDetector {
    /// Load the ONNX model using SessionManager for GPU support.
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session_manager = crate::ml::session::global_session_manager();
        let model_session = session_manager.get_or_load(&model_path.to_path_buf(), ModelType::DepthEstimation)?; // TEED doesn't have a dedicated ModelType yet,  using DepthEstimation as proxy
        
        let session = model_session.session.lock().unwrap();
        let input_name  = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();
        drop(session);

        // We can't introspect dims from ORT 2.x before the first run, so we
        // assume dynamic. If the model is actually fixed-size, the first
        // inference will succeed (we send whatever the image size is) or fail
        // with a clear shape mismatch that surfaces immediately.
        let fixed_dims: Option<(u32, u32)> = None;

        log::info!("TEED loaded (dynamic input assumed): input_name='{}' (backend: {:?})", input_name, session_manager.backend());

        Ok(Self { session_manager, model_path: model_path.to_path_buf(), input_name, output_name, fixed_dims })
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

        let model_session = self.session_manager.get_or_load(&self.model_path, ModelType::DepthEstimation)?;
        let mut session = model_session.session.lock().unwrap();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow!("TEED produced no output"))?;

        let (shape, raw) = output.try_extract_tensor::<f32>()?;

        let (map_h, map_w) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            _ => return Err(anyhow!("Unexpected TEED output shape: {:?}", shape)),
        };

        let map_size = map_h * map_w;
        if raw.len() < map_size {
            return Err(anyhow!("TEED output too small: expected {}, got {}", map_size, raw.len()));
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
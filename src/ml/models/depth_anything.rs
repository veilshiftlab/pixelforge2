//! Depth-Anything V2: Monocular Depth Estimation
//!
//! # Model Details
//!
//! - **Recommended**: `depth-anything-v2-small` (~100 MB) — `onnx-community/depth-anything-v2-small`
//! - **Input**: RGB float32 NCHW, 0–1 normalized, default 518×518 (dynamic if re-exported)
//! - **Output**: `[1, H, W]` or `[H, W]` — relative depth (higher = farther for ViT models)
//!
//! # Output semantics
//!
//! Depth-Anything produces *relative* depth. We normalize to [0, 1] post-inference
//! (0 = nearest, 1 = farthest).
//!
//! # Coordinate mapping
//!
//! The output is bilinearly upsampled to original resolution using per-axis scale
//! factors from [`ResizeScale`]. This is essential for non-square inputs — a 256×512
//! image has different X/Y resize ratios when squashed to a square model input, and
//! using a single shared scale factor introduces spatial distortion in the depth map.

use crate::ml::preprocess::{
    normalize_min_max, upsample_map_to_original, PreprocessConfig, preprocess,
};
use crate::ml::session::{SessionManager, ModelType};
use anyhow::{anyhow, Result};
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default input size (ViT-S/B/L all use 518; patch size 14 × 37 = 518)
const DEFAULT_INPUT_SIZE: u32 = 518;

// ─────────────────────────────────────────────────────────────────────────────
// Estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Depth estimator using Depth-Anything V2 ONNX model.
pub struct DepthAnythingEstimator {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
    input_name: String,
    output_name: String,
    /// Model input width (static from export, or DEFAULT_INPUT_SIZE for dynamic models)
    pub model_w: u32,
    /// Model input height
    pub model_h: u32,
}

impl DepthAnythingEstimator {
    /// Load the ONNX model using SessionManager for GPU support.
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session_manager = crate::ml::session::global_session_manager();
        let model_session = session_manager.get_or_load(&model_path.to_path_buf(), ModelType::DepthEstimation)?;
        
        let session = model_session.session.lock().unwrap();
        let input_name  = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();
        drop(session);

        log::info!("Depth-Anything V2 loaded: input_name='{}' (backend: {:?})", input_name, session_manager.backend());

        Ok(Self {
            session_manager,
            model_path: model_path.to_path_buf(),
            input_name,
            output_name,
            model_w: DEFAULT_INPUT_SIZE,
            model_h: DEFAULT_INPUT_SIZE,
        })
    }

    /// Estimate a depth map for `image`.
    ///
    /// Returns a flat `Vec<f32>` in row-major order at the *original* image's resolution.
    /// Values are normalized to [0.0, 1.0] where 0 = nearest, 1 = farthest.
    pub fn estimate(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let cfg = PreprocessConfig::unit(self.model_w, self.model_h);
        let (tensor, scale) = preprocess(image, &cfg)?;

        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, self.model_h as i64, self.model_w as i64],
            tensor,
        ))?;

        let model_session = self.session_manager.get_or_load(&self.model_path, ModelType::DepthEstimation)?;
        let mut session = model_session.session.lock().unwrap();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow!("Model produced no depth output"))?;

        let (shape, raw) = output.try_extract_tensor::<f32>()?;

        // Supported output shapes: [1,H,W], [H,W], [1,1,H,W]
        let (map_h, map_w) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            _ => return Err(anyhow!("Unexpected depth output shape: {:?}", shape)),
        };

        let map_size = map_h * map_w;
        if raw.len() < map_size {
            return Err(anyhow!(
                "Depth output too small: expected {} values, got {}",
                map_size, raw.len()
            ));
        }
        // Strip leading batch/channel dims (all size 1) by taking the last H*W elements
        let map_slice = &raw[raw.len() - map_size..];

        let mut upsampled = upsample_map_to_original(map_slice, map_w, map_h, &scale);
        normalize_min_max(&mut upsampled);

        // Depth-Anything V2 outputs disparity (higher value = closer to camera).
        // Our pipeline convention is depth (0 = nearest, 1 = farthest), so we
        // invert after normalization. Without this, background pixels (low
        // disparity) are treated as "near" and get foreground treatment.
        for d in &mut upsampled {
            *d = 1.0 - *d;
        }

        Ok(upsampled)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder depth estimator — radial gradient blended with inverted luminance.
/// Gives visually plausible depth for testing without a real model.
pub struct PlaceholderDepthEstimator;

impl PlaceholderDepthEstimator {
    pub fn estimate(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        use image::GenericImageView;

        let (w, h) = image.dimensions();
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        let max_r = cx.hypot(cy).max(1.0);

        // Convert to luma once, index by flat offset
        let gray = image.to_luma8();
        let luma: Vec<f32> = gray.pixels().map(|p| p[0] as f32 / 255.0).collect();

        let mut depth: Vec<f32> = (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let r = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                    1.0 - (r / max_r * 0.5).min(1.0)
                })
            })
            .zip(luma.iter())
            .map(|(radial, &lum)| (radial * 0.7 + (1.0 - lum) * 0.3).clamp(0.0, 1.0))
            .collect();

        normalize_min_max(&mut depth);
        Ok(depth)
    }
}
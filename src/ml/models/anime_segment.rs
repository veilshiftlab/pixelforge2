//! Foreground/Background Segmentation — AnimeSegment (IS-Net)
//!
//! # Model Details
//!
//! - **Model**: AnimeSegment (IS-Net — Information Segmentation Network)
//!   - Source: `skytnt/anime-segmentation` on HuggingFace
//!   - Size: ~43 MB
//!   - Input: RGB float32 NCHW, normalized to [0,1], dynamic resolution
//!   - Output: `[1, 1, H, W]` — foreground probability map, values in [0, 1]
//!     where 1.0 = foreground (character), 0.0 = background
//!
//! # Why a dedicated anime segmentation model
//!
//! Generic segmentation models (BiSeNet, DeepLab) are trained on
//! photographic content and fail on anime/cel-shaded images — they can't
//! distinguish flat-color clothing from background, and they misclassify
//! stylized hair as "not person". AnimeSegment is trained specifically on
//! anime illustrations and reliably separates character from background
//! regardless of art style.
//!
//! # Usage in the pipeline
//!
//! The mask is consumed by `depth_to_flat::classify_background` to
//! accurately identify background pixels for desaturation treatment.
//! This replaces the SLIC-based heuristic (mean_depth > p70 AND
//! touches_border) which misclassified dress regions as background on
//! flat-cel-shaded images.
//!
//! # Fallback
//!
//! When the model is not available (not downloaded, failed to load), the
//! pipeline falls back to the SLIC-based `classify_background`. This is
//! handled in `depth_to_flat.rs` by checking `ml_results.segmentation_mask`.

use crate::ml::preprocess::{upsample_map_to_original, PreprocessConfig, preprocess};
use crate::ml::session::{SessionManager, ModelType};
use anyhow::{anyhow, Result};
use image::{DynamicImage, GenericImageView};
use std::path::Path;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Segmenter
// ─────────────────────────────────────────────────────────────────────────────

/// Foreground/background segmenter using the AnimeSegment ONNX model.
///
/// The model accepts dynamic input resolution. We cap the inference
/// resolution at 1024×1024 to balance speed and quality — larger inputs
/// give slightly better edge detail but take significantly longer. The
/// mask is bilinearly upsampled back to the original image resolution.
pub struct AnimeSegmenter {
    session_manager: Arc<SessionManager>,
    model_path: std::path::PathBuf,
    input_name: String,
    output_name: String,
}

/// Maximum inference dimension. The model supports dynamic input, but we cap
/// it to keep inference time reasonable (~100ms on GPU, ~300ms on CPU at 1024²).
const MAX_INFER_DIM: u32 = 1024;

impl AnimeSegmenter {
    /// Load the ONNX model using SessionManager for GPU support.
    pub fn new(model_path: &Path) -> Result<Self> {
        let session_manager = crate::ml::session::global_session_manager();
        let model_session = session_manager.get_or_load(
            &model_path.to_path_buf(),
            ModelType::Segmentation,
        )?;

        let session = model_session.session.lock().unwrap();
        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        log::info!(
            "AnimeSegment loaded (input='{}', output='{}', backend: {:?})",
            input_name, output_name, session_manager.backend()
        );

        drop(session);

        Ok(Self {
            session_manager,
            model_path: model_path.to_path_buf(),
            input_name,
            output_name,
        })
    }

    /// Segment `image` into a foreground probability mask.
    ///
    /// Returns a flat `Vec<f32>` in row-major order at the *original* image's
    /// resolution. Values are in [0.0, 1.0] — 1.0 = foreground (character),
    /// 0.0 = background.
    pub fn segment(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (orig_w, orig_h) = image.dimensions();

        // Cap inference resolution to MAX_INFER_DIM. The model accepts dynamic
        // input, so we use the smaller of (orig, MAX_INFER_DIM) on each axis.
        let scale = MAX_INFER_DIM as f32 / orig_w.max(orig_h) as f32;
        let (target_w, target_h) = if scale < 1.0 {
            (
                (orig_w as f32 * scale).round() as u32,
                (orig_h as f32 * scale).round() as u32,
            )
        } else {
            (orig_w, orig_h)
        };

        // AnimeSegment uses plain [0,1] normalization (no ImageNet shift).
        let cfg = PreprocessConfig::unit(target_w, target_h);
        let (tensor, scale_info) = preprocess(image, &cfg)?;

        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, target_h as i64, target_w as i64],
            tensor,
        ))?;

        let model_session = self.session_manager.get_or_load(
            &self.model_path,
            ModelType::Segmentation,
        )?;
        let mut session = model_session.session.lock().unwrap();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow!("AnimeSegment produced no output"))?;

        let (shape, raw) = output.try_extract_tensor::<f32>()?;

        // Output shape is [1, 1, H, W] or [1, H, W] or [H, W].
        let (map_h, map_w) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            _ => return Err(anyhow!(
                "Unexpected AnimeSegment output shape: {:?}", shape
            )),
        };

        let map_size = map_h * map_w;
        if raw.len() < map_size {
            return Err(anyhow!(
                "AnimeSegment output too small: expected {}, got {}",
                map_size, raw.len()
            ));
        }
        let map_slice = &raw[raw.len() - map_size..];

        // If the model ran at original resolution, skip upsampling.
        let result = if map_w == orig_w as usize && map_h == orig_h as usize {
            map_slice.to_vec()
        } else {
            upsample_map_to_original(map_slice, map_w, map_h, &scale_info)
        };

        // Clamp to [0, 1] — the model output is already a probability, but
        // floating-point inference can produce values slightly outside range.
        let result: Vec<f32> = result.into_iter()
            .map(|v| v.clamp(0.0, 1.0))
            .collect();

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder segmenter — classifies everything as foreground.
///
/// When the model is not available, the pipeline falls back to SLIC-based
/// background classification, so this placeholder just returns an all-ones
/// mask (everything foreground) which causes `classify_background` to skip
/// the mask-based path and use SLIC instead.
pub struct PlaceholderSegmenter;

impl PlaceholderSegmenter {
    pub fn segment(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (w, h) = image.dimensions();
        Ok(vec![1.0f32; (w * h) as usize])
    }
}

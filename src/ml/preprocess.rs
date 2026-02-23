//! Image preprocessing utilities for ML models
//!
//! Provides normalization, resizing, and — critically — correct coordinate
//! remapping from model output space back to original image space.
//!
//! # Why this matters
//!
//! When an image is resized from (orig_w, orig_h) to a square (size x size),
//! the X and Y axes scale by *different factors* unless the image is already square.
//! Sampling back with a single shared scale factor (orig * size / orig_size) is
//! incorrect and introduces spatial distortion in the output map — which breaks
//! pixel-accurate use cases like depth-guided shading or segmentation-aware dithering.
//!
//! The correct approach is to track each axis scale factor independently and use
//! bilinear interpolation when sampling from model output back to original resolution.

use anyhow::Result;
use image::{DynamicImage, GenericImageView};

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Preprocessing configuration for a specific model
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target width fed to the model
    pub target_w: u32,
    /// Target height fed to the model
    pub target_h: u32,
    /// Normalization mean (RGB)
    pub mean: [f32; 3],
    /// Normalization std (RGB)
    pub std: [f32; 3],
}

impl PreprocessConfig {
    /// Plain 0–1 normalization, no ImageNet shift (Depth-Anything, YOLOv8)
    pub fn unit(target_w: u32, target_h: u32) -> Self {
        Self { target_w, target_h, mean: [0.0, 0.0, 0.0], std: [1.0, 1.0, 1.0] }
    }

    /// ImageNet mean/std normalization (BiSeNet, most classification backbones)
    pub fn imagenet(target_w: u32, target_h: u32) -> Self {
        Self {
            target_w,
            target_h,
            mean: [0.485, 0.456, 0.406],
            std:  [0.229, 0.224, 0.225],
        }
    }

    /// 0.5 mean / 0.5 std normalization used by some MiDaS variants
    pub fn half_norm(target_w: u32, target_h: u32) -> Self {
        Self {
            target_w,
            target_h,
            mean: [0.5, 0.5, 0.5],
            std:  [0.5, 0.5, 0.5],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks the independent scale factors applied when resizing an image to a model's
/// input size. Required for correct inverse mapping from model output → original space.
#[derive(Debug, Clone, Copy)]
pub struct ResizeScale {
    /// orig_w / model_w  — multiply model-space X by this to get original-space X
    pub x: f32,
    /// orig_h / model_h  — multiply model-space Y by this to get original-space Y
    pub y: f32,
    /// Original image width
    pub orig_w: u32,
    /// Original image height
    pub orig_h: u32,
    /// Model input width used for this resize
    pub model_w: u32,
    /// Model input height used for this resize
    pub model_h: u32,
}

impl ResizeScale {
    pub fn new(orig_w: u32, orig_h: u32, model_w: u32, model_h: u32) -> Self {
        Self {
            x: orig_w as f32 / model_w as f32,
            y: orig_h as f32 / model_h as f32,
            orig_w,
            orig_h,
            model_w,
            model_h,
        }
    }

    /// Map a point from model-input space to original-image space (normalized 0–1)
    pub fn to_normalized(&self, model_x: f32, model_y: f32) -> (f32, f32) {
        (
            (model_x * self.x) / self.orig_w as f32,
            (model_y * self.y) / self.orig_h as f32,
        )
    }

    /// Map a bounding box (x1, y1, x2, y2) from model space to normalized image space
    pub fn bbox_to_normalized(
        &self,
        x1: f32, y1: f32, x2: f32, y2: f32,
    ) -> (f32, f32, f32, f32) {
        let (nx1, ny1) = self.to_normalized(x1, y1);
        let (nx2, ny2) = self.to_normalized(x2, y2);
        (nx1, ny1, nx2, ny2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

/// Preprocess an image to a flat NCHW f32 tensor for model inference.
///
/// Returns the tensor data and the [`ResizeScale`] needed to map outputs back
/// to the original image's coordinate space.
pub fn preprocess(
    image: &DynamicImage,
    config: &PreprocessConfig,
) -> Result<(Vec<f32>, ResizeScale)> {
    let (orig_w, orig_h) = image.dimensions();

    // Resize to model input dimensions
    let resized = image.resize_exact(
        config.target_w,
        config.target_h,
        image::imageops::FilterType::Lanczos3,
    );
    let rgba = resized.to_rgba8();

    let w = config.target_w as usize;
    let h = config.target_h as usize;
    let mut tensor = vec![0.0f32; 3 * w * h];

    for y in 0..h {
        for x in 0..w {
            let pixel = rgba.get_pixel(x as u32, y as u32).0;
            let base = y * w + x;

            tensor[base]             = (pixel[0] as f32 / 255.0 - config.mean[0]) / config.std[0];
            tensor[w * h + base]     = (pixel[1] as f32 / 255.0 - config.mean[1]) / config.std[1];
            tensor[2 * w * h + base] = (pixel[2] as f32 / 255.0 - config.mean[2]) / config.std[2];
        }
    }

    let scale = ResizeScale::new(orig_w, orig_h, config.target_w, config.target_h);
    Ok((tensor, scale))
}

// ─────────────────────────────────────────────────────────────────────────────
// Output remapping
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinearly sample a single-channel model output map at a given original-image pixel.
///
/// `map` has dimensions `(map_h, map_w)` in row-major order.
/// `orig_x` and `orig_y` are pixel coordinates in the original image.
/// `scale` is the [`ResizeScale`] for this inference run.
#[inline]
pub fn sample_map_bilinear(
    map: &[f32],
    map_w: usize,
    map_h: usize,
    orig_x: usize,
    orig_y: usize,
    scale: &ResizeScale,
) -> f32 {
    // Map original pixel → fractional model-output coordinate
    // model_x = orig_x / scale.x  (scale.x = orig_w / model_w)
    let mx = orig_x as f32 / scale.x;
    let my = orig_y as f32 / scale.y;

    // Clamp to valid range
    let mx = mx.clamp(0.0, (map_w - 1) as f32);
    let my = my.clamp(0.0, (map_h - 1) as f32);

    let x0 = mx.floor() as usize;
    let y0 = my.floor() as usize;
    let x1 = (x0 + 1).min(map_w - 1);
    let y1 = (y0 + 1).min(map_h - 1);

    let tx = mx - x0 as f32;
    let ty = my - y0 as f32;

    let v00 = map[y0 * map_w + x0];
    let v10 = map[y0 * map_w + x1];
    let v01 = map[y1 * map_w + x0];
    let v11 = map[y1 * map_w + x1];

    // Bilinear interpolation
    v00 * (1.0 - tx) * (1.0 - ty)
        + v10 * tx * (1.0 - ty)
        + v01 * (1.0 - tx) * ty
        + v11 * tx * ty
}

/// Upsample a single-channel model output map to the original image's resolution
/// using bilinear interpolation with correct per-axis scale factors.
///
/// This is the correct way to produce a per-pixel map aligned with the original image.
pub fn upsample_map_to_original(
    map: &[f32],
    map_w: usize,
    map_h: usize,
    scale: &ResizeScale,
) -> Vec<f32> {
    let out_w = scale.orig_w as usize;
    let out_h = scale.orig_h as usize;
    let mut out = Vec::with_capacity(out_w * out_h);

    for orig_y in 0..out_h {
        for orig_x in 0..out_w {
            out.push(sample_map_bilinear(map, map_w, map_h, orig_x, orig_y, scale));
        }
    }

    out
}

/// Upsample a multi-class logit map `[num_classes, map_h, map_w]` to original resolution,
/// returning `(argmax_class, max_prob)` per pixel.
///
/// Used by BiSeNet: the model outputs class logits, we argmax per pixel then upsample
/// the discrete label map via nearest-neighbour (correct for class labels — bilinear
/// on class indices is meaningless).
pub fn upsample_class_map_to_original(
    logits: &[f32],
    num_classes: usize,
    map_w: usize,
    map_h: usize,
    scale: &ResizeScale,
) -> (Vec<usize>, Vec<f32>) {
    let out_w = scale.orig_w as usize;
    let out_h = scale.orig_h as usize;

    let mut classes = Vec::with_capacity(out_w * out_h);
    let mut probs   = Vec::with_capacity(out_w * out_h);

    for orig_y in 0..out_h {
        for orig_x in 0..out_w {
            // Nearest-neighbour map coordinates
            let mx = ((orig_x as f32 / scale.x).round() as usize).min(map_w - 1);
            let my = ((orig_y as f32 / scale.y).round() as usize).min(map_h - 1);

            // Argmax over classes at this spatial position
            let mut best_class = 0usize;
            let mut best_prob  = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let v = logits[c * map_h * map_w + my * map_w + mx];
                if v > best_prob {
                    best_prob  = v;
                    best_class = c;
                }
            }

            classes.push(best_class);
            probs.push(best_prob);
        }
    }

    (classes, probs)
}

/// Normalize a f32 slice to [0.0, 1.0] by min-max scaling.
/// Returns the unmodified input if range is effectively zero.
pub fn normalize_min_max(values: &mut [f32]) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &v in values.iter() {
        if v < min { min = v; }
        if v > max { max = v; }
    }
    let range = (max - min).max(1e-6);
    for v in values.iter_mut() {
        *v = ((*v - min) / range).clamp(0.0, 1.0);
    }
}

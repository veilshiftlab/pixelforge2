//! Depth-to-flat color conversion
//!
//! # Phase 2.3 — SLIC-driven per-region shading
//!
//! Replaces the BiSeNet-dependent per-region shading with a model-free
//! approach using SLIC superpixel labels:
//!
//! 1. **Median filter** (5×5) on the depth map to remove speckle around hair,
//!    glasses, and fine detail boundaries.
//! 2. **Per-region MAD normalization**: for each SLIC cluster, compute the
//!    median and MAD (median absolute deviation) of the depth values. Normalize
//!    each pixel's depth to `s = (depth - median) / (1.4826 * MAD)`, clamped to
//!    [-1, 1]. Regions with MAD < `mad_threshold` get `s = 0` (no shading —
//!    avoids amplifying noise in flat regions).
//! 3. **Contrast curve**: `s' = sign(s) * |s|^gamma`. Lower gamma = more
//!    contrast in midtones.
//! 4. **Lab L bias**: `L' = clamp(L + s' * strength * 100, 0, 100)`. Near
//!    features (nose tip) become highlights, far features become shadows.
//!
//! Background separation (Otsu/manual threshold + Lab desaturation) is also
//! applied — background pixels bypass the shading step.
//!
//! # Depth convention
//!
//! Depth-Anything V2 outputs **0 = nearest, 1 = farthest**. For shading, near
//! features should be highlights (high L), so we invert: depth values below
//! the region median produce positive `s` (lighten), above produce negative
//! `s` (darken).

use super::DepthToFlatConfig;
use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::{IntoColor, Lab, Srgb};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Convert depth map to flat color bands with per-region shading.
///
/// Pipeline:
/// 1. Median-filter the depth map (5×5) to remove speckle.
/// 2. Compute per-region shading signal (MAD normalization per SLIC cluster).
/// 3. Separate background (Otsu/manual threshold).
/// 4. Apply contrast curve + Lab L bias to foreground; desaturate background.
pub fn depth_to_flat(
    input: &DynamicImage,
    ml_results: &MLResults,
    config: &DepthToFlatConfig,
) -> Result<DynamicImage> {
    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    let raw_depth = match &ml_results.depth_map {
        Some(d) => d,
        None    => return Ok(input.clone()),
    };

    // ── 1. Median filter the depth map ───────────────────────────────────────
    let depth = median_filter_5x5(raw_depth, width, height);

    // ── 2. Compute per-region shading signal (local MAD) ──────────────────────
    let local_shading = compute_per_region_shading(&depth, ml_results, width, height, config);

    // ── 2b. Compute global shading signal (min-max over whole image) ──────────
    let global_shading = compute_global_shading(&depth, config);

    // ── 2c. Blend local and global shading ────────────────────────────────────
    // Local preserves fine detail within regions; global preserves relative
    // depth between regions. Blending gives both.
    let gw = config.global_depth_weight.clamp(0.0, 1.0);
    let shading: Vec<f32> = local_shading.iter()
        .zip(global_shading.iter())
        .map(|(&l, &g)| l * (1.0 - gw) + g * gw)
        .collect();

    // ── 3. Background separation threshold ────────────────────────────────────
    let bg_threshold = compute_bg_threshold(&depth, config);

    // ── 4. Apply shading + background treatment ───────────────────────────────
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let d = depth.get(idx).copied().unwrap_or(0.5);
            let original = input.get_pixel(x, y);

            let is_background = d > bg_threshold;

            let adjusted = if is_background {
                apply_background_treatment(original, config)
            } else {
                apply_foreground_shading(original, shading.get(idx).copied().unwrap_or(0.0), config)
            };

            output.put_pixel(x, y, adjusted);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

// ─────────────────────────────────────────────────────────────────────────────
// Global shading (min-max over whole image)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute global shading signal using min-max normalization over the entire
/// depth map. Preserves relative depth between all pixels (a far background
/// pixel gets a different shading value than a near foreground pixel), but
/// doesn't amplify local detail like MAD does.
///
/// Returns `s ∈ [-1, 1]` where positive = nearer (highlight), negative = farther (shadow).
fn compute_global_shading(depth: &[f32], _config: &DepthToFlatConfig) -> Vec<f32> {
    if depth.is_empty() {
        return Vec::new();
    }

    let min = depth.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = depth.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    if range < 1e-6 {
        return vec![0.0f32; depth.len()];
    }

    // Normalize to [0, 1], then shift to [-1, 1] with 0.5 → 0
    // Invert: low depth (near) → positive (highlight)
    depth.iter()
        .map(|&d| {
            let normalized = (d - min) / range;  // 0 = nearest, 1 = farthest
            (0.5 - normalized) * 2.0  // near → +1, far → -1
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-region shading (MAD normalization)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the per-pixel shading signal `s ∈ [-1, 1]` using SLIC labels.
///
/// For each SLIC cluster:
/// 1. Collect all depth values.
/// 2. Compute median and MAD.
/// 3. Normalize: `s = (depth - median) / (1.4826 * MAD)`, clamped to [-1, 1].
/// 4. If MAD < `mad_threshold`, set `s = 0` for all pixels in this cluster.
///
/// If SLIC labels are not available, falls back to a global median/MAD over
/// the entire depth map.
fn compute_per_region_shading(
    depth: &[f32],
    ml_results: &MLResults,
    width: u32,
    height: u32,
    config: &DepthToFlatConfig,
) -> Vec<f32> {
    let n = (width * height) as usize;
    let mut shading = vec![0.0f32; n];

    if depth.len() < n {
        return shading;
    }

    match &ml_results.slic_labels {
        Some(labels) if labels.len() >= n => {
            // ── Per-region MAD normalization ──────────────────────────────────
            // Group depth values by cluster ID
            let mut clusters: HashMap<u32, Vec<f32>> = HashMap::new();
            for i in 0..n {
                clusters.entry(labels[i]).or_default().push(depth[i]);
            }

            // Compute (median, MAD) per cluster
            let mut stats: HashMap<u32, (f32, f32)> = HashMap::new();
            for (cluster, mut values) in clusters {
                let median = compute_median(&mut values);
                let deviations: Vec<f32> = values.iter().map(|&v| (v - median).abs()).collect();
                let mut deviations = deviations;
                let mad = compute_median(&mut deviations);
                stats.insert(cluster, (median, mad));
            }

            // Normalize each pixel
            for i in 0..n {
                let cluster = labels[i];
                if let Some(&(median, mad)) = stats.get(&cluster) {
                    if mad < config.mad_threshold {
                        shading[i] = 0.0; // Low-variance region — skip
                    } else {
                        // Invert: depth below median (nearer) → positive s (highlight)
                        let s = (median - depth[i]) / (1.4826 * mad);
                        shading[i] = s.clamp(-1.0, 1.0);
                    }
                }
            }
        }
        _ => {
            // ── Global fallback (no SLIC labels) ───────────────────────────────
            let mut all_depths: Vec<f32> = depth.to_vec();
            let median = compute_median(&mut all_depths);
            let deviations: Vec<f32> = depth.iter().map(|&v| (v - median).abs()).collect();
            let mut deviations = deviations;
            let mad = compute_median(&mut deviations);

            if mad >= config.mad_threshold {
                for i in 0..n {
                    let s = (median - depth[i]) / (1.4826 * mad);
                    shading[i] = s.clamp(-1.0, 1.0);
                }
            }
        }
    }

    shading
}

// ─────────────────────────────────────────────────────────────────────────────
// Contrast curve + Lab L bias
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the contrast curve and Lab L bias to a foreground pixel.
///
/// - Contrast curve: `s' = sign(s) * |s|^gamma`
/// - Lab L bias: `L' = clamp(L + s' * strength * 100, 0, 100)`
fn apply_foreground_shading(
    original: Rgba<u8>,
    s: f32,
    config: &DepthToFlatConfig,
) -> Rgba<u8> {
    if config.strength < 0.001 || s.abs() < 0.001 {
        return original;
    }

    let lab = rgba_to_lab(original);

    // Contrast curve
    let s_curve = s.signum() * s.abs().powf(config.gamma);

    // Lab L bias
    let new_l = (lab.l + s_curve * config.strength * 100.0).clamp(0.0, 100.0);

    lab_to_rgba(Lab::new(new_l, lab.a, lab.b), original[3])
}

// ─────────────────────────────────────────────────────────────────────────────
// Background treatment
// ─────────────────────────────────────────────────────────────────────────────

/// Desaturate and optionally darken background pixels.
fn apply_background_treatment(original: Rgba<u8>, config: &DepthToFlatConfig) -> Rgba<u8> {
    let lab = rgba_to_lab(original);

    let desat = config.bg_desaturation.clamp(0.0, 1.0);
    let new_a = lab.a * (1.0 - desat);
    let new_b = lab.b * (1.0 - desat);

    let new_l = (lab.l + config.bg_lightness_shift * 100.0).clamp(0.0, 100.0);

    lab_to_rgba(Lab::new(new_l, new_a, new_b), original[3])
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth preprocessing — 5×5 median filter
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a 5×5 median filter to the depth map.
///
/// Depth-Anything produces speckle around hair, glasses, and fine detail
/// boundaries. Median filtering preserves edges while removing speckle —
/// Gaussian would blur edges into neighbors, softening the very depth
/// discontinuities we want to detect.
fn median_filter_5x5(depth: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    if depth.len() < n {
        return depth.to_vec();
    }

    let mut filtered = vec![0.0f32; n];

    // 5×5 window has 25 elements; median is the 13th (index 12) after sorting
    let mut window = [0.0f32; 25];

    for y in 0..h {
        for x in 0..w {
            let mut idx = 0;
            for dy in -2..=2isize {
                for dx in -2..=2isize {
                    let nx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    let ny = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    window[idx] = depth[ny * w + nx];
                    idx += 1;
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            filtered[y * w + x] = window[12];
        }
    }

    filtered
}

// ─────────────────────────────────────────────────────────────────────────────
// Background threshold (Otsu or manual)
// ─────────────────────────────────────────────────────────────────────────────

fn compute_bg_threshold(depth: &[f32], config: &DepthToFlatConfig) -> f32 {
    if config.use_otsu_threshold {
        let raw = otsu_threshold(depth);
        let clamped = raw.clamp(0.3, 0.85);
        log::debug!("Otsu depth threshold: raw={:.3} clamped={:.3}", raw, clamped);
        clamped
    } else {
        config.bg_depth_threshold.clamp(0.01, 0.99)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the median of a slice (sorts in place).
fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Otsu's threshold
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Otsu threshold for a depth map (256-bin histogram).
fn otsu_threshold(depth_map: &[f32]) -> f32 {
    const BINS: usize = 256;

    let mut hist = [0u32; BINS];
    for &d in depth_map {
        let bin = ((d.clamp(0.0, 1.0)) * (BINS - 1) as f32).round() as usize;
        hist[bin] += 1;
    }

    let total = depth_map.len() as f64;
    if total == 0.0 {
        return 0.6;
    }

    let mut sum_all = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        sum_all += i as f64 * h as f64;
    }

    let mut sum_fg = 0.0f64;
    let mut weight_fg = 0u64;
    let mut best_variance = 0.0f64;
    let mut best_threshold = 0.6f32;

    for t in 0..BINS {
        weight_fg += hist[t] as u64;
        if weight_fg == 0 { continue; }

        let weight_bg = total as u64 - weight_fg;
        if weight_bg == 0 { break; }

        sum_fg += t as f64 * hist[t] as f64;

        let mean_fg = sum_fg / weight_fg as f64;
        let mean_bg = (sum_all - sum_fg) / weight_bg as f64;

        let between = weight_fg as f64 * weight_bg as f64
            * (mean_fg - mean_bg).powi(2);

        if between > best_variance {
            best_variance = between;
            best_threshold = t as f32 / (BINS - 1) as f32;
        }
    }

    best_threshold
}

// ─────────────────────────────────────────────────────────────────────────────
// Lab color utilities
// ─────────────────────────────────────────────────────────────────────────────

fn rgba_to_lab(rgba: Rgba<u8>) -> Lab {
    let rgb: Srgb = Srgb::new(
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    );
    rgb.into_color()
}

fn lab_to_rgba(lab: Lab, alpha: u8) -> Rgba<u8> {
    let rgb: Srgb = lab.into_color();
    Rgba([
        (rgb.red   * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.green * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.blue  * 255.0).clamp(0.0, 255.0) as u8,
        alpha,
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities kept for pipeline/analysis use
// ─────────────────────────────────────────────────────────────────────────────

/// Evenly-spaced depth band thresholds (used by analysis / debugging tools).
pub fn analyze_depth_histogram(_depth_map: &[f32], num_bands: u32) -> Vec<f32> {
    (0..=num_bands)
        .map(|i| i as f32 / num_bands.max(1) as f32)
        .collect()
}

/// Per-pixel depth gradient magnitude, normalized to [0, 1].
/// Used by the importance map for edge-preserving downsampling.
pub fn compute_depth_gradients(depth_map: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let len = depth_map.len();
    let mut gradients = vec![0.0f32; len];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y as usize * w + x as usize;
            let gx = (depth_map[(idx + 1).min(len - 1)]
                    - depth_map[idx.saturating_sub(1)]).abs();
            let gy = (depth_map[(idx + w).min(len - 1)]
                    - depth_map[idx.saturating_sub(w)]).abs();
            gradients[idx] = (gx + gy) * 0.5;
        }
    }

    let max = gradients.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradients { *g /= max; }
    }
    gradients
}

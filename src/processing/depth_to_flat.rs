//! Depth-to-flat color conversion
//!
//! # What depth is used for (and not used for)
//!
//! **Background separation** (most reliable):
//! Depth-Anything V2 reliably distinguishes foreground subjects from background
//! in portrait images. Background pixels are desaturated and optionally darkened.
//!
//! **Classification hierarchy:**
//! When BiSeNet segmentation is available it is authoritative — a pixel labeled
//! `Skin` by BiSeNet is foreground regardless of its depth value. Depth is only
//! used to classify pixels that BiSeNet labeled `Background` (confirming they are
//! truly background) or when no segmentation was run at all (pure depth fallback).
//!
//! This prevents the OR-logic mistake of desaturating cheeks and neck just
//! because they are slightly farther from the camera than the nose tip.
//!
//! **Lightness reinforcement** (moderate reliability):
//! Within BiSeNet-defined regions, depth nudges the existing photo shading
//! toward discrete tonal bands.  The photo's own illumination is preserved as
//! the primary signal; depth adds at most `depth_influence` weight (default 0.4).
//! Depth is NOT used to drive shading on high-detail facial regions (eyes, lips,
//! nose) — BiSeNet boundaries are far more reliable there.
//!
//! # Depth convention
//!
//! Depth-Anything V2 outputs **0 = nearest, 1 = farthest** after our normalization.
//! For shading we invert this: near pixels should be highlights (high L), far pixels
//! should be shadows (low L).  So `depth_for_shading = 1.0 - depth`.

use super::DepthToFlatConfig;
use crate::ml::{MLResults, SegmentationRegion};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::{FromColor, IntoColor, Lab, Srgb};

// ─────────────────────────────────────────────────────────────────────────────
// Per-image depth statistics (computed once, shared across all pixels)
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics extracted from the depth map for a single image.
struct DepthStats {
    /// Depth value above which an unlabeled pixel is classified as background.
    /// Only consulted when BiSeNet segmentation is absent (pure depth fallback).
    bg_threshold: f32,
}

impl DepthStats {
    fn compute(depth_map: &[f32], config: &DepthToFlatConfig) -> Self {
        let bg_threshold = if config.use_otsu_threshold {
            // Clamp to [0.3, 0.85]: Otsu on a portrait with a dominant near subject
            // can place the threshold very low (e.g. 0.15), classifying most of the
            // image as background.  The clamp prevents that.
            let raw = otsu_threshold(depth_map);
            let clamped = raw.clamp(0.3, 0.85);
            log::debug!("Otsu depth threshold: raw={:.3} clamped={:.3}", raw, clamped);
            clamped
        } else {
            let t = config.bg_depth_threshold.clamp(0.01, 0.99);
            log::debug!("Manual depth threshold: {:.3}", t);
            t
        };
        Self { bg_threshold }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Convert depth map to flat color bands, with background separation.
pub fn depth_to_flat(
    input: &DynamicImage,
    ml_results: &MLResults,
    config: &DepthToFlatConfig,
) -> Result<DynamicImage> {
    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    let depth_map = match &ml_results.depth_map {
        Some(d) => d,
        None    => return Ok(input.clone()),
    };

    let segmentation = ml_results.segmentation.as_ref();
    let stats = DepthStats::compute(depth_map, config);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let depth = *depth_map.get(idx).unwrap_or(&0.5);

            let region = segmentation
                .and_then(|s| s.regions.get(&(idx as u32)).copied())
                .unwrap_or(SegmentationRegion::Background);

            let original = input.get_pixel(x, y);

            // Classification hierarchy — BiSeNet is authoritative when available.
            //
            // WRONG (old): is_bg = (bisenet_bg) OR (depth > threshold)
            //   This desaturates cheeks/neck because they are slightly farther than
            //   the nose tip, even though BiSeNet correctly labeled them as Skin.
            //
            // CORRECT: BiSeNet label wins for all labeled regions.
            //   Depth threshold only applies to pixels BiSeNet labeled Background,
            //   or when no segmentation was run at all (pure depth fallback).
            let is_background = if segmentation.is_some() {
                // Segmentation available: trust BiSeNet completely for labeled regions.
                // Only pixels BiSeNet already called Background get BG treatment.
                region == SegmentationRegion::Background
            } else {
                // No segmentation: fall back to pure depth threshold.
                depth > stats.bg_threshold
            };

            let adjusted = if is_background {
                apply_background_treatment(original, config)
            } else {
                apply_foreground_shading(original, depth, region, config)
            };

            output.put_pixel(x, y, adjusted);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

// ─────────────────────────────────────────────────────────────────────────────
// Background treatment
// ─────────────────────────────────────────────────────────────────────────────

/// Desaturate and optionally darken background pixels.
///
/// Works in Lab: pulls a* and b* toward 0 (desaturation) and shifts L
/// by `bg_lightness_shift`. The hue is not changed — just drained.
fn apply_background_treatment(original: Rgba<u8>, config: &DepthToFlatConfig) -> Rgba<u8> {
    let lab = rgba_to_lab(original);

    // Desaturate: pull a* and b* toward gray (0, 0)
    let desat = config.bg_desaturation.clamp(0.0, 1.0);
    let new_a = lab.a * (1.0 - desat);
    let new_b = lab.b * (1.0 - desat);

    // Lightness shift (negative = darken)
    let new_l = (lab.l + config.bg_lightness_shift * 100.0).clamp(0.0, 100.0);

    lab_to_rgba(Lab::new(new_l, new_a, new_b), original[3])
}

// ─────────────────────────────────────────────────────────────────────────────
// Foreground shading
// ─────────────────────────────────────────────────────────────────────────────

/// Blend photo lightness with depth-derived shading bands.
///
/// High-detail facial regions (eyes, lips, nose) are passed through with
/// depth influence reduced to near-zero — BiSeNet boundaries are far more
/// reliable for those regions than depth geometry.
fn apply_foreground_shading(
    original: Rgba<u8>,
    depth: f32,
    region: SegmentationRegion,
    config: &DepthToFlatConfig,
) -> Rgba<u8> {
    // High-detail regions: preserve the photo almost entirely.
    // Depth shading on a 2-pixel eye is noise, not signal.
    let effective_influence = if region.is_high_detail() {
        config.depth_influence * 0.15
    } else {
        config.depth_influence
    };

    // Early out if depth has no influence
    if effective_influence < 0.001 {
        return original;
    }

    let num_bands = band_count_for_region(region, config).max(1);

    // Invert depth: Depth-Anything encodes 0=near, 1=far.
    // For shading, near features (nose tip) should be highlights.
    let depth_for_shading = 1.0 - depth;

    let band_index = if num_bands > 1 {
        (depth_for_shading * (num_bands as f32 - 1.0))
            .round()
            .clamp(0.0, (num_bands - 1) as f32) as u32
    } else {
        0
    };

    let band_position = if num_bands > 1 {
        band_index as f32 / (num_bands - 1) as f32
    } else {
        0.5
    };

    let lab = rgba_to_lab(original);

    // Depth-derived target lightness
    let depth_l = depth_target_lightness(lab.l, band_position, config);

    // Blend: photo lightness is primary, depth nudges it
    let blended_l = if config.preserve_gradients && config.gradient_preservation > 0.0 {
        // Three-way blend: photo, depth, and soft gradient
        let preserve = config.gradient_preservation;
        lab.l * preserve
            + depth_l * (1.0 - preserve) * effective_influence
            + lab.l  * (1.0 - preserve) * (1.0 - effective_influence)
    } else {
        lerp(lab.l, depth_l, effective_influence)
    };

    lab_to_rgba(Lab::new(blended_l.clamp(0.0, 100.0), lab.a, lab.b), original[3])
}

/// Target lightness value for a given band position within a region.
///
/// band_position 0.0 = deepest shadow band, 1.0 = brightest highlight band.
/// Adjustments are relative to the pixel's original L, capped so we never
/// push a pixel outside a reasonable range.
fn depth_target_lightness(
    original_l: f32,
    band_position: f32,
    config: &DepthToFlatConfig,
) -> f32 {
    let adjustment = if band_position < config.shadow_threshold {
        // Shadow zone — darken proportional to how far into shadow we are
        let shadow_depth = if config.shadow_threshold > 0.0 {
            1.0 - (band_position / config.shadow_threshold)
        } else {
            1.0
        };
        -20.0 * shadow_depth   // max -20 L units for deepest shadow
    } else if band_position > config.highlight_threshold {
        // Highlight zone — lighten proportional to how far into highlight we are
        let denom = 1.0 - config.highlight_threshold;
        let highlight_depth = if denom > 0.0 {
            (band_position - config.highlight_threshold) / denom
        } else {
            1.0
        };
        15.0 * highlight_depth  // max +15 L units for brightest highlight
    } else {
        // Mid-tone — gentle curve, no strong push
        let mid_range = config.highlight_threshold - config.shadow_threshold;
        if mid_range > 0.0 {
            let t = (band_position - config.shadow_threshold) / mid_range;
            5.0 * (t - 0.5)  // ±2.5 L units across the midtone range
        } else {
            0.0
        }
    };

    (original_l + adjustment).clamp(0.0, 100.0)
}

/// Number of depth bands to use for a given region.
fn band_count_for_region(region: SegmentationRegion, config: &DepthToFlatConfig) -> u32 {
    if region.is_skin_like() || region.is_high_detail() {
        config.skin_tone_bands
    } else if region.is_hair_like() {
        config.hair_bands
    } else {
        // Clothing, accessories, etc. — use clothing_bands
        config.clothing_bands
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Otsu's threshold
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Otsu threshold for a depth map (256-bin histogram).
///
/// Maximizes between-class variance, finding the natural split between
/// the foreground (near, low depth) and background (far, high depth) clusters.
/// For portraits with clear subject/background separation this is very accurate.
/// Fallback: 0.6 if the histogram is unimodal (no clear split).
fn otsu_threshold(depth_map: &[f32]) -> f32 {
    const BINS: usize = 256;

    // Build histogram
    let mut hist = [0u32; BINS];
    for &d in depth_map {
        let bin = ((d.clamp(0.0, 1.0)) * (BINS - 1) as f32).round() as usize;
        hist[bin] += 1;
    }

    let total = depth_map.len() as f64;
    if total == 0.0 {
        return 0.6;
    }

    // Precompute cumulative sums
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
// Utility functions
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

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
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
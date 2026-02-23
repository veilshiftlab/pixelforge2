//! Depth-to-flat color conversion implementation

use super::DepthToFlatConfig;
use crate::ml::{MLResults, SegmentationRegion};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::{FromColor, Lab, Srgb};

/// Convert depth map to flat color bands
pub fn depth_to_flat(
    input: &DynamicImage,
    ml_results: &MLResults,
    config: &DepthToFlatConfig,
) -> Result<DynamicImage> {
    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    let depth_map = match &ml_results.depth_map {
        Some(d) => d,
        None => return Ok(input.clone()),
    };

    let segmentation = ml_results.segmentation.as_ref();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let depth = if idx < depth_map.len() { depth_map[idx] } else { 0.5 };

            let region = segmentation
                .and_then(|s| s.regions.get(&(idx as u32)).copied())
                .unwrap_or(SegmentationRegion::Background);

            // Map BiSeNet regions to depth-band counts using semantic helpers.
            // Skin-like regions (Skin, Neck) → skin_tone_bands
            // Hair-like regions (Hair, Hat)  → hair_bands
            // Facial detail regions          → skin_tone_bands (same shading resolution)
            // Everything else                → background_bands
            let num_bands = if region.is_skin_like() || region.is_high_detail() {
                config.skin_tone_bands
            } else if region.is_hair_like() {
                config.hair_bands
            } else {
                match region {
                    SegmentationRegion::Background => config.background_bands,
                    // Earring, Eyeglasses, etc. — treat as background
                    _ => config.background_bands,
                }
            };

            let num_bands = num_bands.max(1);

            let band_index = if num_bands > 1 {
                (depth * (num_bands as f32 - 1.0)).round() as u32
            } else {
                0
            };
            let band_index = band_index.min(num_bands - 1);

            let original = input.get_pixel(x, y);
            let adjusted = adjust_color_for_band(original, band_index, num_bands, config);
            output.put_pixel(x, y, adjusted);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Adjust color for a specific depth band
fn adjust_color_for_band(
    original: Rgba<u8>,
    band_index: u32,
    num_bands: u32,
    config: &DepthToFlatConfig,
) -> Rgba<u8> {
    let rgb = Srgb::new(
        original[0] as f32 / 255.0,
        original[1] as f32 / 255.0,
        original[2] as f32 / 255.0,
    );
    let lab: Lab = Lab::from_color(rgb);

    let band_position = if num_bands > 1 {
        band_index as f32 / (num_bands - 1) as f32
    } else {
        0.5
    };

    let lightness_adjustment = if band_position < config.shadow_threshold {
        let factor = if config.shadow_threshold > 0.0 {
            band_position / config.shadow_threshold
        } else {
            0.0
        };
        -0.2 * (1.0 - factor)
    } else if band_position > config.highlight_threshold {
        let denom = 1.0 - config.highlight_threshold;
        let factor = if denom > 0.0 {
            (band_position - config.highlight_threshold) / denom
        } else {
            1.0
        };
        0.2 * factor
    } else {
        let mid_range = config.highlight_threshold - config.shadow_threshold;
        if mid_range > 0.0 {
            let mid_position = (band_position - config.shadow_threshold) / mid_range;
            0.05 * (mid_position - 0.5)
        } else {
            0.0
        }
    };

    let new_lightness = (lab.l + lightness_adjustment * 100.0).clamp(0.0, 100.0);

    let final_lightness = if config.preserve_gradients && config.gradient_preservation > 0.0 {
        lab.l * config.gradient_preservation + new_lightness * (1.0 - config.gradient_preservation)
    } else {
        new_lightness
    };

    let new_lab = Lab::new(final_lightness, lab.a, lab.b);
    let new_rgb: Srgb = Srgb::from_color(new_lab);

    Rgba([
        (new_rgb.red   * 255.0).clamp(0.0, 255.0) as u8,
        (new_rgb.green * 255.0).clamp(0.0, 255.0) as u8,
        (new_rgb.blue  * 255.0).clamp(0.0, 255.0) as u8,
        original[3],
    ])
}

/// Analyze depth histogram to find band thresholds (simplified: evenly spaced).
pub fn analyze_depth_histogram(depth_map: &[f32], num_bands: u32) -> Vec<f32> {
    let mut thresholds = Vec::with_capacity(num_bands as usize + 1);
    for i in 0..=num_bands {
        thresholds.push(i as f32 / num_bands.max(1) as f32);
    }
    thresholds
}

/// Compute depth gradients for edge detection.
pub fn compute_depth_gradients(depth_map: &[f32], width: u32, height: u32) -> Vec<f32> {
    let total_pixels = (width * height) as usize;
    let mut gradients = vec![0.0f32; total_pixels];
    let w = width as usize;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let grad_x = (depth_map[(idx + 1).min(depth_map.len()-1)]
                        - depth_map[idx.saturating_sub(1)]).abs();
            let grad_y = (depth_map[(idx + w).min(depth_map.len()-1)]
                        - depth_map[idx.saturating_sub(w)]).abs();
            gradients[idx] = (grad_x + grad_y) * 0.5;
        }
    }

    let max = gradients.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradients { *g /= max; }
    }
    gradients
}
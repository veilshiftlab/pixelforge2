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

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let depth = if idx < depth_map.len() { depth_map[idx] } else { 0.5 };

            // Determine region
            let region = segmentation
                .and_then(|s| s.regions.get(&(idx as u32)).copied())
                .unwrap_or(SegmentationRegion::Background);

            // Get band count for this region
            let num_bands = match region {
                SegmentationRegion::Face => config.skin_tone_bands,
                SegmentationRegion::Hair => config.hair_bands,
                SegmentationRegion::Clothing => config.clothing_bands,
                _ => config.background_bands,
            };

            // Ensure at least 1 band
            let num_bands = num_bands.max(1);

            // Determine which band this pixel belongs to
            let band_index = if num_bands > 1 {
                (depth * (num_bands as f32 - 1.0)).round() as u32
            } else {
                0
            };
            let band_index = band_index.min(num_bands - 1);

            // Get original color
            let original = input.get_pixel(x, y);

            // Adjust color based on band
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
    // Convert to Lab color space
    let rgb = Srgb::new(
        original[0] as f32 / 255.0,
        original[1] as f32 / 255.0,
        original[2] as f32 / 255.0,
    );
    let lab: Lab = Lab::from_color(rgb);

    // Calculate band position (0.0 = shadow, 1.0 = highlight)
    let band_position = if num_bands > 1 {
        band_index as f32 / (num_bands - 1) as f32
    } else {
        0.5
    };

    // Determine if this is a shadow, midtone, or highlight
    let lightness_adjustment = if band_position < config.shadow_threshold {
        // Shadow band - reduce lightness
        let factor = if config.shadow_threshold > 0.0 {
            band_position / config.shadow_threshold
        } else {
            0.0
        };
        -0.2 * (1.0 - factor)
    } else if band_position > config.highlight_threshold {
        // Highlight band - increase lightness
        let denom = 1.0 - config.highlight_threshold;
        let factor = if denom > 0.0 {
            (band_position - config.highlight_threshold) / denom
        } else {
            1.0
        };
        0.2 * factor
    } else {
        // Midtone - slight adjustment based on position
        let mid_range = config.highlight_threshold - config.shadow_threshold;
        if mid_range > 0.0 {
            let mid_position = (band_position - config.shadow_threshold) / mid_range;
            0.05 * (mid_position - 0.5)
        } else {
            0.0
        }
    };

    // Apply adjustment
    let new_lightness = (lab.l + lightness_adjustment * 100.0).clamp(0.0, 100.0);

    // Optionally preserve some gradient
    let final_lightness = if config.preserve_gradients && config.gradient_preservation > 0.0 {
        let gradient_factor = config.gradient_preservation;
        lab.l * gradient_factor + new_lightness * (1.0 - gradient_factor)
    } else {
        new_lightness
    };

    // Convert back to RGB
    let new_lab = Lab::new(final_lightness, lab.a, lab.b);
    let new_rgb: Srgb = Srgb::from_color(new_lab);

    Rgba([
        (new_rgb.red * 255.0).clamp(0.0, 255.0) as u8,
        (new_rgb.green * 255.0).clamp(0.0, 255.0) as u8,
        (new_rgb.blue * 255.0).clamp(0.0, 255.0) as u8,
        original[3],
    ])
}

/// Analyze depth histogram to find optimal band thresholds
pub fn analyze_depth_histogram(depth_map: &[f32], num_bands: u32) -> Vec<f32> {
    // Create histogram
    let mut histogram = vec![0u32; 256];
    for &depth in depth_map {
        let bin = (depth * 255.0).min(255.0) as usize;
        if bin < histogram.len() {
            histogram[bin] += 1;
        }
    }

    // Find thresholds using Otsu's method or k-means
    // Simplified: evenly spaced thresholds
    let mut thresholds = Vec::with_capacity(num_bands as usize + 1);
    for i in 0..=num_bands {
        thresholds.push(i as f32 / num_bands.max(1) as f32);
    }

    thresholds
}

/// Compute depth gradients for edge detection
pub fn compute_depth_gradients(depth_map: &[f32], width: u32, height: u32) -> Vec<f32> {
    let total_pixels = (width * height) as usize;
    let mut gradients = vec![0.0f32; total_pixels];
    let width_usize = width as usize;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            
            // Calculate horizontal gradient
            let idx_left = idx.saturating_sub(1);
            let idx_right = (idx + 1).min(depth_map.len().saturating_sub(1));
            let grad_x = (depth_map[idx_right] - depth_map[idx_left]).abs();
            
            // Calculate vertical gradient
            let idx_up = idx.saturating_sub(width_usize);
            let idx_down = (idx + width_usize).min(depth_map.len().saturating_sub(1));
            let grad_y = (depth_map[idx_down] - depth_map[idx_up]).abs();
            
            gradients[idx] = (grad_x + grad_y) * 0.5;
        }
    }

    // Normalize
    let max = gradients.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradients {
            *g /= max;
        }
    }

    gradients
}

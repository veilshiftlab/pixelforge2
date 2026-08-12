//! Downsampling and Importance Map Implementation
//!
//! This module provides the core downsampling algorithms used to convert
//! high-resolution images into pixel art. The key innovation is
//! **importance-weighted downsampling**, which uses ML analysis results
//! to preserve important details during size reduction.
//!
//! # Importance Map
//!
//! The importance map assigns a weight to each pixel, indicating how
//! important it is to preserve that pixel's color during downsampling.
//! Importance is derived from:
//!
//! 1. **Depth Gradients**: Edges in the depth map indicate form boundaries
//! 2. **Image Edges**: Sobel-detected edges in the original image
//!
//! Facial-landmark and segmentation-boundary importance were removed in the
//! pipeline repurpose (those ML models are gone). TEED-driven importance is
//! intentionally NOT used here — TEED is too aggressive for downsampling and
//! preserves detail that doesn't read as pixel art at small sizes.
//!
//! # Downsampling Methods
//!
//! | Method | Description | Use Case |
//! |--------|-------------|----------|
//! | Weighted | Uses importance map for content-aware sizing | Best quality for portraits |
//! | NearestNeighbor | Simple pixel selection | Retro/pixel art style |
//! | Bilinear | Smooth interpolation | Soft, blended look |

use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

// =============================================================================
// Weighted Downsampling
// =============================================================================

/// Importance-weighted downsampling.
///
/// This is the primary downsampling method for high-quality pixel art.
/// It uses an importance map to weight pixel contributions, ensuring
/// that important details (faces, edges) are preserved even at small sizes.
///
/// # Algorithm
///
/// For each output pixel:
/// 1. Find the corresponding region in the input image
/// 2. Weight each input pixel by its importance value
/// 3. Compute weighted average color
pub fn weighted_downsample(
    input: &DynamicImage,
    importance_map: &[f32],
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    let input_width = input.width();
    let input_height = input.height();
    let rgba = input.to_rgba8();

    // Create output buffer
    let mut output = RgbaImage::new(output_width, output_height);

    // Calculate scale factors
    let scale_x = input_width as f32 / output_width as f32;
    let scale_y = input_height as f32 / output_height as f32;

    // For each output pixel
    for out_y in 0..output_height {
        for out_x in 0..output_width {
            // Calculate input region bounds
            let in_x_start = (out_x as f32 * scale_x).floor() as u32;
            let in_y_start = (out_y as f32 * scale_y).floor() as u32;
            let in_x_end = ((out_x + 1) as f32 * scale_x).ceil().min(input_width as f32) as u32;
            let in_y_end = ((out_y + 1) as f32 * scale_y).ceil().min(input_height as f32) as u32;

            // Weighted color accumulation
            let mut r_sum = 0.0f64;
            let mut g_sum = 0.0f64;
            let mut b_sum = 0.0f64;
            let mut a_sum = 0.0f64;
            let mut weight_sum = 0.0f64;

            // Iterate over all input pixels that contribute to this output pixel
            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = (in_y * input_width + in_x) as usize;
                    // Get importance weight, default to 1.0 if out of bounds
                    let weight = importance_map.get(idx).copied().unwrap_or(1.0) as f64;

                    let pixel = rgba.get_pixel(in_x, in_y);
                    r_sum += pixel[0] as f64 * weight;
                    g_sum += pixel[1] as f64 * weight;
                    b_sum += pixel[2] as f64 * weight;
                    a_sum += pixel[3] as f64 * weight;
                    weight_sum += weight;
                }
            }

            // Normalize and set output pixel
            if weight_sum > 0.0 {
                output.put_pixel(out_x, out_y, Rgba([
                    (r_sum / weight_sum).round().clamp(0.0, 255.0) as u8,
                    (g_sum / weight_sum).round().clamp(0.0, 255.0) as u8,
                    (b_sum / weight_sum).round().clamp(0.0, 255.0) as u8,
                    (a_sum / weight_sum).round().clamp(0.0, 255.0) as u8,
                ]));
            } else {
                // Fallback for zero weight (shouldn't happen)
                output.put_pixel(out_x, out_y, Rgba([0, 0, 0, 255]));
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

// =============================================================================
// Simple Downsampling Methods
// =============================================================================

/// Simple bilinear downsampling.
///
/// Uses the image crate's built-in Triangle filter for smooth interpolation.
/// Fast but may blur important details like eyes and edges.
pub fn bilinear_downsample(
    input: &DynamicImage,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    Ok(input.resize(output_width, output_height, image::imageops::FilterType::Triangle))
}

/// Nearest neighbor downsampling.
///
/// Simply selects the closest pixel without interpolation.
/// Produces blocky results that may be desirable for a retro look.
pub fn nearest_neighbor_downsample(
    input: &DynamicImage,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    Ok(input.resize(output_width, output_height, image::imageops::FilterType::Nearest))
}

/// Palette-mode downsampling — the key fix for "smudge" artifacts.
///
/// For each output pixel, this finds the corresponding region in the input
/// image, counts how many times each palette color appears in that region,
/// and picks the **most common** (mode) palette color.
///
/// This produces crisp, discrete output — no averaging, no smudge. Every
/// output pixel is a real palette color. This is how professional pixel-art
/// tools handle downscaling: quantize first, then pick the dominant color.
///
/// # Arguments
///
/// * `input` - The input image (any size, should already be palette-quantized)
/// * `palette` - The palette to use for per-pixel quantization before counting
/// * `output_width` - Target width in pixels
/// * `output_height` - Target height in pixels
pub fn palette_mode_downsample(
    input: &DynamicImage,
    palette: &super::Palette,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    if palette.colors.is_empty() {
        // No palette — fall back to nearest neighbor
        return nearest_neighbor_downsample(input, output_width, output_height);
    }

    let (input_width, input_height) = input.dimensions();
    let rgba = input.to_rgba8();

    // Pre-quantize every input pixel to its nearest palette color index.
    // This is the crucial step: after this, every pixel is a discrete palette
    // entry, so counting frequencies gives meaningful results (no smudge).
    let palette_indices: Vec<usize> = rgba.pixels()
        .map(|p| {
            let target = Rgba([p[0], p[1], p[2], 255]);
            palette.nearest_index(target)
        })
        .collect();

    let mut output = RgbaImage::new(output_width, output_height);

    let scale_x = input_width as f32 / output_width as f32;
    let scale_y = input_height as f32 / output_height as f32;

    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let in_x_start = (out_x as f32 * scale_x).floor() as u32;
            let in_y_start = (out_y as f32 * scale_y).floor() as u32;
            let in_x_end = ((out_x + 1) as f32 * scale_x).ceil().min(input_width as f32) as u32;
            let in_y_end = ((out_y + 1) as f32 * scale_y).ceil().min(input_height as f32) as u32;

            // Count palette color frequencies in this block
            let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = (in_y * input_width + in_x) as usize;
                    if idx < palette_indices.len() {
                        *counts.entry(palette_indices[idx]).or_insert(0) += 1;
                    }
                }
            }

            // Pick the most common palette color (mode)
            let best_idx = counts.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let color = palette.colors.get(best_idx).copied().unwrap_or(Rgba([0, 0, 0, 255]));
            output.put_pixel(out_x, out_y, color);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

// =============================================================================
// Importance Map Computation
// =============================================================================

/// Compute importance map from ML results.
///
/// The importance map is a per-pixel weight indicating how important
/// that pixel is to preserve during downsampling. Higher values mean
/// the pixel should have more influence on the output.
///
/// # Sources of Importance
///
/// 1. **Depth Gradients**: Areas where depth changes rapidly (edges of 3D forms)
///
/// Note: `depth_map` is at the **original image resolution**. When the input
/// image passed in is at a different resolution (post-transform), depth-gradient
/// lookups silently miss and fall back to the base weight of 1.0. This is
/// acceptable for Phase 1; Phase 2.4 resamples ML maps to post-transform dims.
pub fn compute_importance_map(
    width: u32,
    height: u32,
    ml_results: Option<&MLResults>,
) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut importance = vec![1.0f32; size];

    if let Some(ml) = ml_results {
        // Depth Gradient Importance — boost areas where depth changes rapidly
        // (edges of 3D forms). Guarded against dimension mismatch.
        if let Some(depth_map) = &ml.depth_map {
            let w = width as usize;
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = y as usize * w + x as usize;
                    let idx_left = idx.saturating_sub(1);
                    let idx_right = idx + 1;
                    let idx_up = idx.saturating_sub(w);
                    let idx_down = idx + w;

                    if idx_right < depth_map.len() && idx_down < depth_map.len() {
                        let grad_x = (depth_map[idx_right] - depth_map[idx_left]).abs();
                        let grad_y = (depth_map[idx_down] - depth_map[idx_up]).abs();
                        let gradient = (grad_x + grad_y) * 0.5;

                        // Scale gradient for visible effect
                        importance[idx] += gradient * 5.0;
                    }
                }
            }
        }
    }

    // Normalize to 0.0-1.0 range
    let max = importance.iter().cloned().fold(1.0f32, f32::max);
    if max > 0.0 {
        for imp in &mut importance {
            *imp /= max;
        }
    }

    importance
}

/// Compute combined importance map including edge detection.
///
/// This is the main entry point for importance map computation.
/// It combines ML-derived importance (depth gradients) with image edge
/// detection (Sobel) for the most accurate importance weighting.
///
/// # Algorithm
///
/// 1. Compute base importance from ML results (depth gradients)
/// 2. Compute edge importance from image using Sobel operator
/// 3. Combine: final = base * 0.7 + edge * 0.3 + edge * 2.0 (boost)
/// 4. Normalize to 0.0-1.0
pub fn compute_combined_importance_map(
    input: &DynamicImage,
    ml_results: Option<&MLResults>,
) -> Vec<f32> {
    let (width, height) = input.dimensions();

    // Get base importance from ML results
    let mut importance = compute_importance_map(width, height, ml_results);

    // Add edge-based importance from the image itself
    let edge_importance = compute_edge_importance(input);

    // Combine with edge importance
    // Edges are critical for pixel art, so we boost them significantly
    for (i, base) in importance.iter_mut().enumerate() {
        if let Some(&edge) = edge_importance.get(i) {
            *base = *base * 0.7 + edge * 0.3 + edge * 2.0;
        }
    }

    // Normalize again after combination
    let max = importance.iter().cloned().fold(1.0f32, f32::max);
    if max > 0.0 {
        for imp in &mut importance {
            *imp /= max;
        }
    }

    importance
}

/// Compute edge-based importance using Sobel operator.
///
/// Detects edges in the image using the Sobel operator, which
/// computes horizontal and vertical gradients.
///
/// # Sobel Kernels
///
/// Gx (horizontal):
/// ```text
/// [-1  0  1]
/// [-2  0  2]
/// [-1  0  1]
/// ```
///
/// Gy (vertical):
/// ```text
/// [-1 -2 -1]
/// [ 0  0  0]
/// [ 1  2  1]
/// ```
pub fn compute_edge_importance(input: &DynamicImage) -> Vec<f32> {
    let (width, height) = input.dimensions();
    let gray = input.to_luma8();
    let mut importance = vec![0.0f32; (width * height) as usize];

    // Sobel edge detection
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0i32;
            let mut gy = 0i32;

            // Gx kernel (horizontal gradient)
            gx += gray.get_pixel(x - 1, y - 1)[0] as i32 * -1;
            gx += gray.get_pixel(x + 1, y - 1)[0] as i32 *  1;
            gx += gray.get_pixel(x - 1, y)[0] as i32 * -2;
            gx += gray.get_pixel(x + 1, y)[0] as i32 *  2;
            gx += gray.get_pixel(x - 1, y + 1)[0] as i32 * -1;
            gx += gray.get_pixel(x + 1, y + 1)[0] as i32 *  1;

            // Gy kernel (vertical gradient)
            gy += gray.get_pixel(x - 1, y - 1)[0] as i32 * -1;
            gy += gray.get_pixel(x, y - 1)[0] as i32 * -2;
            gy += gray.get_pixel(x + 1, y - 1)[0] as i32 * -1;
            gy += gray.get_pixel(x - 1, y + 1)[0] as i32 *  1;
            gy += gray.get_pixel(x, y + 1)[0] as i32 *  2;
            gy += gray.get_pixel(x + 1, y + 1)[0] as i32 *  1;

            // Compute magnitude
            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            let idx = (y * width + x) as usize;
            importance[idx] = magnitude / 255.0;
        }
    }

    // Normalize to 0.0-1.0 range
    let max = importance.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for imp in &mut importance {
            *imp /= max;
        }
    }

    importance
}

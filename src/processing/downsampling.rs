//! Downsampling and Importance Map Implementation
//!
//! This module provides the core downsampling algorithms used to convert
//! high-resolution images into pixel art. The key innovation is
//! **importance-weighted downsampling**, which uses ML analysis results
//! to preserve important details during size reduction.
//!
//! NOTE: This file was synced from server - includes compute_combined_importance_map function.
//!
//! # Importance Map
//!
//! The importance map assigns a weight to each pixel, indicating how
//! important it is to preserve that pixel's color during downsampling.
//! Importance is derived from:
//!
//! 1. **Facial Landmarks**: Eyes, nose, and mouth areas are prioritized
//! 2. **Depth Gradients**: Edges in the depth map indicate form boundaries
//! 3. **Segmentation Boundaries**: Region boundaries should be crisp
//! 4. **Image Edges**: Sobel-detected edges in the original image
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
///
/// # Arguments
///
/// * `input` - The input image (any size)
/// * `importance_map` - Per-pixel importance weights (same size as input)
/// * `output_width` - Target width in pixels
/// * `output_height` - Target height in pixels
///
/// # Returns
///
/// Downsampled image at the specified dimensions.
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
/// 1. **Landmarks**: Gaussian boost around facial feature points
/// 2. **Depth Gradients**: Areas where depth changes rapidly (edges)
/// 3. **Segmentation Boundaries**: Pixels at region boundaries
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `ml_results` - Optional ML analysis results
///
/// # Returns
///
/// Vec<f32> with length width * height, normalized to 0.0-1.0 range.
pub fn compute_importance_map(
    width: u32,
    height: u32,
    ml_results: Option<&MLResults>,
) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut importance = vec![1.0f32; size];

    if let Some(ml) = ml_results {
        // =====================================================================
        // Landmark Importance
        // =====================================================================
        // Boost importance around facial landmarks (eyes, nose, mouth)
        
        if let Some(landmarks) = &ml.landmarks {
            let radius = (width as f32 * 0.05).max(5.0);
            let radius_sq = radius * radius;

            for &(lx, ly) in &landmarks.points {
                let px = (lx * width as f32) as i32;
                let py = (ly * height as f32) as i32;
                let radius_i = radius.ceil() as i32;

                // Apply Gaussian-like boost in a circular region
                for dy in -radius_i..=radius_i {
                    for dx in -radius_i..=radius_i {
                        let target_x = px + dx;
                        let target_y = py + dy;

                        if target_x >= 0 && target_y >= 0 
                           && target_x < width as i32 
                           && target_y < height as i32 
                        {
                            let dist_sq = (dx * dx + dy * dy) as f32;
                            if dist_sq < radius_sq {
                                // Gaussian falloff: higher boost near center
                                let boost = (1.0 - dist_sq / radius_sq) * 4.0;
                                let idx = (target_y as u32 * width + target_x as u32) as usize;
                                if idx < importance.len() {
                                    importance[idx] += boost;
                                }
                            }
                        }
                    }
                }
            }
        }

        // =====================================================================
        // Depth Gradient Importance
        // =====================================================================
        // Boost areas where depth changes rapidly (edges of 3D forms)
        
        if let Some(depth_map) = &ml.depth_map {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = (y * width + x) as usize;
                    let idx_left = idx - 1;
                    let idx_right = idx + 1;
                    let idx_up = idx - width as usize;
                    let idx_down = idx + width as usize;

                    if idx_right < depth_map.len() && idx_down < depth_map.len() {
                        // Sobel-like gradient calculation
                        let grad_x = (depth_map[idx_right] - depth_map[idx_left]).abs();
                        let grad_y = (depth_map[idx_down] - depth_map[idx_up]).abs();
                        let gradient = (grad_x + grad_y) * 0.5;

                        // Scale gradient for visible effect
                        importance[idx] += gradient * 5.0;
                    }
                }
            }
        }

        // =====================================================================
        // Segmentation Boundary Importance
        // =====================================================================
        // Boost pixels at boundaries between regions (face/hair, skin/clothes)
        
        if let Some(segmentation) = &ml.segmentation {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = (y * width + x) as u32;
                    let current_region = segmentation.regions.get(&idx);
                    
                    // Check if any neighbor has a different region
                    let is_boundary = [
                        (y > 0).then(|| segmentation.regions.get(&(idx - width))),
                        (y < height - 1).then(|| segmentation.regions.get(&(idx + width))),
                        (x > 0).then(|| segmentation.regions.get(&(idx - 1))),
                        (x < width - 1).then(|| segmentation.regions.get(&(idx + 1))),
                    ].iter().any(|neighbor| {
                        *neighbor != Some(current_region)
                    });
                    
                    if is_boundary {
                        let idx_usize = idx as usize;
                        if idx_usize < importance.len() {
                            importance[idx_usize] += 3.0; // Strong boost for boundaries
                        }
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
/// It combines ML-derived importance with image edge detection
/// for the most accurate importance weighting.
///
/// # Algorithm
///
/// 1. Compute base importance from ML results (landmarks, depth, segmentation)
/// 2. Compute edge importance from image using Sobel operator
/// 3. Combine: final = base * 0.7 + edge * 0.3 + edge * 2.0 (boost)
/// 4. Normalize to 0.0-1.0
///
/// # Arguments
///
/// * `input` - The input image
/// * `ml_results` - Optional ML analysis results
///
/// # Returns
///
/// Vec<f32> with length = width * height of input image.
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
            // Weight edge importance higher - it's critical for pixel art
            *base = *base * 0.7 + edge * 0.3 + edge * 2.0; // Boost edges significantly
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
/// # Algorithm
///
/// For each pixel (except borders):
/// 1. Apply Sobel kernel to compute Gx (horizontal gradient)
/// 2. Apply Sobel kernel to compute Gy (vertical gradient)
/// 3. Magnitude = sqrt(Gx² + Gy²)
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

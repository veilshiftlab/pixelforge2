//! Weighted downsampling implementation

use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

/// Importance-weighted downsampling
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

            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = (in_y * input_width + in_x) as usize;
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
                output.put_pixel(out_x, out_y, Rgba([0, 0, 0, 255]));
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Simple bilinear downsampling (fallback)
pub fn bilinear_downsample(
    input: &DynamicImage,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    Ok(input.resize(output_width, output_height, image::imageops::FilterType::Triangle))
}

/// Nearest neighbor downsampling (for pixel art style)
pub fn nearest_neighbor_downsample(
    input: &DynamicImage,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    Ok(input.resize(output_width, output_height, image::imageops::FilterType::Nearest))
}

/// Compute importance map from ML results and other sources
pub fn compute_importance_map(
    width: u32,
    height: u32,
    ml_results: Option<&MLResults>,
) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut importance = vec![1.0f32; size];

    if let Some(ml) = ml_results {
        // Add landmark importance (gaussian around each point)
        if let Some(landmarks) = &ml.landmarks {
            let radius = (width as f32 * 0.05).max(5.0);
            let radius_sq = radius * radius;

            for &(lx, ly) in &landmarks.points {
                let px = (lx * width as f32) as i32;
                let py = (ly * height as f32) as i32;
                let radius_i = radius.ceil() as i32;

                for dy in -radius_i..=radius_i {
                    for dx in -radius_i..=radius_i {
                        let target_x = px + dx;
                        let target_y = py + dy;

                        if target_x >= 0 && target_y >= 0 && target_x < width as i32 && target_y < height as i32 {
                            let dist_sq = (dx * dx + dy * dy) as f32;
                            if dist_sq < radius_sq {
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

        // Add depth gradient importance
        if let Some(depth_map) = &ml.depth_map {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = (y * width + x) as usize;
                    let idx_left = idx - 1;
                    let idx_right = idx + 1;
                    let idx_up = idx - width as usize;
                    let idx_down = idx + width as usize;

                    if idx_right < depth_map.len() && idx_down < depth_map.len() {
                        let grad_x = (depth_map[idx_right] - depth_map[idx_left]).abs();
                        let grad_y = (depth_map[idx_down] - depth_map[idx_up]).abs();
                        let gradient = (grad_x + grad_y) * 0.5;

                        importance[idx] += gradient * 5.0;
                    }
                }
            }
        }
    }

    // Normalize
    let max = importance.iter().cloned().fold(1.0f32, f32::max);
    if max > 0.0 {
        for imp in &mut importance {
            *imp /= max;
        }
    }

    importance
}

/// Compute edge-based importance
pub fn compute_edge_importance(input: &DynamicImage) -> Vec<f32> {
    let (width, height) = input.dimensions();
    let gray = input.to_luma8();
    let mut importance = vec![0.0f32; (width * height) as usize];

    // Sobel edge detection
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0i32;
            let mut gy = 0i32;

            // Sobel kernels
            gx += gray.get_pixel(x - 1, y - 1)[0] as i32 * -1;
            gx += gray.get_pixel(x + 1, y - 1)[0] as i32 * 1;
            gx += gray.get_pixel(x - 1, y)[0] as i32 * -2;
            gx += gray.get_pixel(x + 1, y)[0] as i32 * 2;
            gx += gray.get_pixel(x - 1, y + 1)[0] as i32 * -1;
            gx += gray.get_pixel(x + 1, y + 1)[0] as i32 * 1;

            gy += gray.get_pixel(x - 1, y - 1)[0] as i32 * -1;
            gy += gray.get_pixel(x, y - 1)[0] as i32 * -2;
            gy += gray.get_pixel(x + 1, y - 1)[0] as i32 * -1;
            gy += gray.get_pixel(x - 1, y + 1)[0] as i32 * 1;
            gy += gray.get_pixel(x, y + 1)[0] as i32 * 2;
            gy += gray.get_pixel(x + 1, y + 1)[0] as i32 * 1;

            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            let idx = (y * width + x) as usize;
            importance[idx] = magnitude / 255.0;
        }
    }

    // Normalize
    let max = importance.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for imp in &mut importance {
            *imp /= max;
        }
    }

    importance
}

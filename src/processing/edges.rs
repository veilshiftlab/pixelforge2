//! Edge detection and drawing implementation

use super::{EdgeConfig, EdgeMode, EdgeColorMode};
use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

/// Draw edges on the output image
pub fn draw_edges(
    input: &DynamicImage,
    ml_results: Option<&MLResults>,
    config: &EdgeConfig,
) -> Result<DynamicImage> {
    if matches!(config.edge_mode, EdgeMode::None) {
        return Ok(input.clone());
    }

    let (width, height) = input.dimensions();
    let mut output = input.to_rgba8();

    // Detect edges based on mode
    let edges = detect_edges(input, config.edge_mode, ml_results)?;

    // Draw detected edges
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            if idx < edges.len() && edges[idx] > 0 {
                let edge_color = get_edge_color(&output, x, y, config);

                // Draw edge pixel(s) based on thickness
                for ty in 0..config.thickness {
                    for tx in 0..config.thickness {
                        let px = x.saturating_add(tx);
                        let py = y.saturating_add(ty);

                        if px < width && py < height {
                            // Apply edge darkener to adjacent pixels
                            if config.edge_darkener_strength > 0.0 {
                                darken_adjacent(&mut output, px, py, config.edge_darkener_strength, width, height);
                            }

                            // Draw the edge pixel
                            output.put_pixel(px, py, edge_color);
                        }
                    }
                }
            }
        }
    }

    // Apply anti-aliasing if enabled
    if config.anti_alias_edges {
        anti_alias_edges(&mut output);
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Detect edges using various methods
fn detect_edges(
    input: &DynamicImage,
    mode: EdgeMode,
    ml_results: Option<&MLResults>,
) -> Result<Vec<u8>> {
    let (width, height) = input.dimensions();
    let mut edges = vec![0u8; (width * height) as usize];

    match mode {
        EdgeMode::None => {}
        EdgeMode::Outlines => {
            // Detect outer edges using Sobel operator
            let gradient = sobel_gradient(input);
            for (i, &g) in gradient.iter().enumerate() {
                edges[i] = if g > 0.3 { 255 } else { 0 };
            }
        }
        EdgeMode::Internal => {
            // Detect internal edges from depth/segmentation
            if let Some(ml) = ml_results {
                if let Some(depth) = &ml.depth_map {
                    // Depth edges
                    for y in 1..height - 1 {
                        for x in 1..width - 1 {
                            let idx = (y * width + x) as usize;
                            let idx_left = idx.saturating_sub(1);
                            let idx_right = (idx + 1).min(depth.len() - 1);
                            let idx_up = idx.saturating_sub(width as usize);
                            let idx_down = (idx + width as usize).min(depth.len() - 1);

                            let grad_x = (depth[idx_right] - depth[idx_left]).abs();
                            let grad_y = (depth[idx_down] - depth[idx_up]).abs();

                            if grad_x > 0.1 || grad_y > 0.1 {
                                edges[idx] = 255;
                            }
                        }
                    }
                }
            }
        }
        EdgeMode::Both => {
            // Combine outline and internal edges
            let gradient = sobel_gradient(input);
            for (i, &g) in gradient.iter().enumerate() {
                if g > 0.3 {
                    edges[i] = 255;
                }
            }

            // Add internal edges from ML
            if let Some(ml) = ml_results {
                if let Some(depth) = &ml.depth_map {
                    for y in 1..height - 1 {
                        for x in 1..width - 1 {
                            let idx = (y * width + x) as usize;
                            let idx_left = idx.saturating_sub(1);
                            let idx_right = (idx + 1).min(depth.len() - 1);
                            let idx_up = idx.saturating_sub(width as usize);
                            let idx_down = (idx + width as usize).min(depth.len() - 1);

                            let grad_x = (depth[idx_right] - depth[idx_left]).abs();
                            let grad_y = (depth[idx_down] - depth[idx_up]).abs();

                            if grad_x > 0.1 || grad_y > 0.1 {
                                edges[idx] = 255;
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply morphological thinning for cleaner edges
    edges = thin_edges(&edges, width, height);

    Ok(edges)
}

/// Compute Sobel gradient magnitude
fn sobel_gradient(input: &DynamicImage) -> Vec<f32> {
    let (width, height) = input.dimensions();
    let gray = input.to_luma8();
    let mut gradient = vec![0.0f32; (width * height) as usize];

    // Sobel kernels
    let sobel_x: [[i32; 3]; 3] = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ];

    let sobel_y: [[i32; 3]; 3] = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0i32;
            let mut gy = 0i32;

            for ky in 0..3i32 {
                for kx in 0..3i32 {
                    let px = (x as i32 + kx - 1) as u32;
                    let py = (y as i32 + ky - 1) as u32;
                    let val = gray.get_pixel(px, py)[0] as i32;

                    gx += val * sobel_x[ky as usize][kx as usize];
                    gy += val * sobel_y[ky as usize][kx as usize];
                }
            }

            let idx = (y * width + x) as usize;
            gradient[idx] = ((gx * gx + gy * gy) as f32).sqrt() / 255.0;
        }
    }

    // Normalize
    let max = gradient.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradient {
            *g /= max;
        }
    }

    gradient
}

/// Morphological thinning for cleaner edges
fn thin_edges(edges: &[u8], width: u32, height: u32) -> Vec<u8> {
    // Simple thinning: remove edge pixels that have edge neighbors on both sides
    // For now, return the original edges (full implementation would thin the edges)
    let _height = height; // suppress unused warning
    let _width = width;
    edges.to_vec()
}

/// Get edge color based on configuration
fn get_edge_color(image: &RgbaImage, x: u32, y: u32, config: &EdgeConfig) -> Rgba<u8> {
    match config.edge_color_mode {
        EdgeColorMode::Black => Rgba([0, 0, 0, 255]),
        EdgeColorMode::DarkestShade => {
            // Find the darkest color in a small region
            let mut darkest_luma = 255u8;
            let mut darkest_color = Rgba([0, 0, 0, 255]);

            let radius = 2u32;
            for dy in 0..=radius {
                for dx in 0..=radius {
                    let px = x.saturating_sub(radius) + dx;
                    let py = y.saturating_sub(radius) + dy;

                    if px < image.width() && py < image.height() {
                        let pixel = image.get_pixel(px, py);
                        let r = pixel[0] as u16;
                        let g = pixel[1] as u16;
                        let b = pixel[2] as u16;
                        let luma = ((r + g + b) / 3) as u8;

                        if luma < darkest_luma {
                            darkest_luma = luma;
                            darkest_color = *pixel;
                        }
                    }
                }
            }

            // Darken by 50%
            Rgba([
                (darkest_color[0] as u16 * 50 / 100) as u8,
                (darkest_color[1] as u16 * 50 / 100) as u8,
                (darkest_color[2] as u16 * 50 / 100) as u8,
                255,
            ])
        }
        EdgeColorMode::Custom => {
            let c = config.custom_edge_color;
            Rgba([c.r(), c.g(), c.b(), 255])
        }
    }
}

/// Darken pixels adjacent to edges
fn darken_adjacent(image: &mut RgbaImage, x: u32, y: u32, strength: f32, width: u32, height: u32) {
    let radius = 1u32;
    let factor = 1.0 - strength;

    for dy in 0..=radius * 2 {
        for dx in 0..=radius * 2 {
            if dx == radius && dy == radius {
                continue;
            }

            let px = x.saturating_sub(radius) + dx;
            let py = y.saturating_sub(radius) + dy;

            if px < width && py < height {
                let pixel = image.get_pixel(px, py);
                let darkened = Rgba([
                    (pixel[0] as f32 * factor) as u8,
                    (pixel[1] as f32 * factor) as u8,
                    (pixel[2] as f32 * factor) as u8,
                    pixel[3],
                ]);
                image.put_pixel(px, py, darkened);
            }
        }
    }
}

/// Apply anti-aliasing to edges
fn anti_alias_edges(image: &mut RgbaImage) {
    let (width, height) = image.dimensions();
    let mut blurred = image.clone();

    // Simple box blur for anti-aliasing
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;

            for dy in 0..3u32 {
                for dx in 0..3u32 {
                    let px = x + dx - 1;
                    let py = y + dy - 1;

                    if px < width && py < height {
                        let pixel = image.get_pixel(px, py);
                        r_sum += pixel[0] as u32;
                        g_sum += pixel[1] as u32;
                        b_sum += pixel[2] as u32;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let pixel = blurred.get_pixel_mut(x, y);
                pixel[0] = (r_sum / count) as u8;
                pixel[1] = (g_sum / count) as u8;
                pixel[2] = (b_sum / count) as u8;
            }
        }
    }

    *image = blurred;
}

/// Detect region boundaries for segmentation-based edges
pub fn detect_region_boundaries(
    segmentation: &crate::ml::SegmentationResult,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut boundaries = vec![0u8; (width * height) as usize];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as u32;
            let current = segmentation.regions.get(&idx).copied().unwrap_or(crate::ml::SegmentationRegion::Background);

            // Check neighbors
            let left = segmentation.regions.get(&(idx - 1)).copied().unwrap_or(crate::ml::SegmentationRegion::Background);
            let right = segmentation.regions.get(&(idx + 1)).copied().unwrap_or(crate::ml::SegmentationRegion::Background);
            let up = segmentation.regions.get(&(idx - width)).copied().unwrap_or(crate::ml::SegmentationRegion::Background);
            let down = segmentation.regions.get(&(idx + width)).copied().unwrap_or(crate::ml::SegmentationRegion::Background);

            // If any neighbor has a different region, this is a boundary
            if left != current || right != current || up != current || down != current {
                boundaries[idx as usize] = 255;
            }
        }
    }

    boundaries
}

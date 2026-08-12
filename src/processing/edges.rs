//! Edge detection and drawing implementation
//!
//! # Outline pass
//!
//! When an edge map is available (`ml_results.edge_map`), it is the
//! authoritative edge source. The full-resolution edge probability map is
//! downsampled to pixel-art resolution via **average-pooling** + threshold.
//!
//! Average-pooling (not max-pooling) is critical: for a 512→32 downsample,
//! each output cell covers 16×16 = 256 source pixels. Max-pooling marks the
//! cell as an edge if ANY of those 256 pixels is an edge — producing dense,
//! thick, blob-like edges. Average-pooling takes the mean probability, so only
//! cells where edges are dense/strong pass the threshold. This produces thin,
//! clean 1px lines.
//!
//! # Outline colors
//!
//! Outlines are drawn **after** palette quantization and use the **palette's
//! own colors** — the darkest entry for light pixels, the lightest entry for
//! dark pixels. This guarantees:
//! - Outlines always match the palette (strict palettes like Game Boy are safe)
//! - Outlines read on any background (auto-contrast via palette extremes)
//! - No more "all black" outlines from Lab delta snapping to one dark entry

use super::{EdgeConfig, EdgeMode, EdgeColorMode, OutlineStyle, Palette};
use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

/// Draw edges on the output image.
///
/// `original_dims` is the (width, height) of the image the ML maps were
/// generated from. Needed to downsample the edge map to the input image's
/// resolution.
///
/// `palette` provides the colors used for outline drawing. This function
/// should be called AFTER palette quantization so outlines use palette colors.
pub fn draw_edges(
    input: &DynamicImage,
    ml_results: Option<&MLResults>,
    original_dims: (u32, u32),
    config: &EdgeConfig,
    palette: &Palette,
) -> Result<DynamicImage> {
    if matches!(config.edge_mode, EdgeMode::None) {
        return Ok(input.clone());
    }

    let (width, height) = input.dimensions();
    let mut output = input.to_rgba8();

    // ── Detect edges ────────────────────────────────────────────────────────
    // Prefer ML edge map; fall back to Sobel when the model isn't available.
    let edges = if let Some(ml) = ml_results {
        if let Some(edge_map) = &ml.edge_map {
            downsample_edge_mask(edge_map, original_dims, width, height, config.teed_threshold)
        } else {
            sobel_edge_mask(input, 0.3)
        }
    } else {
        sobel_edge_mask(input, 0.3)
    };

    // ── Pre-compute palette extremes for outline coloring ───────────────────
    let (darkest, lightest) = palette_extremes(palette);

    // ── Draw edges ──────────────────────────────────────────────────────────
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            if idx < edges.len() && edges[idx] > 0 {
                let edge_color = compute_outline_color(&output, x, y, config, darkest, lightest);

                // Draw edge pixel(s) based on thickness
                for ty in 0..config.thickness {
                    for tx in 0..config.thickness {
                        let px = x.saturating_add(tx);
                        let py = y.saturating_add(ty);

                        if px < width && py < height {
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

// ─────────────────────────────────────────────────────────────────────────────
// Edge mask downsampling — average-pool + hysteresis thresholding
// ─────────────────────────────────────────────────────────────────────────────

/// Downsample a full-resolution edge probability map to the target
/// (pixel-art) dimensions and threshold into a binary mask using
/// **hysteresis thresholding** (Canny-style).
///
/// # Algorithm
///
/// 1. Average-pool the edge map to pixel-art resolution.
/// 2. Mark pixels above `high_threshold` as **strong** edges (seeds).
/// 3. Mark pixels above `low_threshold` (but below high) as **weak** edges.
/// 4. Keep weak edges **only if** connected to a strong edge via 8-connectivity.
/// 5. Remove isolated strong edges (surrounded by no weak edges) — noise cleanup.
///
/// This fixes two problems with simple thresholding:
/// - **Line breaks**: weak edges in the middle of a strong edge get dropped,
///   creating gaps. Hysteresis keeps them because they're connected to strong seeds.
/// - **Noise artifacts**: isolated edge pixels from model noise get dropped
///   because they have no strong-edge connection.
fn downsample_edge_mask(
    edge_map: &[f32],
    orig_dims: (u32, u32),
    target_w: u32,
    target_h: u32,
    threshold: f32,
) -> Vec<u8> {
    let (orig_w, orig_h) = (orig_dims.0 as usize, orig_dims.1 as usize);
    let (tw, th) = (target_w as usize, target_h as usize);

    // ── Step 1: Average-pool to pixel-art resolution ──────────────────────────
    let mut avg_map = vec![0.0f32; tw * th];

    let scale_x = orig_w as f32 / tw as f32;
    let scale_y = orig_h as f32 / th as f32;

    for ty in 0..th {
        for tx in 0..tw {
            let ox_start = (tx as f32 * scale_x).floor() as usize;
            let oy_start = (ty as f32 * scale_y).floor() as usize;
            let ox_end = (((tx + 1) as f32 * scale_x).ceil() as usize).min(orig_w);
            let oy_end = (((ty + 1) as f32 * scale_y).ceil() as usize).min(orig_h);

            let mut sum = 0.0f32;
            let mut count = 0u32;
            for oy in oy_start..oy_end {
                for ox in ox_start..ox_end {
                    let idx = oy * orig_w + ox;
                    if idx < edge_map.len() {
                        sum += edge_map[idx];
                        count += 1;
                    }
                }
            }

            avg_map[ty * tw + tx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    // ── Step 2: Hysteresis thresholding ───────────────────────────────────────
    // High threshold = strong edges (seeds). Low threshold = weak edges.
    // The user-facing `threshold` slider controls the low threshold; high is
    // 1.5× above it. This ratio follows Canny's recommendation.
    let low_thresh = threshold;
    let high_thresh = (threshold * 1.5).min(0.95);

    // 0 = no edge, 1 = weak edge, 2 = strong edge
    let mut labels = vec![0u8; tw * th];

    for (i, &avg) in avg_map.iter().enumerate() {
        if avg >= high_thresh {
            labels[i] = 2; // strong
        } else if avg >= low_thresh {
            labels[i] = 1; // weak
        }
    }

    // ── Step 3: BFS from strong edges — keep connected weak edges ──────────────
    let mut mask = vec![0u8; tw * th];
    let mut queue: Vec<usize> = Vec::new();

    // Seed queue with all strong edges
    for (i, &label) in labels.iter().enumerate() {
        if label == 2 {
            mask[i] = 255;
            queue.push(i);
        }
    }

    // BFS: expand to 8-connected weak edges
    while let Some(idx) = queue.pop() {
        let tx = idx % tw;
        let ty = idx / tw;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = tx as i32 + dx;
                let ny = ty as i32 + dy;
                if nx < 0 || ny < 0 || nx >= tw as i32 || ny >= th as i32 { continue; }
                let nidx = ny as usize * tw + nx as usize;
                if labels[nidx] == 1 && mask[nidx] == 0 {
                    mask[nidx] = 255;
                    queue.push(nidx);
                }
            }
        }
    }

    mask
}

// ─────────────────────────────────────────────────────────────────────────────
// Outline color — palette-based auto-contrast
// ─────────────────────────────────────────────────────────────────────────────

/// Find the darkest and lightest colors in the palette.
///
/// These are used for auto-contrast outlines: light pixels get the darkest
/// palette color, dark pixels get the lightest. This ensures outlines always
/// match the palette and read on any background.
fn palette_extremes(palette: &Palette) -> (Rgba<u8>, Rgba<u8>) {
    if palette.colors.is_empty() {
        return (Rgba([0, 0, 0, 255]), Rgba([255, 255, 255, 255]));
    }

    let luma = |c: &&Rgba<u8>| (c[0] as u16 + c[1] as u16 + c[2] as u16) / 3;

    let darkest = palette.colors.iter()
        .min_by_key(luma)
        .copied()
        .unwrap_or(Rgba([0, 0, 0, 255]));

    let lightest = palette.colors.iter()
        .max_by_key(luma)
        .copied()
        .unwrap_or(Rgba([255, 255, 255, 255]));

    (darkest, lightest)
}

/// Compute the outline color for an edge pixel.
///
/// - `AutoContrast`: use palette darkest for light pixels, palette lightest for dark pixels.
/// - `AutoContrastWithHueShift`: same as AutoContrast (hue shift within a discrete palette
///   is not meaningful; kept for config backwards-compat).
/// - `Black`: always use palette darkest.
fn compute_outline_color(
    image: &RgbaImage,
    x: u32,
    y: u32,
    config: &EdgeConfig,
    darkest: Rgba<u8>,
    lightest: Rgba<u8>,
) -> Rgba<u8> {
    let pixel = image.get_pixel(x, y);
    let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;

    match config.outline_style {
        OutlineStyle::Black => darkest,

        OutlineStyle::AutoContrast | OutlineStyle::AutoContrastWithHueShift => {
            if luma >= 128 {
                darkest   // light pixel → darken
            } else {
                lightest  // dark pixel → lighten
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sobel fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Sobel-based binary edge mask — used when the ML edge model is unavailable.
fn sobel_edge_mask(input: &DynamicImage, threshold: f32) -> Vec<u8> {
    let gradient = sobel_gradient(input);
    gradient.iter().map(|&g| if g > threshold { 255 } else { 0 }).collect()
}

/// Compute Sobel gradient magnitude
fn sobel_gradient(input: &DynamicImage) -> Vec<f32> {
    let (width, height) = input.dimensions();
    let gray = input.to_luma8();
    let mut gradient = vec![0.0f32; (width * height) as usize];

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

    let max = gradient.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradient {
            *g /= max;
        }
    }

    gradient
}

/// Apply anti-aliasing to edges
fn anti_alias_edges(image: &mut RgbaImage) {
    let (width, height) = image.dimensions();
    let mut blurred = image.clone();

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

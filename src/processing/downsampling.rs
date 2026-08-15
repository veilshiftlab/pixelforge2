//! Downsampling and Importance Map Implementation
//!
//! # Phase 4 — Smudge fix
//!
//! Three changes target the "color smudge" artifact reported at small output
//! sizes:
//!
//! 1. **D1a — Bilateral pre-filter before palette snap.** In
//!    `palette_mode_downsample`, a 3×3 bilateral filter on the input image
//!    eliminates the noisy alternation on smooth gradients that broke
//!    mode-picking (a smooth red gradient would snap to alternating
//!    palette entries pixel-by-pixel, and per-block mode-picking became
//!    essentially random).
//!
//! 2. **D1b — Average in Lab space, not linear RGB.** In
//!    `weighted_downsample`, each contributing pixel converts to Lab
//!    before accumulation. Linear RGB averaging of saturated red and
//!    saturated blue produces muddy purple; Lab averaging preserves hue.
//!
//! 3. **D1c — Cap importance weight at 2.0× (was 5.0×).** A single
//!    high-importance edge pixel inside a 1024-pixel block was dominating
//!    the weighted average, producing a "smudge of one color" regardless
//!    of what the other 1023 pixels wanted. 2.0× is strong enough to
//!    preserve edges but no longer washes out the block.
//!
//! Plus a new downsampling method:
//!
//! 4. **D2 — `PerceptualDither`.** Area-average downsample → snap to
//!    nearest palette color in Lab → Floyd-Steinberg error diffusion.
//!    Produces clean pixel-art downscaling with no smudge and organic
//!    dithering patterns. The professional pixel-art approach.
//!
//! 5. **D1 perf — Stack array instead of HashMap.**
//!    `palette_mode_downsample` was allocating a `HashMap<usize, u32>` per
//!    output pixel (1024 allocs for a 32×32 output). Replaced with a
//!    stack-allocated `[u32; 256]` array, zeroed once per output pixel via
//!    `fill(0)`.
//!
//! # Importance Map
//!
//! The importance map assigns a weight to each pixel, indicating how
//! important it is to preserve that pixel's color during downsampling.
//! Importance is derived from:
//!
//! 1. **Depth Gradients** (capped at 2.0×): edges of 3D forms
//! 2. **Image Edges**: Sobel-detected edges in the original image
//!
//! # Downsampling Methods
//!
//! | Method | Description | Use Case |
//! |--------|-------------|----------|
//! | PaletteMode | Bilateral-filter, snap, pick most common per block | Crisp discrete output, no smudge (default) |
//! | PerceptualDither | Area-average + Floyd-Steinberg | Clean pro pixel-art (new in Phase 4) |
//! | Weighted | Lab-space weighted average | Smooth, preserves detail |
//! | NearestNeighbor | Pick center pixel | Retro/pixel art style |
//! | Bilinear | Smooth interpolation | Soft, blended look |

use super::palette::{PaletteLab, rgb_to_lab, lab_to_rgb};
use super::Palette;
use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::Lab;

/// Max palette size we'll handle in stack-allocated per-block counters.
/// Palette colors above this index fall back to the HashMap path (rare).
const MAX_PALETTE_SIZE: usize = 256;

// =============================================================================
// Weighted Downsampling (D1b — Lab-space averaging)
// =============================================================================

/// Importance-weighted downsampling.
///
/// This is the primary downsampling method for high-quality pixel art.
/// It uses an importance map to weight pixel contributions, ensuring
/// that important details (faces, edges) are preserved even at small sizes.
///
/// # Phase 4 — D1b: Lab-space averaging
///
/// Each contributing pixel is converted to Lab before being weighted-averaged.
/// Linear RGB averaging of saturated red + saturated blue gives muddy purple;
/// Lab averaging preserves hue and produces perceptually correct blends.
pub fn weighted_downsample(
    input: &DynamicImage,
    importance_map: &[f32],
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    let input_width = input.width();
    let input_height = input.height();
    let rgba = input.to_rgba8();

    let mut output = RgbaImage::new(output_width, output_height);

    let scale_x = input_width as f32 / output_width as f32;
    let scale_y = input_height as f32 / output_height as f32;

    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let in_x_start = (out_x as f32 * scale_x).floor() as u32;
            let in_y_start = (out_y as f32 * scale_y).floor() as u32;
            let in_x_end = ((out_x + 1) as f32 * scale_x).ceil().min(input_width as f32) as u32;
            let in_y_end = ((out_y + 1) as f32 * scale_y).ceil().min(input_height as f32) as u32;

            // Accumulate in Lab space (perceptually uniform — preserves hue)
            let mut l_sum = 0.0f64;
            let mut a_sum = 0.0f64;
            let mut b_sum = 0.0f64;
            let mut alpha_sum = 0.0f64;
            let mut weight_sum = 0.0f64;

            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = (in_y * input_width + in_x) as usize;
                    let weight = importance_map.get(idx).copied().unwrap_or(1.0) as f64;

                    let pixel = rgba.get_pixel(in_x, in_y);
                    let lab = rgb_to_lab(*pixel);
                    l_sum     += lab.l as f64 * weight;
                    a_sum     += lab.a as f64 * weight;
                    b_sum     += lab.b as f64 * weight;
                    alpha_sum += pixel[3] as f64 * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                let avg_lab = Lab::new(
                    (l_sum / weight_sum) as f32,
                    (a_sum / weight_sum) as f32,
                    (b_sum / weight_sum) as f32,
                );
                let rgb = lab_to_rgb(avg_lab);
                let alpha = (alpha_sum / weight_sum).round().clamp(0.0, 255.0) as u8;
                output.put_pixel(out_x, out_y, Rgba([rgb[0], rgb[1], rgb[2], alpha]));
            } else {
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
// Palette-mode downsampling (D1a — bilateral pre-filter; D1 perf — stack array)
// =============================================================================

/// Palette-mode downsampling — the key fix for "smudge" artifacts.
///
/// For each output pixel, this finds the corresponding region in the input
/// image, counts how many times each palette color appears in that region,
/// and picks the **most common** (mode) palette color.
///
/// # Phase 4 — D1a: bilateral pre-filter
///
/// Before snapping each input pixel to its nearest palette color, the input
/// image is run through a 3×3 bilateral filter. This eliminates the noisy
/// alternation on smooth gradients: a smooth red gradient would otherwise
/// snap to alternating palette entries pixel-by-pixel, and per-block
/// mode-picking became essentially random (the source of "smudge").
///
/// Bilateral filtering preserves edges while smoothing flats — exactly
/// what we want before quantization.
///
/// # Phase 4 — D1 perf: stack array instead of HashMap
///
/// The per-output-pixel `HashMap<usize, u32>` was allocating 1024 HashMaps
/// for a 32×32 output. Replaced with a stack `[u32; 256]` array zeroed
/// via `fill(0)` once per output pixel.
///
/// # Arguments
///
/// * `input` - The input image (any size, should already be palette-quantized)
/// * `palette` - The palette to use for per-pixel quantization before counting
/// * `output_width` - Target width in pixels
/// * `output_height` - Target height in pixels
pub fn palette_mode_downsample(
    input: &DynamicImage,
    palette: &Palette,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    if palette.colors.is_empty() {
        return nearest_neighbor_downsample(input, output_width, output_height);
    }

    let (input_width, input_height) = input.dimensions();

    // ── Phase 4 — D1a: bilateral pre-filter ────────────────────────────────
    // Smooths flat regions while preserving edges, so palette snap doesn't
    // produce noisy alternation on smooth gradients.
    // Phase 6 — perf: `bilateral_filter_3x3_rgba` returns RgbaImage directly,
    // avoiding one redundant `to_rgba8()` conversion.
    let rgba = bilateral_filter_3x3_rgba(input);

    // Pre-quantize every input pixel to its nearest palette color index.
    let palette_lab = PaletteLab::from_palette(palette);
    let palette_indices: Vec<usize> = rgba.pixels()
        .map(|p| {
            let lab = rgb_to_lab(Rgba([p[0], p[1], p[2], 255]));
            // Use PaletteLab's cached Lab values for O(K) lookup instead of
            // recomputing per palette entry.
            palette_index_nearest(&palette_lab, lab)
        })
        .collect();

    let mut output = RgbaImage::new(output_width, output_height);

    let scale_x = input_width as f32 / output_width as f32;
    let scale_y = input_height as f32 / output_height as f32;

    // Phase 4 — D1 perf: stack-allocated counter array, zeroed per output pixel.
    // Avoids 1024 HashMap allocations on a 32×32 output.
    let mut counts: [u32; MAX_PALETTE_SIZE] = [0; MAX_PALETTE_SIZE];

    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let in_x_start = (out_x as f32 * scale_x).floor() as u32;
            let in_y_start = (out_y as f32 * scale_y).floor() as u32;
            let in_x_end = ((out_x + 1) as f32 * scale_x).ceil().min(input_width as f32) as u32;
            let in_y_end = ((out_y + 1) as f32 * scale_y).ceil().min(input_height as f32) as u32;

            // Zero counters for this output pixel
            for c in counts.iter_mut() { *c = 0; }

            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = (in_y * input_width + in_x) as usize;
                    if idx < palette_indices.len() {
                        let pi = palette_indices[idx];
                        if pi < MAX_PALETTE_SIZE {
                            counts[pi] += 1;
                        }
                    }
                }
            }

            // Pick the most common palette color (mode)
            let mut best_idx = 0usize;
            let mut best_count = 0u32;
            for (i, &c) in counts.iter().enumerate() {
                if c > best_count {
                    best_count = c;
                    best_idx = i;
                }
            }

            let color = palette.colors.get(best_idx).copied().unwrap_or(Rgba([0, 0, 0, 255]));
            output.put_pixel(out_x, out_y, color);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Return the index of the palette entry nearest to `target_lab`.
/// Inline mirror of `Palette::nearest_index` but uses the cached Lab values.
#[inline]
fn palette_index_nearest(palette_lab: &PaletteLab, target: Lab) -> usize {
    if palette_lab.lab.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;
    for (i, &lab) in palette_lab.lab.iter().enumerate() {
        let dl = target.l - lab.l;
        let da = target.a - lab.a;
        let db = target.b - lab.b;
        let d = dl * dl + da * da + db * db;
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

// =============================================================================
// Perceptual Dither downsampling (D2 — new in Phase 4)
// =============================================================================

/// Perceptual-dither downsampling — the professional pixel-art approach.
///
/// # Algorithm
///
/// 1. **Area-average downsample** to target resolution (proper anti-aliasing).
/// 2. **Snap each pixel** to nearest palette color in Lab space.
/// 3. **Floyd-Steinberg error diffusion**: propagate the Lab quantization
///    error to unprocessed neighbors:
///    - 7/16 to right, 3/16 to bottom-left, 5/16 to bottom, 1/16 to bottom-right.
///
/// Produces clean pixel-art downscaling with no smudge and organic dithering
/// patterns (vs ordered/Bayer dithering which produces regular grids).
///
/// # When to use
///
/// - Best for portraits and any image with smooth gradients
/// - Better than `PaletteMode` when the palette has few colors (4–16) and
///   you want gradients to render as smooth dithered transitions rather
///   than a single dominant color per block
/// - Slower than `PaletteMode` due to the error-diffusion pass
pub fn perceptual_dither_downsample(
    input: &DynamicImage,
    palette: &Palette,
    output_width: u32,
    output_height: u32,
) -> Result<DynamicImage> {
    if palette.colors.is_empty() {
        return nearest_neighbor_downsample(input, output_width, output_height);
    }

    // ── Step 1: Area-average downsample (proper anti-aliasing) ──────────────
    let area_averaged = area_average_downsample(input, output_width, output_height);
    let mut rgba = area_averaged.to_rgba8();
    let (w, h) = rgba.dimensions();

    // ── Step 2+3: Floyd-Steinberg dithering in Lab space ─────────────────────
    let palette_lab = PaletteLab::from_palette(palette);

    // Working buffer of Lab values (with accumulated error).
    // We mutate this in place as we walk the image.
    let mut lab_buf: Vec<Lab> = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            lab_buf.push(rgb_to_lab(*p));
        }
    }

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let current = lab_buf[idx];

            // Snap to nearest palette color
            let palette_idx = palette_index_nearest(&palette_lab, current);
            let snapped_lab = palette_lab.lab[palette_idx];
            let snapped_rgb = palette_lab.rgb[palette_idx];

            // Replace the pixel with the snapped palette color
            rgba.put_pixel(x, y, snapped_rgb);

            // Compute Lab error
            let err_l = current.l - snapped_lab.l;
            let err_a = current.a - snapped_lab.a;
            let err_b = current.b - snapped_lab.b;

            // Floyd-Steinberg distribution:
            //        | * | 7/16
            // |3/16|5/16|1/16
            diffuse_error(&mut lab_buf, x, y, w, h, err_l, err_a, err_b,  1,  0, 7.0 / 16.0);
            diffuse_error(&mut lab_buf, x, y, w, h, err_l, err_a, err_b, -1,  1, 3.0 / 16.0);
            diffuse_error(&mut lab_buf, x, y, w, h, err_l, err_a, err_b,  0,  1, 5.0 / 16.0);
            diffuse_error(&mut lab_buf, x, y, w, h, err_l, err_a, err_b,  1,  1, 1.0 / 16.0);
        }
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Add a fraction of the quantization error to the pixel at `(x+dx, y+dy)`.
#[inline]
fn diffuse_error(
    lab_buf: &mut [Lab],
    x: u32, y: u32,
    w: u32, h: u32,
    err_l: f32, err_a: f32, err_b: f32,
    dx: i32, dy: i32,
    weight: f32,
) {
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { return; }
    let nidx = (ny as u32 * w + nx as u32) as usize;
    lab_buf[nidx].l += err_l * weight;
    lab_buf[nidx].a += err_a * weight;
    lab_buf[nidx].b += err_b * weight;
}

/// Area-average downsample — proper anti-aliasing for downscaling.
///
/// Each output pixel is the unweighted average of all input pixels in its
/// block. This is the standard "box filter" — mathematically equivalent
/// to area integration, which is the correct way to downscale.
fn area_average_downsample(
    input: &DynamicImage,
    output_width: u32,
    output_height: u32,
) -> DynamicImage {
    let (input_width, input_height) = input.dimensions();
    let rgba = input.to_rgba8();

    let mut output = RgbaImage::new(output_width, output_height);

    let scale_x = input_width as f32 / output_width as f32;
    let scale_y = input_height as f32 / output_height as f32;

    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let in_x_start = (out_x as f32 * scale_x).floor() as u32;
            let in_y_start = (out_y as f32 * scale_y).floor() as u32;
            let in_x_end = ((out_x + 1) as f32 * scale_x).ceil().min(input_width as f32) as u32;
            let in_y_end = ((out_y + 1) as f32 * scale_y).ceil().min(input_height as f32) as u32;

            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut a_sum = 0u32;
            let mut count = 0u32;

            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let pixel = rgba.get_pixel(in_x, in_y);
                    r_sum += pixel[0] as u32;
                    g_sum += pixel[1] as u32;
                    b_sum += pixel[2] as u32;
                    a_sum += pixel[3] as u32;
                    count += 1;
                }
            }

            if count > 0 {
                output.put_pixel(out_x, out_y, Rgba([
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                    (a_sum / count) as u8,
                ]));
            } else {
                output.put_pixel(out_x, out_y, Rgba([0, 0, 0, 255]));
            }
        }
    }

    DynamicImage::ImageRgba8(output)
}

// =============================================================================
// Bilateral filter (D1a helper)
// =============================================================================

/// 3×3 bilateral filter — edge-preserving smoothing.
///
/// Each output pixel is a weighted average of its 3×3 neighborhood where
/// weights depend on **both spatial proximity** (Gaussian on distance) and
/// **color similarity** (Gaussian on Lab ΔE). Pixels far in color from
/// the center contribute almost nothing, so edges are preserved while
/// flat regions are smoothed.
///
/// This is the pre-quantization step in `palette_mode_downsample` —
/// smoothing flats eliminates the noisy palette-snap alternation that
/// was breaking per-block mode-picking (the source of "smudge").
/// Phase 6 — perf: returns `RgbaImage` directly (was `DynamicImage`) so callers
/// that need the rgba buffer (which is everyone — `palette_mode_downsample`
/// calls `.pixels()` on it) skip one `to_rgba8()` conversion.
fn bilateral_filter_3x3_rgba(input: &DynamicImage) -> RgbaImage {
    let (w, h) = input.dimensions();
    let rgba_in = input.to_rgba8();
    let mut rgba_out = RgbaImage::new(w, h);

    // Pre-compute Lab for every input pixel (used for color-similarity weight).
    let mut lab_in: Vec<Lab> = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            lab_in.push(rgb_to_lab(*rgba_in.get_pixel(x, y)));
        }
    }

    // Sigma parameters: tuned for "smooth flats, preserve edges".
    let sigma_space = 1.2;   // ~one pixel radius
    let sigma_color = 25.0;  // Lab ΔE threshold for "same region"

    let wi = w as i32;
    let hi = h as i32;

    for y in 0..hi {
        for x in 0..wi {
            let idx = (y as usize) * (w as usize) + (x as usize);
            let center_lab = lab_in[idx];
            let center_alpha = rgba_in.get_pixel(x as u32, y as u32)[3];

            let mut l_sum = 0.0f64;
            let mut a_sum = 0.0f64;
            let mut b_sum = 0.0f64;
            let mut weight_sum = 0.0f64;

            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= wi || ny >= hi { continue; }
                    let nidx = (ny as usize) * (w as usize) + (nx as usize);
                    let n_lab = lab_in[nidx];

                    // Spatial weight: Gaussian on Euclidean distance (1 or √2)
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let w_space = (-dist_sq / (2.0 * sigma_space * sigma_space)).exp();

                    // Color weight: Gaussian on Lab ΔE
                    let dl = center_lab.l - n_lab.l;
                    let da = center_lab.a - n_lab.a;
                    let db = center_lab.b - n_lab.b;
                    let color_dist_sq = dl * dl + da * da + db * db;
                    let w_color = (-color_dist_sq / (2.0 * sigma_color * sigma_color)).exp();

                    let w_total = w_space * w_color;
                    l_sum += n_lab.l as f64 * w_total as f64;
                    a_sum += n_lab.a as f64 * w_total as f64;
                    b_sum += n_lab.b as f64 * w_total as f64;
                    weight_sum += w_total as f64;
                }
            }

            if weight_sum > 0.0 {
                let filtered_lab = Lab::new(
                    (l_sum / weight_sum) as f32,
                    (a_sum / weight_sum) as f32,
                    (b_sum / weight_sum) as f32,
                );
                let rgb = lab_to_rgb(filtered_lab);
                rgba_out.put_pixel(x as u32, y as u32, Rgba([rgb[0], rgb[1], rgb[2], center_alpha]));
            } else {
                rgba_out.put_pixel(x as u32, y as u32, *rgba_in.get_pixel(x as u32, y as u32));
            }
        }
    }

    rgba_out
}

// =============================================================================
// Importance Map Computation (D1c — cap importance weight at 2.0×)
// =============================================================================

/// Compute importance map from ML results.
///
/// # Phase 4 — D1c: importance weight cap
///
/// Changed from `* 5.0` to `* 2.0`. A 1024-pixel block with a single
/// high-importance edge pixel was being dominated by that one pixel
/// (5× weight made it equivalent to ~5 ordinary pixels, but the other
/// 1023 should outvote). 2× preserves edge influence without smudging
/// the block to one color.
pub fn compute_importance_map(
    width: u32,
    height: u32,
    ml_results: Option<&MLResults>,
) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut importance = vec![1.0f32; size];

    if let Some(ml) = ml_results {
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

                        // Phase 4 — D1c: capped at 2.0× (was 5.0×)
                        importance[idx] += gradient * 2.0;
                    }
                }
            }
        }
    }

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
/// # Phase 5 carryover — C12 fix
///
/// Replaced the buggy `*base * 0.7 + edge * 0.3 + edge * 2.0` formula
/// (which double-counted edges and gave them 2.3× weight instead of 0.3×)
/// with the intended `*base * 0.7 + edge * 0.3`. If extra edge boost is
/// desired, it's now a clean 2.0× multiplier instead of a bug.
pub fn compute_combined_importance_map(
    input: &DynamicImage,
    ml_results: Option<&MLResults>,
) -> Vec<f32> {
    let (width, height) = input.dimensions();

    let mut importance = compute_importance_map(width, height, ml_results);
    let edge_importance = compute_edge_importance(input);

    // Phase 5 — C12 fix: clean formula (was 0.7 + 0.3 + 2.0 = buggy double-count)
    for (i, base) in importance.iter_mut().enumerate() {
        if let Some(&edge) = edge_importance.get(i) {
            *base = *base * 0.7 + edge * 0.3;
        }
    }

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
pub fn compute_edge_importance(input: &DynamicImage) -> Vec<f32> {
    let (width, height) = input.dimensions();
    let gray = input.to_luma8();
    let mut importance = vec![0.0f32; (width * height) as usize];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx = 0i32;
            let mut gy = 0i32;

            gx += gray.get_pixel(x - 1, y - 1)[0] as i32 * -1;
            gx += gray.get_pixel(x + 1, y - 1)[0] as i32 *  1;
            gx += gray.get_pixel(x - 1, y)[0] as i32 * -2;
            gx += gray.get_pixel(x + 1, y)[0] as i32 *  2;
            gx += gray.get_pixel(x - 1, y + 1)[0] as i32 * -1;
            gx += gray.get_pixel(x + 1, y + 1)[0] as i32 *  1;

            gy += gray.get_pixel(x - 1, y - 1)[0] as i32 * -1;
            gy += gray.get_pixel(x, y - 1)[0] as i32 * -2;
            gy += gray.get_pixel(x + 1, y - 1)[0] as i32 * -1;
            gy += gray.get_pixel(x - 1, y + 1)[0] as i32 *  1;
            gy += gray.get_pixel(x, y + 1)[0] as i32 *  2;
            gy += gray.get_pixel(x + 1, y + 1)[0] as i32 *  1;

            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            let idx = (y * width + x) as usize;
            importance[idx] = magnitude / 255.0;
        }
    }

    let max = importance.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for imp in &mut importance {
            *imp /= max;
        }
    }

    importance
}

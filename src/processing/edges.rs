//! Edge detection and drawing implementation
//!
//! # Phase 3 + Phase 4 — Edge pipeline rework
//!
//! Phase 3 adds a cleanup stage between skeletonization and classification:
//! gap-bridging, isolated-noise drop, and short-segment drop. The result is
//! cleaner, more continuous edge lines with less single-pixel noise.
//!
//! Phase 4 replaces the old single `AutoContrast` outline mode with three
//! explicit modes (see `OutlineStyle`):
//! - `LocalColorShift` (default) — edge = local mean Lab, shifted darker
//!   and optionally hue-rotated, snapped to nearest palette entry.
//! - `Black` — fixed darkest palette color.
//! - `MaxContrast` — the old AutoContrast behavior (max-ΔE from local mean).
//!
//! # Phase 6 — Sobel fallback percentile normalization
//!
//! The Sobel fallback path (used when ML edge detection is unavailable) now
//! percentile-normalizes the gradient (5/95) before thresholding, so the
//! `teed_threshold` slider works on both ML and Sobel paths. Previously the
//! Sobel path hardcoded threshold=0.3 and ignored the user's setting.
//!
//! # Phase 7 — EdgeMode::Internal fallback fix
//!
//! When SLIC labels are unavailable, `EdgeMode::Internal` no longer silently
//! behaves like `EdgeMode::Both`. It falls back to border-touching edges only
//! (silhouette edges detectable without region labels) and emits a warning.
//! `Outlines` and `Both` keep their backward-compatible behavior.
//!
//! # Full pipeline
//!
//! 1. **Edge map normalization** (`ml/models/teed.rs`): percentile (5/95)
//!    instead of min-max. A few saturated pixels no longer collapse the
//!    dynamic range. Phase 6: Sobel fallback now does the same.
//! 2. **Downsample to pixel-art resolution**: **max-pool** the full-res edge
//!    map (preserves sparse 1px lines that average-pooling diluted below
//!    threshold — the source of "edges break midway").
//! 3. **3×3 non-max suppression**: thin clusters of strong edges back to
//!    single-pixel-wide ridges. Output is a binary mask of edge candidates.
//! 4. **Hysteresis thresholding**: `low = threshold * 0.4`, `high = threshold`.
//!    Weak edges connected to strong seeds survive; isolated noise drops.
//! 5. **Zhang-Suen skeletonization**: thins the binary mask to a true
//!    1-pixel-wide line network. This is the "make it lines not pixels" step.
//! 6. **Phase 3 — Cleanup**: gap-bridging (connect skeleton segments
//!    separated by 1-2px gaps), isolated-noise drop (remove skeleton pixels
//!    with zero skeleton neighbors), short-segment drop (remove connected
//!    components shorter than `min_segment_length`).
//! 7. **Edge-mode classification** (C9 + Phase 7): if SLIC labels are
//!    available, an edge pixel is classified as `Outline` if its 3×3
//!    neighborhood spans ≥2 cluster labels (silhouette), else `Internal`
//!    (within-region detail). `EdgeMode::Outlines` keeps only outlines;
//!    `Internal` keeps only internal; `Both` keeps everything. When SLIC
//!    is missing, `Internal` falls back to border-touching edges + warns.
//! 8. **Phase 4 — Compositing**: color each edge pixel according to
//!    `outline_style` (see `compute_outline_color`).
//! 9. **Edge-direction AA**: if `anti_alias_edges` is on, apply 2-tap
//!    blending **only along the line's normal direction** (per-polyline).
//!    Interior palette pixels are untouched — no more full-image blur.
//!
//! ## Outline color
//!
//! Outlines are drawn **after** palette quantization and use the palette's
//! own colors (never out-of-palette colors). Strict palettes like Game Boy
//! are safe — `OutlineStyle::Black` falls back to the palette's darkest
//! entry. `LocalColorShift` snaps its shifted color to the nearest palette
//! entry, so it's also palette-legal.

use super::palette::{PaletteLab, rgb_to_lab};
use super::{EdgeConfig, EdgeMode, OutlineStyle, Palette};
use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::Lab;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Draw edges on the output image.
///
/// `original_dims` is the (width, height) of the image the ML maps were
/// generated from. Needed to downsample the edge map to the input image's
/// resolution.
///
/// `palette` provides the colors used for outline drawing. This function
/// should be called AFTER palette quantization so outlines use palette colors.
///
/// `slic_labels` (optional, post-transform resolution) drives the
/// `EdgeMode::Internal` vs `Outlines` classification. When `None`, all
/// edges are treated as outlines (backward-compatible behavior).
///
/// Phase 8: `warnings` is appended to when a silent fallback occurs inside
/// this function (e.g. the Phase 7 Internal-mode SLIC fallback in
/// `classify_edges`). The caller surfaces these to the UI.
pub fn draw_edges(
    input: &DynamicImage,
    ml_results: Option<&MLResults>,
    original_dims: (u32, u32),
    config: &EdgeConfig,
    palette: &Palette,
    warnings: &mut Vec<String>,
) -> Result<DynamicImage> {
    if matches!(config.edge_mode, EdgeMode::None) {
        return Ok(input.clone());
    }

    let (width, height) = input.dimensions();
    let mut output = input.to_rgba8();

    // ── Detect edges ────────────────────────────────────────────────────────
    // Prefer ML edge map; fall back to Sobel when the model isn't available.
    // Phase 6: Sobel fallback now respects `config.teed_threshold` (was
    // hardcoded 0.3). The gradient is percentile-normalized (5/95) so the
    // threshold is on the same scale as ML edge probabilities.
    // Phase 8: if ML results exist but no edge_map, warn that Sobel fallback
    // is in use (the user enabled ML but the edge model didn't load).
    let edge_mask = if let Some(ml) = ml_results {
        if let Some(edge_map) = &ml.edge_map {
            downsample_edge_mask(edge_map, original_dims, width, height, config.teed_threshold)
        } else {
            let msg = "ML results present but edge_map missing; using Sobel fallback. \
                       (Edge detection model may not have loaded.)";
            log::warn!("{}", msg);
            warnings.push(msg.into());
            sobel_edge_mask(input, config.teed_threshold)
        }
    } else {
        // No ML at all — Sobel fallback. Don't warn here; the user explicitly
        // didn't run ML analysis. The pipeline caller can detect ml_results
        // absence separately if desired.
        sobel_edge_mask(input, config.teed_threshold)
    };

    // ── Skeletonize (C1) — thin to 1px-wide ridges ──────────────────────────
    let skeleton = skeletonize(&edge_mask, width, height);

    // ── Phase 3 — Cleanup: gap-bridging + noise drop + short-segment drop ───
    // Produces a cleaner line network with fewer single-pixel noise artifacts
    // and fewer broken-line gaps. Output is a binary mask (255 = edge, 0 = bg).
    let cleaned = cleanup_edge_mask(&skeleton, width, height, config.min_segment_length);

    // ── Edge-mode classification — Internal vs Outlines ─────────────────────
    // Priority 1: segmentation mask (foreground/bg boundary = outline).
    // Priority 2: SLIC labels (label span ≥2 = outline).
    // Phase 8: pass `warnings` so the Internal-mode fallback can push.
    let slic_labels: Option<&[u32]> = ml_results
        .and_then(|ml| ml.slic_labels.as_deref());
    let seg_mask: Option<&[f32]> = ml_results
        .and_then(|ml| ml.segmentation_mask.as_deref());
    let classification = classify_edges(
        &cleaned, slic_labels, seg_mask, width, height, config.edge_mode, warnings,
    );

    // ── Phase 4 — Compositing with chosen outline style ─────────────────────
    let palette_lab = PaletteLab::from_palette(palette);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if !classification[idx] {
                continue;
            }

            let edge_color = compute_outline_color(
                &output, x, y, width, height, config, &palette_lab,
            );

            // Phase 2 — C11: centered thickness (was asymmetric bottom-right).
            // thickness=1 → 1px; thickness=2 → 3×3 with corners skipped (cross);
            // thickness=3 → 3×3 solid; thickness=4 → 5×5 cross.
            draw_thickness(&mut output, x, y, width, height, config.thickness, edge_color);
        }
    }

    // ── Edge-direction AA (C10) ─────────────────────────────────────────────
    if config.anti_alias_edges {
        anti_alias_edges_directional(&mut output, &cleaned, width, height);
    }

    Ok(DynamicImage::ImageRgba8(output))
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge mask downsampling — max-pool + NMS + widened hysteresis (C5)
// ─────────────────────────────────────────────────────────────────────────────

/// Downsample a full-resolution edge probability map to the target
/// (pixel-art) dimensions and threshold into a binary mask.
///
/// # Phase 2 algorithm
///
/// 1. **Max-pool** the edge map to pixel-art resolution. A 16×16 source
///    block contributing even one strong edge pixel now produces a strong
///    output cell — average-pooling diluted sparse 1px lines below the
///    threshold, which was the source of "edges break midway".
/// 2. **3×3 non-max suppression**: any cell that has a stronger neighbor
///    gets suppressed. Thins cluster blobs back to single-pixel ridges.
/// 3. **Hysteresis thresholding** with widened band:
///    - `low  = threshold * 0.4`
///    - `high = threshold`
///    A weak edge survives if connected (8-conn) to a strong seed. The
///    widened band lets single-pixel gaps in a strong chain be bridged.
///
/// Returns a binary mask: 255 = edge, 0 = no edge.
fn downsample_edge_mask(
    edge_map: &[f32],
    orig_dims: (u32, u32),
    target_w: u32,
    target_h: u32,
    threshold: f32,
) -> Vec<u8> {
    let (orig_w, orig_h) = (orig_dims.0 as usize, orig_dims.1 as usize);
    let (tw, th) = (target_w as usize, target_h as usize);

    // ── Step 1: Max-pool to pixel-art resolution ──────────────────────────────
    let mut max_map = vec![0.0f32; tw * th];

    let scale_x = orig_w as f32 / tw as f32;
    let scale_y = orig_h as f32 / th as f32;

    for ty in 0..th {
        for tx in 0..tw {
            let ox_start = (tx as f32 * scale_x).floor() as usize;
            let oy_start = (ty as f32 * scale_y).floor() as usize;
            let ox_end = (((tx + 1) as f32 * scale_x).ceil() as usize).min(orig_w);
            let oy_end = (((ty + 1) as f32 * scale_y).ceil() as usize).min(orig_h);

            let mut mx = 0.0f32;
            for oy in oy_start..oy_end {
                for ox in ox_start..ox_end {
                    let idx = oy * orig_w + ox;
                    if idx < edge_map.len() {
                        let v = edge_map[idx];
                        if v > mx { mx = v; }
                    }
                }
            }
            max_map[ty * tw + tx] = mx;
        }
    }

    // ── Step 2: 3×3 non-max suppression ──────────────────────────────────────
    // Suppress a cell if any of its 8 neighbors is strictly greater. Cells
    // that are local maxima (or equal to the local max) survive.
    let nms = nms_3x3(&max_map, tw, th);

    // ── Step 3: Hysteresis thresholding with widened band ────────────────────
    // low = threshold * 0.4, high = threshold. Weak edges connected to a
    // strong seed survive; isolated noise drops.
    let low_thresh = threshold * 0.4;
    let high_thresh = threshold;

    let mut labels = vec![0u8; tw * th]; // 0 = none, 1 = weak, 2 = strong
    for (i, &v) in nms.iter().enumerate() {
        if v >= high_thresh {
            labels[i] = 2;
        } else if v >= low_thresh {
            labels[i] = 1;
        }
    }

    // BFS from strong edges — keep connected weak edges
    let mut mask = vec![0u8; tw * th];
    let mut queue: Vec<usize> = Vec::new();

    for (i, &label) in labels.iter().enumerate() {
        if label == 2 {
            mask[i] = 255;
            queue.push(i);
        }
    }

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

/// 3×3 non-max suppression. Suppresses any cell that has a strictly greater
/// 8-neighbor. Cells equal to the local max survive (so wide plateaus don't
/// collapse to a single pixel — Zhang-Suen handles that).
fn nms_3x3(values: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let v = values[idx];
            let mut is_max = true;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let nv = values[(ny as usize) * w + (nx as usize)];
                    if nv > v {
                        is_max = false;
                        break;
                    }
                }
                if !is_max { break; }
            }
            out[idx] = if is_max { v } else { 0.0 };
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Zhang-Suen skeletonization (C1)
// ─────────────────────────────────────────────────────────────────────────────

/// Thin a binary edge mask to single-pixel-wide ridges via the Zhang-Suen
/// algorithm. Two sub-iterations per pass: A (removes south/east "stair"
/// pixels) and B (removes north/west). Repeats until no changes occur.
///
/// Input: 0 = no edge, nonzero = edge.
/// Output: 255 = skeleton pixel, 0 = background.
fn skeletonize(mask: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if mask.len() < n { return mask.to_vec(); }

    // Working buffer: 1 = edge, 0 = background.
    let mut img: Vec<u8> = mask.iter().map(|&v| if v > 0 { 1 } else { 0 }).collect();

    loop {
        let mut removed = 0u32;

        // Sub-iteration A
        let to_remove_a = zhang_suen_pass(&img, w, h, true);
        for &idx in &to_remove_a {
            img[idx] = 0;
        }
        removed += to_remove_a.len() as u32;

        // Sub-iteration B
        let to_remove_b = zhang_suen_pass(&img, w, h, false);
        for &idx in &to_remove_b {
            img[idx] = 0;
        }
        removed += to_remove_b.len() as u32;

        if removed == 0 { break; }
    }

    img.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — Edge mask cleanup (gap-bridging + noise drop + short-segment drop)
// ─────────────────────────────────────────────────────────────────────────────

/// Clean up a skeletonized edge mask.
///
/// Three sub-stages, applied in order:
///
/// 1. **Gap-bridging**: for each skeleton pixel, look in a 5×5 neighborhood
///    for other skeleton pixels that aren't 8-connected to it. If found,
///    draw a Bresenham line connecting them. One pass only — avoids
///    over-connecting distinct features.
///
/// 2. **Isolated-noise drop**: remove skeleton pixels with zero skeleton
///    neighbors in 8-connectivity. These are single-pixel noise that
///    can't be part of a real line.
///
/// 3. **Short-segment drop**: run connected-components labeling (8-conn)
///    on the skeleton and drop components shorter than `min_segment_length`
///    pixels. Removes small noise clusters that survived step 2.
///
/// Input/output: binary mask, 255 = edge, 0 = background.
fn cleanup_edge_mask(
    skeleton: &[u8],
    width: u32,
    height: u32,
    min_segment_length: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if skeleton.len() < n {
        return skeleton.to_vec();
    }

    // Working copy as bool for efficiency.
    let mut mask: Vec<bool> = skeleton.iter().map(|&v| v > 0).collect();

    // ── Step 1: Gap-bridging (single pass) ──────────────────────────────────
    // Snapshot before modification so we read original skeleton state.
    let snap = mask.clone();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !snap[idx] { continue; }

            // Look in 5×5 (Chebyshev distance 2) for other skeleton pixels
            // that aren't 8-connected (distance > 1).
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    if dx.abs() <= 1 && dy.abs() <= 1 { continue; }  // skip 3×3
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let nidx = (ny as usize) * w + (nx as usize);
                    if !snap[nidx] { continue; }

                    // Found a candidate endpoint within distance 2. Bridge
                    // with a Bresenham line (interior pixels get set).
                    draw_line_bresenham(&mut mask, x as i32, y as i32, nx, ny, w, h);
                }
            }
        }
    }

    // ── Step 2: Isolated-noise drop ─────────────────────────────────────────
    let snap = mask.clone();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !snap[idx] { continue; }

            let mut has_neighbor = false;
            'neigh: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let nidx = (ny as usize) * w + (nx as usize);
                    if snap[nidx] {
                        has_neighbor = true;
                        break 'neigh;
                    }
                }
            }
            if !has_neighbor {
                mask[idx] = false;
            }
        }
    }

    // ── Step 3: Short-segment drop (connected-components labeling, 8-conn) ──
    if min_segment_length > 1 {
        let mut labels: Vec<u32> = vec![0; n];
        let mut component_sizes: Vec<u32> = vec![0];  // label 0 = background
        let mut next_label: u32 = 1;
        let mut queue: Vec<usize> = Vec::new();

        for seed in 0..n {
            if !mask[seed] || labels[seed] != 0 { continue; }

            // BFS the connected component starting at `seed`.
            let label = next_label;
            next_label += 1;
            component_sizes.push(0);
            let mut size = 0u32;

            queue.clear();
            queue.push(seed);
            labels[seed] = label;

            while let Some(i) = queue.pop() {
                size += 1;
                let cx = i % w;
                let cy = i / w;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                        let ni = (ny as usize) * w + (nx as usize);
                        if mask[ni] && labels[ni] == 0 {
                            labels[ni] = label;
                            queue.push(ni);
                        }
                    }
                }
            }

            component_sizes[label as usize] = size;
        }

        // Drop components shorter than the threshold.
        for i in 0..n {
            let lbl = labels[i] as usize;
            if lbl == 0 { continue; }
            if component_sizes[lbl] < min_segment_length {
                mask[i] = false;
            }
        }
    }

    mask.iter().map(|&v| if v { 255 } else { 0 }).collect()
}

/// Draw a Bresenham line between two points, setting interior pixels to `true`.
/// Endpoints are assumed already set by the caller. Used by gap-bridging.
fn draw_line_bresenham(
    mask: &mut [bool],
    x0: i32, y0: i32,
    x1: i32, y1: i32,
    w: usize, h: usize,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let idx = (y as usize) * w + (x as usize);
            mask[idx] = true;
        }
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// One sub-iteration of Zhang-Suen. Returns the list of pixel indices to
/// remove. `sub_a = true` for sub-iteration A, `false` for B.
///
/// Pixel labeling (P8 neighbors, P2 = north, P4 = east, P6 = south, P8 = west):
/// ```text
///   P9 P2 P3
///   P8 P1 P4
///   P7 P6 P5
/// ```
fn zhang_suen_pass(img: &[u8], w: usize, h: usize, sub_a: bool) -> Vec<usize> {
    let mut remove = Vec::new();

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            if img[idx] == 0 { continue; }

            // 8-neighborhood (clockwise from north)
            let p2 = img[(y - 1) * w + x] as u32;
            let p3 = img[(y - 1) * w + (x + 1)] as u32;
            let p4 = img[y * w + (x + 1)] as u32;
            let p5 = img[(y + 1) * w + (x + 1)] as u32;
            let p6 = img[(y + 1) * w + x] as u32;
            let p7 = img[(y + 1) * w + (x - 1)] as u32;
            let p8 = img[y * w + (x - 1)] as u32;
            let p9 = img[(y - 1) * w + (x - 1)] as u32;

            // Condition (a): 2 ≤ B(P1) ≤ 6, where B = count of black neighbors
            let b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            if b < 2 || b > 6 { continue; }

            // Condition (b): A(P1) = 1, where A = count of 0→1 transitions
            // in the clockwise sequence P2,P3,P4,P5,P6,P7,P8,P9,P2
            let seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
            let mut a = 0u32;
            for i in 0..8 {
                if seq[i] == 0 && seq[i + 1] == 1 { a += 1; }
            }
            if a != 1 { continue; }

            // Condition (c)/(d): differ between sub-iterations A and B
            if sub_a {
                // A: P2 * P4 * P6 == 0 AND P4 * P6 * P8 == 0
                if p2 * p4 * p6 != 0 { continue; }
                if p4 * p6 * p8 != 0 { continue; }
            } else {
                // B: P2 * P4 * P8 == 0 AND P2 * P6 * P8 == 0
                if p2 * p4 * p8 != 0 { continue; }
                if p2 * p6 * p8 != 0 { continue; }
            }

            remove.push(idx);
        }
    }

    remove
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge-mode classification (C9)
// ─────────────────────────────────────────────────────────────────────────────

/// Classify each skeleton pixel as keep/drop based on `EdgeMode` and SLIC labels.
///
/// Returns a `Vec<bool>` (one per pixel, row-major) — `true` = render this
/// edge pixel, `false` = skip it.
///
/// - `EdgeMode::None`: handled by `draw_edges` early-return; never reaches here.
/// - `EdgeMode::Outlines`: keep only pixels whose 3×3 neighborhood spans
///   ≥2 SLIC labels (silhouette edges at region boundaries).
/// - `EdgeMode::Internal`: keep only pixels whose 3×3 neighborhood spans
///   exactly 1 SLIC label (within-region detail).
/// - `EdgeMode::Both`: keep all skeleton pixels.
///
/// When SLIC labels are unavailable:
/// - `Outlines` and `Both` fall back to keeping all skeleton pixels (every
///   edge is treated as an outline — backward-compatible).
/// - `Internal` falls back to border-touching edges only (silhouette edges
///   detected without region labels) and emits a warning. Previously it
///   silently behaved like `Both`, which was incorrect — internal edges
///   cannot be identified without region labels. Phase 7 fix.
///   Phase 8: the warning is now also pushed to `warnings` so the UI can
///   surface it (was previously only `log::warn!`).
fn classify_edges(
    skeleton: &[u8],
    slic_labels: Option<&[u32]>,
    seg_mask: Option<&[f32]>,
    width: u32,
    height: u32,
    mode: EdgeMode,
    warnings: &mut Vec<String>,
) -> Vec<bool> {
    let n = (width * height) as usize;
    let mut keep = vec![false; n];

    let w = width as i32;
    let h = height as i32;

    // ── Priority 1: Segmentation mask ──────────────────────────────────────
    // When the segmentation mask is available, outline = edge pixel whose 3×3
    // neighborhood spans the foreground/background boundary. Internal = edge
    // pixel entirely within foreground. This is far more reliable than SLIC
    // label span because the mask directly encodes the character boundary.
    if let Some(mask) = seg_mask {
        if mask.len() >= n {
            for y in 0..h {
                for x in 0..w {
                    let idx = (y as usize) * (w as usize) + (x as usize);
                    if skeleton[idx] == 0 { continue; }

                    // Check if 3×3 neighborhood spans fg/bg boundary.
                    let mut has_fg = false;
                    let mut has_bg = false;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
                            let nidx = (ny as usize) * (w as usize) + (nx as usize);
                            if mask[nidx] >= 0.5 {
                                has_fg = true;
                            } else {
                                has_bg = true;
                            }
                        }
                    }

                    let is_outline = has_fg && has_bg;
                    let keep_it = match mode {
                        EdgeMode::None     => false,
                        EdgeMode::Outlines => is_outline,
                        EdgeMode::Internal => !is_outline && has_fg,
                        EdgeMode::Both     => true,
                    };
                    keep[idx] = keep_it;
                }
            }
            return keep;
        }
    }

    // ── Priority 2: SLIC labels (fallback) ──────────────────────────────────
    let labels = match slic_labels {
        Some(l) if l.len() >= n => l,
        _ => {
            // No SLIC labels — mode-aware fallback (Phase 7).
            match mode {
                EdgeMode::Outlines | EdgeMode::Both => {
                    // No SLIC → treat all edges as outlines (backward-compat).
                    for i in 0..n {
                        keep[i] = skeleton[i] > 0;
                    }
                }
                EdgeMode::Internal => {
                    // No SLIC → can't identify internal edges.
                    // Fall back to border-touching edges only (detectable
                    // without SLIC) and warn. Phase 7 fix: previously
                    // this silently behaved like Both.
                    // Phase 8: also push to `warnings` so the UI surfaces it.
                    let msg = "EdgeMode::Internal requires SLIC labels or segmentation mask; \
                               falling back to border-touching edges only.";
                    log::warn!("{}", msg);
                    warnings.push(msg.into());
                    let wu = w as usize;
                    let hu = h as usize;
                    for y in 0..hu {
                        for x in 0..wu {
                            let idx = y * wu + x;
                            if skeleton[idx] == 0 { continue; }
                            // Border-touching = pixel is on image edge OR
                            // has a non-skeleton 4-neighbor (silhouette).
                            let on_border = x == 0 || y == 0
                                || x == wu - 1 || y == hu - 1;
                            let has_bg_neighbor = (x > 0 && skeleton[idx - 1] == 0)
                                || (x < wu - 1 && skeleton[idx + 1] == 0)
                                || (y > 0 && skeleton[idx - wu] == 0)
                                || (y < hu - 1 && skeleton[idx + wu] == 0);
                            keep[idx] = on_border || has_bg_neighbor;
                        }
                    }
                }
                EdgeMode::None => {}  // already handled by early return in draw_edges
            }
            return keep;
        }
    };

    for y in 0..h {
        for x in 0..w {
            let idx = (y as usize) * (w as usize) + (x as usize);
            if skeleton[idx] == 0 { continue; }

            // Span of distinct labels in 3×3 neighborhood
            let mut first_label: Option<u32> = None;
            let mut spans_multiple = false;
            'neigh: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
                    let nidx = (ny as usize) * (w as usize) + (nx as usize);
                    let lbl = labels[nidx];
                    match first_label {
                        None => first_label = Some(lbl),
                        Some(l) if l != lbl => {
                            spans_multiple = true;
                            break 'neigh;
                        }
                        _ => {}
                    }
                }
            }

            let is_outline = spans_multiple;
            let keep_it = match mode {
                EdgeMode::None     => false,
                EdgeMode::Outlines => is_outline,
                EdgeMode::Internal => !is_outline,
                EdgeMode::Both     => true,
            };
            keep[idx] = keep_it;
        }
    }

    keep
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4 — Outline color: LocalColorShift / Black / MaxContrast
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the outline color for an edge pixel.
///
/// Phase 4: three explicit modes (see `OutlineStyle`).
///
/// - `LocalColorShift` (default): edge color = local 3×3 mean Lab, shifted
///   darker by `edge_l_shift` L* units and optionally hue-rotated by
///   `edge_hue_shift` degrees, then snapped to nearest palette entry.
///   Subtle, follows local color family.
/// - `Black`: always use the palette's darkest color (lowest Lab L*).
///   For strict retro palettes (Game Boy, NES).
/// - `MaxContrast`: the old AutoContrast behavior — max-ΔE from local mean,
///   optionally blended toward darkest by `edge_darkener_strength`. Loud.
fn compute_outline_color(
    image: &RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    config: &EdgeConfig,
    palette_lab: &PaletteLab,
) -> Rgba<u8> {
    if palette_lab.rgb.is_empty() {
        return Rgba([0, 0, 0, 255]);
    }

    match config.outline_style {
        OutlineStyle::Black => {
            darkest_palette_color(palette_lab)
        }

        OutlineStyle::MaxContrast => {
            // Old AutoContrast: max ΔE from local mean.
            let mean_lab = neighborhood_mean_lab(image, x, y, width, height);
            let local_contrast = palette_lab.furthest_from(mean_lab);

            // ET-5(b): blend local-contrast with darkest palette color.
            let t = config.edge_darkener_strength.clamp(0.0, 1.0);
            if t < 0.001 {
                local_contrast
            } else {
                blend_rgb(local_contrast, darkest_palette_color(palette_lab), t)
            }
        }

        OutlineStyle::LocalColorShift => {
            // Local 3×3 mean Lab, shifted darker + optional hue rotation,
            // then snapped to nearest palette entry.
            let mean_lab = neighborhood_mean_lab(image, x, y, width, height);
            let shifted = shift_lab(
                mean_lab,
                -config.edge_l_shift,  // negative = darker
                config.edge_hue_shift,
            );
            palette_lab.nearest_to(shifted)
        }
    }
}

/// Return the palette entry with the lowest Lab L* (darkest color).
/// Used by `Black` and `MaxContrast` modes.
fn darkest_palette_color(palette_lab: &PaletteLab) -> Rgba<u8> {
    if palette_lab.rgb.is_empty() {
        return Rgba([0, 0, 0, 255]);
    }
    let mut darkest_idx = 0usize;
    let mut darkest_l = f32::MAX;
    for (i, &lab) in palette_lab.lab.iter().enumerate() {
        if lab.l < darkest_l {
            darkest_l = lab.l;
            darkest_idx = i;
        }
    }
    palette_lab.rgb[darkest_idx]
}

/// Shift a Lab color: apply `dl` to L* (clamped to [0, 100]) and rotate the
/// a*b* hue by `hue_deg` degrees (counter-clockwise in the a*b* plane).
///
/// Fast path: if `hue_deg` is approximately 0, skip the polar conversion.
fn shift_lab(lab: Lab, dl: f32, hue_deg: f32) -> Lab {
    let new_l = (lab.l + dl).clamp(0.0, 100.0);

    if hue_deg.abs() < 0.001 {
        return Lab::new(new_l, lab.a, lab.b);
    }

    // Convert (a, b) → polar (C, h), rotate h, convert back.
    let a = lab.a as f64;
    let b = lab.b as f64;
    let chroma = (a * a + b * b).sqrt();
    let hue = b.atan2(a);  // radians
    let rotated = hue + (hue_deg as f64).to_radians();
    let new_a = (chroma * rotated.cos()) as f32;
    let new_b = (chroma * rotated.sin()) as f32;

    Lab::new(new_l, new_a, new_b)
}

/// Linear RGB blend: `result = a * (1 - t) + b * t`.
/// Used by ET-5(b) to bias local-contrast outline color toward darkest.
#[inline]
fn blend_rgb(a: Rgba<u8>, b: Rgba<u8>, t: f32) -> Rgba<u8> {
    let inv_t = 1.0 - t;
    Rgba([
        (a[0] as f32 * inv_t + b[0] as f32 * t).round().clamp(0.0, 255.0) as u8,
        (a[1] as f32 * inv_t + b[1] as f32 * t).round().clamp(0.0, 255.0) as u8,
        (a[2] as f32 * inv_t + b[2] as f32 * t).round().clamp(0.0, 255.0) as u8,
        255,
    ])
}

/// Compute the mean Lab color of the 3×3 neighborhood centered at (x, y).
/// Edges of image clamp to nearest valid pixel (so corners still get a
/// meaningful mean rather than skipping).
fn neighborhood_mean_lab(
    image: &RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Lab {
    let mut sum_l = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut count = 0u32;

    let x0 = x.saturating_sub(1);
    let x1 = (x + 1).min(width - 1);
    let y0 = y.saturating_sub(1);
    let y1 = (y + 1).min(height - 1);

    for ny in y0..=y1 {
        for nx in x0..=x1 {
            let p = image.get_pixel(nx, ny);
            let lab = rgb_to_lab(*p);
            sum_l += lab.l;
            sum_a += lab.a;
            sum_b += lab.b;
            count += 1;
        }
    }

    let c = count.max(1) as f32;
    Lab::new(sum_l / c, sum_a / c, sum_b / c)
}

// ─────────────────────────────────────────────────────────────────────────────
// Centered thickness drawing (C11)
// ─────────────────────────────────────────────────────────────────────────────

/// Draw an edge pixel with the requested thickness, centered on (x, y).
///
/// Phase 2 — C11 fix: the old code did `for ty in 0..thickness` which grew
/// the stroke only toward bottom-right (1px top-left, Npx bottom-right).
/// The new code uses a symmetric kernel:
/// - thickness=1 → single pixel
/// - thickness=2 → 3×3 cross (center + 4-conn neighbors)
/// - thickness=3 → 3×3 solid
/// - thickness=4 → 5×5 cross
///
/// Corners are skipped for even thicknesses to keep strokes looking like
/// lines, not squares.
fn draw_thickness(
    output: &mut RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    thickness: u32,
    color: Rgba<u8>,
) {
    match thickness {
        1 => {
            put_pixel_safe(output, x, y, width, height, color);
        }
        2 => {
            // 3×3 cross
            for &(dx, dy) in &[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                put_pixel_safe_offset(output, x, y, dx, dy, width, height, color);
            }
        }
        3 => {
            // 3×3 solid
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    put_pixel_safe_offset(output, x, y, dx, dy, width, height, color);
                }
            }
        }
        _ => {
            // thickness ≥ 4: 5×5 cross (manhattan distance ≤ thickness/2)
            let radius = (thickness as i32) / 2;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx.abs() + dy.abs() <= radius {
                        put_pixel_safe_offset(output, x, y, dx, dy, width, height, color);
                    }
                }
            }
        }
    }
}

#[inline]
fn put_pixel_safe(
    output: &mut RgbaImage,
    x: u32, y: u32,
    width: u32, height: u32,
    color: Rgba<u8>,
) {
    if x < width && y < height {
        output.put_pixel(x, y, color);
    }
}

#[inline]
fn put_pixel_safe_offset(
    output: &mut RgbaImage,
    x: u32, y: u32,
    dx: i32, dy: i32,
    width: u32, height: u32,
    color: Rgba<u8>,
) {
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
        output.put_pixel(nx as u32, ny as u32, color);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sobel fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Sobel-based binary edge mask — used when the ML edge model is unavailable.
///
/// Phase 6: percentile-normalizes the gradient (5/95) before thresholding so
/// the threshold is on the same scale as ML edge probabilities. Previously
/// the gradient was raw magnitude / 255 (different distribution than ML
/// probabilities), and the threshold was hardcoded to 0.3 — so the user's
/// `teed_threshold` slider did nothing when ML was unavailable.
fn sobel_edge_mask(input: &DynamicImage, threshold: f32) -> Vec<u8> {
    let gradient = sobel_gradient(input);
    let normalized = percentile_normalize_5_95(&gradient);
    normalized.iter().map(|&g| if g > threshold { 255 } else { 0 }).collect()
}

/// Percentile-normalize a slice to [0, 1] using the 5th and 95th percentiles
/// as endpoints (robust to outliers, same scheme as ML edge map normalization
/// in `ml/models/teed.rs`).
fn percentile_normalize_5_95(values: &[f32]) -> Vec<f32> {
    if values.is_empty() { return Vec::new(); }
    // `percentiles` is `pub(crate)` in `depth_to_flat.rs`, re-exported at
    // `crate::processing::*` via `pub use depth_to_flat::*`.
    let (p5, p95) = crate::processing::percentiles(values, 0.05, 0.95);
    let range = p95 - p5;
    if range < 1e-6 {
        return vec![0.0f32; values.len()];
    }
    values.iter()
        .map(|&v| ((v - p5) / range).clamp(0.0, 1.0))
        .collect()
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

// ─────────────────────────────────────────────────────────────────────────────
// Edge-direction-aware anti-aliasing (C10)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply edge-direction-aware AA: for each skeleton pixel, blend the edge
/// color toward the background in the direction **perpendicular** to the
/// line's local tangent. Interior palette pixels are not touched.
///
/// The local tangent is estimated by looking at which 8-connected neighbors
/// are also skeleton pixels. If most neighbors are horizontal, the line is
/// horizontal → AA vertically. If most are vertical → AA horizontally.
/// Diagonal lines get 2-tap AA on both axes.
///
/// This replaces the old `anti_alias_edges` which was a full-image 3×3 box
/// blur — it destroyed palette quantization across the entire output.
fn anti_alias_edges_directional(
    image: &mut RgbaImage,
    skeleton: &[u8],
    width: u32,
    height: u32,
) {
    let w = width as i32;
    let h = height as i32;

    // Work on a snapshot so we read original colors while writing AA blends.
    let snapshot = image.clone();

    for y in 0..h {
        for x in 0..w {
            let idx = (y as usize) * (w as usize) + (x as usize);
            if skeleton[idx] == 0 { continue; }

            // Count skeleton neighbors in each axis
            let mut horiz = 0i32; // east+west neighbors
            let mut vert = 0i32;  // north+south neighbors
            for &(dx, dy) in &[(1, 0), (-1, 0)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx >= 0 && ny >= 0 && nx < w && ny < h {
                    let nidx = (ny as usize) * (w as usize) + (nx as usize);
                    if skeleton[nidx] > 0 { horiz += 1; }
                }
            }
            for &(dx, dy) in &[(0, 1), (0, -1)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx >= 0 && ny >= 0 && nx < w && ny < h {
                    let nidx = (ny as usize) * (w as usize) + (nx as usize);
                    if skeleton[nidx] > 0 { vert += 1; }
                }
            }

            // AA perpendicular to the dominant line direction.
            // horiz ≥ vert → line is horizontal → AA north & south.
            // vert > horiz → line is vertical → AA east & west.
            let edge_color = *image.get_pixel(x as u32, y as u32);

            if horiz >= vert && vert == 0 {
                // Pure horizontal line: AA north + south
                blend_neighbor(&snapshot, image, x, y, w, h, 0, -1, edge_color, 0.35);
                blend_neighbor(&snapshot, image, x, y, w, h, 0,  1, edge_color, 0.35);
            } else if vert > horiz && horiz == 0 {
                // Pure vertical line: AA east + west
                blend_neighbor(&snapshot, image, x, y, w, h, -1, 0, edge_color, 0.35);
                blend_neighbor(&snapshot, image, x, y, w, h,  1, 0, edge_color, 0.35);
            } else {
                // Diagonal / junction: 2-tap on both axes, weaker blend
                blend_neighbor(&snapshot, image, x, y, w, h, -1, 0, edge_color, 0.2);
                blend_neighbor(&snapshot, image, x, y, w, h,  1, 0, edge_color, 0.2);
                blend_neighbor(&snapshot, image, x, y, w, h, 0, -1, edge_color, 0.2);
                blend_neighbor(&snapshot, image, x, y, w, h, 0,  1, edge_color, 0.2);
            }
        }
    }
}

/// Blend `edge_color` into the pixel at `(x+dx, y+dy)` with weight `t`.
/// Background color comes from `snapshot` (the pre-AA image). Output goes
/// into `image`. Skips out-of-bounds and skips pixels that are themselves
/// skeleton (we don't want to AA over another edge pixel).
#[inline]
fn blend_neighbor(
    snapshot: &RgbaImage,
    image: &mut RgbaImage,
    x: i32, y: i32,
    w: i32, h: i32,
    dx: i32, dy: i32,
    edge_color: Rgba<u8>,
    t: f32,
) {
    let nx = x + dx;
    let ny = y + dy;
    if nx < 0 || ny < 0 || nx >= w || ny >= h { return; }
    let nidx = (ny as usize) * (w as usize) + (nx as usize);
    // Don't AA over another edge pixel
    // (we can't see `skeleton` here, but `image` already has the edge color
    // painted — if the neighbor's color equals edge_color, skip).
    let bg = snapshot.get_pixel(nx as u32, ny as u32);
    let out = image.get_pixel_mut(nx as u32, ny as u32);
    let _ = nidx; // (kept for future use; the equality check below is sufficient)
    if *out == edge_color { return; }
    out[0] = (bg[0] as f32 * (1.0 - t) + edge_color[0] as f32 * t).round() as u8;
    out[1] = (bg[1] as f32 * (1.0 - t) + edge_color[1] as f32 * t).round() as u8;
    out[2] = (bg[2] as f32 * (1.0 - t) + edge_color[2] as f32 * t).round() as u8;
}

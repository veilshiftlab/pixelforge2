//! SLIC superpixels — model-free region classification
//!
//! # Phase 3 — Gradient-aware features + region growing
//!
//! The 6D feature vector `(L, a, b, depth, x·s, y·s)` only responded to
//! hard color/depth steps. Smooth gradient transitions (cheek shading,
//! hair strand flow, soft depth roll-off) didn't create region boundaries
//! because the centroid's color match was "close enough" across the
//! transition. This is the user's "SLIC doesn't group based on colors
//! and gradients" complaint.
//!
//! ## Phase 3 changes
//!
//! 1. **10D feature vector** `(L, a, b, depth, dL/dx, dL/dy, dDepth/dx,
//!    dDepth/dy, x·s, y·s)`. The four new gradient terms make SLIC follow
//!    gradient transitions, not just hard steps. Each gradient term is
//!    normalized by its 95th percentile so a single sharp edge doesn't
//!    dominate the distance metric.
//! 2. **Region-growing post-processing**: after k-means converges, run a
//!    4-connectivity pass on each cluster. Any cluster with >1 disconnected
//!    component gets split into separate labels. Eliminates the
//!    "salt-and-pepper" assignments where a centroid's color matches a
//!    far-away region (e.g., a forehead centroid matching a far-off
//!    background blob of similar color).
//!
//! # Algorithm
//!
//! 1. Build the 10D feature vector per pixel.
//! 2. Initialize K cluster centers by random sampling (deterministic seed).
//! 3. Run 10 iterations of k-means (assign + update).
//! 4. Split disconnected components into fresh cluster IDs.
//! 5. Return the label map (one cluster ID per pixel).
//!
//! # Parameters
//!
//! - `K` (default 5, range 3–8): cluster count. Higher = more regions.
//! - `spatial_weight` (default 0.5, range 0–1): controls blobbiness vs.
//!   detail fidelity. Higher = blobbier regions (boundaries follow the
//!   spatial grid). Lower = regions follow color/depth/gradient boundaries
//!   more faithfully.
//!
//! # Normalization
//!
//! All feature components are normalized to ~[-1, 1] so no single dimension
//! dominates the Euclidean distance:
//! - L → /50  (range [-1, 1] centered on mid-gray; default 0.5 was too
//!   biased toward dark pixels)
//! - a, b → /128 (range [-1, 1])
//! - depth → (d - 0.5) * 2 (range [-1, 1])
//! - dL/dx, dL/dy → /p95_L_grad (95th percentile of |dL| gradients)
//! - dDepth/dx, dDepth/dy → /p95_D_grad (95th percentile of |dDepth| gradients)
//! - x, y → /max_dim × `spatial_weight` × 2 - 1 (range [-1, 1])

use crate::processing::SlicConfig;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba};
use palette::{IntoColor, Lab, Srgb};
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// 10D feature vector (Phase 3):
/// `(L, a, b, depth, dL/dx, dL/dy, dDepth/dx, dDepth/dy, x·s, y·s)`
type Feature = [f32; 10];

/// Run SLIC superpixel clustering on an image with an optional depth map.
///
/// Returns a `Vec<u32>` of cluster IDs, one per pixel, row-major, at the
/// input image's resolution.
///
/// If `depth_map` is `None`, the depth component is set to 0.5 (mid-gray)
/// for all pixels — SLIC still works, just without depth-informed boundaries.
pub fn slic(
    image: &DynamicImage,
    depth_map: Option<&[f32]>,
    config: &SlicConfig,
) -> Result<Vec<u32>> {
    let (width, height) = image.dimensions();
    let n_pixels = (width * height) as usize;

    if n_pixels == 0 {
        return Ok(Vec::new());
    }

    let k = config.k.clamp(5, 128) as usize;
    let s = config.spatial_weight.clamp(0.0, 1.0);

    // ── Build feature vectors (10D, with gradient terms) ────────────────────
    let features = build_features(image, depth_map, width, height, s);

    // ── Initialize cluster centers (deterministic random sampling) ──────────
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut centroids: Vec<Feature> = if features.len() >= k {
        features.choose_multiple(&mut rng, k).copied().collect()
    } else {
        features.clone()
    };
    let actual_k = centroids.len();

    // ── Phase 6 — P2: SLIC 2S×2S search window setup ────────────────────────
    // Standard SLIC: each centroid only searches pixels within a 2S×2S window
    // where S = sqrt(N/K) is the expected cluster grid spacing. Reduces
    // assignment complexity from O(N*K) to O(N * 4) per iteration.
    // The windowed search requires tracking each centroid's pixel-space
    // position separately (feature vector stores normalized spatial coords).
    let grid_s = ((n_pixels as f32 / actual_k as f32).sqrt() as u32).max(1);
    let wi = width as i32;
    let hi = height as i32;

    // Recover initial centroid positions from their spatial feature components.
    // spatial_x = (x / max_dim) * 2.0 - 1.0) * spatial_weight
    // → x = ((spatial_x / spatial_weight) + 1.0) * 0.5 * max_dim
    let max_dim = width.max(height) as f32;
    let mut centroid_pos: Vec<(i32, i32)> = centroids.iter().map(|f| {
        let nx = if s > 1e-6 { (f[8] / s) + 1.0 } else { 1.0 };
        let ny = if s > 1e-6 { (f[9] / s) + 1.0 } else { 1.0 };
        let x = ((nx * 0.5) * max_dim).round() as i32;
        let y = ((ny * 0.5) * max_dim).round() as i32;
        (x.clamp(0, wi - 1), y.clamp(0, hi - 1))
    }).collect();

    // ── K-means iterations (windowed) ────────────────────────────────────────
    let mut labels = vec![0u32; n_pixels];

    // Performance: reuse the distances buffer across iterations instead of
    // allocating 4MB per iteration.
    let mut distances = vec![f32::MAX; n_pixels];
    let w_usize = width as usize;

    // Standard SLIC uses 3-5 iterations. Was 10, which is overkill and
    // doubles runtime. Early-exit on convergence still applies.
    for _iteration in 0..5 {
        // ── Assignment step: for each centroid, search only its 2S×2S window ─
        // Track minimum distance per pixel across all centroids.
        for d in distances.iter_mut() { *d = f32::MAX; }
        for (c_idx, centroid) in centroids.iter().enumerate() {
            let (cx, cy) = centroid_pos[c_idx];
            let x_start = (cx - grid_s as i32).max(0);
            let x_end   = (cx + grid_s as i32).min(wi - 1);
            let y_start = (cy - grid_s as i32).max(0);
            let y_end   = (cy + grid_s as i32).min(hi - 1);

            for y in y_start..=y_end {
                let row_start = (y as usize) * w_usize;
                for x in x_start..=x_end {
                    let i = row_start + (x as usize);
                    let d = squared_distance(&features[i], centroid);
                    if d < distances[i] {
                        distances[i] = d;
                        labels[i] = c_idx as u32;
                    }
                }
            }
        }

        // ── Update step: recompute centroids as cluster means ────────────────
        let mut sums = vec![[0.0f64; 10]; actual_k];
        let mut counts = vec![0u64; actual_k];
        let mut pos_sums = vec![[0.0f64; 2]; actual_k];

        for (i, feat) in features.iter().enumerate() {
            let c = labels[i] as usize;
            for j in 0..10 {
                sums[c][j] += feat[j] as f64;
            }
            counts[c] += 1;
            // Track pixel-space position for windowed search next iteration
            let px = (i % (width as usize)) as f64;
            let py = (i / (width as usize)) as f64;
            pos_sums[c][0] += px;
            pos_sums[c][1] += py;
        }

        let mut changed = false;
        for c in 0..actual_k {
            if counts[c] > 0 {
                let count = counts[c] as f64;
                for j in 0..10 {
                    let new_val = (sums[c][j] / count) as f32;
                    if (new_val - centroids[c][j]).abs() > 1e-6 {
                        changed = true;
                    }
                    centroids[c][j] = new_val;
                }
                // Update centroid pixel position for next iteration's windowed search
                centroid_pos[c] = (
                    ((pos_sums[c][0] / count).round() as i32).clamp(0, wi - 1),
                    ((pos_sums[c][1] / count).round() as i32).clamp(0, hi - 1),
                );
            }
        }

        // Early exit if converged
        if !changed && _iteration > 0 {
            break;
        }
    }

    // ── Phase 3 — C7: split disconnected cluster components ──────────────────
    // After k-means, a single cluster ID may label multiple disconnected
    // blobs of pixels (e.g., two patches of similar sky color in opposite
    // corners). Split each multi-component cluster into separate IDs so
    // downstream stages (bg classification, edge Internal-vs-Outlines)
    // see spatially-coherent regions.
    let labels = split_disconnected_clusters(labels, width, height);

    Ok(labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature construction (10D — Phase 3)
// ─────────────────────────────────────────────────────────────────────────────

/// Build the 10D feature vector for every pixel.
///
/// Phase 3 adds 4 gradient terms (dL/dx, dL/dy, dDepth/dx, dDepth/dy) so
/// SLIC follows gradient transitions, not just hard color/depth steps. Each
/// gradient term is normalized by its 95th percentile (computed once over
/// the whole image) so a single sharp edge doesn't dominate.
fn build_features(
    image: &DynamicImage,
    depth_map: Option<&[f32]>,
    width: u32,
    height: u32,
    spatial_weight: f32,
) -> Vec<Feature> {
    let rgba = image.to_rgba8();
    let n_pixels = (width * height) as usize;
    let w = width as usize;
    let h = height as usize;
    let max_dim = width.max(height) as f32;

    // ── Lab color per pixel (single conversion, not 3×) ──────────────────────
    // Performance fix: was calling rgba_to_lab() 3 times per pixel (once for
    // each of L, a, b). Now calls it once and splits the result. Also uses
    // raw pixel buffer indexing instead of get_pixel() to avoid bounds-check
    // overhead on 1M+ pixels.
    let mut lab_l = Vec::with_capacity(n_pixels);
    let mut lab_a = Vec::with_capacity(n_pixels);
    let mut lab_b = Vec::with_capacity(n_pixels);
    let rgba_buf = rgba.as_raw();  // &Vec<u8>, 4 bytes per pixel
    for i in 0..n_pixels {
        let off = i * 4;
        let p = Rgba([rgba_buf[off], rgba_buf[off+1], rgba_buf[off+2], rgba_buf[off+3]]);
        let lab = rgba_to_lab(p);
        lab_l.push(lab.l);
        lab_a.push(lab.a);
        lab_b.push(lab.b);
    }

    // ── Depth (with 0.5 fallback when missing) ────────────────────────────────
    let depth: Vec<f32> = (0..n_pixels)
        .map(|i| depth_map.and_then(|d| d.get(i).copied()).unwrap_or(0.5))
        .collect();

    // ── Compute gradients via forward differences (clamped at image edge) ─────
    let mut d_l_dx = vec![0.0f32; n_pixels];
    let mut d_l_dy = vec![0.0f32; n_pixels];
    let mut d_d_dx = vec![0.0f32; n_pixels];
    let mut d_d_dy = vec![0.0f32; n_pixels];

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let x1 = (x + 1).min(w - 1);
            let y1 = (y + 1).min(h - 1);
            d_l_dx[i] = (lab_l[y * w + x1] - lab_l[i]).abs();
            d_l_dy[i] = (lab_l[y1 * w + x] - lab_l[i]).abs();
            d_d_dx[i] = (depth[y * w + x1] - depth[i]).abs();
            d_d_dy[i] = (depth[y1 * w + x] - depth[i]).abs();
        }
    }

    // ── Normalize gradients by 95th percentile (robust to outliers) ──────────
    let p95_l = percentile(&d_l_dx, &d_l_dy, 0.95).max(1e-6);
    let p95_d = percentile(&d_d_dx, &d_d_dy, 0.95).max(1e-6);

    // ── Assemble 10D feature vector ───────────────────────────────────────────
    let mut features = Vec::with_capacity(n_pixels);
    for y in 0..height {
        for x in 0..width {
            let i = (y as usize) * w + (x as usize);

            // Spatial: normalize to [-1, 1] × spatial_weight
            let x_norm = ((x as f32 / max_dim) * 2.0 - 1.0) * spatial_weight;
            let y_norm = ((y as f32 / max_dim) * 2.0 - 1.0) * spatial_weight;

            features.push([
                (lab_l[i] / 50.0) - 1.0,            // L: [-1, 1] centered on mid-gray
                lab_a[i] / 128.0,                  // a: [-1, 1]
                lab_b[i] / 128.0,                  // b: [-1, 1]
                (depth[i] - 0.5) * 2.0,            // depth: [-1, 1]
                (d_l_dx[i] / p95_l).clamp(-1.0, 1.0),  // dL/dx: [-1, 1]
                (d_l_dy[i] / p95_l).clamp(-1.0, 1.0),  // dL/dy: [-1, 1]
                (d_d_dx[i] / p95_d).clamp(-1.0, 1.0),  // dDepth/dx: [-1, 1]
                (d_d_dy[i] / p95_d).clamp(-1.0, 1.0),  // dDepth/dy: [-1, 1]
                x_norm,                            // spatial x
                y_norm,                            // spatial y
            ]);
        }
    }

    features
}

/// Compute a single percentile across the union of two gradient slices.
/// Used to get one normalization factor for both dL/dx and dL/dy (they
/// share the same physical meaning — magnitude of L* gradient).
///
/// Performance: uses `select_nth_unstable_by` which is O(N) average
/// (quickselect partition) instead of O(N log N) full sort. On a 2M-element
/// array this is ~10× faster than sorting.
fn percentile(a: &[f32], b: &[f32], pct: f32) -> f32 {
    let mut combined: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
    if combined.is_empty() { return 1.0; }
    let idx = ((combined.len() as f32 - 1.0) * pct).round() as usize;
    let idx = idx.min(combined.len() - 1);
    // Partition so that the element at index `idx` is in its sorted position.
    combined.select_nth_unstable_by(idx, |x, y| {
        x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
    });
    combined[idx]
}

// ─────────────────────────────────────────────────────────────────────────────
// Region-growing post-processing (C7 — Phase 3)
// ─────────────────────────────────────────────────────────────────────────────

/// Split any cluster whose pixels form multiple disconnected components
/// into separate cluster IDs.
///
/// Uses 8-connectivity (N/NE/E/SE/S/SW/W/NW — includes diagonals).
/// ET-3/ET-4 fix: was 4-connectivity which fragmented k=32 clusters into
/// 4502 pieces on a 1024² image because diagonal-only connections were
/// treated as disconnected. 8-connectivity produces ~k to 2k regions.
///
/// New IDs are assigned starting from `max_existing_id + 1` to avoid
/// collisions with the original k-means IDs.
fn split_disconnected_clusters(mut labels: Vec<u32>, width: u32, height: u32) -> Vec<u32> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if n == 0 || labels.len() < n {
        return labels;
    }

    // Find the highest existing cluster ID so new IDs start above it.
    let max_existing_id = labels.iter().copied().max().unwrap_or(0);
    let mut next_id = max_existing_id + 1;

    // Visited mask so we don't process the same pixel twice.
    let mut visited = vec![false; n];

    // For each pixel, if not yet visited, BFS its 8-connected component.
    for seed in 0..n {
        if visited[seed] { continue; }
        let original_label = labels[seed];

        // BFS the connected component starting at `seed`.
        let mut queue: Vec<usize> = vec![seed];
        visited[seed] = true;
        let mut component_pixels: Vec<usize> = Vec::new();

        while let Some(i) = queue.pop() {
            component_pixels.push(i);
            let x = i % w;
            let y = i / w;

            // 8-connectivity neighbors (ET-3/ET-4 fix: was 4-connectivity)
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let ni = (ny as usize) * w + (nx as usize);
                    if visited[ni] { continue; }
                    if labels[ni] == original_label {
                        visited[ni] = true;
                        queue.push(ni);
                    }
                }
            }
        }

        // If this is the first component we've seen for `original_label`,
        // keep the original ID. Otherwise, reassign to a new ID.
        let first_pixel_of_label = (0..seed).all(|i| labels[i] != original_label);
        if !first_pixel_of_label {
            for &px in &component_pixels {
                labels[px] = next_id;
            }
            next_id += 1;
        }
    }

    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn squared_distance(a: &Feature, b: &Feature) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..10 {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn rgba_to_lab(rgba: Rgba<u8>) -> Lab {
    let rgb: Srgb = Srgb::new(
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    );
    rgb.into_color()
}

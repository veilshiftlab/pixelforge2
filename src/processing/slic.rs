//! SLIC superpixels — model-free region classification
//!
//! Simple Linear Iterative Clustering on the 6D feature vector
//! `(L, a, b, depth, x·s, y·s)`.
//!
//! Replaces BiSeNet segmentation (which doesn't work on anime-style images).
//! SLIC is domain-agnostic: it produces color-coherent AND depth-coherent AND
//! spatially contiguous regions, which is exactly what the downstream
//! per-region shading needs.
//!
//! # Algorithm
//!
//! 1. Build a 6D feature vector per pixel (Lab color + depth + spatial coords
//!    scaled by `spatial_weight`).
//! 2. Initialize K cluster centers by random sampling.
//! 3. Run 10 iterations of k-means (assign + update).
//! 4. Return the label map (one cluster ID per pixel).
//!
//! # Parameters
//!
//! - `K` (default 5, range 3–8): cluster count. Higher = more regions.
//! - `spatial_weight` (default 0.5, range 0–1): controls blobbiness vs.
//!   detail fidelity. Higher = blobbier regions (boundaries follow the spatial
//!   grid). Lower = regions follow color/depth boundaries more faithfully.
//!
//! # Normalization
//!
//! All feature components are normalized to ~[0, 1] so no single dimension
//! dominates the Euclidean distance:
//! - L → L/100  (range [0, 1])
//! - a, b → /128 (range [-1, 1])
//! - depth → already [0, 1]
//! - x, y → /max_dim × `spatial_weight` (range [0, spatial_weight])

use crate::processing::SlicConfig;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba};
use palette::{IntoColor, Lab, Srgb};
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// 6D feature vector: (L, a, b, depth, x_norm * s, y_norm * s)
type Feature = [f32; 6];

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

    let k = config.k.clamp(3, 8) as usize;
    let s = config.spatial_weight.clamp(0.0, 1.0);

    // ── Build feature vectors ────────────────────────────────────────────────
    let features = build_features(image, depth_map, width, height, s);

    // ── Initialize cluster centers (deterministic random sampling) ──────────
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut centroids: Vec<Feature> = if features.len() >= k {
        features.choose_multiple(&mut rng, k).copied().collect()
    } else {
        features.clone()
    };
    let actual_k = centroids.len();

    // ── K-means iterations ───────────────────────────────────────────────────
    let mut labels = vec![0u32; n_pixels];

    for _iteration in 0..10 {
        // ── Assignment step: assign each pixel to nearest centroid ───────────
        for (i, feat) in features.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = f32::MAX;
            for (c_idx, centroid) in centroids.iter().enumerate() {
                let d = squared_distance(feat, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c_idx;
                }
            }
            labels[i] = best_idx as u32;
        }

        // ── Update step: recompute centroids as cluster means ────────────────
        let mut sums = vec![[0.0f64; 6]; actual_k];
        let mut counts = vec![0u64; actual_k];

        for (i, feat) in features.iter().enumerate() {
            let c = labels[i] as usize;
            for j in 0..6 {
                sums[c][j] += feat[j] as f64;
            }
            counts[c] += 1;
        }

        let mut changed = false;
        for c in 0..actual_k {
            if counts[c] > 0 {
                let count = counts[c] as f64;
                for j in 0..6 {
                    let new_val = (sums[c][j] / count) as f32;
                    if (new_val - centroids[c][j]).abs() > 1e-6 {
                        changed = true;
                    }
                    centroids[c][j] = new_val;
                }
            }
        }

        // Early exit if converged
        if !changed && _iteration > 0 {
            break;
        }
    }

    Ok(labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build the 6D feature vector for every pixel.
///
/// Normalization:
/// - L → /100, a,b → /128 (Lab components → roughly [-1, 1])
/// - depth → as-is (already [0, 1])
/// - x, y → normalized to [0, 1] by max_dim, then × `spatial_weight`
fn build_features(
    image: &DynamicImage,
    depth_map: Option<&[f32]>,
    width: u32,
    height: u32,
    spatial_weight: f32,
) -> Vec<Feature> {
    let rgba = image.to_rgba8();
    let n_pixels = (width * height) as usize;
    let max_dim = width.max(height) as f32;

    let mut features = Vec::with_capacity(n_pixels);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            let lab = rgba_to_lab(*pixel);

            let depth = depth_map
                .and_then(|d| d.get((y * width + x) as usize).copied())
                .unwrap_or(0.5);

            let x_norm = (x as f32 / max_dim) * spatial_weight;
            let y_norm = (y as f32 / max_dim) * spatial_weight;

            features.push([
                lab.l / 100.0,       // L: [0, 1]
                lab.a / 128.0,       // a: [-1, 1]
                lab.b / 128.0,       // b: [-1, 1]
                depth,               // depth: [0, 1]
                x_norm,              // spatial x: [0, s]
                y_norm,              // spatial y: [0, s]
            ]);
        }
    }

    features
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn squared_distance(a: &Feature, b: &Feature) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..6 {
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

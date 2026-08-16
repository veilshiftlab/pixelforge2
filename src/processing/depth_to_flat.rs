//! Depth-to-flat color conversion
//!
//! # Phase 3 — Depth Signal Architecture
//!
//! Replaces the old `local * (1-gw) + global * gw` blend (which destroyed
//! both signals — neither local detail nor global depth relationships
//! survived) with **hierarchical depth normalization**:
//!
//! 1. **Median filter** (5×5) on the depth map to remove speckle.
//! 2. **Global shading** via percentile normalization (2nd/98th, robust
//!    to outliers). Preserves global truth: far background pixels get
//!    different shading than near foreground pixels, across SLIC regions.
//! 3. **Local shading** via per-region MAD normalization (current behavior).
//! 4. **Hierarchical blend**: `shading = global * gw + local * lw`, then
//!    clamp to [-1, 1]. Default gw=0.6, lw=0.4 — biases toward global truth
//!    per user complaint "missing out on global depth values".
//! 5. **Background classification** is now **region-based** (per-SLIC-cluster):
//!    a cluster is background if `mean_depth > p70 AND (touches_border OR
//!    size_pct > bg_cluster_size_pct)`. Eliminates the depth-bleed leakage
//!    onto subject edges that the old per-pixel `d > threshold` produced.
//!    Falls back to Otsu/manual threshold when SLIC labels are unavailable.
//! 6. **Contrast curve + Lab L bias**: `s' = sign(s) * |s|^gamma`,
//!    `L' = clamp(L + s' * strength * l_shift_scale, 0, 100)`.
//!    Phase 2: `l_shift_scale` (default 40) replaces the implicit ×100
//!    that produced ±60 L* shifts at default strength=0.6.
//! 7. **Phase 2 — Flat-region mask**: per-pixel mask `∈ {0, 1}` built from
//!    per-cluster MAD. When a cluster's MAD < `mad_threshold`, ALL shading
//!    (global + local) is zeroed for its pixels. Previously only the local
//!    signal was gated, leaving flat regions (e.g. 2D anime faces with
//!    ML-imposed spurious depth variance) vulnerable to extreme shifts.
//!
//! # Depth convention
//!
//! Depth-Anything V2 outputs **0 = nearest, 1 = farthest**. For shading,
//! near features should be highlights (high L), so we invert: depth values
//! below the percentile/median baseline produce positive `s` (lighten),
//! above produce negative `s` (darken).

use super::DepthToFlatConfig;
use crate::ml::MLResults;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::{IntoColor, Lab, Srgb};
// Phase 6 — P5: HashMap removed; per-region shading and bg classification
// now use Vec indexed by cluster ID (cheaper, no hashing overhead).

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Convert depth map to flat color bands with per-region shading.
///
/// Pipeline:
/// 1. Median-filter the depth map (5×5) to remove speckle.
/// 2. Compute hierarchical shading: global (percentile 2/98) + local (per-region MAD).
/// 3. Classify background per SLIC cluster (mean depth + border/size rule).
/// 4. Apply contrast curve + Lab L bias to foreground; desaturate background.
pub fn depth_to_flat(
    input: &DynamicImage,
    ml_results: &MLResults,
    config: &DepthToFlatConfig,
) -> Result<DynamicImage> {
    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    let raw_depth = match &ml_results.depth_map {
        Some(d) => d,
        None    => return Ok(input.clone()),
    };

    // ── 1. Median filter the depth map ───────────────────────────────────────
    // Phase 6 — P1: use cached filtered_depth_map if present (computed once
    // in MLAnalysis::analyze). Falls back to local computation when missing
    // (e.g., when the pipeline is called with hand-constructed MLResults from
    // a test binary, or when MLResults was deserialized without the cache).
    let depth: Vec<f32> = if let Some(cached) = &ml_results.filtered_depth_map {
        if cached.len() == raw_depth.len() {
            cached.clone()
        } else {
            // Dim mismatch — recompute (defensive against stale cache)
            median_filter_5x5(raw_depth, width, height)
        }
    } else {
        median_filter_5x5(raw_depth, width, height)
    };

    // ── 2. Hierarchical shading (C6) ─────────────────────────────────────────
    // Global preserves relative depth between all pixels (percentile-normalized
    // so outliers don't collapse the range). Local preserves fine detail
    // within each SLIC region (MAD-normalized). We sum them with separate
    // weights instead of averaging — both signals survive.
    //
    // Phase 2: also compute a per-pixel flat mask from the per-cluster MAD.
    // When a cluster's MAD is below `mad_threshold`, ALL shading (global +
    // local) is zeroed for its pixels. This prevents the global signal from
    // amplifying ML depth noise on flat regions (e.g. 2D anime faces where
    // ML imposes spurious depth variance on what should be flat).
    let (local_shading, per_cluster_mad) = compute_per_region_shading(
        &depth, ml_results, width, height, config,
    );
    let global_shading = compute_global_shading_percentile(&depth);
    let flat_mask = compute_flat_mask(&depth, ml_results, width, height, config, &per_cluster_mad);

    let gw = config.global_depth_weight.clamp(0.0, 1.0);
    let lw = 1.0 - gw; // local weight mirrors global weight for symmetry
    let shading: Vec<f32> = local_shading.iter()
        .zip(global_shading.iter())
        .zip(flat_mask.iter())
        .map(|((&l, &g), &fm)| (g * gw + l * lw).clamp(-1.0, 1.0) * fm)
        .collect();

    // ── 3. Region-based background classification (C8, C13) ──────────────────
    // Per-SLIC-cluster: bg if mean_depth > p70 AND (touches_border OR size > 15%).
    // Falls back to per-pixel Otsu/manual threshold when SLIC is unavailable.
    let bg_mask = classify_background(&depth, ml_results, width, height, config);

    // ── 4. Apply shading + background treatment ───────────────────────────────
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let original = input.get_pixel(x, y);
            let is_background = bg_mask
                .get(idx)
                .copied()
                .unwrap_or(false);

            let adjusted = if is_background {
                apply_background_treatment(original, config)
            } else {
                apply_foreground_shading(original, shading.get(idx).copied().unwrap_or(0.0), config)
            };

            output.put_pixel(x, y, adjusted);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

// ─────────────────────────────────────────────────────────────────────────────
// Global shading — percentile normalization (C6)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute global shading signal using **percentile normalization** (2nd/98th).
///
/// Returns `s ∈ [-1, 1]` where positive = nearer (highlight), negative = farther
/// (shadow). Uses 2nd/98th percentiles as the [0,1] endpoints instead of
/// min/max — robust to outliers (a few saturated edge pixels won't collapse
/// the dynamic range, unlike the old min-max approach).
fn compute_global_shading_percentile(depth: &[f32]) -> Vec<f32> {
    if depth.is_empty() {
        return Vec::new();
    }

    let (p2, p98) = percentiles(depth, 0.02, 0.98);
    let range = p98 - p2;

    if range < 1e-6 {
        return vec![0.0f32; depth.len()];
    }

    // Normalize to [0, 1] via percentile, then shift to [-1, 1] with 0.5 → 0.
    // Invert: low depth (near) → positive (highlight).
    depth.iter()
        .map(|&d| {
            let normalized = ((d - p2) / range).clamp(0.0, 1.0);  // 0 = nearest, 1 = farthest
            (0.5 - normalized) * 2.0  // near → +1, far → -1
        })
        .collect()
}

/// Lookup the `lo_pct`-th and `hi_pct`-th percentiles of a slice.
/// Sorts a copy — callers should cache the result if reused.
///
/// Phase 6: made `pub(crate)` so `edges::sobel_edge_mask` can reuse it for
/// percentile-normalizing the Sobel gradient (puts the fallback on the same
/// scale as ML edge probabilities, so `teed_threshold` means the same thing
/// for both paths).
pub(crate) fn percentiles(values: &[f32], lo_pct: f32, hi_pct: f32) -> (f32, f32) {
    if values.is_empty() { return (0.0, 0.0); }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo_idx = ((n as f32 - 1.0) * lo_pct).round() as usize;
    let hi_idx = ((n as f32 - 1.0) * hi_pct).round() as usize;
    (sorted[lo_idx.min(n - 1)], sorted[hi_idx.min(n - 1)])
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-region shading (MAD normalization) — unchanged
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the per-pixel shading signal `s ∈ [-1, 1]` using SLIC labels.
///
/// For each SLIC cluster:
/// 1. Collect all depth values.
/// 2. Compute median and MAD.
/// 3. Normalize: `s = (depth - median) / (1.4826 * MAD)`, clamped to [-1, 1].
/// 4. If MAD < `mad_threshold`, set `s = 0` for all pixels in this cluster.
///
/// If SLIC labels are not available, falls back to a global median/MAD over
/// the entire depth map.
///
/// Phase 2: returns `(shading, per_cluster_mad)` where `per_cluster_mad` is
/// a `Vec<f32>` indexed by cluster ID. The caller uses this to build a flat
/// mask that gates the *global* shading signal too (previously only the
/// local signal was gated, leaving flat regions vulnerable to ML depth noise
/// amplified by global percentile normalization).
fn compute_per_region_shading(
    depth: &[f32],
    ml_results: &MLResults,
    width: u32,
    height: u32,
    config: &DepthToFlatConfig,
) -> (Vec<f32>, Vec<f32>) {
    let n = (width * height) as usize;
    let mut shading = vec![0.0f32; n];

    if depth.len() < n {
        return (shading, Vec::new());
    }

    match &ml_results.slic_labels {
        Some(labels) if labels.len() >= n => {
            // Phase 6 — P5: use Vec<Vec<f32>> indexed by cluster ID instead of
            // HashMap. The HashMap had per-pixel hashing overhead on the hot
            // assignment loop. Vec indexing is O(1) with no hashing.
            //
            // SLIC IDs may not be contiguous after Phase 3's
            // `split_disconnected_clusters`, so size by max(labels)+1.
            let max_id = labels.iter().copied().max().unwrap_or(0) as usize;
            let n_clusters = max_id + 1;

            let mut clusters: Vec<Vec<f32>> = vec![Vec::new(); n_clusters];
            for i in 0..n {
                let c = labels[i] as usize;
                if c < n_clusters {
                    clusters[c].push(depth[i]);
                }
            }

            let mut stats: Vec<(f32, f32)> = vec![(0.0, 0.0); n_clusters];
            for (c, mut values) in clusters.into_iter().enumerate() {
                if values.is_empty() { continue; }
                let median = compute_median(&mut values);
                let deviations: Vec<f32> = values.iter().map(|&v| (v - median).abs()).collect();
                let mut deviations = deviations;
                let mad = compute_median(&mut deviations);
                stats[c] = (median, mad);
            }

            // Phase 2: extract per-cluster MAD so the caller can build a flat mask.
            let per_cluster_mad: Vec<f32> = stats.iter().map(|&(_, m)| m).collect();

            for i in 0..n {
                let cluster = labels[i] as usize;
                if cluster < n_clusters {
                    let (median, mad) = stats[cluster];
                    if mad < config.mad_threshold {
                        shading[i] = 0.0;
                    } else {
                        // Invert: depth below median (nearer) → positive s (highlight)
                        let s = (median - depth[i]) / (1.4826 * mad);
                        shading[i] = s.clamp(-1.0, 1.0);
                    }
                }
            }

            (shading, per_cluster_mad)
        }
        _ => {
            // ── Global fallback (no SLIC labels) ───────────────────────────────
            let mut all_depths: Vec<f32> = depth.to_vec();
            let median = compute_median(&mut all_depths);
            let deviations: Vec<f32> = depth.iter().map(|&v| (v - median).abs()).collect();
            let mut deviations = deviations;
            let mad = compute_median(&mut deviations);

            // Phase 2: single-element MAD vector signals "whole-image flat"
            // to the flat-mask builder.
            let per_cluster_mad = vec![mad];

            if mad >= config.mad_threshold {
                for i in 0..n {
                    let s = (median - depth[i]) / (1.4826 * mad);
                    shading[i] = s.clamp(-1.0, 1.0);
                }
            }

            (shading, per_cluster_mad)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — Flat-region mask
// ─────────────────────────────────────────────────────────────────────────────

/// Build a per-pixel flat mask `∈ {0.0, 1.0}`.
///
/// `1.0` = region has meaningful depth variance → apply shading.
/// `0.0` = region is flat (MAD < `mad_threshold`) → skip all shading.
///
/// When SLIC labels are available, the mask is per-cluster (all pixels in a
/// flat cluster get 0.0). When SLIC is unavailable, the mask is whole-image
/// (the single global MAD from `compute_per_region_shading` decides).
///
/// This gate is applied to the *final blended* shading (global + local), so
/// flat regions are protected from both signals. Previously only the local
/// signal was gated, leaving flat regions vulnerable to the global percentile
/// normalization amplifying ML depth noise (e.g. 2D anime faces with ML-imposed
/// spurious depth variance on what should be flat cheek/forehead regions).
fn compute_flat_mask(
    _depth: &[f32],
    ml_results: &MLResults,
    width: u32,
    height: u32,
    config: &DepthToFlatConfig,
    per_cluster_mad: &[f32],
) -> Vec<f32> {
    let n = (width * height) as usize;
    let mut mask = vec![1.0f32; n];

    if per_cluster_mad.is_empty() {
        return mask;
    }

    match &ml_results.slic_labels {
        Some(labels) if labels.len() >= n => {
            for i in 0..n {
                let cluster = labels[i] as usize;
                if cluster < per_cluster_mad.len() {
                    if per_cluster_mad[cluster] < config.mad_threshold {
                        mask[i] = 0.0;
                    }
                }
            }
        }
        _ => {
            // No SLIC labels — `per_cluster_mad` is the single global MAD.
            // If global MAD is below threshold, the whole image is flat.
            if per_cluster_mad[0] < config.mad_threshold {
                for m in &mut mask { *m = 0.0; }
            }
        }
    }

    mask
}

// ─────────────────────────────────────────────────────────────────────────────
// Region-based background classification (C8, C13)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a per-pixel boolean mask: `true` = background, `false` = foreground.
///
/// When SLIC labels are available, classification is per-cluster:
/// ```text
/// for each SLIC cluster:
///   mean_depth = mean(depth[cluster])
///   touches_border = any pixel in cluster is on the image edge
///   size_pct = len(cluster) / total_pixels
///   is_background = (mean_depth > p70_global_depth)
///                && (touches_border || size_pct > bg_cluster_size_pct)
/// ```
/// This eliminates the depth-bleed leakage onto subject edges that the old
/// per-pixel `d > threshold` produced.
///
/// When SLIC labels are unavailable, falls back to per-pixel Otsu/manual
/// threshold (the Phase 1 behavior).
fn classify_background(
    depth: &[f32],
    ml_results: &MLResults,
    width: u32,
    height: u32,
    config: &DepthToFlatConfig,
) -> Vec<bool> {
    let n = (width * height) as usize;
    let mut mask = vec![false; n];

    if depth.len() < n {
        return mask;
    }

    let labels = match &ml_results.slic_labels {
        Some(l) if l.len() >= n => l,
        _ => {
            // ── Fallback: per-pixel threshold (Otsu or manual) ─────────────────
            let threshold = compute_bg_threshold(depth, config);
            for i in 0..n {
                mask[i] = depth[i] > threshold;
            }
            return mask;
        }
    };

    // ── Region-based classification (Phase 3 — C8, C13) ───────────────────────
    // Phase 6 — P5: Vec-indexed by cluster ID instead of HashMap.
    // SLIC IDs may not be contiguous after Phase 3's split_disconnected_clusters,
    // so size by max(labels)+1.
    let max_id = labels.iter().copied().max().unwrap_or(0) as usize;
    let n_clusters = max_id + 1;
    let mut cluster_sum:    Vec<f64> = vec![0.0; n_clusters];
    let mut cluster_count:  Vec<u32> = vec![0;    n_clusters];
    let mut cluster_border: Vec<bool> = vec![false; n_clusters];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let lbl = labels[i] as usize;
            if lbl >= n_clusters { continue; }
            cluster_sum[lbl]   += depth[i] as f64;
            cluster_count[lbl] += 1;
            // Border touch: pixel on the outermost row/column of the image
            let on_border = x == 0 || y == 0 || x == width - 1 || y == height - 1;
            if on_border {
                cluster_border[lbl] = true;
            }
        }
    }

    // Compute p70 of the per-cluster mean depth values.
    let mut cluster_means: Vec<f32> = (0..n_clusters)
        .filter(|&c| cluster_count[c] > 0)
        .map(|c| {
            let count = cluster_count[c] as f32;
            (cluster_sum[c] as f32 / count.max(1.0)) as f32
        })
        .collect();
    cluster_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p70 = if cluster_means.is_empty() {
        0.6
    } else {
        let idx = ((cluster_means.len() as f32 - 1.0) * 0.70).round() as usize;
        cluster_means[idx.min(cluster_means.len() - 1)]
    };

    // Decide per-cluster
    let mut cluster_is_bg: Vec<bool> = vec![false; n_clusters];
    let size_threshold = (n as f32 * config.bg_cluster_size_pct.clamp(0.0, 1.0)) as u32;
    for c in 0..n_clusters {
        if cluster_count[c] == 0 { continue; }
        let count = cluster_count[c];
        let mean = cluster_sum[c] as f32 / count.max(1) as f32;
        let touches_border = cluster_border[c];
        cluster_is_bg[c] = mean > p70 && (touches_border || count > size_threshold);
    }

    // Apply to per-pixel mask
    for i in 0..n {
        let lbl = labels[i] as usize;
        if lbl < n_clusters {
            mask[i] = cluster_is_bg[lbl];
        }
    }

    mask
}

// ─────────────────────────────────────────────────────────────────────────────
// Contrast curve + Lab L bias
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the contrast curve and Lab L bias to a foreground pixel.
///
/// - Contrast curve: `s' = sign(s) * |s|^gamma`
/// - Lab L bias: `L' = clamp(L + s' * strength * l_shift_scale, 0, 100)`
///   Phase 2: `l_shift_scale` (default 40) replaces the implicit ×100 that
///   produced ±60 L* shifts at default strength=0.6.
fn apply_foreground_shading(
    original: Rgba<u8>,
    s: f32,
    config: &DepthToFlatConfig,
) -> Rgba<u8> {
    if config.strength < 0.001 || s.abs() < 0.001 {
        return original;
    }

    let lab = rgba_to_lab(original);

    // Contrast curve
    let s_curve = s.signum() * s.abs().powf(config.gamma);

    // Lab L bias
    // Phase 2: uses configurable `l_shift_scale` (default 40) instead of the
    // implicit ×100 that produced ±60 L* shifts at default strength=0.6.
    let new_l = (lab.l + s_curve * config.strength * config.l_shift_scale).clamp(0.0, 100.0);

    lab_to_rgba(Lab::new(new_l, lab.a, lab.b), original[3])
}

// ─────────────────────────────────────────────────────────────────────────────
// Background treatment
// ─────────────────────────────────────────────────────────────────────────────

/// Desaturate and optionally darken background pixels.
fn apply_background_treatment(original: Rgba<u8>, config: &DepthToFlatConfig) -> Rgba<u8> {
    let lab = rgba_to_lab(original);

    let desat = config.bg_desaturation.clamp(0.0, 1.0);
    let new_a = lab.a * (1.0 - desat);
    let new_b = lab.b * (1.0 - desat);

    let new_l = (lab.l + config.bg_lightness_shift * 100.0).clamp(0.0, 100.0);

    lab_to_rgba(Lab::new(new_l, new_a, new_b), original[3])
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth preprocessing — 5×5 median filter
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a 5×5 median filter to the depth map.
///
/// Depth-Anything produces speckle around hair, glasses, and fine detail
/// boundaries. Median filtering preserves edges while removing speckle —
/// Gaussian would blur edges into neighbors, softening the very depth
/// discontinuities we want to detect.
///
/// Phase 6 — P1: made `pub` so `ml::analysis` can call it once after
/// depth inference and cache the result in `MLResults.filtered_depth_map`.
/// `depth_to_flat` then reads the cache instead of recomputing on every
/// pipeline invocation.
pub fn median_filter_5x5(depth: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    if depth.len() < n {
        return depth.to_vec();
    }

    let mut filtered = vec![0.0f32; n];

    // 5×5 window has 25 elements; median is the 13th (index 12) after sorting
    let mut window = [0.0f32; 25];

    for y in 0..h {
        for x in 0..w {
            let mut idx = 0;
            for dy in -2..=2isize {
                for dx in -2..=2isize {
                    let nx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    let ny = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    window[idx] = depth[ny * w + nx];
                    idx += 1;
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            filtered[y * w + x] = window[12];
        }
    }

    filtered
}

// ─────────────────────────────────────────────────────────────────────────────
// Background threshold fallback (Otsu or manual) — used only when SLIC missing
// ─────────────────────────────────────────────────────────────────────────────

fn compute_bg_threshold(depth: &[f32], config: &DepthToFlatConfig) -> f32 {
    if config.use_otsu_threshold {
        let raw = otsu_threshold(depth);
        let clamped = raw.clamp(0.3, 0.85);
        log::debug!("Otsu depth threshold: raw={:.3} clamped={:.3}", raw, clamped);
        clamped
    } else {
        config.bg_depth_threshold.clamp(0.01, 0.99)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the median of a slice (sorts in place).
fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Otsu's threshold
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Otsu threshold for a depth map (256-bin histogram).
fn otsu_threshold(depth_map: &[f32]) -> f32 {
    const BINS: usize = 256;

    let mut hist = [0u32; BINS];
    for &d in depth_map {
        let bin = ((d.clamp(0.0, 1.0)) * (BINS - 1) as f32).round() as usize;
        hist[bin] += 1;
    }

    let total = depth_map.len() as f64;
    if total == 0.0 {
        return 0.6;
    }

    let mut sum_all = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        sum_all += i as f64 * h as f64;
    }

    let mut sum_fg = 0.0f64;
    let mut weight_fg = 0u64;
    let mut best_variance = 0.0f64;
    let mut best_threshold = 0.6f32;

    for t in 0..BINS {
        weight_fg += hist[t] as u64;
        if weight_fg == 0 { continue; }

        let weight_bg = total as u64 - weight_fg;
        if weight_bg == 0 { break; }

        sum_fg += t as f64 * hist[t] as f64;

        let mean_fg = sum_fg / weight_fg as f64;
        let mean_bg = (sum_all - sum_fg) / weight_bg as f64;

        let between = weight_fg as f64 * weight_bg as f64
            * (mean_fg - mean_bg).powi(2);

        if between > best_variance {
            best_variance = between;
            best_threshold = t as f32 / (BINS - 1) as f32;
        }
    }

    best_threshold
}

// ─────────────────────────────────────────────────────────────────────────────
// Lab color utilities
// ─────────────────────────────────────────────────────────────────────────────

fn rgba_to_lab(rgba: Rgba<u8>) -> Lab {
    let rgb: Srgb = Srgb::new(
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    );
    rgb.into_color()
}

fn lab_to_rgba(lab: Lab, alpha: u8) -> Rgba<u8> {
    let rgb: Srgb = lab.into_color();
    Rgba([
        (rgb.red   * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.green * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.blue  * 255.0).clamp(0.0, 255.0) as u8,
        alpha,
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities kept for pipeline/analysis use
// ─────────────────────────────────────────────────────────────────────────────

/// Evenly-spaced depth band thresholds (used by analysis / debugging tools).
pub fn analyze_depth_histogram(_depth_map: &[f32], num_bands: u32) -> Vec<f32> {
    (0..=num_bands)
        .map(|i| i as f32 / num_bands.max(1) as f32)
        .collect()
}

/// Per-pixel depth gradient magnitude, normalized to [0, 1].
/// Used by the importance map for edge-preserving downsampling.
pub fn compute_depth_gradients(depth_map: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let len = depth_map.len();
    let mut gradients = vec![0.0f32; len];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y as usize * w + x as usize;
            let gx = (depth_map[(idx + 1).min(len - 1)]
                    - depth_map[idx.saturating_sub(1)]).abs();
            let gy = (depth_map[(idx + w).min(len - 1)]
                    - depth_map[idx.saturating_sub(w)]).abs();
            gradients[idx] = (gx + gy) * 0.5;
        }
    }

    let max = gradients.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradients { *g /= max; }
    }
    gradients
}

//! Palette generation and quantization

use super::{PaletteConfig, PaletteMode, PresetPalette};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::{FromColor, Lab, Srgb};
use rand::prelude::*;

/// Generated palette
#[derive(Debug, Clone)]
pub struct Palette {
    /// Colors in the palette
    pub colors: Vec<Rgba<u8>>,
}

impl Palette {
    /// Create a new palette
    pub fn new(colors: Vec<Rgba<u8>>) -> Self {
        Self { colors }
    }

    /// Find the nearest color in the palette
    ///
    /// Phase 5 — numerical robustness: uses `total_cmp` instead of
    /// `partial_cmp(...).unwrap()` so NaN distances don't panic. NaN
    /// compares as greater than any finite value, so NaN-distance palette
    /// entries are deprioritized (kept as last resort).
    pub fn nearest(&self, color: Rgba<u8>) -> Rgba<u8> {
        if self.colors.is_empty() {
            return color;
        }

        let target_lab = rgb_to_lab(color);

        self.colors
            .iter()
            .map(|&c| (c, color_distance_lab(target_lab, rgb_to_lab(c))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(c, _)| c)
            .unwrap_or(color)
    }

    /// Find the index of the nearest palette color. Used by palette-mode
    /// downsampling to pre-quantize every pixel before counting frequencies.
    ///
    /// Phase 5 — numerical robustness: uses `total_cmp` instead of
    /// `partial_cmp(...).unwrap()` so NaN distances don't panic.
    pub fn nearest_index(&self, color: Rgba<u8>) -> usize {
        if self.colors.is_empty() {
            return 0;
        }

        let target_lab = rgb_to_lab(color);

        self.colors
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, color_distance_lab(target_lab, rgb_to_lab(c))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — C2: PaletteLab cache + local-contrast color picker
// ─────────────────────────────────────────────────────────────────────────────

/// Cache of palette colors pre-converted to Lab. Built once per `draw_edges`
/// call so we don't redo the (relatively expensive) RGB→Lab conversion for
/// every edge pixel.
///
/// Use [`PaletteLab::furthest_from`] to pick the palette color that maximizes
/// perceptual distance (Lab ΔE) from a local neighborhood mean — this is the
/// local-contrast outline color.
pub struct PaletteLab {
    pub rgb: Vec<Rgba<u8>>,
    pub lab: Vec<Lab>,
}

impl PaletteLab {
    /// Build the cache from a `Palette`. Empty palettes produce an empty cache.
    pub fn from_palette(p: &Palette) -> Self {
        let rgb = p.colors.clone();
        let lab = p.colors.iter().map(|&c| rgb_to_lab(c)).collect();
        Self { rgb, lab }
    }

    /// Return the palette color furthest (in Lab ΔE) from `target`.
    ///
    /// Used by `edges::local_contrast_color` to pick an outline color that
    /// maximizes contrast against the local 3×3 neighborhood mean. If the
    /// palette is empty, returns the input color unchanged.
    ///
    /// Ties broken by darker-first (gives outlines a slight dark bias which
    /// reads better on light backgrounds).
    pub fn furthest_from(&self, target: Lab) -> Rgba<u8> {
        if self.rgb.is_empty() {
            return Rgba([0, 0, 0, 255]);
        }

        let mut best_idx = 0usize;
        let mut best_dist = f32::MIN;
        for (i, &lab) in self.lab.iter().enumerate() {
            let d = color_distance_lab(target, lab);
            if d > best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        self.rgb[best_idx]
    }

    /// Return the palette color nearest (in Lab ΔE) to `target`.
    /// Mirrors `Palette::nearest` but skips the per-call Lab conversion.
    pub fn nearest_to(&self, target: Lab) -> Rgba<u8> {
        if self.rgb.is_empty() {
            return Rgba([0, 0, 0, 255]);
        }
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        for (i, &lab) in self.lab.iter().enumerate() {
            let d = color_distance_lab(target, lab);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        self.rgb[best_idx]
    }
}

/// Generate palette from image.
///
/// Phase 9: removed the unused `_ml_results` parameter (was a leftover from
/// the dropped face-parsing pipeline; never read by any code path).
pub fn generate_palette(
    input: &DynamicImage,
    config: &PaletteConfig,
) -> Result<Palette> {
    match config.mode {
        PaletteMode::Auto => generate_auto_palette(input, config),
        PaletteMode::Preset => generate_preset_palette(config.preset),
        PaletteMode::Custom => Ok(Palette::new(config.custom_colors.iter()
            .map(|&c| Rgba([c.r(), c.g(), c.b(), 255]))
            .collect())),
        PaletteMode::Hybrid => generate_auto_palette(input, config),
    }
}

/// Auto-generate palette using k-means clustering.
///
/// # Chroma-weighted subsampling (desaturation fix)
///
/// The palette is extracted from the image via k-means in Lab space. On
/// real images, vibrant colors (blue dress, warm skin) are often a small
/// minority of total pixels — the image is dominated by gray/dark regions
/// (background, hair, shadows). Uniform random subsampling under-represents
/// vibrant colors, causing k-means to allocate most centroids to gray
/// regions and completely miss vibrant colors.
///
/// Fix: subsample with probability proportional to **chroma**
/// (C = sqrt(a² + b²)). High-chroma pixels are oversampled, so k-means
/// sees enough vibrant pixels to allocate dedicated centroids. This
/// ensures the palette always contains entries for vibrant colors
/// regardless of their pixel count in the original image.
fn generate_auto_palette(
    input: &DynamicImage,
    config: &PaletteConfig,
) -> Result<Palette> {
    let (width, height) = input.dimensions();

    // Extract all colors as Lab
    let mut colors: Vec<Lab> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let pixel = input.get_pixel(x, y);
            colors.push(rgb_to_lab(pixel));
        }
    }

    // Chroma-weighted subsampling to 16K pixels.
    //
    // Each pixel gets a weight of (1 + chroma). High-chroma pixels are
    // oversampled so k-means allocates centroids to vibrant regions.
    //
    // Why (1 + chroma) and not just chroma?
    // - Pixels with chroma=0 (pure gray) still need representation (gray
    //   is a valid and important color in most images).
    // - (1 + chroma) ensures grays have a base weight of 1, while vibrant
    //   pixels (chroma 30-80) get weight 31-81 — ~40-80× more likely to
    //   be sampled than gray.
    //
    // Algorithm: weighted sampling without replacement via prefix-sum binary
    // search. Builds a cumulative-weight array once (O(N)), then each sample
    // is O(log N) via binary search. Total: O(N + S·log N) where S=16K and
    // N=~1M → ~400K operations (was ~16 billion with the old linear scan).
    // Duplicates handled by over-sampling 10% and deduplicating — collision
    // rate is ~S/N ≈ 1.6%, so one extra pass is enough.
    const MAX_SAMPLES: usize = 16_384;
    let colors: Vec<Lab> = if colors.len() > MAX_SAMPLES {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42); // deterministic

        // Compute weights: w[i] = 1 + chroma(colors[i])
        let weights: Vec<f32> = colors.iter()
            .map(|&c| 1.0 + (c.a * c.a + c.b * c.b).sqrt())
            .collect();

        // Build prefix-sum array for binary search.
        // cumsum[i] = sum of weights[0..=i].
        let mut cumsum: Vec<f64> = Vec::with_capacity(weights.len());
        let mut acc = 0.0f64;
        for &w in &weights {
            acc += w as f64;
            cumsum.push(acc);
        }
        let total_weight = acc;

        // Weighted sampling with over-sampling + dedup.
        // Generate ~10% extra samples to compensate for duplicate collisions,
        // then dedup and take the first MAX_SAMPLES unique indices.
        let target_unique = MAX_SAMPLES;
        let mut sampled_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(target_unique);

        // Keep sampling in batches until we have enough unique indices.
        while sampled_indices.len() < target_unique {
            let needed = target_unique - sampled_indices.len();
            let batch_size = (needed * 110 / 100).max(64); // 10% extra, min 64
            for _ in 0..batch_size {
                let target = rng.gen::<f64>() * total_weight;
                // Binary search for the first index where cumsum[i] >= target.
                let idx = match cumsum.binary_search_by(|&c| {
                    c.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    Ok(i) => i,
                    Err(i) => i.min(weights.len() - 1),
                };
                sampled_indices.insert(idx);
            }
            // Safety: if we somehow can't make progress (extremely degenerate
            // weight distribution), break to avoid infinite loop.
            if sampled_indices.len() >= colors.len() {
                break;
            }
        }

        sampled_indices.into_iter()
            .take(target_unique)
            .map(|i| colors[i])
            .collect()
    } else {
        colors
    };

    // Run k-means clustering
    let palette_colors = k_means(&colors, config.max_colors as usize);

    Ok(Palette::new(palette_colors))
}

/// Generate preset palette
fn generate_preset_palette(preset: PresetPalette) -> Result<Palette> {
    let colors = match preset {
        PresetPalette::None => return Ok(Palette::new(vec![])),
        PresetPalette::GameBoy => vec![
            Rgba([15, 56, 15, 255]),
            Rgba([48, 98, 48, 255]),
            Rgba([139, 172, 15, 255]),
            Rgba([155, 188, 15, 255]),
        ],
        PresetPalette::GameBoyColor => vec![
            Rgba([15, 56, 15, 255]),
            Rgba([48, 98, 48, 255]),
            Rgba([139, 172, 15, 255]),
            Rgba([155, 188, 15, 255]),
            Rgba([139, 80, 80, 255]),
            Rgba([80, 100, 140, 255]),
            Rgba([200, 150, 100, 255]),
            Rgba([255, 200, 150, 255]),
        ],
        PresetPalette::NES => nes_palette(),
        PresetPalette::PICO8 => vec![
            Rgba([0, 0, 0, 255]),
            Rgba([29, 43, 83, 255]),
            Rgba([126, 37, 83, 255]),
            Rgba([0, 135, 81, 255]),
            Rgba([171, 82, 54, 255]),
            Rgba([95, 87, 79, 255]),
            Rgba([194, 195, 199, 255]),
            Rgba([255, 241, 232, 255]),
            Rgba([255, 0, 77, 255]),
            Rgba([255, 163, 0, 255]),
            Rgba([255, 236, 39, 255]),
            Rgba([0, 228, 54, 255]),
            Rgba([41, 173, 255, 255]),
            Rgba([131, 118, 156, 255]),
            Rgba([255, 119, 168, 255]),
            Rgba([255, 204, 170, 255]),
        ],
        PresetPalette::DawnBringer32 => dawnbringer_palette(),
        PresetPalette::AAP64 => aap64_palette(),
    };

    Ok(Palette::new(colors))
}

/// Apply palette quantization to image
///
/// Phase 6 — P4: builds a `PaletteLab` cache once (instead of recomputing
/// Lab for every palette entry on every pixel), reducing per-pixel work
/// from O(K * Lab_conversion) to O(K * distance) — the Lab conversions
/// are amortized to once per palette entry instead of once per pixel.
pub fn apply_palette(
    input: &DynamicImage,
    palette: &Palette,
) -> Result<DynamicImage> {
    if palette.colors.is_empty() {
        return Ok(input.clone());
    }

    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    // Phase 6 — P4: build PaletteLab cache once.
    let palette_lab = PaletteLab::from_palette(palette);

    for y in 0..height {
        for x in 0..width {
            let pixel = input.get_pixel(x, y);  // returns Rgba<u8> by value
            let target = rgb_to_lab(pixel);
            let quantized = palette_lab.nearest_to(target);
            output.put_pixel(x, y, quantized);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// K-means clustering for palette extraction
///
/// Uses a fixed seed (42) for deterministic output — the same input image
/// and config always produces the same palette. Without this, `thread_rng()`
/// gave different initial centroids on every call, causing the pipeline to
/// produce different results on reprocess.
///
/// # Saturation preservation (desaturation fix)
///
/// Two changes prevent the "washed-out colors" bug where vibrant subject
/// colors (blue dress, warm skin) collapsed to grayish centroids:
///
/// 1. **K-means++ initialization** (replaces uniform random sampling):
///    spreads initial centroids across the color space so vibrant outliers
///    get their own centroid from the start. Uniform random could place
///    all centroids in the dominant gray/background region, causing
///    vibrant colors to be absorbed into grayish clusters.
///
/// 2. **Medoid snap after convergence**: replaces each centroid with the
///    actual pixel closest to it (the "medoid"). Mean-based centroids are
///    synthetic averages that reduce chroma (opposite hue directions
///    cancel). The medoid is a real color that exists in the image, so
///    vibrant regions stay vibrant — the medoid of a blue-dress cluster
///    is an actual blue pixel, not a grayish average.
fn k_means(colors: &[Lab], k: usize) -> Vec<Rgba<u8>> {
    if colors.is_empty() || k == 0 {
        return vec![];
    }

    let k = k.min(colors.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // ── K-means++ initialization ────────────────────────────────────────────
    // Pick the first centroid randomly, then each subsequent centroid with
    // probability proportional to D(x)² (squared distance from nearest
    // existing centroid). This spreads centroids across the color space.
    let mut centroids: Vec<Lab> = Vec::with_capacity(k);

    // First centroid: uniform random.
    let first_idx = rng.gen_range(0..colors.len());
    centroids.push(colors[first_idx]);

    // Subsequent centroids: weighted by D(x)².
    // `min_dists[i]` = squared distance from colors[i] to nearest centroid.
    let mut min_dists: Vec<f32> = colors
        .iter()
        .map(|&c| squared_distance_lab(c, centroids[0]))
        .collect();

    for _ in 1..k {
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        if total <= 0.0 {
            // All remaining pixels are duplicates of existing centroids —
            // just pick randomly to fill the remaining slots.
            let idx = rng.gen_range(0..colors.len());
            centroids.push(colors[idx]);
        } else {
            // Weighted random selection: pick index with probability
            // proportional to min_dists[i].
            let target = rng.gen::<f64>() * total;
            let mut acc = 0.0f64;
            let mut chosen = colors.len() - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                acc += d as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            centroids.push(colors[chosen]);
        }

        // Update min_dists with the new centroid.
        let new_centroid = centroids[centroids.len() - 1];
        for (i, &color) in colors.iter().enumerate() {
            let d = squared_distance_lab(color, new_centroid);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    // ── Lloyd's iterations (assign + update) ────────────────────────────────
    for _ in 0..20 {
        // Assign colors to nearest centroid
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];

        for (i, &color) in colors.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(j, &c)| (j, squared_distance_lab(color, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);

            clusters[nearest].push(i);
        }

        // Update centroids as cluster means
        let mut changed = false;
        for (j, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() { continue; }
            let sum_l: f32 = cluster.iter().map(|&i| colors[i].l).sum();
            let sum_a: f32 = cluster.iter().map(|&i| colors[i].a).sum();
            let sum_b: f32 = cluster.iter().map(|&i| colors[i].b).sum();
            let count = cluster.len() as f32;
            let new_centroid = Lab::new(sum_l / count, sum_a / count, sum_b / count);
            if squared_distance_lab(new_centroid, centroids[j]) > 1e-6 {
                changed = true;
            }
            centroids[j] = new_centroid;
        }

        if !changed { break; }
    }

    // ── Medoid snap: replace each centroid with the actual pixel closest to it ──
    // Mean-based centroids reduce chroma (opposite hues cancel). The medoid
    // is a real pixel color, so vibrant regions stay vibrant.
    let mut final_palette: Vec<Lab> = Vec::with_capacity(k);

    // Final assignment to get clusters.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &color) in colors.iter().enumerate() {
        let nearest = centroids
            .iter()
            .enumerate()
            .map(|(j, &c)| (j, squared_distance_lab(color, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0);
        clusters[nearest].push(i);
    }

    for (j, cluster) in clusters.iter().enumerate() {
        if cluster.is_empty() {
            // Empty cluster — keep the centroid as-is (will be converted to RGB).
            final_palette.push(centroids[j]);
            continue;
        }
        // Find the pixel in this cluster closest to the centroid (medoid).
        let centroid = centroids[j];
        let mut best_dist = f32::MAX;
        let mut best_idx = cluster[0];
        for &i in cluster {
            let d = squared_distance_lab(colors[i], centroid);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        final_palette.push(colors[best_idx]);
    }

    // ── Debug log: dump palette for verification ────────────────────────────
    if log::log_enabled!(log::Level::Debug) {
        let palette_str: Vec<String> = final_palette.iter()
            .map(|&lab| {
                let rgb = lab_to_rgb(lab);
                format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2])
            })
            .collect();
        log::debug!("k-means palette ({} colors): {}", final_palette.len(), palette_str.join(" "));
    }

    // Convert to RGB
    final_palette
        .iter()
        .map(|&lab| lab_to_rgb(lab))
        .collect()
}

/// Squared Euclidean distance in Lab space (no sqrt — cheaper, same ordering).
#[inline]
fn squared_distance_lab(a: Lab, b: Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    dl * dl + da * da + db * db
}

/// Convert RGB to Lab color space
pub(crate) fn rgb_to_lab(rgba: Rgba<u8>) -> Lab {
    let rgb = Srgb::new(
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    );
    Lab::from_color(rgb)
}

/// Convert Lab to RGB
pub(crate) fn lab_to_rgb(lab: Lab) -> Rgba<u8> {
    let rgb: Srgb = Srgb::from_color(lab);
    Rgba([
        (rgb.red * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.green * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.blue * 255.0).clamp(0.0, 255.0) as u8,
        255,
    ])
}

/// Calculate color distance in Lab space
pub(crate) fn color_distance_lab(a: Lab, b: Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

// ============================================================================
// PRESET PALETTE DEFINITIONS
// ============================================================================

fn nes_palette() -> Vec<Rgba<u8>> {
    vec![
        Rgba([0, 0, 0, 255]),
        Rgba([252, 252, 252, 255]),
        Rgba([188, 188, 188, 255]),
        Rgba([124, 124, 124, 255]),
        Rgba([164, 228, 252, 255]),
        Rgba([60, 188, 252, 255]),
        Rgba([0, 120, 248, 255]),
        Rgba([0, 0, 252, 255]),
        Rgba([184, 184, 248, 255]),
        Rgba([104, 136, 252, 255]),
        Rgba([0, 88, 248, 255]),
        Rgba([0, 0, 248, 255]),
        Rgba([216, 184, 248, 255]),
        Rgba([152, 120, 248, 255]),
        Rgba([104, 68, 252, 255]),
        Rgba([68, 0, 252, 255]),
        Rgba([248, 184, 248, 255]),
        Rgba([248, 120, 248, 255]),
        Rgba([216, 0, 204, 255]),
        Rgba([148, 0, 132, 255]),
        Rgba([248, 164, 192, 255]),
        Rgba([248, 88, 152, 255]),
        Rgba([228, 0, 88, 255]),
        Rgba([168, 0, 32, 255]),
        Rgba([240, 208, 176, 255]),
        Rgba([248, 120, 88, 255]),
        Rgba([248, 56, 0, 255]),
        Rgba([136, 20, 0, 255]),
        Rgba([252, 224, 168, 255]),
        Rgba([252, 160, 68, 255]),
        Rgba([228, 92, 16, 255]),
        Rgba([172, 124, 0, 255]),
        Rgba([248, 216, 120, 255]),
        Rgba([248, 184, 0, 255]),
        Rgba([184, 144, 0, 255]),
        Rgba([120, 120, 0, 255]),
        Rgba([216, 248, 120, 255]),
        Rgba([184, 248, 24, 255]),
        Rgba([0, 184, 0, 255]),
        Rgba([0, 128, 0, 255]),
        Rgba([184, 248, 184, 255]),
        Rgba([88, 216, 84, 255]),
        Rgba([0, 168, 0, 255]),
        Rgba([0, 136, 0, 255]),
        Rgba([184, 248, 216, 255]),
        Rgba([88, 248, 152, 255]),
        Rgba([0, 184, 104, 255]),
        Rgba([0, 136, 88, 255]),
        Rgba([0, 252, 252, 255]),
        Rgba([0, 232, 216, 255]),
        Rgba([0, 136, 136, 255]),
        Rgba([0, 104, 104, 255]),
        Rgba([248, 216, 248, 255]),
        Rgba([248, 184, 248, 255]),
    ]
}

fn dawnbringer_palette() -> Vec<Rgba<u8>> {
    vec![
        Rgba([0, 0, 0, 255]),
        Rgba([34, 32, 52, 255]),
        Rgba([69, 40, 60, 255]),
        Rgba([102, 57, 49, 255]),
        Rgba([143, 86, 59, 255]),
        Rgba([185, 122, 87, 255]),
        Rgba([215, 171, 105, 255]),
        Rgba([238, 219, 139, 255]),
        Rgba([250, 242, 188, 255]),
        Rgba([138, 111, 48, 255]),
        Rgba([87, 71, 47, 255]),
        Rgba([49, 38, 29, 255]),
        Rgba([49, 26, 47, 255]),
        Rgba([62, 39, 35, 255]),
        Rgba([75, 53, 34, 255]),
        Rgba([93, 69, 55, 255]),
        Rgba([107, 84, 55, 255]),
        Rgba([122, 96, 59, 255]),
        Rgba([130, 109, 64, 255]),
        Rgba([153, 133, 85, 255]),
        Rgba([189, 166, 115, 255]),
        Rgba([217, 204, 164, 255]),
        Rgba([251, 246, 212, 255]),
        Rgba([91, 120, 143, 255]),
        Rgba([63, 76, 96, 255]),
        Rgba([48, 52, 70, 255]),
        Rgba([24, 20, 37, 255]),
        Rgba([60, 44, 52, 255]),
        Rgba([119, 54, 66, 255]),
        Rgba([177, 72, 83, 255]),
        Rgba([222, 114, 102, 255]),
        Rgba([247, 164, 139, 255]),
    ]
}

fn aap64_palette() -> Vec<Rgba<u8>> {
    vec![
        Rgba([0, 0, 0, 255]),
        Rgba([25, 22, 20, 255]),
        Rgba([40, 35, 33, 255]),
        Rgba([56, 50, 47, 255]),
        Rgba([75, 66, 62, 255]),
        Rgba([89, 82, 80, 255]),
        Rgba([115, 107, 104, 255]),
        Rgba([148, 140, 135, 255]),
        Rgba([181, 172, 165, 255]),
        Rgba([201, 192, 186, 255]),
        Rgba([223, 215, 208, 255]),
        Rgba([245, 238, 231, 255]),
        Rgba([63, 40, 32, 255]),
        Rgba([89, 57, 45, 255]),
        Rgba([117, 77, 60, 255]),
        Rgba([149, 100, 76, 255]),
        Rgba([179, 124, 97, 255]),
        Rgba([209, 152, 123, 255]),
        Rgba([232, 184, 154, 255]),
        Rgba([251, 219, 190, 255]),
        Rgba([51, 70, 30, 255]),
        Rgba([72, 97, 44, 255]),
        Rgba([94, 122, 58, 255]),
        Rgba([122, 148, 75, 255]),
        Rgba([149, 173, 98, 255]),
        Rgba([178, 198, 126, 255]),
        Rgba([207, 222, 159, 255]),
        Rgba([236, 245, 203, 255]),
        Rgba([31, 60, 50, 255]),
        Rgba([43, 85, 68, 255]),
        Rgba([57, 110, 86, 255]),
        Rgba([76, 137, 103, 255]),
        Rgba([98, 163, 122, 255]),
        Rgba([126, 189, 146, 255]),
        Rgba([160, 214, 176, 255]),
        Rgba([199, 236, 210, 255]),
        Rgba([21, 51, 67, 255]),
        Rgba([34, 77, 96, 255]),
        Rgba([52, 103, 123, 255]),
        Rgba([77, 131, 151, 255]),
        Rgba([108, 158, 178, 255]),
        Rgba([144, 187, 204, 255]),
        Rgba([183, 214, 228, 255]),
        Rgba([222, 239, 249, 255]),
        Rgba([41, 31, 58, 255]),
        Rgba([58, 44, 84, 255]),
        Rgba([79, 62, 112, 255]),
        Rgba([104, 86, 142, 255]),
        Rgba([134, 115, 171, 255]),
        Rgba([168, 148, 199, 255]),
        Rgba([204, 185, 224, 255]),
        Rgba([238, 224, 246, 255]),
        Rgba([68, 24, 44, 255]),
        Rgba([101, 39, 60, 255]),
        Rgba([139, 60, 82, 255]),
        Rgba([180, 89, 109, 255]),
        Rgba([214, 126, 141, 255]),
        Rgba([240, 173, 181, 255]),
        Rgba([252, 219, 221, 255]),
        Rgba([255, 245, 245, 255]),
        Rgba([90, 50, 10, 255]),
        Rgba([138, 82, 23, 255]),
        Rgba([188, 119, 39, 255]),
    ]
}

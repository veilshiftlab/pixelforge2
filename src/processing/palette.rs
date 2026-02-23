//! Palette generation and quantization

use super::{PaletteConfig, PaletteMode, PresetPalette};
use crate::ml::{MLResults, SegmentationRegion};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use palette::{FromColor, Lab, Srgb};
use rand::prelude::*;
use std::collections::HashMap;

/// Generated palette
#[derive(Debug, Clone)]
pub struct Palette {
    /// Colors in the palette
    pub colors: Vec<Rgba<u8>>,

    /// Per-region colors (if available)
    pub regions: HashMap<SegmentationRegion, Vec<Rgba<u8>>>,
}

impl Palette {
    /// Create a new palette
    pub fn new(colors: Vec<Rgba<u8>>) -> Self {
        Self {
            colors,
            regions: HashMap::new(),
        }
    }

    /// Find the nearest color in the palette
    pub fn nearest(&self, color: Rgba<u8>) -> Rgba<u8> {
        if self.colors.is_empty() {
            return color;
        }
        
        let target_lab = rgb_to_lab(color);

        self.colors
            .iter()
            .map(|&c| (c, color_distance_lab(target_lab, rgb_to_lab(c))))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(c, _)| c)
            .unwrap_or(color)
    }

    /// Find nearest color within a region
    pub fn nearest_in_region(&self, color: Rgba<u8>, region: SegmentationRegion) -> Rgba<u8> {
        if let Some(region_colors) = self.regions.get(&region) {
            if region_colors.is_empty() {
                return self.nearest(color);
            }
            
            let target_lab = rgb_to_lab(color);

            region_colors
                .iter()
                .map(|&c| (c, color_distance_lab(target_lab, rgb_to_lab(c))))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(c, _)| c)
                .unwrap_or(color)
        } else {
            self.nearest(color)
        }
    }
}

/// Generate palette from image
pub fn generate_palette(
    input: &DynamicImage,
    config: &PaletteConfig,
    ml_results: Option<&MLResults>,
) -> Result<Palette> {
    match config.mode {
        PaletteMode::Auto => generate_auto_palette(input, config, ml_results),
        PaletteMode::Preset => generate_preset_palette(config.preset),
        PaletteMode::Custom => Ok(Palette::new(config.custom_colors.iter()
            .map(|&c| Rgba([c.r(), c.g(), c.b(), 255]))
            .collect())),
        PaletteMode::Hybrid => generate_hybrid_palette(input, config, ml_results),
    }
}

/// Auto-generate palette using k-means clustering
fn generate_auto_palette(
    input: &DynamicImage,
    config: &PaletteConfig,
    ml_results: Option<&MLResults>,
) -> Result<Palette> {
    let (width, height) = input.dimensions();

    // Extract all colors
    let mut colors: Vec<Lab> = Vec::new();
    let mut region_colors: HashMap<SegmentationRegion, Vec<Lab>> = HashMap::new();

    for y in 0..height {
        for x in 0..width {
            let pixel = input.get_pixel(x, y);
            let lab = rgb_to_lab(pixel);
            colors.push(lab);

            // Track per-region colors if segmentation available
            if config.per_region_limit {
                if let Some(ml) = ml_results {
                    if let Some(seg) = &ml.segmentation {
                        let idx = (y * width + x) as u32;
                        if let Some(&region) = seg.regions.get(&idx) {
                            region_colors.entry(region).or_default().push(lab);
                        }
                    }
                }
            }
        }
    }

    // Run k-means clustering
    let palette_colors = k_means(&colors, config.max_colors as usize);

    let mut palette = Palette::new(palette_colors);

    // Generate per-region palettes
    if config.per_region_limit {
        let region_max = (config.max_colors / 4).max(2);

        for (region, colors) in region_colors {
            if colors.len() > region_max as usize {
                let region_palette = k_means(&colors, region_max as usize);
                palette.regions.insert(region, region_palette);
            }
        }
    }

    Ok(palette)
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

/// Generate hybrid palette (auto + overrides)
fn generate_hybrid_palette(
    input: &DynamicImage,
    config: &PaletteConfig,
    ml_results: Option<&MLResults>,
) -> Result<Palette> {
    let mut palette = generate_auto_palette(input, config, ml_results)?;

    // Apply overrides
    if let Some(skin) = &config.skin_override {
        palette.regions.insert(
            SegmentationRegion::Skin,
            skin.iter().map(|&c| Rgba([c.r(), c.g(), c.b(), 255])).collect(),
        );
    }

    if let Some(hair) = &config.hair_override {
        palette.regions.insert(
            SegmentationRegion::Hair,
            hair.iter().map(|&c| Rgba([c.r(), c.g(), c.b(), 255])).collect(),
        );
    }

    if let Some(bg) = &config.background_override {
        palette.regions.insert(
            SegmentationRegion::Background,
            bg.iter().map(|&c| Rgba([c.r(), c.g(), c.b(), 255])).collect(),
        );
    }

    Ok(palette)
}

/// Apply palette quantization to image
pub fn apply_palette(
    input: &DynamicImage,
    palette: &Palette,
) -> Result<DynamicImage> {
    if palette.colors.is_empty() {
        return Ok(input.clone());
    }

    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = input.get_pixel(x, y);
            let quantized = palette.nearest(pixel);
            output.put_pixel(x, y, quantized);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Apply palette with region awareness
pub fn apply_palette_with_regions(
    input: &DynamicImage,
    palette: &Palette,
    ml_results: Option<&MLResults>,
) -> Result<DynamicImage> {
    if palette.colors.is_empty() {
        return Ok(input.clone());
    }

    let (width, height) = input.dimensions();
    let mut output = RgbaImage::new(width, height);

    let segmentation = ml_results.and_then(|ml| ml.segmentation.as_ref());

    for y in 0..height {
        for x in 0..width {
            let pixel = input.get_pixel(x, y);

            // Find region if available
            let region = segmentation.and_then(|seg| {
                let idx = (y * width + x) as u32;
                seg.regions.get(&idx).copied()
            });

            // Quantize to nearest palette color
            let quantized = if let Some(r) = region {
                palette.nearest_in_region(pixel, r)
            } else {
                palette.nearest(pixel)
            };

            output.put_pixel(x, y, quantized);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// K-means clustering for palette extraction
fn k_means(colors: &[Lab], k: usize) -> Vec<Rgba<u8>> {
    if colors.is_empty() || k == 0 {
        return vec![];
    }

    let k = k.min(colors.len());
    let mut rng = rand::thread_rng();

    // Initialize centroids randomly using slice sampling
    let mut centroids: Vec<Lab> = colors
        .choose_multiple(&mut rng, k)
        .copied()
        .collect();

    // Iterate
    for _ in 0..20 {
        // Assign colors to nearest centroid
        let mut clusters: Vec<Vec<Lab>> = vec![Vec::new(); k];

        for &color in colors {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, color_distance_lab(color, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            clusters[nearest].push(color);
        }

        // Update centroids
        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let sum_l: f32 = cluster.iter().map(|c| c.l).sum();
                let sum_a: f32 = cluster.iter().map(|c| c.a).sum();
                let sum_b: f32 = cluster.iter().map(|c| c.b).sum();
                let count = cluster.len() as f32;

                centroids[i] = Lab::new(sum_l / count, sum_a / count, sum_b / count);
            }
        }
    }

    // Convert centroids to RGB
    centroids
        .iter()
        .map(|&lab| lab_to_rgb(lab))
        .collect()
}

/// Convert RGB to Lab color space
fn rgb_to_lab(rgba: Rgba<u8>) -> Lab {
    let rgb = Srgb::new(
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    );
    Lab::from_color(rgb)
}

/// Convert Lab to RGB
fn lab_to_rgb(lab: Lab) -> Rgba<u8> {
    let rgb: Srgb = Srgb::from_color(lab);
    Rgba([
        (rgb.red * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.green * 255.0).clamp(0.0, 255.0) as u8,
        (rgb.blue * 255.0).clamp(0.0, 255.0) as u8,
        255,
    ])
}

/// Calculate color distance in Lab space
fn color_distance_lab(a: Lab, b: Lab) -> f32 {
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

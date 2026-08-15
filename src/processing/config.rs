//! Processing configuration structures

use serde::{Deserialize, Serialize};

// =============================================================================
// Downsampling Method
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DownsamplingMethod {
    /// Quantize the full-res image to the palette first, then pick the most
    /// common palette color in each downsample block. Eliminates the "smudge"
    /// effect — every output pixel is a discrete palette color, no averaging.
    /// Phase 4 — D1a: now with bilateral pre-filter for smoother palette snap.
    #[default]
    PaletteMode,
    /// Phase 4 — D2: area-average downsample + Floyd-Steinberg error diffusion.
    /// Produces clean pixel-art downscaling with organic dithering. Best for
    /// small palettes (4-16 colors) where you want smooth gradient transitions.
    PerceptualDither,
    Weighted,
    NearestNeighbor,
    Bilinear,
}

// =============================================================================
// Transform Configuration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub output_size: u32,
    pub scale: f32,
    pub rotation: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub export_scale: u32,
    pub downsampling_method: DownsamplingMethod,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            output_size: 32,
            scale: 1.0,
            rotation: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            flip_horizontal: false,
            flip_vertical: false,
            export_scale: 1,
            // Phase 4 — D3: align code with README. Was NearestNeighbor
            // which produced aliasing artifacts; PaletteMode (the new
            // default with bilateral pre-filter) gives crisp output.
            downsampling_method: DownsamplingMethod::PaletteMode,
        }
    }
}

// =============================================================================
// SLIC Superpixel Configuration
// =============================================================================

/// Configuration for SLIC superpixel clustering.
///
/// SLIC replaces BiSeNet segmentation: it's model-free, domain-agnostic, and
/// works equally on photo and anime. The 6D feature vector is
/// `(L, a, b, depth, x·spatial_weight, y·spatial_weight)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicConfig {
    /// Number of clusters (superpixels). Default 32, range 5–128.
    /// Higher = more, smaller regions = finer detail control.
    /// For portraits: 32-64 preserves limbs/face/dress separation.
    /// For simple images: 8-16 is sufficient.
    #[serde(default = "default_slic_k")]
    pub k: u32,

    /// Spatial weight (compactness). Default 0.5, range 0–1.
    /// Higher = blobbier regions (boundaries follow the spatial grid).
    /// Lower = regions follow color/depth boundaries more faithfully.
    #[serde(default = "default_slic_spatial_weight")]
    pub spatial_weight: f32,
}

fn default_slic_k() -> u32 { 32 }
fn default_slic_spatial_weight() -> f32 { 0.7 }

impl Default for SlicConfig {
    fn default() -> Self {
        Self {
            k: default_slic_k(),
            spatial_weight: default_slic_spatial_weight(),
        }
    }
}

// =============================================================================
// Depth-to-Flat Configuration
// =============================================================================

/// Depth-to-flat color conversion configuration.
///
/// After the pipeline repurpose, region classification comes from SLIC
/// superpixels (`crate::processing::slic`) rather than BiSeNet segmentation.
/// Within each SLIC region, depth is normalized by MAD (median absolute
/// deviation) and used to bias Lab L* — producing discrete shading bands
/// that follow both color and depth boundaries.
///
/// # Depth convention
///
/// Depth-Anything V2 outputs **0 = nearest, 1 = farthest**. The implementation
/// inverts this for shading so near features (nose tip) become highlights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthToFlatConfig {
    // ── Per-region shading (Phase 2.3) ────────────────────────────────────────

    /// Shading intensity. How strongly the depth-derived shading signal biases
    /// Lab L*. 0.0 = no shading, 0.6 = balanced (default), 1.0 = max contrast.
    #[serde(default = "default_dtf_strength")]
    pub strength: f32,

    /// Contrast curve exponent. `s' = sign(s) * |s|^gamma`.
    /// 1.0 = linear, <1.0 = more contrast in midtones (default 0.8), >1.0 = compressed.
    #[serde(default = "default_dtf_gamma")]
    pub gamma: f32,

    /// MAD (median absolute deviation) threshold for low-variance region skip.
    /// Regions with MAD below this get no shading (avoids amplifying noise in
    /// flat backgrounds / solid-color hair). Default 0.02, range 0.01–0.2.
    /// Lower = shade more regions (only truly flat regions are skipped).
    #[serde(default = "default_dtf_mad_threshold")]
    pub mad_threshold: f32,

    /// Blend between local (per-region MAD) and global (whole-image min-max)
    /// depth shading. 0.0 = pure local (current behavior), 1.0 = pure global,
    /// 0.5 = balanced blend (default).
    ///
    /// Local shading preserves fine depth detail within each SLIC region but
    /// loses global depth relationships (a far background region and a near
    /// foreground region both get normalized to the same [-1, 1] range).
    /// Global shading preserves relative depth between all pixels but flattens
    /// local detail. Blending gives both: local detail with global context.
    #[serde(default = "default_dtf_global_depth_weight")]
    pub global_depth_weight: f32,

    // ── Background separation ─────────────────────────────────────────────────

    /// When true, compute the foreground/background depth split automatically
    /// per image using Otsu's method on the depth histogram. Default: true.
    ///
    /// **Phase 3**: this is now a *fallback* when SLIC labels are unavailable.
    /// When SLIC labels are present, region-based background classification
    /// is used instead (per-SLIC-cluster mean depth + border/size rules).
    pub use_otsu_threshold: bool,

    /// Manual depth threshold: pixels with depth > this are background.
    /// Only used when `use_otsu_threshold = false`. Range 0.01–0.99. Default 0.6.
    pub bg_depth_threshold: f32,

    /// How much to pull background pixel saturation toward gray.
    /// 0.0 = no change, 1.0 = fully desaturate. Default 0.65.
    pub bg_desaturation: f32,

    /// L* shift applied to background pixels in Lab space.
    /// Negative = darken, positive = lighten. Scaled by ×100 before applying.
    /// Range −1.0 to +1.0. Default −0.08 (subtle darkening).
    pub bg_lightness_shift: f32,

    // ── Phase 3 — region-based background classification ──────────────────────

    /// Minimum cluster size (as a fraction of total image pixels) for a
    /// SLIC cluster to be classified as background even if it doesn't touch
    /// the image border. Default 0.15 (15%). Range 0.0–1.0.
    ///
    /// A cluster is background if `mean_depth > p70 AND (touches_border OR
    /// size_pct > bg_cluster_size_pct)`. The border-touch rule catches
    /// skies / horizon backgrounds; the size rule catches large flat regions
    /// that don't reach the edge (e.g., a wall behind the subject).
    #[serde(default = "default_dtf_bg_cluster_size_pct")]
    pub bg_cluster_size_pct: f32,
}

fn default_dtf_strength() -> f32 { 0.6 }
fn default_dtf_gamma() -> f32 { 0.8 }
fn default_dtf_mad_threshold() -> f32 { 0.02 }
/// Phase 3 — C6: bias toward global truth (0.6/0.4 split). User's complaint
/// was "missing out on global depth values" — pure 0.5/0.5 blend destroyed
/// both signals; 0.6/0.4 keeps global depth relationships dominant while
/// still letting local detail through.
fn default_dtf_global_depth_weight() -> f32 { 0.6 }
fn default_dtf_bg_cluster_size_pct() -> f32 { 0.15 }

impl Default for DepthToFlatConfig {
    fn default() -> Self {
        Self {
            strength:              default_dtf_strength(),
            gamma:                 default_dtf_gamma(),
            mad_threshold:         default_dtf_mad_threshold(),
            global_depth_weight:   default_dtf_global_depth_weight(),
            use_otsu_threshold:    true,
            bg_depth_threshold:    0.6,
            bg_desaturation:       0.65,
            bg_lightness_shift:    -0.08,
            bg_cluster_size_pct:   default_dtf_bg_cluster_size_pct(),
        }
    }
}

// =============================================================================
// Edge Configuration
// =============================================================================

/// Outline coloring strategy for the edge pass.
///
/// Phase 7 — H4: removed `AutoContrastWithHueShift` variant (it was a no-op
/// kept for backwards-compat; the local-contrast `AutoContrast` mode from
/// Phase 2 supersedes it). Also removed the associated `delta` and `hue_shift`
/// fields from `EdgeConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutlineStyle {
    /// Local-contrast: pick the palette color that maximizes Lab ΔE from
    /// the 3×3 neighborhood mean of the edge pixel. Outline color varies
    /// across the image based on local background. (Phase 2 implementation.)
    #[default]
    AutoContrast,
    /// Fixed darkest palette color — for retro palettes (Game Boy, NES).
    Black,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    pub edge_mode: EdgeMode,
    pub thickness: u32,
    pub edge_color_mode: EdgeColorMode,
    pub custom_edge_color: egui::Color32,
    pub edge_darkener_strength: f32,
    pub anti_alias_edges: bool,
    // ── Edge pass ────────────────────────────────────────────────────────────
    /// How outline colors are chosen. See [`OutlineStyle`].
    #[serde(default)]
    pub outline_style: OutlineStyle,
    /// TEED edge probability threshold. Pixels above this are edges.
    /// Default 0.3, range 0.1–0.7. Lower = more edges, higher = only strong edges.
    #[serde(default = "default_teed_threshold")]
    pub teed_threshold: f32,
}

fn default_teed_threshold() -> f32 { 0.3 }

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_mode: EdgeMode::Outlines,
            thickness: 1,
            edge_color_mode: EdgeColorMode::DarkestShade,
            custom_edge_color: egui::Color32::BLACK,
            edge_darkener_strength: 0.3,
            anti_alias_edges: false,
            outline_style: OutlineStyle::AutoContrast,
            teed_threshold: default_teed_threshold(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeMode {
    None,
    #[default]
    Outlines,
    Internal,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeColorMode {
    Black,
    #[default]
    DarkestShade,
    Custom,
}

// =============================================================================
// Palette Configuration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteConfig {
    pub mode: PaletteMode,
    pub max_colors: u32,
    /// Inert after the repurpose: per-region palette limits required BiSeNet
    /// segmentation, which is gone. Kept for preset-file backwards-compat.
    #[serde(default)]
    pub per_region_limit: bool,
    pub preset: PresetPalette,
    pub custom_colors: Vec<egui::Color32>,
    pub skin_override: Option<Vec<egui::Color32>>,
    pub hair_override: Option<Vec<egui::Color32>>,
    pub eye_override: Option<egui::Color32>,
    pub lip_override: Option<egui::Color32>,
    pub background_override: Option<Vec<egui::Color32>>,
}

impl Default for PaletteConfig {
    fn default() -> Self {
        Self {
            mode: PaletteMode::Auto,
            max_colors: 32,
            per_region_limit: false,
            preset: PresetPalette::None,
            custom_colors: Vec::new(),
            skin_override: None,
            hair_override: None,
            eye_override: None,
            lip_override: None,
            background_override: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaletteMode {
    #[default]
    Auto,
    Preset,
    Custom,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PresetPalette {
    #[default]
    None,
    GameBoy,
    GameBoyColor,
    PICO8,
    NES,
    DawnBringer32,
    AAP64,
}

// =============================================================================
// Processing State
// =============================================================================

#[derive(Debug, Clone)]
pub enum ProcessingState {
    Idle,
    Running(ProcessingStatus),
    Complete,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ProcessingStatus {
    pub progress: f32,
    pub stage: String,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self::Idle
    }
}

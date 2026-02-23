//! Processing configuration structures

use serde::{Deserialize, Serialize};

// =============================================================================
// Downsampling Method
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DownsamplingMethod {
    #[default]
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
    pub clip_to_face: bool,
    pub clip_padding: f32,
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
            clip_to_face: false,
            clip_padding: 0.2,
            export_scale: 1,
            downsampling_method: DownsamplingMethod::Weighted,
        }
    }
}

// =============================================================================
// Depth-to-Flat Configuration
// =============================================================================

/// Depth-to-flat color conversion configuration.
///
/// Controls two distinct jobs:
///
/// ## Background separation (most reliable depth use)
///
/// Pixels classified as background (by BiSeNet label OR by depth beyond the
/// threshold) are desaturated and optionally darkened. When `use_otsu_threshold`
/// is true (default), the foreground/background split is computed automatically
/// per image using Otsu's method on the depth histogram.
///
/// ## Foreground shading reinforcement (moderate reliability)
///
/// Within BiSeNet-defined foreground regions, depth nudges the photo's existing
/// lightness toward discrete tonal bands. `depth_influence` controls how much
/// weight depth has vs. the original photo shading.  High-detail facial regions
/// (eyes, lips, nose) automatically receive a greatly reduced influence — depth
/// geometry at that scale is too noisy to be useful.
///
/// # Depth convention
///
/// Depth-Anything V2 outputs **0 = nearest, 1 = farthest**. The implementation
/// inverts this for shading so near features (nose tip) become highlights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthToFlatConfig {
    // ── Tonal band counts ─────────────────────────────────────────────────────

    /// Number of discrete tonal bands for skin regions (Skin, Neck). Range 1–16.
    pub skin_tone_bands: u32,

    /// Number of tonal bands for hair regions (Hair, Hat). Range 1–16.
    pub hair_bands: u32,

    /// Number of tonal bands for clothing and non-face accessories. Range 1–16.
    pub clothing_bands: u32,

    /// Number of tonal bands when depth is used for background shading.
    /// Usually irrelevant (background is desaturated instead), but kept
    /// for backwards compatibility. Range 1–8.
    pub background_bands: u32,

    // ── Shading thresholds ────────────────────────────────────────────────────

    /// Band position (0–1) below which pixels enter the shadow zone. Default 0.25.
    pub shadow_threshold: f32,

    /// Band position (0–1) above which pixels enter the highlight zone. Default 0.75.
    pub highlight_threshold: f32,

    // ── Gradient preservation ─────────────────────────────────────────────────

    /// When true, blends in some of the original lightness to soften band edges.
    pub preserve_gradients: bool,

    /// Fraction of original L to preserve when banding is active (0 = pure bands,
    /// 1 = no banding). Only used when `preserve_gradients` is true. Default 0.0.
    pub gradient_preservation: f32,

    // ── Depth influence ───────────────────────────────────────────────────────

    /// How strongly depth-derived shading overrides the photo's own lighting.
    ///
    /// 0.0 = photo lightness unchanged (depth has no effect on shading)  
    /// 0.4 = balanced blend (recommended default)  
    /// 1.0 = pure depth-derived bands, photo lighting ignored
    ///
    /// High-detail facial regions (eyes, lips, nose) are automatically clamped
    /// to ~15% of this value regardless of the setting.
    pub depth_influence: f32,

    // ── Background separation ─────────────────────────────────────────────────

    /// When true, compute the foreground/background depth split automatically
    /// per image using Otsu's method on the depth histogram. Default: true.
    ///
    /// Disable and set `bg_depth_threshold` manually for fine control.
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
}

impl Default for DepthToFlatConfig {
    fn default() -> Self {
        Self {
            skin_tone_bands:       4,
            hair_bands:            3,
            clothing_bands:        3,
            background_bands:      2,
            shadow_threshold:      0.25,
            highlight_threshold:   0.75,
            preserve_gradients:    false,
            gradient_preservation: 0.0,
            // New fields
            depth_influence:       0.4,
            use_otsu_threshold:    true,
            bg_depth_threshold:    0.6,
            bg_desaturation:       0.65,
            bg_lightness_shift:    -0.08,
        }
    }
}

// =============================================================================
// Feature Preservation Configuration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePreserveConfig {
    pub eye_size: EyeSize,
    pub eye_detail: DetailLevel,
    pub lip_detail: DetailLevel,
    pub nose_detail: DetailLevel,
    pub force_eye_highlights: bool,
    pub distinct_nostrils: bool,
    pub feature_sharpening: f32,
}

impl Default for FeaturePreserveConfig {
    fn default() -> Self {
        Self {
            eye_size: EyeSize::Small,
            eye_detail: DetailLevel::Minimal,
            lip_detail: DetailLevel::Minimal,
            nose_detail: DetailLevel::Minimal,
            force_eye_highlights: true,
            distinct_nostrils: false,
            feature_sharpening: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EyeSize {
    Auto,
    #[default]
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetailLevel {
    #[default]
    Minimal,
    Standard,
    Full,
}

// =============================================================================
// Edge Configuration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    pub edge_mode: EdgeMode,
    pub thickness: u32,
    pub edge_color_mode: EdgeColorMode,
    pub custom_edge_color: egui::Color32,
    pub edge_darkener_strength: f32,
    pub anti_alias_edges: bool,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_mode: EdgeMode::Outlines,
            thickness: 1,
            edge_color_mode: EdgeColorMode::DarkestShade,
            custom_edge_color: egui::Color32::BLACK,
            edge_darkener_strength: 0.3,
            anti_alias_edges: false,
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
            max_colors: 16,
            per_region_limit: true,
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
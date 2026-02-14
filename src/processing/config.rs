//! Processing configuration structures

use serde::{Deserialize, Serialize};

/// Transform configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    /// Output size in pixels
    pub output_size: u32,

    /// Scale factor for input image
    pub scale: f32,

    /// Rotation in degrees
    pub rotation: f32,

    /// X offset (normalized -1 to 1)
    pub offset_x: f32,

    /// Y offset (normalized -1 to 1)
    pub offset_y: f32,

    /// Whether to clip to face region
    pub clip_to_face: bool,

    /// Padding around face when clipping
    pub clip_padding: f32,

    /// Export scale multiplier
    pub export_scale: u32,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            output_size: 32,
            scale: 1.0,
            rotation: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            clip_to_face: false,
            clip_padding: 0.2,
            export_scale: 1,
        }
    }
}

/// Depth-to-flat color conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthToFlatConfig {
    /// Number of color bands for skin tones
    pub skin_tone_bands: u32,

    /// Number of color bands for hair
    pub hair_bands: u32,

    /// Number of color bands for clothing
    pub clothing_bands: u32,

    /// Number of color bands for background
    pub background_bands: u32,

    /// Depth threshold for shadows (0.0-1.0)
    pub shadow_threshold: f32,

    /// Depth threshold for highlights (0.0-1.0)
    pub highlight_threshold: f32,

    /// Whether to preserve some gradients
    pub preserve_gradients: bool,

    /// Gradient preservation strength (0.0-1.0)
    pub gradient_preservation: f32,
}

impl Default for DepthToFlatConfig {
    fn default() -> Self {
        Self {
            skin_tone_bands: 4,
            hair_bands: 3,
            clothing_bands: 3,
            background_bands: 2,
            shadow_threshold: 0.25,
            highlight_threshold: 0.75,
            preserve_gradients: false,
            gradient_preservation: 0.0,
        }
    }
}

/// Feature preservation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePreserveConfig {
    /// Eye size mode
    pub eye_size: EyeSize,

    /// Eye detail level
    pub eye_detail: DetailLevel,

    /// Lip detail level
    pub lip_detail: DetailLevel,

    /// Nose detail level
    pub nose_detail: DetailLevel,

    /// Force eye highlights
    pub force_eye_highlights: bool,

    /// Render distinct nostrils
    pub distinct_nostrils: bool,

    /// Feature sharpening strength (0.0-1.0)
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

/// Eye size modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EyeSize {
    Auto,
    #[default]
    Small,
    Medium,
    Large,
}

/// Detail level for features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetailLevel {
    #[default]
    Minimal,
    Standard,
    Full,
}

/// Edge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Edge drawing mode
    pub edge_mode: EdgeMode,

    /// Edge thickness in pixels
    pub thickness: u32,

    /// Edge color mode
    pub edge_color_mode: EdgeColorMode,

    /// Custom edge color (RGBA)
    pub custom_edge_color: egui::Color32,

    /// Edge darkener strength (0.0-1.0)
    pub edge_darkener_strength: f32,

    /// Anti-alias edges
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

/// Edge drawing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeMode {
    None,
    #[default]
    Outlines,
    Internal,
    Both,
}

/// Edge color modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeColorMode {
    Black,
    #[default]
    DarkestShade,
    Custom,
}

/// Palette configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteConfig {
    /// Palette generation mode
    pub mode: PaletteMode,

    /// Maximum colors for auto mode
    pub max_colors: u32,

    /// Limit colors per region
    pub per_region_limit: bool,

    /// Preset palette selection
    pub preset: PresetPalette,

    /// Custom colors
    pub custom_colors: Vec<egui::Color32>,

    /// Skin tone overrides
    pub skin_override: Option<Vec<egui::Color32>>,

    /// Hair color overrides
    pub hair_override: Option<Vec<egui::Color32>>,

    /// Eye color override
    pub eye_override: Option<egui::Color32>,

    /// Lip color override
    pub lip_override: Option<egui::Color32>,

    /// Background color overrides
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

/// Palette generation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaletteMode {
    #[default]
    Auto,
    Preset,
    Custom,
    Hybrid,
}

/// Built-in preset palettes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PresetPalette {
    #[default]
    None,
    GameBoy,
    GameBoyColor,
    NES,
    PICO8,
    DawnBringer32,
    AAP64,
}

/// Complete processing configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub transform: TransformConfig,
    pub depth_to_flat: DepthToFlatConfig,
    pub features: FeaturePreserveConfig,
    pub edges: EdgeConfig,
    pub palette: PaletteConfig,
}

/// Processing state
#[derive(Debug, Clone)]
pub enum ProcessingState {
    Idle,
    Running(ProcessingStatus),
    Complete,
    Error(String),
}

/// Processing status for progress display
#[derive(Debug, Clone)]
pub struct ProcessingStatus {
    /// Progress (0.0-1.0)
    pub progress: f32,

    /// Current stage description
    pub stage: String,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self::Idle
    }
}

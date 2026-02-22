//! Processing configuration structures
//!
//! This module defines all configuration structures used throughout
//! the pixel art processing pipeline:
//!
//! - **TransformConfig**: Image transformations (scale, rotation, flip)
//! - **DepthToFlatConfig**: Depth-to-flat color conversion settings
//! - **FeaturePreserveConfig**: Facial feature preservation settings
//! - **EdgeConfig**: Edge detection and rendering settings
//! - **PaletteConfig**: Color palette generation settings
//!
//! NOTE: This file was synced from server - includes flip_horizontal/flip_vertical fields.

use serde::{Deserialize, Serialize};

// =============================================================================
// Downsampling Method
// =============================================================================

/// Downsampling method for reducing image resolution.
///
/// Each method has different trade-offs between quality and speed:
///
/// | Method | Quality | Speed | Best For |
/// |--------|---------|-------|----------|
/// | Weighted | High | Slow | Portraits with ML analysis |
/// | NearestNeighbor | Low | Fast | Retro/pixel art style |
/// | Bilinear | Medium | Medium | Smooth results |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DownsamplingMethod {
    /// Content-aware weighted downsampling.
    /// Uses importance maps from ML analysis to preserve important details.
    /// Best quality but slowest.
    #[default]
    Weighted,
    
    /// Simple nearest-neighbor downsampling.
    /// Fast but produces blocky results. Good for retro style.
    NearestNeighbor,
    
    /// Bilinear interpolation downsampling.
    /// Smooth results, may blur fine details.
    Bilinear,
}

// =============================================================================
// Transform Configuration
// =============================================================================

/// Image transformation configuration.
///
/// Controls how the input image is transformed before pixel art processing:
///
/// 1. **Scale**: Resize the input image
/// 2. **Rotation**: Rotate the image
/// 3. **Offset**: Pan the image
/// 4. **Flip**: Mirror horizontally/vertically
/// 5. **Clip to Face**: Crop to detected face region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    // -------------------------------------------------------------------------
    // Output Size
    // -------------------------------------------------------------------------
    
    /// Output size in pixels.
    ///
    /// For Square mode: Both width and height
    /// For Preserve mode: Maximum dimension
    pub output_size: u32,

    // -------------------------------------------------------------------------
    // Transformations
    // -------------------------------------------------------------------------
    
    /// Scale factor for input image.
    ///
    /// - 1.0 = original size
    /// - 0.5 = half size
    /// - 2.0 = double size
    pub scale: f32,

    /// Rotation in degrees.
    ///
    /// Positive = clockwise, Negative = counter-clockwise
    /// Range: -180.0 to 180.0
    pub rotation: f32,

    /// X offset (normalized -1 to 1).
    ///
    /// -1.0 = shift left, 0.0 = center, 1.0 = shift right
    pub offset_x: f32,

    /// Y offset (normalized -1 to 1).
    ///
    /// -1.0 = shift up, 0.0 = center, 1.0 = shift down
    pub offset_y: f32,

    /// Flip horizontally (mirror left-right).
    pub flip_horizontal: bool,

    /// Flip vertically (mirror top-bottom).
    pub flip_vertical: bool,

    // -------------------------------------------------------------------------
    // Face Clipping
    // -------------------------------------------------------------------------
    
    /// Whether to crop the image to the detected face region.
    ///
    /// When enabled, the image is cropped to focus on the face,
    /// which produces better results for portrait pixel art.
    pub clip_to_face: bool,

    /// Padding around face when clipping (0.0 to 1.0).
    ///
    /// Fraction of face size to add as padding.
    /// - 0.0 = tight crop
    /// - 0.2 = 20% padding (default)
    /// - 1.0 = 100% padding (face in center with room for hair/shoulders)
    pub clip_padding: f32,

    // -------------------------------------------------------------------------
    // Export
    // -------------------------------------------------------------------------
    
    /// Export scale multiplier.
    ///
    /// Multiplies the output dimensions for export:
    /// - 1 = original pixel art size
    /// - 2 = 2x size (good for sharing)
    /// - 4 = 4x size
    /// - 8 = 8x size
    pub export_scale: u32,
    
    // -------------------------------------------------------------------------
    // Downsampling
    // -------------------------------------------------------------------------
    
    /// Method used for downsampling the image to pixel art size.
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
/// Controls how the depth map is converted to discrete color bands,
/// simulating the limited palette of pixel art while preserving
/// the 3D form through shading.
///
/// # Color Bands
///
/// Each region type (skin, hair, clothes, background) can have a
/// different number of color bands:
/// - More bands = smoother gradients, less "pixel art" feel
/// - Fewer bands = higher contrast, more stylized
///
/// # Example
///
/// For a 32x32 output with 4 skin bands:
/// - Each band covers approximately 25% of the depth range
/// - Creates 4 distinct skin tones (highlight, light, shadow, dark)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthToFlatConfig {
    /// Number of color bands for skin tones (1-16).
    ///
    /// More bands = smoother skin, less pixel art style
    pub skin_tone_bands: u32,

    /// Number of color bands for hair (1-16).
    pub hair_bands: u32,

    /// Number of color bands for clothing (1-16).
    pub clothing_bands: u32,

    /// Number of color bands for background (1-8).
    ///
    /// Usually fewer than face/hair to keep focus on subject
    pub background_bands: u32,

    /// Depth threshold for shadows (0.0-1.0).
    ///
    /// Pixels with depth below this value are darkened.
    /// Lower = more shadows, Higher = fewer shadows
    pub shadow_threshold: f32,

    /// Depth threshold for highlights (0.0-1.0).
    ///
    /// Pixels with depth above this value are brightened.
    /// Higher = more highlights, Lower = fewer highlights
    pub highlight_threshold: f32,

    /// Whether to preserve some gradient information.
    ///
    /// When true, adds subtle gradients between color bands
    /// for a more natural look.
    pub preserve_gradients: bool,

    /// Gradient preservation strength (0.0-1.0).
    ///
    /// How much gradient information to preserve.
    /// Only used when preserve_gradients is true.
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

// =============================================================================
// Feature Preservation Configuration
// =============================================================================

/// Feature preservation configuration.
///
/// Controls how facial features are preserved and rendered in the
/// pixel art output. At small sizes, features need special handling
/// to remain recognizable.
///
/// # Eye Size
///
/// Eyes are critical for recognition. The minimum size ensures
/// eyes are always visible:
/// - Auto: Scale based on output size
/// - Small: 2 pixels (good for 16x16)
/// - Medium: 3 pixels (good for 32x32)
/// - Large: 4 pixels (good for 64x64+)
///
/// # Detail Level
///
/// Controls how much detail to render in facial features:
/// - Minimal: Single pixels for features (16x16)
/// - Standard: Basic shapes (32x32)
/// - Full: Detailed with shading (64x64+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePreserveConfig {
    /// Eye size mode.
    pub eye_size: EyeSize,

    /// Eye detail level.
    pub eye_detail: DetailLevel,

    /// Lip detail level.
    pub lip_detail: DetailLevel,

    /// Nose detail level.
    pub nose_detail: DetailLevel,

    /// Force eye highlights.
    ///
    /// When true, always adds a white pixel for eye reflection,
    /// making eyes appear more alive.
    pub force_eye_highlights: bool,

    /// Render distinct nostrils.
    ///
    /// When true, adds dark pixels for nostril shadows.
    pub distinct_nostrils: bool,

    /// Feature sharpening strength (0.0-1.0).
    ///
    /// Higher values make features more defined.
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

/// Eye size modes for feature preservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EyeSize {
    /// Automatically scale based on output size.
    Auto,
    
    /// Small eyes (2 pixels minimum).
    #[default]
    Small,
    
    /// Medium eyes (3 pixels minimum).
    Medium,
    
    /// Large eyes (4 pixels minimum).
    Large,
}

/// Detail level for facial features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetailLevel {
    /// Minimal detail - single pixel features.
    /// Best for 16x16 or smaller.
    #[default]
    Minimal,
    
    /// Standard detail - basic shapes.
    /// Best for 32x32.
    Standard,
    
    /// Full detail - shading and gradients.
    /// Best for 64x64 or larger.
    Full,
}

// =============================================================================
// Edge Configuration
// =============================================================================

/// Edge detection and rendering configuration.
///
/// Edges define the pixel art style by creating clear boundaries
/// between regions. This config controls how edges are detected
/// and rendered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Edge drawing mode.
    pub edge_mode: EdgeMode,

    /// Edge thickness in pixels (1-4).
    pub thickness: u32,

    /// Edge color mode.
    pub edge_color_mode: EdgeColorMode,

    /// Custom edge color (RGBA).
    /// Only used when edge_color_mode is Custom.
    pub custom_edge_color: egui::Color32,

    /// Edge darkener strength (0.0-1.0).
    ///
    /// How much to darken the color of edge pixels.
    /// Higher = darker edges, more contrast.
    pub edge_darkener_strength: f32,

    /// Anti-alias edges.
    ///
    /// When true, smooths edge pixels for a softer look.
    /// Usually false for pixel art to maintain sharp edges.
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

/// Edge drawing modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeMode {
    /// No edges - clean pixel art without outlines.
    None,
    
    /// Outline edges only - outer boundaries of regions.
    /// Classic pixel art style.
    #[default]
    Outlines,
    
    /// Internal edges only - edges within regions.
    /// Good for adding detail without outlining.
    Internal,
    
    /// Both outline and internal edges.
    /// Maximum definition, may look busy.
    Both,
}

/// Edge color modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeColorMode {
    /// Pure black edges (#000000).
    Black,
    
    /// Use the darkest shade in each region.
    /// Softer look, more integrated with the image.
    #[default]
    DarkestShade,
    
    /// Custom color defined by user.
    Custom,
}

// =============================================================================
// Palette Configuration
// =============================================================================

/// Color palette configuration.
///
/// Controls how colors are quantized for the pixel art output.
/// Three modes are available:
///
/// # Auto Mode
///
/// Automatically extracts colors from the image:
/// - Analyzes color distribution
/// - Limits to max_colors
/// - Optional per-region limiting
///
/// # Preset Mode
///
/// Uses a built-in palette:
/// - Game Boy: 4 colors, green monochrome
/// - Game Boy Color: 8 colors
/// - PICO-8: 16 colors, distinctive retro look
/// - NES: 54 colors, 8-bit console palette
/// - DawnBringer32: 32 colors, popular for pixel art
/// - AAP64: 64 colors, extended palette
///
/// # Custom Mode
///
/// User-defined colors (not yet implemented).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteConfig {
    /// Palette generation mode.
    pub mode: PaletteMode,

    /// Maximum colors for auto mode (2-256).
    pub max_colors: u32,

    /// Limit colors per region.
    ///
    /// When true, each segmented region (face, hair, etc.)
    /// gets its own limited palette for better color separation.
    pub per_region_limit: bool,

    /// Preset palette selection.
    pub preset: PresetPalette,

    /// Custom colors for custom mode.
    pub custom_colors: Vec<egui::Color32>,

    /// Skin tone color overrides.
    /// When set, replaces detected skin tones.
    pub skin_override: Option<Vec<egui::Color32>>,

    /// Hair color overrides.
    pub hair_override: Option<Vec<egui::Color32>>,

    /// Eye color override.
    pub eye_override: Option<egui::Color32>,

    /// Lip color override.
    pub lip_override: Option<egui::Color32>,

    /// Background color overrides.
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

/// Palette generation modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaletteMode {
    /// Automatically extract colors from the image.
    #[default]
    Auto,
    
    /// Use a built-in preset palette.
    Preset,
    
    /// Use user-defined custom colors.
    Custom,
    
    /// Combine auto extraction with presets.
    Hybrid,
}

/// Built-in preset palettes.
///
/// These palettes are designed for pixel art and provide
/// distinctive visual styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PresetPalette {
    /// No preset selected.
    #[default]
    None,
    
    /// Nintendo Game Boy (4 colors).
    /// Green monochrome, iconic retro look.
    GameBoy,
    
    /// Nintendo Game Boy Color (8 colors).
    /// Limited but versatile.
    GameBoyColor,
    
    /// PICO-8 fantasy console (16 colors).
    /// Bright, distinctive, very popular.
    PICO8,
    
    /// Nintendo Entertainment System (~54 colors).
    /// Classic 8-bit console palette.
    NES,
    
    /// DawnBringer 32 (32 colors).
    /// Designed specifically for pixel art.
    DawnBringer32,
    
    /// Arne's 64 color palette.
    /// Extended version of DawnBringer.
    AAP64,
}

// =============================================================================
// Processing State
// =============================================================================

/// Processing state for tracking progress.
///
/// Used by the UI to display the current state of processing.
#[derive(Debug, Clone)]
pub enum ProcessingState {
    /// No processing in progress.
    Idle,
    
    /// Processing is running with progress.
    Running(ProcessingStatus),
    
    /// Processing completed successfully.
    Complete,
    
    /// Processing failed with an error.
    Error(String),
}

/// Processing status for progress display.
#[derive(Debug, Clone)]
pub struct ProcessingStatus {
    /// Progress (0.0-1.0).
    pub progress: f32,

    /// Current stage description.
    pub stage: String,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self::Idle
    }
}

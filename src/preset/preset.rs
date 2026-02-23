//! Preset save/load functionality

use crate::processing::{
    DepthToFlatConfig, EdgeConfig, FeaturePreserveConfig, PaletteConfig, TransformConfig,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Processing preset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    /// Preset name
    pub name: String,

    /// Transform settings
    pub transform: TransformConfig,

    /// Depth-to-flat settings
    pub depth_to_flat: DepthToFlatConfig,

    /// Feature preservation settings
    pub features: FeaturePreserveConfig,

    /// Edge settings
    pub edges: EdgeConfig,

    /// Palette settings
    pub palette: PaletteConfig,
}

impl Default for Preset {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            transform: TransformConfig::default(),
            depth_to_flat: DepthToFlatConfig::default(),
            features: FeaturePreserveConfig::default(),
            edges: EdgeConfig::default(),
            palette: PaletteConfig::default(),
        }
    }
}

impl Preset {
    /// Save preset to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        log::info!("Saved preset: {:?}", path);
        Ok(())
    }

    /// Load preset from file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let preset: Preset = serde_json::from_str(&json)?;
        log::info!("Loaded preset: {:?}", path);
        Ok(preset)
    }

    /// Get built-in presets
    pub fn built_in_presets() -> Vec<Preset> {
        vec![
            // Portrait - Minimal
            Preset {
                name: "Portrait - Minimal".to_string(),
                transform: TransformConfig {
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
                    downsampling_method: crate::processing::DownsamplingMethod::Weighted,
                },
                depth_to_flat: DepthToFlatConfig {
                    skin_tone_bands: 3,
                    hair_bands: 2,
                    clothing_bands: 2,
                    background_bands: 1,
                    shadow_threshold: 0.25,
                    highlight_threshold: 0.75,
                    preserve_gradients: false,
                    gradient_preservation: 0.0,
                    depth_influence: 0.4,
                    use_otsu_threshold: true,
                    bg_depth_threshold: 0.6,
                    bg_desaturation: 0.65,
                    bg_lightness_shift: 0.0,
                },
                features: FeaturePreserveConfig {
                    eye_size: crate::processing::EyeSize::Small,
                    eye_detail: crate::processing::DetailLevel::Minimal,
                    lip_detail: crate::processing::DetailLevel::Minimal,
                    nose_detail: crate::processing::DetailLevel::Minimal,
                    force_eye_highlights: true,
                    distinct_nostrils: false,
                    feature_sharpening: 0.5,
                },
                edges: EdgeConfig {
                    edge_mode: crate::processing::EdgeMode::Outlines,
                    thickness: 1,
                    edge_color_mode: crate::processing::EdgeColorMode::DarkestShade,
                    custom_edge_color: egui::Color32::BLACK,
                    edge_darkener_strength: 0.3,
                    anti_alias_edges: false,
                },
                palette: PaletteConfig {
                    mode: crate::processing::PaletteMode::Auto,
                    max_colors: 12,
                    per_region_limit: true,
                    preset: crate::processing::PresetPalette::None,
                    custom_colors: Vec::new(),
                    skin_override: None,
                    hair_override: None,
                    eye_override: None,
                    lip_override: None,
                    background_override: None,
                },
            },

            // Portrait - Detailed
            Preset {
                name: "Portrait - Detailed".to_string(),
                transform: TransformConfig {
                    output_size: 64,
                    scale: 1.0,
                    rotation: 0.0,
                    offset_x: 0.0,
                    offset_y: 0.0,
                    flip_horizontal: false,
                    flip_vertical: false,
                    clip_to_face: false,
                    clip_padding: 0.2,
                    export_scale: 1,
                    downsampling_method: crate::processing::DownsamplingMethod::Weighted,
                },
                depth_to_flat: DepthToFlatConfig {
                    skin_tone_bands: 5,
                    hair_bands: 4,
                    clothing_bands: 3,
                    background_bands: 2,
                    shadow_threshold: 0.2,
                    highlight_threshold: 0.8,
                    preserve_gradients: false,
                    gradient_preservation: 0.0,
                    depth_influence: 0.4,
                    use_otsu_threshold: true,
                    bg_depth_threshold: 0.6,
                    bg_desaturation: 0.65,
                    bg_lightness_shift: 0.0,
                },
                features: FeaturePreserveConfig {
                    eye_size: crate::processing::EyeSize::Medium,
                    eye_detail: crate::processing::DetailLevel::Standard,
                    lip_detail: crate::processing::DetailLevel::Standard,
                    nose_detail: crate::processing::DetailLevel::Standard,
                    force_eye_highlights: true,
                    distinct_nostrils: true,
                    feature_sharpening: 0.7,
                },
                edges: EdgeConfig {
                    edge_mode: crate::processing::EdgeMode::Both,
                    thickness: 1,
                    edge_color_mode: crate::processing::EdgeColorMode::DarkestShade,
                    custom_edge_color: egui::Color32::BLACK,
                    edge_darkener_strength: 0.4,
                    anti_alias_edges: false,
                },
                palette: PaletteConfig {
                    mode: crate::processing::PaletteMode::Auto,
                    max_colors: 32,
                    per_region_limit: true,
                    preset: crate::processing::PresetPalette::None,
                    custom_colors: Vec::new(),
                    skin_override: None,
                    hair_override: None,
                    eye_override: None,
                    lip_override: None,
                    background_override: None,
                },
            },

            // Game Boy Style
            Preset {
                name: "Game Boy Style".to_string(),
                transform: TransformConfig {
                    output_size: 48,
                    scale: 1.0,
                    rotation: 0.0,
                    offset_x: 0.0,
                    offset_y: 0.0,
                    flip_horizontal: false,
                    flip_vertical: false,
                    clip_to_face: false,
                    clip_padding: 0.2,
                    export_scale: 2,
                    downsampling_method: crate::processing::DownsamplingMethod::NearestNeighbor,
                },
                depth_to_flat: DepthToFlatConfig {
                    skin_tone_bands: 2,
                    hair_bands: 2,
                    clothing_bands: 2,
                    background_bands: 1,
                    shadow_threshold: 0.3,
                    highlight_threshold: 0.7,
                    preserve_gradients: false,
                    gradient_preservation: 0.0,
                    depth_influence: 0.4,
                    use_otsu_threshold: true,
                    bg_depth_threshold: 0.6,
                    bg_desaturation: 0.65,
                    bg_lightness_shift: 0.0,
                },
                features: FeaturePreserveConfig {
                    eye_size: crate::processing::EyeSize::Small,
                    eye_detail: crate::processing::DetailLevel::Minimal,
                    lip_detail: crate::processing::DetailLevel::Minimal,
                    nose_detail: crate::processing::DetailLevel::Minimal,
                    force_eye_highlights: true,
                    distinct_nostrils: false,
                    feature_sharpening: 0.3,
                },
                edges: EdgeConfig {
                    edge_mode: crate::processing::EdgeMode::Outlines,
                    thickness: 1,
                    edge_color_mode: crate::processing::EdgeColorMode::Black,
                    custom_edge_color: egui::Color32::BLACK,
                    edge_darkener_strength: 0.2,
                    anti_alias_edges: false,
                },
                palette: PaletteConfig {
                    mode: crate::processing::PaletteMode::Preset,
                    max_colors: 4,
                    per_region_limit: false,
                    preset: crate::processing::PresetPalette::GameBoy,
                    custom_colors: Vec::new(),
                    skin_override: None,
                    hair_override: None,
                    eye_override: None,
                    lip_override: None,
                    background_override: None,
                },
            },
        ]
    }
}

//! Application configuration management

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Model quality setting
    pub model_quality: ModelQuality,

    /// Use sequential processing (for low VRAM)
    pub sequential_processing: bool,

    /// UI preferences
    pub ui: UiConfig,

    /// Processing defaults
    pub processing: ProcessingDefaults,

    /// Last used directories
    pub directories: DirectoryConfig,

    /// Model management
    pub models: ModelConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelQuality {
    #[default]
    Minimal,
    Standard,
    High,
}

impl ModelQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelQuality::Minimal => "minimal",
            ModelQuality::Standard => "standard",
            ModelQuality::High => "high",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ModelQuality::Minimal => "Minimal (Fast)",
            ModelQuality::Standard => "Standard",
            ModelQuality::High => "High Quality",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Auto-download missing models
    pub auto_download: bool,

    /// Check for model updates on startup
    pub check_updates: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            auto_download: true,
            check_updates: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Theme preference
    pub theme: Theme,

    /// Show ML overlay by default
    pub show_overlays: bool,

    /// Default zoom level for previews
    pub default_zoom: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Theme {
    #[default]
    Dark,
    Light,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingDefaults {
    /// Default output size
    pub default_output_size: u32,

    /// Default edge thickness
    pub default_edge_thickness: u32,
}

impl Default for ProcessingDefaults {
    fn default() -> Self {
        Self {
            default_output_size: 32,
            default_edge_thickness: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryConfig {
    /// Last opened image directory
    pub last_image_dir: Option<PathBuf>,

    /// Last export directory
    pub last_export_dir: Option<PathBuf>,

    /// Custom model directory
    pub custom_model_dir: Option<PathBuf>,
}

impl Default for DirectoryConfig {
    fn default() -> Self {
        Self {
            last_image_dir: None,
            last_export_dir: None,
            custom_model_dir: None,
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            show_overlays: true,
            default_zoom: 1.0,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_quality: ModelQuality::default(),
            sequential_processing: false,
            ui: UiConfig::default(),
            processing: ProcessingDefaults::default(),
            directories: DirectoryConfig::default(),
            models: ModelConfig::default(),
        }
    }
}

impl AppConfig {
    /// Get the config file path
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?;
        Ok(config_dir.join("pixelforge").join("config.json"))
    }

    /// Load configuration from disk
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: AppConfig = serde_json::from_str(&content)?;
            Ok(config)
        } else {
            let config = AppConfig::default();
            let _ = config.save();
            Ok(config)
        }
    }

    /// Save configuration to disk
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;

        Ok(())
    }

    /// Get the models directory
    pub fn models_dir() -> Result<PathBuf> {
        let data_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
        Ok(data_dir.join("pixelforge").join("models"))
    }

    /// Get the presets directory
    pub fn presets_dir() -> Result<PathBuf> {
        let data_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
        Ok(data_dir.join("pixelforge").join("presets"))
    }
}

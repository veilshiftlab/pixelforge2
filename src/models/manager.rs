//! Model manager for loading and managing ML models

use crate::config::{AppConfig, ModelQuality};
use anyhow::Result;
use std::path::PathBuf;

/// Model manager
pub struct ModelManager {
    /// Currently loaded quality level
    pub loaded_quality: Option<ModelQuality>,
    
    /// Path to models directory
    models_dir: PathBuf,
    
    /// Whether models are available
    models_available: bool,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_dir();
        let models_available = models_dir.exists();
        
        Ok(Self {
            loaded_quality: None,
            models_dir,
            models_available,
        })
    }

    /// Get models directory
    fn get_models_dir() -> PathBuf {
        // Check for models alongside executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let bundled_dir = exe_dir.join("models");
                if bundled_dir.exists() {
                    return bundled_dir;
                }
            }
        }

        // Fall back to config models dir
        AppConfig::models_dir().unwrap_or_else(|_| {
            let data_dir = dirs::data_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
            data_dir.join("pixelforge").join("models")
        })
    }

    /// Set model quality
    pub fn set_quality(&mut self, quality: ModelQuality) -> Result<()> {
        // In a real implementation, this would load/unload models
        self.loaded_quality = Some(quality);
        Ok(())
    }

    /// Check if models are loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded_quality.is_some()
    }

    /// Check if models are available
    pub fn models_available(&self) -> bool {
        self.models_available
    }

    /// Get estimated VRAM usage
    pub fn estimated_vram_usage(&self) -> u64 {
        match self.loaded_quality {
            Some(ModelQuality::High) => 2 * 1024 * 1024 * 1024, // 2GB
            Some(ModelQuality::Standard) => 512 * 1024 * 1024, // 512MB
            Some(ModelQuality::Minimal) | Some(ModelQuality::Sequential) => 256 * 1024 * 1024, // 256MB
            None => 0,
        }
    }

    /// Get models directory path
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Unload all models (free memory)
    pub fn unload_all(&mut self) {
        self.loaded_quality = None;
        log::info!("Unloaded all models");
    }

    /// Check for available model files
    pub fn check_model_files(&self) -> Vec<String> {
        let mut available = Vec::new();
        
        if self.models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&self.models_dir) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(".onnx") {
                            available.push(name.to_string());
                        }
                    }
                }
            }
        }
        
        available
    }

    /// Get path to a specific model file
    pub fn get_model_path(&self, model_type: &str) -> Option<PathBuf> {
        let model_name = match model_type {
            "face_detection" => match self.loaded_quality {
                Some(ModelQuality::Minimal) => "yunet_2023.onnx",
                _ => "mediapipe_face.onnx",
            },
            "depth" => match self.loaded_quality {
                Some(ModelQuality::Minimal) => "midas_small_q8.onnx",
                Some(ModelQuality::High) | Some(ModelQuality::Sequential) => "dpt_large.onnx",
                _ => "midas_small.onnx",
            },
            "segmentation" => match self.loaded_quality {
                Some(ModelQuality::Minimal) => "segformer_b0.onnx",
                Some(ModelQuality::High) | Some(ModelQuality::Sequential) => "sam_vit_b.onnx",
                _ => "segformer_b2.onnx",
            },
            _ => return None,
        };

        let path = self.models_dir.join(model_name);
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    /// List required models for each quality level
    pub fn required_models(quality: ModelQuality) -> &'static [&'static str] {
        match quality {
            ModelQuality::Minimal => &[
                "yunet_2023.onnx",
                "midas_small_q8.onnx",
                "segformer_b0.onnx",
            ],
            ModelQuality::Standard => &[
                "mediapipe_face.onnx",
                "midas_small.onnx",
                "segformer_b2.onnx",
            ],
            ModelQuality::High => &[
                "mediapipe_face.onnx",
                "dpt_large.onnx",
                "sam_vit_b.onnx",
            ],
            ModelQuality::Sequential => &[
                "mediapipe_face.onnx",
                "dpt_large.onnx",
                "sam_vit_b.onnx",
            ],
        }
    }

    /// Download a model (placeholder - would use reqwest in production)
    pub fn download_model(&self, model_name: &str) -> Result<()> {
        log::info!("Would download model: {}", model_name);
        // In production, this would:
        // 1. Check if model already exists
        // 2. Download from remote URL
        // 3. Verify checksum
        // 4. Save to models directory
        Ok(())
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            loaded_quality: None,
            models_dir: std::path::PathBuf::from("models"),
            models_available: false,
        })
    }
}

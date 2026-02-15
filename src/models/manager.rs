//! Model manager for loading and managing ML models

use crate::config::{AppConfig, ModelQuality};
use crate::models::{ModelDownloader, AsyncDownloader};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;

/// Model manager
pub struct ModelManager {
    /// Currently loaded quality level
    pub loaded_quality: Option<ModelQuality>,
    
    /// Path to models directory
    models_dir: PathBuf,
    
    /// Whether models are available
    models_available: bool,

    /// Model downloader
    downloader: Option<AsyncDownloader>,

    /// Sequential processing mode (for low VRAM)
    sequential_mode: bool,

    /// Download progress
    download_progress: Arc<RwLock<DownloadProgress>>,
}

/// Download progress tracking
#[derive(Debug, Clone, Default)]
pub struct DownloadProgress {
    pub current_model: String,
    pub progress: f32,
    pub is_downloading: bool,
    pub last_error: Option<String>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_dir();
        let models_available = models_dir.exists();
        
        let downloader = AsyncDownloader::new(&models_dir).ok();

        Ok(Self {
            loaded_quality: None,
            models_dir,
            models_available,
            downloader,
            sequential_mode: false,
            download_progress: Arc::new(RwLock::new(DownloadProgress::default())),
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
        self.loaded_quality = Some(quality);
        log::info!("Model quality set to: {:?}", quality);
        Ok(())
    }

    /// Set sequential processing mode
    pub fn set_sequential_mode(&mut self, enabled: bool) {
        self.sequential_mode = enabled;
        log::info!("Sequential processing: {}", enabled);
    }

    /// Check if sequential mode is enabled
    pub fn is_sequential(&self) -> bool {
        self.sequential_mode
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
            Some(ModelQuality::High) => {
                if self.sequential_mode {
                    512 * 1024 * 1024 // 512MB when sequential
                } else {
                    2 * 1024 * 1024 * 1024 // 2GB
                }
            }
            Some(ModelQuality::Standard) => {
                if self.sequential_mode {
                    256 * 1024 * 1024 // 256MB
                } else {
                    512 * 1024 * 1024 // 512MB
                }
            }
            Some(ModelQuality::Minimal) => {
                128 * 1024 * 1024 // 128MB
            }
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
        let quality = self.loaded_quality?;
        let quality_str = quality.as_str();

        if let Some(info) = ModelDownloader::get_model_info(model_type, quality_str) {
            let path = self.models_dir.join(&info.filename);
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    /// List required models for current quality
    pub fn list_required_models(&self) -> Vec<ModelStatus> {
        let quality = self.loaded_quality.unwrap_or(ModelQuality::Minimal);
        let quality_str = quality.as_str();

        let required = ModelDownloader::get_required_models(quality_str);
        
        required
            .into_iter()
            .map(|(model_type, info)| {
                let path = self.models_dir.join(&info.filename);
                ModelStatus {
                    model_type: model_type.to_string(),
                    name: info.name,
                    filename: info.filename,
                    downloaded: path.exists(),
                    size_mb: info.size_mb,
                    description: info.description,
                }
            })
            .collect()
    }

    /// Get missing models for current quality
    pub fn get_missing_models(&self) -> Vec<ModelStatus> {
        self.list_required_models()
            .into_iter()
            .filter(|m| !m.downloaded)
            .collect()
    }

    /// Download missing models
    pub fn download_missing_models(&self) -> Result<()> {
        if let Some(ref downloader) = self.downloader {
            let quality = self.loaded_quality.unwrap_or(ModelQuality::Minimal);
            
            // Set up progress callback
            let progress = self.download_progress.clone();
            downloader.set_progress_callback(move |name, prog| {
                let mut p = progress.write();
                p.current_model = name.to_string();
                p.progress = prog;
                p.is_downloading = prog < 1.0;
            });

            // Download
            {
                let mut p = self.download_progress.write();
                p.is_downloading = true;
                p.last_error = None;
            }

            let result = downloader.download_all(quality.as_str());

            {
                let mut p = self.download_progress.write();
                p.is_downloading = false;
                if let Err(ref e) = result {
                    p.last_error = Some(e.to_string());
                }
            }

            result?;
        }

        Ok(())
    }
    
    /// Set HuggingFace token for downloads
    pub fn set_huggingface_token(&self, token: String) {
        if let Some(ref downloader) = self.downloader {
            downloader.set_huggingface_token(token);
        }
    }

    /// Download a specific model
    pub fn download_model(&self, model_type: &str) -> Result<PathBuf> {
        if let Some(ref downloader) = self.downloader {
            let quality = self.loaded_quality.unwrap_or(ModelQuality::Minimal);
            return downloader.download_model(model_type, quality.as_str());
        }

        Err(anyhow::anyhow!("Downloader not available"))
    }

    /// Get download progress
    pub fn get_download_progress(&self) -> DownloadProgress {
        self.download_progress.read().clone()
    }

    /// Get total download size for a quality level
    pub fn get_download_size(quality: ModelQuality) -> f64 {
        ModelDownloader::get_total_download_size(quality.as_str())
    }

    /// Check if all required models are downloaded
    pub fn all_models_downloaded(&self) -> bool {
        self.get_missing_models().is_empty()
    }
}

/// Status of a model
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub model_type: String,
    pub name: String,
    pub filename: String,
    pub downloaded: bool,
    pub size_mb: f64,
    pub description: String,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            loaded_quality: None,
            models_dir: std::path::PathBuf::from("models"),
            models_available: false,
            downloader: None,
            sequential_mode: false,
            download_progress: Arc::new(RwLock::new(DownloadProgress::default())),
        })
    }
}

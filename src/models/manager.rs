//! Model manager for loading and managing ML models

use crate::models::{ModelDownloader, AsyncDownloader, MODELS};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;

/// Model manager
pub struct ModelManager {
    /// Path to models directory
    models_dir: PathBuf,

    /// Model downloader
    downloader: Option<AsyncDownloader>,

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
        let downloader = AsyncDownloader::new(&models_dir).ok();

        Ok(Self {
            models_dir,
            downloader,
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

        // Fall back to app data directory
        let data_dir = dirs::data_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        data_dir.join("pixelforge").join("models")
    }

    /// Get models directory path
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Get path to a specific model file
    pub fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        let info = ModelDownloader::get_model_info(model_id)?;
        let path = self.models_dir.join(info.filename);
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    /// List all models with their status
    pub fn list_models(&self) -> Vec<ModelStatus> {
        MODELS.iter()
            .map(|info| {
                let path = self.models_dir.join(info.filename);
                ModelStatus {
                    id: info.id,
                    name: info.name,
                    filename: info.filename,
                    downloaded: path.exists(),
                    size_mb: info.size_mb,
                    description: info.description,
                }
            })
            .collect()
    }

    /// Get missing models
    pub fn get_missing_models(&self) -> Vec<ModelStatus> {
        self.list_models()
            .into_iter()
            .filter(|m| !m.downloaded)
            .collect()
    }

    /// Check if all models are downloaded
    pub fn all_models_downloaded(&self) -> bool {
        self.get_missing_models().is_empty()
    }

    /// Download a specific model
    pub fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        if let Some(ref downloader) = self.downloader {
            let progress = self.download_progress.clone();
            downloader.set_progress_callback(move |name, prog| {
                let mut p = progress.write();
                p.current_model = name.to_string();
                p.progress = prog;
                p.is_downloading = prog < 1.0;
            });

            {
                let mut p = self.download_progress.write();
                p.is_downloading = true;
                p.last_error = None;
            }

            let result = downloader.download_model(model_id);

            {
                let mut p = self.download_progress.write();
                p.is_downloading = false;
                if let Err(ref e) = result {
                    p.last_error = Some(e.to_string());
                }
            }

            result
        } else {
            Err(anyhow::anyhow!("Downloader not available"))
        }
    }

    /// Download all missing models
    pub fn download_all_missing(&self) -> Result<Vec<PathBuf>> {
        if let Some(ref downloader) = self.downloader {
            let progress = self.download_progress.clone();
            downloader.set_progress_callback(move |name, prog| {
                let mut p = progress.write();
                p.current_model = name.to_string();
                p.progress = prog;
                p.is_downloading = prog < 1.0;
            });

            {
                let mut p = self.download_progress.write();
                p.is_downloading = true;
                p.last_error = None;
            }

            let result = downloader.download_all_missing();

            {
                let mut p = self.download_progress.write();
                p.is_downloading = false;
                if let Err(ref e) = result {
                    p.last_error = Some(e.to_string());
                }
            }

            result
        } else {
            Err(anyhow::anyhow!("Downloader not available"))
        }
    }

    /// Set HuggingFace token for downloads
    pub fn set_huggingface_token(&self, token: String) {
        if let Some(ref downloader) = self.downloader {
            downloader.set_huggingface_token(token);
        }
    }

    /// Get download progress
    pub fn get_download_progress(&self) -> DownloadProgress {
        self.download_progress.read().clone()
    }

    /// Get total size of all models
    pub fn get_total_size() -> f64 {
        ModelDownloader::get_total_size()
    }

    /// Get estimated VRAM usage for running all models sequentially
    pub fn estimated_vram_usage() -> u64 {
        // Peak VRAM during depth estimation (largest model)
        // Depth-Anything V2 Base: ~800MB with intermediate tensors
        800 * 1024 * 1024
    }
}

/// Status of a model
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub downloaded: bool,
    pub size_mb: f64,
    pub description: &'static str,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            models_dir: std::path::PathBuf::from("models"),
            downloader: None,
            download_progress: Arc::new(RwLock::new(DownloadProgress::default())),
        })
    }
}

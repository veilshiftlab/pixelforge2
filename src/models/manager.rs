//! Model Manager Module

use crate::models::{ModelDownloader, AsyncDownloader};
use anyhow::Result;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;

/// High-level model management for PixelForge.
pub struct ModelManager {
    models_dir: PathBuf,
    downloader: Option<AsyncDownloader>,
    download_progress: Arc<RwLock<DownloadProgress>>,
}

#[derive(Debug, Clone, Default)]
pub struct DownloadProgress {
    pub current_model: String,
    pub progress: f32,
    pub is_downloading: bool,
    pub last_error: Option<String>,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_dir();
        let downloader = AsyncDownloader::new(&models_dir).ok();

        Ok(Self {
            models_dir,
            downloader,
            download_progress: Arc::new(RwLock::new(DownloadProgress::default())),
        })
    }

    fn get_models_dir() -> PathBuf {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let bundled_dir = exe_dir.join("models");
                if bundled_dir.exists() {
                    return bundled_dir;
                }
            }
        }
        let data_dir = dirs::data_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        data_dir.join("pixelforge").join("models")
    }

    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    pub fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        let info = ModelDownloader::get_model_info(model_id)?;
        let path = self.models_dir.join(&info.filename);
        if path.exists() { Some(path) } else { None }
    }

    pub fn list_models(&self) -> Vec<ModelStatus> {
        ModelDownloader::get_all_models()
            .into_iter()
            .map(|info| {
                let path = self.models_dir.join(&info.filename);
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

    pub fn get_missing_models(&self) -> Vec<ModelStatus> {
        self.list_models().into_iter().filter(|m| !m.downloaded).collect()
    }

    pub fn all_models_downloaded(&self) -> bool {
        self.get_missing_models().is_empty()
    }

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

    pub fn set_huggingface_token(&self, token: String) {
        if let Some(ref downloader) = self.downloader {
            downloader.set_huggingface_token(token);
        }
    }

    pub fn get_download_progress(&self) -> DownloadProgress {
        self.download_progress.read().clone()
    }

    pub fn get_total_size() -> f64 {
        ModelDownloader::get_total_size()
    }

    pub fn estimated_vram_usage(&self) -> u64 {
        400 * 1024 * 1024
    }

    /// Check which model files exist and return their status
    pub fn check_model_files(&self) -> Vec<(&'static str, bool)> {
        self.list_models()
            .into_iter()
            .map(|m| (m.id, m.downloaded))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub id: &'static str,
    pub name: String,
    pub filename: String,
    pub downloaded: bool,
    pub size_mb: f64,
    pub description: String,
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

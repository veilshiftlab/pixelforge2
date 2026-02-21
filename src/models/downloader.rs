//! Model downloader with progress reporting
//!
//! Downloads the 3 fixed models for PixelForge:
//! - YOLOv8n-Face: Face detection with 5 landmarks (6MB)
//! - Depth-Anything V2 Base: Depth estimation (370MB)  
//! - BiSeNet: Face parsing/segmentation (48MB)

use anyhow::Result;
use reqwest::blocking::Client;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Fixed model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub size_mb: f64,
    pub description: &'static str,
}

/// The 3 fixed models used by PixelForge
pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "yolov8n-face",
        name: "YOLOv8n-Face",
        filename: "yolov8n-face.onnx",
        url: "https://huggingface.co/onnx-community/yolov8n-face/resolve/main/onnx/model.onnx",
        size_mb: 6.0,
        description: "Face detection with 5 landmarks",
    },
    ModelInfo {
        id: "depth-anything-v2",
        name: "Depth-Anything V2 Base",
        filename: "depth-anything-v2-base.onnx",
        url: "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.onnx",
        size_mb: 370.0,
        description: "Depth estimation for depth-to-flat conversion",
    },
    ModelInfo {
        id: "bisenet",
        name: "BiSeNet Face Parsing",
        filename: "bisenet.onnx",
        url: "https://huggingface.co/yakhyo/face-parsing/resolve/main/face_parsing.onnx",
        size_mb: 48.0,
        description: "Face parsing for region-aware processing",
    },
];

/// Download progress callback
pub type ProgressCallback = Box<dyn Fn(&str, f32) + Send + Sync>;

/// Model downloader
pub struct ModelDownloader {
    client: Client,
    models_dir: PathBuf,
    progress_callback: Option<Arc<Mutex<ProgressCallback>>>,
    huggingface_token: Option<String>,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub fn new(models_dir: &Path) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(30))
            .user_agent("PixelForge/0.1.0")
            .build()?;

        Ok(Self {
            client,
            models_dir: models_dir.to_path_buf(),
            progress_callback: None,
            huggingface_token: None,
        })
    }
    
    /// Set HuggingFace API token
    pub fn set_huggingface_token(&mut self, token: String) {
        self.huggingface_token = Some(token);
    }

    /// Set progress callback
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str, f32) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(Mutex::new(Box::new(callback))));
    }

    /// Report progress
    fn report_progress(&self, model_name: &str, progress: f32) {
        if let Some(cb) = &self.progress_callback {
            if let Ok(guard) = cb.lock() {
                guard(model_name, progress);
            }
        }
    }

    /// Get model info by ID
    pub fn get_model_info(model_id: &str) -> Option<&'static ModelInfo> {
        MODELS.iter().find(|m| m.id == model_id)
    }

    /// Get all models
    pub fn get_all_models() -> &'static [ModelInfo] {
        MODELS
    }

    /// Check if a model is already downloaded
    pub fn is_model_downloaded(&self, model_id: &str) -> bool {
        if let Some(info) = Self::get_model_info(model_id) {
            self.models_dir.join(info.filename).exists()
        } else {
            false
        }
    }

    /// Download a single model
    pub fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        let info = Self::get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_id))?;

        let output_path = self.models_dir.join(info.filename);

        // Check if already exists
        if output_path.exists() {
            log::info!("Model {} already exists, skipping download", info.filename);
            return Ok(output_path);
        }

        // Ensure models directory exists
        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)?;
        }

        log::info!("Downloading {} from {}", info.name, info.url);
        self.report_progress(info.name, 0.0);

        // Build request with optional Bearer token for HuggingFace URLs
        let mut request = self.client.get(info.url);
        
        if let Some(ref token) = self.huggingface_token {
            if !token.is_empty() {
                request = request.bearer_auth(token);
                log::info!("Using HuggingFace Bearer token for authentication");
            }
        }
        
        let response = request.send()?;
        
        let status = response.status();
        if !status.is_success() {
            let error_msg = format!("HTTP {} - {} for URL: {}", status.as_u16(), status.canonical_reason().unwrap_or("Unknown"), info.url);
            log::error!("Download failed: {}", error_msg);
            return Err(anyhow::anyhow!("{}", error_msg));
        }
        
        let total_size = response.content_length().unwrap_or(0) as usize;

        // Read entire response body
        let bytes = response.bytes()?;
        let downloaded = bytes.len();
        
        if downloaded == 0 {
            return Err(anyhow::anyhow!("Empty response from {}", info.url));
        }

        if total_size > 0 {
            let progress = downloaded as f32 / total_size as f32;
            self.report_progress(info.name, progress);
        } else {
            self.report_progress(info.name, 0.5);
        }

        std::fs::write(&output_path, &bytes)?;
        log::info!("Saved model to {} ({} bytes)", output_path.display(), downloaded);
        self.report_progress(info.name, 1.0);

        Ok(output_path)
    }

    /// Download all missing models
    pub fn download_all_missing(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let mut errors = Vec::new();

        for info in MODELS {
            if !self.is_model_downloaded(info.id) {
                match self.download_model(info.id) {
                    Ok(path) => paths.push(path),
                    Err(e) => {
                        let error_msg = format!("{}: {}", info.name, e);
                        log::error!("Failed to download {}", error_msg);
                        errors.push(error_msg);
                    }
                }
            }
        }

        if !errors.is_empty() && paths.is_empty() {
            return Err(anyhow::anyhow!("All downloads failed:\n{}", errors.join("\n")));
        } else if !errors.is_empty() {
            log::warn!("Some downloads failed: {}", errors.join(", "));
        }

        Ok(paths)
    }

    /// Get total size of all models
    pub fn get_total_size() -> f64 {
        MODELS.iter().map(|m| m.size_mb).sum()
    }

    /// List models with their download status
    pub fn list_models_status(&self) -> Vec<(&'static ModelInfo, bool)> {
        MODELS.iter()
            .map(|info| {
                let downloaded = self.models_dir.join(info.filename).exists();
                (*info, downloaded)
            })
            .collect()
    }
}

/// Async model downloader for use in UI
pub struct AsyncDownloader {
    downloader: Arc<Mutex<ModelDownloader>>,
}

impl AsyncDownloader {
    pub fn new(models_dir: &Path) -> Result<Self> {
        Ok(Self {
            downloader: Arc::new(Mutex::new(ModelDownloader::new(models_dir)?)),
        })
    }

    pub fn set_progress_callback<F>(&self, callback: F)
    where
        F: Fn(&str, f32) + Send + Sync + 'static,
    {
        if let Ok(mut d) = self.downloader.lock() {
            d.set_progress_callback(callback);
        }
    }
    
    pub fn set_huggingface_token(&self, token: String) {
        if let Ok(mut d) = self.downloader.lock() {
            d.set_huggingface_token(token);
        }
    }

    pub fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        if let Ok(d) = self.downloader.lock() {
            d.download_model(model_id)
        } else {
            Err(anyhow::anyhow!("Downloader is locked"))
        }
    }

    pub fn download_all_missing(&self) -> Result<Vec<PathBuf>> {
        if let Ok(d) = self.downloader.lock() {
            d.download_all_missing()
        } else {
            Err(anyhow::anyhow!("Downloader is locked"))
        }
    }

    pub fn is_model_downloaded(&self, model_id: &str) -> bool {
        if let Ok(d) = self.downloader.lock() {
            d.is_model_downloaded(model_id)
        } else {
            false
        }
    }

    pub fn list_models_status(&self) -> Vec<(&'static ModelInfo, bool)> {
        if let Ok(d) = self.downloader.lock() {
            d.list_models_status()
        } else {
            Vec::new()
        }
    }
}

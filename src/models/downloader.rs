//! Model downloader with progress reporting

use anyhow::Result;
use reqwest::blocking::Client;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Model download configuration
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub filename: String,
    pub url: String,
    pub size_mb: f64,
    pub description: String,
}

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
            .timeout(Duration::from_secs(300))
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

    /// Get model info for a specific model type and quality
    pub fn get_model_info(model_type: &str, quality: &str) -> Option<ModelInfo> {
        match (model_type, quality) {
            // Face Detection - SCRFD from HuggingFace (requires token for some repos)
            ("face_detection", "minimal") => Some(ModelInfo {
                name: "SCRFD Face Detector (500M)".to_string(),
                filename: "scrfd_500m.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/face-detection/resolve/main/scrfd_500m_bnkps.onnx".to_string(),
                size_mb: 2.5,
                description: "Lightweight face detector".to_string(),
            }),
            ("face_detection", "standard") => Some(ModelInfo {
                name: "SCRFD Face Detector (2.5G)".to_string(),
                filename: "scrfd_2.5g.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/face-detection/resolve/main/scrfd_2.5g_bnkps.onnx".to_string(),
                size_mb: 5.2,
                description: "Standard face detector".to_string(),
            }),
            ("face_detection", "high") => Some(ModelInfo {
                name: "SCRFD Face Detector (10G)".to_string(),
                filename: "scrfd_10g.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/face-detection/resolve/main/scrfd_10g_bnkps.onnx".to_string(),
                size_mb: 18.0,
                description: "High accuracy face detector".to_string(),
            }),

            // Depth Estimation - MiDaS from HuggingFace
            ("depth", "minimal") => Some(ModelInfo {
                name: "MiDaS Small".to_string(),
                filename: "midas_small.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/MiDaS/resolve/main/midas_v21_small_256.onnx".to_string(),
                size_mb: 21.0,
                description: "Fast depth estimation".to_string(),
            }),
            ("depth", "standard") => Some(ModelInfo {
                name: "MiDaS Hybrid".to_string(),
                filename: "midas_hybrid.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/MiDaS/resolve/main/dpt_hybrid_384.onnx".to_string(),
                size_mb: 130.0,
                description: "Standard depth estimation".to_string(),
            }),
            ("depth", "high") => Some(ModelInfo {
                name: "DPT Large".to_string(),
                filename: "dpt_large.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/MiDaS/resolve/main/dpt_large_384.onnx".to_string(),
                size_mb: 340.0,
                description: "High quality depth estimation".to_string(),
            }),

            // Segmentation - MODNet for portrait matting
            ("segmentation", "minimal") => Some(ModelInfo {
                name: "MODNet Portrait".to_string(),
                filename: "modnet.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/MODNet/resolve/main/modnet_photographic_portrait_matting.onnx".to_string(),
                size_mb: 25.0,
                description: "Portrait segmentation".to_string(),
            }),
            ("segmentation", "standard") => Some(ModelInfo {
                name: "MODNet Portrait".to_string(),
                filename: "modnet.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/MODNet/resolve/main/modnet_photographic_portrait_matting.onnx".to_string(),
                size_mb: 25.0,
                description: "Portrait segmentation".to_string(),
            }),
            ("segmentation", "high") => Some(ModelInfo {
                name: "MODNet Portrait".to_string(),
                filename: "modnet.onnx".to_string(),
                url: "https://huggingface.co/pinto0309/MODNet/resolve/main/modnet_photographic_portrait_matting.onnx".to_string(),
                size_mb: 25.0,
                description: "Portrait segmentation".to_string(),
            }),

            _ => None,
        }
    }

    /// Check if a model is already downloaded
    pub fn is_model_downloaded(&self, model_type: &str, quality: &str) -> bool {
        if let Some(info) = Self::get_model_info(model_type, quality) {
            self.models_dir.join(&info.filename).exists()
        } else {
            false
        }
    }

    /// Get all required models for a quality level
    pub fn get_required_models(quality: &str) -> Vec<(&'static str, ModelInfo)> {
        let mut models = Vec::new();

        for model_type in &["face_detection", "depth", "segmentation"] {
            if let Some(info) = Self::get_model_info(model_type, quality) {
                models.push((*model_type, info));
            }
        }

        models
    }

    /// Download a single model
    pub fn download_model(&self, model_type: &str, quality: &str) -> Result<PathBuf> {
        let info = Self::get_model_info(model_type, quality)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {} / {}", model_type, quality))?;

        let output_path = self.models_dir.join(&info.filename);

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
        self.report_progress(&info.name, 0.0);

        // Build request with optional Bearer token for HuggingFace URLs
        let mut request = self.client.get(&info.url);
        
        // Check if URL is a HuggingFace URL and add auth if token is available
        if info.url.contains("huggingface.co") {
            if let Some(ref token) = self.huggingface_token {
                if !token.is_empty() {
                    request = request.bearer_auth(token);
                    log::info!("Using HuggingFace Bearer token for authentication");
                } else {
                    log::warn!("HuggingFace URL but no token provided - may fail with 401");
                }
            } else {
                log::warn!("HuggingFace URL but no token set - may fail with 401");
            }
        }
        
        // Download with status check
        let response = request.send()?;
        
        // Check HTTP status
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
        
        // Check if we got any data
        if downloaded == 0 {
            return Err(anyhow::anyhow!("Empty response from {}", info.url));
        }

        // Report progress based on expected size
        if total_size > 0 {
            let progress = downloaded as f32 / total_size as f32;
            self.report_progress(&info.name, progress);
        } else {
            // If no content-length, report based on downloaded bytes
            self.report_progress(&info.name, 0.5);
        }

        // Save to file
        std::fs::write(&output_path, &bytes)?;
        log::info!("Saved model to {} ({} bytes)", output_path.display(), downloaded);
        self.report_progress(&info.name, 1.0);

        Ok(output_path)
    }

    /// Download all models for a quality level
    pub fn download_all_for_quality(&self, quality: &str) -> Result<Vec<PathBuf>> {
        let models = Self::get_required_models(quality);
        let mut paths = Vec::new();
        let mut errors = Vec::new();

        for (model_type, info) in &models {
            match self.download_model(model_type, quality) {
                Ok(path) => paths.push(path),
                Err(e) => {
                    let error_msg = format!("{}: {}", info.name, e);
                    log::error!("Failed to download {}", error_msg);
                    errors.push(error_msg);
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

    /// Get download size for all models of a quality level
    pub fn get_total_download_size(quality: &str) -> f64 {
        Self::get_required_models(quality)
            .iter()
            .map(|(_, info)| info.size_mb)
            .sum()
    }

    /// List available models with their status
    pub fn list_models(&self, quality: &str) -> Vec<(String, String, bool, f64)> {
        Self::get_required_models(quality)
            .into_iter()
            .map(|(_, info)| {
                let downloaded = self.models_dir.join(&info.filename).exists();
                (info.name, info.description, downloaded, info.size_mb)
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

    pub fn download_model(&self, model_type: &str, quality: &str) -> Result<PathBuf> {
        if let Ok(d) = self.downloader.lock() {
            d.download_model(model_type, quality)
        } else {
            Err(anyhow::anyhow!("Downloader is locked"))
        }
    }

    pub fn download_all(&self, quality: &str) -> Result<Vec<PathBuf>> {
        if let Ok(d) = self.downloader.lock() {
            d.download_all_for_quality(quality)
        } else {
            Err(anyhow::anyhow!("Downloader is locked"))
        }
    }

    pub fn list_models(&self, quality: &str) -> Vec<(String, String, bool, f64)> {
        if let Ok(d) = self.downloader.lock() {
            d.list_models(quality)
        } else {
            Vec::new()
        }
    }

    pub fn get_total_size(quality: &str) -> f64 {
        ModelDownloader::get_total_download_size(quality)
    }
}

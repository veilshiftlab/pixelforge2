//! Model Downloader Module
//!
//! Handles downloading ONNX models defined in models.toml.

use anyhow::Result;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// =============================================================================
// TOML Configuration Structures
// =============================================================================

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelsConfig {
    pub models: HashMap<String, ModelDefinition>,
    pub download: Option<DownloadSettings>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    pub filename: String,
    pub url: String,
    pub size_mb: f64,
    pub description: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadSettings {
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_retries")]
    pub retries: u32,
    #[serde(default = "default_verify")]
    pub verify_exists: bool,
}

fn default_max_concurrent() -> usize { 2 }
fn default_timeout() -> u64 { 300 }
fn default_retries() -> u32 { 3 }
fn default_verify() -> bool { true }

impl Default for DownloadSettings {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent(),
            timeout: default_timeout(),
            retries: default_retries(),
            verify_exists: default_verify(),
        }
    }
}

// =============================================================================
// Model Info
// =============================================================================

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub url: String,
    pub size_mb: f64,
    pub description: String,
}

// =============================================================================
// Progress Callback
// =============================================================================

pub type ProgressCallback = Box<dyn Fn(&str, f32) + Send + Sync>;

// =============================================================================
// Model Downloader
// =============================================================================

pub struct ModelDownloader {
    client: Client,
    models_dir: PathBuf,
    progress_callback: Option<Arc<Mutex<ProgressCallback>>>,
    huggingface_token: Option<String>,
    config: ModelsConfig,
}

impl ModelDownloader {
    pub fn new(models_dir: &Path) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(30))
            .user_agent("PixelForge/0.1.0")
            .build()?;

        let config = Self::load_config()?;

        Ok(Self {
            client,
            models_dir: models_dir.to_path_buf(),
            progress_callback: None,
            huggingface_token: None,
            config,
        })
    }

    fn load_config() -> Result<ModelsConfig> {
        let toml_content = include_str!("models.toml");
        let config: ModelsConfig = toml::from_str(toml_content)?;
        Ok(config)
    }

    pub fn set_huggingface_token(&mut self, token: String) {
        self.huggingface_token = Some(token);
    }

    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str, f32) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(Mutex::new(Box::new(callback))));
    }

    fn report_progress(&self, model_name: &str, progress: f32) {
        if let Some(cb) = &self.progress_callback {
            if let Ok(guard) = cb.lock() {
                guard(model_name, progress);
            }
        }
    }

    pub fn get_model_info(model_id: &str) -> Option<ModelInfo> {
        let config = Self::load_config().ok()?;
        let def = config.models.get(model_id)?;

        Some(ModelInfo {
            id: model_id.to_string(),
            name: def.name.clone(),
            filename: def.filename.clone(),
            url: def.url.clone(),
            size_mb: def.size_mb,
            description: def.description.clone(),
        })
    }

    pub fn get_all_models() -> Vec<ModelInfo> {
        let config = match Self::load_config() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut models: Vec<ModelInfo> = config
            .models
            .iter()
            .filter(|(_, def)| def.enabled)
            .map(|(id, def)| ModelInfo {
                id: id.clone(),
                name: def.name.clone(),
                filename: def.filename.clone(),
                url: def.url.clone(),
                size_mb: def.size_mb,
                description: def.description.clone(),
            })
            .collect();

        // Sort by ID for stable UI ordering — HashMap iteration is random per
        // call, which caused the Models panel to reorganize every frame.
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    pub fn is_model_downloaded(&self, model_id: &str) -> bool {
        if let Some(def) = self.config.models.get(model_id) {
            self.models_dir.join(&def.filename).exists()
        } else {
            false
        }
    }

    pub fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        let def = self
            .config
            .models
            .get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_id))?;

        let output_path = self.models_dir.join(&def.filename);

        if output_path.exists() {
            log::info!("Model {} already exists, skipping download", def.filename);
            return Ok(output_path);
        }

        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)?;
        }

        log::info!("Downloading {} from {}", def.name, def.url);
        self.report_progress(&def.name, 0.0);

        let mut request = self.client.get(&def.url);

        if let Some(ref token) = self.huggingface_token {
            if !token.is_empty() {
                request = request.bearer_auth(token);
            }
        }

        let response = request.send()?;

        let status = response.status();
        if !status.is_success() {
            let error_msg = format!(
                "HTTP {} - {} for URL: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown"),
                def.url
            );
            log::error!("Download failed: {}", error_msg);
            return Err(anyhow::anyhow!("{}", error_msg));
        }

        let total_size = response.content_length().unwrap_or(0) as usize;
        let bytes = response.bytes()?;
        let downloaded = bytes.len();

        if downloaded == 0 {
            return Err(anyhow::anyhow!("Empty response from {}", def.url));
        }

        if total_size > 0 {
            let progress = downloaded as f32 / total_size as f32;
            self.report_progress(&def.name, progress);
        } else {
            self.report_progress(&def.name, 0.5);
        }

        std::fs::write(&output_path, &bytes)?;
        log::info!("Saved model to {} ({} bytes)", output_path.display(), downloaded);
        self.report_progress(&def.name, 1.0);

        Ok(output_path)
    }

    pub fn download_all_missing(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let mut errors = Vec::new();

        for (id, def) in &self.config.models {
            if !def.enabled {
                continue;
            }
            if !self.is_model_downloaded(id) {
                match self.download_model(id) {
                    Ok(path) => paths.push(path),
                    Err(e) => {
                        let error_msg = format!("{}: {}", def.name, e);
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

    pub fn get_total_size() -> f64 {
        let config = match Self::load_config() {
            Ok(c) => c,
            Err(_) => return 0.0,
        };
        config.models.values().filter(|def| def.enabled).map(|def| def.size_mb).sum()
    }

    pub fn list_models_status(&self) -> Vec<(String, bool)> {
        let mut models: Vec<(String, bool)> = self.config
            .models
            .iter()
            .filter(|(_, def)| def.enabled)
            .map(|(id, def)| {
                let downloaded = self.models_dir.join(&def.filename).exists();
                (id.clone(), downloaded)
            })
            .collect();

        // Sort by ID for stable ordering (HashMap iteration is random).
        models.sort_by(|a, b| a.0.cmp(&b.0));
        models
    }
}

// =============================================================================
// Async Downloader Wrapper
// =============================================================================

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

    pub fn list_models_status(&self) -> Vec<(String, bool)> {
        if let Ok(d) = self.downloader.lock() {
            d.list_models_status()
        } else {
            Vec::new()
        }
    }
}

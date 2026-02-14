//! Model download functionality

use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender, Receiver};

/// Model download information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub url: String,
    pub size_mb: u64,
    pub model_type: ModelType,
}

/// Types of models
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    FaceDetection,
    DepthEstimation,
    Segmentation,
}

/// Download progress
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub model: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub complete: bool,
    pub error: Option<String>,
}

/// Model downloader
pub struct ModelDownloader {
    models_dir: PathBuf,
    progress_receiver: Option<Receiver<DownloadProgress>>,
}

impl ModelDownloader {
    /// Create a new downloader
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir,
            progress_receiver: None,
        }
    }
    
    /// Get available models for download
    pub fn get_available_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                name: "MediaPipe Face Mesh".to_string(),
                description: "High-quality face landmark detection (468 points)".to_string(),
                url: "https://github.com/PixelForge/models/raw/main/mediapipe_face.onnx".to_string(),
                size_mb: 12,
                model_type: ModelType::FaceDetection,
            },
            ModelInfo {
                name: "DPT-Large".to_string(),
                description: "Superior depth estimation for better 3D understanding".to_string(),
                url: "https://github.com/PixelForge/models/raw/main/dpt_large.onnx".to_string(),
                size_mb: 340,
                model_type: ModelType::DepthEstimation,
            },
            ModelInfo {
                name: "SAM-ViT-B".to_string(),
                description: "Superior segmentation for precise region detection".to_string(),
                url: "https://github.com/PixelForge/models/raw/main/sam_vit_b.onnx".to_string(),
                size_mb: 375,
                model_type: ModelType::Segmentation,
            },
        ]
    }
    
    /// Check if a model is downloaded
    pub fn is_downloaded(&self, model: &ModelInfo) -> bool {
        let filename = model.url.rsplit('/').next().unwrap_or("");
        self.models_dir.join(filename).exists()
    }
    
    /// Start downloading a model
    pub fn download(&mut self, model: &ModelInfo) -> Result<()> {
        let (sender, receiver) = channel();
        self.progress_receiver = Some(receiver);
        
        let url = model.url.clone();
        let filename = url.rsplit('/').next().unwrap_or("model.onnx").to_string();
        let dest_path = self.models_dir.join(&filename);
        let model_name = model.name.clone();
        let total_bytes = model.size_mb * 1024 * 1024;
        
        // Create models directory if needed
        std::fs::create_dir_all(&self.models_dir)?;
        
        // Spawn download thread
        std::thread::spawn(move || {
            let result = download_file(&url, &dest_path, sender.clone(), total_bytes, model_name);
            
            if let Err(e) = result {
                let _ = sender.send(DownloadProgress {
                    model: String::new(),
                    bytes_downloaded: 0,
                    total_bytes: 0,
                    complete: false,
                    error: Some(e.to_string()),
                });
            }
        });
        
        Ok(())
    }
    
    /// Get download progress
    pub fn get_progress(&self) -> Option<DownloadProgress> {
        self.progress_receiver.as_ref()?.try_recv().ok()
    }
    
    /// Cancel download (if in progress)
    pub fn cancel(&mut self) {
        self.progress_receiver = None;
    }
}

/// Download a file with progress reporting
fn download_file(
    url: &str,
    dest: &PathBuf,
    progress: Sender<DownloadProgress>,
    expected_size: u64,
    model_name: String,
) -> Result<()> {
    use std::fs::File;
    use std::io::{Read, Write};
    
    // Send initial progress
    progress.send(DownloadProgress {
        model: model_name.clone(),
        bytes_downloaded: 0,
        total_bytes: expected_size,
        complete: false,
        error: None,
    })?;
    
    // In a real implementation, use reqwest for HTTP download
    // This is a placeholder that would use blocking reqwest
    
    let response = reqwest::blocking::get(url)?;
    let total_size = response.content_length().unwrap_or(expected_size);
    
    let mut file = File::create(dest)?;
    let mut downloaded = 0u64;
    
    for chunk in response.bytes()? {
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        
        // Report progress every 100KB
        if downloaded % (100 * 1024) < chunk.len() as u64 {
            progress.send(DownloadProgress {
                model: model_name.clone(),
                bytes_downloaded: downloaded,
                total_bytes: total_size,
                complete: false,
                error: None,
            })?;
        }
    }
    
    // Send completion
    progress.send(DownloadProgress {
        model: model_name,
        bytes_downloaded: downloaded,
        total_bytes: total_size,
        complete: true,
        error: None,
    })?;
    
    Ok(())
}

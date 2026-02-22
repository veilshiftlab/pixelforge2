//! ML Model wrappers using unified SessionManager
//!
//! Provides simplified model interfaces that use cached sessions.
//! This module maintains backward compatibility while providing
//! new implementations with dynamic input support.

// Re-export new model implementations
mod models;
pub use models::{YOLOv8FaceDetector, DepthAnythingV2, BiSeNetSegmenter};

use super::session::{ModelType, SessionManager};
use super::types::*;
use super::config::*;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use std::path::PathBuf;
use std::sync::Arc;

// =============================================================================
// Legacy Face Detection (Backward Compatible)
// =============================================================================

/// Face detector using ONNX model with session caching
/// Legacy interface - wraps YOLOv8FaceDetector
pub struct FaceDetector {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
}

impl FaceDetector {
    /// Create a new face detector
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self {
            session_manager,
            model_path: PathBuf::new(),
        }
    }

    /// Set the model path
    pub fn with_model(mut self, model_path: PathBuf) -> Self {
        self.model_path = model_path;
        self
    }

    /// Detect faces in an image (legacy interface)
    pub fn detect(&self, image: &DynamicImage) -> Result<FaceDetectionResult> {
        if self.model_path.as_os_str().is_empty() {
            return Ok(FaceDetectionResult::default());
        }

        // Use new YOLOv8FaceDetector with default config
        let detector = YOLOv8FaceDetector::new(
            Arc::clone(&self.session_manager),
            self.model_path.clone(),
        );

        let config = FaceDetectionConfig::default();
        let output = detector.detect(image, &config)?;

        // Convert to legacy format
        Ok(FaceDetectionResult {
            bounds: output.bounds,
            landmarks: output.landmarks.first().map(|_| {
                // Create legacy FaceLandmarks from the 5-point output
                FaceLandmarks {
                    points: output.landmarks.clone(),
                    left_eye: None,
                    right_eye: None,
                    nose: None,
                    lips: None,
                    face_outline: Vec::new(),
                }
            }),
            confidence: output.confidence,
        })
    }
}

// =============================================================================
// Legacy Depth Estimation (Backward Compatible)
// =============================================================================

/// Depth estimator using ONNX model with session caching
/// Legacy interface - wraps DepthAnythingV2
pub struct DepthEstimator {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
}

impl DepthEstimator {
    /// Create a new depth estimator
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self {
            session_manager,
            model_path: PathBuf::new(),
        }
    }

    /// Set the model path
    pub fn with_model(mut self, model_path: PathBuf) -> Self {
        self.model_path = model_path;
        self
    }

    /// Estimate depth map for an image (legacy interface)
    pub fn estimate(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        if self.model_path.as_os_str().is_empty() {
            return create_placeholder_depth(image);
        }

        // Use new DepthAnythingV2 with default config
        let estimator = DepthAnythingV2::new(
            Arc::clone(&self.session_manager),
            self.model_path.clone(),
        );

        let config = DepthConfig::default();
        let output = estimator.estimate(image, &config)?;

        Ok(output.depth_map)
    }
}

// =============================================================================
// Legacy Segmentation (Backward Compatible)
// =============================================================================

/// Semantic segmenter using ONNX model with session caching
/// Legacy interface - wraps BiSeNetSegmenter
pub struct Segmenter {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
}

impl Segmenter {
    /// Create a new segmenter
    pub fn new(session_manager: Arc<SessionManager>) -> Self {
        Self {
            session_manager,
            model_path: PathBuf::new(),
        }
    }

    /// Set the model path
    pub fn with_model(mut self, model_path: PathBuf) -> Self {
        self.model_path = model_path;
        self
    }

    /// Segment an image into semantic regions (legacy interface)
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        if self.model_path.as_os_str().is_empty() {
            return create_placeholder_segmentation(image);
        }

        // Use new BiSeNetSegmenter with default config
        let segmenter = BiSeNetSegmenter::new(
            Arc::clone(&self.session_manager),
            self.model_path.clone(),
        );

        let config = SegmentationConfig::default();
        let output = segmenter.segment(image, &config)?;

        // Convert to legacy SegmentationResult
        Ok(SegmentationResult::from(output))
    }
}

// =============================================================================
// Helper Functions (Legacy)
// =============================================================================

/// Create placeholder depth map
fn create_placeholder_depth(image: &DynamicImage) -> Result<Vec<f32>> {
    let (width, height) = image.dimensions();
    let mut depth_map = vec![0.5f32; (width * height) as usize];

    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let max_dist = center_x.hypot(center_y).max(1.0);

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = (y * width + x) as usize;
            depth_map[idx] = 1.0 - (dist / max_dist * 0.5).min(1.0);
        }
    }

    let gray = image.to_luma8();
    for (i, depth) in depth_map.iter_mut().enumerate() {
        let y = (i / width as usize) as u32;
        let x = (i % width as usize) as u32;
        if x < width && y < height {
            let pixel = gray.get_pixel(x, y);
            let luminance = pixel[0] as f32 / 255.0;
            *depth = (*depth * 0.7 + luminance * 0.3).clamp(0.0, 1.0);
        }
    }

    Ok(depth_map)
}

/// Create placeholder segmentation
fn create_placeholder_segmentation(image: &DynamicImage) -> Result<SegmentationResult> {
    let (width, height) = image.dimensions();
    let mut regions = std::collections::HashMap::new();

    let face_y_start = (height as f32 * 0.15) as u32;
    let face_y_end = (height as f32 * 0.75) as u32;
    let face_x_start = (width as f32 * 0.25) as u32;
    let face_x_end = (width as f32 * 0.75) as u32;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as u32;

            let region = if y >= face_y_start
                && y <= face_y_end
                && x >= face_x_start
                && x <= face_x_end
            {
                let rel_y = (y - face_y_start) as f32 / (face_y_end - face_y_start).max(1) as f32;
                let rel_x = (x - face_x_start) as f32 / (face_x_end - face_x_start).max(1) as f32;

                if rel_y > 0.25 && rel_y < 0.4 {
                    if (rel_x > 0.2 && rel_x < 0.4) || (rel_x > 0.6 && rel_x < 0.8) {
                        SegmentationRegion::Eyes
                    } else {
                        SegmentationRegion::Face
                    }
                } else if rel_y > 0.4 && rel_y < 0.6 && rel_x > 0.35 && rel_x < 0.65 {
                    SegmentationRegion::Nose
                } else if rel_y > 0.65 && rel_y < 0.8 && rel_x > 0.3 && rel_x < 0.7 {
                    SegmentationRegion::Lips
                } else {
                    SegmentationRegion::Face
                }
            } else if y < face_y_start {
                SegmentationRegion::Hair
            } else {
                SegmentationRegion::Background
            };

            regions.insert(idx, region);
        }
    }

    Ok(SegmentationResult {
        regions,
        region_bounds: std::collections::HashMap::new(),
        confidence: vec![0.9; (width * height) as usize],
    })
}

//! ML Analysis orchestration

use super::{MLConfig, MLResults, FaceDetectionResult, SegmentationResult};
use super::{FaceDetector, PlaceholderDetector, DepthEstimator, PlaceholderDepthEstimator, Segmenter, PlaceholderSegmenter};
use crate::models::ModelManager;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use parking_lot::RwLock;
use std::sync::Arc;

/// ML Analysis orchestrator
pub struct MLAnalysis;

impl MLAnalysis {
    /// Run full ML analysis on an image
    pub fn analyze(
        image: &DynamicImage,
        config: &MLConfig,
        model_manager: &Arc<RwLock<ModelManager>>,
    ) -> Result<MLResults> {
        let mut results = MLResults::default();

        let (width, height) = image.dimensions();
        let _total_pixels = (width * height) as usize;

        // Run face detection first (provides bounds for other models)
        if config.face_detection_enabled {
            let detection = run_face_detection(image, model_manager)?;
            results.face_bounds = detection.bounds;
            results.landmarks = detection.landmarks;
        }

        // Run depth estimation
        if config.depth_estimation_enabled {
            results.depth_map = Some(run_depth_estimation(image, model_manager)?);
        }

        // Run segmentation
        if config.segmentation_enabled {
            results.segmentation = Some(run_segmentation(image, model_manager)?);
        }

        Ok(results)
    }
}

/// Run face detection (uses model if available, placeholder otherwise)
fn run_face_detection(image: &DynamicImage, model_manager: &Arc<RwLock<ModelManager>>) -> Result<FaceDetectionResult> {
    let manager = model_manager.read();
    
    // Try to get the face detection model path
    if let Some(model_path) = manager.get_model_path("face_detection") {
        drop(manager); // Release lock before creating detector
        let mut detector = FaceDetector::new(&model_path)?;
        detector.detect(image)
    } else {
        drop(manager);
        let detector = PlaceholderDetector;
        detector.detect(image)
    }
}

/// Run depth estimation (uses model if available, placeholder otherwise)
fn run_depth_estimation(image: &DynamicImage, model_manager: &Arc<RwLock<ModelManager>>) -> Result<Vec<f32>> {
    let manager = model_manager.read();
    
    if let Some(model_path) = manager.get_model_path("depth") {
        drop(manager);
        let mut estimator = DepthEstimator::new(&model_path)?;
        estimator.estimate(image)
    } else {
        drop(manager);
        let estimator = PlaceholderDepthEstimator;
        estimator.estimate(image)
    }
}

/// Run segmentation (uses model if available, placeholder otherwise)
fn run_segmentation(image: &DynamicImage, model_manager: &Arc<RwLock<ModelManager>>) -> Result<SegmentationResult> {
    let manager = model_manager.read();
    
    if let Some(model_path) = manager.get_model_path("segmentation") {
        drop(manager);
        let mut segmenter = Segmenter::new(&model_path)?;
        segmenter.segment(image)
    } else {
        drop(manager);
        let segmenter = PlaceholderSegmenter;
        segmenter.segment(image)
    }
}

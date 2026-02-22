//! ML Analysis Orchestration Module

use super::{MLConfig, MLResults, FaceDetectionResult, SegmentationResult};
use super::models::{
    YoloV8FaceDetector, PlaceholderFaceDetector,
    DepthAnythingEstimator, PlaceholderDepthEstimator,
    BiSeNetSegmenter, PlaceholderSegmenter,
};
use crate::models::ModelManager;
use anyhow::Result;
use image::DynamicImage;
use parking_lot::RwLock;
use std::sync::Arc;

pub struct MLAnalysis;

impl MLAnalysis {
    pub fn analyze(
        image: &DynamicImage,
        config: &MLConfig,
        model_manager: &Arc<RwLock<ModelManager>>,
    ) -> Result<MLResults> {
        let mut results = MLResults::default();

        // Step 1: Face Detection
        if config.face_detection_enabled {
            let detection = run_face_detection(image, model_manager)?;
            // FaceBounds implements Copy, so no clone needed
            results.face_bounds = detection.bounds;
            results.landmarks = detection.landmarks.clone();
            results.face = Some(super::FaceDetectionOutput {
                bounds: detection.bounds,
                landmarks: detection.landmarks.clone().map(|l| l.points).unwrap_or_default(),
                confidence: detection.confidence,
            });
        }

        // Step 2: Depth Estimation
        if config.depth_estimation_enabled {
            results.depth_map = Some(run_depth_estimation(image, model_manager)?);
        }

        // Step 3: Segmentation
        if config.segmentation_enabled {
            results.segmentation = Some(run_segmentation(image, model_manager)?);
        }

        Ok(results)
    }
}

fn run_face_detection(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<FaceDetectionResult> {
    let manager = model_manager.read();

    if let Some(model_path) = manager.get_model_path("yolov8_face") {
        drop(manager);
        log::info!("Running face detection with YOLOv8n-Face model");
        let detector = YoloV8FaceDetector::new(&model_path)?;
        detector.detect(image)
    } else {
        drop(manager);
        log::warn!("Face detection model not available, using placeholder");
        let detector = PlaceholderFaceDetector;
        detector.detect(image)
    }
}

fn run_depth_estimation(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<Vec<f32>> {
    let manager = model_manager.read();

    if let Some(model_path) = manager.get_model_path("depth_anything") {
        drop(manager);
        log::info!("Running depth estimation with Depth-Anything V2 Small model");
        let estimator = DepthAnythingEstimator::new(&model_path)?;
        estimator.estimate(image)
    } else {
        drop(manager);
        log::warn!("Depth estimation model not available, using placeholder");
        let estimator = PlaceholderDepthEstimator;
        estimator.estimate(image)
    }
}

fn run_segmentation(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<SegmentationResult> {
    let manager = model_manager.read();

    if let Some(model_path) = manager.get_model_path("bisenet") {
        drop(manager);
        log::info!("Running segmentation with BiSeNet model");
        let segmenter = BiSeNetSegmenter::new(&model_path)?;
        segmenter.segment(image)
    } else {
        drop(manager);
        log::warn!("Segmentation model not available, using placeholder");
        let segmenter = PlaceholderSegmenter;
        segmenter.segment(image)
    }
}

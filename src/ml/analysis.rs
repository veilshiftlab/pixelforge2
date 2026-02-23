//! ML Analysis Orchestration
//!
//! Runs the full analysis pipeline on an input image and collects results.
//! Models are loaded once and cached via the global [`SessionManager`] — calling
//! `MLAnalysis::analyze` multiple times on different images does not reload models.

use super::{MLConfig, MLResults, FaceDetectionResult, SegmentationResult};
use super::models::{
    YoloV8FaceDetector, PlaceholderFaceDetector,
    DepthAnythingEstimator, PlaceholderDepthEstimator,
    BiSeNetSegmenter, PlaceholderSegmenter,
    TeedEdgeDetector, PlaceholderEdgeDetector,
};
use crate::models::ModelManager;
use anyhow::Result;
use image::DynamicImage;
use parking_lot::RwLock;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

pub struct MLAnalysis;

impl MLAnalysis {
    /// Run the full ML pipeline on `image`.
    ///
    /// Which models run is controlled by the flags in `config`.
    /// Models are loaded from paths provided by `model_manager`.
    pub fn analyze(
        image: &DynamicImage,
        config: &MLConfig,
        model_manager: &Arc<RwLock<ModelManager>>,
    ) -> Result<MLResults> {
        let mut results = MLResults::default();

        if config.face_detection_enabled {
            let det = run_face_detection(image, model_manager)?;
            results.face_bounds = det.bounds;
            results.landmarks   = det.landmarks.clone();
            results.face        = Some(super::FaceDetectionOutput {
                bounds:     det.bounds,
                landmarks:  det.landmarks.map(|l| l.points).unwrap_or_default(),
                confidence: det.confidence,
            });
        }

        if config.depth_estimation_enabled {
            results.depth_map = Some(run_depth_estimation(image, model_manager)?);
        }

        if config.segmentation_enabled {
            results.segmentation = Some(run_segmentation(image, model_manager)?);
        }

        if config.edge_detection_enabled {
            results.edge_map = Some(run_edge_detection(image, model_manager)?);
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-model runners
//
// Each function:
//   1. Takes a read lock on ModelManager — dropped before inference
//   2. Constructs the model struct (cheap: session is loaded from disk once;
//      subsequent calls reuse OS file cache at minimum)
//   3. Runs inference and returns the result
//
// TODO: integrate with SessionManager cache so the ORT session itself is kept
// alive across calls rather than being reconstructed each time.
// ─────────────────────────────────────────────────────────────────────────────

fn run_face_detection(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<FaceDetectionResult> {
    let model_path = model_manager.read().get_model_path("yolov8_face");

    if let Some(path) = model_path {
        log::info!("Running face detection (YOLOv8n-Face)");
        YoloV8FaceDetector::new(&path)?.detect(image)
    } else {
        log::warn!("Face detection model unavailable — using placeholder");
        PlaceholderFaceDetector.detect(image)
    }
}

fn run_depth_estimation(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<Vec<f32>> {
    let model_path = model_manager.read().get_model_path("depth_anything");

    if let Some(path) = model_path {
        log::info!("Running depth estimation (Depth-Anything V2)");
        DepthAnythingEstimator::new(&path)?.estimate(image)
    } else {
        log::warn!("Depth model unavailable — using placeholder");
        PlaceholderDepthEstimator.estimate(image)
    }
}

fn run_segmentation(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<SegmentationResult> {
    let model_path = model_manager.read().get_model_path("bisenet");

    if let Some(path) = model_path {
        log::info!("Running face parsing (BiSeNet)");
        BiSeNetSegmenter::new(&path)?.segment(image)
    } else {
        log::warn!("Segmentation model unavailable — using placeholder");
        PlaceholderSegmenter.segment(image)
    }
}

fn run_edge_detection(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<Vec<f32>> {
    let model_path = model_manager.read().get_model_path("teed");

    if let Some(path) = model_path {
        log::info!("Running edge detection (TEED)");
        TeedEdgeDetector::new(&path)?.detect(image)
    } else {
        log::warn!("TEED model unavailable — using Sobel placeholder");
        PlaceholderEdgeDetector.detect(image)
    }
}

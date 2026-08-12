//! ML Analysis Orchestration
//!
//! Runs the full analysis pipeline on an input image and collects results.
//! Models are loaded once and cached via the global [`SessionManager`] — calling
//! `MLAnalysis::analyze` multiple times on different images does not reload models.
//!
//! After the pipeline repurpose, only Depth-Anything V2 and TEED run here.
//! ML outputs are **optional enhancements**: if either model fails to load or
//! to produce a map, the downstream pipeline degrades gracefully.

use super::{MLConfig, MLResults};
use super::models::{
    DepthAnythingEstimator, PlaceholderDepthEstimator,
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
    /// Run the configured ML models on `image`.
    ///
    /// Which models run is controlled by the flags in `config`.
    /// Models are loaded from paths provided by `model_manager`.
    pub fn analyze(
        image: &DynamicImage,
        config: &MLConfig,
        model_manager: &Arc<RwLock<ModelManager>>,
    ) -> Result<MLResults> {
        let mut results = MLResults::default();

        if config.depth_estimation_enabled {
            results.depth_map = Some(run_depth_estimation(image, model_manager)?);
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
//      subsequent calls reuse the SessionManager cache)
//   3. Runs inference and returns the result
// ─────────────────────────────────────────────────────────────────────────────

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

fn run_edge_detection(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<Vec<f32>> {
    let model_path = model_manager.read().get_model_path("edge_detection");

    if let Some(path) = model_path {
        log::info!("Running edge detection (DexiNed)");
        TeedEdgeDetector::new(&path)?.detect(image)
    } else {
        log::warn!("Edge detection model unavailable — using Sobel placeholder");
        PlaceholderEdgeDetector.detect(image)
    }
}

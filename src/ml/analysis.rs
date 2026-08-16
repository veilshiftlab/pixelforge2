//! ML Analysis Orchestration
//!
//! Runs the full analysis pipeline on an input image and collects results.
//! Models are loaded once and cached via the global [`SessionManager`] — calling
//! `MLAnalysis::analyze` multiple times on different images does not reload models.
//!
//! Three ML models run here: Depth-Anything V2 (depth), DexiNed (edges), and
//! AnimeSegment (foreground/background mask). ML outputs are **optional
//! enhancements**: if any model fails to load or produce a map, the downstream
//! pipeline degrades gracefully (SLIC fallback for background classification).

use super::{MLConfig, MLResults};
use super::models::{
    AnimeSegmenter, PlaceholderSegmenter,
    DepthAnythingEstimator, PlaceholderDepthEstimator,
    TeedEdgeDetector, PlaceholderEdgeDetector,
};
use crate::models::ModelManager;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
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

            // Phase 6 — P1: cache the 5×5 median-filtered depth map so
            // `depth_to_flat` doesn't recompute it on every pipeline
            // invocation (every slider tweak). The filter is a pure
            // function of the depth map, so it's safe to share across
            // pipeline runs that use the same ML results.
            let (w, h) = image.dimensions();
            if let Some(ref raw) = results.depth_map {
                results.filtered_depth_map =
                    Some(crate::processing::median_filter_5x5(raw, w, h));
            }
        }

        if config.edge_detection_enabled {
            results.edge_map = Some(run_edge_detection(image, model_manager)?);
        }

        if config.segmentation_enabled {
            results.segmentation_mask = Some(run_segmentation(image, model_manager)?);
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

fn run_segmentation(
    image: &DynamicImage,
    model_manager: &Arc<RwLock<ModelManager>>,
) -> Result<Vec<f32>> {
    let model_path = model_manager.read().get_model_path("segmentation");

    if let Some(path) = model_path {
        log::info!("Running segmentation (AnimeSegment)");
        AnimeSegmenter::new(&path)?.segment(image)
    } else {
        log::warn!(
            "Segmentation model unavailable — using placeholder. \
             Background classification will fall back to SLIC heuristic."
        );
        PlaceholderSegmenter.segment(image)
    }
}

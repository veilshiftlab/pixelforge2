//! ML analysis module
//!
//! Provides ML-based image analysis for pixel art style transfer:
//! - Face detection and landmark estimation
//! - Depth map estimation
//! - Semantic segmentation
//!
//! Uses ONNX Runtime with unified session management for efficient
//! model caching and optional GPU acceleration.

// Module declarations
mod config;
mod types;
mod session;
mod preprocess;
mod analysis;
mod visualization;

// Legacy model wrappers
mod face_detection;
mod depth;
mod segmentation;

// Public exports - Configuration
pub use config::MLConfig;

// Public exports - Types
pub use types::{
    FaceBounds,
    FaceDetectionResult,
    FaceLandmarks,
    LandmarkRegion,
    SegmentationRegion,
    SegmentationResult,
    MLResults,
};

// Public exports - Legacy Model Wrappers
pub use face_detection::{FaceDetector, PlaceholderDetector};
pub use depth::{DepthEstimator, PlaceholderDepthEstimator};
pub use segmentation::{Segmenter, PlaceholderSegmenter};

// Public exports - Analysis
pub use analysis::MLAnalysis;

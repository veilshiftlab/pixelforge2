//! ML Analysis Module
//!
//! This module provides machine learning capabilities for PixelForge:
//!
//! - **Face Detection**: YOLOv8n-Face with 5 facial landmarks
//! - **Depth Estimation**: Depth-Anything V2 Small
//! - **Segmentation**: BiSeNet face parsing (19 classes)
//!
//! # Module Structure
//!
//! ```text
//! src/ml/
//! ├── mod.rs           - Module exports and types
//! ├── config.rs        - ML configuration structures
//! ├── analysis.rs      - Pipeline orchestration
//! ├── types.rs         - Result types
//! └── models/
//!     ├── mod.rs
//!     ├── yolov8_face.rs    - Face detection model
//!     ├── depth_anything.rs - Depth estimation model
//!     └── bisenet.rs        - Segmentation model
//! ```

mod config;
mod analysis;
mod types;
pub mod models;

pub use config::*;
pub use analysis::*;
pub use types::*;

// Re-export model structs for convenience
pub use models::{
    YoloV8FaceDetector, PlaceholderFaceDetector,
    DepthAnythingEstimator, PlaceholderDepthEstimator,
    BiSeNetSegmenter, PlaceholderSegmenter,
};

// Legacy re-exports (for backward compatibility)
pub use models::YoloV8FaceDetector as FaceDetector;
pub use models::DepthAnythingEstimator as DepthEstimator;
pub use models::BiSeNetSegmenter as Segmenter;

//! ML Analysis Module
//!
//! Provides machine learning map generation for PixelForge:
//!
//! - **Face Detection** (`yolov8_face`): YOLOv8n-Face — bounding box + 5 real landmarks
//! - **Depth Estimation** (`depth_anything`): Depth-Anything V2 — per-pixel relative depth
//! - **Segmentation** (`bisenet`): BiSeNet — 19-class face parsing
//! - **Edge Detection** (`teed`): TEED — crisp perceptual edge map
//!
//! # Module layout
//!
//! ```text
//! src/ml/
//! ├── mod.rs            — re-exports
//! ├── config.rs         — per-model and combined configuration
//! ├── analysis.rs       — pipeline orchestration (MLAnalysis::analyze)
//! ├── types.rs          — result types (MLResults, SegmentationRegion, …)
//! ├── preprocess.rs     — tensor preparation and bilinear coordinate remapping
//! └── models/
//!     ├── mod.rs
//!     ├── yolov8_face.rs
//!     ├── depth_anything.rs
//!     ├── bisenet.rs
//!     └── teed.rs
//! ```

mod analysis;
mod config;
mod types;
pub mod models;
pub mod preprocess;

pub use analysis::MLAnalysis;
pub use config::*;
pub use types::*;

// Public model structs — needed by ModelManager and tests
pub use models::yolov8_face::YoloV8FaceDetector;
pub use models::depth_anything::DepthAnythingEstimator;
pub use models::bisenet::BiSeNetSegmenter;
pub use models::teed::TeedEdgeDetector;
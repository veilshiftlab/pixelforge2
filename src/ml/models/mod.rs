//! ML Model Implementations
//!
//! Each submodule provides:
//! - A concrete model struct that loads and runs an ONNX model
//! - A `Placeholder*` struct for testing when models are not yet downloaded
//!
//! All models use [`crate::ml::preprocess`] for consistent input preparation
//! and coordinate remapping.

pub mod depth_anything;
pub mod teed;

pub use depth_anything::{DepthAnythingEstimator, PlaceholderDepthEstimator};
pub use teed::{PlaceholderEdgeDetector, TeedEdgeDetector};

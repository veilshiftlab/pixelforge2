//! ML Model Implementations
//!
//! This module contains the ONNX model implementations for each ML task:
//!
//! - **yolov8_face**: Face detection with 5 landmarks
//! - **depth_anything**: Monocular depth estimation
//! - **bisenet**: Face parsing / semantic segmentation
//!
//! Each model module provides:
//! - A detector/estimator struct that loads and runs the ONNX model
//! - A placeholder implementation for when the model is not available

pub mod yolov8_face;
pub mod depth_anything;
pub mod bisenet;

pub use yolov8_face::{YoloV8FaceDetector, PlaceholderFaceDetector};
pub use depth_anything::{DepthAnythingEstimator, PlaceholderDepthEstimator};
pub use bisenet::{BiSeNetSegmenter, PlaceholderSegmenter};

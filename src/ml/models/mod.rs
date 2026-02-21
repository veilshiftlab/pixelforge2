//! ML Model Implementations
//!
//! Provides model-specific implementations:
//! - YOLOv8n-Face: Face detection with 5 landmarks
//! - Depth-Anything V2: Depth estimation
//! - BiSeNet: Face parsing segmentation

mod yolov8_face;
mod depth_anything;
mod bisenet;

// Re-export
pub use yolov8_face::YOLOv8FaceDetector;
pub use depth_anything::DepthAnythingV2;
pub use bisenet::BiSeNetSegmenter;

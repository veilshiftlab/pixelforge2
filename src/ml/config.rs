//! ML configuration structures

use serde::{Deserialize, Serialize};

/// ML Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Enable face detection
    pub face_detection_enabled: bool,

    /// Enable depth estimation
    pub depth_estimation_enabled: bool,

    /// Enable segmentation
    pub segmentation_enabled: bool,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            face_detection_enabled: true,
            depth_estimation_enabled: true,
            segmentation_enabled: true,
        }
    }
}

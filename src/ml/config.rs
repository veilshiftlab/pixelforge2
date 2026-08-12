//! ML Configuration Structures
//!
//! After the pipeline repurpose, the only ML models are Depth-Anything V2
//! and TEED. Face-detection and segmentation configs were removed along with
//! the models themselves.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Depth Estimation (Depth-Anything V2)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DepthColormap {
    #[default]
    Turbo,
    Viridis,
    Grayscale,
    Plasma,
    Inferno,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthConfig {
    /// Normalize output to 0–1 (default: true; should almost always be true)
    pub normalize_output: bool,
    /// Invert depth values — true: nearer = higher value (default: false)
    pub invert: bool,
    /// Gamma correction for visualization (default: 1.0)
    pub gamma: f32,
    /// Visualization colormap
    pub colormap: DepthColormap,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            normalize_output: true,
            invert: false,
            gamma: 1.0,
            colormap: DepthColormap::Turbo,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge Detection (TEED)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Threshold above which a pixel is considered an edge (default: 0.3)
    /// Lower values = more edges detected; higher = only strong edges
    pub threshold: f32,
    /// Dilate detected edges by N pixels for downstream use (default: 0)
    pub dilation_px: u32,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self { threshold: 0.3, dilation_px: 0 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined ML configuration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub depth: DepthConfig,
    pub edge: EdgeConfig,

    /// Enable depth estimation
    pub depth_estimation_enabled: bool,
    /// Enable TEED edge detection
    pub edge_detection_enabled: bool,

    /// Execution mode (CPU / GPU sequential / GPU parallel)
    pub execution: ExecutionConfig,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            depth: DepthConfig::default(),
            edge: EdgeConfig::default(),
            depth_estimation_enabled: true,
            edge_detection_enabled: true,
            execution: ExecutionConfig::default(),
        }
    }
}

impl MLConfig {
    pub fn depth_only() -> Self {
        Self {
            depth_estimation_enabled: true,
            edge_detection_enabled: false,
            ..Default::default()
        }
    }

    pub fn edges_only() -> Self {
        Self {
            depth_estimation_enabled: false,
            edge_detection_enabled: true,
            ..Default::default()
        }
    }

    pub fn any_enabled(&self) -> bool {
        self.depth_estimation_enabled || self.edge_detection_enabled
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution configuration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionMode {
    CpuOnly,
    #[default]
    GpuSequential,
    GpuParallel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub mode: ExecutionMode,
    pub gpu_device: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self { mode: ExecutionMode::GpuSequential, gpu_device: 0 }
    }
}

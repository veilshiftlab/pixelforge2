//! ML Configuration Structures

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Face Detection (YOLOv8n-Face)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetectionConfig {
    /// Minimum objectness confidence (default: 0.5)
    pub confidence_threshold: f32,
    /// IoU threshold for NMS (default: 0.45)
    pub iou_threshold: f32,
}

impl Default for FaceDetectionConfig {
    fn default() -> Self {
        Self { confidence_threshold: 0.5, iou_threshold: 0.45 }
    }
}

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
// Segmentation (BiSeNet)
// ─────────────────────────────────────────────────────────────────────────────

/// BiSeNet class identifiers — all 19 classes, no collapsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentationClass {
    Background      = 0,
    Skin            = 1,
    LeftEyebrow     = 2,
    RightEyebrow    = 3,
    LeftEye         = 4,
    RightEye        = 5,
    Eyeglasses      = 6,
    LeftEar         = 7,
    RightEar        = 8,
    Earring         = 9,
    Nose            = 10,
    InnerMouth      = 11,
    UpperLip        = 12,
    LowerLip        = 13,
    Neck            = 14,
    Hair            = 15,
    Hat             = 16,
    LeftEarringDetail  = 17,
    RightEarringDetail = 18,
}

impl SegmentationClass {
    pub fn from_index(i: u8) -> Option<Self> {
        match i {
            0  => Some(Self::Background),
            1  => Some(Self::Skin),
            2  => Some(Self::LeftEyebrow),
            3  => Some(Self::RightEyebrow),
            4  => Some(Self::LeftEye),
            5  => Some(Self::RightEye),
            6  => Some(Self::Eyeglasses),
            7  => Some(Self::LeftEar),
            8  => Some(Self::RightEar),
            9  => Some(Self::Earring),
            10 => Some(Self::Nose),
            11 => Some(Self::InnerMouth),
            12 => Some(Self::UpperLip),
            13 => Some(Self::LowerLip),
            14 => Some(Self::Neck),
            15 => Some(Self::Hair),
            16 => Some(Self::Hat),
            17 => Some(Self::LeftEarringDetail),
            18 => Some(Self::RightEarringDetail),
            _  => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Background         => "Background",
            Self::Skin               => "Skin",
            Self::LeftEyebrow        => "Left Eyebrow",
            Self::RightEyebrow       => "Right Eyebrow",
            Self::LeftEye            => "Left Eye",
            Self::RightEye           => "Right Eye",
            Self::Eyeglasses         => "Eyeglasses",
            Self::LeftEar            => "Left Ear",
            Self::RightEar           => "Right Ear",
            Self::Earring            => "Earring",
            Self::Nose               => "Nose",
            Self::InnerMouth         => "Inner Mouth",
            Self::UpperLip           => "Upper Lip",
            Self::LowerLip           => "Lower Lip",
            Self::Neck               => "Neck",
            Self::Hair               => "Hair",
            Self::Hat                => "Hat",
            Self::LeftEarringDetail  => "Left Earring Detail",
            Self::RightEarringDetail => "Right Earring Detail",
        }
    }

    pub fn is_high_detail(&self) -> bool {
        matches!(
            self,
            Self::LeftEye | Self::RightEye | Self::UpperLip | Self::LowerLip |
            Self::InnerMouth | Self::Nose | Self::LeftEyebrow | Self::RightEyebrow
        )
    }

    pub fn all() -> Vec<Self> {
        (0u8..=18).filter_map(Self::from_index).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationConfig {
    /// Confidence threshold for class assignment (default: 0.5)
    pub confidence_threshold: f32,
    /// Discard isolated regions smaller than N pixels (default: 16)
    pub min_region_size: usize,
    /// Classes to show in visualization (empty = all)
    pub visible_classes: Vec<SegmentationClass>,
    /// Overlay opacity for visualization (0=original, 1=seg only)
    pub overlay_opacity: f32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_region_size: 16,
            visible_classes: Vec::new(),
            overlay_opacity: 0.5,
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
    pub face_detection: FaceDetectionConfig,
    pub depth: DepthConfig,
    pub segmentation: SegmentationConfig,
    pub edge: EdgeConfig,

    /// Enable face detection (portrait toggle)
    pub face_detection_enabled: bool,
    /// Enable depth estimation
    pub depth_estimation_enabled: bool,
    /// Enable BiSeNet segmentation
    pub segmentation_enabled: bool,
    /// Enable TEED edge detection
    pub edge_detection_enabled: bool,

    /// Execution mode (CPU / GPU sequential / GPU parallel)
    pub execution: ExecutionConfig,

    // ── Flat shims used by panels.rs ──────────────────────────────────────
    // These mirror values in the nested sub-configs; keep them in sync via
    // the Default impl below.  Longer-term, panels.rs should access
    // ml_config.face_detection.confidence_threshold etc. directly.

    /// Face detection confidence threshold (mirrors face_detection.confidence_threshold)
    pub face_confidence_threshold: f32,
    /// Depth edge sensitivity — used as edge detection threshold (mirrors edge.threshold)
    pub depth_edge_sensitivity: f32,
    /// Segmentation class confidence threshold (mirrors segmentation.confidence_threshold)
    pub segmentation_sensitivity: f32,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            face_detection: FaceDetectionConfig::default(),
            depth: DepthConfig::default(),
            segmentation: SegmentationConfig::default(),
            edge: EdgeConfig::default(),
            face_detection_enabled: true,
            depth_estimation_enabled: true,
            segmentation_enabled: true,
            edge_detection_enabled: true,
            execution: ExecutionConfig::default(),
            face_confidence_threshold: 0.5,
            depth_edge_sensitivity: 0.3,
            segmentation_sensitivity: 0.5,
        }
    }
}

impl MLConfig {
    pub fn face_only() -> Self {
        Self {
            face_detection_enabled: true,
            depth_estimation_enabled: false,
            segmentation_enabled: false,
            edge_detection_enabled: false,
            ..Default::default()
        }
    }

    pub fn depth_only() -> Self {
        Self {
            face_detection_enabled: false,
            depth_estimation_enabled: true,
            segmentation_enabled: false,
            edge_detection_enabled: false,
            ..Default::default()
        }
    }

    pub fn segmentation_only() -> Self {
        Self {
            face_detection_enabled: false,
            depth_estimation_enabled: false,
            segmentation_enabled: true,
            edge_detection_enabled: false,
            ..Default::default()
        }
    }

    pub fn edges_only() -> Self {
        Self {
            face_detection_enabled: false,
            depth_estimation_enabled: false,
            segmentation_enabled: false,
            edge_detection_enabled: true,
            ..Default::default()
        }
    }

    pub fn any_enabled(&self) -> bool {
        self.face_detection_enabled
            || self.depth_estimation_enabled
            || self.segmentation_enabled
            || self.edge_detection_enabled
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


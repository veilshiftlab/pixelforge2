//! ML Configuration Structures
//!
//! Provides configurable parameters for each ML component:
//! - Face Detection (YOLOv8n-Face)
//! - Depth Estimation (Depth-Anything V2)
//! - Segmentation (BiSeNet)

use serde::{Deserialize, Serialize};

// =============================================================================
// Face Detection Configuration (YOLOv8n-Face)
// =============================================================================

/// Configuration for YOLOv8n-Face face detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetectionConfig {
    /// Minimum confidence threshold for detection (default: 0.25)
    /// Lower values detect more faces but may have false positives
    pub confidence_threshold: f32,

    /// IoU threshold for Non-Maximum Suppression (default: 0.45)
    /// Higher values keep more overlapping detections
    pub iou_threshold: f32,
}

impl Default for FaceDetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
        }
    }
}

// =============================================================================
// Depth Estimation Configuration (Depth-Anything V2)
// =============================================================================

/// Color map options for depth visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DepthColormap {
    /// Perceptually uniform, vibrant colors (default)
    #[default]
    Turbo,
    /// Colorblind-friendly, blue-yellow gradient
    Viridis,
    /// Simple black-white gradient
    Grayscale,
    /// High contrast purple-yellow
    Plasma,
    /// Inferno colormap (black-red-yellow)
    Inferno,
}

/// Configuration for Depth-Anything V2 depth estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthConfig {
    /// Normalize output to 0-1 range (default: true)
    /// Recommended for consistent processing
    pub normalize_output: bool,

    /// Invert depth values (default: false)
    /// false: nearer objects have lower values (standard)
    /// true: nearer objects have higher values
    pub invert: bool,

    /// Gamma correction for visualization (default: 1.0)
    /// Values > 1.0 brighten mid-tones
    /// Values < 1.0 darken mid-tones
    pub gamma: f32,

    /// Color map for visualization
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

// =============================================================================
// Segmentation Configuration (BiSeNet)
// =============================================================================

/// Segmentation class identifiers (BiSeNet 19 classes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentationClass {
    Background = 0,
    Skin = 1,
    LeftEyebrow = 2,
    RightEyebrow = 3,
    LeftEye = 4,
    RightEye = 5,
    Eyeglasses = 6,
    LeftEar = 7,
    RightEar = 8,
    Earring = 9,
    Nose = 10,
    InnerMouth = 11,
    UpperLip = 12,
    LowerLip = 13,
    Neck = 14,
    Hair = 15,
    Hat = 16,
    LeftEarringDetail = 17,
    RightEarringDetail = 18,
}

impl SegmentationClass {
    /// Get class from index (0-18)
    pub fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Background),
            1 => Some(Self::Skin),
            2 => Some(Self::LeftEyebrow),
            3 => Some(Self::RightEyebrow),
            4 => Some(Self::LeftEye),
            5 => Some(Self::RightEye),
            6 => Some(Self::Eyeglasses),
            7 => Some(Self::LeftEar),
            8 => Some(Self::RightEar),
            9 => Some(Self::Earring),
            10 => Some(Self::Nose),
            11 => Some(Self::InnerMouth),
            12 => Some(Self::UpperLip),
            13 => Some(Self::LowerLip),
            14 => Some(Self::Neck),
            15 => Some(Self::Hair),
            16 => Some(Self::Hat),
            17 => Some(Self::LeftEarringDetail),
            18 => Some(Self::RightEarringDetail),
            _ => None,
        }
    }

    /// Get display name for the class
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Background => "Background",
            Self::Skin => "Skin",
            Self::LeftEyebrow => "Left Eyebrow",
            Self::RightEyebrow => "Right Eyebrow",
            Self::LeftEye => "Left Eye",
            Self::RightEye => "Right Eye",
            Self::Eyeglasses => "Eyeglasses",
            Self::LeftEar => "Left Ear",
            Self::RightEar => "Right Ear",
            Self::Earring => "Earring",
            Self::Nose => "Nose",
            Self::InnerMouth => "Inner Mouth",
            Self::UpperLip => "Upper Lip",
            Self::LowerLip => "Lower Lip",
            Self::Neck => "Neck",
            Self::Hair => "Hair",
            Self::Hat => "Hat",
            Self::LeftEarringDetail => "Left Earring Detail",
            Self::RightEarringDetail => "Right Earring Detail",
        }
    }

    /// Check if this class represents facial features (for special processing)
    pub fn is_facial_feature(&self) -> bool {
        matches!(
            self,
            Self::LeftEye
                | Self::RightEye
                | Self::Nose
                | Self::UpperLip
                | Self::LowerLip
                | Self::InnerMouth
        )
    }

    /// Check if this class represents hair
    pub fn is_hair(&self) -> bool {
        matches!(self, Self::Hair)
    }

    /// Check if this class represents skin
    pub fn is_skin(&self) -> bool {
        matches!(self, Self::Skin)
    }

    /// Get all classes as a vector
    pub fn all() -> Vec<Self> {
        (0..=18)
            .filter_map(|i| Self::from_index(i))
            .collect()
    }
}

/// Configuration for BiSeNet face parsing segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationConfig {
    /// Minimum confidence threshold for class assignment (default: 0.5)
    pub confidence_threshold: f32,

    /// Merge isolated regions smaller than N pixels (default: 16)
    /// Helps reduce noise in segmentation output
    pub min_region_size: usize,

    /// Which classes to include in visualization (default: all)
    /// Empty vector means show all classes
    pub visible_classes: Vec<SegmentationClass>,

    /// Overlay opacity for visualization (default: 0.5)
    /// 0.0 = original image only, 1.0 = segmentation only
    pub overlay_opacity: f32,

    /// Whether to merge eyebrows into a single region
    pub merge_eyebrows: bool,

    /// Whether to merge eyes into a single region
    pub merge_eyes: bool,

    /// Whether to merge lips into a single region
    pub merge_lips: bool,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_region_size: 16,
            visible_classes: Vec::new(), // Empty = all visible
            overlay_opacity: 0.5,
            merge_eyebrows: true,
            merge_eyes: true,
            merge_lips: true,
        }
    }
}

// =============================================================================
// Combined ML Configuration
// =============================================================================

/// Main ML analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Face detection settings
    pub face_detection: FaceDetectionConfig,

    /// Depth estimation settings
    pub depth: DepthConfig,

    /// Segmentation settings
    pub segmentation: SegmentationConfig,

    /// Enable face detection
    pub enable_face_detection: bool,

    /// Enable depth estimation
    pub enable_depth: bool,

    /// Enable segmentation
    pub enable_segmentation: bool,
    
    // Legacy fields for backward compatibility
    pub face_detection_enabled: bool,
    pub depth_estimation_enabled: bool,
    pub segmentation_enabled: bool,
    pub face_confidence_threshold: f32,
    pub depth_edge_sensitivity: f32,
    pub segmentation_sensitivity: f32,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            face_detection: FaceDetectionConfig::default(),
            depth: DepthConfig::default(),
            segmentation: SegmentationConfig::default(),
            enable_face_detection: true,
            enable_depth: true,
            enable_segmentation: true,
            // Legacy fields
            face_detection_enabled: true,
            depth_estimation_enabled: true,
            segmentation_enabled: true,
            face_confidence_threshold: 0.5,
            depth_edge_sensitivity: 0.15,
            segmentation_sensitivity: 0.5,
        }
    }
}

impl MLConfig {
    /// Create config with only face detection enabled
    pub fn face_only() -> Self {
        Self {
            enable_face_detection: true,
            enable_depth: false,
            enable_segmentation: false,
            face_detection_enabled: true,
            depth_estimation_enabled: false,
            segmentation_enabled: false,
            ..Default::default()
        }
    }

    /// Create config with only depth estimation enabled
    pub fn depth_only() -> Self {
        Self {
            enable_face_detection: false,
            enable_depth: true,
            enable_segmentation: false,
            face_detection_enabled: false,
            depth_estimation_enabled: true,
            segmentation_enabled: false,
            ..Default::default()
        }
    }

    /// Create config with only segmentation enabled
    pub fn segmentation_only() -> Self {
        Self {
            enable_face_detection: false,
            enable_depth: false,
            enable_segmentation: true,
            face_detection_enabled: false,
            depth_estimation_enabled: false,
            segmentation_enabled: true,
            ..Default::default()
        }
    }

    /// Check if any ML component is enabled
    pub fn any_enabled(&self) -> bool {
        self.enable_face_detection || self.enable_depth || self.enable_segmentation
            || self.face_detection_enabled || self.depth_estimation_enabled || self.segmentation_enabled
    }
}

// =============================================================================
// Execution Configuration
// =============================================================================

/// Execution backend preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionMode {
    /// Use CPU only (slowest, most compatible)
    CpuOnly,
    /// Use GPU with sequential model execution (recommended for 4GB VRAM)
    #[default]
    GpuSequential,
    /// Use GPU with parallel model execution (faster, needs more VRAM)
    GpuParallel,
}

/// Model variant selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelVariant {
    /// Small/fast variant (~550MB total, lower quality)
    Small,
    /// Base variant (~1.1GB total, higher quality) - RECOMMENDED
    #[default]
    Base,
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Execution mode
    pub mode: ExecutionMode,

    /// Model variant
    pub variant: ModelVariant,

    /// GPU device index (0 = first GPU)
    pub gpu_device: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::GpuSequential,
            variant: ModelVariant::Base,
            gpu_device: 0,
        }
    }
}

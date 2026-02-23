//! ML data types and structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Results from ML analysis
#[derive(Debug, Clone, Default)]
pub struct MLResults {
    /// Detected face landmarks
    pub landmarks: Option<FaceLandmarks>,

    /// Depth map (normalized 0.0–1.0, row-major, same dims as input image)
    pub depth_map: Option<Vec<f32>>,

    /// Segmentation results
    pub segmentation: Option<SegmentationResult>,

    /// Edge map from TEED (normalized 0.0–1.0, row-major, same dims as input image)
    pub edge_map: Option<Vec<f32>>,

    /// Face bounding box
    pub face_bounds: Option<FaceBounds>,

    /// Face detection output
    pub face: Option<FaceDetectionOutput>,
}

/// Face bounding box — coordinates normalized to 0.0–1.0 relative to original image
#[derive(Debug, Clone, Copy)]
pub struct FaceBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl FaceBounds {
    /// Convert normalized bounds to pixel coordinates for a given image size
    pub fn to_pixels(&self, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
        let x = (self.x * img_w as f32) as u32;
        let y = (self.y * img_h as f32) as u32;
        let w = (self.width * img_w as f32) as u32;
        let h = (self.height * img_h as f32) as u32;
        (x, y, w, h)
    }
}

/// Facial landmarks — all coordinates normalized to 0.0–1.0 relative to original image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarks {
    /// All landmark points (normalized)
    pub points: Vec<(f32, f32)>,

    /// Left eye region
    pub left_eye: Option<LandmarkRegion>,

    /// Right eye region
    pub right_eye: Option<LandmarkRegion>,

    /// Nose region
    pub nose: Option<LandmarkRegion>,

    /// Lips region
    pub lips: Option<LandmarkRegion>,

    /// Face outline
    pub face_outline: Vec<(f32, f32)>,
}

impl Default for FaceLandmarks {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            left_eye: None,
            right_eye: None,
            nose: None,
            lips: None,
            face_outline: Vec::new(),
        }
    }
}

/// A landmark region with center and approximate size (normalized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkRegion {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
}

/// Per-pixel segmentation results at original image resolution
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Per-pixel region labels (pixel index = y * width + x)
    pub regions: HashMap<u32, SegmentationRegion>,

    /// Bounding box per region (x_min, y_min, x_max, y_max) normalized
    pub region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)>,

    /// Per-pixel confidence scores
    pub confidence: Vec<f32>,
}

impl Default for SegmentationResult {
    fn default() -> Self {
        Self {
            regions: HashMap::new(),
            region_bounds: HashMap::new(),
            confidence: Vec::new(),
        }
    }
}

/// Segmentation region types — maps to BiSeNet's 19 classes without discarding information.
///
/// Preserving fine-grained regions allows downstream pixel art processing to apply
/// distinct palette, dithering, and detail strategies per region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SegmentationRegion {
    #[default]
    Background,
    /// General skin / face surface
    Skin,
    /// Left eyebrow
    LeftEyebrow,
    /// Right eyebrow
    RightEyebrow,
    /// Left eye (iris + sclera)
    LeftEye,
    /// Right eye (iris + sclera)
    RightEye,
    /// Eyeglasses frame and lenses
    Eyeglasses,
    /// Left ear
    LeftEar,
    /// Right ear
    RightEar,
    /// Earring / jewelry (generic)
    Earring,
    /// Nose
    Nose,
    /// Inner mouth / teeth
    InnerMouth,
    /// Upper lip
    UpperLip,
    /// Lower lip
    LowerLip,
    /// Neck
    Neck,
    /// Hair
    Hair,
    /// Hat or head covering
    Hat,
}

impl SegmentationRegion {
    /// Returns true for regions that contain fine detail important for pixel art fidelity
    pub fn is_high_detail(self) -> bool {
        matches!(
            self,
            Self::LeftEye
                | Self::RightEye
                | Self::UpperLip
                | Self::LowerLip
                | Self::InnerMouth
                | Self::Nose
                | Self::LeftEyebrow
                | Self::RightEyebrow
        )
    }

    /// Returns true for regions that benefit from special hair-rendering treatment
    pub fn is_hair_like(self) -> bool {
        matches!(self, Self::Hair | Self::Hat)
    }

    /// Returns true for skin-tone regions
    pub fn is_skin_like(self) -> bool {
        matches!(self, Self::Skin | Self::Neck)
    }

    /// Human-readable name for UI display
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Background   => "Background",
            Self::Skin         => "Skin",
            Self::LeftEyebrow  => "Left Eyebrow",
            Self::RightEyebrow => "Right Eyebrow",
            Self::LeftEye      => "Left Eye",
            Self::RightEye     => "Right Eye",
            Self::Eyeglasses   => "Eyeglasses",
            Self::LeftEar      => "Left Ear",
            Self::RightEar     => "Right Ear",
            Self::Earring      => "Earring",
            Self::Nose         => "Nose",
            Self::InnerMouth   => "Inner Mouth",
            Self::UpperLip     => "Upper Lip",
            Self::LowerLip     => "Lower Lip",
            Self::Neck         => "Neck",
            Self::Hair         => "Hair",
            Self::Hat          => "Hat / Head Covering",
        }
    }

    /// Canonical RGBA color for visualization overlays.
    /// Returns `(r, g, b, a)` with a=255.
    pub fn rgba(self) -> (u8, u8, u8, u8) {
        match self {
            Self::Background             => ( 30,  30,  30, 255),
            Self::Skin                   => (255, 200, 150, 255),
            Self::LeftEyebrow
            | Self::RightEyebrow         => (139,  90,  43, 255),
            Self::LeftEye
            | Self::RightEye             => (100, 150, 255, 255),
            Self::Eyeglasses             => (180, 180, 180, 255),
            Self::LeftEar
            | Self::RightEar             => (230, 180, 140, 255),
            Self::Earring                => (255, 215,   0, 255),
            Self::Nose                   => (255, 160,  80, 255),
            Self::InnerMouth             => (200,  20,  20, 255),
            Self::UpperLip
            | Self::LowerLip             => (255, 100, 100, 255),
            Self::Neck                   => (220, 185, 140, 255),
            Self::Hair                   => (100,  60,  20, 255),
            Self::Hat                    => (200, 200, 200, 255),
        }
    }
}

/// Face detection result
#[derive(Debug, Clone)]
pub struct FaceDetectionResult {
    /// Detected face bounds (normalized)
    pub bounds: Option<FaceBounds>,

    /// Facial landmarks (normalized)
    pub landmarks: Option<FaceLandmarks>,

    /// Detection confidence 0.0–1.0
    pub confidence: f32,
}

impl Default for FaceDetectionResult {
    fn default() -> Self {
        Self {
            bounds: None,
            landmarks: None,
            confidence: 0.0,
        }
    }
}

/// Face detection output stored in MLResults
#[derive(Debug, Clone, Default)]
pub struct FaceDetectionOutput {
    /// Detected face bounds (normalized)
    pub bounds: Option<FaceBounds>,

    /// 5 facial landmark points in model order:
    /// [left_eye, right_eye, nose_tip, left_mouth, right_mouth] (normalized)
    pub landmarks: Vec<(f32, f32)>,

    /// Detection confidence
    pub confidence: f32,
}
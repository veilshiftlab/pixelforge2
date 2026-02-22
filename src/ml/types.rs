//! ML data types and structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Results from ML analysis
#[derive(Debug, Clone, Default)]
pub struct MLResults {
    /// Detected face landmarks
    pub landmarks: Option<FaceLandmarks>,

    /// Depth map (normalized 0.0-1.0)
    pub depth_map: Option<Vec<f32>>,

    /// Segmentation results
    pub segmentation: Option<SegmentationResult>,

    /// Face bounding box
    pub face_bounds: Option<FaceBounds>,

    /// Face detection output
    pub face: Option<FaceDetectionOutput>,
}

/// Face bounding box
#[derive(Debug, Clone, Copy)]
pub struct FaceBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Facial landmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarks {
    /// All landmark points (normalized 0.0-1.0)
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

/// A landmark region with bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkRegion {
    /// Center X (normalized)
    pub center_x: f32,

    /// Center Y (normalized)
    pub center_y: f32,

    /// Width (normalized)
    pub width: f32,

    /// Height (normalized)
    pub height: f32,
}

/// Segmentation results
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Region map (pixel index -> region type)
    pub regions: HashMap<u32, SegmentationRegion>,

    /// Region bounding boxes
    pub region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)>,
    
    /// Confidence map
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

/// Segmentation region types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SegmentationRegion {
    #[default]
    Background,
    Face,
    Hair,
    Eyes,
    Nose,
    Lips,
    Ears,
    Neck,
    Clothing,
}

/// Face detection result
#[derive(Debug, Clone)]
pub struct FaceDetectionResult {
    /// Detected face bounds
    pub bounds: Option<FaceBounds>,

    /// Facial landmarks
    pub landmarks: Option<FaceLandmarks>,

    /// Detection confidence
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

/// Face detection output for ML results
#[derive(Debug, Clone, Default)]
pub struct FaceDetectionOutput {
    /// Detected face bounds
    pub bounds: Option<FaceBounds>,

    /// Facial landmark points
    pub landmarks: Vec<(f32, f32)>,

    /// Detection confidence
    pub confidence: f32,
}

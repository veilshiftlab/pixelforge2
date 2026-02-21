//! ML Data Types and Output Structures
//!
//! Provides output types for ML analysis:
//! - Face Detection (YOLOv8n-Face): bounding boxes + 5 landmarks
//! - Depth Estimation (Depth-Anything V2): depth map
//! - Segmentation (BiSeNet): class map with 19 classes

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::SegmentationClass;

// =============================================================================
// Face Detection Types
// =============================================================================

/// Face bounding box (normalized 0.0-1.0)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceBounds {
    /// X position (left edge, normalized)
    pub x: f32,
    /// Y position (top edge, normalized)
    pub y: f32,
    /// Width (normalized)
    pub width: f32,
    /// Height (normalized)
    pub height: f32,
}

impl FaceBounds {
    /// Create from pixel coordinates
    pub fn from_pixels(x: u32, y: u32, w: u32, h: u32, img_width: u32, img_height: u32) -> Self {
        Self {
            x: x as f32 / img_width as f32,
            y: y as f32 / img_height as f32,
            width: w as f32 / img_width as f32,
            height: h as f32 / img_height as f32,
        }
    }

    /// Convert to pixel coordinates
    pub fn to_pixels(&self, img_width: u32, img_height: u32) -> (u32, u32, u32, u32) {
        (
            (self.x * img_width as f32) as u32,
            (self.y * img_height as f32) as u32,
            (self.width * img_width as f32) as u32,
            (self.height * img_height as f32) as u32,
        )
    }

    /// Get center point (normalized)
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Check if a point is inside the bounds
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}

/// YOLOv8-Face landmark indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceLandmarkIndex {
    LeftEye = 0,
    RightEye = 1,
    Nose = 2,
    LeftMouth = 3,
    RightMouth = 4,
}

/// Output from YOLOv8n-Face detection
#[derive(Debug, Clone)]
pub struct FaceDetectionOutput {
    /// Detected face bounding box (normalized 0-1)
    /// None if no face detected
    pub bounds: Option<FaceBounds>,

    /// 5 facial landmarks (normalized 0-1):
    /// [0] Left eye center
    /// [1] Right eye center
    /// [2] Nose tip
    /// [3] Left mouth corner
    /// [4] Right mouth corner
    pub landmarks: Vec<(f32, f32)>,

    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
}

impl Default for FaceDetectionOutput {
    fn default() -> Self {
        Self {
            bounds: None,
            landmarks: Vec::new(),
            confidence: 0.0,
        }
    }
}

impl FaceDetectionOutput {
    /// Get a specific landmark by index
    pub fn get_landmark(&self, index: FaceLandmarkIndex) -> Option<(f32, f32)> {
        self.landmarks.get(index as usize).copied()
    }

    /// Get eye centers (left, right)
    pub fn eye_centers(&self) -> Option<((f32, f32), (f32, f32))> {
        let left = self.get_landmark(FaceLandmarkIndex::LeftEye)?;
        let right = self.get_landmark(FaceLandmarkIndex::RightEye)?;
        Some((left, right))
    }

    /// Calculate eye distance (normalized)
    pub fn eye_distance(&self) -> Option<f32> {
        let (left, right) = self.eye_centers()?;
        let dx = right.0 - left.0;
        let dy = right.1 - left.1;
        Some((dx * dx + dy * dy).sqrt())
    }
}

/// Legacy face detection result (for backward compatibility)
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

impl From<FaceDetectionOutput> for FaceDetectionResult {
    fn from(output: FaceDetectionOutput) -> Self {
        let landmarks = output.bounds.as_ref().map(|b| {
            FaceLandmarks::from_yolo(&output.landmarks, b)
        });
        Self {
            bounds: output.bounds,
            landmarks,
            confidence: output.confidence,
        }
    }
}

/// Facial landmarks structure
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

impl FaceLandmarks {
    /// Create from YOLOv8-Face 5-point landmarks
    pub fn from_yolo(landmarks: &[(f32, f32)], bounds: &FaceBounds) -> Self {
        if landmarks.len() < 5 {
            return Self::default();
        }

        // Calculate regions from landmarks
        let left_eye = Some(LandmarkRegion {
            center_x: landmarks[0].0,
            center_y: landmarks[0].1,
            width: bounds.width * 0.1,
            height: bounds.height * 0.05,
        });

        let right_eye = Some(LandmarkRegion {
            center_x: landmarks[1].0,
            center_y: landmarks[1].1,
            width: bounds.width * 0.1,
            height: bounds.height * 0.05,
        });

        let nose = Some(LandmarkRegion {
            center_x: landmarks[2].0,
            center_y: landmarks[2].1,
            width: bounds.width * 0.08,
            height: bounds.height * 0.1,
        });

        // Lips region from mouth corners
        let mouth_left = landmarks[3];
        let mouth_right = landmarks[4];
        let mouth_center = ((mouth_left.0 + mouth_right.0) / 2.0, (mouth_left.1 + mouth_right.1) / 2.0);

        let lips = Some(LandmarkRegion {
            center_x: mouth_center.0,
            center_y: mouth_center.1,
            width: (mouth_right.0 - mouth_left.0).abs() * 1.2,
            height: bounds.height * 0.05,
        });

        Self {
            points: landmarks.to_vec(),
            left_eye,
            right_eye,
            nose,
            lips,
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

// =============================================================================
// Depth Estimation Types
// =============================================================================

/// Output from Depth-Anything V2 depth estimation
#[derive(Debug, Clone)]
pub struct DepthOutput {
    /// Raw depth values (width × height)
    /// Values are relative depth (not metric)
    /// Higher values = further from camera (by default)
    pub depth_map: Vec<f32>,

    /// Width of the depth map
    pub width: u32,

    /// Height of the depth map
    pub height: u32,

    /// Min/max depth values in this image
    /// Useful for normalization and visualization
    pub depth_range: (f32, f32),
}

impl DepthOutput {
    /// Get depth value at pixel coordinates
    pub fn get_depth(&self, x: u32, y: u32) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.depth_map[(y * self.width + x) as usize])
        } else {
            None
        }
    }

    /// Get normalized depth value (0-1) at pixel coordinates
    pub fn get_normalized_depth(&self, x: u32, y: u32) -> Option<f32> {
        self.get_depth(x, y).map(|d| {
            let (min, max) = self.depth_range;
            if max - min > 0.0 {
                (d - min) / (max - min)
            } else {
                0.5
            }
        })
    }

    /// Get depth statistics
    pub fn statistics(&self) -> DepthStatistics {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;

        for &d in &self.depth_map {
            min = min.min(d);
            max = max.max(d);
            sum += d;
        }

        let mean = sum / self.depth_map.len() as f32;

        // Calculate standard deviation
        let variance: f32 = self.depth_map.iter()
            .map(|&d| (d - mean).powi(2))
            .sum::<f32>() / self.depth_map.len() as f32;
        let std = variance.sqrt();

        DepthStatistics {
            min,
            max,
            mean,
            std,
        }
    }
}

/// Depth map statistics
#[derive(Debug, Clone, Copy)]
pub struct DepthStatistics {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

// =============================================================================
// Segmentation Types
// =============================================================================

/// Output from BiSeNet face parsing
#[derive(Debug, Clone)]
pub struct SegmentationOutput {
    /// Class ID per pixel (0-18)
    /// Maps directly to SegmentationClass enum
    pub class_map: Vec<u8>,

    /// Confidence per pixel (0.0-1.0)
    /// Only populated if model outputs probability distribution
    pub confidence_map: Vec<f32>,

    /// Width of the segmentation map
    pub width: u32,

    /// Height of the segmentation map
    pub height: u32,

    /// Bounding box per detected class
    /// Useful for region-specific processing
    pub class_bounds: HashMap<SegmentationClass, FaceBounds>,
}

impl SegmentationOutput {
    /// Get class at pixel coordinates
    pub fn get_class(&self, x: u32, y: u32) -> Option<SegmentationClass> {
        if x < self.width && y < self.height {
            let class_id = self.class_map[(y * self.width + x) as usize];
            SegmentationClass::from_index(class_id)
        } else {
            None
        }
    }

    /// Get confidence at pixel coordinates
    pub fn get_confidence(&self, x: u32, y: u32) -> Option<f32> {
        if x < self.width && y < self.height && !self.confidence_map.is_empty() {
            Some(self.confidence_map[(y * self.width + x) as usize])
        } else {
            None
        }
    }

    /// Count pixels per class
    pub fn class_pixel_counts(&self) -> HashMap<SegmentationClass, usize> {
        let mut counts = HashMap::new();
        for &class_id in &self.class_map {
            if let Some(class) = SegmentationClass::from_index(class_id) {
                *counts.entry(class).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Get percentage coverage for each class
    pub fn class_percentages(&self) -> HashMap<SegmentationClass, f32> {
        let total = self.class_map.len() as f32;
        self.class_pixel_counts()
            .into_iter()
            .map(|(class, count)| (class, count as f32 / total * 100.0))
            .collect()
    }

    /// Check if a pixel belongs to a facial feature
    pub fn is_facial_feature(&self, x: u32, y: u32) -> bool {
        self.get_class(x, y)
            .map(|c| c.is_facial_feature())
            .unwrap_or(false)
    }

    /// Create a mask for specific classes
    pub fn create_mask(&self, classes: &[SegmentationClass]) -> Vec<bool> {
        let class_set: std::collections::HashSet<_> = classes.iter().collect();
        self.class_map
            .iter()
            .map(|&id| {
                SegmentationClass::from_index(id)
                    .map(|c| class_set.contains(&c))
                    .unwrap_or(false)
            })
            .collect()
    }
}

// =============================================================================
// Legacy Types (Backward Compatibility)
// =============================================================================

/// Legacy segmentation region types (for backward compatibility)
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

impl From<SegmentationClass> for SegmentationRegion {
    fn from(class: SegmentationClass) -> Self {
        match class {
            SegmentationClass::Background => SegmentationRegion::Background,
            SegmentationClass::Skin => SegmentationRegion::Face,
            SegmentationClass::LeftEyebrow | SegmentationClass::RightEyebrow => SegmentationRegion::Face,
            SegmentationClass::LeftEye | SegmentationClass::RightEye => SegmentationRegion::Eyes,
            SegmentationClass::Eyeglasses => SegmentationRegion::Face,
            SegmentationClass::LeftEar | SegmentationClass::RightEar => SegmentationRegion::Ears,
            SegmentationClass::Earring | SegmentationClass::LeftEarringDetail | SegmentationClass::RightEarringDetail => SegmentationRegion::Ears,
            SegmentationClass::Nose => SegmentationRegion::Nose,
            SegmentationClass::InnerMouth | SegmentationClass::UpperLip | SegmentationClass::LowerLip => SegmentationRegion::Lips,
            SegmentationClass::Neck => SegmentationRegion::Neck,
            SegmentationClass::Hair => SegmentationRegion::Hair,
            SegmentationClass::Hat => SegmentationRegion::Hair, // Treat hat as hair
        }
    }
}

/// Legacy segmentation result (for backward compatibility)
#[derive(Debug, Clone, Default)]
pub struct SegmentationResult {
    /// Region map (pixel index -> region type)
    pub regions: HashMap<u32, SegmentationRegion>,

    /// Region bounding boxes
    pub region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)>,

    /// Confidence map
    pub confidence: Vec<f32>,
}

impl From<SegmentationOutput> for SegmentationResult {
    fn from(output: SegmentationOutput) -> Self {
        let mut regions = HashMap::new();
        let mut region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)> = HashMap::new();

        let width = output.width;
        let height = output.height;

        for (i, &class_id) in output.class_map.iter().enumerate() {
            let x = (i % width as usize) as u32;
            let y = (i / width as usize) as u32;

            if let Some(class) = SegmentationClass::from_index(class_id) {
                let region = SegmentationRegion::from(class);
                regions.insert(i as u32, region);

                // Update bounds
                let entry = region_bounds.entry(region).or_insert((
                    x as f32 / width as f32,
                    y as f32 / height as f32,
                    x as f32 / width as f32,
                    y as f32 / height as f32,
                ));
                entry.0 = entry.0.min(x as f32 / width as f32);
                entry.1 = entry.1.min(y as f32 / height as f32);
                entry.2 = entry.2.max(x as f32 / width as f32);
                entry.3 = entry.3.max(y as f32 / height as f32);
            }
        }

        Self {
            regions,
            region_bounds,
            confidence: output.confidence_map,
        }
    }
}

// =============================================================================
// Combined ML Results
// =============================================================================

/// Complete ML analysis results
#[derive(Debug, Clone, Default)]
pub struct MLResults {
    // New structured outputs
    /// Face detection output
    pub face: Option<FaceDetectionOutput>,

    /// Depth estimation output
    pub depth: Option<DepthOutput>,

    /// Segmentation output
    pub segmentation_output: Option<SegmentationOutput>,

    // Legacy fields (for backward compatibility)
    /// Detected face landmarks (legacy)
    pub landmarks: Option<FaceLandmarks>,

    /// Depth map (legacy)
    pub depth_map: Option<Vec<f32>>,

    /// Segmentation results (legacy)
    pub segmentation: Option<SegmentationResult>,

    /// Face bounding box (legacy)
    pub face_bounds: Option<FaceBounds>,
}

impl MLResults {
    /// Create from new output types, populating legacy fields
    pub fn from_outputs(
        face: Option<FaceDetectionOutput>,
        depth: Option<DepthOutput>,
        segmentation: Option<SegmentationOutput>,
    ) -> Self {
        // Derive legacy fields
        let landmarks = face.as_ref().and_then(|f| {
            f.bounds.as_ref().map(|b| FaceLandmarks::from_yolo(&f.landmarks, b))
        });

        let face_bounds = face.as_ref().and_then(|f| f.bounds);

        let depth_map = depth.as_ref().map(|d| d.depth_map.clone());

        let segmentation_result = segmentation.clone().map(SegmentationResult::from);

        Self {
            face,
            depth,
            segmentation_output: segmentation,
            landmarks,
            depth_map,
            segmentation: segmentation_result,
            face_bounds,
        }
    }

    /// Check if any analysis was performed
    pub fn has_any(&self) -> bool {
        self.face.is_some() || self.depth.is_some() || self.segmentation_output.is_some()
    }

    /// Check if face was detected
    pub fn has_face(&self) -> bool {
        self.face.as_ref().map(|f| f.bounds.is_some()).unwrap_or(false)
    }

    /// Check if depth was estimated
    pub fn has_depth(&self) -> bool {
        self.depth.is_some()
    }

    /// Check if segmentation was performed
    pub fn has_segmentation(&self) -> bool {
        self.segmentation_output.is_some()
    }
}

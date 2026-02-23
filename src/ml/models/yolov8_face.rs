//! YOLOv8n-Face: Face Detection with 5 Landmarks
//!
//! # Model Details
//!
//! - **Source**: https://huggingface.co/akanametov/yolov8-face (`.pt` for re-export)
//!   or `onnx-community/yolov8n-face` for a pre-exported ONNX.
//! - **Size**: ~6 MB
//! - **Architecture**: YOLOv8 nano
//! - **Input**: RGB float32 NCHW, 0–1 normalized, default 640×640 (dynamic if re-exported)
//! - **Output shape**: `[1, 15, N]`  — channel-first, NOT `[1, N, 15]`
//!
//! # Output format (per detection column)
//!
//! ```text
//! [0]  cx          — box center-x in model input space
//! [1]  cy          — box center-y in model input space
//! [2]  w           — box width in model input space
//! [3]  h           — box height in model input space
//! [4]  confidence  — objectness × class score
//! [5]  lm0_x       — left eye x
//! [6]  lm0_y       — left eye y
//! [7]  lm1_x       — right eye x
//! [8]  lm1_y       — right eye y
//! [9]  lm2_x       — nose tip x
//! [10] lm2_y       — nose tip y
//! [11] lm3_x       — left mouth corner x
//! [12] lm3_y       — left mouth corner y
//! [13] lm4_x       — right mouth corner x
//! [14] lm4_y       — right mouth corner y
//! ```
//!
//! All spatial values are in **model input pixel space** (e.g. 0–640).
//! We convert them to normalized [0,1] coordinates relative to the original image
//! using the [`ResizeScale`] from preprocessing.

use crate::ml::{FaceBounds, FaceDetectionResult, FaceLandmarks, LandmarkRegion};
use crate::ml::preprocess::{PreprocessConfig, ResizeScale, preprocess};
use anyhow::{anyhow, Result};
use image::DynamicImage;
use ort::session::builder::GraphOptimizationLevel;
use std::cell::RefCell;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default model input resolution (used when the ONNX has fixed dims)
const DEFAULT_INPUT_SIZE: u32 = 640;

/// Confidence threshold below which detections are discarded
const CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Number of output channels per detection
const DET_CHANNELS: usize = 15;

// ─────────────────────────────────────────────────────────────────────────────
// Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Face detector using YOLOv8n-Face ONNX model.
pub struct YoloV8FaceDetector {
    session: RefCell<ort::session::Session>,
    input_name: String,
    output_name: String,
    /// Model input dimensions resolved at load time (falls back to 640×640)
    model_w: u32,
    model_h: u32,
}

impl YoloV8FaceDetector {
    /// Load the ONNX model and resolve its input dimensions.
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let input_name  = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        // Try to read static input dims from the model metadata.
        // Dynamic axes appear as -1; we keep the default for those.
        let (model_w, model_h) = resolve_input_dims(&session, DEFAULT_INPUT_SIZE);

        log::info!(
            "YOLOv8n-Face loaded: input={}x{}, input_name='{}'",
            model_w, model_h, input_name
        );

        Ok(Self {
            session: RefCell::new(session),
            input_name,
            output_name,
            model_w,
            model_h,
        })
    }

    /// Detect the most confident face in `image`.
    ///
    /// Returns normalized coordinates (0–1) relative to the *original* image size.
    pub fn detect(&self, image: &DynamicImage) -> Result<FaceDetectionResult> {
        let cfg = PreprocessConfig::unit(self.model_w, self.model_h);
        let (tensor, scale) = preprocess(image, &cfg)?;

        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, self.model_h as i64, self.model_w as i64],
            tensor,
        ))?;

        let mut session = self.session.borrow_mut();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow!("Model produced no output"))?;

        let (shape, data) = output.try_extract_tensor::<f32>()?;

        parse_detections(&shape, &data, &scale)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the YOLOv8-Face output tensor into a [`FaceDetectionResult`].
///
/// Expected shape: `[1, 15, N]` — channel-first layout.
/// We select the detection with the highest confidence above the threshold.
fn parse_detections(
    shape: &[i64],
    data: &[f32],
    scale: &ResizeScale,
) -> Result<FaceDetectionResult> {
    // Validate and extract dimensions
    // YOLOv8 ONNX output is [batch=1, channels=15, num_detections]
    let (channels, num_det) = match shape.len() {
        3 => (shape[1] as usize, shape[2] as usize),
        2 => (shape[0] as usize, shape[1] as usize), // batch squeezed out
        _ => return Err(anyhow!("Unexpected output shape: {:?}", shape)),
    };

    if channels < DET_CHANNELS {
        return Err(anyhow!(
            "Expected ≥{} output channels, got {}",
            DET_CHANNELS, channels
        ));
    }

    // Each feature i for detection j lives at: data[i * num_det + j]
    let feat = |i: usize, j: usize| -> f32 {
        let idx = i * num_det + j;
        if idx < data.len() { data[idx] } else { 0.0 }
    };

    let mut best_j    = None;
    let mut best_conf = CONFIDENCE_THRESHOLD;

    for j in 0..num_det {
        let conf = feat(4, j);
        if conf > best_conf {
            best_conf = conf;
            best_j = Some(j);
        }
    }

    let Some(j) = best_j else {
        return Ok(FaceDetectionResult::default());
    };

    // YOLOv8 box format: cx, cy, w, h in model-input pixel space
    let cx = feat(0, j);
    let cy = feat(1, j);
    let bw = feat(2, j);
    let bh = feat(3, j);

    let x1 = cx - bw * 0.5;
    let y1 = cy - bh * 0.5;
    let x2 = cx + bw * 0.5;
    let y2 = cy + bh * 0.5;

    // Convert box from model-input space to normalized original-image coords
    let (nx1, ny1, nx2, ny2) = scale.bbox_to_normalized(x1, y1, x2, y2);
    let face_bounds = FaceBounds {
        x: nx1.clamp(0.0f32, 1.0f32),
        y: ny1.clamp(0.0f32, 1.0f32),
        width:  (nx2 - nx1).clamp(0.0f32, 1.0f32),
        height: (ny2 - ny1).clamp(0.0f32, 1.0f32),
    };

    // Extract the 5 real landmarks from model output
    let raw_landmarks: Vec<(f32, f32)> = (0..5)
        .map(|k| {
            let lx = feat(5 + k * 2,     j);
            let ly = feat(5 + k * 2 + 1, j);
            // Convert from model-input space to normalized original-image coords
            let (nx, ny) = scale.to_normalized(lx, ly);
            (nx.clamp(0.0f32, 1.0f32), ny.clamp(0.0f32, 1.0f32))
        })
        .collect();

    // Build structured landmarks from the 5-point set
    let landmarks = build_face_landmarks(&raw_landmarks, &face_bounds);

    Ok(FaceDetectionResult {
        bounds: Some(face_bounds),
        landmarks: Some(landmarks),
        confidence: best_conf,
    })
}

/// Build a [`FaceLandmarks`] struct from the 5 raw YOLOv8-Face landmark points.
///
/// Landmark order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
/// Region sizes are estimated as a fraction of the face bounding box.
fn build_face_landmarks(
    pts: &[(f32, f32)],
    bounds: &FaceBounds,
) -> FaceLandmarks {
    debug_assert_eq!(pts.len(), 5, "Expected exactly 5 landmarks");

    let eye_w  = bounds.width  * 0.12;
    let eye_h  = bounds.height * 0.07;
    let nose_w = bounds.width  * 0.10;
    let nose_h = bounds.height * 0.10;
    let lip_w  = bounds.width  * 0.18;
    let lip_h  = bounds.height * 0.06;

    FaceLandmarks {
        points: pts.to_vec(),
        left_eye: Some(LandmarkRegion {
            center_x: pts[0].0,
            center_y: pts[0].1,
            width: eye_w,
            height: eye_h,
        }),
        right_eye: Some(LandmarkRegion {
            center_x: pts[1].0,
            center_y: pts[1].1,
            width: eye_w,
            height: eye_h,
        }),
        nose: Some(LandmarkRegion {
            center_x: pts[2].0,
            center_y: pts[2].1,
            width: nose_w,
            height: nose_h,
        }),
        lips: Some(LandmarkRegion {
            center_x: (pts[3].0 + pts[4].0) * 0.5,
            center_y: (pts[3].1 + pts[4].1) * 0.5,
            width:  lip_w,
            height: lip_h,
        }),
        face_outline: vec![],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dimension resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt to read static input H/W from the session metadata.
/// Falls back to `default_size × default_size` if the axes are dynamic or unavailable.

fn resolve_input_dims(
    _session: &ort::session::Session,
    default_size: u32,
) -> (u32, u32) {
    // ORT 2.x does not expose a stable API to read symbolic input dims before inference.
    // We use well-known defaults; dynamic-axes exports accept any size anyway.
    (default_size, default_size)
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder face detector — used when the ONNX model is not yet available.
/// Returns a centered face estimate with geometrically derived landmarks.
pub struct PlaceholderFaceDetector;

impl PlaceholderFaceDetector {
    pub fn detect(&self, _image: &DynamicImage) -> Result<FaceDetectionResult> {
        let bounds = FaceBounds { x: 0.25, y: 0.15, width: 0.5, height: 0.6 };
        let cx = bounds.x + bounds.width  * 0.5;
        let cy = bounds.y + bounds.height * 0.5;
        let (w, h) = (bounds.width, bounds.height);

        let pts = vec![
            (cx - w * 0.15, cy - h * 0.10), // left eye
            (cx + w * 0.15, cy - h * 0.10), // right eye
            (cx,            cy + h * 0.05), // nose tip
            (cx - w * 0.12, cy + h * 0.22), // left mouth
            (cx + w * 0.12, cy + h * 0.22), // right mouth
        ];

        let landmarks = build_face_landmarks(&pts, &bounds);

        Ok(FaceDetectionResult {
            bounds: Some(bounds),
            landmarks: Some(landmarks),
            confidence: 0.95,
        })
    }
}

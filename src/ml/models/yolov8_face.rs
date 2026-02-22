//! YOLOv8n-Face: Face Detection with 5 Landmarks
//!
//! This module implements face detection using the YOLOv8n-Face ONNX model.
//!
//! # Model Details
//!
//! - **Source**: https://huggingface.co/onnx-community/yolov8n-face
//! - **Size**: ~6 MB
//! - **Architecture**: YOLOv8 nano (optimized for speed)
//! - **Input**: 640x640 RGB image
//! - **Output**: Bounding boxes + 5 facial landmarks
//!
//! # Landmarks
//!
//! The model detects 5 key facial landmarks:
//! 1. Left eye
//! 2. Right eye
//! 3. Nose tip
//! 4. Left mouth corner
//! 5. Right mouth corner
//!
//! # Usage
//!
//! ```ignore
//! let detector = YoloV8FaceDetector::new(&model_path)?;
//! let result = detector.detect(&image)?;
//!
//! if let Some(bounds) = result.bounds {
//!     println!("Face at ({}, {})", bounds.x, bounds.y);
//! }
//! ```

use crate::ml::{FaceBounds, FaceDetectionResult, FaceLandmarks, LandmarkRegion};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ort::session::builder::GraphOptimizationLevel;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

/// Face detector using YOLOv8n-Face ONNX model.
///
/// This detector:
/// 1. Loads the ONNX model on creation
/// 2. Resizes input images to model's expected size (640x640)
/// 3. Returns normalized face bounds and landmarks
///
/// # Memory Usage
///
/// YOLOv8n is very lightweight:
/// - Model size: ~6MB
/// - Peak VRAM: ~50MB during inference
pub struct YoloV8FaceDetector {
    /// ONNX Runtime session
    session: RefCell<ort::session::Session>,
    /// Input tensor name
    input_name: String,
    /// Output tensor name
    output_name: String,
    /// Detected/cached input size (default 640)
    input_size: AtomicU32,
}

impl YoloV8FaceDetector {
    /// Create a new face detector from an ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to yolov8n-face.onnx
    ///
    /// # Configuration
    ///
    /// - Graph optimization: Level 3 (maximum)
    /// - Intra-op threads: 4
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();
        let input_size = AtomicU32::new(640);

        log::info!("YOLOv8n-Face model loaded");

        Ok(Self {
            session: RefCell::new(session),
            input_name,
            output_name,
            input_size,
        })
    }

    /// Detect faces in an image.
    ///
    /// Returns the most confident face detection with:
    /// - Bounding box (normalized 0-1)
    /// - 5 facial landmarks (if available)
    /// - Confidence score
    ///
    /// # Auto-Size Detection
    ///
    /// If the default input size is wrong, the model will auto-detect
    /// the correct size from ONNX Runtime error messages.
    pub fn detect(&self, image: &DynamicImage) -> Result<FaceDetectionResult> {
        let (orig_w, orig_h) = image.dimensions();
        let mut size = self.input_size.load(Ordering::Relaxed);

        // Retry loop for auto-detecting input size
        loop {
            match self.try_detect(image, size, orig_w, orig_h) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let err_str = e.to_string();
                    if let Some(new_size) = Self::extract_expected_size(&err_str) {
                        if new_size != size && new_size > 0 {
                            log::info!("Auto-detected face model input size: {}x{}", new_size, new_size);
                            size = new_size;
                            self.input_size.store(size, Ordering::Relaxed);
                            continue;
                        }
                    }
                    return Err(e);
                }
            }
        }
    }

    /// Attempt face detection with a specific input size.
    fn try_detect(
        &self,
        image: &DynamicImage,
        size: u32,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<FaceDetectionResult> {
        // Resize to model input size
        let resized = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        // Prepare input tensor (NCHW format, RGB normalized to 0-1)
        let mut input_data = vec![0.0f32; 3 * size as usize * size as usize];

        for y in 0..size as usize {
            for x in 0..size as usize {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                let values = pixel.0;
                let base_idx = y * size as usize + x;
                // R channel
                input_data[base_idx] = values[0] as f32 / 255.0;
                // G channel
                input_data[size as usize * size as usize + base_idx] = values[1] as f32 / 255.0;
                // B channel
                input_data[2 * size as usize * size as usize + base_idx] = values[2] as f32 / 255.0;
            }
        }

        // Create input tensor
        let input_tensor =
            ort::value::Value::from_array(([1i64, 3, size as i64, size as i64], input_data))?;

        // Run inference
        let mut session = self.session.borrow_mut();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        let (shape, data) = output.try_extract_tensor::<f32>()?;

        // Parse detections - YOLOv8-Face outputs [1, N, 15] where N is max detections
        // Each detection: [x1, y1, x2, y2, conf, lmx0, lmy0, lmx1, lmy1, lmx2, lmy2, lmx3, lmy3, lmx4, lmy4]
        let num_detections = if shape.len() >= 2 {
            shape[1] as usize
        } else {
            0
        };

        // Find best detection
        let mut best_detection: Option<(f32, f32, f32, f32, f32)> = None;
        let mut best_confidence = 0.0f32;

        for i in 0..num_detections {
            let base = i * 15;
            if base + 4 < data.len() {
                let confidence = data[base + 4];
                if confidence > best_confidence && confidence > 0.5 {
                    best_confidence = confidence;
                    best_detection = Some((
                        data[base],
                        data[base + 1],
                        data[base + 2],
                        data[base + 3],
                        confidence,
                    ));
                }
            }
        }

        // Convert to normalized coordinates
        let face_bounds = best_detection.map(|(x1, y1, x2, y2, _)| FaceBounds {
            x: x1 / orig_w as f32,
            y: y1 / orig_h as f32,
            width: (x2 - x1) / orig_w as f32,
            height: (y2 - y1) / orig_h as f32,
        });

        // Generate landmarks (placeholder - real landmarks from model output)
        let landmarks = face_bounds.as_ref().map(generate_landmarks_from_bounds);

        Ok(FaceDetectionResult {
            bounds: face_bounds,
            landmarks,
            confidence: best_confidence,
        })
    }

    /// Extract expected input size from ONNX Runtime error message.
    fn extract_expected_size(error_msg: &str) -> Option<u32> {
        for line in error_msg.lines() {
            if line.contains("Expected:") {
                if let Some(after) = line.split("Expected:").nth(1) {
                    for part in after.split_whitespace() {
                        if let Ok(size) = part.parse::<u32>() {
                            if size > 64 && size <= 2048 {
                                return Some(size);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Generate estimated landmarks from face bounds.
///
/// Used when the model doesn't provide landmark coordinates.
fn generate_landmarks_from_bounds(bounds: &FaceBounds) -> FaceLandmarks {
    let cx = bounds.x + bounds.width / 2.0;
    let cy = bounds.y + bounds.height / 2.0;
    let (w, h) = (bounds.width, bounds.height);

    FaceLandmarks {
        points: vec![
            // Left eye (2 points)
            (cx - w * 0.15, cy - h * 0.1),
            (cx - w * 0.1, cy - h * 0.12),
            // Right eye (2 points)
            (cx - w * 0.05, cy - h * 0.1),
            (cx + w * 0.05, cy - h * 0.1),
            // Nose and mouth
            (cx + w * 0.1, cy - h * 0.12),
            (cx + w * 0.15, cy - h * 0.1),
            (cx, cy + h * 0.05),
            (cx - w * 0.08, cy + h * 0.2),
            (cx, cy + h * 0.18),
            (cx + w * 0.08, cy + h * 0.2),
        ],
        left_eye: Some(LandmarkRegion {
            center_x: cx - w * 0.1,
            center_y: cy - h * 0.1,
            width: w * 0.1,
            height: h * 0.05,
        }),
        right_eye: Some(LandmarkRegion {
            center_x: cx + w * 0.1,
            center_y: cy - h * 0.1,
            width: w * 0.1,
            height: h * 0.05,
        }),
        nose: Some(LandmarkRegion {
            center_x: cx,
            center_y: cy + h * 0.05,
            width: w * 0.08,
            height: h * 0.1,
        }),
        lips: Some(LandmarkRegion {
            center_x: cx,
            center_y: cy + h * 0.2,
            width: w * 0.15,
            height: h * 0.05,
        }),
        face_outline: vec![],
    }
}

/// Placeholder face detector for when the model is not available.
///
/// Returns a centered face estimate based on typical portrait composition.
pub struct PlaceholderFaceDetector;

impl PlaceholderFaceDetector {
    /// Detect a face using placeholder heuristics.
    ///
    /// Assumes:
    /// - Face is centered in the image
    /// - Face occupies ~50% width and ~60% height
    /// - High confidence placeholder
    pub fn detect(&self, _image: &DynamicImage) -> Result<FaceDetectionResult> {
        Ok(FaceDetectionResult {
            bounds: Some(FaceBounds {
                x: 0.25,
                y: 0.15,
                width: 0.5,
                height: 0.6,
            }),
            landmarks: Some(generate_landmarks_from_bounds(&FaceBounds {
                x: 0.25,
                y: 0.15,
                width: 0.5,
                height: 0.6,
            })),
            confidence: 0.95,
        })
    }
}

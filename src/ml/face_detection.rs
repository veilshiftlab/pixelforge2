//! Face detection using ONNX Runtime

use super::{FaceBounds, FaceDetectionResult, FaceLandmarks, LandmarkRegion};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ort::session::builder::GraphOptimizationLevel;

/// Face detector using ONNX model
pub struct FaceDetector {
    session: ort::session::Session,
    input_name: String,
    output_name: String,
}

impl FaceDetector {
    /// Create a new face detector
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Get input/output names from model
        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        Ok(Self { session, input_name, output_name })
    }

    /// Detect faces in an image
    pub fn detect(&mut self, image: &DynamicImage) -> Result<FaceDetectionResult> {
        let (orig_w, orig_h) = image.dimensions();

        // Preprocess
        let target_size = 160u32;
        let resized = image.resize(target_size, target_size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        // Create input tensor as flat vec (NCHW format)
        let mut input_data = vec![0.0f32; 1 * 3 * target_size as usize * target_size as usize];
        
        for y in 0..target_size as usize {
            for x in 0..target_size as usize {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                let values = pixel.0;
                let base_idx = y * target_size as usize + x;
                // Channel 0
                input_data[base_idx] = values[0] as f32 / 255.0;
                // Channel 1
                input_data[target_size as usize * target_size as usize + base_idx] = values[1] as f32 / 255.0;
                // Channel 2
                input_data[2 * target_size as usize * target_size as usize + base_idx] = values[2] as f32 / 255.0;
            }
        }

        // Create input value
        let input_tensor = ort::value::Value::from_array((
            [1, 3, target_size as i64, target_size as i64],
            input_data.clone(),
        ))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            &self.input_name => input_tensor
        ])?;

        // Get output
        let output = outputs.get(&self.output_name)
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        // Extract tensor data
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        
        // Parse detections - shape is typically [batch, num_detections, detection_data]
        let num_detections = if shape.len() >= 2 { shape[1] as usize } else { 0 };

        let mut best_detection: Option<(f32, f32, f32, f32, f32)> = None;
        let mut best_confidence = 0.0f32;

        for i in 0..num_detections {
            let base = i * 15; // Each detection has multiple values
            if base + 4 < data.len() {
                let confidence = data[base + 4];
                if confidence > best_confidence && confidence > 0.5 {
                    best_confidence = confidence;
                    let x1 = data[base];
                    let y1 = data[base + 1];
                    let x2 = data[base + 2];
                    let y2 = data[base + 3];
                    best_detection = Some((x1, y1, x2, y2, confidence));
                }
            }
        }

        // Convert to original image coordinates
        let face_bounds = best_detection.map(|(x1, y1, x2, y2, _conf)| {
            FaceBounds {
                x: x1 / orig_w as f32,
                y: y1 / orig_h as f32,
                width: (x2 - x1) / orig_w as f32,
                height: (y2 - y1) / orig_h as f32,
            }
        });

        let landmarks = face_bounds.as_ref().map(|bounds| {
            generate_landmarks_from_bounds(bounds)
        });

        Ok(FaceDetectionResult {
            bounds: face_bounds,
            landmarks,
            confidence: best_confidence,
        })
    }
}

/// Generate approximate landmarks from face bounds
fn generate_landmarks_from_bounds(bounds: &FaceBounds) -> FaceLandmarks {
    let cx = bounds.x + bounds.width / 2.0;
    let cy = bounds.y + bounds.height / 2.0;
    let w = bounds.width;
    let h = bounds.height;

    FaceLandmarks {
        points: vec![
            (cx - w * 0.15, cy - h * 0.1),
            (cx - w * 0.1, cy - h * 0.12),
            (cx - w * 0.05, cy - h * 0.1),
            (cx + w * 0.05, cy - h * 0.1),
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

/// Placeholder detector for when models aren't available
pub struct PlaceholderDetector;

impl PlaceholderDetector {
    pub fn detect(&self, image: &DynamicImage) -> Result<FaceDetectionResult> {
        let (width, height) = image.dimensions();
        let _ = (width, height);

        Ok(FaceDetectionResult {
            bounds: Some(FaceBounds {
                x: 0.25,
                y: 0.15,
                width: 0.5,
                height: 0.6,
            }),
            landmarks: Some(generate_landmarks_from_bounds(
                &FaceBounds { x: 0.25, y: 0.15, width: 0.5, height: 0.6 },
            )),
            confidence: 0.95,
        })
    }
}

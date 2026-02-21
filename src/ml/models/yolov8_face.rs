//! YOLOv8n-Face Face Detection Model
//!
//! Provides face detection with 5 facial landmarks using YOLOv8n-Face.
//! Supports dynamic input resolution (any size divisible by 32).

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};

use crate::ml::{
    FaceBounds, FaceDetectionOutput, FaceLandmarkIndex,
    FaceDetectionConfig, SessionManager, ModelType,
};

/// YOLOv8n-Face face detector
pub struct YOLOv8FaceDetector {
    session_manager: std::sync::Arc<SessionManager>,
    model_path: std::path::PathBuf,
}

impl YOLOv8FaceDetector {
    /// Create a new detector with session manager
    pub fn new(
        session_manager: std::sync::Arc<SessionManager>,
        model_path: std::path::PathBuf,
    ) -> Self {
        Self {
            session_manager,
            model_path,
        }
    }

    /// Detect faces in an image
    pub fn detect(
        &self,
        image: &DynamicImage,
        config: &FaceDetectionConfig,
    ) -> Result<FaceDetectionOutput> {
        let session = self.session_manager
            .get_or_load(&self.model_path, ModelType::FaceDetection)
            .context("Failed to load face detection model")?;

        let (orig_width, orig_height) = image.dimensions();

        // Prepare input with letterbox resize (maintains aspect ratio)
        let (input_tensor, scale, pad_x, pad_y, input_size) =
            prepare_yolo_input(image)?;

        // Run inference
        let mut sess = session.session.lock().unwrap();
        let outputs = sess.run(ort::inputs![
            &session.input_name => input_tensor
        ]).context("Face detection inference failed")?;

        // Get output tensor
        let output = outputs
            .get(&session.output_name)
            .context("No output from face detection model")?;

        let (shape, data) = output.try_extract_tensor::<f32>()
            .context("Failed to extract face detection output")?;

        // Parse YOLOv8 output format
        // Output shape: [1, 20, N] where N is number of proposals
        // 20 = 4 (bbox) + 1 (conf) + 15 (5 landmarks * 3)
        let detections = parse_yolo_output(
            &data,
            &shape,
            config.confidence_threshold,
            config.iou_threshold,
        )?;

        // Scale detections back to original image coordinates
        let result = scale_detections(
            detections,
            scale,
            pad_x,
            pad_y,
            orig_width,
            orig_height,
            input_size,
        );

        Ok(result)
    }
}

/// Prepare image for YOLOv8 input with letterbox resize
fn prepare_yolo_input(
    image: &DynamicImage,
) -> Result<(ort::value::Value, f32, f32, f32, u32)> {
    let (width, height) = image.dimensions();

    // Calculate target size (divisible by 32, max 640 for efficiency)
    let max_size = 640u32;
    let stride = 32u32;

    // Calculate scale to fit within max_size
    let scale = if width > height {
        max_size as f32 / width as f32
    } else {
        max_size as f32 / height as f32
    };

    let new_width = ((width as f32 * scale) / stride as f32).round() as u32 * stride;
    let new_height = ((height as f32 * scale) / stride as f32).round() as u32 * stride;

    // Resize image
    let resized = image.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);
    let rgba = resized.to_rgba8();

    // Create input tensor (NCHW format)
    let mut input_data = vec![0.0f32; 1 * 3 * (new_width * new_height) as usize];

    for y in 0..new_height as usize {
        for x in 0..new_width as usize {
            let pixel = rgba.get_pixel(x as u32, y as u32);
            let values = pixel.0;
            let base_idx = y * new_width as usize + x;

            // Normalize to 0-1 and arrange in CHW order
            input_data[base_idx] = values[0] as f32 / 255.0;
            input_data[new_width as usize * new_height as usize + base_idx] = values[1] as f32 / 255.0;
            input_data[2 * new_width as usize * new_height as usize + base_idx] = values[2] as f32 / 255.0;
        }
    }

    let input_tensor = ort::value::Value::from_array((
        [1i64, 3, new_height as i64, new_width as i64],
        input_data,
    )).context("Failed to create input tensor")?;

    // Calculate padding (for letterbox)
    let pad_x = 0.0f32; // No padding needed with resize_exact
    let pad_y = 0.0f32;

    Ok((input_tensor, scale, pad_x, pad_y, new_width.max(new_height)))
}

/// Parsed YOLO detection
struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    confidence: f32,
    landmarks: Vec<(f32, f32)>,
}

/// Parse YOLOv8-Face output tensor
fn parse_yolo_output(
    data: &[f32],
    shape: &[i64],
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    // YOLOv8-Face output: [1, 20, N] or [1, N, 20]
    let (num_attrs, num_proposals) = if shape.len() == 3 {
        if shape[1] == 20 {
            (shape[1] as usize, shape[2] as usize)
        } else {
            (shape[2] as usize, shape[1] as usize)
        }
    } else {
        return Ok(Vec::new());
    };

    let mut detections = Vec::new();

    for i in 0..num_proposals {
        // Get detection data
        let base = if shape[1] == 20 {
            i * num_attrs // [1, 20, N] format
        } else {
            i // [1, N, 20] format - data is interleaved
        };

        // Extract confidence
        let conf = if shape[1] == 20 {
            data[4 * num_proposals + i]
        } else {
            data[base + 4]
        };

        if conf < conf_threshold {
            continue;
        }

        // Extract bounding box (center format)
        let (cx, cy, w, h) = if shape[1] == 20 {
            (
                data[i],
                data[num_proposals + i],
                data[2 * num_proposals + i],
                data[3 * num_proposals + i],
            )
        } else {
            (
                data[base],
                data[base + 1],
                data[base + 2],
                data[base + 3],
            )
        };

        // Extract 5 landmarks (each has x, y, and conf)
        let mut landmarks = Vec::new();
        for lm in 0..5 {
            let (lm_x, lm_y) = if shape[1] == 20 {
                (
                    data[(5 + lm * 3) * num_proposals + i],
                    data[(6 + lm * 3) * num_proposals + i],
                )
            } else {
                (
                    data[base + 5 + lm * 3],
                    data[base + 6 + lm * 3],
                )
            };
            landmarks.push((lm_x, lm_y));
        }

        detections.push(Detection {
            x: cx - w / 2.0, // Convert to top-left format
            y: cy - h / 2.0,
            w,
            h,
            confidence: conf,
            landmarks,
        });
    }

    // Apply Non-Maximum Suppression
    apply_nms(&mut detections, iou_threshold);

    Ok(detections)
}

/// Apply Non-Maximum Suppression
fn apply_nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    // Sort by confidence
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; detections.len()];

    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }

        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }

            let iou = compute_iou(&detections[i], &detections[j]);
            if iou > iou_threshold {
                keep[j] = false;
            }
        }
    }

    // Retain only kept detections
    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Compute Intersection over Union
fn compute_iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union = area_a + area_b - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Scale detections back to original image coordinates
fn scale_detections(
    mut detections: Vec<Detection>,
    scale: f32,
    _pad_x: f32,
    _pad_y: f32,
    orig_width: u32,
    orig_height: u32,
    _input_size: u32,
) -> FaceDetectionOutput {
    if detections.is_empty() {
        return FaceDetectionOutput::default();
    }

    // Take best detection
    let best = detections.remove(0);

    // Scale coordinates to original image
    let inv_scale = 1.0 / scale;

    // Normalize to 0-1
    let bounds = FaceBounds {
        x: (best.x * inv_scale / orig_width as f32).clamp(0.0, 1.0),
        y: (best.y * inv_scale / orig_height as f32).clamp(0.0, 1.0),
        width: (best.w * inv_scale / orig_width as f32).clamp(0.0, 1.0),
        height: (best.h * inv_scale / orig_height as f32).clamp(0.0, 1.0),
    };

    // Scale and normalize landmarks
    let landmarks: Vec<(f32, f32)> = best.landmarks
        .iter()
        .map(|&(lx, ly)| {
            (
                (lx * inv_scale / orig_width as f32).clamp(0.0, 1.0),
                (ly * inv_scale / orig_height as f32).clamp(0.0, 1.0),
            )
        })
        .collect();

    FaceDetectionOutput {
        bounds: Some(bounds),
        landmarks,
        confidence: best.confidence,
    }
}

//! Semantic segmentation using ONNX Runtime

use super::{SegmentationRegion, SegmentationResult};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use std::collections::HashMap;

/// Semantic segmenter using ONNX model
pub struct Segmenter {
    #[allow(dead_code)]
    session: ort::session::Session,
}

impl Segmenter {
    /// Create a new segmenter
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::SessionBuilder::new()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Segment an image into semantic regions
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        let (orig_w, orig_h) = image.dimensions();

        // SegFormer typically expects 512x512
        let size = 512u32;
        let resized = image.resize(size, size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        // Create input tensor (NCHW format)
        let mut input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));

        for y in 0..size as usize {
            for x in 0..size as usize {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                let values = pixel.0;
                // Normalize using ImageNet mean/std
                input[[0, 0, y, x]] = (values[0] as f32 / 255.0 - 0.485) / 0.229;
                input[[0, 1, y, x]] = (values[1] as f32 / 255.0 - 0.456) / 0.224;
                input[[0, 2, y, x]] = (values[2] as f32 / 255.0 - 0.406) / 0.225;
            }
        }

        // Run inference using ort 2.0 API
        let input_values = vec![ort::value::Value::from_array(input)?];
        let outputs = self.session.run(input_values)?;

        // Get output
        let output = outputs.get(0)
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        let view = output.try_extract_tensor::<f32>()?;
        let seg_data = view.view();

        // Parse segmentation output
        // Usually [1, num_classes, H, W] or [1, H, W]
        let shape = seg_data.shape();
        let (seg_h, seg_w) = match shape.len() {
            4 => (shape[2], shape[3]),
            3 => (shape[1], shape[2]),
            _ => (size as usize, size as usize),
        };

        let num_classes = match shape.len() {
            4 => shape[1],
            _ => 20, // Default for face parsing
        };

        let mut regions = HashMap::new();
        let mut confidence = vec![0.0f32; (orig_w * orig_h) as usize];
        let mut region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)> = HashMap::new();

        for orig_y in 0..orig_h as usize {
            for orig_x in 0..orig_w as usize {
                // Map to segmentation map coordinates
                let seg_y = (orig_y * seg_h / orig_h as usize).min(seg_h - 1);
                let seg_x = (orig_x * seg_w / orig_w as usize).min(seg_w - 1);

                // Find class with highest probability
                let mut best_class = 0;
                let mut best_prob = 0.0f32;

                for c in 0..num_classes {
                    let idx: [usize; 4] = [0, c, seg_y, seg_x];
                    let prob = if c < shape.get(1).copied().unwrap_or(1) {
                        seg_data[&idx]
                    } else {
                        0.0
                    };

                    if prob > best_prob {
                        best_prob = prob;
                        best_class = c;
                    }
                }

                let idx = (orig_y * orig_w as usize + orig_x) as u32;
                let region = map_class_to_region(best_class);
                regions.insert(idx, region);
                confidence[(orig_y * orig_w as usize + orig_x)] = best_prob;

                // Update region bounds
                let entry = region_bounds.entry(region).or_insert((
                    orig_x as f32 / orig_w as f32,
                    orig_y as f32 / orig_h as f32,
                    orig_x as f32 / orig_w as f32,
                    orig_y as f32 / orig_h as f32,
                ));
                entry.0 = entry.0.min(orig_x as f32 / orig_w as f32);
                entry.1 = entry.1.min(orig_y as f32 / orig_h as f32);
                entry.2 = entry.2.max(orig_x as f32 / orig_w as f32);
                entry.3 = entry.3.max(orig_y as f32 / orig_h as f32);
            }
        }

        Ok(SegmentationResult {
            regions,
            region_bounds,
            confidence,
        })
    }
}

/// Map semantic class ID to region type
fn map_class_to_region(class_id: usize) -> SegmentationRegion {
    // Face parsing classes (typical BiSeNet format)
    match class_id {
        0 => SegmentationRegion::Background,
        1..=2 => SegmentationRegion::Face,    // skin
        3..=8 => SegmentationRegion::Eyes,    // eyebrows, eyes
        9..=10 => SegmentationRegion::Eyes,   // eyes
        11..=14 => SegmentationRegion::Nose,  // nose
        15..=16 => SegmentationRegion::Lips,  // lips, mouth
        17 => SegmentationRegion::Hair,
        18 => SegmentationRegion::Ears,
        19 => SegmentationRegion::Neck,
        20 => SegmentationRegion::Clothing,
        _ => SegmentationRegion::Background,
    }
}

/// Placeholder segmenter for when models aren't available
pub struct PlaceholderSegmenter;

impl PlaceholderSegmenter {
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        let (width, height) = image.dimensions();
        let mut regions = HashMap::new();

        // Simple segmentation based on position
        let face_y_start = (height as f32 * 0.15) as u32;
        let face_y_end = (height as f32 * 0.75) as u32;
        let face_x_start = (width as f32 * 0.25) as u32;
        let face_x_end = (width as f32 * 0.75) as u32;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as u32;

                let region = if y >= face_y_start && y <= face_y_end && x >= face_x_start && x <= face_x_end {
                    // Within face region
                    let rel_y = (y - face_y_start) as f32 / (face_y_end - face_y_start).max(1) as f32;
                    let rel_x = (x - face_x_start) as f32 / (face_x_end - face_x_start).max(1) as f32;

                    // Eye region (upper part)
                    if rel_y > 0.25 && rel_y < 0.4 {
                        if rel_x > 0.2 && rel_x < 0.4 {
                            SegmentationRegion::Eyes
                        } else if rel_x > 0.6 && rel_x < 0.8 {
                            SegmentationRegion::Eyes
                        } else {
                            SegmentationRegion::Face
                        }
                    }
                    // Nose region (middle)
                    else if rel_y > 0.4 && rel_y < 0.6 && rel_x > 0.35 && rel_x < 0.65 {
                        SegmentationRegion::Nose
                    }
                    // Lips region (lower middle)
                    else if rel_y > 0.65 && rel_y < 0.8 && rel_x > 0.3 && rel_x < 0.7 {
                        SegmentationRegion::Lips
                    }
                    else {
                        SegmentationRegion::Face
                    }
                } else if y < face_y_start {
                    // Hair region (top)
                    SegmentationRegion::Hair
                } else {
                    // Background
                    SegmentationRegion::Background
                };

                regions.insert(idx, region);
            }
        }

        Ok(SegmentationResult {
            regions,
            region_bounds: HashMap::new(),
            confidence: vec![0.9; (width * height) as usize],
        })
    }
}

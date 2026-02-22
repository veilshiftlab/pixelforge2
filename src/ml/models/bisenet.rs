//! BiSeNet: Face Parsing / Semantic Segmentation
//!
//! This module implements face parsing using the BiSeNet ONNX model.
//!
//! # Model Details
//!
//! - **Source**: https://huggingface.co/szhubel/face-parsing-bisenet
//! - **Size**: ~48 MB
//! - **Architecture**: BiSeNet (Bilateral Segmentation Network)
//! - **Input**: 512x512 RGB image
//! - **Output**: Per-pixel class labels (19 classes)
//!
//! # Segmentation Classes
//!
//! BiSeNet outputs 19 semantic classes for face parsing:
//!
//! | ID | Class | ID | Class |
//! |----|-------|-----|-------|
//! | 0 | Background | 10 | Nose |
//! | 1 | Skin | 11 | Inner Mouth |
//! | 2 | Left Eyebrow | 12 | Upper Lip |
//! | 3 | Right Eyebrow | 13 | Lower Lip |
//! | 4 | Left Eye | 14 | Neck |
//! | 5 | Right Eye | 15 | Hair |
//! | 6 | Eyeglasses | 16 | Hat |
//! | 7 | Left Ear | 17 | Left Earring |
//! | 8 | Right Ear | 18 | Right Earring |
//! | 9 | Earring | | |
//!
//! # Usage
//!
//! ```ignore
//! let segmenter = BiSeNetSegmenter::new(&model_path)?;
//! let result = segmenter.segment(&image)?;
//!
//! // Get region for a pixel
//! let region = result.regions.get(&pixel_idx);
//! ```

use crate::ml::{SegmentationRegion, SegmentationResult};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ort::session::builder::GraphOptimizationLevel;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Face parsing segmenter using BiSeNet ONNX model.
///
/// This segmenter:
/// 1. Loads the ONNX model on creation
/// 2. Resizes input images to model's expected size
/// 3. Returns per-pixel region labels
///
/// # Memory Usage
///
/// - Model: ~48MB
/// - Peak VRAM: ~150MB during inference
pub struct BiSeNetSegmenter {
    /// ONNX Runtime session
    session: RefCell<ort::session::Session>,
    /// Input tensor name
    input_name: String,
    /// Output tensor name
    output_name: String,
    /// Cached input size (auto-detected)
    input_size: AtomicU32,
}

impl BiSeNetSegmenter {
    /// Create a new segmenter from an ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to bisenet.onnx
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

        // Default size for BiSeNet, will be auto-detected
        let input_size = AtomicU32::new(512);

        log::info!("BiSeNet model loaded (will auto-detect input size)");

        Ok(Self {
            session: RefCell::new(session),
            input_name,
            output_name,
            input_size,
        })
    }

    /// Segment an image into semantic regions.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image (any size)
    ///
    /// # Returns
    ///
    /// `SegmentationResult` containing:
    /// - Per-pixel region labels
    /// - Region bounding boxes
    /// - Confidence map
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        let (orig_w, orig_h) = image.dimensions();
        let mut size = self.input_size.load(Ordering::Relaxed);

        // Retry loop for auto-detecting input size
        loop {
            match self.try_segment(image, size, orig_w, orig_h) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let err_str = e.to_string();
                    if let Some(new_size) = Self::extract_expected_size(&err_str) {
                        if new_size != size && new_size > 0 {
                            log::info!("Auto-detected segmentation model input size: {}x{}", new_size, new_size);
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

    /// Attempt segmentation with a specific input size.
    fn try_segment(
        &self,
        image: &DynamicImage,
        size: u32,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<SegmentationResult> {
        // Resize to model input size
        let resized = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        // Prepare input tensor with ImageNet normalization
        let mut input_data = vec![0.0f32; 3 * size as usize * size as usize];

        for y in 0..size as usize {
            for x in 0..size as usize {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                let values = pixel.0;
                let base_idx = y * size as usize + x;

                // ImageNet normalization
                input_data[base_idx] = (values[0] as f32 / 255.0 - 0.485) / 0.229;
                input_data[size as usize * size as usize + base_idx] =
                    (values[1] as f32 / 255.0 - 0.456) / 0.224;
                input_data[2 * size as usize * size as usize + base_idx] =
                    (values[2] as f32 / 255.0 - 0.406) / 0.225;
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

        let (shape, seg_data) = output.try_extract_tensor::<f32>()?;

        // Determine output dimensions
        let (seg_h, seg_w) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            _ => (size as usize, size as usize),
        };
        let num_classes = match shape.len() {
            4 => shape[1] as usize,
            _ => 20,
        };

        // Build segmentation result
        let mut regions = HashMap::new();
        let mut confidence = vec![0.0f32; (orig_w * orig_h) as usize];
        let mut region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)> = HashMap::new();

        for orig_y in 0..orig_h as usize {
            for orig_x in 0..orig_w as usize {
                // Map to segmentation output coordinates
                let seg_y = (orig_y * seg_h / orig_h as usize).min(seg_h - 1);
                let seg_x = (orig_x * seg_w / orig_w as usize).min(seg_w - 1);

                // Find best class for this pixel
                let mut best_class = 0;
                let mut best_prob = 0.0f32;

                for c in 0..num_classes {
                    let idx = if shape.len() == 4 {
                        c * seg_h * seg_w + seg_y * seg_w + seg_x
                    } else {
                        seg_y * seg_w + seg_x
                    };
                    let prob = if idx < seg_data.len() { seg_data[idx] } else { 0.0 };
                    if prob > best_prob {
                        best_prob = prob;
                        best_class = c;
                    }
                }

                // Map class to region
                let idx = (orig_y * orig_w as usize + orig_x) as u32;
                let region = map_class_to_region(best_class);
                regions.insert(idx, region);
                confidence[orig_y * orig_w as usize + orig_x] = best_prob;

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

/// Map BiSeNet class ID to SegmentationRegion.
///
/// BiSeNet uses 19 classes for face parsing.
/// This function groups them into simpler categories.
fn map_class_to_region(class_id: usize) -> SegmentationRegion {
    match class_id {
        0 => SegmentationRegion::Background,
        1 => SegmentationRegion::Face,       // Skin
        2 | 3 => SegmentationRegion::Face,   // Eyebrows
        4 | 5 => SegmentationRegion::Eyes,   // Eyes
        6 => SegmentationRegion::Face,       // Eyeglasses
        7 | 8 => SegmentationRegion::Ears,   // Ears
        9 => SegmentationRegion::Ears,       // Earring
        10 => SegmentationRegion::Nose,      // Nose
        11..=13 => SegmentationRegion::Lips, // Mouth/Lips
        14 => SegmentationRegion::Neck,      // Neck
        15 => SegmentationRegion::Hair,      // Hair
        16 => SegmentationRegion::Hair,      // Hat
        17 | 18 => SegmentationRegion::Ears, // Earrings
        _ => SegmentationRegion::Background,
    }
}

/// Placeholder segmenter for when the model is not available.
///
/// Generates approximate face regions based on typical portrait layout.
pub struct PlaceholderSegmenter;

impl PlaceholderSegmenter {
    /// Segment using placeholder heuristics.
    ///
    /// Assumes a centered face with typical proportions:
    /// - Face in center 50% width, 60% height
    /// - Eyes in upper face region
    /// - Nose in middle
    /// - Lips in lower face region
    /// - Hair above face
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        let (width, height) = image.dimensions();
        let mut regions = HashMap::new();

        // Define face region (centered)
        let face_y_start = (height as f32 * 0.15) as u32;
        let face_y_end = (height as f32 * 0.75) as u32;
        let face_x_start = (width as f32 * 0.25) as u32;
        let face_x_end = (width as f32 * 0.75) as u32;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as u32;

                let region =
                    if y >= face_y_start && y <= face_y_end && x >= face_x_start && x <= face_x_end
                    {
                        // Inside face region
                        let rel_y = (y - face_y_start) as f32
                            / (face_y_end - face_y_start).max(1) as f32;
                        let rel_x = (x - face_x_start) as f32
                            / (face_x_end - face_x_start).max(1) as f32;

                        // Eye region: upper face, sides
                        if rel_y > 0.25
                            && rel_y < 0.4
                            && ((rel_x > 0.2 && rel_x < 0.4) || (rel_x > 0.6 && rel_x < 0.8))
                        {
                            SegmentationRegion::Eyes
                        }
                        // Nose region: middle face, center
                        else if rel_y > 0.4 && rel_y < 0.6 && rel_x > 0.35 && rel_x < 0.65 {
                            SegmentationRegion::Nose
                        }
                        // Lips region: lower face, center
                        else if rel_y > 0.65 && rel_y < 0.8 && rel_x > 0.3 && rel_x < 0.7 {
                            SegmentationRegion::Lips
                        }
                        // Rest is face/skin
                        else {
                            SegmentationRegion::Face
                        }
                    } else if y < face_y_start {
                        // Above face = hair
                        SegmentationRegion::Hair
                    } else {
                        // Outside face region
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

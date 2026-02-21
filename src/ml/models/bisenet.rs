//! BiSeNet Face Parsing Segmentation Model
//!
//! Provides face parsing with 19 semantic classes using BiSeNet.
//! Supports dynamic input resolution (any size).

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use std::collections::HashMap;

use crate::ml::{
    SegmentationOutput, SegmentationConfig, SegmentationClass,
    FaceBounds, SessionManager, ModelType,
};

/// BiSeNet face parsing segmenter
pub struct BiSeNetSegmenter {
    session_manager: std::sync::Arc<SessionManager>,
    model_path: std::path::PathBuf,
}

impl BiSeNetSegmenter {
    /// Create a new segmenter with session manager
    pub fn new(
        session_manager: std::sync::Arc<SessionManager>,
        model_path: std::path::PathBuf,
    ) -> Self {
        Self {
            session_manager,
            model_path,
        }
    }

    /// Segment an image into semantic regions
    pub fn segment(
        &self,
        image: &DynamicImage,
        config: &SegmentationConfig,
    ) -> Result<SegmentationOutput> {
        let session = self.session_manager
            .get_or_load(&self.model_path, ModelType::Segmentation)
            .context("Failed to load segmentation model")?;

        let (width, height) = image.dimensions();

        // Prepare input (dynamic size)
        let input_tensor = prepare_segmentation_input(image)?;

        // Run inference
        let mut sess = session.session.lock().unwrap();
        let outputs = sess.run(ort::inputs![
            &session.input_name => input_tensor
        ]).context("Segmentation inference failed")?;

        // Get output tensor
        let output = outputs
            .get(&session.output_name)
            .context("No output from segmentation model")?;

        let (shape, data) = output.try_extract_tensor::<f32>()
            .context("Failed to extract segmentation output")?;

        // Parse segmentation output
        let (class_map, confidence_map) = parse_segmentation_output(
            &data,
            &shape,
            width,
            height,
            config.confidence_threshold,
        )?;

        // Compute class bounds
        let class_bounds = compute_class_bounds(&class_map, width, height);

        // Apply optional merging
        let class_map = if config.merge_eyebrows || config.merge_eyes || config.merge_lips {
            merge_classes(&class_map, width, height, config)
        } else {
            class_map
        };

        // Apply small region filtering
        let class_map = if config.min_region_size > 0 {
            filter_small_regions(&class_map, width, height, config.min_region_size)
        } else {
            class_map
        };

        Ok(SegmentationOutput {
            class_map,
            confidence_map,
            width,
            height,
            class_bounds,
        })
    }
}

/// Prepare image for BiSeNet input (dynamic size)
fn prepare_segmentation_input(image: &DynamicImage) -> Result<ort::value::Value> {
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8();

    // Create input tensor (NCHW format)
    // BiSeNet expects RGB normalized with ImageNet stats
    let mut input_data = vec![0.0f32; 3 * (width * height) as usize];

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = rgba.get_pixel(x as u32, y as u32);
            let values = pixel.0;
            let base_idx = y * width as usize + x;

            // Normalize with ImageNet stats
            input_data[base_idx] = (values[0] as f32 / 255.0 - mean[0]) / std[0];
            input_data[width as usize * height as usize + base_idx] =
                (values[1] as f32 / 255.0 - mean[1]) / std[1];
            input_data[2 * width as usize * height as usize + base_idx] =
                (values[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    ort::value::Value::from_array((
        [1i64, 3, height as i64, width as i64],
        input_data,
    )).context("Failed to create segmentation input tensor")
}

/// Parse BiSeNet output to class map and confidence map
fn parse_segmentation_output(
    data: &[f32],
    shape: &[i64],
    orig_width: u32,
    orig_height: u32,
    conf_threshold: f32,
) -> Result<(Vec<u8>, Vec<f32>)> {
    // BiSeNet output: [1, 19, H, W] (class scores) or [1, H, W] (argmax result)
    let num_classes = if shape.len() == 4 {
        shape[1] as usize
    } else {
        19 // Default for face parsing
    };

    let (out_height, out_width) = match shape.len() {
        4 => (shape[2] as usize, shape[3] as usize),
        3 => (shape[1] as usize, shape[2] as usize),
        2 => (shape[0] as usize, shape[1] as usize),
        _ => {
            return Err(anyhow::anyhow!(
                "Unexpected segmentation output shape: {:?}",
                shape
            ));
        }
    };

    let total_pixels = (orig_width * orig_height) as usize;
    let mut class_map = vec![0u8; total_pixels];
    let mut confidence_map = vec![0.0f32; total_pixels];

    // If output has class dimension, apply argmax
    if shape.len() == 4 && shape[1] > 1 {
        // Softmax-like: find max class for each pixel
        for y in 0..orig_height as usize {
            for x in 0..orig_width as usize {
                // Map to output coordinates
                let out_y = (y * out_height / orig_height as usize).min(out_height - 1);
                let out_x = (x * out_width / orig_width as usize).min(out_width - 1);
                let out_idx = out_y * out_width + out_x;

                // Find class with max probability
                let mut best_class = 0u8;
                let mut best_prob = f32::NEG_INFINITY;

                for c in 0..num_classes.min(19) {
                    let idx = c * out_height * out_width + out_idx;
                    let prob = if idx < data.len() { data[idx] } else { 0.0 };

                    if prob > best_prob {
                        best_prob = prob;
                        best_class = c as u8;
                    }
                }

                // Apply softmax for confidence
                let mut sum_exp = 0.0f32;
                for c in 0..num_classes.min(19) {
                    let idx = c * out_height * out_width + out_idx;
                    let prob = if idx < data.len() { data[idx] } else { 0.0 };
                    sum_exp += (prob - best_prob).exp();
                }
                let confidence = 1.0 / sum_exp;

                let in_idx = y * orig_width as usize + x;
                class_map[in_idx] = best_class;
                confidence_map[in_idx] = confidence;
            }
        }
    } else {
        // Output is already class indices
        for y in 0..orig_height as usize {
            for x in 0..orig_width as usize {
                let out_y = (y * out_height / orig_height as usize).min(out_height - 1);
                let out_x = (x * out_width / orig_width as usize).min(out_width - 1);
                let out_idx = out_y * out_width + out_x;

                let in_idx = y * orig_width as usize + x;
                class_map[in_idx] = if out_idx < data.len() {
                    data[out_idx] as u8
                } else {
                    0
                };
                confidence_map[in_idx] = 1.0;
            }
        }
    }

    // Apply confidence threshold (set low-confidence pixels to background)
    if conf_threshold > 0.0 {
        for (class, &conf) in class_map.iter_mut().zip(confidence_map.iter()) {
            if conf < conf_threshold {
                *class = 0; // Set to background
            }
        }
    }

    Ok((class_map, confidence_map))
}

/// Compute bounding box for each class
fn compute_class_bounds(
    class_map: &[u8],
    width: u32,
    height: u32,
) -> HashMap<SegmentationClass, FaceBounds> {
    let mut bounds: HashMap<SegmentationClass, (f32, f32, f32, f32, u32, u32)> = HashMap::new();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let class_id = class_map[idx];

            if let Some(class) = SegmentationClass::from_index(class_id) {
                let entry = bounds.entry(class).or_insert((
                    x as f32, y as f32, x as f32, y as f32, // min_x, min_y, max_x, max_y
                    0u32, 0u32, // pixel count, placeholder
                ));
                entry.0 = entry.0.min(x as f32);
                entry.1 = entry.1.min(y as f32);
                entry.2 = entry.2.max(x as f32);
                entry.3 = entry.3.max(y as f32);
                entry.4 += 1;
            }
        }
    }

    bounds
        .into_iter()
        .filter_map(|(class, (min_x, min_y, max_x, max_y, count, _))| {
            // Skip classes with very few pixels
            if count < 10 {
                return None;
            }

            Some((
                class,
                FaceBounds {
                    x: min_x / width as f32,
                    y: min_y / height as f32,
                    width: (max_x - min_x + 1.0) / width as f32,
                    height: (max_y - min_y + 1.0) / height as f32,
                },
            ))
        })
        .collect()
}

/// Merge related classes (eyebrows, eyes, lips)
fn merge_classes(
    class_map: &[u8],
    width: u32,
    height: u32,
    config: &SegmentationConfig,
) -> Vec<u8> {
    let mut result = class_map.to_vec();

    for (y, x) in (0..height).flat_map(|y| (0..width).map(move |x| (y, x))) {
        let idx = (y * width + x) as usize;
        let class_id = result[idx];

        if let Some(class) = SegmentationClass::from_index(class_id) {
            let new_class = match class {
                SegmentationClass::LeftEyebrow | SegmentationClass::RightEyebrow
                    if config.merge_eyebrows => {
                    Some(SegmentationClass::Skin) // Merge into skin
                }
                SegmentationClass::LeftEye | SegmentationClass::RightEye
                    if config.merge_eyes => {
                    // Keep as eyes but unified
                    Some(SegmentationClass::LeftEye)
                }
                SegmentationClass::UpperLip | SegmentationClass::LowerLip | SegmentationClass::InnerMouth
                    if config.merge_lips => {
                    Some(SegmentationClass::UpperLip) // Merge into single lip class
                }
                _ => None,
            };

            if let Some(new) = new_class {
                result[idx] = new as u8;
            }
        }
    }

    result
}

/// Filter out small isolated regions
fn filter_small_regions(
    class_map: &[u8],
    width: u32,
    height: u32,
    min_size: usize,
) -> Vec<u8> {
    // Simple approach: for each pixel, check if neighbors have same class
    // If too few neighbors, set to most common neighbor class
    let mut result = class_map.to_vec();

    let get_neighbors = |x: u32, y: u32| -> Vec<(u32, u32)> {
        let mut neighbors = Vec::new();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    neighbors.push((nx as u32, ny as u32));
                }
            }
        }
        neighbors
    };

    // Count connected components (simple flood-fill approach)
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;
            let class_id = result[idx];

            // Count same-class neighbors
            let same_count = get_neighbors(x, y)
                .iter()
                .filter(|&&(nx, ny)| {
                    let nidx = (ny * width + nx) as usize;
                    result[nidx] == class_id
                })
                .count();

            // If isolated, replace with most common neighbor class
            if same_count < 2 {
                let mut class_counts: [usize; 19] = [0; 19];
                for &(nx, ny) in &get_neighbors(x, y) {
                    let nidx = (ny * width + nx) as usize;
                    let nclass = result[nidx] as usize;
                    if nclass < 19 {
                        class_counts[nclass] += 1;
                    }
                }

                let most_common = class_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, count)| count)
                    .map(|(class, _)| class as u8)
                    .unwrap_or(0);

                result[idx] = most_common;
            }
        }
    }

    result
}

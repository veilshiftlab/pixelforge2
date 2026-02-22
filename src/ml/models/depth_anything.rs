//! Depth-Anything V2: Monocular Depth Estimation

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ort::session::builder::GraphOptimizationLevel;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

/// Depth estimator using Depth-Anything V2 ONNX model.
pub struct DepthAnythingEstimator {
    session: RefCell<ort::session::Session>,
    input_name: String,
    output_name: String,
    input_size: AtomicU32,
}

impl DepthAnythingEstimator {
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();
        let input_size = AtomicU32::new(256);

        log::info!("Depth-Anything V2 model loaded");

        Ok(Self {
            session: RefCell::new(session),
            input_name,
            output_name,
            input_size,
        })
    }

    pub fn estimate(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (orig_w, orig_h) = image.dimensions();
        let mut size = self.input_size.load(Ordering::Relaxed);

        loop {
            match self.try_estimate(image, size, orig_w, orig_h) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let err_str = e.to_string();
                    if let Some(new_size) = Self::extract_expected_size(&err_str) {
                        if new_size != size && new_size > 0 {
                            log::info!("Auto-detected depth model input size: {}x{}", new_size, new_size);
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

    fn try_estimate(
        &self,
        image: &DynamicImage,
        size: u32,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<Vec<f32>> {
        let resized = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        let mut input_data = vec![0.0f32; 3 * size as usize * size as usize];

        for y in 0..size as usize {
            for x in 0..size as usize {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                let values = pixel.0;
                let base_idx = y * size as usize + x;
                input_data[base_idx] = values[0] as f32 / 255.0;
                input_data[size as usize * size as usize + base_idx] = values[1] as f32 / 255.0;
                input_data[2 * size as usize * size as usize + base_idx] = values[2] as f32 / 255.0;
            }
        }

        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, size as i64, size as i64],
            input_data,
        ))?;

        let mut session = self.session.borrow_mut();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        // Get output by name (using String directly)
        let output_name = self.output_name.clone();
        let output = outputs.get(&output_name)
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        let (shape, depth_data) = output.try_extract_tensor::<f32>()?;

        let (dh, dw) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            _ => (size as usize, size as usize),
        };

        let mut min_depth = f32::MAX;
        let mut max_depth = f32::MIN;
        for &d in depth_data.iter() {
            min_depth = min_depth.min(d);
            max_depth = max_depth.max(d);
        }
        let range = (max_depth - min_depth).max(0.001);

        let mut depth_map = Vec::with_capacity((orig_w * orig_h) as usize);

        for orig_y in 0..orig_h as usize {
            for orig_x in 0..orig_w as usize {
                let depth_y = (orig_y * dh / orig_h as usize).min(dh - 1);
                let depth_x = (orig_x * dw / orig_w as usize).min(dw - 1);

                let idx = match shape.len() {
                    4 => depth_y * dw + depth_x,
                    3 => depth_y * dw + depth_x,
                    _ => depth_y * dw + depth_x,
                };

                let d = if idx < depth_data.len() { depth_data[idx] } else { 0.5 };
                depth_map.push(((d - min_depth) / range).clamp(0.0, 1.0));
            }
        }

        Ok(depth_map)
    }

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

/// Placeholder depth estimator.
pub struct PlaceholderDepthEstimator;

impl PlaceholderDepthEstimator {
    pub fn estimate(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (width, height) = image.dimensions();
        let mut depth_map = vec![0.5f32; (width * height) as usize];

        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        let max_dist = center_x.hypot(center_y).max(1.0);

        for y in 0..height {
            for x in 0..width {
                let dist = ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2)).sqrt();
                depth_map[(y * width + x) as usize] = 1.0 - (dist / max_dist * 0.5).min(1.0);
            }
        }

        let gray = image.to_luma8();
        for (i, depth) in depth_map.iter_mut().enumerate() {
            let x = (i % width as usize) as u32;
            let y = (i / width as usize) as u32;
            if x < width && y < height {
                let luminance = gray.get_pixel(x, y)[0] as f32 / 255.0;
                *depth = (*depth * 0.7 + (1.0 - luminance) * 0.3).clamp(0.0, 1.0);
            }
        }

        Ok(depth_map)
    }
}

//! Depth estimation using ONNX Runtime

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ort::session::builder::GraphOptimizationLevel;

/// Depth estimator using ONNX model
pub struct DepthEstimator {
    session: ort::session::Session,
    input_name: String,
    output_name: String,
}

impl DepthEstimator {
    /// Create a new depth estimator
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        Ok(Self { session, input_name, output_name })
    }

    /// Estimate depth map for an image
    pub fn estimate(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (orig_w, orig_h) = image.dimensions();

        // MiDaS expects 384x384 input
        let size = 384u32;
        let resized = image.resize(size, size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        // Create input tensor as flat vec (NCHW format)
        let mut input_data = vec![0.0f32; 1 * 3 * size as usize * size as usize];
        
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

        // Create input value
        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, size as i64, size as i64],
            input_data,
        ))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            &self.input_name => input_tensor
        ])?;

        // Get output
        let output = outputs.get(&self.output_name)
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        // Extract tensor data
        let (shape, depth_data) = output.try_extract_tensor::<f32>()?;

        // Get dimensions
        let (dh, dw) = match shape.len() {
            4 => (shape[2] as usize, shape[3] as usize),
            3 => (shape[1] as usize, shape[2] as usize),
            _ => (size as usize, size as usize),
        };

        // Find min/max for normalization
        let mut min_depth = f32::MAX;
        let mut max_depth = f32::MIN;

        for &d in depth_data.iter() {
            min_depth = min_depth.min(d);
            max_depth = max_depth.max(d);
        }

        let range = (max_depth - min_depth).max(0.001);

        // Normalize and interpolate to original size
        let mut depth_map = Vec::with_capacity((orig_w * orig_h) as usize);
        
        for orig_y in 0..orig_h as usize {
            for orig_x in 0..orig_w as usize {
                let depth_y = (orig_y * dh / orig_h as usize).min(dh - 1);
                let depth_x = (orig_x * dw / orig_w as usize).min(dw - 1);
                
                let idx = match shape.len() {
                    4 => depth_y * dw + depth_x + (shape[1] * shape[2] * shape[3]) as usize, // Skip batch and channel
                    3 => depth_y * dw + depth_x + (shape[1] * shape[2]) as usize,
                    _ => depth_y * dw + depth_x,
                };
                
                let d = if idx < depth_data.len() { depth_data[idx] } else { 0.5 };
                let normalized = (d - min_depth) / range;
                depth_map.push(normalized.clamp(0.0, 1.0));
            }
        }

        Ok(depth_map)
    }
}

/// Placeholder depth estimator for when models aren't available
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
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();
                let idx = (y * width + x) as usize;
                depth_map[idx] = 1.0 - (dist / max_dist * 0.5).min(1.0);
            }
        }

        let gray = image.to_luma8();
        for (i, depth) in depth_map.iter_mut().enumerate() {
            let y = (i / width as usize) as u32;
            let x = (i % width as usize) as u32;
            if x < width && y < height {
                let pixel = gray.get_pixel(x, y);
                let luminance = pixel[0] as f32 / 255.0;
                *depth = (*depth * 0.7 + luminance * 0.3).clamp(0.0, 1.0);
            }
        }

        Ok(depth_map)
    }
}

//! Depth estimation using ONNX Runtime

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// Depth estimator using ONNX model
pub struct DepthEstimator {
    #[allow(dead_code)]
    session: ort::session::Session,
}

impl DepthEstimator {
    /// Create a new depth estimator
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session = ort::session::SessionBuilder::new()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Estimate depth map for an image
    pub fn estimate(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let (orig_w, orig_h) = image.dimensions();

        // MiDaS expects 384x384 input
        let size = 384u32;
        let resized = image.resize(size, size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();

        // Create input tensor (NCHW format)
        let mut input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));

        for y in 0..size as usize {
            for x in 0..size as usize {
                let pixel = rgba.get_pixel(x as u32, y as u32);
                let values = pixel.0;
                // Normalize to 0-1 and convert to NCHW
                input[[0, 0, y, x]] = values[0] as f32 / 255.0;
                input[[0, 1, y, x]] = values[1] as f32 / 255.0;
                input[[0, 2, y, x]] = values[2] as f32 / 255.0;
            }
        }

        // Run inference using ort 2.0 API
        let input_values = vec![ort::value::Value::from_array(input)?];
        let outputs = self.session.run(input_values)?;

        // Get output
        let output = outputs.get(0)
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        let view = output.try_extract_tensor::<f32>()?;
        let depth_data = view.view();

        // Extract depth map (usually [1, 1, H, W] or [1, H, W])
        let mut depth_map = Vec::with_capacity((orig_w * orig_h) as usize);

        // Get dimensions
        let shape = depth_data.shape();
        let (dh, dw) = match shape.len() {
            4 => (shape[2], shape[3]),
            3 => (shape[1], shape[2]),
            _ => (size as usize, size as usize),
        };

        // Normalize depth values
        let mut min_depth = f32::MAX;
        let mut max_depth = f32::MIN;

        // First pass: find min/max
        for y in 0..dh {
            for x in 0..dw {
                let idx = match shape.len() {
                    4 => [0, 0, y, x],
                    3 => [0, y, x],
                    _ => [y, x],
                };
                let d = depth_data[&idx];
                min_depth = min_depth.min(d);
                max_depth = max_depth.max(d);
            }
        }

        let range = (max_depth - min_depth).max(0.001);

        // Second pass: normalize and interpolate to original size
        for orig_y in 0..orig_h as usize {
            for orig_x in 0..orig_w as usize {
                // Map to depth map coordinates
                let depth_y = (orig_y * dh / orig_h as usize).min(dh - 1);
                let depth_x = (orig_x * dw / orig_w as usize).min(dw - 1);

                let idx = match shape.len() {
                    4 => [0, 0, depth_y, depth_x],
                    3 => [0, depth_y, depth_x],
                    _ => [depth_y, depth_x],
                };

                let d = depth_data[&idx];
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

        // Create a simple synthetic depth map (center is closer)
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        let max_dist = (center_x.hypot(center_y)).max(1.0);

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Closer to center = higher depth value (closer)
                let idx = (y * width + x) as usize;
                depth_map[idx] = 1.0 - (dist / max_dist * 0.5).min(1.0);
            }
        }

        // Add some variation based on luminance
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

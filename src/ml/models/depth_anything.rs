//! Depth-Anything V2 Depth Estimation Model
//!
//! Provides depth estimation using Depth-Anything V2.
//! Supports dynamic input resolution (any size).

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};

use crate::ml::{
    DepthOutput, DepthConfig, SessionManager, ModelType,
};

/// Depth-Anything V2 depth estimator
pub struct DepthAnythingV2 {
    session_manager: std::sync::Arc<SessionManager>,
    model_path: std::path::PathBuf,
}

impl DepthAnythingV2 {
    /// Create a new depth estimator with session manager
    pub fn new(
        session_manager: std::sync::Arc<SessionManager>,
        model_path: std::path::PathBuf,
    ) -> Self {
        Self {
            session_manager,
            model_path,
        }
    }

    /// Estimate depth map for an image
    pub fn estimate(
        &self,
        image: &DynamicImage,
        config: &DepthConfig,
    ) -> Result<DepthOutput> {
        let session = self.session_manager
            .get_or_load(&self.model_path, ModelType::DepthEstimation)
            .context("Failed to load depth estimation model")?;

        let (width, height) = image.dimensions();

        // Prepare input (dynamic size, no resize needed for Depth-Anything)
        let input_tensor = prepare_depth_input(image)?;

        // Run inference
        let mut sess = session.session.lock().unwrap();
        let outputs = sess.run(ort::inputs![
            &session.input_name => input_tensor
        ]).context("Depth estimation inference failed")?;

        // Get output tensor
        let output = outputs
            .get(&session.output_name)
            .context("No output from depth estimation model")?;

        let (shape, data) = output.try_extract_tensor::<f32>()
            .context("Failed to extract depth output")?;

        // Extract depth map
        let depth_map = extract_depth_map(&data, &shape, width, height)?;

        // Compute range for normalization
        let (min_val, max_val) = compute_range(&depth_map);

        // Normalize if requested
        let (depth_range, final_depth) = if config.normalize_output {
            let normalized = normalize_depth(&depth_map, min_val, max_val);
            ((0.0f32, 1.0f32), normalized)
        } else {
            ((min_val, max_val), depth_map)
        };

        Ok(DepthOutput {
            depth_map: final_depth,
            width,
            height,
            depth_range,
        })
    }
}

/// Prepare image for Depth-Anything input (dynamic size)
fn prepare_depth_input(image: &DynamicImage) -> Result<ort::value::Value> {
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8();

    // Create input tensor (NCHW format)
    // Depth-Anything expects RGB normalized with ImageNet stats
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
    )).context("Failed to create depth input tensor")
}

/// Extract depth map from model output
fn extract_depth_map(
    data: &[f32],
    shape: &[i64],
    orig_width: u32,
    orig_height: u32,
) -> Result<Vec<f32>> {
    // Depth-Anything output shape: [1, 1, H, W] or [1, H, W]
    let (out_height, out_width) = match shape.len() {
        4 => (shape[2] as usize, shape[3] as usize),
        3 => (shape[1] as usize, shape[2] as usize),
        2 => (shape[0] as usize, shape[1] as usize),
        _ => {
            return Err(anyhow::anyhow!(
                "Unexpected depth output shape: {:?}",
                shape
            ));
        }
    };

    // If output matches input size, use directly
    if out_width == orig_width as usize && out_height == orig_height as usize {
        return Ok(data.to_vec());
    }

    // Otherwise, interpolate to original size
    let mut depth_map = Vec::with_capacity((orig_width * orig_height) as usize);

    for y in 0..orig_height as usize {
        for x in 0..orig_width as usize {
            // Bilinear interpolation
            let src_y = y as f32 * (out_height - 1) as f32 / (orig_height - 1) as f32;
            let src_x = x as f32 * (out_width - 1) as f32 / (orig_width - 1) as f32;

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(out_height - 1);
            let x1 = (x0 + 1).min(out_width - 1);

            let fy = src_y - y0 as f32;
            let fx = src_x - x0 as f32;

            let v00 = data[y0 * out_width + x0];
            let v10 = data[y0 * out_width + x1];
            let v01 = data[y1 * out_width + x0];
            let v11 = data[y1 * out_width + x1];

            let v = v00 * (1.0 - fx) * (1.0 - fy)
                  + v10 * fx * (1.0 - fy)
                  + v01 * (1.0 - fx) * fy
                  + v11 * fx * fy;

            depth_map.push(v);
        }
    }

    Ok(depth_map)
}

/// Compute min/max range of depth values
fn compute_range(depth_map: &[f32]) -> (f32, f32) {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &v in depth_map {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    (min_val, max_val)
}

/// Normalize depth values to 0-1 range
fn normalize_depth(depth_map: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    let range = (max_val - min_val).max(0.0001);

    depth_map
        .iter()
        .map(|&v| ((v - min_val) / range).clamp(0.0, 1.0))
        .collect()
}

/// Invert depth values (swap near/far)
pub fn invert_depth(depth_map: &[f32]) -> Vec<f32> {
    depth_map.iter().map(|&v| 1.0 - v).collect()
}

/// Apply gamma correction to depth values
pub fn apply_gamma(depth_map: &[f32], gamma: f32) -> Vec<f32> {
    depth_map
        .iter()
        .map(|&v| v.powf(gamma).clamp(0.0, 1.0))
        .collect()
}

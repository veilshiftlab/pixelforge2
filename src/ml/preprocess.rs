//! Image preprocessing utilities for ML models

use anyhow::Result;
use image::{DynamicImage, GenericImageView};

/// Preprocessing configuration
#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    /// Target size (width and height)
    pub target_size: u32,
    /// Whether to maintain aspect ratio
    pub keep_aspect_ratio: bool,
    /// Normalization mean (RGB)
    pub mean: [f32; 3],
    /// Normalization std (RGB)
    pub std: [f32; 3],
}

impl PreprocessConfig {
    /// Default config for ImageNet-normalized models
    pub fn imagenet() -> Self {
        Self {
            target_size: 224,
            keep_aspect_ratio: false,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Default config for MiDaS depth models
    pub fn midas() -> Self {
        Self {
            target_size: 384,
            keep_aspect_ratio: false,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

/// Preprocess image with simple resize (fixed size)
pub fn preprocess_image(
    image: &DynamicImage,
    config: &PreprocessConfig,
) -> Result<Vec<f32>> {
    let resized = if config.keep_aspect_ratio {
        image.resize(config.target_size, config.target_size, image::imageops::FilterType::Triangle)
    } else {
        image.resize_exact(config.target_size, config.target_size, image::imageops::FilterType::Triangle)
    };

    let rgba = resized.to_rgba8();
    let mut input_data = vec![0.0f32; 3 * (config.target_size * config.target_size) as usize];

    for y in 0..config.target_size as usize {
        for x in 0..config.target_size as usize {
            let pixel = rgba.get_pixel(x as u32, y as u32);
            let values = pixel.0;
            let base_idx = y * config.target_size as usize + x;

            // Normalize with mean/std
            input_data[base_idx] = (values[0] as f32 / 255.0 - config.mean[0]) / config.std[0];
            input_data[config.target_size as usize * config.target_size as usize + base_idx] =
                (values[1] as f32 / 255.0 - config.mean[1]) / config.std[1];
            input_data[2 * config.target_size as usize * config.target_size as usize + base_idx] =
                (values[2] as f32 / 255.0 - config.mean[2]) / config.std[2];
        }
    }

    Ok(input_data)
}

/// Preprocess image with simple normalization (no resize)
pub fn preprocess_image_simple(image: &DynamicImage) -> Result<Vec<f32>> {
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8();

    let mut input_data = vec![0.0f32; 3 * (width * height) as usize];
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = rgba.get_pixel(x as u32, y as u32);
            let values = pixel.0;
            let base_idx = y * width as usize + x;

            input_data[base_idx] = (values[0] as f32 / 255.0 - mean[0]) / std[0];
            input_data[width as usize * height as usize + base_idx] =
                (values[1] as f32 / 255.0 - mean[1]) / std[1];
            input_data[2 * width as usize * height as usize + base_idx] =
                (values[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    Ok(input_data)
}

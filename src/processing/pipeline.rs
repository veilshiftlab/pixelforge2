//! Main processing pipeline implementation

use super::{ProcessingConfig, TransformConfig};
use super::{depth_to_flat, weighted_downsample, compute_importance_map, preserve_features, draw_edges, generate_palette, apply_palette};
use crate::ml::MLResults;
use crate::image::ImageTransform;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};

/// Processing pipeline
pub struct ProcessingPipeline;

impl ProcessingPipeline {
    /// Execute the full processing pipeline
    pub fn process(
        input: &DynamicImage,
        ml_results: Option<&MLResults>,
        config: &ProcessingConfig,
        progress_callback: impl Fn(f32, &str) + Send + Sync + 'static,
    ) -> Result<DynamicImage> {
        // Stage 1: Preprocessing (5%)
        progress_callback(0.05, "Preparing image...");
        let preprocessed = Self::preprocess(input, &config.transform)?;

        // Stage 2: Depth-to-flat conversion (if ML results available)
        progress_callback(0.10, "Converting depth to flat colors...");
        let flat_color = if let Some(ml) = ml_results {
            depth_to_flat(&preprocessed, ml, &config.depth_to_flat)?
        } else {
            preprocessed
        };

        // Stage 3: Compute importance map
        progress_callback(0.55, "Computing importance map...");
        let (width, height) = flat_color.dimensions();
        let importance = compute_importance_map(width, height, ml_results);

        // Stage 4: Weighted downsampling
        progress_callback(0.60, "Downsampling...");
        let downsampled = weighted_downsample(
            &flat_color,
            &importance,
            config.transform.output_size,
            config.transform.output_size,
        )?;

        // Stage 5: Feature preservation (if ML results available)
        progress_callback(0.85, "Preserving features...");
        let with_features = if let Some(ml) = ml_results {
            preserve_features(&downsampled, ml, &config.features)?
        } else {
            downsampled
        };

        // Stage 6: Palette quantization
        progress_callback(0.90, "Applying palette...");
        let palette = generate_palette(&with_features, &config.palette, ml_results)?;
        let quantized = apply_palette(&with_features, &palette)?;

        // Stage 7: Edge enhancement
        progress_callback(0.95, "Drawing edges...");
        let with_edges = draw_edges(&quantized, ml_results, &config.edges)?;

        // Stage 8: Final output
        progress_callback(1.0, "Complete!");
        Ok(with_edges)
    }

    /// Preprocess image (scale, rotate, clip)
    fn preprocess(input: &DynamicImage, config: &TransformConfig) -> Result<DynamicImage> {
        let mut result = input.clone();

        // Scale
        if config.scale != 1.0 {
            let new_width = (result.width() as f32 * config.scale) as u32;
            let new_height = (result.height() as f32 * config.scale) as u32;
            result = ImageTransform::resize(&result, new_width, new_height)?;
        }

        // Rotate
        if config.rotation != 0.0 {
            result = ImageTransform::rotate(&result, config.rotation)?;
        }

        // Offset
        if config.offset_x != 0.0 || config.offset_y != 0.0 {
            result = ImageTransform::offset(&result, config.offset_x, config.offset_y)?;
        }

        Ok(result)
    }
}

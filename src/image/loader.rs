//! Image loading utilities

use anyhow::Result;
use image::DynamicImage;
use std::path::Path;
use super::transform::ImageProcessor;

/// Image loader
pub struct ImageLoader;

impl ImageLoader {
    /// Load an image from a file
    pub fn load(path: &Path) -> Result<DynamicImage> {
        let image = image::open(path)?;

        log::info!(
            "Loaded image: {:?} ({}x{})",
            path.file_name(),
            image.width(),
            image.height()
        );

        Ok(image)
    }

    /// Load image from bytes
    pub fn load_from_bytes(bytes: &[u8]) -> Result<DynamicImage> {
        let image = image::load_from_memory(bytes)?;
        Ok(image)
    }

    /// Load an image from a file into an ImageProcessor
    /// `max_dimension`: maximum width or height (0 = no auto-resize)
    pub fn load_with_processor(path: &Path, max_dimension: u32) -> Result<ImageProcessor> {
        let image = Self::load(path)?;
        Ok(ImageProcessor::new(image, Some(path.to_path_buf()), max_dimension))
    }

    /// Load image from bytes into an ImageProcessor
    pub fn load_from_bytes_with_processor(bytes: &[u8], max_dimension: u32) -> Result<ImageProcessor> {
        let image = Self::load_from_bytes(bytes)?;
        Ok(ImageProcessor::new(image, None, max_dimension))
    }
}

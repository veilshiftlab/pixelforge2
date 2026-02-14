//! Image loading utilities

use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// Image data container
pub struct ImageData {
    /// The loaded image
    pub image: DynamicImage,

    /// Original file path
    pub path: Option<std::path::PathBuf>,
}

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
}

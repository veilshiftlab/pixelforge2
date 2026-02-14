//! Image export utilities

use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::path::Path;

/// Export configuration
pub struct ExportConfig {
    /// Output format
    pub format: ImageFormat,

    /// Scale multiplier
    pub scale: u32,

    /// JPG quality (1-100)
    pub quality: u8,

    /// Include metadata
    pub include_metadata: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ImageFormat::Png,
            scale: 1,
            quality: 90,
            include_metadata: false,
        }
    }
}

/// Image exporter
pub struct ImageExporter;

impl ImageExporter {
    /// Export an image to a file
    pub fn export(image: &DynamicImage, path: &Path, config: &ExportConfig) -> Result<()> {
        // Apply scale
        let scaled = if config.scale > 1 {
            let (w, h) = image.dimensions();
            image.resize(
                w * config.scale,
                h * config.scale,
                image::imageops::FilterType::Nearest,
            )
        } else {
            image.clone()
        };

        // Determine format from extension
        let format = ImageFormat::from_path(path).unwrap_or(config.format);

        // Save
        match format {
            ImageFormat::Jpeg => {
                let rgba = scaled.to_rgba8();
                let rgb = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_fn(
                    rgba.width(),
                    rgba.height(),
                    |x, y| {
                        let p = rgba.get_pixel(x, y);
                        image::Rgb([p[0], p[1], p[2]])
                    },
                );

                let mut output = std::fs::File::create(path)?;
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, config.quality);
                encoder.encode(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    image::ExtendedColorType::Rgb8,
                )?;
            }
            ImageFormat::Png => {
                scaled.save_with_format(path, ImageFormat::Png)?;
            }
            _ => {
                scaled.save(path)?;
            }
        }

        log::info!("Exported image to: {}", path.display());
        Ok(())
    }

    /// Export to PNG with optional metadata
    pub fn export_png(
        image: &DynamicImage,
        path: &Path,
        _metadata: Option<&[(String, String)]>,
    ) -> Result<()> {
        image.save_with_format(path, ImageFormat::Png)?;
        log::info!("Exported PNG to: {}", path.display());
        Ok(())
    }

    /// Export to JPEG with quality
    pub fn export_jpeg(image: &DynamicImage, path: &Path, quality: u8) -> Result<()> {
        let rgba = image.to_rgba8();
        let rgb = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_fn(
            rgba.width(),
            rgba.height(),
            |x, y| {
                let p = rgba.get_pixel(x, y);
                image::Rgb([p[0], p[1], p[2]])
            },
        );

        let mut output = std::fs::File::create(path)?;
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, quality);
        encoder.encode(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )?;

        log::info!("Exported JPEG to: {}", path.display());
        Ok(())
    }
}

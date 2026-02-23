//! Image transformation utilities

use anyhow::Result;
use image::{DynamicImage, GenericImageView};

/// Image transform utilities
pub struct ImageTransform;

impl ImageTransform {
    /// Resize an image
    pub fn resize(image: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage> {
        Ok(image.resize(width, height, image::imageops::FilterType::Lanczos3))
    }

    /// Rotate an image by degrees
    pub fn rotate(image: &DynamicImage, degrees: f32) -> Result<DynamicImage> {
        let radians = degrees.to_radians();

        let (width, height) = image.dimensions();
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        let cos = radians.cos().abs();
        let sin = radians.sin().abs();

        let new_width = (width as f32 * cos + height as f32 * sin) as u32;
        let new_height = (width as f32 * sin + height as f32 * cos) as u32;

        let rgba = image.to_rgba8();
        let mut output = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(new_width, new_height);

        let new_center_x = new_width as f32 / 2.0;
        let new_center_y = new_height as f32 / 2.0;

        for y in 0..new_height {
            for x in 0..new_width {
                let dx = x as f32 - new_center_x;
                let dy = y as f32 - new_center_y;

                let src_x = dx * cos + dy * sin + center_x;
                let src_y = -dx * sin + dy * cos + center_y;

                if src_x >= 0.0 && src_x < width as f32 && src_y >= 0.0 && src_y < height as f32 {
                    let x0 = src_x as u32;
                    let y0 = src_y as u32;
                    let x1 = (x0 + 1).min(width - 1);
                    let y1 = (y0 + 1).min(height - 1);

                    let fx = src_x - x0 as f32;
                    let fy = src_y - y0 as f32;

                    let p00 = rgba.get_pixel(x0, y0);
                    let p10 = rgba.get_pixel(x1, y0);
                    let p01 = rgba.get_pixel(x0, y1);
                    let p11 = rgba.get_pixel(x1, y1);

                    let interpolate = |c: usize| -> u8 {
                        let v00 = p00[c] as f32;
                        let v10 = p10[c] as f32;
                        let v01 = p01[c] as f32;
                        let v11 = p11[c] as f32;

                        let v0 = v00 * (1.0 - fx) + v10 * fx;
                        let v1 = v01 * (1.0 - fx) + v11 * fx;
                        let v = v0 * (1.0 - fy) + v1 * fy;

                        v as u8
                    };

                    output.put_pixel(x, y, image::Rgba([
                        interpolate(0),
                        interpolate(1),
                        interpolate(2),
                        interpolate(3),
                    ]));
                }
            }
        }

        Ok(DynamicImage::ImageRgba8(output))
    }

    /// Apply offset to image (pan)
    pub fn offset(image: &DynamicImage, offset_x: f32, offset_y: f32) -> Result<DynamicImage> {
        let (width, height) = image.dimensions();
        let rgba = image.to_rgba8();

        let mut output = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(width, height);

        let dx = (offset_x * width as f32) as i32;
        let dy = (offset_y * height as f32) as i32;

        for y in 0..height {
            for x in 0..width {
                let src_x = x as i32 - dx;
                let src_y = y as i32 - dy;

                if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
                    let pixel = rgba.get_pixel(src_x as u32, src_y as u32);
                    output.put_pixel(x, y, *pixel);
                }
            }
        }

        Ok(DynamicImage::ImageRgba8(output))
    }

    /// Clip image to a region
    pub fn clip(image: &DynamicImage, x: f32, y: f32, width: f32, height: f32) -> Result<DynamicImage> {
        let (img_w, img_h) = image.dimensions();

        let clip_x = (x * img_w as f32) as u32;
        let clip_y = (y * img_h as f32) as u32;
        let clip_w = (width * img_w as f32) as u32;
        let clip_h = (height * img_h as f32) as u32;

        let clip_x = clip_x.min(img_w - 1);
        let clip_y = clip_y.min(img_h - 1);
        let clip_w = clip_w.min(img_w - clip_x);
        let clip_h = clip_h.min(img_h - clip_y);

        Ok(image.crop_imm(clip_x, clip_y, clip_w, clip_h))
    }

    /// Flip image horizontally
    pub fn flip_horizontal(image: &DynamicImage) -> DynamicImage {
        image.fliph()
    }

    /// Flip image vertically
    pub fn flip_vertical(image: &DynamicImage) -> DynamicImage {
        image.flipv()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image Processor: stateful transformation manager
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks transformation state for an image (flip directions, rotation, max dimension).
/// Provides methods to apply transforms and retrieve the current processed image.
#[derive(Clone)]
pub struct ImageProcessor {
    /// Original/base image
    original: DynamicImage,
    /// Path to original file (if loaded from file)
    path: Option<std::path::PathBuf>,
    /// Whether image is flipped horizontally
    flip_h: bool,
    /// Whether image is flipped vertically
    flip_v: bool,
    /// Rotation in degrees
    rotation: f32,
    /// Max dimension for auto-resize (0 = no limit)
    max_dimension: u32,
}

impl ImageProcessor {
    /// Create a new processor from a loaded image
    pub fn new(original: DynamicImage, path: Option<std::path::PathBuf>, max_dimension: u32) -> Self {
        Self {
            original,
            path,
            flip_h: false,
            flip_v: false,
            rotation: 0.0,
            max_dimension,
        }
    }

    /// Toggle horizontal flip
    pub fn flip_horizontal(&mut self) {
        self.flip_h = !self.flip_h;
    }

    /// Toggle vertical flip
    pub fn flip_vertical(&mut self) {
        self.flip_v = !self.flip_v;
    }

    /// Set rotation in degrees (absolute, not incremental)
    pub fn set_rotation(&mut self, degrees: f32) {
        self.rotation = degrees;
    }

    /// Rotate by adding degrees (incremental)
    pub fn rotate(&mut self, degrees: f32) {
        self.rotation += degrees;
        // Normalize to [0, 360)
        self.rotation = self.rotation.rem_euclid(360.0);
    }

    /// Set the maximum dimension (0 = no resize)
    pub fn set_max_dimension(&mut self, max_dim: u32) {
        self.max_dimension = max_dim;
    }

    /// Get original image dimensions
    pub fn original_dimensions(&self) -> (u32, u32) {
        self.original.dimensions()
    }

    /// Get original image (unmodified)
    pub fn get_original_image(&self) -> &DynamicImage {
        &self.original
    }

    /// Get current transformation state
    pub fn get_state(&self) -> (bool, bool, f32, u32) {
        (self.flip_h, self.flip_v, self.rotation, self.max_dimension)
    }

    /// Reset all transformations to original state
    pub fn reset(&mut self) {
        self.flip_h = false;
        self.flip_v = false;
        self.rotation = 0.0;
    }

    /// Compute and return the processed image based on current transformation state
    pub fn get_processed_image(&self) -> Result<DynamicImage> {
        let mut img = self.original.clone();

        // Apply flips
        if self.flip_h {
            img = ImageTransform::flip_horizontal(&img);
        }
        if self.flip_v {
            img = ImageTransform::flip_vertical(&img);
        }

        // Apply rotation if non-zero
        if self.rotation.abs() > f32::EPSILON {
            img = ImageTransform::rotate(&img, self.rotation)?;
        }

        // Apply resize if needed
        if self.max_dimension > 0 {
            let (w, h) = img.dimensions();
            let max_side = w.max(h);
            if max_side > self.max_dimension {
                let factor = (self.max_dimension as f32) / (max_side as f32);
                let new_w = ((w as f32) * factor).round().max(1.0) as u32;
                let new_h = ((h as f32) * factor).round().max(1.0) as u32;
                img = ImageTransform::resize(&img, new_w, new_h)?;
            }
        }

        Ok(img)
    }

    /// Get the path to the original image file (if loaded from file)
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }
}

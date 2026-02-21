//! ML Visualization Utilities
//!
//! Provides visualization functions for ML outputs:
//! - Depth heatmap generation with various colormaps
//! - Segmentation overlay rendering
//! - Face detection annotation

use image::{DynamicImage, Rgba, RgbaImage};
use std::collections::HashSet;

use super::config::{DepthColormap, SegmentationClass, SegmentationConfig};
use super::types::{DepthOutput, FaceDetectionOutput, SegmentationOutput};

// =============================================================================
// Colormap Functions (computed at runtime)
// =============================================================================

/// Generate Turbo colormap value for a given position (0.0 to 1.0)
fn turbo_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    
    // Turbo colormap polynomial approximation
    let r = 34.61 + t * (1172.33 - t * (10793.56 - t * (40788.29 - t * 71637.28)));
    let g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * 1073.77)));
    let b = 138.70 + t * (358.80 + t * (1234.90 - t * (1500.90 - t * 357.89)));
    
    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Generate Viridis colormap value for a given position (0.0 to 1.0)
fn viridis_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    
    let r = 68.0 + t * 187.0;
    let g = 84.0 + t * 171.0;
    let b = 106.0 + t * 149.0;
    
    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Generate Plasma colormap value for a given position (0.0 to 1.0)
fn plasma_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    
    let r = 12.0 + t * 243.0;
    let g = 7.0 + t * 248.0;
    let b = 134.0 + t * 121.0;
    
    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Generate Inferno colormap value for a given position (0.0 to 1.0)
fn inferno_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    
    let r = if t < 0.5 { t * 200.0 } else { 100.0 + (t - 0.5) * 310.0 };
    let g = if t < 0.5 { t * 100.0 } else { 50.0 + (t - 0.5) * 205.0 };
    let b = if t < 0.5 { 4.0 + t * 130.0 } else { 69.0 + (t - 0.5) * 186.0 };
    
    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

/// Get color from colormap at normalized position
fn get_colormap_color(t: f32, colormap: DepthColormap) -> (u8, u8, u8) {
    match colormap {
        DepthColormap::Turbo => turbo_color(t),
        DepthColormap::Viridis => viridis_color(t),
        DepthColormap::Plasma => plasma_color(t),
        DepthColormap::Inferno => inferno_color(t),
        DepthColormap::Grayscale => {
            let v = (t * 255.0) as u8;
            (v, v, v)
        }
    }
}

// =============================================================================
// Segmentation Colors
// =============================================================================

/// Default colors for each segmentation class (RGBA)
pub const SEGMENTATION_COLORS: [(u8, u8, u8, u8); 19] = [
    (0, 0, 0, 255),        // Background - Black
    (204, 153, 153, 255),  // Skin - Light pink
    (139, 90, 43, 255),    // LeftEyebrow - Brown
    (139, 90, 43, 255),    // RightEyebrow - Brown
    (0, 255, 255, 255),    // LeftEye - Cyan
    (0, 255, 255, 255),    // RightEye - Cyan
    (128, 128, 128, 255),  // Eyeglasses - Gray
    (255, 182, 193, 255),  // LeftEar - Light pink
    (255, 182, 193, 255),  // RightEar - Light pink
    (255, 215, 0, 255),    // Earring - Gold
    (255, 140, 0, 255),    // Nose - Orange
    (200, 0, 0, 255),      // InnerMouth - Dark red
    (255, 105, 180, 255),  // UpperLip - Hot pink
    (255, 105, 180, 255),  // LowerLip - Hot pink
    (144, 238, 144, 255),  // Neck - Light green
    (139, 69, 19, 255),    // Hair - Saddle brown
    (255, 255, 255, 255),  // Hat - White
    (255, 215, 0, 255),    // LeftEarringDetail - Gold
    (255, 215, 0, 255),    // RightEarringDetail - Gold
];

// =============================================================================
// Depth Visualization
// =============================================================================

/// Apply colormap to normalized depth values
pub fn apply_colormap(
    values: &[f32],
    width: u32,
    height: u32,
    colormap: DepthColormap,
) -> DynamicImage {
    let mut img = RgbaImage::new(width, height);

    for (i, &v) in values.iter().enumerate() {
        let normalized = v.clamp(0.0, 1.0);
        let (r, g, b) = get_colormap_color(normalized, colormap);
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        img.put_pixel(x, y, Rgba([r, g, b, 255]));
    }

    DynamicImage::ImageRgba8(img)
}

/// Create depth heatmap visualization
pub fn visualize_depth(
    depth: &DepthOutput,
    invert: bool,
    gamma: f32,
    colormap: DepthColormap,
) -> DynamicImage {
    let (min, max) = depth.depth_range;
    let range = (max - min).max(0.0001);

    let processed: Vec<f32> = depth.depth_map
        .iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            let inverted = if invert { 1.0 - normalized } else { normalized };
            inverted.powf(gamma).clamp(0.0, 1.0)
        })
        .collect();

    apply_colormap(&processed, depth.width, depth.height, colormap)
}

// =============================================================================
// Segmentation Visualization
// =============================================================================

/// Create segmentation color overlay
pub fn visualize_segmentation(
    segmentation: &SegmentationOutput,
    original: &DynamicImage,
    config: &SegmentationConfig,
) -> DynamicImage {
    let (width, height) = (segmentation.width, segmentation.height);

    let visible_set: HashSet<_> = if config.visible_classes.is_empty() {
        SegmentationClass::all().into_iter().collect()
    } else {
        config.visible_classes.iter().copied().collect()
    };

    let original_rgba = original.to_rgba8();
    let mut result = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let class_id = segmentation.class_map[idx];

            let orig_pixel = if x < original_rgba.width() && y < original_rgba.height() {
                *original_rgba.get_pixel(x, y)
            } else {
                Rgba([128, 128, 128, 255])
            };

            if let Some(class) = SegmentationClass::from_index(class_id) {
                if visible_set.contains(&class) && (class_id as usize) < SEGMENTATION_COLORS.len() {
                    let (r, g, b, _) = SEGMENTATION_COLORS[class_id as usize];
                    let opacity = config.overlay_opacity;

                    let r = (r as f32 * opacity + orig_pixel[0] as f32 * (1.0 - opacity)) as u8;
                    let g = (g as f32 * opacity + orig_pixel[1] as f32 * (1.0 - opacity)) as u8;
                    let b = (b as f32 * opacity + orig_pixel[2] as f32 * (1.0 - opacity)) as u8;

                    result.put_pixel(x, y, Rgba([r, g, b, 255]));
                } else {
                    result.put_pixel(x, y, orig_pixel);
                }
            } else {
                result.put_pixel(x, y, orig_pixel);
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Create segmentation color-only image (no blending)
pub fn visualize_segmentation_colors(
    segmentation: &SegmentationOutput,
) -> DynamicImage {
    let (width, height) = (segmentation.width, segmentation.height);
    let mut result = RgbaImage::new(width, height);

    for (i, &class_id) in segmentation.class_map.iter().enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;

        if (class_id as usize) < SEGMENTATION_COLORS.len() {
            let (r, g, b, a) = SEGMENTATION_COLORS[class_id as usize];
            result.put_pixel(x, y, Rgba([r, g, b, a]));
        } else {
            result.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
    }

    DynamicImage::ImageRgba8(result)
}

// =============================================================================
// Face Detection Visualization
// =============================================================================

/// Draw face detection results on original image
pub fn visualize_face_detection(
    face: &FaceDetectionOutput,
    original: &DynamicImage,
) -> DynamicImage {
    let mut result = original.to_rgba8();
    let (width, height) = result.dimensions();

    if let Some(ref bounds) = face.bounds {
        let (x, y, w, h) = bounds.to_pixels(width, height);

        for thickness in 0..4u32 {
            for px in x..(x + w).min(width) {
                if y.saturating_add(thickness) < height {
                    result.put_pixel(px, y.saturating_add(thickness), Rgba([0, 255, 0, 255]));
                }
                if y + h > thickness && (y + h - thickness) < height {
                    result.put_pixel(px, y + h - thickness, Rgba([0, 255, 0, 255]));
                }
            }

            for py in y..(y + h).min(height) {
                if x.saturating_add(thickness) < width {
                    result.put_pixel(x.saturating_add(thickness), py, Rgba([0, 255, 0, 255]));
                }
                if x + w > thickness && (x + w - thickness) < width {
                    result.put_pixel(x + w - thickness, py, Rgba([0, 255, 0, 255]));
                }
            }
        }

        for (i, &(lx, ly)) in face.landmarks.iter().enumerate() {
            let px = (lx * width as f32) as u32;
            let py = (ly * height as f32) as u32;

            let color = match i {
                0 | 1 => Rgba([255, 0, 0, 255]),
                2 => Rgba([0, 255, 0, 255]),
                3 | 4 => Rgba([255, 0, 255, 255]),
                _ => Rgba([255, 255, 0, 255]),
            };

            for dx in -2i32..=2 {
                for dy in -2i32..=2 {
                    let draw_x = (px as i32 + dx).max(0) as u32;
                    let draw_y = (py as i32 + dy).max(0) as u32;
                    if draw_x < width && draw_y < height {
                        result.put_pixel(draw_x, draw_y, color);
                    }
                }
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Draw face bounding box with landmarks on original image
pub fn draw_face_annotation(
    original: &DynamicImage,
    bounds: Option<&super::types::FaceBounds>,
    landmarks: &[(f32, f32)],
    confidence: f32,
) -> DynamicImage {
    let face_output = FaceDetectionOutput {
        bounds: bounds.cloned(),
        landmarks: landmarks.to_vec(),
        confidence,
    };
    visualize_face_detection(&face_output, original)
}

// =============================================================================
// Combined Visualization
// =============================================================================

/// Create combined visualization showing all ML results
pub fn visualize_combined(
    original: &DynamicImage,
    face: Option<&FaceDetectionOutput>,
    depth: Option<&DepthOutput>,
    segmentation: Option<&SegmentationOutput>,
    depth_colormap: DepthColormap,
) -> DynamicImage {
    let base = if let Some(seg) = segmentation {
        visualize_segmentation_colors(seg)
    } else {
        original.clone()
    };

    let with_depth = if let Some(dep) = depth {
        let depth_vis = visualize_depth(dep, false, 1.0, depth_colormap);
        blend_images(&base, &depth_vis, 0.3)
    } else {
        base
    };

    if let Some(f) = face {
        visualize_face_detection(f, &with_depth)
    } else {
        with_depth
    }
}

/// Blend two images with given opacity
fn blend_images(bottom: &DynamicImage, top: &DynamicImage, top_opacity: f32) -> DynamicImage {
    let bottom_rgba = bottom.to_rgba8();
    let top_rgba = top.to_rgba8();

    let (width, height) = bottom_rgba.dimensions();
    let mut result = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let bottom_pixel = bottom_rgba.get_pixel(x, y);
            let top_pixel = if x < top_rgba.width() && y < top_rgba.height() {
                top_rgba.get_pixel(x, y)
            } else {
                bottom_pixel
            };

            let r = (bottom_pixel[0] as f32 * (1.0 - top_opacity) + top_pixel[0] as f32 * top_opacity) as u8;
            let g = (bottom_pixel[1] as f32 * (1.0 - top_opacity) + top_pixel[1] as f32 * top_opacity) as u8;
            let b = (bottom_pixel[2] as f32 * (1.0 - top_opacity) + top_pixel[2] as f32 * top_opacity) as u8;

            result.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    DynamicImage::ImageRgba8(result)
}
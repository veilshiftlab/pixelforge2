//! Feature preservation implementation

use super::{FeaturePreserveConfig, EyeSize, DetailLevel};
use crate::ml::{FaceLandmarks, MLResults};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

/// Preserve facial features in the downsampled image
pub fn preserve_features(
    input: &DynamicImage,
    ml_results: &MLResults,
    config: &FeaturePreserveConfig,
) -> Result<DynamicImage> {
    let landmarks = match &ml_results.landmarks {
        Some(l) => l,
        None => return Ok(input.clone()),
    };

    let (width, height) = input.dimensions();
    let mut output = input.to_rgba8();

    // Preserve eyes
    preserve_eyes(&mut output, landmarks, config, width, height)?;

    // Preserve lips
    preserve_lips(&mut output, landmarks, config, width, height)?;

    // Preserve nose
    preserve_nose(&mut output, landmarks, config, width, height)?;

    Ok(DynamicImage::ImageRgba8(output))
}

/// Eye region bounds from landmarks
#[derive(Debug, Clone)]
struct EyeBounds {
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
}

impl EyeBounds {
    fn from_landmarks(landmarks: &FaceLandmarks, is_left: bool) -> Option<Self> {
        // Standard landmark indices (68-point model)
        // Left eye: 36-41, Right eye: 42-47
        let start_idx = if is_left { 36 } else { 42 };
        let end_idx = if is_left { 41 } else { 47 };

        if landmarks.points.len() <= end_idx {
            return None;
        }

        let eye_points: Vec<(f32, f32)> = landmarks.points[start_idx..=end_idx].to_vec();

        let min_x = eye_points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let max_x = eye_points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let min_y = eye_points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let max_y = eye_points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);

        Some(Self {
            center_x: (min_x + max_x) / 2.0,
            center_y: (min_y + max_y) / 2.0,
            width: max_x - min_x,
            height: max_y - min_y,
        })
    }
}

/// Preserve eye features
fn preserve_eyes(
    output: &mut RgbaImage,
    landmarks: &FaceLandmarks,
    config: &FeaturePreserveConfig,
    width: u32,
    height: u32,
) -> Result<()> {
    let left_eye = EyeBounds::from_landmarks(landmarks, true);
    let right_eye = EyeBounds::from_landmarks(landmarks, false);

    // Calculate minimum eye size based on config
    let min_size = match config.eye_size {
        EyeSize::Auto => (width as f32 * 0.06).ceil() as u32,
        EyeSize::Small => 2,
        EyeSize::Medium => 3,
        EyeSize::Large => 4,
    };

    // Process each eye
    for eye_bounds in [left_eye, right_eye].iter().flatten() {
        // Calculate eye region in pixel coordinates
        let eye_width = (eye_bounds.width * width as f32).max(min_size as f32) as u32;
        let eye_height = (eye_bounds.height * height as f32).max(min_size as f32) as u32;

        let center_x = (eye_bounds.center_x * width as f32) as u32;
        let center_y = (eye_bounds.center_y * height as f32) as u32;

        // Draw simplified eye based on detail level
        draw_simplified_eye(
            output,
            center_x,
            center_y,
            eye_width,
            eye_height,
            config.eye_detail,
            config.force_eye_highlights,
            width,
            height,
        );
    }

    Ok(())
}

/// Draw a simplified eye shape
fn draw_simplified_eye(
    output: &mut RgbaImage,
    center_x: u32,
    center_y: u32,
    width: u32,
    height: u32,
    detail: DetailLevel,
    force_highlight: bool,
    img_width: u32,
    img_height: u32,
) {
    // Determine colors based on detail level
    let (white_color, iris_color, pupil_color, highlight_color) = match detail {
        DetailLevel::Minimal => {
            // Just a dark pixel for the eye
            (None, None, Some(Rgba([20, 20, 30, 255])), None)
        }
        DetailLevel::Standard => {
            // White + pupil
            (
                Some(Rgba([240, 240, 245, 255])),
                None,
                Some(Rgba([20, 20, 30, 255])),
                if force_highlight { Some(Rgba([255, 255, 255, 255])) } else { None },
            )
        }
        DetailLevel::Full => {
            // White + iris + pupil + highlight
            (
                Some(Rgba([240, 240, 245, 255])),
                Some(Rgba([80, 100, 140, 255])),
                Some(Rgba([10, 10, 15, 255])),
                Some(Rgba([255, 255, 255, 255])),
            )
        }
    };

    let half_w = (width / 2) as i32;
    let half_h = (height / 2) as i32;

    // Draw white (eye white) - elliptical
    if let Some(white) = white_color {
        for dy in -half_h..=half_h {
            for dx in -half_w..=half_w {
                let px = center_x as i32 + dx;
                let py = center_y as i32 + dy;

                if px >= 0 && py >= 0 && px < img_width as i32 && py < img_height as i32 {
                    // Elliptical mask
                    let ellipse = if half_w > 0 && half_h > 0 {
                        (dx * dx) as f32 / (half_w * half_w) as f32 +
                        (dy * dy) as f32 / (half_h * half_h) as f32
                    } else {
                        0.0
                    };
                    
                    if ellipse <= 1.0 {
                        output.put_pixel(px as u32, py as u32, white);
                    }
                }
            }
        }
    }

    // Draw iris (if present)
    if let Some(iris) = iris_color {
        let iris_radius = (half_w.min(half_h) * 2 / 3).max(1);
        for dy in -iris_radius..=iris_radius {
            for dx in -iris_radius..=iris_radius {
                if dx * dx + dy * dy <= iris_radius * iris_radius {
                    let px = center_x as i32 + dx;
                    let py = center_y as i32 + dy;

                    if px >= 0 && py >= 0 && px < img_width as i32 && py < img_height as i32 {
                        output.put_pixel(px as u32, py as u32, iris);
                    }
                }
            }
        }
    }

    // Draw pupil
    if let Some(pupil) = pupil_color {
        let pupil_radius = (half_w.min(half_h) / 3).max(1);
        for dy in -pupil_radius..=pupil_radius {
            for dx in -pupil_radius..=pupil_radius {
                if dx * dx + dy * dy <= pupil_radius * pupil_radius {
                    let px = center_x as i32 + dx;
                    let py = center_y as i32 + dy;

                    if px >= 0 && py >= 0 && px < img_width as i32 && py < img_height as i32 {
                        output.put_pixel(px as u32, py as u32, pupil);
                    }
                }
            }
        }
    }

    // Draw highlight
    if let Some(highlight) = highlight_color {
        let hx = center_x as i32 - half_w / 2;
        let hy = center_y as i32 - half_h / 2;

        if hx >= 0 && hy >= 0 && hx < img_width as i32 && hy < img_height as i32 {
            output.put_pixel(hx as u32, hy as u32, highlight);
        }
    }
}

/// Preserve lip features
fn preserve_lips(
    output: &mut RgbaImage,
    landmarks: &FaceLandmarks,
    config: &FeaturePreserveConfig,
    width: u32,
    height: u32,
) -> Result<()> {
    // Standard landmark indices for lips: 48-59 (outer), 60-67 (inner)
    if landmarks.points.len() < 60 {
        return Ok(());
    }

    let lip_points: Vec<(f32, f32)> = landmarks.points[48..=59].to_vec();
    
    let min_x = lip_points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let max_x = lip_points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let min_y = lip_points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y = lip_points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);

    let center_x = ((min_x + max_x) / 2.0 * width as f32) as u32;
    let center_y = ((min_y + max_y) / 2.0 * height as f32) as u32;
    let lip_width = ((max_x - min_x) * width as f32) as u32;
    let lip_height = ((max_y - min_y) * height as f32) as u32;

    match config.lip_detail {
        DetailLevel::Minimal => {
            // Single line for mouth
            let start_x = (min_x * width as f32) as u32;
            let end_x = (max_x * width as f32) as u32;

            let lip_color = Rgba([150, 80, 80, 255]);
            for x in start_x..=end_x.min(width - 1) {
                if center_y < height {
                    output.put_pixel(x, center_y, lip_color);
                }
            }
        }
        DetailLevel::Standard => {
            // Upper and lower lip with slight curve
            let lip_color = Rgba([170, 90, 90, 255]);
            let dark_lip = Rgba([130, 70, 70, 255]);

            // Simple filled rectangle with center line
            let half_w = (lip_width / 2) as i32;
            let half_h = (lip_height / 2) as i32;

            for dy in -half_h..=half_h {
                for dx in -half_w..=half_w {
                    let px = center_x as i32 + dx;
                    let py = center_y as i32 + dy;

                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        let color = if dy == 0 { dark_lip } else { lip_color };
                        output.put_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
        DetailLevel::Full => {
            // Detailed lip shape with shading
            let lip_color_light = Rgba([190, 100, 100, 255]);
            let lip_color_mid = Rgba([170, 85, 85, 255]);
            let lip_color_dark = Rgba([140, 70, 70, 255]);

            let half_w = (lip_width / 2) as i32;
            let half_h = (lip_height / 2) as i32;

            for dy in -half_h..=half_h {
                for dx in -half_w..=half_w {
                    let px = center_x as i32 + dx;
                    let py = center_y as i32 + dy;

                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        // Simple shading based on position
                        let color = if dy < -half_h / 3 {
                            lip_color_light
                        } else if dy > half_h / 3 {
                            lip_color_mid
                        } else {
                            lip_color_dark
                        };
                        output.put_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Preserve nose features
fn preserve_nose(
    output: &mut RgbaImage,
    landmarks: &FaceLandmarks,
    config: &FeaturePreserveConfig,
    width: u32,
    height: u32,
) -> Result<()> {
    // Standard landmark indices for nose: 27-35
    if landmarks.points.len() < 36 {
        return Ok(());
    }

    let nose_tip = landmarks.points[30];
    let nose_left = landmarks.points[31];
    let nose_right = landmarks.points[35];

    if config.distinct_nostrils {
        // Draw nostril indicators
        let nostril_y = ((nose_tip.1 + (nose_tip.1 - landmarks.points[27].1) * 0.3) * height as f32) as u32;
        let left_x = (nose_left.0 * width as f32) as u32;
        let right_x = (nose_right.0 * width as f32) as u32;

        let nostril_color = Rgba([60, 40, 40, 255]);

        if nostril_y < height {
            if left_x < width {
                output.put_pixel(left_x, nostril_y, nostril_color);
            }
            if right_x < width {
                output.put_pixel(right_x, nostril_y, nostril_color);
            }
        }
    }

    match config.nose_detail {
        DetailLevel::Minimal => {
            // No additional rendering
        }
        DetailLevel::Standard => {
            // Simple nose bridge highlight
            let bridge_top = landmarks.points[27];
            let bridge_bottom = nose_tip;

            let highlight_color = Rgba([240, 220, 210, 255]);

            // Draw thin highlight line
            let y_start = (bridge_top.1 * height as f32) as u32;
            let y_end = (bridge_bottom.1 * height as f32) as u32;
            let x_center = (bridge_top.0 * width as f32) as u32;

            for y in y_start..=y_end.min(height - 1) {
                if x_center < width {
                    output.put_pixel(x_center, y, highlight_color);
                }
            }
        }
        DetailLevel::Full => {
            // Detailed nose with shading
            // TODO: Implement detailed nose rendering with proper shading
        }
    }

    Ok(())
}

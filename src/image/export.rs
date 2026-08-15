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

// ─────────────────────────────────────────────────────────────────────────────
// Contact-sheet composition (Phase 1 — U1)
// ─────────────────────────────────────────────────────────────────────────────

/// One row of the contact sheet: a label and either an image or a settings
/// text block. Image is optional so missing intermediates (e.g., ML not yet
/// run) render as a placeholder row instead of aborting the whole sheet.
///
/// ET-1: if `settings` is `Some`, the row renders as a multi-line text block
/// listing key/value configuration pairs instead of an image. Used for the
/// 7th "Settings" row so contact sheets are reproducible.
pub struct ContactSheetRow<'a> {
    pub label:    &'a str,
    pub image:    Option<DynamicImage>,
    /// Optional source-vs-displayed resolution hint, e.g. "518×518 → 1024×1024".
    pub subtitle: Option<&'a str>,
    /// ET-1: if present, render as a settings text block (one "key: value"
    /// per line) instead of an image. When `settings` is `Some`, `image`
    /// should be `None`.
    pub settings: Option<Vec<(String, String)>>,
}

/// Compose a vertical stack of labeled rows into a single PNG.
///
/// Each row is resized to `row_width` using nearest-neighbor (pixel art
/// should never be bilinearly resampled). Rows are stacked top-to-bottom
/// with a small label band above each image.
///
/// Returns a `DynamicImage` suitable for `ImageExporter::export_png`.
pub fn compose_contact_sheet(
    rows: &[ContactSheetRow<'_>],
    row_width: u32,
) -> Result<DynamicImage> {
    const LABEL_HEIGHT:   u32 = 22;
    const SUBTITLE_HEIGHT: u32 = 14;
    const ROW_GAP:        u32 = 8;
    const BG_COLOR:       [u8; 4] = [24, 24, 28, 255];
    const LABEL_COLOR:    [u8; 4] = [40, 40, 46, 255];
    const PLACEHOLDER_BG: [u8; 4] = [44, 44, 52, 255];
    const SETTINGS_BG:    [u8; 4] = [28, 28, 34, 255];
    const SETTINGS_LINE_H: u32 = 11;  // 7px glyph + 4px spacing

    let row_width = row_width.max(64);

    // Pre-compute each row's height.
    // ET-1: added settings_pairs to the layout tuple.
    type RowLayout<'a> = (u32, Option<DynamicImage>, u32, Option<&'a str>, Option<&'a [(String, String)]>);
    let mut row_layout: Vec<RowLayout<'_>> =
        Vec::with_capacity(rows.len());

    let mut total_height: u32 = ROW_GAP; // top padding
    for row in rows {
        let label_h   = LABEL_HEIGHT;
        let subtitle_h = if row.subtitle.is_some() { SUBTITLE_HEIGHT } else { 0 };

        // ET-1: settings row has a different height computation
        if let Some(ref pairs) = row.settings {
            let settings_h = (pairs.len() as u32 * SETTINGS_LINE_H) + 8; // +8 padding
            let row_h = label_h + subtitle_h + settings_h + ROW_GAP;
            row_layout.push((settings_h, None, 0, row.subtitle, Some(pairs)));
            total_height += row_h;
            continue;
        }

        let (img_w, img_h) = match &row.image {
            Some(img) => img.dimensions(),
            None      => (row_width, 64), // placeholder row height
        };
        // Resize preserving aspect ratio to fit row_width
        let scaled_h = if img_w > 0 {
            let aspect = img_h as f32 / img_w as f32;
            (row_width as f32 * aspect).round().max(8.0) as u32
        } else {
            64
        };

        let row_h = label_h + subtitle_h + scaled_h + ROW_GAP;
        row_layout.push((scaled_h, row.image.clone(), img_h, row.subtitle, None));
        total_height += row_h;
    }

    let mut sheet = image::RgbaImage::from_pixel(row_width, total_height, image::Rgba(BG_COLOR));

    let mut y: u32 = ROW_GAP;
    for (i, (content_h, img_opt, src_h, subtitle, settings_pairs)) in row_layout.drain(..).enumerate() {
        let label = rows[i].label;

        // ── Label band ───────────────────────────────────────────────────────
        for ly in 0..LABEL_HEIGHT {
            for x in 0..row_width {
                sheet.put_pixel(x, y + ly, image::Rgba(LABEL_COLOR));
            }
        }
        draw_text_simple(&mut sheet, label, 6, y + 4, row_width);
        y += LABEL_HEIGHT;

        // ── Subtitle band ─────────────────────────────────────────────────────
        if let Some(sub) = subtitle {
            for sy in 0..SUBTITLE_HEIGHT {
                for x in 0..row_width {
                    sheet.put_pixel(x, y + sy, image::Rgba(BG_COLOR));
                }
            }
            draw_text_small(&mut sheet, sub, 6, y + 3, row_width);
            y += SUBTITLE_HEIGHT;
        }

        // ── Content row ───────────────────────────────────────────────────────
        // ET-1: if settings_pairs is present, render as text block instead of image
        if let Some(pairs) = settings_pairs {
            // Fill background
            for sy in 0..content_h {
                for x in 0..row_width {
                    sheet.put_pixel(x, y + sy, image::Rgba(SETTINGS_BG));
                }
            }
            // Render each "key: value" pair as a line
            for (line_i, (key, val)) in pairs.iter().enumerate() {
                let line_y = y + 4 + (line_i as u32) * SETTINGS_LINE_H;
                let text = format!("{}: {}", key, val);
                draw_text_small(&mut sheet, &text, 8, line_y, row_width);
            }
        } else if let Some(img) = img_opt {
            // Resize preserving aspect, nearest-neighbor
            let resized = img.resize(
                row_width,
                content_h,
                image::imageops::FilterType::Nearest,
            );
            let rgba = resized.to_rgba8();
            for sy in 0..content_h.min(rgba.height()) {
                for x in 0..row_width.min(rgba.width()) {
                    let p = rgba.get_pixel(x, sy);
                    sheet.put_pixel(x, y + sy, *p);
                }
            }
            // Pad any shortfall with background
            for sy in rgba.height()..content_h {
                for x in 0..row_width {
                    sheet.put_pixel(x, y + sy, image::Rgba(BG_COLOR));
                }
            }
            let dim_note = format!("{}x{}", img.dimensions().0, src_h);
            let _ = dim_note;
        } else {
            // Placeholder: gray box with "not available" text
            for sy in 0..content_h {
                for x in 0..row_width {
                    sheet.put_pixel(x, y + sy, image::Rgba(PLACEHOLDER_BG));
                }
            }
            draw_text_small(
                &mut sheet,
                "(not available — run analysis / process first)",
                8,
                y + (content_h / 2).saturating_sub(6),
                row_width,
            );
        }
        y += content_h + ROW_GAP;
    }

    Ok(DynamicImage::ImageRgba8(sheet))
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal 5×7 bitmap font for labels (no external font dependency).
// Enough for ASCII printable range — we only need labels here.
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a single ASCII string into the image at (x, y) using a built-in 5×7
/// bitmap font. Uppercases letters; passes digits, punctuation, and spaces
/// through. Clips at the row width.
fn draw_text_simple(
    img: &mut image::RgbaImage,
    text: &str,
    x0: u32,
    y0: u32,
    max_x: u32,
) {
    let color = image::Rgba([235, 235, 240, 255]);
    let mut x = x0;
    for ch in text.chars().take(64) {
        let upper = ch.to_ascii_uppercase();
        if let Some(glyph) = font5x7::glyph(upper) {
            for gy in 0..7u32 {
                for gx in 0..5u32 {
                    // ET-fix: binary literals store row 0 at the MSB end,
                    // so bit position = (6-gy)*5 + (4-gx) to read top-left
                    // as row 0, col 0.
                    let bit = (6 - gy) * 5 + (4 - gx);
                    if (glyph >> bit) & 1 != 0 {
                        let px = x + gx;
                        let py = y0 + gy;
                        if px < max_x && px < img.width() && py < img.height() {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
        x += 6; // 5 px glyph + 1 px spacing
        if x + 5 > max_x { break; }
    }
}

/// Same as `draw_text_simple` but smaller color (gray) — for subtitles.
fn draw_text_small(
    img: &mut image::RgbaImage,
    text: &str,
    x0: u32,
    y0: u32,
    max_x: u32,
) {
    let color = image::Rgba([160, 160, 170, 255]);
    let mut x = x0;
    for ch in text.chars().take(96) {
        let upper = ch.to_ascii_uppercase();
        if let Some(glyph) = font5x7::glyph(upper) {
            for gy in 0..7u32 {
                for gx in 0..5u32 {
                    // ET-fix: same bit-order fix as draw_text_simple
                    let bit = (6 - gy) * 5 + (4 - gx);
                    if (glyph >> bit) & 1 != 0 {
                        let px = x + gx;
                        let py = y0 + gy;
                        if px < max_x && px < img.width() && py < img.height() {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
        x += 6;
        if x + 5 > max_x { break; }
    }
}

/// Inline 5×7 bitmap font for ASCII printables (uppercase + digits + punctuation).
/// Each glyph is a u64 bitmask where the first row (top) is stored at the MSB
/// end and the last row (bottom) at the LSB end. Within each row, column 0
/// (left) is at the higher bit position. Rendering code accesses bit
/// `(6-gy)*5 + (4-gx)` to map screen position (gy=top, gx=left) to the
/// correct glyph bit.
mod font5x7 {
    pub fn glyph(ch: char) -> Option<u64> {
        let bits = match ch {
            ' ' => 0u64,
            '!' => 0b00001_00001_00001_00001_00001_00000_00001,
            '"' => 0b00011_00011_00000_00000_00000_00000_00000,
            '#' => 0b01010_01010_11111_01010_11111_01010_01010,
            '$' => 0b00100_01111_10100_01110_00101_11110_00100,
            '%' => 0b11000_11001_00010_00100_01000_10011_00011,
            '&' => 0b01100_10010_01100_10001_01011_10010_01101,
            '\'' => 0b00001_00001_00000_00000_00000_00000_00000,
            '(' => 0b00001_00010_00010_00010_00010_00010_00001,
            ')' => 0b01000_00100_00100_00100_00100_00100_01000,
            '*' => 0b00000_00100_10101_01110_10101_00100_00000,
            '+' => 0b00000_00100_00100_11111_00100_00100_00000,
            ',' => 0b00000_00000_00000_00000_00000_00001_00010,
            '-' => 0b00000_00000_00000_11111_00000_00000_00000,
            '.' => 0b00000_00000_00000_00000_00000_00000_00001,
            '/' => 0b00001_00010_00010_00100_01000_01000_10000,
            '0' => 0b01110_10001_10011_10101_11001_10001_01110,
            '1' => 0b00100_01100_00100_00100_00100_00100_01110,
            '2' => 0b01110_10001_00001_00010_00100_01000_11111,
            '3' => 0b11111_00010_00100_00010_00001_10001_01110,
            '4' => 0b00010_00110_01010_10010_11111_00010_00010,
            '5' => 0b11111_10000_11110_00001_00001_10001_01110,
            '6' => 0b00110_01000_10000_11110_10001_10001_01110,
            '7' => 0b11111_00001_00010_00100_01000_01000_01000,
            '8' => 0b01110_10001_10001_01110_10001_10001_01110,
            '9' => 0b01110_10001_10001_01111_00001_00010_01100,
            ':' => 0b00000_00000_00001_00000_00000_00001_00000,
            ';' => 0b00000_00000_00001_00000_00000_00001_00010,
            '<' => 0b00010_00100_01000_10000_01000_00100_00010,
            '=' => 0b00000_00000_11111_00000_11111_00000_00000,
            '>' => 0b01000_00100_00010_00001_00010_00100_01000,
            '?' => 0b01110_10001_00001_00010_00100_00000_00100,
            '@' => 0b01110_10001_10111_10101_10111_10000_01110,
            'A' => 0b01110_10001_10001_11111_10001_10001_10001,
            'B' => 0b11110_10001_10001_11110_10001_10001_11110,
            'C' => 0b01110_10001_10000_10000_10000_10001_01110,
            'D' => 0b11110_10001_10001_10001_10001_10001_11110,
            'E' => 0b11111_10000_10000_11110_10000_10000_11111,
            'F' => 0b11111_10000_10000_11110_10000_10000_10000,
            'G' => 0b01110_10001_10000_10111_10001_10001_01111,
            'H' => 0b10001_10001_10001_11111_10001_10001_10001,
            'I' => 0b01110_00100_00100_00100_00100_00100_01110,
            'J' => 0b00111_00010_00010_00010_00010_10010_01100,
            'K' => 0b10001_10010_10100_11000_10100_10010_10001,
            'L' => 0b10000_10000_10000_10000_10000_10000_11111,
            'M' => 0b10001_11011_10101_10101_10001_10001_10001,
            'N' => 0b10001_10001_11001_10101_10011_10001_10001,
            'O' => 0b01110_10001_10001_10001_10001_10001_01110,
            'P' => 0b11110_10001_10001_11110_10000_10000_10000,
            'Q' => 0b01110_10001_10001_10001_10101_10010_01101,
            'R' => 0b11110_10001_10001_11110_10100_10010_10001,
            'S' => 0b01111_10000_10000_01110_00001_00001_11110,
            'T' => 0b11111_00100_00100_00100_00100_00100_00100,
            'U' => 0b10001_10001_10001_10001_10001_10001_01110,
            'V' => 0b10001_10001_10001_10001_10001_01010_00100,
            'W' => 0b10001_10001_10001_10101_10101_11011_10001,
            'X' => 0b10001_10001_01010_00100_01010_10001_10001,
            'Y' => 0b10001_10001_01010_00100_00100_00100_00100,
            'Z' => 0b11111_00001_00010_00100_01000_10000_11111,
            '[' => 0b01110_01000_01000_01000_01000_01000_01110,
            '\\' => 0b10000_01000_01000_00100_00010_00010_00001,
            ']' => 0b01110_00010_00010_00010_00010_00010_01110,
            '^' => 0b00100_01010_10001_00000_00000_00000_00000,
            '_' => 0b00000_00000_00000_00000_00000_00000_11111,
            _ => return None,
        };
        Some(bits)
    }
}

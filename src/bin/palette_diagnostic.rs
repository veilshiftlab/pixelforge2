//! Diagnostic: run the actual pipeline on the real test image and trace
//! colors through every stage. No ML — just palette extraction + downsample
//! + palette snap, which is where the user sees the "retro palette" shift.

use image::{DynamicImage, GenericImageView, Rgba};
use palette::{IntoColor, FromColor, Lab, Srgb};
use pixelforge::processing::{
    generate_palette, apply_palette, palette_mode_downsample,
    Palette, PaletteConfig, PaletteMode, PresetPalette,
};

fn rgb_to_lab(rgba: Rgba<u8>) -> Lab {
    let rgb = Srgb::new(
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    );
    rgb.into_color()
}

fn lab_to_rgb(lab: Lab) -> Rgba<u8> {
    let rgb: Srgb = Srgb::from_color(lab);
    Rgba([
        (rgb.red * 255.0).clamp(0.0, 255.0).round() as u8,
        (rgb.green * 255.0).clamp(0.0, 255.0).round() as u8,
        (rgb.blue * 255.0).clamp(0.0, 255.0).round() as u8,
        255,
    ])
}

fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if max == r {
        (g - b) / d + (if g < b { 6.0 } else { 0.0 })
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h * 60.0, s, l)
}

fn hue_name(h: f32) -> &'static str {
    if h < 15.0 || h >= 345.0 { "RED" }
    else if h < 45.0 { "ORANGE" }
    else if h < 75.0 { "YELLOW" }
    else if h < 105.0 { "YELLOW-GREEN" }
    else if h < 165.0 { "GREEN" }
    else if h < 195.0 { "CYAN" }
    else if h < 255.0 { "BLUE" }
    else if h < 285.0 { "PURPLE" }
    else { "MAGENTA" }
}

/// Sample the most common colors in a region of the image.
fn sample_region(img: &DynamicImage, x0: u32, y0: u32, x1: u32, y1: u32, label: &str) {
    let rgba = img.to_rgba8();
    use std::collections::HashMap;
    let mut counts: HashMap<(u8,u8,u8), u32> = HashMap::new();
    let mut total = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = rgba.get_pixel(x, y);
            let key = (p[0], p[1], p[2]);
            *counts.entry(key).or_insert(0) += 1;
            total += 1;
        }
    }
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  {} region ({}x{}, {} pixels):", label, x1-x0, y1-y0, total);
    for (i, ((r, g, b), count)) in sorted.iter().take(8).enumerate() {
        let (h, s, l) = rgb_to_hsl(*r, *g, *b);
        let lab = rgb_to_lab(Rgba([*r, *g, *b, 255]));
        let pct = 100.0 * *count as f32 / total as f32;
        println!("    [{}] #{:02X}{:02X}{:02X}  count={:>5} ({:>5.1}%)  HSL=({:.0}°, {:.2}, {:.2}) {}  Lab=({:.0},{:.0},{:.0})",
                 i, r, g, b, count, pct, h, s, l, hue_name(h), lab.l, lab.a, lab.b);
    }
}

fn main() {
    println!("=== Real Image Pipeline Diagnostic ===\n");

    let img_path = "/home/z/my-project/upload/Anima_00022_.png";
    let input = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load {}: {}", img_path, e);
            std::process::exit(1);
        }
    };
    let (w, h) = input.dimensions();
    println!("Loaded: {} ({}x{})\n", img_path, w, h);

    // ── Sample the original image in key regions ───────────────────────────
    // The image is 1024x1024. Based on the contact sheet, the character
    // occupies roughly the center. Let's sample several regions.
    println!("── Stage 1: Original image regions ──");
    // Dress region (center-bottom area)
    sample_region(&input, 350, 600, 650, 900, "Dress (center-bottom)");
    // Skin region (face area, upper-center)
    sample_region(&input, 400, 250, 600, 400, "Skin (face)");
    // Hair region (top)
    sample_region(&input, 400, 100, 600, 250, "Hair (top)");
    // Background (corners)
    sample_region(&input, 50, 50, 200, 200, "Background (top-left)");
    sample_region(&input, 850, 850, 1000, 1000, "Background (bottom-right)");

    // ── Extract palette at various sizes ───────────────────────────────────
    for &max_colors in &[12u32, 32, 64, 128] {
        println!("\n── Stage 2: Palette extraction (max_colors={}) ──", max_colors);
        let config = PaletteConfig {
            mode: PaletteMode::Auto,
            max_colors,
            preset: PresetPalette::None,
            custom_colors: Vec::new(),
        };
        let palette: Palette = generate_palette(&input, &config).expect("palette failed");
        println!("  Extracted {} colors:", palette.colors.len());

        // Classify each palette entry by hue
        let mut by_hue: std::collections::HashMap<&str, Vec<(usize, Rgba<u8>)>> = std::collections::HashMap::new();
        for (i, &c) in palette.colors.iter().enumerate() {
            let (hue, _, _) = rgb_to_hsl(c[0], c[1], c[2]);
            let name = hue_name(hue);
            by_hue.entry(name).or_insert_with(Vec::new).push((i, c));
        }

        // Print palette grouped by hue
        for hue_group in ["RED", "ORANGE", "YELLOW", "GREEN", "CYAN", "BLUE", "PURPLE", "MAGENTA"] {
            if let Some(entries) = by_hue.get(hue_group) {
                println!("    {} ({} entries):", hue_group, entries.len());
                for (i, c) in entries {
                    let (h, s, l) = rgb_to_hsl(c[0], c[1], c[2]);
                    let lab = rgb_to_lab(*c);
                    println!("      [{}] #{:02X}{:02X}{:02X}  HSL=({:.0}°,{:.2},{:.2})  Lab=({:.0},{:.0},{:.0})",
                             i, c[0], c[1], c[2], h, s, l, lab.l, lab.a, lab.b);
                }
            }
        }
    }

    // ── Full pipeline: palette_mode_downsample at 32x32 ────────────────────
    println!("\n── Stage 3: Full PaletteMode downsample (32x32) ──");
    let config = PaletteConfig {
        mode: PaletteMode::Auto,
        max_colors: 32,
        preset: PresetPalette::None,
        custom_colors: Vec::new(),
    };
    let palette: Palette = generate_palette(&input, &config).expect("palette failed");
    let out = palette_mode_downsample(&input, &palette, 32, 32).expect("downsample failed");

    // Sample the output in the dress region (proportional to input)
    // Input dress was 350-650, 600-900 in 1024x1024
    // Output is 32x32, so dress region is roughly 11-20, 18-28
    println!("  Output palette colors used in dress region (11-20, 19-28):");
    use std::collections::HashMap;
    let mut counts: HashMap<(u8,u8,u8), u32> = HashMap::new();
    let out_rgba = out.to_rgba8();
    for y in 19..28 {
        for x in 11..20 {
            let p = out_rgba.get_pixel(x, y);
            *counts.entry((p[0], p[1], p[2])).or_insert(0) += 1;
        }
    }
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for ((r, g, b), count) in sorted.iter().take(10) {
        let (h, s, l) = rgb_to_hsl(*r, *g, *b);
        println!("    #{:02X}{:02X}{:02X}  count={}  HSL=({:.0}°,{:.2},{:.2}) {}",
                 r, g, b, count, h, s, l, hue_name(h));
    }

    // ── Trace specific pixels: what does the dress pixel snap to? ──────────
    println!("\n── Stage 4: Trace specific dress pixels through snap ──");
    // Sample a few specific pixels from the dress region in the original
    let rgba = input.to_rgba8();
    let sample_points = [
        (450, 700, "dress center"),
        (500, 750, "dress center 2"),
        (400, 650, "dress left"),
        (550, 800, "dress bottom"),
    ];
    let palette_lab = pixelforge::processing::PaletteLab::from_palette(&palette);
    for (x, y, label) in &sample_points {
        let p = rgba.get_pixel(*x, *y);
        let orig_lab = rgb_to_lab(*p);
        let nearest = palette_lab.nearest_to(orig_lab);
        let (h1, s1, _) = rgb_to_hsl(p[0], p[1], p[2]);
        let (h2, s2, _) = rgb_to_hsl(nearest[0], nearest[1], nearest[2]);
        let dl = orig_lab.l - rgb_to_lab(nearest).l;
        let da = orig_lab.a - rgb_to_lab(nearest).a;
        let db = orig_lab.b - rgb_to_lab(nearest).b;
        let dist = (dl*dl + da*da + db*db).sqrt();
        println!("  {} ({}x{}): orig=#{:02X}{:02X}{:02X} ({:.0}° {}) → nearest=#{:02X}{:02X}{:02X} ({:.0}° {})  ΔE={:.1}",
                 label, x, y,
                 p[0], p[1], p[2], h1, hue_name(h1),
                 nearest[0], nearest[1], nearest[2], h2, hue_name(h2),
                 dist);
    }

    println!("\n=== Diagnostic complete ===");
}

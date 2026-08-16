//! Benchmark: measure each pipeline stage's runtime on the real image.
//! Run with: cargo run --bin bench_pipeline --release

use image::{DynamicImage, GenericImageView};
use std::time::Instant;
use pixelforge::processing::{
    generate_palette, palette_mode_downsample, apply_palette,
    compute_combined_importance_map, weighted_downsample,
    Palette, PaletteConfig, PaletteMode, PresetPalette, TransformConfig,
};

fn main() {
    let img_path = "/home/z/my-project/upload/Anima_00022_.png";
    let input = image::open(img_path).expect("load failed");
    let (w, h) = input.dimensions();
    println!("Image: {}x{} ({} pixels)\n", w, h, w*h);

    // ── Stage A: Importance map (runs at full preprocessed resolution) ──
    let t0 = Instant::now();
    let importance = compute_combined_importance_map(&input, None);
    println!("compute_combined_importance_map: {:.1}ms (no ML)", t0.elapsed().as_secs_f32() * 1000.0);

    // ── Stage B: Palette extraction ──
    let t0 = Instant::now();
    let config = PaletteConfig {
        mode: PaletteMode::Auto,
        max_colors: 32,
        preset: PresetPalette::None,
        custom_colors: Vec::new(),
    };
    let palette: Palette = generate_palette(&input, &config).expect("palette failed");
    println!("generate_palette (32 colors): {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // ── Stage C: PaletteMode downsample ──
    let t0 = Instant::now();
    let out = palette_mode_downsample(&input, &palette, 32, 32).expect("downsample failed");
    println!("palette_mode_downsample (32x32): {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // ── Stage D: Weighted downsample ──
    let t0 = Instant::now();
    let _wout = weighted_downsample(&input, &importance, 32, 32).expect("weighted failed");
    println!("weighted_downsample (32x32): {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // ── Stage E: apply_palette on full-res ──
    let t0 = Instant::now();
    let _q = apply_palette(&input, &palette).expect("apply failed");
    println!("apply_palette (full-res 1024x1024): {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // ── Stage F: apply_palette on 32x32 ──
    let t0 = Instant::now();
    let _q = apply_palette(&out, &palette).expect("apply failed");
    println!("apply_palette (32x32): {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    println!("\n=== Summary ===");
    println!("If the app freezes for several seconds, the culprit is the slowest stage above.");
}

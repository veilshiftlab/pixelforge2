//! Determinism test — runs the pipeline twice on the same input and verifies
//! the outputs are byte-identical. Run with: `cargo run --bin determinism_test`

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use pixelforge::ml::MLResults;
use pixelforge::processing::pipeline::{PipelineInput, run};
use pixelforge::processing::{
    DepthToFlatConfig, EdgeConfig, PaletteConfig, PaletteMode,
    SlicConfig, TransformConfig,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .init();

    println!("=== Determinism Test ===\n");

    let image = generate_test_image(128, 128);
    let depth_map = synthesize_depth_map(128, 128);
    let edge_map = synthesize_edge_map(&image);
    let slic_labels = pixelforge::processing::slic::slic(&image, Some(&depth_map), &SlicConfig::default())
        .expect("SLIC should not fail");

    let ml = MLResults {
        depth_map: Some(depth_map),
        filtered_depth_map: None,
        edge_map: Some(edge_map),
        slic_labels: Some(slic_labels),
        slic_labels_k: Some(5),
        slic_labels_s: Some(0.5),
    };

    let transform = TransformConfig { output_size: 32, ..Default::default() };
    let edges = EdgeConfig::default();
    let palette = PaletteConfig { mode: PaletteMode::Auto, max_colors: 16, ..Default::default() };
    let dtf = DepthToFlatConfig::default();

    let make_input = || PipelineInput {
        image: &image,
        ml_results: Some(&ml),
        transform: &transform,
        depth_to_flat: &dtf,
        edges: &edges,
        palette: &palette,
        output_width: 32,
        output_height: 32,
    };

    // Run the pipeline 3 times
    let out1 = run(&make_input());
    let out2 = run(&make_input());
    let out3 = run(&make_input());

    // Compare bytes
    let bytes1 = out1.image.to_rgba8().into_raw();
    let bytes2 = out2.image.to_rgba8().into_raw();
    let bytes3 = out3.image.to_rgba8().into_raw();

    let palette1 = out1.palette_colors;
    let palette2 = out2.palette_colors;
    let palette3 = out3.palette_colors;

    println!("Run 1: {} image bytes, {} palette colors", bytes1.len(), palette1.len());
    println!("Run 2: {} image bytes, {} palette colors", bytes2.len(), palette2.len());
    println!("Run 3: {} image bytes, {} palette colors", bytes3.len(), palette3.len());

    let img_match_12 = bytes1 == bytes2;
    let img_match_13 = bytes1 == bytes3;
    let pal_match_12 = palette1 == palette2;
    let pal_match_13 = palette1 == palette3;

    println!("\nImage bytes identical (run 1 == run 2): {}", if img_match_12 { "✅ YES" } else { "❌ NO" });
    println!("Image bytes identical (run 1 == run 3): {}", if img_match_13 { "✅ YES" } else { "❌ NO" });
    println!("Palette identical (run 1 == run 2):      {}", if pal_match_12 { "✅ YES" } else { "❌ NO" });
    println!("Palette identical (run 1 == run 3):      {}", if pal_match_13 { "✅ YES" } else { "❌ NO" });

    if img_match_12 && img_match_13 && pal_match_12 && pal_match_13 {
        println!("\n✅ Pipeline is deterministic — same input always produces same output.");
    } else {
        println!("\n❌ Pipeline is NOT deterministic — outputs differ between runs.");
        if !img_match_12 {
            let diffs = bytes1.iter().zip(bytes2.iter()).filter(|(a, b)| a != b).count();
            println!("   Image bytes differing (run 1 vs 2): {} / {}", diffs, bytes1.len());
        }
        std::process::exit(1);
    }
}

fn generate_test_image(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (w + h) as f32) * 255.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn synthesize_depth_map(w: u32, h: u32) -> Vec<f32> {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let mut depth = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / cx;
            let dy = (y as f32 - cy) / cy;
            let dist = (dx * dx + dy * dy).sqrt();
            depth.push(dist.clamp(0.0, 1.0));
        }
    }
    depth
}

fn synthesize_edge_map(image: &DynamicImage) -> Vec<f32> {
    let (w, h) = image.dimensions();
    let gray = image.to_luma8();
    let mut edges = vec![0.0f32; (w * h) as usize];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (gray.get_pixel(x + 1, y)[0] as i32 - gray.get_pixel(x - 1, y)[0] as i32).abs();
            let gy = (gray.get_pixel(x, y + 1)[0] as i32 - gray.get_pixel(x, y - 1)[0] as i32).abs();
            let idx = (y * w + x) as usize;
            edges[idx] = ((gx + gy) as f32 / 510.0).clamp(0.0, 1.0);
        }
    }
    edges
}

//! Headless pipeline test harness (Phase 4).
//!
//! Generates synthetic test images and runs the processing pipeline on them
//! to verify no crashes, correct output dimensions, and reasonable behavior.
//! ML models (Depth-Anything, TEED) are unavailable in this sandbox, so we
//! synthesize depth and edge maps programmatically.
//!
//! Run with: `cargo run --bin pipeline_test`

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use pixelforge::ml::MLResults;
use pixelforge::processing::pipeline::{PipelineInput, run};
use pixelforge::processing::{
    DepthToFlatConfig, EdgeConfig, EdgeMode, OutlineStyle,
    PaletteConfig, PaletteMode, SlicConfig, TransformConfig,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .init();

    println!("=== PixelForge Pipeline Test Harness ===\n");

    let mut all_pass = true;

    // Test 1: Photo portrait (synthetic — gradient with face-like region)
    all_pass &= test_case(
        "Photo Portrait (synthetic)",
        generate_portrait(256, 256),
        DepthToFlatConfig::default(),
        SlicConfig::default(),
    );

    // Test 2: Anime portrait (flat colors with hard edges)
    all_pass &= test_case(
        "Anime Portrait (flat colors)",
        generate_anime_portrait(256, 256),
        DepthToFlatConfig::default(),
        SlicConfig::default(),
    );

    // Test 3: Busy background (Otsu stress test)
    all_pass &= test_case(
        "Busy Background (Otsu stress)",
        generate_busy_background(256, 256),
        DepthToFlatConfig::default(),
        SlicConfig::default(),
    );

    // Test 4: Low-detail image (MAD threshold stress)
    all_pass &= test_case(
        "Low-detail (MAD threshold stress)",
        generate_low_detail(128, 128),
        DepthToFlatConfig::default(),
        SlicConfig::default(),
    );

    // Test 5: High-detail image (TEED threshold stress)
    all_pass &= test_case(
        "High-detail (TEED threshold stress)",
        generate_high_detail(256, 256),
        DepthToFlatConfig::default(),
        SlicConfig::default(),
    );

    // Test 6: Vary SLIC K (Phase 7+: range widened to 5-128)
    for k in [5, 16, 32, 64] {
        let name = format!("SLIC K={}", k);
        all_pass &= test_case(
            &name,
            generate_portrait(128, 128),
            DepthToFlatConfig::default(),
            SlicConfig { k, spatial_weight: 0.5 },
        );
    }

    // Test 7: Vary shading strength
    for strength in [0.0, 0.4, 1.0] {
        let name = format!("Shading strength={}", strength);
        all_pass &= test_case(
            &name,
            generate_portrait(128, 128),
            DepthToFlatConfig { strength, ..Default::default() },
            SlicConfig::default(),
        );
    }

    // Test 8: Phase 4 outline styles (3 variants)
    for style in [OutlineStyle::LocalColorShift, OutlineStyle::Black, OutlineStyle::MaxContrast] {
        let name = format!("Outline style={:?}", style);
        all_pass &= test_case_with_edges(
            &name,
            generate_portrait(128, 128),
            EdgeConfig { outline_style: style, edge_mode: EdgeMode::Outlines, ..Default::default() },
        );
    }

    // Test 9: No ML results (graceful degradation)
    all_pass &= test_no_ml("No ML (degraded mode)", generate_portrait(64, 64));

    println!("\n=== Results ===");
    if all_pass {
        println!("✅ All tests passed — no crashes, correct dimensions.");
    } else {
        println!("❌ Some tests failed — see above.");
        std::process::exit(1);
    }
}

fn test_case(
    name: &str,
    image: DynamicImage,
    dtf_config: DepthToFlatConfig,
    slic_config: SlicConfig,
) -> bool {
    let (w, h) = image.dimensions();

    let depth_map = synthesize_depth_map(w, h);
    let edge_map = synthesize_edge_map(&image);

    let slic_labels = pixelforge::processing::slic::slic(&image, Some(&depth_map), &slic_config)
        .expect("SLIC should not fail");

    let ml = MLResults {
        depth_map: Some(depth_map),
        filtered_depth_map: None,
        edge_map: Some(edge_map),
        slic_labels: Some(slic_labels),
        slic_labels_k: Some(slic_config.k),
        slic_labels_s: Some(slic_config.spatial_weight),
    };

    let transform = TransformConfig {
        output_size: 32,
        ..Default::default()
    };
    let edges = EdgeConfig::default();
    let palette = PaletteConfig {
        mode: PaletteMode::Auto,
        max_colors: 16,
        ..Default::default()
    };

    let input = PipelineInput {
        image: &image,
        ml_results: Some(&ml),
        transform: &transform,
        depth_to_flat: &dtf_config,
        edges: &edges,
        palette: &palette,
        output_width: 32,
        output_height: 32,
    };

    let output = run(&input);
    let (ow, oh) = output.image.dimensions();
    if ow == 32 && oh == 32 {
        println!("  ✅ {} ({}×{} → {}×{}, {} colors)", name, w, h, ow, oh, output.palette_colors.len());
        true
    } else {
        println!("  ❌ {} — wrong output dims: {}×{} (expected 32×32)", name, ow, oh);
        false
    }
}

fn test_case_with_edges(
    name: &str,
    image: DynamicImage,
    edges_config: EdgeConfig,
) -> bool {
    let (w, h) = image.dimensions();
    let depth_map = synthesize_depth_map(w, h);
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
    let palette = PaletteConfig { mode: PaletteMode::Auto, max_colors: 16, ..Default::default() };

    let input = PipelineInput {
        image: &image,
        ml_results: Some(&ml),
        transform: &transform,
        depth_to_flat: &DepthToFlatConfig::default(),
        edges: &edges_config,
        palette: &palette,
        output_width: 32,
        output_height: 32,
    };

    let output = run(&input);
    let (ow, oh) = output.image.dimensions();
    if ow == 32 && oh == 32 {
        println!("  ✅ {} ({}×{} → {}×{}, {} colors)", name, w, h, ow, oh, output.palette_colors.len());
        true
    } else {
        println!("  ❌ {} — wrong output dims: {}×{}", name, ow, oh);
        false
    }
}

fn test_no_ml(name: &str, image: DynamicImage) -> bool {
    let (w, h) = image.dimensions();
    let transform = TransformConfig { output_size: 32, ..Default::default() };
    let edges = EdgeConfig::default();
    let palette = PaletteConfig { mode: PaletteMode::Auto, max_colors: 16, ..Default::default() };

    let input = PipelineInput {
        image: &image,
        ml_results: None,
        transform: &transform,
        depth_to_flat: &DepthToFlatConfig::default(),
        edges: &edges,
        palette: &palette,
        output_width: 32,
        output_height: 32,
    };

    let output = run(&input);
    let (ow, oh) = output.image.dimensions();
    if ow == 32 && oh == 32 {
        println!("  ✅ {} ({}×{} → {}×{}, {} colors)", name, w, h, ow, oh, output.palette_colors.len());
        true
    } else {
        println!("  ❌ {} — wrong output dims: {}×{}", name, ow, oh);
        false
    }
}

// ─── Synthetic image generators ──────────────────────────────────────────────

fn generate_portrait(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / cx;
            let dy = (y as f32 - cy) / cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 0.6 {
                let shade = 1.0 - dist * 0.5;
                img.put_pixel(x, y, Rgba([
                    (220.0 * shade) as u8,
                    (180.0 * shade) as u8,
                    (150.0 * shade) as u8,
                    255,
                ]));
            } else {
                img.put_pixel(x, y, Rgba([40, 40, 60, 255]));
            }
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn generate_anime_portrait(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let color = if y < h / 3 {
                Rgba([80, 120, 200, 255])
            } else if y < h * 2 / 3 {
                Rgba([255, 220, 180, 255])
            } else {
                Rgba([200, 60, 60, 255])
            };
            img.put_pixel(x, y, color);
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn generate_busy_background(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = ((x ^ y) * 7 + (x * y) % 13) % 256;
            img.put_pixel(x, y, Rgba([v as u8, (v / 2) as u8, (255 - v) as u8, 255]));
        }
    }
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / cx;
            let dy = (y as f32 - cy) / cy;
            if dx * dx + dy * dy < 0.25 {
                img.put_pixel(x, y, Rgba([220, 180, 150, 255]));
            }
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn generate_low_detail(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::from_pixel(w, h, Rgba([128, 128, 128, 255]));
    for y in h / 3..h * 2 / 3 {
        for x in w / 3..w * 2 / 3 {
            img.put_pixel(x, y, Rgba([200, 200, 200, 255]));
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn generate_high_detail(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let checker = if (x / 8 + y / 8) % 2 == 0 { 200 } else { 50 };
            let grad = (x + y) as u8 / 2;
            let v = checker.min(grad).max(50);
            img.put_pixel(x, y, Rgba([v, v, v, 255]));
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

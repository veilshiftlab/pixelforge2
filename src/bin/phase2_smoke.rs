// Quick smoke test: drive the pipeline with a known-shaped edge map and
// verify the skeletonization + classification code doesn't panic.
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use pixelforge::ml::MLResults;
use pixelforge::processing::pipeline::{PipelineInput, run};
use pixelforge::processing::{
    DepthToFlatConfig, EdgeConfig, EdgeMode, OutlineStyle,
    PaletteConfig, PaletteMode, SlicConfig, TransformConfig,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    println!("=== Phase 2 Smoke Test ===\n");

    let image = generate_test_image(128, 128);
    let depth_map = synthesize_depth_map(128, 128);
    let edge_map = synthesize_strong_edge_map(128, 128);
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

    let mut all_pass = true;

    // Test all 4 EdgeMode variants
    for mode in [EdgeMode::None, EdgeMode::Outlines, EdgeMode::Internal, EdgeMode::Both] {
        let edges = EdgeConfig { edge_mode: mode, ..Default::default() };
        all_pass &= test_one(&image, &ml, edges, "default");
    }

    // Test all 3 OutlineStyle variants (Phase 7: AutoContrastWithHueShift removed → 2 variants)
    for style in [OutlineStyle::AutoContrast, OutlineStyle::Black] {
        let edges = EdgeConfig { outline_style: style, edge_mode: EdgeMode::Both, ..Default::default() };
        all_pass &= test_one(&image, &ml, edges, "style");
    }

    // Test AA on/off
    for aa in [false, true] {
        let edges = EdgeConfig { anti_alias_edges: aa, edge_mode: EdgeMode::Both, ..Default::default() };
        all_pass &= test_one(&image, &ml, edges, "AA");
    }

    // Test all 4 thicknesses
    for t in [1, 2, 3, 4] {
        let edges = EdgeConfig { thickness: t, edge_mode: EdgeMode::Both, ..Default::default() };
        all_pass &= test_one(&image, &ml, edges, "thickness");
    }

    // Test threshold sweep
    for thr in [0.1, 0.3, 0.5, 0.7] {
        let edges = EdgeConfig { teed_threshold: thr, edge_mode: EdgeMode::Both, ..Default::default() };
        all_pass &= test_one(&image, &ml, edges, "threshold");
    }

    println!("\n=== Result ===");
    if all_pass { println!("✅ All Phase 2 smoke tests passed."); }
    else { println!("❌ Some failed."); std::process::exit(1); }
}

fn test_one(image: &DynamicImage, ml: &MLResults, edges: EdgeConfig, label: &str) -> bool {
    let transform = TransformConfig { output_size: 32, ..Default::default() };
    let palette = PaletteConfig { mode: PaletteMode::Auto, max_colors: 16, ..Default::default() };
    let dtf = DepthToFlatConfig::default();
    let input = PipelineInput {
        image, ml_results: Some(ml), transform: &transform,
        depth_to_flat: &dtf, edges: &edges, palette: &palette,
        output_width: 32, output_height: 32,
    };
    let out = run(&input);
    let (ow, oh) = out.image.dimensions();
    if ow == 32 && oh == 32 {
        println!("  ✅ {:?} {} → {}×{}, {} colors", edges.edge_mode, label, ow, oh, out.palette_colors.len());
        true
    } else {
        println!("  ❌ {:?} {} → {}×{} (wrong)", edges.edge_mode, label, ow, oh);
        false
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

// Strong vertical + horizontal edges to actually trigger skeletonization
fn synthesize_strong_edge_map(w: u32, h: u32) -> Vec<f32> {
    let mut edges = vec![0.0f32; (w * h) as usize];
    // Vertical line at x = w/4
    let x_line = (w / 4) as usize;
    for y in 0..h as usize {
        for dx in 0..3 {
            let x = x_line + dx;
            if x < w as usize { edges[y * w as usize + x] = 0.9; }
        }
    }
    // Horizontal line at y = h/2
    let y_line = (h / 2) as usize;
    for x in 0..w as usize {
        for dy in 0..3 {
            let y = y_line + dy;
            if y < h as usize { edges[y * w as usize + x] = 0.9; }
        }
    }
    // Diagonal
    for i in 0..(w.min(h) as usize) {
        edges[i * w as usize + i] = 0.8;
    }
    edges
}

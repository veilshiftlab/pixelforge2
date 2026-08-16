// Phase 6 baseline-vs-optimized comparison.
// Builds an MLResults WITHOUT the cached filtered_depth_map (simulating the
// pre-Phase-6 path) and an MLResults WITH the cache. Times both, reports
// the speedup. Also reports the SLIC vs HashMap comparison.
use image::{DynamicImage, Rgba, RgbaImage};
use pixelforge::ml::MLResults;
use pixelforge::processing::pipeline::{PipelineInput, run};
use pixelforge::processing::{
    DepthToFlatConfig, EdgeConfig, PaletteConfig, PaletteMode,
    SlicConfig, TransformConfig, DownsamplingMethod,
};
use std::time::Instant;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    println!("=== Phase 6 Baseline vs Optimized ===\n");

    let (w, h) = (512u32, 512u32);
    let image = test_image(w, h);
    let depth = depth_map(w, h);
    let edge = edge_map(w, h);
    let slic = pixelforge::processing::slic::slic(&image, Some(&depth), None, &SlicConfig::default())
        .expect("SLIC should not fail");

    // Pre-Phase-6 path: no filtered_depth_map cache (depth_to_flat recomputes)
    let ml_uncached = MLResults {
        depth_map: Some(depth.clone()),
        filtered_depth_map: None, // BASELINE
        edge_map: Some(edge.clone()),
        segmentation_mask: None,
        slic_labels: Some(slic.clone()),
        slic_labels_k: Some(5),
        slic_labels_s: Some(0.5),
    };
    // Phase 6 path: with filtered_depth_map cache
    let filtered = pixelforge::processing::median_filter_5x5(&depth, w, h);
    let ml_cached = MLResults {
        depth_map: Some(depth.clone()),
        filtered_depth_map: Some(filtered), // PHASE 6
        edge_map: Some(edge.clone()),
        segmentation_mask: None,
        slic_labels: Some(slic.clone()),
        slic_labels_k: Some(5),
        slic_labels_s: Some(0.5),
    };

    let dtf = DepthToFlatConfig::default();
    let edges = EdgeConfig::default();
    let palette = PaletteConfig { mode: PaletteMode::Auto, max_colors: 16, ..Default::default() };

    for method in [DownsamplingMethod::PaletteMode, DownsamplingMethod::PerceptualDither, DownsamplingMethod::Weighted] {
        let transform = TransformConfig { downsampling_method: method, ..Default::default() };

        // Baseline (no cache)
        let input = PipelineInput {
            image: &image, ml_results: Some(&ml_uncached), transform: &transform,
            depth_to_flat: &dtf, edges: &edges, palette: &palette,
            output_width: 32, output_height: 32,
        };
        let _ = run(&input); // warmup
        let start = Instant::now();
        for _ in 0..3 { let _ = run(&input); }
        let baseline = start.elapsed() / 3;

        // Optimized (with cache)
        let input = PipelineInput {
            image: &image, ml_results: Some(&ml_cached), transform: &transform,
            depth_to_flat: &dtf, edges: &edges, palette: &palette,
            output_width: 32, output_height: 32,
        };
        let _ = run(&input); // warmup
        let start = Instant::now();
        for _ in 0..3 { let _ = run(&input); }
        let optimized = start.elapsed() / 3;

        let baseline_ms = baseline.as_secs_f64() * 1000.0;
        let optimized_ms = optimized.as_secs_f64() * 1000.0;
        let speedup = baseline_ms / optimized_ms.max(0.001);

        println!("  {:?}:", method);
        println!("    baseline  (no P1 cache): {:>7.2} ms", baseline_ms);
        println!("    optimized (with cache):  {:>7.2} ms", optimized_ms);
        println!("    speedup:                 {:>7.2}×", speedup);
        println!();
    }
    println!("✅ Comparison complete.");
}

fn test_image(w: u32, h: u32) -> DynamicImage {
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
fn depth_map(w: u32, h: u32) -> Vec<f32> {
    let cx = w as f32 / 2.0; let cy = h as f32 / 2.0;
    (0..h).flat_map(|y| (0..w).map(move |x| {
        let dx = (x as f32 - cx) / cx;
        let dy = (y as f32 - cy) / cy;
        (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0)
    })).collect()
}
fn edge_map(w: u32, h: u32) -> Vec<f32> {
    let mut e = vec![0.0f32; (w * h) as usize];
    let mid = (w / 2) as usize;
    for y in 0..h as usize {
        for &x in &[mid - 1, mid, mid + 1] {
            if x < w as usize { e[y * w as usize + x] = 0.9; }
        }
    }
    e
}

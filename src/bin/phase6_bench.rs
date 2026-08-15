// Phase 6 timing benchmark: measure pipeline runtime on a 512×512 image
// (the size where Phase 6 optimizations matter most — large enough that
// the median filter, SLIC brute-force, and k-means all show up).
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
    println!("=== Phase 6 Performance Benchmark ===\n");

    let sizes: [(u32, u32); 3] = [(256, 256), (512, 512), (1024, 1024)];
    let methods = [
        (DownsamplingMethod::PaletteMode,    "PaletteMode   "),
        (DownsamplingMethod::PerceptualDither, "PerceptualDither"),
        (DownsamplingMethod::Weighted,       "Weighted      "),
    ];

    let all_ok = true;  // placeholder for future failure tracking

    for &(w, h) in &sizes {
        let image = test_image(w, h);
        let depth = depth_map(w, h);
        let edge = edge_map(w, h);
        let slic = pixelforge::processing::slic::slic(&image, Some(&depth), &SlicConfig::default())
            .expect("SLIC should not fail");
        // Simulate MLResults WITH filtered_depth_map cache (Phase 6 P1)
        let filtered = pixelforge::processing::median_filter_5x5(&depth, w, h);
        let ml = MLResults {
            depth_map: Some(depth),
            filtered_depth_map: Some(filtered),
            edge_map: Some(edge),
            slic_labels: Some(slic),
            slic_labels_k: Some(5),
            slic_labels_s: Some(0.5),
        };
        let dtf = DepthToFlatConfig::default();
        let edges = EdgeConfig::default();
        let palette = PaletteConfig { mode: PaletteMode::Auto, max_colors: 16, ..Default::default() };

        for &(method, method_name) in &methods {
            let transform = TransformConfig { downsampling_method: method, ..Default::default() };
            let input = PipelineInput {
                image: &image, ml_results: Some(&ml), transform: &transform,
                depth_to_flat: &dtf, edges: &edges, palette: &palette,
                output_width: 32, output_height: 32,
            };

            // Warm-up (cache JIT/branch predictor)
            let _ = run(&input);

            let n_runs = 3;
            let start = Instant::now();
            for _ in 0..n_runs {
                let _ = run(&input);
            }
            let elapsed = start.elapsed() / n_runs;
            let ms = elapsed.as_secs_f64() * 1000.0;

            println!("  {}×{}  method={}  →  {:>7.2} ms / run (avg of {})",
                w, h, method_name, ms, n_runs);
        }
        println!();
    }

    if all_ok {
        println!("✅ All benchmarks completed.");
    } else {
        println!("❌ Some benchmarks failed.");
        std::process::exit(1);
    }
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

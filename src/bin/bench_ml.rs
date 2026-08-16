//! Benchmark ML-related stages.

use image::{DynamicImage, GenericImageView};
use std::time::Instant;
use pixelforge::processing::{median_filter_5x5, SlicConfig};
use pixelforge::processing::slic::slic;

fn main() {
    let img_path = "/home/z/my-project/upload/Anima_00022_.png";
    let input = image::open(img_path).expect("load failed");
    let (w, h) = input.dimensions();
    println!("Image: {}x{}", w, h);

    // Synthesize a fake depth map for benchmarking (we don't have ML models here)
    let n = (w * h) as usize;
    let depth_map: Vec<f32> = (0..n).map(|i| {
        let x = (i % w as usize) as f32 / w as f32;
        let y = (i / w as usize) as f32 / h as f32;
        ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt()
    }).collect();

    // ── Median filter 5x5 on depth map ──
    let t0 = Instant::now();
    let _filtered = median_filter_5x5(&depth_map, w, h);
    println!("median_filter_5x5 (depth map): {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // ── SLIC superpixels ──
    let t0 = Instant::now();
    let _labels = slic(&input, Some(&depth_map), None, &SlicConfig::default()).expect("slic failed");
    println!("slic (default K={}): {:.1}ms", SlicConfig::default().k, t0.elapsed().as_secs_f32() * 1000.0);

    // ── SLIC with higher K (like the user might use) ──
    for &k in &[32u32, 64, 128] {
        let cfg = SlicConfig { k, spatial_weight: 0.7 };
        let t0 = Instant::now();
        let _labels = slic(&input, Some(&depth_map), None, &cfg).expect("slic failed");
        println!("slic (K={}): {:.1}ms", k, t0.elapsed().as_secs_f32() * 1000.0);
    }
}

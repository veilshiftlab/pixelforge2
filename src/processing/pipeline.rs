//! Processing pipeline — pure data transformation, no egui dependency.
//!
//! This module owns the multi-stage pixel-art conversion pipeline.
//! It takes plain config structs and returns a [`PipelineOutput`],
//! keeping all egui / texture concerns in `app/processing.rs`.
//!
//! # Pipeline stages (post-repurpose)
//!
//! 1. **Depth → flat** — per-region shading at original resolution (must precede
//!    transforms; depth/SLIC maps are keyed to original pixel coordinates)
//! 2. **Transform** — scale, rotate, offset, flip
//! 3. **Importance map** — compute per-pixel weights for downsampling
//! 4. **Downsample** — reduce to target pixel-art resolution
//! 5. **Palette quantize** — map to a limited color palette
//! 6. **Edge render** — draw single-pixel outlines / internal contours
//!
//! The feature-preservation stage (eyes/lips/nose via dlib landmarks) and the
//! `clip_to_face` transform were removed in the pipeline repurpose — both
//! depended on face detection, which is gone now that BiSeNet was dropped.

use crate::ml::MLResults;
use crate::processing::{
    apply_palette, bilinear_downsample,
    compute_combined_importance_map, depth_to_flat, draw_edges, generate_palette,
    nearest_neighbor_downsample, palette_mode_downsample, weighted_downsample,
    DepthToFlatConfig, DownsamplingMethod, EdgeConfig,
    Palette, PaletteConfig, TransformConfig,
};
use crate::image::ImageTransform;
use image::{DynamicImage, Rgba};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline input / output
// ─────────────────────────────────────────────────────────────────────────────

/// Everything the pipeline needs, assembled by `app/processing.rs`.
pub struct PipelineInput<'a> {
    pub image:            &'a DynamicImage,
    pub ml_results:       Option<&'a MLResults>,
    pub transform:        &'a TransformConfig,
    pub depth_to_flat:    &'a DepthToFlatConfig,
    pub edges:            &'a EdgeConfig,
    pub palette:          &'a PaletteConfig,
    pub output_width:     u32,
    pub output_height:    u32,
}

/// What the pipeline returns — pure image data, no GPU types.
pub struct PipelineOutput {
    /// Final pixel-art image.
    pub image: DynamicImage,
    /// Extracted palette as raw RGBA tuples.
    pub palette_colors: Vec<Rgba<u8>>,
    /// Intermediate: post-transform, pre-downsample.
    pub preprocessed: Option<DynamicImage>,
    /// Intermediate: post depth-to-flat.
    pub flat: Option<DynamicImage>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full pixel-art pipeline.
///
/// All stages are infallible from the caller's perspective: failures fall
/// back gracefully (log a warning, pass the image through unchanged) rather
/// than aborting. Only genuinely unrecoverable errors (OOM, etc.) propagate.
pub fn run(input: &PipelineInput<'_>) -> PipelineOutput {
    let PipelineInput {
        image, ml_results, transform, depth_to_flat: dtf_config,
        edges, palette: pal_config, output_width, output_height,
    } = input;

    // ── 1. Depth → flat color ────────────────────────────────────────────────
    // MUST run before geometric transforms: depth_map and SLIC labels are
    // keyed to the ORIGINAL image dimensions.  Transforming first invalidates
    // all pixel index lookups (y * width + x uses the wrong width).
    let depth_processed = if let Some(ml) = ml_results {
        depth_to_flat(image, ml, dtf_config).unwrap_or_else(|e| {
            log::warn!("depth_to_flat: {e}");
            (*image).clone()
        })
    } else {
        (*image).clone()
    };

    // ── 2. Geometric transforms ───────────────────────────────────────────────
    // Now safe to resize/rotate/flip — depth coloring is baked in at original res.
    let preprocessed = apply_transforms(&depth_processed, transform);

    // `flat` alias kept so later stages read naturally.
    // Clone retained for PipelineOutput debug preview before pipeline consumes it.
    let flat = preprocessed;
    let flat_for_output = flat.clone();

    // ── 2b. Resample ML maps to post-transform dimensions ───────────────────────
    // ML maps (depth, edge) are at the original image resolution. After geometric
    // transforms the image dimensions may differ. Resample so downstream stages
    // (importance map, edge pass) index the correct pixels.
    // Bilinear for depth (continuous data); nearest for edge (probability data).
    let original_dims = (image.width(), image.height());
    let transformed_dims = (flat.width(), flat.height());
    let resampled_ml = resample_ml_maps(*ml_results, original_dims, transformed_dims);

    // ── 3. Importance map ────────────────────────────────────────────────────
    // Uses the resampled depth map (matches `flat` dimensions).
    let importance = compute_combined_importance_map(&flat, resampled_ml.as_ref());

    // ── 4. Palette extraction (from full-res image) ──────────────────────────
    // Palette is extracted from the FULL-RES transformed image, before
    // downsampling. This gives the k-means algorithm the full color
    // distribution to work with, producing a more representative palette
    // than extracting from the already-downsampled image.
    let palette = generate_palette(&flat, pal_config, resampled_ml.as_ref())
        .unwrap_or_else(|_| Palette::new(vec![]));

    // ── 5. Downsample ────────────────────────────────────────────────────────
    let ow = *output_width;
    let oh = *output_height;
    let fallback = |img: &DynamicImage| img.resize_exact(ow, oh, image::imageops::FilterType::Nearest);

    let downsampled = match transform.downsampling_method {
        DownsamplingMethod::PaletteMode => {
            // Pre-quantize at full res, then pick most common palette color per block.
            // This eliminates the "smudge" — every output pixel is a discrete
            // palette color, no averaging.
            palette_mode_downsample(&flat, &palette, ow, oh).unwrap_or_else(|_| fallback(&flat))
        }
        DownsamplingMethod::Weighted =>
            weighted_downsample(&flat, &importance, ow, oh).unwrap_or_else(|_| fallback(&flat)),
        DownsamplingMethod::NearestNeighbor =>
            nearest_neighbor_downsample(&flat, ow, oh).unwrap_or_else(|_| fallback(&flat)),
        DownsamplingMethod::Bilinear =>
            bilinear_downsample(&flat, ow, oh).unwrap_or_else(|_| fallback(&flat)),
    };

    // ── 6. Palette quantization (snap to palette) ───────────────────────────
    // For PaletteMode downsampling, the image is already quantized (each pixel
    // is a palette color). For other methods, snap to palette now.
    let quantized = apply_palette(&downsampled, &palette)
        .unwrap_or_else(|e| { log::warn!("apply_palette: {e}"); downsampled.clone() });

    // ── 7. Edge rendering (outline pass) ────────────────────────────────────
    // Runs AFTER palette quantization so outlines use the palette's own colors
    // (darkest for light pixels, lightest for dark pixels).
    let final_image = draw_edges(&quantized, resampled_ml.as_ref(), transformed_dims, edges, &palette).unwrap_or_else(|e| {
        log::warn!("draw_edges: {e}");
        quantized
    });

    PipelineOutput {
        image: final_image,
        palette_colors: palette.colors,
        preprocessed: None,
        flat: Some(flat_for_output),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML map resampling (Phase 2.4 — dimension mismatch fix)
// ─────────────────────────────────────────────────────────────────────────────

/// Resample ML maps (depth, edge) from original dimensions to transformed
/// dimensions. Bilinear for depth (continuous), nearest for edge (probability).
///
/// Returns `None` if `ml_results` is `None` or no maps need resampling.
/// Returns a new `MLResults` with resampled maps; `slic_labels` is dropped
/// (only needed by `depth_to_flat`, which already ran).
fn resample_ml_maps(
    ml_results: Option<&MLResults>,
    original_dims: (u32, u32),
    target_dims: (u32, u32),
) -> Option<MLResults> {
    let ml = ml_results?;

    // No resampling needed if dimensions match
    if original_dims == target_dims {
        // Still return a shallow copy so downstream stages don't mutate the original
        return Some(MLResults {
            depth_map: ml.depth_map.clone(),
            edge_map: ml.edge_map.clone(),
            slic_labels: None, // not needed downstream
            slic_labels_k: None,
            slic_labels_s: None,
        });
    }

    let resampled_depth = ml.depth_map.as_ref().map(|d| {
        resample_bilinear(d, original_dims, target_dims)
    });

    let resampled_edge = ml.edge_map.as_ref().map(|e| {
        resample_nearest(e, original_dims, target_dims)
    });

    log::debug!(
        "Resampled ML maps: {}x{} → {}x{}",
        original_dims.0, original_dims.1, target_dims.0, target_dims.1
    );

    Some(MLResults {
        depth_map: resampled_depth,
        edge_map: resampled_edge,
        slic_labels: None,
        slic_labels_k: None,
        slic_labels_s: None,
    })
}

/// Bilinear resampling for a flat `Vec<f32>` (continuous data like depth).
fn resample_bilinear(
    data: &[f32],
    orig_dims: (u32, u32),
    target_dims: (u32, u32),
) -> Vec<f32> {
    let (ow, oh) = (orig_dims.0 as usize, orig_dims.1 as usize);
    let (tw, th) = (target_dims.0 as usize, target_dims.1 as usize);

    if ow == tw && oh == th || data.len() < ow * oh {
        return data.to_vec();
    }

    let mut result = vec![0.0f32; tw * th];
    let sx = if tw > 1 { (ow - 1) as f32 / (tw - 1) as f32 } else { 0.0 };
    let sy = if th > 1 { (oh - 1) as f32 / (th - 1) as f32 } else { 0.0 };

    for ty in 0..th {
        for tx in 0..tw {
            let fx = tx as f32 * sx;
            let fy = ty as f32 * sy;
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min(ow - 1);
            let y1 = (y0 + 1).min(oh - 1);
            let dx = fx - x0 as f32;
            let dy = fy - y0 as f32;

            let v00 = data[y0 * ow + x0];
            let v01 = data[y0 * ow + x1];
            let v10 = data[y1 * ow + x0];
            let v11 = data[y1 * ow + x1];

            let v0 = v00 * (1.0 - dx) + v01 * dx;
            let v1 = v10 * (1.0 - dx) + v11 * dx;
            result[ty * tw + tx] = v0 * (1.0 - dy) + v1 * dy;
        }
    }

    result
}

/// Nearest-neighbor resampling for a flat `Vec<f32>` (probability/discrete data
/// like edge maps — bilinear would introduce halos).
fn resample_nearest(
    data: &[f32],
    orig_dims: (u32, u32),
    target_dims: (u32, u32),
) -> Vec<f32> {
    let (ow, oh) = (orig_dims.0 as usize, orig_dims.1 as usize);
    let (tw, th) = (target_dims.0 as usize, target_dims.1 as usize);

    if ow == tw && oh == th || data.len() < ow * oh {
        return data.to_vec();
    }

    let mut result = vec![0.0f32; tw * th];
    let sx = ow as f32 / tw as f32;
    let sy = oh as f32 / th as f32;

    for ty in 0..th {
        for tx in 0..tw {
            let ox = ((tx as f32 + 0.5) * sx).floor() as usize;
            let oy = ((ty as f32 + 0.5) * sy).floor() as usize;
            let ox = ox.min(ow - 1);
            let oy = oy.min(oh - 1);
            result[ty * tw + tx] = data[oy * ow + ox];
        }
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage implementations
// ─────────────────────────────────────────────────────────────────────────────

fn apply_transforms(
    image: &DynamicImage,
    config: &TransformConfig,
) -> DynamicImage {
    let mut current = image.clone();

    if config.scale != 1.0 {
        let nw = ((current.width()  as f32) * config.scale) as u32;
        let nh = ((current.height() as f32) * config.scale) as u32;
        if nw > 0 && nh > 0 {
            current = ImageTransform::resize(&current, nw, nh).unwrap_or(current);
        }
    }
    if config.rotation != 0.0 {
        current = ImageTransform::rotate(&current, config.rotation).unwrap_or(current);
    }
    if config.offset_x != 0.0 || config.offset_y != 0.0 {
        current = ImageTransform::offset(&current, config.offset_x, config.offset_y)
            .unwrap_or(current);
    }
    if config.flip_horizontal {
        current = ImageTransform::flip_horizontal(&current);
    }
    if config.flip_vertical {
        current = ImageTransform::flip_vertical(&current);
    }

    current
}

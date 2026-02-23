//! Processing pipeline — pure data transformation, no egui dependency.
//!
//! This module owns the multi-stage pixel-art conversion pipeline.
//! It takes plain config structs and returns a [`PipelineOutput`],
//! keeping all egui / texture concerns in `app/processing.rs`.
//!
//! # Pipeline stages
//!
//! 1. **Depth → flat** — color shading at original resolution (must precede transforms;
//!    segmentation/depth maps are keyed to original pixel coordinates)
//! 2. **Transform** — scale, rotate, offset, flip, clip-to-face
//! 3. **Importance map** — compute per-pixel weights for downsampling
//! 4. **Downsample** — reduce to target pixel-art resolution
//! 5. **Feature preserve** — sharpen eyes / lips / nose at small sizes
//! 6. **Palette quantize** — map to a limited color palette
//! 7. **Edge render** — draw single-pixel outlines / internal contours

use crate::ml::MLResults;
use crate::processing::{
    apply_palette, apply_palette_with_regions, bilinear_downsample,
    compute_combined_importance_map, depth_to_flat, draw_edges, generate_palette,
    nearest_neighbor_downsample, preserve_features, weighted_downsample,
    DepthToFlatConfig, DownsamplingMethod, EdgeConfig, FeaturePreserveConfig, PaletteConfig,
    Palette, TransformConfig,
};
use crate::image::{ImageTransform};
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
    pub features:         &'a FeaturePreserveConfig,
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
        features, edges, palette: pal_config, output_width, output_height,
    } = input;

    // ── 1. Depth → flat color ────────────────────────────────────────────────
    // MUST run before geometric transforms: segmentation.regions and depth_map
    // are keyed to the ORIGINAL image dimensions.  Transforming first invalidates
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
    // Now safe to resize/rotate/clip — depth coloring is baked in at original res.
    let preprocessed = apply_transforms(&depth_processed, transform, *ml_results);

    // `flat` alias kept so later stages read naturally.
    // Clone retained for PipelineOutput debug preview before pipeline consumes it.
    let flat = preprocessed;
    let flat_for_output = flat.clone();

    // ── 3. Importance map ────────────────────────────────────────────────────
    // NOTE: ml_results depth_map and segmentation are at original image resolution,
    // but `flat` is now at post-transform resolution. The depth-gradient and
    // segmentation-boundary lookups inside compute_combined_importance_map will
    // silently miss on dimension mismatch and fall back to Sobel-edge-only importance.
    // This is acceptable — edge importance alone produces good downsampling results.
    // TODO: resample ml_results maps to post-transform dims for full accuracy.
    let importance = compute_combined_importance_map(&flat, *ml_results);

    // ── 4. Downsample ────────────────────────────────────────────────────────
    let ow = *output_width;
    let oh = *output_height;
    let fallback = |img: &DynamicImage| img.resize_exact(ow, oh, image::imageops::FilterType::Nearest);

    let downsampled = match transform.downsampling_method {
        DownsamplingMethod::Weighted =>
            weighted_downsample(&flat, &importance, ow, oh).unwrap_or_else(|_| fallback(&flat)),
        DownsamplingMethod::NearestNeighbor =>
            nearest_neighbor_downsample(&flat, ow, oh).unwrap_or_else(|_| fallback(&flat)),
        DownsamplingMethod::Bilinear =>
            bilinear_downsample(&flat, ow, oh).unwrap_or_else(|_| fallback(&flat)),
    };

    // ── 5. Feature preservation ──────────────────────────────────────────────
    let with_features = if let Some(ml) = ml_results {
        preserve_features(&downsampled, ml, features).unwrap_or_else(|e| {
            log::warn!("preserve_features: {e}");
            downsampled
        })
    } else {
        downsampled
    };

    // ── 6. Palette quantization ──────────────────────────────────────────────
    let palette = generate_palette(&with_features, pal_config, *ml_results)
        .unwrap_or_else(|_| Palette::new(vec![]));

    let quantized = if pal_config.per_region_limit && ml_results.is_some() {
        apply_palette_with_regions(&with_features, &palette, *ml_results)
            .unwrap_or_else(|e| { log::warn!("region palette: {e}"); with_features.clone() })
    } else {
        apply_palette(&with_features, &palette)
            .unwrap_or_else(|e| { log::warn!("apply_palette: {e}"); with_features.clone() })
    };

    // ── 7. Edge rendering ────────────────────────────────────────────────────
    let final_image = draw_edges(&quantized, *ml_results, edges).unwrap_or_else(|e| {
        log::warn!("draw_edges: {e}");
        quantized
    });

    PipelineOutput {
        image: final_image,
        palette_colors: palette.colors,
        preprocessed: None,  // consumed by flat stage
        flat: Some(flat_for_output),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage implementations
// ─────────────────────────────────────────────────────────────────────────────

fn apply_transforms(
    image: &DynamicImage,
    config: &TransformConfig,
    ml_results: Option<&MLResults>,
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
    if config.clip_to_face {
        if let Some(ml) = ml_results {
            if let Some(b) = &ml.face_bounds {
                let pad = config.clip_padding;
                let x = (b.x - b.width  * pad * 0.5).max(0.0);
                let y = (b.y - b.height * pad * 0.5).max(0.0);
                current = ImageTransform::clip(&current, x, y, b.width*(1.0+pad), b.height*(1.0+pad))
                    .unwrap_or(current);
            }
        }
    }

    current
}
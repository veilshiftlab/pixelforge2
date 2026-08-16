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
    nearest_neighbor_downsample, palette_mode_downsample,
    perceptual_dither_downsample, weighted_downsample,
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
    /// Final pixel-art image (post-downsample + post-edges).
    pub image: DynamicImage,
    /// Extracted palette as raw RGBA tuples.
    pub palette_colors: Vec<Rgba<u8>>,
    /// Intermediate: **post-transform** (scale/rotate/offset/flip applied),
    /// pre-downsample. Useful for verifying transform effects.
    pub preprocessed: Option<DynamicImage>,
    /// Intermediate: **post-depth-to-flat** (shading + bg treatment applied),
    /// pre-transform. Useful for verifying depth-to-flat output before any
    /// geometric manipulation. Set even when ML is absent (returns the
    /// unmodified input in that case).
    pub flat: Option<DynamicImage>,
    /// Phase 8 — Human-readable warnings emitted by silent pipeline
    /// fallbacks (depth_to_flat failure, palette empty, edge render failure,
    /// EdgeMode::Internal SLIC fallback, etc.). Empty when all stages ran
    /// cleanly. The UI surfaces these in a dismissible banner so users see
    /// *why* output is degenerate instead of guessing from logs.
    pub warnings: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full pixel-art pipeline.
///
/// All stages are infallible from the caller's perspective: failures fall
/// back gracefully (log a warning, pass the image through unchanged) rather
/// than aborting. Only genuinely unrecoverable errors (OOM, etc.) propagate.
///
/// Phase 8: every silent fallback appends a human-readable message to
/// `PipelineOutput.warnings`. The UI surfaces these in a dismissible banner
/// so users see *why* output is degenerate instead of guessing from logs.
pub fn run(input: &PipelineInput<'_>) -> PipelineOutput {
    let PipelineInput {
        image, ml_results, transform, depth_to_flat: dtf_config,
        edges, palette: pal_config, output_width, output_height,
    } = input;

    // Phase 8 — collect warnings from silent fallbacks. Each fallback below
    // pushes a human-readable message here in addition to logging.
    let mut warnings: Vec<String> = Vec::new();

    // ── 1. Depth → flat color ────────────────────────────────────────────────
    // MUST run before geometric transforms: depth_map and SLIC labels are
    // keyed to the ORIGINAL image dimensions.  Transforming first invalidates
    // all pixel index lookups (y * width + x uses the wrong width).
    // Result: post-depth-to-flat intermediate (shaded, bg treated, original res).
    //
    // Phase 6 — perf: skip the `(*image).clone()` when ML is absent by
    // returning `Cow<DynamicImage>` so we either borrow or own.
    use std::borrow::Cow;
    let depth_processed: Cow<'_, DynamicImage> = if let Some(ml) = ml_results {
        Cow::Owned(
            depth_to_flat(image, ml, dtf_config).unwrap_or_else(|e| {
                let msg = format!("Depth-to-flat conversion failed: {e}. Using original image.");
                log::warn!("{}", msg);
                warnings.push(msg);
                (*image).clone()
            })
        )
    } else {
        Cow::Borrowed(image)
    };

    // Phase 1 — U6: expose post-depth-to-flat image for the preview tab + contact sheet.
    // Phase 6 — perf: only clone if we own; if borrowed, clone once for the output
    // (this is the unavoidable cost of supporting a preview tab + the actual pipeline).
    let flat_for_output: DynamicImage = depth_processed.clone().into_owned();

    // ── 2. Geometric transforms ───────────────────────────────────────────────
    // Now safe to resize/rotate/flip — depth coloring is baked in at original res.
    // Result: post-transform image (what downsample/palette/edge stages see).
    //
    // Phase 6 — perf: skip the transforms entirely when no transform is configured
    // (scale == 1.0, no rotation, no offset, no flips). Returns a borrow in that case.
    let transform_is_noop = transform.scale == 1.0
        && transform.rotation == 0.0
        && transform.offset_x == 0.0
        && transform.offset_y == 0.0
        && !transform.flip_horizontal
        && !transform.flip_vertical;

    let preprocessed: Cow<'_, DynamicImage> = if transform_is_noop {
        depth_processed.clone()
    } else {
        Cow::Owned(apply_transforms(&depth_processed, transform))
    };

    // Phase 1 — U7: expose post-transform image for the preview tab + contact sheet.
    let preprocessed_for_output: DynamicImage = preprocessed.clone().into_owned();

    // ── 2b. Resample ML maps to post-transform dimensions ───────────────────────
    // ML maps (depth, edge) are at the original image resolution. After geometric
    // transforms the image dimensions may differ. Resample so downstream stages
    // (importance map, edge pass) index the correct pixels.
    // Bilinear for depth (continuous data); nearest for edge (probability data).
    let original_dims = (image.width(), image.height());
    let transformed_dims = (preprocessed.width(), preprocessed.height());
    let resampled_ml = resample_ml_maps(*ml_results, original_dims, transformed_dims);

    // ── 3. Importance map ────────────────────────────────────────────────────
    // Uses the resampled depth map (matches `preprocessed` dimensions).
    let importance = compute_combined_importance_map(&preprocessed, resampled_ml.as_ref());

    // ── 4. Palette extraction (from POST-DTF image, same as downsampling input) ─
    // Palette must be extracted from the same image that downsampling quantizes
    // (i.e. `preprocessed` = post-DTF + post-transform). Otherwise the palette
    // reflects original colors while quantization operates on shaded colors, and
    // palette-snap partially undoes the DTF shading.
    //
    // The old "ET-3/ET-4 fix" extracted the palette from the original image to
    // avoid washed-out colors — but that diagnosis was a false flag. The real
    // cause of washed-out colors was DTF applying ±60 L* shifts (default
    // strength=0.6, implicit ×100 scale). Phase 2 introduces an explicit
    // `l_shift_scale` config (default 40) that caps the shift at a more
    // reasonable ±24 L*, so the palette can safely come from post-DTF.
    let palette = generate_palette(&preprocessed, pal_config)
        .unwrap_or_else(|e| {
            let msg = format!("Palette extraction failed: {e}. Using empty palette.");
            log::warn!("{}", msg);
            warnings.push(msg);
            Palette::new(vec![])
        });

    // Phase 8 — also warn if the palette ended up empty (e.g. Preset mode with
    // `PresetPalette::None`, or Auto mode on a degenerate input). Empty palette
    // makes PaletteMode/PerceptualDither fall back to nearest-neighbor, and
    // makes Black/LocalColorShift edge modes fall back to pure black.
    if palette.colors.is_empty() {
        warnings.push(
            "Palette is empty (no colors extracted or selected). \
             PaletteMode/PerceptualDither will fall back to nearest-neighbor; \
             edge colors will be pure black.".into()
        );
    }

    // ── 5. Downsample ────────────────────────────────────────────────────────
    let ow = *output_width;
    let oh = *output_height;
    // Phase 8 — fallback closure now also pushes a warning so the user sees
    // that their chosen downsampling method failed (e.g. PaletteMode with an
    // empty palette) and was replaced with nearest-neighbor resize.
    // `mut` is required because the closure captures `&mut warnings`.
    let mut fallback = |img: &DynamicImage, method_name: &str| {
        let msg = format!(
            "{method_name} downsampling failed; falling back to nearest-neighbor resize."
        );
        log::warn!("{}", msg);
        warnings.push(msg);
        img.resize_exact(ow, oh, image::imageops::FilterType::Nearest)
    };

    // Phase 6 — perf: `PaletteMode` and `PerceptualDither` both produce
    // already-quantized output (every pixel is a palette color), so we can
    // skip `apply_palette` for them. Other methods need the snap-to-palette pass.
    let downsample_already_quantized = matches!(
        transform.downsampling_method,
        DownsamplingMethod::PaletteMode | DownsamplingMethod::PerceptualDither
    );

    let downsampled = match transform.downsampling_method {
        DownsamplingMethod::PaletteMode => {
            // Pre-quantize at full res, then pick most common palette color per block.
            // Phase 4 — D1a: now with bilateral pre-filter to eliminate noisy
            // alternation on smooth gradients (the source of "smudge").
            palette_mode_downsample(&preprocessed, &palette, ow, oh)
                .unwrap_or_else(|_| fallback(&preprocessed, "PaletteMode"))
        }
        DownsamplingMethod::PerceptualDither => {
            // Phase 4 — D2: area-average downsample + Floyd-Steinberg error
            // diffusion. Clean pixel-art downscaling with organic dithering.
            perceptual_dither_downsample(&preprocessed, &palette, ow, oh)
                .unwrap_or_else(|_| fallback(&preprocessed, "PerceptualDither"))
        }
        DownsamplingMethod::Weighted =>
            weighted_downsample(&preprocessed, &importance, ow, oh)
                .unwrap_or_else(|_| fallback(&preprocessed, "Weighted")),
        DownsamplingMethod::NearestNeighbor =>
            nearest_neighbor_downsample(&preprocessed, ow, oh)
                .unwrap_or_else(|_| fallback(&preprocessed, "NearestNeighbor")),
        DownsamplingMethod::Bilinear =>
            bilinear_downsample(&preprocessed, ow, oh)
                .unwrap_or_else(|_| fallback(&preprocessed, "Bilinear")),
    };

    // ── 6. Palette quantization (snap to palette) ───────────────────────────
    // For PaletteMode/PerceptualDither downsampling, the image is already
    // quantized (each pixel is a palette color) — skip the redundant pass.
    // For other methods, snap to palette now.
    let quantized = if downsample_already_quantized {
        downsampled
    } else {
        apply_palette(&downsampled, &palette)
            .unwrap_or_else(|e| {
                let msg = format!("Palette quantization failed: {e}. Using unquantized image.");
                log::warn!("{}", msg);
                warnings.push(msg);
                downsampled
            })
    };

    // ── 7. Edge rendering (outline pass) ────────────────────────────────────
    // Runs AFTER palette quantization so outlines use the palette's own colors
    // (darkest for light pixels, lightest for dark pixels).
    // Phase 8: passes `&mut warnings` so the Phase 7 Internal-mode SLIC
    // fallback inside `classify_edges` can surface its message to the UI.
    let final_image = draw_edges(
        &quantized, resampled_ml.as_ref(), transformed_dims, edges, &palette,
        &mut warnings,
    ).unwrap_or_else(|e| {
        let msg = format!("Edge rendering failed: {e}. Output has no edges.");
        log::warn!("{}", msg);
        warnings.push(msg);
        quantized
    });

    PipelineOutput {
        image: final_image,
        palette_colors: palette.colors,
        preprocessed: Some(preprocessed_for_output),
        flat: Some(flat_for_output),
        warnings,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML map resampling (Phase 2.4 — dimension mismatch fix)
// ─────────────────────────────────────────────────────────────────────────────

/// Resample ML maps (depth, edge) from original dimensions to transformed
/// dimensions. Bilinear for depth (continuous), nearest for edge (probability).
///
/// Returns `None` if `ml_results` is `None` or no maps need resampling.
/// Returns a new `MLResults` with resampled maps.
///
/// Phase 2 — C9: SLIC labels are now resampled via nearest-neighbor and
/// carried through to `draw_edges`, which uses them for the Internal vs
/// Outlines edge-mode classification. Previously they were dropped here.
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
            filtered_depth_map: ml.filtered_depth_map.clone(),
            edge_map: ml.edge_map.clone(),
            slic_labels: ml.slic_labels.clone(), // Phase 2: carry through
            slic_labels_k: ml.slic_labels_k,
            slic_labels_s: ml.slic_labels_s,
        });
    }

    let resampled_depth = ml.depth_map.as_ref().map(|d| {
        resample_bilinear(d, original_dims, target_dims)
    });

    let resampled_edge = ml.edge_map.as_ref().map(|e| {
        resample_nearest(e, original_dims, target_dims)
    });

    // Phase 2 — C9: resample SLIC labels via nearest-neighbor (labels are
    // discrete — bilinear would produce fractional cluster IDs).
    let resampled_slic = ml.slic_labels.as_ref().map(|l| {
        resample_labels_nearest(l, original_dims, target_dims)
    });

    log::debug!(
        "Resampled ML maps: {}x{} → {}x{}",
        original_dims.0, original_dims.1, target_dims.0, target_dims.1
    );

    Some(MLResults {
        depth_map: resampled_depth,
        // filtered_depth_map stays at original resolution — depth_to_flat
        // already ran before transforms, so this resampled MLResults is
        // only used by the importance map and edge pass, which use the
        // raw `depth_map`.
        filtered_depth_map: None,
        edge_map: resampled_edge,
        slic_labels: resampled_slic,
        slic_labels_k: ml.slic_labels_k,
        slic_labels_s: ml.slic_labels_s,
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

/// Nearest-neighbor resampling for SLIC cluster labels (discrete `Vec<u32>`).
/// Phase 2 — C9: needed so `draw_edges` can do Internal-vs-Outlines
/// classification at the post-transform resolution.
fn resample_labels_nearest(
    labels: &[u32],
    orig_dims: (u32, u32),
    target_dims: (u32, u32),
) -> Vec<u32> {
    let (ow, oh) = (orig_dims.0 as usize, orig_dims.1 as usize);
    let (tw, th) = (target_dims.0 as usize, target_dims.1 as usize);

    if ow == tw && oh == th || labels.len() < ow * oh {
        return labels.to_vec();
    }

    let mut result = vec![0u32; tw * th];
    let sx = ow as f32 / tw as f32;
    let sy = oh as f32 / th as f32;

    for ty in 0..th {
        for tx in 0..tw {
            let ox = ((tx as f32 + 0.5) * sx).floor() as usize;
            let oy = ((ty as f32 + 0.5) * sy).floor() as usize;
            let ox = ox.min(ow - 1);
            let oy = oy.min(oh - 1);
            result[ty * tw + tx] = labels[oy * ow + ox];
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

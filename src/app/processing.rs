//! App-level processing callbacks
//!
//! This module handles only egui-aware concerns:
//! - File dialogs and image loading/exporting
//! - Triggering ML analysis
//! - Calling the pure pipeline (`processing::pipeline::run`) and
//!   converting the result into egui textures + app state
//!
//! Heavy pixel-art pipeline logic lives in `crate::processing::pipeline`.

use super::state::{AspectRatioMode, OutputImage, PixelForgeApp};
use crate::image::{ExportConfig, ImageExporter};
use crate::processing::{ProcessingState, ProcessingStatus};
use crate::processing::pipeline::{PipelineInput};
use eframe::egui::{self, ColorImage};
use image::GenericImageView;

// ─────────────────────────────────────────────────────────────────────────────
// Image I/O
// ─────────────────────────────────────────────────────────────────────────────

pub fn load_image(app: &mut PixelForgeApp, path: &std::path::Path, ctx: &egui::Context) {
    match crate::image::ImageLoader::load_with_processor(path, 1024) {
        Ok(processor) => {
            let img = processor.get_processed_image()
                .unwrap_or_else(|_| processor.get_original_image().clone());
            let texture = upload_texture(ctx, "input_image", &img);

            app.custom_output_width  = img.width().min(512);
            app.custom_output_height = img.height().min(512);

            app.input_image = Some(super::state::InputImage {
                image: img,
                texture,
                path: Some(path.to_path_buf()),
            });
            app.image_processor = Some(processor);

            app.preprocessed_image = None;
            app.flat_color_image   = None;
            app.ml_results         = None;
            app.output_image       = None;
            app.processing         = ProcessingState::Idle;

            log::info!("Loaded: {}", path.display());
        }
        Err(e) => log::error!("Failed to load {}: {}", path.display(), e),
    }
}

pub fn clear_image(app: &mut PixelForgeApp) {
    app.input_image        = None;
    app.image_processor    = None;
    app.preprocessed_image = None;
    app.flat_color_image   = None;
    app.ml_results         = None;
    app.output_image       = None;
    app.processing         = ProcessingState::Idle;
}

/// Refresh the input image texture to reflect current processor transformations
/// Call this after any flip/rotate operation to update the preview
pub fn refresh_input_preview(app: &mut PixelForgeApp, ctx: &egui::Context) {
    if let Some(ref mut input) = app.input_image {
        if let Some(ref processor) = app.image_processor {
            match processor.get_processed_image() {
                Ok(processed_img) => {
                    let texture = upload_texture(ctx, "input_image", &processed_img);
                    input.image = processed_img;
                    input.texture = texture;
                    ctx.request_repaint();
                }
                Err(e) => log::error!("Failed to refresh preview: {}", e),
            }
        }
    }
}

pub fn open_file_dialog(app: &mut PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp"])
        .pick_file()
    {
        app.pending_file_load = Some(path);
    }
}

pub fn export_image(app: &PixelForgeApp) {
    let Some(output) = &app.output_image else { return };
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("PNG",  &["png"])
        .add_filter("JPEG", &["jpg", "jpeg"])
        .save_file()
    {
        let cfg = ExportConfig { scale: app.transform_config.export_scale, ..Default::default() };
        match ImageExporter::export(&output.image, &path, &cfg) {
            Ok(_)  => log::info!("Exported to {}", path.display()),
            Err(e) => log::error!("Export failed: {}", e),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Presets
// ─────────────────────────────────────────────────────────────────────────────

pub fn save_preset(app: &PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Preset", &["pixelforge"])
        .save_file()
    {
        let preset = crate::preset::Preset {
            name:          path.file_stem().and_then(|s| s.to_str()).unwrap_or("Custom").to_string(),
            transform:     app.transform_config.clone(),
            depth_to_flat: app.depth_to_flat_config.clone(),
            edges:         app.edge_config.clone(),
            palette:       app.palette_config.clone(),
        };
        match preset.save(&path) {
            Ok(_)  => log::info!("Preset saved: {}", path.display()),
            Err(e) => log::error!("Preset save failed: {}", e),
        }
    }
}

pub fn load_preset(app: &mut PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Preset", &["pixelforge"])
        .pick_file()
    {
        match crate::preset::Preset::load(&path) {
            Ok(preset) => {
                app.transform_config     = preset.transform;
                app.depth_to_flat_config = preset.depth_to_flat;
                app.edge_config          = preset.edges;
                app.palette_config       = preset.palette;
                app.current_preset       = Some(preset.name);
            }
            Err(e) => log::error!("Preset load failed: {}", e),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output dimensions
// ─────────────────────────────────────────────────────────────────────────────

pub fn get_output_dimensions(app: &PixelForgeApp) -> (u32, u32) {
    match app.aspect_mode {
        AspectRatioMode::Square => {
            let s = app.transform_config.output_size;
            (s, s)
        }
        AspectRatioMode::Preserve => {
            let Some(input) = &app.input_image else {
                let s = app.transform_config.output_size;
                return (s, s);
            };
            let (w, h) = input.image.dimensions();
            let target = app.transform_config.output_size as f32;
            let aspect = w as f32 / h as f32;
            if aspect >= 1.0 {
                let ow = target as u32;
                let oh = (target / aspect).round().max(8.0) as u32;
                (ow, oh)
            } else {
                let oh = target as u32;
                let ow = (target * aspect).round().max(8.0) as u32;
                (ow, oh)
            }
        }
        AspectRatioMode::Custom => {
            (app.custom_output_width.max(8), app.custom_output_height.max(8))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Analysis
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_ml_analysis(app: &mut PixelForgeApp, ctx: &egui::Context) {
    let Some(_input) = &app.input_image else { return };
    let Some(ref processor) = app.image_processor else { return };

    // Use the processed image (with flips/rotations applied) from the ImageProcessor
    let image = match processor.get_processed_image() {
        Ok(img) => img,
        Err(e) => {
            log::error!("Failed to get processed image for ML: {}", e);
            app.processing = ProcessingState::Error(e.to_string());
            ctx.request_repaint();
            return;
        }
    };

    let ml_config = app.ml_config.clone();
    let mgr       = app.model_manager.clone();

    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.05,
        stage: "Running ML analysis…".into(),
    });
    ctx.request_repaint();

    // Spawn a background thread so the UI doesn't freeze during ML inference.
    // Results are sent back via an mpsc channel, polled in `update()`.
    let (tx, rx) = std::sync::mpsc::channel();
    app.ml_analysis_receiver = Some(rx);

    let ctx_clone = ctx.clone();
    std::thread::spawn(move || {
        let result = crate::ml::MLAnalysis::analyze(&image, &ml_config, &mgr);
        let _ = tx.send(result);
        // Wake the UI so it picks up the result immediately
        ctx_clone.request_repaint();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Processing pipeline
// ─────────────────────────────────────────────────────────────────────────────

pub fn process_image(app: &mut PixelForgeApp, ctx: &egui::Context) {
    if app.input_image.is_none() || app.image_processor.is_none() { return; }

    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.05,
        stage: "Starting…".into(),
    });
    ctx.request_repaint();

    let (ow, oh) = get_output_dimensions(app);
    
    // Use the processed image (with flips/rotations applied) from the ImageProcessor
    let input = if let Some(ref processor) = app.image_processor {
        match processor.get_processed_image() {
            Ok(img) => img,
            Err(e) => {
                log::error!("Failed to get processed image: {}", e);
                app.processing = ProcessingState::Error(e.to_string());
                ctx.request_repaint();
                return;
            }
        }
    } else {
        app.input_image.as_ref().unwrap().image.clone()
    };

    // ── Compute SLIC superpixels (cached in ml_results) ───────────────────────
    // SLIC runs on the same processed image the ML analysis used. The label
    // map is cached in ml_results.slic_labels so palette/edge tweaks don't
    // trigger a re-cluster. Recompute when depth_map is available and either
    // the labels are missing or the config changed.
    if let Some(ref mut ml) = app.ml_results {
        let need_recompute = ml.slic_labels.is_none()
            || ml.slic_labels_k != Some(app.slic_config.k)
            || ml.slic_labels_s != Some(app.slic_config.spatial_weight);
        if need_recompute {
            if let Some(ref depth) = ml.depth_map {
                match crate::processing::slic::slic(&input, Some(depth), &app.slic_config) {
                    Ok(labels) => {
                        ml.slic_labels = Some(labels);
                        ml.slic_labels_k = Some(app.slic_config.k);
                        ml.slic_labels_s = Some(app.slic_config.spatial_weight);
                    }
                    Err(e) => {
                        log::warn!("SLIC failed: {e}");
                        ml.slic_labels = None;
                    }
                }
            } else {
                // No depth map — still run SLIC with depth=0.5 fallback
                match crate::processing::slic::slic(&input, None, &app.slic_config) {
                    Ok(labels) => {
                        ml.slic_labels = Some(labels);
                        ml.slic_labels_k = Some(app.slic_config.k);
                        ml.slic_labels_s = Some(app.slic_config.spatial_weight);
                    }
                    Err(e) => {
                        log::warn!("SLIC (no depth) failed: {e}");
                        ml.slic_labels = None;
                    }
                }
            }
        }
    }

    let pipeline_input = PipelineInput {
        image:         &input,
        ml_results:    app.ml_results.as_ref(),
        transform:     &app.transform_config,
        depth_to_flat: &app.depth_to_flat_config,
        edges:         &app.edge_config,
        palette:       &app.palette_config,
        output_width:  ow,
        output_height: oh,
    };

    // ── Progress updates are reported before and after the pipeline.
    // For more granular in-pipeline updates, PipelineInput could carry
    // a progress callback, but that adds complexity for marginal UX gain
    // on a synchronous pipeline.
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.10,
        stage: "Processing…".into(),
    });
    ctx.request_repaint();

    let output = crate::processing::pipeline::run(&pipeline_input);

    // Store intermediates for debugging
    app.preprocessed_image = output.preprocessed;
    app.flat_color_image   = output.flat;

    // Upload final texture
    let texture = upload_texture(ctx, "output_image", &output.image);
    let palette_colors: Vec<egui::Color32> = output.palette_colors.iter()
        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
        .collect();

    app.output_image = Some(OutputImage {
        image:   output.image,
        texture,
        palette: palette_colors,
    });
    app.processing = ProcessingState::Complete;
    ctx.request_repaint();
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn upload_texture(ctx: &egui::Context, name: &str, img: &image::DynamicImage) -> egui::TextureHandle {
    let rgba = img.to_rgba8();
    let pixels: Vec<egui::Color32> = rgba.pixels()
        .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
        .collect();
    // NEAREST filtering is critical for pixel art — linear filtering would
    // interpolate between pixels, producing the "gradient" artifacts the user
    // reported in the preview. This applies to output AND intermediate previews.
    ctx.load_texture(
        name,
        ColorImage { size: [rgba.width() as _, rgba.height() as _], pixels },
        egui::TextureOptions::NEAREST,
    )
}
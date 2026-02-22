//! Image processing functions

use super::state::{PixelForgeApp, AspectRatioMode, OutputImage};
use crate::image::{ImageTransform, ImageExporter, ExportConfig};
use crate::models::ModelManager;
use crate::processing::{
    depth_to_flat, weighted_downsample, bilinear_downsample, nearest_neighbor_downsample,
    compute_combined_importance_map, preserve_features, draw_edges, generate_palette, apply_palette, 
    apply_palette_with_regions, Palette,
    ProcessingState, ProcessingStatus, DownsamplingMethod,
};
use eframe::egui::{self, ColorImage};
use image::GenericImageView;

/// Load an image from a file
pub fn load_image(app: &mut PixelForgeApp, path: &std::path::Path, ctx: &egui::Context) {
    match image::open(path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let size = [rgba.width() as _, rgba.height() as _];
            let pixels: Vec<egui::Color32> = rgba
                .pixels()
                .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
                .collect();

            let color_image = ColorImage { size, pixels };
            let texture = ctx.load_texture("input_image", color_image, egui::TextureOptions::default());

            app.custom_output_width = img.width().min(512);
            app.custom_output_height = img.height().min(512);

            app.input_image = Some(super::state::InputImage {
                image: img,
                texture,
                path: Some(path.to_path_buf()),
            });

            app.preprocessed_image = None;
            app.flat_color_image = None;
            app.ml_results = None;
            app.output_image = None;

            log::info!("Loaded image: {} ({}x{})", path.display(), size[0], size[1]);
        }
        Err(e) => {
            log::error!("Failed to load image: {}", e);
        }
    }
}

/// Clear the current image
pub fn clear_image(app: &mut PixelForgeApp) {
    app.input_image = None;
    app.preprocessed_image = None;
    app.flat_color_image = None;
    app.ml_results = None;
    app.output_image = None;
    app.processing = ProcessingState::Idle;
    log::info!("Cleared current image");
}

/// Open file dialog
pub fn open_file_dialog(app: &mut PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp"])
        .pick_file()
    {
        app.pending_file_load = Some(path);
    }
}

/// Export the output image
pub fn export_image(app: &PixelForgeApp) {
    if let Some(output) = &app.output_image {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG", &["png"])
            .add_filter("JPEG", &["jpg", "jpeg"])
            .save_file()
        {
            let export_config = ExportConfig {
                scale: app.transform_config.export_scale,
                ..Default::default()
            };

            match ImageExporter::export(&output.image, &path, &export_config) {
                Ok(_) => log::info!("Exported to: {}", path.display()),
                Err(e) => log::error!("Export failed: {}", e),
            }
        }
    }
}

/// Save preset
pub fn save_preset(app: &PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Preset", &["pixelforge"])
        .save_file()
    {
        let preset = crate::preset::Preset {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Custom")
                .to_string(),
            transform: app.transform_config.clone(),
            depth_to_flat: app.depth_to_flat_config.clone(),
            features: app.feature_config.clone(),
            edges: app.edge_config.clone(),
            palette: app.palette_config.clone(),
        };

        match preset.save(&path) {
            Ok(_) => log::info!("Saved preset: {}", path.display()),
            Err(e) => log::error!("Failed to save preset: {}", e),
        }
    }
}

/// Load preset
pub fn load_preset(app: &mut PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Preset", &["pixelforge"])
        .pick_file()
    {
        match crate::preset::Preset::load(&path) {
            Ok(preset) => {
                app.transform_config = preset.transform;
                app.depth_to_flat_config = preset.depth_to_flat;
                app.feature_config = preset.features;
                app.edge_config = preset.edges;
                app.palette_config = preset.palette;
                app.current_preset = Some(preset.name);
                log::info!("Loaded preset: {}", path.display());
            }
            Err(e) => log::error!("Failed to load preset: {}", e),
        }
    }
}

/// Update model settings
pub fn update_model_settings(app: &mut PixelForgeApp) {
    let manager = app.model_manager.read();
    app.vram_usage = manager.estimated_vram_usage() / (1024 * 1024);
}

/// Get output dimensions based on aspect mode
pub fn get_output_dimensions(app: &PixelForgeApp) -> (u32, u32) {
    match app.aspect_mode {
        AspectRatioMode::Square => {
            let size = app.transform_config.output_size;
            (size, size)
        }
        AspectRatioMode::Preserve => {
            if let Some(input) = &app.input_image {
                let (w, h) = input.image.dimensions();
                let target_size = app.transform_config.output_size as f32;
                let aspect = w as f32 / h as f32;
                
                if aspect > 1.0 {
                    let out_w = target_size;
                    let out_h = (target_size / aspect).max(8.0) as u32;
                    (out_w as u32, out_h)
                } else {
                    let out_h = target_size;
                    let out_w = (target_size * aspect).max(8.0) as u32;
                    (out_w, out_h as u32)
                }
            } else {
                let size = app.transform_config.output_size;
                (size, size)
            }
        }
        AspectRatioMode::Custom => {
            (app.custom_output_width.max(8), app.custom_output_height.max(8))
        }
    }
}

/// Run ML analysis
pub fn run_ml_analysis(app: &mut PixelForgeApp, ctx: &egui::Context) {
    if app.input_image.is_none() {
        return;
    }

    let image = app.input_image.as_ref().unwrap().image.clone();
    let model_manager = app.model_manager.clone();
    let ml_config = app.ml_config.clone();

    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.1,
        stage: "Running ML analysis...".to_string(),
    });
    ctx.request_repaint();

    match crate::ml::MLAnalysis::analyze(&image, &ml_config, &model_manager) {
        Ok(results) => {
            app.ml_results = Some(results);
            app.processing = ProcessingState::Complete;
            
            let manager = app.model_manager.read();
            app.vram_usage = manager.estimated_vram_usage() / (1024 * 1024);
        }
        Err(e) => {
            log::error!("ML analysis failed: {}", e);
            app.processing = ProcessingState::Error(e.to_string());
        }
    }
    ctx.request_repaint();
}

/// Process image to pixel art
pub fn process_image(app: &mut PixelForgeApp, ctx: &egui::Context) {
    if app.input_image.is_none() {
        return;
    }

    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.05,
        stage: "Preprocessing...".to_string(),
    });
    ctx.request_repaint();

    run_full_pipeline(app, ctx);
}

/// Run the complete processing pipeline
fn run_full_pipeline(app: &mut PixelForgeApp, ctx: &egui::Context) {
    let input = match &app.input_image {
        Some(img) => img.image.clone(),
        None => return,
    };

    // Stage 1: Preprocessing
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.1,
        stage: "Transforming...".to_string(),
    });
    ctx.request_repaint();

    let mut current = input.clone();
    
    // Apply scale transform
    if app.transform_config.scale != 1.0 {
        let new_w = (current.width() as f32 * app.transform_config.scale) as u32;
        let new_h = (current.height() as f32 * app.transform_config.scale) as u32;
        if new_w > 0 && new_h > 0 {
            current = ImageTransform::resize(&current, new_w, new_h).unwrap_or(current);
        }
    }

    // Apply rotation
    if app.transform_config.rotation != 0.0 {
        current = ImageTransform::rotate(&current, app.transform_config.rotation).unwrap_or(current);
    }

    // Apply offset
    if app.transform_config.offset_x != 0.0 || app.transform_config.offset_y != 0.0 {
        current = ImageTransform::offset(&current, 
            app.transform_config.offset_x, 
            app.transform_config.offset_y
        ).unwrap_or(current);
    }

    // Apply flips
    if app.transform_config.flip_horizontal {
        current = ImageTransform::flip_horizontal(&current);
    }
    if app.transform_config.flip_vertical {
        current = ImageTransform::flip_vertical(&current);
    }

    // Clip to face region if enabled
    if app.transform_config.clip_to_face {
        if let Some(ref ml) = app.ml_results {
            if let Some(ref bounds) = ml.face_bounds {
                let padding = app.transform_config.clip_padding;
                let x = (bounds.x - bounds.width * padding / 2.0).max(0.0);
                let y = (bounds.y - bounds.height * padding / 2.0).max(0.0);
                let w = bounds.width * (1.0 + padding);
                let h = bounds.height * (1.0 + padding);
                
                current = ImageTransform::clip(&current, x, y, w, h).unwrap_or(current);
            }
        }
    }

    app.preprocessed_image = Some(current.clone());

    // Stage 2: Depth-to-flat
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.25,
        stage: "Converting depth to flat colors...".to_string(),
    });
    ctx.request_repaint();

    let flat_color = if let Some(ref ml) = app.ml_results {
        depth_to_flat(&current, ml, &app.depth_to_flat_config).unwrap_or_else(|e| {
            log::warn!("Depth-to-flat failed: {}", e);
            current.clone()
        })
    } else {
        current.clone()
    };

    app.flat_color_image = Some(flat_color.clone());

    // Stage 3: Importance map (now includes edge detection)
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.4,
        stage: "Computing importance map...".to_string(),
    });
    ctx.request_repaint();

    let importance = compute_combined_importance_map(&flat_color, app.ml_results.as_ref());

    // Stage 4: Downsampling
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.5,
        stage: "Downsampling...".to_string(),
    });
    ctx.request_repaint();

    let (out_w, out_h) = get_output_dimensions(app);
    
    let downsampled = match app.transform_config.downsampling_method {
        DownsamplingMethod::Weighted => {
            weighted_downsample(&flat_color, &importance, out_w, out_h)
                .unwrap_or_else(|e| {
                    log::warn!("Weighted downsampling failed: {}", e);
                    flat_color.resize(out_w, out_h, image::imageops::FilterType::Nearest)
                })
        }
        DownsamplingMethod::NearestNeighbor => {
            nearest_neighbor_downsample(&flat_color, out_w, out_h)
                .unwrap_or_else(|e| {
                    log::warn!("Nearest neighbor downsampling failed: {}", e);
                    flat_color.resize(out_w, out_h, image::imageops::FilterType::Nearest)
                })
        }
        DownsamplingMethod::Bilinear => {
            bilinear_downsample(&flat_color, out_w, out_h)
                .unwrap_or_else(|e| {
                    log::warn!("Bilinear downsampling failed: {}", e);
                    flat_color.resize(out_w, out_h, image::imageops::FilterType::Nearest)
                })
        }
    };

    // Stage 5: Feature preservation
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.65,
        stage: "Preserving features...".to_string(),
    });
    ctx.request_repaint();

    let with_features = if let Some(ref ml) = app.ml_results {
        preserve_features(&downsampled, ml, &app.feature_config).unwrap_or_else(|e| {
            log::warn!("Feature preservation failed: {}", e);
            downsampled
        })
    } else {
        downsampled
    };

    // Stage 6: Palette quantization
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.75,
        stage: "Applying palette...".to_string(),
    });
    ctx.request_repaint();

    let palette = generate_palette(&with_features, &app.palette_config, app.ml_results.as_ref())
        .unwrap_or_else(|_| Palette::new(vec![]));

    // Use region-aware palette application when ML results are available
    let quantized = if app.ml_results.is_some() && app.palette_config.per_region_limit {
        apply_palette_with_regions(&with_features, &palette, app.ml_results.as_ref())
            .unwrap_or_else(|e| {
                log::warn!("Region-aware palette application failed: {}", e);
                with_features.clone()
            })
    } else {
        apply_palette(&with_features, &palette).unwrap_or_else(|e| {
            log::warn!("Palette application failed: {}", e);
            with_features.clone()
        })
    };

    // Stage 7: Edge enhancement
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.9,
        stage: "Drawing edges...".to_string(),
    });
    ctx.request_repaint();

    let with_edges = draw_edges(&quantized, app.ml_results.as_ref(), &app.edge_config)
        .unwrap_or_else(|e| {
            log::warn!("Edge drawing failed: {}", e);
            quantized
        });

    // Final: Create output
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.95,
        stage: "Finalizing...".to_string(),
    });
    ctx.request_repaint();

    // Create texture
    let rgba = with_edges.to_rgba8();
    let size = [rgba.width() as _, rgba.height() as _];
    let pixels: Vec<egui::Color32> = rgba
        .pixels()
        .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
        .collect();

    let color_image = ColorImage { size, pixels };
    let texture = ctx.load_texture("output_image", color_image, egui::TextureOptions::default());

    let palette_colors: Vec<egui::Color32> = palette.colors.iter()
        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
        .collect();

    app.output_image = Some(OutputImage {
        image: with_edges,
        texture,
        palette: palette_colors,
    });

    app.processing = ProcessingState::Complete;
    ctx.request_repaint();
}

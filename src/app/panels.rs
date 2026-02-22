//! Side and bottom panels implementation

use super::state::{PixelForgeApp, AspectRatioMode};
use crate::config::ModelQuality;
use crate::processing::{EyeSize, DetailLevel, PaletteMode, PresetPalette};
use eframe::egui;
use image::GenericImageView;

/// Draw left controls panel
pub fn draw_left(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::SidePanel::left("controls_panel")
        .default_width(280.0)
        .min_width(250.0)
        .max_width(360.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;

                model_management_section(app, ui, ctx);
                ui.add_space(8.0);
                ml_analysis_section(app, ui, ctx);
                ui.add_space(8.0);
                depth_to_flat_section(app, ui);
                ui.add_space(8.0);
                feature_section(app, ui);
                ui.add_space(8.0);
                edge_section(app, ui);
            });
        });
}

fn model_management_section(app: &mut PixelForgeApp, ui: &mut egui::Ui, _ctx: &egui::Context) {
    ui.collapsing("📦 Model Management", |ui| {
        // Show model status
        let manager = app.model_manager.read();
        let required_models = manager.list_models();
        let missing_models: Vec<_> = required_models.iter().filter(|m| !m.downloaded).collect();
        
        ui.label(format!("Quality: {}", app.config.model_quality.display_name()));
        ui.label(format!("Models directory: {}", manager.models_dir().display()));
        
        ui.separator();
        
        ui.label("Model Status:");
        for model in &required_models {
            ui.horizontal(|ui| {
                if model.downloaded {
                    ui.label(egui::RichText::new("✅").color(egui::Color32::GREEN));
                } else {
                    ui.label(egui::RichText::new("❌").color(egui::Color32::RED));
                }
                ui.label(&model.name);
                ui.label(egui::RichText::new(format!("({:.1} MB)", model.size_mb)).small());
            });
        }
        
        drop(manager); // Release the lock before potential mutable operations
        
        ui.separator();
        
        // Download progress
        let progress = app.model_manager.read().get_download_progress();
        if progress.is_downloading {
            ui.label(format!("Downloading: {}", progress.current_model));
            ui.add(egui::ProgressBar::new(progress.progress).text(format!("{:.0}%", progress.progress * 100.0)));
        } else if let Some(error) = &progress.last_error {
            ui.label(egui::RichText::new(format!("Error: {}", error)).color(egui::Color32::RED));
        }
        
        // HuggingFace Token input
        ui.label("HuggingFace Token:");
        ui.add(egui::TextEdit::singleline(&mut app.huggingface_token)
            .password(true)
            .hint_text("hf_xxx..."));
        
        // Download button
        if !missing_models.is_empty() {
            let total_size: f64 = missing_models.iter().map(|m| m.size_mb).sum();
            let download_text = format!("⬇ Download {} models ({:.1} MB)", missing_models.len(), total_size);
            
            if ui.button(&download_text).clicked() {
                // Set token if provided
                if !app.huggingface_token.is_empty() {
                    app.model_manager.read().set_huggingface_token(app.huggingface_token.clone());
                }
                // Start download
                if let Err(e) = app.model_manager.read().download_all_missing() {
                    log::error!("Download failed: {}", e);
                }
            }
        } else {
            ui.label(egui::RichText::new("✅ All models downloaded").color(egui::Color32::GREEN));
        }
        
        // Available model files
        let available = app.model_manager.read().check_model_files();
        if !available.is_empty() {
            ui.add_space(4.0);
            ui.label(format!("Found {} model files", available.len()));
        }
    });
}

fn ml_analysis_section(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    ui.collapsing("🤖 ML Analysis", |ui| {
        ui.checkbox(&mut app.ml_config.face_detection_enabled, "Face Detection");
        ui.checkbox(&mut app.ml_config.depth_estimation_enabled, "Depth Estimation");
        ui.checkbox(&mut app.ml_config.segmentation_enabled, "Segmentation");

        ui.separator();

        ui.label("Model Quality:");
        egui::ComboBox::from_id_salt("model_quality")
            .selected_text(app.config.model_quality.display_name())
            .show_ui(ui, |ui| {
                for quality in [ModelQuality::Minimal, ModelQuality::Standard, ModelQuality::High] {
                    ui.selectable_value(&mut app.config.model_quality, quality, quality.display_name());
                }
            });

        ui.checkbox(&mut app.config.sequential_processing, "Sequential Mode (Low VRAM)");

        ui.separator();
        
        // ML Thresholds
        ui.label("Detection Thresholds:");
        ui.add(egui::Slider::new(&mut app.ml_config.face_confidence_threshold, 0.1..=0.95)
            .text("Face Confidence"));
        ui.add(egui::Slider::new(&mut app.ml_config.depth_edge_sensitivity, 0.05..=0.5)
            .text("Depth Sensitivity"));
        ui.add(egui::Slider::new(&mut app.ml_config.segmentation_sensitivity, 0.1..=0.9)
            .text("Segmentation Sensitivity"));

        ui.separator();

        let analysis_enabled = app.input_image.is_some() && 
            !matches!(app.processing, crate::processing::ProcessingState::Running(_));

        if ui.add_enabled(analysis_enabled, egui::Button::new("▶ Run ML Analysis")).clicked() {
            super::processing::update_model_settings(app);
            super::processing::run_ml_analysis(app, ctx);
        }
    });
}

fn depth_to_flat_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("🎨 Depth-to-Flat", |ui| {
        ui.label("Color Bands:");
        
        ui.horizontal(|ui| {
            ui.label("Skin:");
            ui.add(egui::DragValue::new(&mut app.depth_to_flat_config.skin_tone_bands).range(1..=16));
        });
        ui.horizontal(|ui| {
            ui.label("Hair:");
            ui.add(egui::DragValue::new(&mut app.depth_to_flat_config.hair_bands).range(1..=16));
        });
        ui.horizontal(|ui| {
            ui.label("Clothes:");
            ui.add(egui::DragValue::new(&mut app.depth_to_flat_config.clothing_bands).range(1..=16));
        });
        ui.horizontal(|ui| {
            ui.label("Background:");
            ui.add(egui::DragValue::new(&mut app.depth_to_flat_config.background_bands).range(1..=8));
        });

        ui.add_space(4.0);
        ui.add(egui::Slider::new(&mut app.depth_to_flat_config.shadow_threshold, 0.0..=1.0).text("Shadow"));
        ui.add(egui::Slider::new(&mut app.depth_to_flat_config.highlight_threshold, 0.0..=1.0).text("Highlight"));
        ui.checkbox(&mut app.depth_to_flat_config.preserve_gradients, "Preserve Gradients");
    });
}

fn feature_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("👁 Feature Preservation", |ui| {
        ui.label("Eye Size:");
        ui.horizontal_wrapped(|ui| {
            for size in [EyeSize::Auto, EyeSize::Small, EyeSize::Medium, EyeSize::Large] {
                if ui.selectable_label(app.feature_config.eye_size == size, format!("{:?}", size)).clicked() {
                    app.feature_config.eye_size = size;
                }
            }
        });

        ui.label("Detail Level:");
        ui.horizontal_wrapped(|ui| {
            for level in [DetailLevel::Minimal, DetailLevel::Standard, DetailLevel::Full] {
                if ui.selectable_label(app.feature_config.eye_detail == level, format!("{:?}", level)).clicked() {
                    app.feature_config.eye_detail = level;
                    app.feature_config.lip_detail = level;
                    app.feature_config.nose_detail = level;
                }
            }
        });

        ui.checkbox(&mut app.feature_config.force_eye_highlights, "Force Eye Highlights");
        ui.checkbox(&mut app.feature_config.distinct_nostrils, "Distinct Nostrils");
    });
}

fn edge_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("✏ Edge Controls", |ui| {
        ui.label("Edge Mode:");
        ui.horizontal_wrapped(|ui| {
            for mode in [
                crate::processing::EdgeMode::None, 
                crate::processing::EdgeMode::Outlines, 
                crate::processing::EdgeMode::Internal, 
                crate::processing::EdgeMode::Both
            ] {
                if ui.selectable_label(app.edge_config.edge_mode == mode, format!("{:?}", mode)).clicked() {
                    app.edge_config.edge_mode = mode;
                }
            }
        });

        ui.add(egui::Slider::new(&mut app.edge_config.thickness, 1..=4).text("Thickness"));
        ui.add(egui::Slider::new(&mut app.edge_config.edge_darkener_strength, 0.0..=1.0).text("Darkener"));
        ui.checkbox(&mut app.edge_config.anti_alias_edges, "Anti-alias Edges");
    });
}

/// Draw right output settings panel
pub fn draw_right(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::SidePanel::right("output_panel")
        .default_width(220.0)
        .min_width(200.0)
        .max_width(300.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                output_size_section(app, ui);
                ui.separator();
                transform_section(app, ui);
                ui.separator();
                export_section(app, ui);
                ui.separator();
                palette_section(app, ui);
                ui.separator();
                preset_section(app, ui);
                palette_display(app, ui);
            });
        });
}

fn output_size_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.heading("📐 Output Settings");
    ui.separator();

    ui.label("Aspect Ratio:");
    ui.horizontal_wrapped(|ui| {
        for (mode, label) in [
            (AspectRatioMode::Square, "Square"),
            (AspectRatioMode::Preserve, "Original"),
            (AspectRatioMode::Custom, "Custom"),
        ] {
            if ui.selectable_label(app.aspect_mode == mode, label).clicked() {
                app.aspect_mode = mode;
            }
        }
    });

    match app.aspect_mode {
        AspectRatioMode::Square => {
            ui.label("Output Size:");
            egui::ComboBox::from_id_salt("output_size")
                .selected_text(format!("{}×{}", app.transform_config.output_size, app.transform_config.output_size))
                .show_ui(ui, |ui| {
                    for size in [16, 24, 32, 48, 64, 96, 128, 192, 256] {
                        ui.selectable_value(&mut app.transform_config.output_size, size, format!("{}×{}", size, size));
                    }
                });
        }
        AspectRatioMode::Preserve => {
            ui.label("Max Dimension:");
            egui::ComboBox::from_id_salt("max_dim")
                .selected_text(format!("{}", app.transform_config.output_size))
                .show_ui(ui, |ui| {
                    for size in [32, 48, 64, 96, 128, 192, 256] {
                        ui.selectable_value(&mut app.transform_config.output_size, size, format!("{}", size));
                    }
                });
        }
        AspectRatioMode::Custom => {
            ui.horizontal(|ui| {
                ui.label("W:");
                ui.add(egui::DragValue::new(&mut app.custom_output_width).range(8..=512));
            });
            ui.horizontal(|ui| {
                ui.label("H:");
                ui.add(egui::DragValue::new(&mut app.custom_output_height).range(8..=512));
            });
        }
    }

    let (ow, oh) = super::processing::get_output_dimensions(app);
    ui.label(format!("Output: {}×{}", ow, oh));
}

fn transform_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.label("Downsampling:");
    egui::ComboBox::from_id_salt("downsample_method")
        .selected_text(format!("{:?}", app.transform_config.downsampling_method))
        .show_ui(ui, |ui| {
            for method in [
                crate::processing::DownsamplingMethod::Weighted,
                crate::processing::DownsamplingMethod::NearestNeighbor,
                crate::processing::DownsamplingMethod::Bilinear,
            ] {
                ui.selectable_value(
                    &mut app.transform_config.downsampling_method,
                    method,
                    format!("{:?}", method)
                );
            }
        });
    
    ui.add(egui::Slider::new(&mut app.transform_config.scale, 0.1..=4.0).text("Input Scale"));
    ui.add(egui::Slider::new(&mut app.transform_config.rotation, -180.0..=180.0).text("Rotation (°)"));
    ui.add(egui::Slider::new(&mut app.transform_config.offset_x, -1.0..=1.0).text("Offset X"));
    ui.add(egui::Slider::new(&mut app.transform_config.offset_y, -1.0..=1.0).text("Offset Y"));
    
    ui.separator();
    ui.label("Flip:");
    ui.horizontal(|ui| {
        ui.checkbox(&mut app.transform_config.flip_horizontal, "Horizontal");
        ui.checkbox(&mut app.transform_config.flip_vertical, "Vertical");
    });
    
    // Clip to face option
    ui.checkbox(&mut app.transform_config.clip_to_face, "Clip to Face")
        .on_hover_text("Crop the image to focus on the detected face region");
    if app.transform_config.clip_to_face {
        ui.add(egui::Slider::new(&mut app.transform_config.clip_padding, 0.0..=1.0).text("Padding"));
    }
}

fn export_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.label("Export Scale:");
    ui.horizontal(|ui| {
        for scale in [1, 2, 4, 8] {
            if ui.selectable_label(app.transform_config.export_scale == scale, format!("{}×", scale)).clicked() {
                app.transform_config.export_scale = scale;
            }
        }
    });

    let export_enabled = app.output_image.is_some();
    if ui.add_enabled(export_enabled, egui::Button::new("💾 Export Image")).clicked() {
        super::processing::export_image(app);
    }
}

fn palette_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.heading("🎨 Palette");
    
    ui.horizontal_wrapped(|ui| {
        for mode in [PaletteMode::Auto, PaletteMode::Preset, PaletteMode::Custom] {
            if ui.selectable_label(app.palette_config.mode == mode, format!("{:?}", mode)).clicked() {
                app.palette_config.mode = mode;
            }
        }
    });

    if matches!(app.palette_config.mode, PaletteMode::Auto) {
        ui.add(egui::Slider::new(&mut app.palette_config.max_colors, 2..=256).text("Max Colors"));
        ui.checkbox(&mut app.palette_config.per_region_limit, "Per-region palette")
            .on_hover_text("Use separate palette for each segmented region (face, hair, etc.)");
    }

    if matches!(app.palette_config.mode, PaletteMode::Preset) {
        ui.label("Preset:");
        egui::ComboBox::from_id_salt("palette_preset")
            .selected_text(format!("{:?}", app.palette_config.preset))
            .show_ui(ui, |ui| {
                for preset in [
                    PresetPalette::GameBoy, 
                    PresetPalette::GameBoyColor, 
                    PresetPalette::PICO8, 
                    PresetPalette::NES, 
                    PresetPalette::DawnBringer32, 
                    PresetPalette::AAP64
                ] {
                    ui.selectable_value(&mut app.palette_config.preset, preset, format!("{:?}", preset));
                }
            });
    }
}

fn preset_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.heading("📁 Presets");
    
    // Built-in presets dropdown
    ui.label("Built-in:");
    egui::ComboBox::from_id_salt("builtin_presets")
        .selected_text(app.current_preset.as_deref().unwrap_or("Select preset..."))
        .show_ui(ui, |ui| {
            for preset in crate::preset::Preset::built_in_presets() {
                if ui.selectable_label(
                    app.current_preset.as_deref() == Some(&preset.name),
                    &preset.name
                ).clicked() {
                    app.transform_config = preset.transform;
                    app.depth_to_flat_config = preset.depth_to_flat;
                    app.feature_config = preset.features;
                    app.edge_config = preset.edges;
                    app.palette_config = preset.palette;
                    app.current_preset = Some(preset.name);
                }
            }
        });
    
    ui.add_space(4.0);
    
    // Save/Load buttons
    ui.horizontal(|ui| {
        if ui.button("Save...").clicked() {
            super::processing::save_preset(app);
        }
        if ui.button("Load...").clicked() {
            super::processing::load_preset(app);
        }
    });
    
    if let Some(ref name) = app.current_preset {
        ui.label(egui::RichText::new(name).small());
    }
}

fn palette_display(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    if let Some(output) = &app.output_image {
        ui.separator();
        ui.label(format!("Generated Palette ({} colors):", output.palette.len()));
        
        let colors_per_row = 8;
        let rows = (output.palette.len() + colors_per_row - 1) / colors_per_row;
        
        for row in 0..rows.min(8) {
            ui.horizontal(|ui| {
                for col in 0..colors_per_row {
                    let idx = row * colors_per_row + col;
                    if idx < output.palette.len() {
                        let color = output.palette[idx];
                        let (r, g, b) = (color.r(), color.g(), color.b());
                        
                        let response = ui.add(
                            egui::Label::new(
                                egui::RichText::new("  ")
                                    .background_color(egui::Color32::from_rgb(r, g, b))
                            )
                        );
                        
                        response.on_hover_text(format!("RGB({},{},{})", r, g, b));
                    }
                }
            });
        }
        
        if output.palette.len() > 64 {
            ui.label(format!("... and {} more", output.palette.len() - 64));
        }
    }
}

/// Draw bottom processing panel
pub fn draw_bottom(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("processing_panel")
        .default_height(60.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let (progress, status_text) = match &app.processing {
                    crate::processing::ProcessingState::Idle => (0.0, "Ready".to_string()),
                    crate::processing::ProcessingState::Running(status) => (status.progress, status.stage.clone()),
                    crate::processing::ProcessingState::Complete => (1.0, "✅ Complete!".to_string()),
                    crate::processing::ProcessingState::Error(e) => (0.0, format!("❌ Error: {}", e)),
                };

                ui.vertical(|ui| {
                    ui.label(&status_text);
                    ui.add(egui::ProgressBar::new(progress).text(format!("{:.0}%", progress * 100.0)));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let process_enabled = app.input_image.is_some() && 
                        !matches!(app.processing, crate::processing::ProcessingState::Running(_));

                    if ui.add_enabled(process_enabled, egui::Button::new("▶ Process")).clicked() {
                        super::processing::process_image(app, ctx);
                    }

                    if matches!(app.processing, crate::processing::ProcessingState::Running(_)) {
                        if ui.button("⏹ Cancel").clicked() {
                            app.processing = crate::processing::ProcessingState::Idle;
                        }
                    }
                });
            });
        });
}

/// Draw status bar
pub fn draw_status(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("status_bar")
        .default_height(24.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let status = match &app.processing {
                    crate::processing::ProcessingState::Idle => "Ready",
                    crate::processing::ProcessingState::Running(_) => "Processing...",
                    crate::processing::ProcessingState::Complete => "Complete",
                    crate::processing::ProcessingState::Error(_) => "Error",
                };
                ui.label(status);

                ui.separator();

                if let Some(input) = &app.input_image {
                    let (w, h) = input.image.dimensions();
                    ui.label(format!("Input: {}×{}", w, h));
                    if let Some(path) = &input.path {
                        if let Some(name) = path.file_name() {
                            ui.label(name.to_string_lossy());
                        }
                    }
                } else {
                    ui.label("No image loaded");
                }

                ui.separator();

                let (ow, oh) = super::processing::get_output_dimensions(app);
                ui.label(format!("Output: {}×{}", ow, oh));

                ui.separator();

                let vram_percent = if app.total_vram > 0 {
                    (app.vram_usage as f64 / app.total_vram as f64 * 100.0) as u32
                } else {
                    0
                };
                ui.label(format!("VRAM: {}MB ({}%)", app.vram_usage, vram_percent));
                
                ui.separator();
                ui.label(format!("Quality: {}", app.config.model_quality.display_name()));
                
                if app.config.sequential_processing {
                    ui.label("[Sequential]");
                }
            });
        });
}

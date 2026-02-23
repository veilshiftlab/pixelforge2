//! Side and bottom panels

use super::state::{AspectRatioMode, PixelForgeApp};
use crate::processing::{
    DetailLevel, DownsamplingMethod, EdgeMode, EyeSize, PaletteMode, PresetPalette,
};
use eframe::egui;
use image::GenericImageView;

// ─────────────────────────────────────────────────────────────────────────────
// Left panel
// ─────────────────────────────────────────────────────────────────────────────

pub fn draw_left(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::SidePanel::left("controls_panel")
        .default_width(280.0)
        .min_width(240.0)
        .max_width(380.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                models_section(app, ui);
                ui.add_space(6.0);
                ml_section(app, ui, ctx);
                ui.add_space(6.0);
                depth_to_flat_section(app, ui);
                ui.add_space(6.0);
                feature_section(app, ui);
                ui.add_space(6.0);
                edge_section(app, ui);
            });
        });
}

// ─── Models ───────────────────────────────────────────────────────────────────

fn models_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("📦 Models", |ui| {
        let manager = app.model_manager.read();
        let models  = manager.list_models();
        let missing: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();
        let progress = manager.get_download_progress();

        // Directory
        ui.label(egui::RichText::new(manager.models_dir().to_string_lossy()).small().weak());
        ui.add_space(4.0);

        // Per-model status rows
        for model in &models {
            ui.horizontal(|ui| {
                let icon = if model.downloaded {
                    egui::RichText::new("✅").small()
                } else {
                    egui::RichText::new("⬜").small().color(egui::Color32::GRAY)
                };
                ui.label(icon);
                ui.label(egui::RichText::new(&model.name).small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(format!("{:.0} MB", model.size_mb)).small().weak());
                });
            });
        }

        // Active download
        if progress.is_downloading {
            ui.add_space(4.0);
            ui.horizontal(|ui| { ui.spinner(); ui.label(egui::RichText::new(&progress.current_model).small()); });
            ui.add(egui::ProgressBar::new(progress.progress).desired_width(f32::INFINITY)
                .text(format!("{:.0}%", progress.progress * 100.0)));
        } else if let Some(ref err) = progress.last_error {
            ui.colored_label(egui::Color32::RED, egui::RichText::new(err).small());
        }

        drop(manager);
        ui.add_space(4.0);

        // HF token indicator + quick-edit
        ui.horizontal(|ui| {
            if app.huggingface_token.is_empty() {
                ui.label(egui::RichText::new("⚠ HF token not set").color(egui::Color32::YELLOW).small());
            } else {
                ui.label(egui::RichText::new("🔑 Token configured").color(egui::Color32::GREEN).small());
            }
            if ui.small_button("✏").on_hover_text("Edit token").clicked() {
                app.token_input = app.huggingface_token.clone();
                app.show_token_dialog = true;
            }
        });

        if missing.is_empty() {
            ui.colored_label(egui::Color32::GREEN, "✅ All models ready");
        } else {
            let total: f64 = missing.iter().map(|m| m.size_mb).sum();
            let can = !app.huggingface_token.is_empty();
            let btn = egui::Button::new(format!("⬇ Download missing ({:.0} MB)", total));
            if ui.add_enabled(can, btn).clicked() {
                let mgr = app.model_manager.read();
                mgr.set_huggingface_token(app.huggingface_token.clone());
                if let Err(e) = mgr.download_all_missing() {
                    log::error!("Download failed: {}", e);
                }
            }
            if !can {
                ui.label(egui::RichText::new("Set HF token to download").small().color(egui::Color32::YELLOW));
            }
        }

        if ui.small_button("Manage…").clicked() {
            app.show_model_dialog = true;
        }
    });
}

// ─── ML analysis ─────────────────────────────────────────────────────────────

fn ml_section(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    ui.collapsing("🤖 ML Analysis", |ui| {
        // Enable toggles — one per model
        ui.checkbox(&mut app.ml_config.face_detection_enabled,  "Face Detection (YOLOv8)");
        ui.checkbox(&mut app.ml_config.depth_estimation_enabled, "Depth Estimation (Depth-Anything V2)");
        ui.checkbox(&mut app.ml_config.segmentation_enabled,    "Face Parsing (BiSeNet)");
        ui.checkbox(&mut app.ml_config.edge_detection_enabled,  "Edge Detection (TEED)");

        ui.separator();
        ui.label("Thresholds:");

        // Sliders wire directly to the nested sub-configs — no shim sync needed
        ui.add(
            egui::Slider::new(&mut app.ml_config.face_detection.confidence_threshold, 0.1..=0.95)
                .text("Face"),
        );
        ui.add(
            egui::Slider::new(&mut app.ml_config.segmentation.confidence_threshold, 0.1..=0.9)
                .text("Segmentation"),
        );
        ui.add(
            egui::Slider::new(&mut app.ml_config.edge.threshold, 0.05..=0.95)
                .text("Edge"),
        );

        ui.separator();

        // ML-result status badges
        if let Some(ref r) = app.ml_results {
            ui.horizontal_wrapped(|ui| {
                badge(ui, "👤", r.face_bounds.is_some());
                badge(ui, "📐", r.depth_map.is_some());
                badge(ui, "🎨", r.segmentation.is_some());
                badge(ui, "✏",  r.edge_map.is_some());
            });
        }

        let can = app.input_image.is_some()
            && !matches!(app.processing, crate::processing::ProcessingState::Running(_));

        if ui.add_enabled(can, egui::Button::new("▶ Run ML Analysis")).clicked() {
            super::processing::run_ml_analysis(app, ctx);
        }
    });
}

/// Small colored circle badge — green if `active`, gray otherwise
fn badge(ui: &mut egui::Ui, label: &str, active: bool) {
    let color = if active { egui::Color32::GREEN } else { egui::Color32::from_gray(80) };
    ui.label(egui::RichText::new(label).color(color));
}

// ─── Depth-to-flat ───────────────────────────────────────────────────────────

fn depth_to_flat_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("🎨 Depth → Flat", |ui| {
        // Shading band counts per region — DragValue is best for small integers
        for (label, val) in [
            ("Skin",       &mut app.depth_to_flat_config.skin_tone_bands),
            ("Hair",       &mut app.depth_to_flat_config.hair_bands),
            ("Clothing",   &mut app.depth_to_flat_config.clothing_bands),
            ("Background", &mut app.depth_to_flat_config.background_bands),
        ] {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(egui::DragValue::new(val).range(1..=16).suffix(" bands"));
                });
            });
        }

        ui.add_space(4.0);
        ui.add(egui::Slider::new(&mut app.depth_to_flat_config.shadow_threshold,    0.0..=0.5).text("Shadow"));
        ui.add(egui::Slider::new(&mut app.depth_to_flat_config.highlight_threshold, 0.5..=1.0).text("Highlight"));
        ui.checkbox(&mut app.depth_to_flat_config.preserve_gradients, "Preserve gradients");
    });
}

// ─── Feature preservation ─────────────────────────────────────────────────────

fn feature_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("👁 Features", |ui| {
        // Eye size
        ui.label("Eye size:");
        ui.horizontal_wrapped(|ui| {
            for size in [EyeSize::Auto, EyeSize::Small, EyeSize::Medium, EyeSize::Large] {
                if ui.selectable_label(app.feature_config.eye_size == size, format!("{:?}", size)).clicked() {
                    app.feature_config.eye_size = size;
                }
            }
        });

        ui.label("Detail:");
        ui.horizontal_wrapped(|ui| {
            for level in [DetailLevel::Minimal, DetailLevel::Standard, DetailLevel::Full] {
                if ui.selectable_label(app.feature_config.eye_detail == level, format!("{:?}", level)).clicked() {
                    // Apply to all facial features at once
                    app.feature_config.eye_detail  = level;
                    app.feature_config.lip_detail  = level;
                    app.feature_config.nose_detail = level;
                }
            }
        });

        ui.add_space(4.0);
        ui.checkbox(&mut app.feature_config.force_eye_highlights, "Force eye highlights");
        ui.checkbox(&mut app.feature_config.distinct_nostrils,    "Distinct nostrils");
    });
}

// ─── Pixel-art edges ──────────────────────────────────────────────────────────

fn edge_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("✏ Edges", |ui| {
        ui.label("Mode:");
        ui.horizontal_wrapped(|ui| {
            for mode in [EdgeMode::None, EdgeMode::Outlines, EdgeMode::Internal, EdgeMode::Both] {
                if ui.selectable_label(app.edge_config.edge_mode == mode, format!("{:?}", mode)).clicked() {
                    app.edge_config.edge_mode = mode;
                }
            }
        });

        ui.add_space(4.0);
        ui.add(egui::Slider::new(&mut app.edge_config.thickness,              1..=4).text("Thickness"));
        ui.add(egui::Slider::new(&mut app.edge_config.edge_darkener_strength, 0.0..=1.0).text("Darkener"));
        ui.checkbox(&mut app.edge_config.anti_alias_edges, "Anti-alias");
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Right panel
// ─────────────────────────────────────────────────────────────────────────────

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
                palette_section(app, ui);
                ui.separator();
                preset_section(app, ui);
                ui.separator();
                export_section(app, ui);
                palette_swatch_display(app, ui);
            });
        });
}

fn output_size_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.heading("📐 Output");
    ui.separator();

    ui.horizontal_wrapped(|ui| {
        for (mode, label) in [
            (AspectRatioMode::Square,   "Square"),
            (AspectRatioMode::Preserve, "Original"),
            (AspectRatioMode::Custom,   "Custom"),
        ] {
            if ui.selectable_label(app.aspect_mode == mode, label).clicked() {
                app.aspect_mode = mode;
            }
        }
    });

    match app.aspect_mode {
        AspectRatioMode::Square => {
            egui::ComboBox::from_id_salt("output_size")
                .selected_text(format!(
                    "{}×{}",
                    app.transform_config.output_size,
                    app.transform_config.output_size
                ))
                .show_ui(ui, |ui| {
                    for size in [16, 24, 32, 48, 64, 96, 128, 192, 256] {
                        ui.selectable_value(
                            &mut app.transform_config.output_size,
                            size,
                            format!("{}×{}", size, size),
                        );
                    }
                });
        }
        AspectRatioMode::Preserve => {
            ui.label("Max dimension:");
            egui::ComboBox::from_id_salt("max_dim")
                .selected_text(format!("{}", app.transform_config.output_size))
                .show_ui(ui, |ui| {
                    for size in [32, 48, 64, 96, 128, 192, 256] {
                        ui.selectable_value(
                            &mut app.transform_config.output_size,
                            size,
                            format!("{}", size),
                        );
                    }
                });
        }
        AspectRatioMode::Custom => {
            ui.horizontal(|ui| {
                ui.label("W:");
                ui.add(egui::DragValue::new(&mut app.custom_output_width).range(8..=512));
                ui.label("H:");
                ui.add(egui::DragValue::new(&mut app.custom_output_height).range(8..=512));
            });
        }
    }

    let (ow, oh) = super::processing::get_output_dimensions(app);
    ui.label(egui::RichText::new(format!("→ {}×{} px", ow, oh)).small().weak());
}

fn transform_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.label("Downsampling:");
    egui::ComboBox::from_id_salt("ds_method")
        .selected_text(format!("{:?}", app.transform_config.downsampling_method))
        .show_ui(ui, |ui| {
            for m in [
                DownsamplingMethod::Weighted,
                DownsamplingMethod::NearestNeighbor,
                DownsamplingMethod::Bilinear,
            ] {
                ui.selectable_value(&mut app.transform_config.downsampling_method, m, format!("{:?}", m));
            }
        });

    ui.add(egui::Slider::new(&mut app.transform_config.scale,     0.1..=4.0  ).text("Scale"));
    ui.add(egui::Slider::new(&mut app.transform_config.rotation, -180.0..=180.0).text("Rotate °"));
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut app.transform_config.offset_x, -1.0..=1.0).text("X"));
        ui.add(egui::Slider::new(&mut app.transform_config.offset_y, -1.0..=1.0).text("Y"));
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut app.transform_config.flip_horizontal, "Flip H");
        ui.checkbox(&mut app.transform_config.flip_vertical,   "Flip V");
    });

    ui.separator();
    ui.checkbox(&mut app.transform_config.clip_to_face, "Clip to face")
        .on_hover_text("Crop to the detected face region before downsampling");
    if app.transform_config.clip_to_face {
        if app.ml_results.as_ref().and_then(|r| r.face_bounds.as_ref()).is_none() {
            ui.label(egui::RichText::new("⚠ No face detected — run ML first").color(egui::Color32::YELLOW).small());
        }
        ui.add(egui::Slider::new(&mut app.transform_config.clip_padding, 0.0..=1.0).text("Padding"));
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

    match app.palette_config.mode {
        PaletteMode::Auto => {
            ui.add(egui::Slider::new(&mut app.palette_config.max_colors, 2..=256).text("Max colors"));
            ui.checkbox(&mut app.palette_config.per_region_limit, "Per-region palette")
                .on_hover_text("Each segmented region gets its own sub-palette — useful for portraits");
        }
        PaletteMode::Preset => {
            egui::ComboBox::from_id_salt("palette_preset")
                .selected_text(format!("{:?}", app.palette_config.preset))
                .show_ui(ui, |ui| {
                    for p in [
                        PresetPalette::GameBoy,
                        PresetPalette::GameBoyColor,
                        PresetPalette::PICO8,
                        PresetPalette::NES,
                        PresetPalette::DawnBringer32,
                        PresetPalette::AAP64,
                    ] {
                        ui.selectable_value(&mut app.palette_config.preset, p, format!("{:?}", p));
                    }
                });
        }
        PaletteMode::Custom => {
            ui.label(egui::RichText::new("Custom palette editing coming soon").small().weak());
        }
        PaletteMode::Hybrid => {
            // Hybrid combines auto-extracted colors with preset/override colors.
            // The palette section for Hybrid shows the same auto controls.
            ui.add(egui::Slider::new(&mut app.palette_config.max_colors, 2..=256).text("Auto colors"));
            ui.label(egui::RichText::new("Region overrides configured in color settings.").small().weak());
        }
    }
}

fn preset_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.heading("📁 Presets");

    egui::ComboBox::from_id_salt("builtin_presets")
        .selected_text(app.current_preset.as_deref().unwrap_or("Select…"))
        .show_ui(ui, |ui| {
            for preset in crate::preset::Preset::built_in_presets() {
                if ui.selectable_label(
                    app.current_preset.as_deref() == Some(&preset.name),
                    &preset.name,
                ).clicked() {
                    app.transform_config     = preset.transform;
                    app.depth_to_flat_config = preset.depth_to_flat;
                    app.feature_config       = preset.features;
                    app.edge_config          = preset.edges;
                    app.palette_config       = preset.palette;
                    app.current_preset       = Some(preset.name);
                }
            }
        });

    ui.horizontal(|ui| {
        if ui.small_button("Save…").clicked() { super::processing::save_preset(app); }
        if ui.small_button("Load…").clicked() { super::processing::load_preset(app); }
    });

    if let Some(name) = &app.current_preset {
        ui.label(egui::RichText::new(name).small().italics().weak());
    }
}

fn export_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.label("Export scale:");
    ui.horizontal(|ui| {
        for scale in [1u32, 2, 4, 8] {
            if ui.selectable_label(app.transform_config.export_scale == scale, format!("{}×", scale)).clicked() {
                app.transform_config.export_scale = scale;
            }
        }
    });
    if ui.add_enabled(app.output_image.is_some(), egui::Button::new("💾 Export")).clicked() {
        super::processing::export_image(app);
    }
}

fn palette_swatch_display(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    let Some(output) = &app.output_image else { return };
    if output.palette.is_empty() { return }

    ui.separator();
    ui.label(egui::RichText::new(format!("{} colors", output.palette.len())).small().weak());

    // Tightly packed swatches — fill available width
    let swatch = 14.0;
    let gap    = 2.0;
    let per_row = ((ui.available_width() / (swatch + gap)).floor() as usize).max(1);

    egui::Grid::new("palette_swatches")
        .min_col_width(swatch)
        .spacing([gap, gap])
        .show(ui, |ui| {
            for (i, &color) in output.palette.iter().take(256).enumerate() {
                let (r, g, b) = (color.r(), color.g(), color.b());
                ui.add(egui::Label::new(
                    egui::RichText::new("  ").background_color(color).monospace()
                )).on_hover_text(format!("#{:02X}{:02X}{:02X}", r, g, b));
                if (i + 1) % per_row == 0 { ui.end_row(); }
            }
        });

    if output.palette.len() > 256 {
        ui.label(egui::RichText::new(format!("…+{} more", output.palette.len() - 256)).small().weak());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bottom panels
// ─────────────────────────────────────────────────────────────────────────────

pub fn draw_bottom(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("processing_bar")
        .exact_height(62.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                // Progress area
                let (progress, stage) = processing_state_label(&app.processing);
                ui.vertical(|ui| {
                    ui.set_min_width(260.0);
                    ui.label(egui::RichText::new(&stage).small());
                    ui.add(egui::ProgressBar::new(progress).desired_width(f32::INFINITY)
                        .text(format!("{:.0}%", progress * 100.0)));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let running = matches!(app.processing, crate::processing::ProcessingState::Running(_));

                    if running {
                        if ui.button("⏹ Cancel").clicked() {
                            app.processing = crate::processing::ProcessingState::Idle;
                        }
                    }

                    let can = app.input_image.is_some() && !running;
                    if ui.add_enabled(can, egui::Button::new("▶ Process")).clicked() {
                        super::processing::process_image(app, ctx);
                    }
                });
            });
        });
}

pub fn draw_status(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(22.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Processing badge
                let (_, stage) = processing_state_label(&app.processing);
                ui.label(egui::RichText::new(stage).small());

                ui.separator();

                // Input info
                if let Some(input) = &app.input_image {
                    let (w, h) = input.image.dimensions();
                    ui.label(egui::RichText::new(format!("{}×{}", w, h)).small());
                    if let Some(path) = &input.path {
                        if let Some(name) = path.file_name() {
                            ui.label(egui::RichText::new(name.to_string_lossy()).small().weak());
                        }
                    }
                } else {
                    ui.label(egui::RichText::new("No image").small().weak());
                }

                ui.separator();

                let (ow, oh) = super::processing::get_output_dimensions(app);
                ui.label(egui::RichText::new(format!("→ {}×{}", ow, oh)).small().weak());

                // ML result badges
                if let Some(ref ml) = app.ml_results {
                    ui.separator();
                    for (icon, active) in [
                        ("👤", ml.face_bounds.is_some()),
                        ("📐", ml.depth_map.is_some()),
                        ("🎨", ml.segmentation.is_some()),
                        ("✏",  ml.edge_map.is_some()),
                    ] {
                        if active {
                            ui.label(egui::RichText::new(icon).small());
                        }
                    }
                }

                // VRAM — right-aligned
                if app.total_vram > 0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let pct = app.vram_usage * 100 / app.total_vram;
                        ui.label(egui::RichText::new(
                            format!("VRAM {}MB / {}MB ({}%)", app.vram_usage, app.total_vram, pct)
                        ).small().weak());
                    });
                }
            });
        });
}

fn processing_state_label(state: &crate::processing::ProcessingState) -> (f32, String) {
    match state {
        crate::processing::ProcessingState::Idle        => (0.0,  "Ready".into()),
        crate::processing::ProcessingState::Running(s)  => (s.progress, s.stage.clone()),
        crate::processing::ProcessingState::Complete    => (1.0,  "✅ Complete".into()),
        crate::processing::ProcessingState::Error(e)    => (0.0,  format!("❌ {}", e)),
    }
}
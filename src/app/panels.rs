//! Side and bottom panels

use super::state::{AspectRatioMode, PixelForgeApp};
use crate::processing::{
    DepthToFlatConfig, DownsamplingMethod, EdgeConfig, EdgeMode, OutlineStyle,
    PaletteMode, PresetPalette, SlicConfig, TransformConfig,
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
                slic_section(app, ui);
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
        // Enable toggles — one per remaining model
        ui.checkbox(&mut app.ml_config.depth_estimation_enabled, "Depth Estimation (Depth-Anything V2)");
        ui.checkbox(&mut app.ml_config.edge_detection_enabled,  "Edge Detection (DexiNed)");

        // Note: edge threshold is in the Edges panel (edge_config.teed_threshold).
        // It controls which pixels become outlines during the pixel-art pass.

        ui.separator();

        // ML-result status badges (depth + edges only)
        if let Some(ref r) = app.ml_results {
            ui.horizontal_wrapped(|ui| {
                badge(ui, "📐", r.depth_map.is_some());
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
        // ── Per-section reset ──────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Per-region shading (SLIC + MAD)").small().weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("↺ Reset").clicked() {
                    app.depth_to_flat_config = DepthToFlatConfig::default();
                }
            });
        });

        let dtf = &mut app.depth_to_flat_config;

        // ── Strength (slider + numeric entry) ────────────────────────────────
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.strength, 0.0..=1.0)
                    .text("Strength")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "How strongly depth-derived shading biases Lab L*.\n0 = no shading · 0.4 = balanced · 1.0 = max contrast"
            );
        });

        // ── L* shift scale (Phase 2) ─────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.l_shift_scale, 0.0..=100.0)
                    .text("L shift scale")
                    .fixed_decimals(0)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Phase 2 — Maximum Lab L* shift DTF can apply.\n\
                 Actual shift = strength × l_shift_scale (clamped via shading signal).\n\
                 40 = balanced (default, ±24 L* at strength 0.6)\n\
                 100 = old behavior (±60 L*, washes out vibrant colors)\n\
                 0 = no shift (DTF produces no shading)"
            );
        });

        // ── Gamma ───────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.gamma, 0.5..=2.0)
                    .text("Gamma")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Contrast curve exponent: s' = sign(s) × |s|^gamma.\n1.0 = linear · <1.0 = more midtone contrast · >1.0 = compressed"
            );
        });

        // ── MAD threshold (range widened to 0.01..=0.4 per plan) ──────────────
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.mad_threshold, 0.01..=0.4)
                    .text("MAD threshold")
                    .fixed_decimals(3)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Regions with MAD below this get no shading (avoids noise in flat areas).\nLower = shade more regions · Higher = skip noisier regions"
            );
        });

        // ── Global depth weight (with preset marker at 0.5) ────────────────────
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.global_depth_weight, 0.0..=1.0)
                    .text("Global depth")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Blend between local (per-region MAD) and global (whole-image min-max) depth shading.\n\
                 0.0 = pure local (fine detail, loses global relationships)\n\
                 0.5 = balanced (default)\n\
                 1.0 = pure global (preserves relative depth, flattens local detail)"
            );
        });

        ui.separator();

        // ── Background separation ────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Background separation").small().weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("↺ Reset BG").clicked() {
                    let d = DepthToFlatConfig::default();
                    dtf.use_otsu_threshold = d.use_otsu_threshold;
                    dtf.bg_depth_threshold = d.bg_depth_threshold;
                    dtf.bg_desaturation    = d.bg_desaturation;
                    dtf.bg_lightness_shift = d.bg_lightness_shift;
                }
            });
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut dtf.use_otsu_threshold, "Auto threshold");
            ui.label(egui::RichText::new("(Otsu per-image)").small().weak());
        });

        if !dtf.use_otsu_threshold {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut dtf.bg_depth_threshold, 0.1..=0.95)
                        .text("BG threshold")
                        .fixed_decimals(2)
                        .clamping(egui::SliderClamping::Always),
                ).on_hover_text("Depth value above which pixels are classified as background.\n0 = nearest · 1 = farthest");
            });
        }

        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.bg_desaturation, 0.0..=1.0)
                    .text("BG desaturate")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            );
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut dtf.bg_lightness_shift, -0.5..=0.2)
                    .text("BG darken")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text("Negative = darken background, positive = lighten");
        });
    });
}

// ─── SLIC superpixels ───────────────────────────────────────────────────────

fn slic_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("🧩 Regions (SLIC)", |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Superpixel clustering").small().weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("↺ Reset").clicked() {
                    app.slic_config = SlicConfig::default();
                }
            });
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut app.slic_config.k, 5..=128)
                    .text("Clusters (K)")
                    .fixed_decimals(0)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Number of superpixel regions. Higher = more, smaller regions = finer detail control.\n\
                 Portraits: 32-64 preserves limbs/face/dress separation.\n\
                 Simple images: 8-16 is sufficient.\n\
                 Re-clusters on next Process."
            );
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut app.slic_config.spatial_weight, 0.0..=1.0)
                    .text("Spatial weight")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Controls blobbiness vs. detail fidelity.\nHigher = blobbier regions (boundaries follow the spatial grid).\nLower = regions follow color/depth boundaries more faithfully."
            );
        });

        // Show cluster info if labels are cached
        if let Some(ref ml) = app.ml_results {
            if let Some(ref labels) = ml.slic_labels {
                let unique = {
                    let mut s: Vec<u32> = labels.iter().copied().collect();
                    s.sort_unstable();
                    s.dedup();
                    s.len()
                };
                ui.label(egui::RichText::new(format!(
                    "Cached: {} pixels, {} active clusters",
                    labels.len(),
                    unique
                )).small().weak());
            }
        }
    });
}

// ─── Pixel-art edges ──────────────────────────────────────────────────────────

fn edge_section(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.collapsing("✏ Edges", |ui| {
        ui.horizontal(|ui| {
            ui.label("Mode:");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("↺ Reset").clicked() {
                    app.edge_config = EdgeConfig::default();
                }
            });
        });
        ui.horizontal_wrapped(|ui| {
            for mode in [EdgeMode::None, EdgeMode::Outlines, EdgeMode::Internal, EdgeMode::Both] {
                if ui.selectable_label(app.edge_config.edge_mode == mode, format!("{:?}", mode)).clicked() {
                    app.edge_config.edge_mode = mode;
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut app.edge_config.thickness, 1..=4)
                    .text("Thickness")
                    .clamping(egui::SliderClamping::Always),
            );
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut app.edge_config.edge_darkener_strength, 0.0..=1.0)
                    .text("Darken blend")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "ET-5(b): Blends local-contrast outline color toward the darkest palette color.\n\
                 0.0 = pure local-contrast (varies per pixel)\n\
                 0.3 = default (70% local-contrast + 30% darkest)\n\
                 1.0 = pure darkest (same as Black style)"
            );
        });
        ui.checkbox(&mut app.edge_config.anti_alias_edges, "Anti-alias");

        ui.separator();
        ui.label(egui::RichText::new("DexiNed outline pass").small().weak());

        // Outline style — determines how outline colors are chosen from the palette
        // Phase 4: three modes — LocalColorShift (default), Black, MaxContrast.
        ui.label("Outline style:");
        ui.horizontal_wrapped(|ui| {
            for style in [
                OutlineStyle::LocalColorShift,
                OutlineStyle::Black,
                OutlineStyle::MaxContrast,
            ] {
                if ui.selectable_label(app.edge_config.outline_style == style, format!("{:?}", style)).clicked() {
                    app.edge_config.outline_style = style;
                }
            }
        });

        // Phase 4 — LocalColorShift parameters (only relevant when that mode is active)
        if app.edge_config.outline_style == OutlineStyle::LocalColorShift {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut app.edge_config.edge_l_shift, 0.0..=60.0)
                        .text("Edge L* shift")
                        .fixed_decimals(0)
                        .clamping(egui::SliderClamping::Always),
                ).on_hover_text(
                    "Phase 4 — L* shift applied to local 3×3 mean Lab for LocalColorShift mode.\n\
                     Edge color = local mean shifted darker by this many L* units,\n\
                     then snapped to nearest palette entry.\n\
                     25 = default (visible darkening, still in local color family)\n\
                     0 = no shift (edge = local mean, blends in)\n\
                     60 = strong darkening (near-black on light backgrounds)"
                );
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut app.edge_config.edge_hue_shift, -180.0..=180.0)
                        .text("Edge hue shift°")
                        .fixed_decimals(0)
                        .clamping(egui::SliderClamping::Always),
                ).on_hover_text(
                    "Phase 4 — Optional hue rotation (degrees) for LocalColorShift mode.\n\
                     0 = no rotation (default).\n\
                     Positive = counter-clockwise in Lab a*b* plane.\n\
                     Useful for stylized effects (e.g. shift all edges slightly cooler)."
                );
            });
        }

        // Phase 3 — Min segment length slider
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut app.edge_config.min_segment_length, 1..=20)
                    .text("Min segment")
                    .fixed_decimals(0)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Phase 3 — Minimum length (in pixels) for a skeleton segment to survive cleanup.\n\
                 Segments shorter than this are dropped as noise.\n\
                 3 = default (drops single-pixel and 2px noise)\n\
                 1 = keep all (no segment drop)\n\
                 10+ = keep only long, confident lines"
            );
        });

        // Edge threshold with preset markers in the tooltip
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut app.edge_config.teed_threshold, 0.05..=0.8)
                    .text("Edge threshold")
                    .fixed_decimals(2)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text(
                "Edge probability threshold (average-pooled to pixel-art resolution).\n\
                 Presets: 0.15 = many edges · 0.30 = balanced · 0.50 = thin · 0.70 = minimal\n\
                 Lower = more edges (thicker, denser); higher = fewer edges (thinner, cleaner).\n\
                 Falls back to Sobel when the model is unavailable."
            );
        });
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
                transform_section(app, ui, ctx);
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

fn transform_section(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    // ── Input image transforms (using ImageProcessor from image crate) ────────────
    if app.image_processor.is_some() {
        ui.label(egui::RichText::new("Input Transforms").small().weak());

        ui.horizontal(|ui| {
            if ui.button("🔄 Flip H").on_hover_text("Flip horizontally").clicked() {
                if let Some(ref mut proc) = app.image_processor {
                    proc.flip_horizontal();
                }
                super::processing::refresh_input_preview(app, ctx);
            }
            if ui.button("🔃 Flip V").on_hover_text("Flip vertically").clicked() {
                if let Some(ref mut proc) = app.image_processor {
                    proc.flip_vertical();
                }
                super::processing::refresh_input_preview(app, ctx);
            }
            if ui.button("↺ Reset").on_hover_text("Reset all transformations").clicked() {
                if let Some(ref mut proc) = app.image_processor {
                    proc.reset();
                }
                super::processing::refresh_input_preview(app, ctx);
            }
        });

        let mut rotation = 0.0;
        if let Some(ref proc) = app.image_processor {
            rotation = proc.get_state().2;
        }
        ui.horizontal(|ui| {
            if ui.add(
                egui::Slider::new(&mut rotation, -180.0..=180.0)
                    .text("Rotate °")
                    .fixed_decimals(1)
                    .clamping(egui::SliderClamping::Always),
            ).changed() {
                if let Some(ref mut proc) = app.image_processor {
                    proc.set_rotation(rotation);
                }
                super::processing::refresh_input_preview(app, ctx);
            }
        });

        ui.separator();
    } else {
        ui.label(egui::RichText::new("(Load an image to transform)").small().italics());
        ui.separator();
    }

    // ── Pipeline output transforms ────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Output").small().weak());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("↺ Reset").clicked() {
                app.transform_config = TransformConfig::default();
            }
        });
    });
    ui.label("Downsampling:");
    egui::ComboBox::from_id_salt("ds_method")
        .selected_text(format!("{:?}", app.transform_config.downsampling_method))
        .show_ui(ui, |ui| {
            for m in [
                DownsamplingMethod::PaletteMode,
                DownsamplingMethod::PerceptualDither,
                DownsamplingMethod::NearestNeighbor,
                DownsamplingMethod::Weighted,
                DownsamplingMethod::Bilinear,
            ] {
                ui.selectable_value(&mut app.transform_config.downsampling_method, m, format!("{:?}", m));
            }
        });

    match app.transform_config.downsampling_method {
        DownsamplingMethod::PaletteMode => ui.label(
            egui::RichText::new("Palette-mode: bilateral pre-filter, quantize at full res, pick most common color per block. Crisp, no smudge (default).")
            .small().color(egui::Color32::from_rgb(100, 200, 100))
        ),
        DownsamplingMethod::PerceptualDither => ui.label(
            egui::RichText::new("Perceptual-dither: area-average + Floyd-Steinberg error diffusion. Clean gradients via dithering (best for small palettes).")
            .small().color(egui::Color32::from_rgb(100, 200, 100))
        ),
        DownsamplingMethod::NearestNeighbor => ui.label(
            egui::RichText::new("Nearest: picks center pixel. Fast but may miss detail.")
            .small().weak()
        ),
        DownsamplingMethod::Weighted => ui.label(
            egui::RichText::new("Weighted: Lab-space weighted average. May smudge.")
            .small().weak()
        ),
        DownsamplingMethod::Bilinear => ui.label(
            egui::RichText::new("Bilinear: smooth interpolation. May blur.")
            .small().weak()
        ),
    };

    ui.horizontal(|ui| {
        ui.add(
            egui::Slider::new(&mut app.transform_config.scale, 0.1..=4.0)
                .text("Scale")
                .fixed_decimals(2)
                .clamping(egui::SliderClamping::Always),
        );
    });
    ui.horizontal(|ui| {
        ui.add(
            egui::Slider::new(&mut app.transform_config.offset_x, -1.0..=1.0)
                .text("X")
                .fixed_decimals(2)
                .clamping(egui::SliderClamping::Always),
        );
    });
    ui.horizontal(|ui| {
        ui.add(
            egui::Slider::new(&mut app.transform_config.offset_y, -1.0..=1.0)
                .text("Y")
                .fixed_decimals(2)
                .clamping(egui::SliderClamping::Always),
        );
    });
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
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut app.palette_config.max_colors, 2..=128)
                        .text("Max colors")
                        .clamping(egui::SliderClamping::Always),
                );
            });
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
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut app.palette_config.max_colors, 2..=128)
                        .text("Auto colors")
                        .clamping(egui::SliderClamping::Always),
                );
            });
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

                // ML result badges (depth + edges only)
                if let Some(ref ml) = app.ml_results {
                    ui.separator();
                    for (icon, active) in [
                        ("📐", ml.depth_map.is_some()),
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

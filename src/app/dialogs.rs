//! Modal dialogs

use super::state::PixelForgeApp;
use eframe::egui;

// ─────────────────────────────────────────────────────────────────────────────
// Model manager dialog
// ─────────────────────────────────────────────────────────────────────────────

pub fn model_dialog(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::Window::new("Model Manager")
        .collapsible(false)
        .resizable(true)
        .default_size([580.0, 520.0])
        .show(ctx, |ui| {
            let manager  = app.model_manager.read();
            let models   = manager.list_models();
            let progress = manager.get_download_progress();
            let missing: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();

            ui.horizontal(|ui| {
                ui.heading("Models");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(manager.models_dir().display().to_string()).small().weak());
                });
            });

            ui.separator();

            // ── Token section ───────────────────────────────────────────────
            ui.collapsing("🔑 HuggingFace Token", |ui| {
                ui.label("Required to download models from HuggingFace.");
                ui.horizontal(|ui| {
                    ui.label("Get one at:");
                    ui.hyperlink_to("huggingface.co/settings/tokens", "https://huggingface.co/settings/tokens");
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if app.huggingface_token.is_empty() {
                        ui.label(egui::RichText::new("⚠ Not set").color(egui::Color32::YELLOW));
                    } else {
                        ui.label(egui::RichText::new("✓ Configured").color(egui::Color32::GREEN));
                        // Show first 8 chars masked
                        let masked = "*".repeat(8);
                        ui.label(egui::RichText::new(format!("{}…", masked)).weak());
                    }
                    if ui.small_button("Edit…").clicked() {
                        app.token_input = app.huggingface_token.clone();
                        app.show_token_dialog = true;
                    }
                });
            });

            ui.separator();

            // ── Model list ──────────────────────────────────────────────────
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                egui::Grid::new("model_grid")
                    .num_columns(4)
                    .spacing([12.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("").strong());
                        ui.label(egui::RichText::new("Model").strong());
                        ui.label(egui::RichText::new("Size").strong());
                        ui.label(egui::RichText::new("Status").strong());
                        ui.end_row();

                        for model in &models {
                            if model.downloaded {
                                ui.label(egui::RichText::new("✅").color(egui::Color32::GREEN));
                            } else {
                                ui.label(egui::RichText::new("⬜").color(egui::Color32::GRAY));
                            }
                            ui.label(&model.name);
                            ui.label(egui::RichText::new(format!("{:.0} MB", model.size_mb)).weak());
                            if model.downloaded {
                                ui.label(egui::RichText::new("Ready").color(egui::Color32::GREEN));
                            } else {
                                ui.label(egui::RichText::new("Not downloaded").color(egui::Color32::GRAY));
                            }
                            ui.end_row();
                        }
                    });
            });

            ui.separator();

            // ── Download progress ───────────────────────────────────────────
            if progress.is_downloading {
                ui.horizontal(|ui| { ui.spinner(); ui.label(&progress.current_model); });
                ui.add(egui::ProgressBar::new(progress.progress)
                    .text(format!("{:.0}%", progress.progress * 100.0)));
            } else if let Some(ref err) = progress.last_error {
                ui.colored_label(egui::Color32::RED, format!("❌ {}", err));
                if err.contains("401") || err.contains("Unauthorized") {
                    ui.label(egui::RichText::new("💡 Check your HuggingFace token.").color(egui::Color32::YELLOW).small());
                }
            }

            // ── Download button ─────────────────────────────────────────────
            if !missing.is_empty() && !progress.is_downloading {
                let total: f64 = missing.iter().map(|m| m.size_mb).sum();
                let can = !app.huggingface_token.is_empty();

                if !can {
                    ui.label(egui::RichText::new("⚠ Set a HuggingFace token first.").color(egui::Color32::YELLOW));
                }

                // Drop the read lock BEFORE we start the download (which takes a write lock)
                drop(manager);

                let btn = egui::Button::new(format!("⬇ Download all missing ({:.0} MB)", total));
                if ui.add_enabled(can, btn).clicked() {
                    let mgr = app.model_manager.read();
                    mgr.set_huggingface_token(app.huggingface_token.clone());
                    if let Err(e) = mgr.download_all_missing() {
                        log::error!("Download failed: {}", e);
                    }
                }
            } else {
                drop(manager);
            }

            // ── File size verification ──────────────────────────────────────
            {
                let manager = app.model_manager.read();
                let verified: Vec<_> = manager.list_models().into_iter()
                    .filter(|m| m.downloaded)
                    .filter_map(|m| {
                        let path = manager.models_dir().join(&m.filename);
                        std::fs::metadata(&path).ok().map(|meta| {
                            let actual = meta.len() as f64 / (1024.0 * 1024.0);
                            let ok = (actual - m.size_mb).abs() < 5.0;
                            (m.filename.clone(), actual, m.size_mb, ok)
                        })
                    })
                    .collect();

                if !verified.is_empty() {
                    ui.separator();
                    ui.label(egui::RichText::new("Downloaded files").small().weak());
                    egui::ScrollArea::vertical().max_height(80.0).id_salt("verified_scroll").show(ui, |ui| {
                        for (fname, actual, expected, ok) in &verified {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(fname).small().monospace());
                                ui.label(egui::RichText::new(format!("{:.1} MB", actual)).small().weak());
                                if *ok {
                                    ui.label(egui::RichText::new("✓").color(egui::Color32::GREEN).small());
                                } else {
                                    ui.label(egui::RichText::new(
                                        format!("⚠ expected {:.0} MB", expected)
                                    ).color(egui::Color32::YELLOW).small());
                                }
                            });
                        }
                    });
                }
            }

            ui.separator();
            if ui.button("Close").clicked() {
                app.show_model_dialog = false;
            }
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Token dialog
// ─────────────────────────────────────────────────────────────────────────────

pub fn token_dialog(app: &mut PixelForgeApp, ctx: &egui::Context) {
    let mut close = false;
    let mut save  = false;

    egui::Window::new("HuggingFace Token")
        .collapsible(false)
        .resizable(false)
        .default_size([460.0, 200.0])
        .show(ctx, |ui| {
            ui.label("Enter your HuggingFace access token:");
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Get one at:");
                ui.hyperlink_to("huggingface.co/settings/tokens", "https://huggingface.co/settings/tokens");
            });
            ui.add_space(8.0);

            ui.add(
                egui::TextEdit::singleline(&mut app.token_input)
                    .password(true)
                    .hint_text("hf_xxxxxxxxxxxxxxxxxxxx")
                    .desired_width(380.0),
            );

            ui.add_space(4.0);
            ui.label(egui::RichText::new("Stored locally, never transmitted.").small().italics().weak());
            ui.add_space(12.0);

            ui.horizontal(|ui| {
                if ui.button("Save").clicked() { save = true; close = true; }
                if ui.button("Cancel").clicked() { close = true; }
                if ui.button("Clear").clicked() {
                    app.huggingface_token.clear();
                    app.token_input.clear();
                    close = true;
                }
            });
        });

    if save {
        app.huggingface_token = app.token_input.clone();
        app.model_manager.read().set_huggingface_token(app.huggingface_token.clone());
    }
    if close {
        app.show_token_dialog = false;
        app.token_input.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// About dialog
// ─────────────────────────────────────────────────────────────────────────────

pub fn about_dialog(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::Window::new("About PixelForge")
        .collapsible(false)
        .resizable(false)
        .default_size([360.0, 280.0])
        .show(ctx, |ui| {
            ui.heading("🎨 PixelForge");
            ui.label(egui::RichText::new("ML-Enhanced Pixel Art Style Transfer").italics().weak());
            ui.separator();

            ui.label("Version 0.1.0");
            ui.add_space(8.0);

            ui.label("Converts portraits and scenes to pixel art using a multi-stage ML pipeline:");
            ui.add_space(6.0);

            for (icon, text) in [
                ("👤", "YOLOv8n-Face — bounding box + 5 real landmarks"),
                ("📐", "Depth-Anything V2 — per-pixel relative depth for shading"),
                ("🎨", "BiSeNet — 19-class face parsing for region-aware palette"),
                ("✏",  "TEED — perceptual edge detection for crisp outlines"),
            ] {
                ui.horizontal(|ui| {
                    ui.label(icon);
                    ui.label(egui::RichText::new(text).small());
                });
            }

            ui.add_space(8.0);
            ui.label(egui::RichText::new("Depth maps drive 3D shading bands; edge maps produce single-pixel-width contours.").small().weak());
            ui.separator();

            if ui.button("Close").clicked() {
                app.show_about_dialog = false;
            }
        });
}

//! Modal dialogs

use super::state::PixelForgeApp;
use eframe::egui;

/// Model management dialog
pub fn model_dialog(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::Window::new("Model Manager")
        .collapsible(false)
        .resizable(true)
        .default_size([550.0, 500.0])
        .show(ctx, |ui| {
            let manager = app.model_manager.read();
            let models = manager.list_models();
            let download_progress = manager.get_download_progress();

            ui.heading("Required Models");
            ui.label(format!("Models directory: {}", manager.models_dir().display()));
            ui.separator();

            // HuggingFace Token Section
            ui.collapsing("🔑 HuggingFace Token (Optional)", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Some models may require a HuggingFace token for download.");
                });
                ui.horizontal_wrapped(|ui| {
                    ui.hyperlink_to("Get your token", "https://huggingface.co/settings/tokens");
                });
                
                ui.add_space(4.0);
                
                ui.horizontal(|ui| {
                    if app.huggingface_token.is_empty() {
                        ui.label(egui::RichText::new("No token set (may still work)").color(egui::Color32::GRAY));
                    } else {
                        ui.label(egui::RichText::new("✓ Token configured").color(egui::Color32::GREEN));
                    }
                });
                
                if ui.button("Configure Token...").clicked() {
                    app.token_input = app.huggingface_token.clone();
                    app.show_token_dialog = true;
                }
            });

            ui.separator();

            // Model list
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                egui::Grid::new("model_grid")
                    .num_columns(4)
                    .spacing([10.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("");
                        ui.label("Model");
                        ui.label("Size");
                        ui.label("Status");
                        ui.end_row();

                        for model in &models {
                            if model.downloaded {
                                ui.label(egui::RichText::new("✅").color(egui::Color32::GREEN));
                            } else {
                                ui.label(egui::RichText::new("⬜").color(egui::Color32::GRAY));
                            }
                            ui.vertical(|ui| {
                                ui.label(model.name);
                                ui.label(egui::RichText::new(model.description).small().weak());
                            });
                            ui.label(format!("{:.0} MB", model.size_mb));
                            ui.label(if model.downloaded { "Ready" } else { "Not downloaded" });
                            ui.end_row();
                        }
                    });
            });

            ui.separator();

            // Download progress
            if download_progress.is_downloading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(&download_progress.current_model);
                });
                ui.add(egui::ProgressBar::new(download_progress.progress)
                    .text(format!("{:.0}%", download_progress.progress * 100.0)));
            }

            // Show error if any
            if let Some(ref error) = download_progress.last_error {
                ui.add_space(8.0);
                ui.colored_label(egui::Color32::RED, format!("❌ Error: {}", error));
                
                if error.contains("401") || error.contains("Unauthorized") {
                    ui.colored_label(egui::Color32::YELLOW, "💡 This may be a token issue. Check your HuggingFace token.");
                }
            }

            // Download button
            let missing: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();
            
            ui.horizontal(|ui| {
                if !missing.is_empty() && !download_progress.is_downloading {
                    let total_size: f64 = missing.iter().map(|m| m.size_mb).sum();
                    
                    if ui.button(format!("⬇ Download All Missing ({:.0} MB)", total_size)).clicked() {
                        manager.set_huggingface_token(app.huggingface_token.clone());
                        drop(manager);
                        let mgr = app.model_manager.read();
                        if let Err(e) = mgr.download_all_missing() {
                            log::error!("Download failed: {}", e);
                        }
                    }
                } else if missing.is_empty() {
                    ui.label(egui::RichText::new("✅ All models downloaded").color(egui::Color32::GREEN));
                }
            });

            // Downloaded files verification
            ui.separator();
            ui.label("Downloaded Files:");
            egui::ScrollArea::vertical().max_height(80.0).show(ui, |ui| {
                let manager = app.model_manager.read();
                for model in &models {
                    if model.downloaded {
                        let path = manager.models_dir().join(model.filename);
                        if let Ok(metadata) = std::fs::metadata(&path) {
                            let actual_size = metadata.len() as f64 / (1024.0 * 1024.0);
                            ui.horizontal(|ui| {
                                ui.label(model.filename);
                                ui.label(format!("{:.1} MB", actual_size));
                                ui.label(egui::RichText::new("✓").color(egui::Color32::GREEN));
                            });
                        }
                    }
                }
            });

            ui.separator();

            if ui.button("Close").clicked() {
                app.show_model_dialog = false;
            }
        });
}

/// Token configuration dialog
pub fn token_dialog(app: &mut PixelForgeApp, ctx: &egui::Context) {
    let mut close_dialog = false;
    let mut save_token = false;
    
    egui::Window::new("HuggingFace Token")
        .collapsible(false)
        .resizable(false)
        .default_size([450.0, 200.0])
        .show(ctx, |ui| {
            ui.label("Enter your HuggingFace access token (optional for most models).");
            ui.add_space(8.0);
            
            ui.horizontal(|ui| {
                ui.label("Get a token:");
                ui.hyperlink_to("https://huggingface.co/settings/tokens", "https://huggingface.co/settings/tokens");
            });
            
            ui.add_space(8.0);
            
            let text_edit = egui::TextEdit::singleline(&mut app.token_input)
                .password(true)
                .hint_text("hf_xxxxxxxxxxxxxxxxxxxx")
                .desired_width(350.0);
            ui.add(text_edit);
            
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Your token is stored locally and never shared.").small().italics());
            
            ui.add_space(12.0);
            
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    save_token = true;
                    close_dialog = true;
                }
                if ui.button("Cancel").clicked() {
                    close_dialog = true;
                }
                if ui.button("Clear Token").clicked() {
                    app.huggingface_token.clear();
                    app.token_input.clear();
                    close_dialog = true;
                }
            });
        });
    
    if save_token {
        app.huggingface_token = app.token_input.clone();
        let manager = app.model_manager.read();
        manager.set_huggingface_token(app.huggingface_token.clone());
    }
    
    if close_dialog {
        app.show_token_dialog = false;
        app.token_input.clear();
    }
}

/// About dialog
pub fn about_dialog(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::Window::new("About PixelForge")
        .collapsible(false)
        .resizable(false)
        .default_size([350.0, 280.0])
        .show(ctx, |ui| {
            ui.heading("🎨 PixelForge");
            ui.label(egui::RichText::new("ML-Enhanced Pixel Art Style Transfer").italics());
            ui.separator();
            ui.label("Version 0.1.0");
            ui.add_space(10.0);
            ui.label("A tool for converting portraits to pixel art");
            ui.label("using depth estimation and face detection.");
            ui.add_space(10.0);
            ui.label("Models used:");
            ui.label("• YOLOv8n-Face - Face detection");
            ui.label("• Depth-Anything V2 - Depth estimation");
            ui.label("• BiSeNet - Face parsing");
            ui.separator();
            if ui.button("Close").clicked() {
                app.show_about_dialog = false;
            }
        });
}

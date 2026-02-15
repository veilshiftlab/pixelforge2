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
            let models = manager.list_required_models();
            let download_progress = manager.get_download_progress();

            ui.horizontal(|ui| {
                ui.heading(format!("Models for {}", app.config.model_quality.display_name()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("Dir: {}", manager.models_dir().display()));
                });
            });
            
            ui.separator();

            // HuggingFace Token Section
            ui.collapsing("🔑 HuggingFace Token", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Token required for downloading models from HuggingFace.");
                });
                ui.horizontal_wrapped(|ui| {
                    ui.hyperlink_to("Get your token", "https://huggingface.co/settings/tokens");
                });
                
                ui.add_space(4.0);
                
                ui.horizontal(|ui| {
                    // Show masked token if set
                    if app.huggingface_token.is_empty() {
                        ui.label(egui::RichText::new("⚠ No token set").color(egui::Color32::YELLOW));
                    } else {
                        ui.label(egui::RichText::new("✓ Token configured").color(egui::Color32::GREEN));
                        let masked = "*".repeat(app.huggingface_token.len().min(20));
                        ui.label(format!("({}...)", &masked[..8.min(masked.len())]));
                    }
                });
                
                if ui.button("Configure Token...").clicked() {
                    app.token_input = app.huggingface_token.clone();
                    app.show_token_dialog = true;
                }
            });

            ui.separator();

            // Model list
            egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
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
                            ui.label(&model.name);
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
                
                // Check if error suggests auth failure
                if error.contains("401") || error.contains("Unauthorized") {
                    ui.colored_label(egui::Color32::YELLOW, "💡 This may be a token issue. Check your HuggingFace token.");
                }
            }

            // Download button
            let missing: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();
            
            ui.horizontal(|ui| {
                if !missing.is_empty() && !download_progress.is_downloading {
                    let total_size: f64 = missing.iter().map(|m| m.size_mb).sum();
                    
                    // Warn if no token set and trying to download
                    let can_download = !app.huggingface_token.is_empty();
                    
                    if !can_download {
                        ui.label(egui::RichText::new("⚠ Set HuggingFace token first").color(egui::Color32::YELLOW));
                    }
                    
                    let button = if can_download {
                        egui::Button::new(format!("⬇ Download All ({:.0} MB)", total_size))
                    } else {
                        egui::Button::new(format!("⬇ Download All ({:.0} MB)", total_size))
                    };
                    
                    if ui.add_enabled(can_download, button).clicked() {
                        // Set token before downloading
                        manager.set_huggingface_token(app.huggingface_token.clone());
                        drop(manager);
                        let mgr = app.model_manager.read();
                        if let Err(e) = mgr.download_missing_models() {
                            log::error!("Download failed: {}", e);
                        }
                    }
                }
            });

            // Downloaded files verification
            ui.separator();
            ui.label("Downloaded Files:");
            egui::ScrollArea::vertical().max_height(80.0).show(ui, |ui| {
                let manager = app.model_manager.read();
                for model in &models {
                    if model.downloaded {
                        let path = manager.models_dir().join(&model.filename);
                        if let Ok(metadata) = std::fs::metadata(&path) {
                            let actual_size = metadata.len() as f64 / (1024.0 * 1024.0);
                            let size_match = (actual_size - model.size_mb).abs() < 5.0;
                            ui.horizontal(|ui| {
                                ui.label(&model.filename);
                                ui.label(format!("{:.1} MB", actual_size));
                                if size_match {
                                    ui.label(egui::RichText::new("✓").color(egui::Color32::GREEN));
                                } else {
                                    ui.label(egui::RichText::new("!").color(egui::Color32::YELLOW));
                                    ui.label(format!("(expected {:.0} MB)", model.size_mb));
                                }
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
            ui.label("Enter your HuggingFace access token to download models.");
            ui.add_space(8.0);
            
            ui.horizontal(|ui| {
                ui.label("Get a token:");
                ui.hyperlink_to("https://huggingface.co/settings/tokens", "https://huggingface.co/settings/tokens");
            });
            
            ui.add_space(8.0);
            
            // Password-style text input (shows dots)
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
        // Also set it in the model manager
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
        .default_size([350.0, 250.0])
        .show(ctx, |ui| {
            ui.heading("🎨 PixelForge");
            ui.label(egui::RichText::new("ML-Enhanced Pixel Art Style Transfer").italics());
            ui.separator();
            ui.label("Version 0.1.0");
            ui.add_space(10.0);
            ui.label("A tool for converting portraits to pixel art");
            ui.label("using depth estimation and face detection.");
            ui.add_space(10.0);
            ui.label("Features:");
            ui.label("• Face detection with landmark extraction");
            ui.label("• Depth-based color flattening");
            ui.label("• Semantic segmentation");
            ui.label("• Customizable palette quantization");
            ui.separator();
            if ui.button("Close").clicked() {
                app.show_about_dialog = false;
            }
        });
}

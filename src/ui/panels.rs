//! UI Panel implementations for PixelForgeApp

use super::super::app::PixelForgeApp;
use eframe::egui;

/// UI extension trait for PixelForgeApp
impl PixelForgeApp {
    /// Draw the menu bar
    pub fn menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Image...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["png", "jpg", "jpeg", "webp"])
                            .open_file()
                        {
                            self.load_image(&path, ctx);
                        }
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Export Image...").clicked() {
                        self.export_image();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                
                ui.menu_button("Edit", |ui| {
                    if ui.button("Reset to Defaults").clicked() {
                        // Reset all configs
                        ui.close_menu();
                    }
                });
                
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_landmarks, "Show Landmarks");
                    ui.checkbox(&mut self.show_depth_heatmap, "Show Depth Heatmap");
                    ui.checkbox(&mut self.show_segmentation, "Show Segmentation");
                });
                
                ui.menu_button("Presets", |ui| {
                    if ui.button("Save Preset...").clicked() {
                        self.save_preset();
                        ui.close_menu();
                    }
                    
                    if ui.button("Load Preset...").clicked() {
                        self.load_preset();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    // Built-in presets
                    if ui.button("Portrait - Minimal").clicked() {
                        // Apply minimal preset
                        ui.close_menu();
                    }
                    if ui.button("Portrait - Detailed").clicked() {
                        // Apply detailed preset
                        ui.close_menu();
                    }
                    if ui.button("Game Boy Style").clicked() {
                        // Apply game boy preset
                        ui.close_menu();
                    }
                });
                
                ui.menu_button("Settings", |ui| {
                    ui.label("Model Quality:");
                    if ui.radio_value(&mut self.config.model_quality, crate::config::ModelQuality::Minimal, "Minimal (Fast)").clicked() {
                        ui.close_menu();
                    }
                    if ui.radio_value(&mut self.config.model_quality, crate::config::ModelQuality::Standard, "Standard").clicked() {
                        ui.close_menu();
                    }
                    if ui.radio_value(&mut self.config.model_quality, crate::config::ModelQuality::High, "High Quality").clicked() {
                        ui.close_menu();
                    }
                    if ui.radio_value(&mut self.config.model_quality, crate::config::ModelQuality::Sequential, "Sequential (Low VRAM)").clicked() {
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Download Better Models...").clicked() {
                        // Open model download dialog
                        ui.close_menu();
                    }
                });
                
                ui.menu_button("Help", |ui| {
                    if ui.button("About PixelForge").clicked() {
                        // Show about dialog
                        ui.close_menu();
                    }
                    if ui.button("Documentation").clicked() {
                        ui.close_menu();
                    }
                });
            });
        });
    }
    
    /// Draw the left controls panel
    pub fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls_panel")
            .default_width(280.0)
            .min_width(240.0)
            .max_width(360.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                
                // ML Analysis Section
                ui.collapsing("ML Analysis", |ui| {
                    self.ml_analysis_controls(ui);
                });
                
                ui.add_space(8.0);
                
                // Depth-to-Flat Section
                ui.collapsing("Depth-to-Flat", |ui| {
                    self.depth_to_flat_controls(ui);
                });
                
                ui.add_space(8.0);
                
                // Feature Preservation Section
                ui.collapsing("Feature Preservation", |ui| {
                    self.feature_preserve_controls(ui);
                });
                
                ui.add_space(8.0);
                
                // Edge Controls Section
                ui.collapsing("Edge Controls", |ui| {
                    self.edge_controls(ui);
                });
            });
    }
    
    /// Draw the right output settings panel
    pub fn right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("output_panel")
            .default_width(200.0)
            .min_width(180.0)
            .max_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Output Settings");
                ui.separator();
                
                // Output Size
                ui.label("Output Size:");
                egui::ComboBox::from_id_source("output_size")
                    .selected_text(format!("{} px", self.transform_config.output_size))
                    .show_ui(ui, |ui| {
                        let sizes = [16, 24, 32, 48, 64, 96, 128];
                        for size in sizes {
                            ui.selectable_value(
                                &mut self.transform_config.output_size,
                                size,
                                format!("{} px", size),
                            );
                        }
                    });
                
                ui.add_space(8.0);
                
                // Scale
                ui.label("Scale:");
                ui.add(egui::Slider::new(&mut self.transform_config.scale, 0.1..=4.0)
                    .text("x")
                    .step_by(0.1));
                
                // Rotation
                ui.label("Rotation:");
                ui.add(egui::Slider::new(&mut self.transform_config.rotation, -180.0..=180.0)
                    .text("°")
                    .step_by(1.0));
                
                ui.separator();
                
                // Export Format
                ui.label("Format:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Auto, "PNG");
                    ui.radio_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Preset, "JPG");
                });
                
                ui.add_space(8.0);
                
                // Export Scale
                ui.label("Export Scale:");
                ui.horizontal(|ui| {
                    let scales = [1, 2, 4];
                    for &scale in &scales {
                        if ui.selectable_label(
                            self.transform_config.export_scale == scale,
                            format!("{}x", scale)
                        ).clicked() {
                            self.transform_config.export_scale = scale;
                        }
                    }
                });
                
                ui.separator();
                
                // Export Button
                if ui.button("Export Image...").clicked() {
                    self.export_image();
                }
                
                if ui.button("Batch Process...").clicked() {
                    // Open batch dialog
                }
                
                ui.separator();
                
                // Presets
                ui.heading("Presets");
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.save_preset();
                    }
                    if ui.button("Load").clicked() {
                        self.load_preset();
                    }
                });
                
                ui.separator();
                
                // Palette Settings
                ui.heading("Palette");
                ui.label("Mode:");
                ui.horizontal(|ui| {
                    if ui.radio_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Auto, "Auto").changed() {}
                    if ui.radio_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Preset, "Preset").changed() {}
                });
                
                if matches!(self.palette_config.mode, crate::processing::PaletteMode::Auto) {
                    ui.label("Max Colors:");
                    egui::ComboBox::from_id_source("max_colors")
                        .selected_text(format!("{}", self.palette_config.max_colors))
                        .show_ui(ui, |ui| {
                            for count in [4, 8, 12, 16, 24, 32, 48, 64, 128, 256] {
                                ui.selectable_value(&mut self.palette_config.max_colors, count, format!("{}", count));
                            }
                        });
                    
                    ui.checkbox(&mut self.palette_config.per_region_limit, "Per-region limit");
                }
                
                // Palette preview
                if let Some(output) = &self.output_image {
                    ui.add_space(8.0);
                    ui.label("Generated Palette:");
                    ui.horizontal_wrapped(|ui| {
                        for &color in &output.palette {
                            let (r, g, b) = (color.r(), color.g(), color.b());
                            ui.add(egui::Label::new(
                                egui::RichText::new("  ").background_color(color)
                            ));
                        }
                    });
                }
            });
    }
    
    /// Draw the bottom processing panel
    pub fn bottom_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("processing_panel")
            .default_height(60.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Progress bar
                    let progress = match &self.processing {
                        crate::processing::ProcessingState::Idle => 0.0,
                        crate::processing::ProcessingState::Running(status) => status.progress,
                        crate::processing::ProcessingState::Complete => 1.0,
                        crate::processing::ProcessingState::Error(_) => 0.0,
                    };
                    
                    let status_text = match &self.processing {
                        crate::processing::ProcessingState::Idle => "Ready".to_string(),
                        crate::processing::ProcessingState::Running(status) => status.stage.clone(),
                        crate::processing::ProcessingState::Complete => "Complete!".to_string(),
                        crate::processing::ProcessingState::Error(e) => format!("Error: {}", e),
                    };
                    
                    ui.vertical(|ui| {
                        ui.label(&status_text);
                        ui.add(egui::ProgressBar::new(progress)
                            .text(format!("{:.0}%", progress * 100.0)));
                    });
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Process button
                        let process_enabled = self.input_image.is_some() && !matches!(self.processing, crate::processing::ProcessingState::Running(_));
                        
                        if ui.add_enabled(process_enabled, egui::Button::new("Process")).clicked() {
                            self.process_image(ctx);
                        }
                        
                        // Cancel button
                        if matches!(self.processing, crate::processing::ProcessingState::Running(_)) {
                            if ui.button("Cancel").clicked() {
                                // Cancel processing
                            }
                        }
                    });
                });
            });
    }
    
    /// Draw the status bar
    pub fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .default_height(24.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Status
                    let status = match &self.processing {
                        crate::processing::ProcessingState::Idle => "Ready",
                        crate::processing::ProcessingState::Running(_) => "Processing...",
                        crate::processing::ProcessingState::Complete => "Complete",
                        crate::processing::ProcessingState::Error(_) => "Error",
                    };
                    ui.label(status);
                    
                    ui.separator();
                    
                    // Image info
                    if let Some(input) = &self.input_image {
                        let (w, h) = input.image.dimensions();
                        ui.label(format!("Input: {}x{}", w, h));
                        
                        if let Some(path) = &input.path {
                            if let Some(name) = path.file_name() {
                                ui.label(name.to_string_lossy());
                            }
                        }
                    } else {
                        ui.label("No image loaded");
                    }
                    
                    ui.separator();
                    
                    // Output info
                    ui.label(format!("Output: {}x{}", self.transform_config.output_size, self.transform_config.output_size));
                    
                    ui.separator();
                    
                    // VRAM usage
                    let vram_percent = if self.total_vram > 0 {
                        (self.vram_usage as f64 / self.total_vram as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    ui.label(format!("VRAM: {}MB / {}MB ({}%)", 
                        self.vram_usage, self.total_vram, vram_percent));
                    
                    // Model quality indicator
                    ui.separator();
                    let quality_text = match self.config.model_quality {
                        crate::config::ModelQuality::Minimal => "Minimal",
                        crate::config::ModelQuality::Standard => "Standard",
                        crate::config::ModelQuality::High => "High",
                        crate::config::ModelQuality::Sequential => "Sequential",
                    };
                    ui.label(format!("Model: {}", quality_text));
                });
            });
    }
    
    /// Draw the central preview panel
    pub fn central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Create three preview panes
            ui.horizontal(|ui| {
                // Original Image
                ui.vertical(|ui| {
                    ui.heading("Original");
                    ui.separator();
                    
                    if let Some(input) = &self.input_image {
                        // Calculate display size maintaining aspect ratio
                        let available = ui.available_size();
                        let texture_size = input.texture.size_vec2();
                        let scale = (available.x / texture_size.x).min(available.y / texture_size.y).min(1.0);
                        let display_size = texture_size * scale;
                        
                        ui.image(&input.texture);
                    } else {
                        // Drop zone
                        let available = ui.available_size();
                        let rect = ui.allocate_space(available);
                        
                        ui.allocate_ui_at_rect(rect.0, |ui| {
                            ui.centered_and_justified(|ui| {
                                ui.vertical_centered(|ui| {
                                    ui.add_space(40.0);
                                    ui.label(egui::RichText::new("📷").size(48.0));
                                    ui.add_space(10.0);
                                    ui.label(egui::RichText::new("Drop image here").size(16.0));
                                    ui.label(egui::RichText::new("or").size(12.0).weak());
                                    if ui.button("Browse...").clicked() {
                                        if let Some(path) = rfd::FileDialog::new()
                                            .add_filter("Image", &["png", "jpg", "jpeg", "webp"])
                                            .open_file()
                                        {
                                            self.load_image(&path, ctx);
                                        }
                                    }
                                });
                            });
                        });
                    }
                });
                
                ui.separator();
                
                // ML Overlays
                ui.vertical(|ui| {
                    ui.heading("ML Analysis");
                    ui.separator();
                    
                    // Overlay toggles
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.show_landmarks, "Landmarks");
                        ui.checkbox(&mut self.show_depth_heatmap, "Depth");
                        ui.checkbox(&mut self.show_segmentation, "Segment");
                    });
                    
                    if let Some(results) = &self.ml_results {
                        // Show ML analysis visualization
                        // TODO: Render combined overlay
                        ui.label("ML analysis complete");
                        
                        // Show analysis stats
                        if let Some(landmarks) = &results.landmarks {
                            ui.label(format!("Detected {} landmarks", landmarks.points.len()));
                        }
                        if results.depth_map.is_some() {
                            ui.label("Depth map generated");
                        }
                        if results.segmentation.is_some() {
                            ui.label("Segmentation complete");
                        }
                    } else if self.input_image.is_some() {
                        ui.centered_and_justified(|ui| {
                            ui.label("Click 'Run Analysis' to process");
                        });
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Load an image first");
                        });
                    }
                });
                
                ui.separator();
                
                // Output
                ui.vertical(|ui| {
                    ui.heading("Output");
                    ui.separator();
                    
                    if let Some(output) = &self.output_image {
                        // Show output image
                        ui.image(&output.texture);
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.vertical_centered(|ui| {
                                ui.label("Output will appear here");
                                ui.label("after processing");
                            });
                        });
                    }
                });
            });
        });
    }
}

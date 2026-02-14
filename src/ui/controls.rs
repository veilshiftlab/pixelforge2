//! Control panel implementations

use super::super::app::PixelForgeApp;
use eframe::egui;

impl PixelForgeApp {
    /// ML Analysis controls
    pub fn ml_analysis_controls(&mut self, ui: &mut egui::Ui) {
        // Face Detection
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.ml_config.face_detection_enabled, "Face Detection");
        });
        
        if self.ml_config.face_detection_enabled {
            ui.indent("face_indent", |ui| {
                ui.checkbox(&mut self.show_landmarks, "Show landmarks overlay");
            });
        }
        
        // Depth Estimation
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.ml_config.depth_estimation_enabled, "Depth Estimation");
        });
        
        if self.ml_config.depth_estimation_enabled {
            ui.indent("depth_indent", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Depth Bands:");
                    egui::ComboBox::from_id_source("depth_bands")
                        .selected_text(format!("{}", self.depth_to_flat_config.skin_tone_bands))
                        .show_ui(ui, |ui| {
                            for count in 2..=8 {
                                ui.selectable_value(
                                    &mut self.depth_to_flat_config.skin_tone_bands,
                                    count,
                                    count.to_string(),
                                );
                            }
                        });
                });
                ui.checkbox(&mut self.show_depth_heatmap, "Show depth heatmap");
            });
        }
        
        // Segmentation
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.ml_config.segmentation_enabled, "Segmentation");
        });
        
        if self.ml_config.segmentation_enabled {
            ui.indent("seg_indent", |ui| {
                ui.checkbox(&mut self.show_segmentation, "Show segmentation mask");
            });
        }
        
        ui.add_space(8.0);
        
        // Model Quality
        ui.label("Model Quality:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.config.model_quality, crate::config::ModelQuality::Minimal),
                "Minimal"
            ).clicked() {
                self.config.model_quality = crate::config::ModelQuality::Minimal;
            }
            if ui.selectable_label(
                matches!(self.config.model_quality, crate::config::ModelQuality::Standard),
                "Standard"
            ).clicked() {
                self.config.model_quality = crate::config::ModelQuality::Standard;
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.config.model_quality, crate::config::ModelQuality::High),
                "High"
            ).clicked() {
                self.config.model_quality = crate::config::ModelQuality::High;
            }
            if ui.selectable_label(
                matches!(self.config.model_quality, crate::config::ModelQuality::Sequential),
                "Sequential"
            ).clicked() {
                self.config.model_quality = crate::config::ModelQuality::Sequential;
            }
        });
        
        ui.add_space(8.0);
        
        // Run Analysis Button
        let analysis_enabled = self.input_image.is_some() && 
            !matches!(self.processing, crate::processing::ProcessingState::Running(_));
        
        if ui.add_enabled(analysis_enabled, egui::Button::new("Run ML Analysis")).clicked() {
            self.run_ml_analysis(ui.ctx().clone());
        }
    }
    
    /// Depth-to-Flat controls
    pub fn depth_to_flat_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Bands per Region:");
        
        ui.horizontal(|ui| {
            ui.label("Skin:");
            egui::ComboBox::from_id_source("skin_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.skin_tone_bands))
                .show_ui(ui, |ui| {
                    for count in 2..=8 {
                        ui.selectable_value(
                            &mut self.depth_to_flat_config.skin_tone_bands,
                            count,
                            count.to_string(),
                        );
                    }
                });
        });
        
        ui.horizontal(|ui| {
            ui.label("Hair:");
            egui::ComboBox::from_id_source("hair_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.hair_bands))
                .show_ui(ui, |ui| {
                    for count in 2..=8 {
                        ui.selectable_value(
                            &mut self.depth_to_flat_config.hair_bands,
                            count,
                            count.to_string(),
                        );
                    }
                });
        });
        
        ui.horizontal(|ui| {
            ui.label("Clothes:");
            egui::ComboBox::from_id_source("clothes_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.clothing_bands))
                .show_ui(ui, |ui| {
                    for count in 2..=8 {
                        ui.selectable_value(
                            &mut self.depth_to_flat_config.clothing_bands,
                            count,
                            count.to_string(),
                        );
                    }
                });
        });
        
        ui.horizontal(|ui| {
            ui.label("Background:");
            egui::ComboBox::from_id_source("bg_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.background_bands))
                .show_ui(ui, |ui| {
                    for count in 1..=4 {
                        ui.selectable_value(
                            &mut self.depth_to_flat_config.background_bands,
                            count,
                            count.to_string(),
                        );
                    }
                });
        });
        
        ui.add_space(8.0);
        
        // Thresholds
        ui.label("Thresholds:");
        
        ui.horizontal(|ui| {
            ui.label("Shadow:");
            ui.add(egui::Slider::new(&mut self.depth_to_flat_config.shadow_threshold, 0.0..=1.0)
                .text("")
                .step_by(0.05));
            ui.label(format!("{:.0}%", self.depth_to_flat_config.shadow_threshold * 100.0));
        });
        
        ui.horizontal(|ui| {
            ui.label("Highlight:");
            ui.add(egui::Slider::new(&mut self.depth_to_flat_config.highlight_threshold, 0.0..=1.0)
                .text("")
                .step_by(0.05));
            ui.label(format!("{:.0}%", self.depth_to_flat_config.highlight_threshold * 100.0));
        });
        
        ui.add_space(8.0);
        
        ui.checkbox(&mut self.depth_to_flat_config.preserve_gradients, "Preserve gradients");
        
        if self.depth_to_flat_config.preserve_gradients {
            ui.indent("gradient_indent", |ui| {
                ui.label("Gradient strength:");
                ui.add(egui::Slider::new(&mut self.depth_to_flat_config.gradient_preservation, 0.0..=1.0)
                    .text(""));
            });
        }
        
        ui.add_space(8.0);
        
        if ui.button("Preview Flat Colors").clicked() {
            // Generate preview
        }
    }
    
    /// Feature Preservation controls
    pub fn feature_preserve_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Eye Size:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.feature_config.eye_size, crate::processing::EyeSize::Auto),
                "Auto"
            ).clicked() {
                self.feature_config.eye_size = crate::processing::EyeSize::Auto;
            }
            if ui.selectable_label(
                matches!(self.feature_config.eye_size, crate::processing::EyeSize::Small),
                "Small"
            ).clicked() {
                self.feature_config.eye_size = crate::processing::EyeSize::Small;
            }
            if ui.selectable_label(
                matches!(self.feature_config.eye_size, crate::processing::EyeSize::Medium),
                "Medium"
            ).clicked() {
                self.feature_config.eye_size = crate::processing::EyeSize::Medium;
            }
            if ui.selectable_label(
                matches!(self.feature_config.eye_size, crate::processing::EyeSize::Large),
                "Large"
            ).clicked() {
                self.feature_config.eye_size = crate::processing::EyeSize::Large;
            }
        });
        
        ui.add_space(4.0);
        
        ui.label("Eye Detail:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.feature_config.eye_detail, crate::processing::DetailLevel::Minimal),
                "Min"
            ).clicked() {
                self.feature_config.eye_detail = crate::processing::DetailLevel::Minimal;
            }
            if ui.selectable_label(
                matches!(self.feature_config.eye_detail, crate::processing::DetailLevel::Standard),
                "Std"
            ).clicked() {
                self.feature_config.eye_detail = crate::processing::DetailLevel::Standard;
            }
            if ui.selectable_label(
                matches!(self.feature_config.eye_detail, crate::processing::DetailLevel::Full),
                "Full"
            ).clicked() {
                self.feature_config.eye_detail = crate::processing::DetailLevel::Full;
            }
        });
        
        ui.add_space(4.0);
        
        ui.label("Lip Detail:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.feature_config.lip_detail, crate::processing::DetailLevel::Minimal),
                "Min"
            ).clicked() {
                self.feature_config.lip_detail = crate::processing::DetailLevel::Minimal;
            }
            if ui.selectable_label(
                matches!(self.feature_config.lip_detail, crate::processing::DetailLevel::Standard),
                "Std"
            ).clicked() {
                self.feature_config.lip_detail = crate::processing::DetailLevel::Standard;
            }
            if ui.selectable_label(
                matches!(self.feature_config.lip_detail, crate::processing::DetailLevel::Full),
                "Full"
            ).clicked() {
                self.feature_config.lip_detail = crate::processing::DetailLevel::Full;
            }
        });
        
        ui.add_space(4.0);
        
        ui.label("Nose Detail:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.feature_config.nose_detail, crate::processing::DetailLevel::Minimal),
                "Min"
            ).clicked() {
                self.feature_config.nose_detail = crate::processing::DetailLevel::Minimal;
            }
            if ui.selectable_label(
                matches!(self.feature_config.nose_detail, crate::processing::DetailLevel::Standard),
                "Std"
            ).clicked() {
                self.feature_config.nose_detail = crate::processing::DetailLevel::Standard;
            }
            if ui.selectable_label(
                matches!(self.feature_config.nose_detail, crate::processing::DetailLevel::Full),
                "Full"
            ).clicked() {
                self.feature_config.nose_detail = crate::processing::DetailLevel::Full;
            }
        });
        
        ui.add_space(8.0);
        
        ui.checkbox(&mut self.feature_config.force_eye_highlights, "Force eye highlights");
        ui.checkbox(&mut self.feature_config.distinct_nostrils, "Distinct nostrils");
        
        ui.add_space(8.0);
        
        ui.label("Feature Sharpening:");
        ui.add(egui::Slider::new(&mut self.feature_config.feature_sharpening, 0.0..=1.0)
            .text(""));
    }
    
    /// Edge controls
    pub fn edge_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Edge Mode:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.edge_config.edge_mode, crate::processing::EdgeMode::None),
                "None"
            ).clicked() {
                self.edge_config.edge_mode = crate::processing::EdgeMode::None;
            }
            if ui.selectable_label(
                matches!(self.edge_config.edge_mode, crate::processing::EdgeMode::Outlines),
                "Outlines"
            ).clicked() {
                self.edge_config.edge_mode = crate::processing::EdgeMode::Outlines;
            }
            if ui.selectable_label(
                matches!(self.edge_config.edge_mode, crate::processing::EdgeMode::Internal),
                "Internal"
            ).clicked() {
                self.edge_config.edge_mode = crate::processing::EdgeMode::Internal;
            }
            if ui.selectable_label(
                matches!(self.edge_config.edge_mode, crate::processing::EdgeMode::Both),
                "Both"
            ).clicked() {
                self.edge_config.edge_mode = crate::processing::EdgeMode::Both;
            }
        });
        
        ui.add_space(8.0);
        
        ui.label("Thickness:");
        ui.add(egui::Slider::new(&mut self.edge_config.thickness, 1..=4)
            .text("px"));
        
        ui.add_space(8.0);
        
        ui.label("Edge Color:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(
                matches!(self.edge_config.edge_color_mode, crate::processing::EdgeColorMode::Black),
                "Black"
            ).clicked() {
                self.edge_config.edge_color_mode = crate::processing::EdgeColorMode::Black;
            }
            if ui.selectable_label(
                matches!(self.edge_config.edge_color_mode, crate::processing::EdgeColorMode::DarkestShade),
                "Dark"
            ).clicked() {
                self.edge_config.edge_color_mode = crate::processing::EdgeColorMode::DarkestShade;
            }
        });
        
        ui.horizontal(|ui| {
            if ui.selectable_label(
                matches!(self.edge_config.edge_color_mode, crate::processing::EdgeColorMode::Custom),
                "Custom"
            ).clicked() {
                self.edge_config.edge_color_mode = crate::processing::EdgeColorMode::Custom;
            }
            
            if matches!(self.edge_config.edge_color_mode, crate::processing::EdgeColorMode::Custom) {
                ui.color_edit_button_srgba(&mut self.edge_config.custom_edge_color);
            }
        });
        
        ui.add_space(8.0);
        
        ui.label("Edge Darkener:");
        ui.add(egui::Slider::new(&mut self.edge_config.edge_darkener_strength, 0.0..=1.0)
            .text(""));
        ui.label("(darkens adjacent pixels)");
        
        ui.add_space(4.0);
        
        ui.checkbox(&mut self.edge_config.anti_alias_edges, "Anti-alias edges");
    }
}

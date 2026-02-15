//! Central preview panel with tabs

use super::state::{PixelForgeApp, PreviewTab};
use eframe::egui;
use image::GenericImageView;

/// Draw central preview panel
pub fn draw(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::CentralPanel::default().show(ctx, |ui| {
        // Tab bar
        ui.horizontal(|ui| {
            for (tab, label) in [
                (PreviewTab::Original, "📷 Original"),
                (PreviewTab::MLAnalysis, "🤖 ML Analysis"),
                (PreviewTab::Output, "🎨 Output"),
            ] {
                if ui.selectable_label(app.preview_tab == tab, label).clicked() {
                    app.preview_tab = tab;
                }
            }
        });
        
        ui.separator();

        // Content based on selected tab
        match app.preview_tab {
            PreviewTab::Original => original_panel(app, ui, ctx),
            PreviewTab::MLAnalysis => ml_preview_panel(app, ui, ctx),
            PreviewTab::Output => output_panel(app, ui, ctx),
        }
    });
}

fn original_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, _ctx: &egui::Context) {
    if let Some(input) = &app.input_image {
        let (img_w, img_h) = input.image.dimensions();
        let available = ui.available_size();
        
        let scale = calculate_fit_scale(img_w, img_h, available);
        let display_w = (img_w as f32 * scale * app.preview_zoom) as u32;

        ui.vertical(|ui| {
            ui.label(format!("{}×{} pixels | Zoom: {:.1}×", img_w, img_h, app.preview_zoom));
            
            ui.horizontal(|ui| {
                let spacer = (available.x - display_w as f32).max(0.0) / 2.0;
                ui.add_space(spacer);
                ui.image(&input.texture);
            });
        });
    } else {
        drop_zone(app, ui);
    }
}

fn ml_preview_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    // Extract data before the closure to avoid borrow conflicts
    let ml_results = app.ml_results.clone();
    let has_input = app.input_image.is_some();
    let is_processing = matches!(app.processing, crate::processing::ProcessingState::Running(_));
    let skin_bands = app.depth_to_flat_config.skin_tone_bands;
    let hair_bands = app.depth_to_flat_config.hair_bands;
    let clothes_bands = app.depth_to_flat_config.clothing_bands;
    
    if let Some(results) = ml_results {
        let mut rerun_clicked = false;
        
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("ML Analysis Results");
            ui.separator();

            // Face Detection Section
            ui.collapsing("👤 Face Detection", |ui| {
                if results.landmarks.is_some() {
                    ui.label(egui::RichText::new("✅ Face detected").color(egui::Color32::GREEN));
                    
                    if let Some(ref bounds) = results.face_bounds {
                        ui.label(format!("Position: ({:.1}%, {:.1}%)", bounds.x * 100.0, bounds.y * 100.0));
                        ui.label(format!("Size: {:.1}% × {:.1}%", bounds.width * 100.0, bounds.height * 100.0));
                    }
                    
                    if let Some(ref lm) = results.landmarks {
                        ui.label(format!("Landmarks: {} points", lm.points.len()));
                        
                        if let Some(ref eye) = lm.left_eye {
                            ui.label(format!("Left eye: center ({:.2}, {:.2}), size {:.2}×{:.2}", 
                                eye.center_x, eye.center_y, eye.width, eye.height));
                        }
                        if let Some(ref eye) = lm.right_eye {
                            ui.label(format!("Right eye: center ({:.2}, {:.2}), size {:.2}×{:.2}", 
                                eye.center_x, eye.center_y, eye.width, eye.height));
                        }
                        if let Some(ref nose) = lm.nose {
                            ui.label(format!("Nose: center ({:.2}, {:.2})", nose.center_x, nose.center_y));
                        }
                        if let Some(ref lips) = lm.lips {
                            ui.label(format!("Lips: center ({:.2}, {:.2}), size {:.2}×{:.2}", 
                                lips.center_x, lips.center_y, lips.width, lips.height));
                        }
                    }
                } else {
                    ui.label(egui::RichText::new("⚠ No face detected").color(egui::Color32::YELLOW));
                }
            });

            // Depth Map Section
            ui.collapsing("📐 Depth Estimation", |ui| {
                if let Some(ref depth) = results.depth_map {
                    ui.label(egui::RichText::new("✅ Depth map generated").color(egui::Color32::GREEN));
                    ui.label(format!("Resolution: {} values", depth.len()));
                    
                    let min = depth.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = depth.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let avg = depth.iter().sum::<f32>() / depth.len() as f32;
                    
                    ui.label(format!("Range: {:.3} - {:.3}", min, max));
                    ui.label(format!("Average: {:.3}", avg));
                    
                    ui.add_space(8.0);
                    ui.label("Depth Histogram:");
                    
                    let num_bins = 16;
                    let mut histogram = vec![0u32; num_bins];
                    for &d in depth {
                        let bin = (d * (num_bins as f32 - 1.0)).min((num_bins - 1) as f32) as usize;
                        histogram[bin] += 1;
                    }
                    
                    let max_count = histogram.iter().cloned().max().unwrap_or(1).max(1);
                    let bar_width = 12.0;
                    let max_height = 40.0;
                    
                    ui.horizontal(|ui| {
                        for (i, &count) in histogram.iter().enumerate() {
                            let height = (count as f32 / max_count as f32 * max_height).max(2.0);
                            let depth_val = i as f32 / (num_bins - 1) as f32;
                            
                            let color = egui::Color32::from_rgb(
                                (depth_val * 200.0 + 55.0) as u8,
                                (depth_val * 200.0 + 55.0) as u8,
                                (depth_val * 200.0 + 55.0) as u8,
                            );
                            
                            let (rect, response) = ui.allocate_exact_size(
                                egui::Vec2::new(bar_width, height),
                                egui::Sense::hover(),
                            );
                            
                            ui.painter().rect_filled(rect, 0.0, color);
                            response.on_hover_text(format!("Depth: {:.2}\nPixels: {}", depth_val, count));
                        }
                    });
                    
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Near").small());
                        ui.add_space(140.0);
                        ui.label(egui::RichText::new("Far").small());
                    });
                    
                    ui.add_space(8.0);
                    ui.label(format!("Band thresholds: {} skin, {} hair, {} clothes", 
                        skin_bands, hair_bands, clothes_bands));
                } else {
                    ui.label("⚠ Depth estimation not run");
                }
            });

            // Segmentation Section
            ui.collapsing("🎨 Segmentation", |ui| {
                if let Some(ref seg) = results.segmentation {
                    ui.label(egui::RichText::new("✅ Segmentation complete").color(egui::Color32::GREEN));
                    ui.label(format!("Pixels analyzed: {}", seg.regions.len()));
                    
                    use std::collections::HashMap;
                    let mut region_counts: HashMap<crate::ml::SegmentationRegion, usize> = HashMap::new();
                    for region in seg.regions.values() {
                        *region_counts.entry(*region).or_insert(0) += 1;
                    }
                    
                    ui.label("Region breakdown:");
                    for (region, count) in region_counts {
                        let pct = count as f32 / seg.regions.len() as f32 * 100.0;
                        ui.label(format!("  {:?}: {} pixels ({:.1}%)", region, count, pct));
                    }
                    
                    if !seg.region_bounds.is_empty() {
                        ui.label("Region bounds:");
                        for (region, (x, y, w, h)) in &seg.region_bounds {
                            ui.label(format!("  {:?}: ({:.1}%, {:.1}%) {:.1}%×{:.1}%", 
                                region, x*100.0, y*100.0, w*100.0, h*100.0));
                        }
                    }
                } else {
                    ui.label("⚠ Segmentation not run");
                }
            });
            
            ui.add_space(10.0);
            
            let enabled = !is_processing;
            if ui.add_enabled(enabled, egui::Button::new("🔄 Re-run Analysis")).clicked() {
                rerun_clicked = true;
            }
        });
        
        if rerun_clicked {
            super::processing::run_ml_analysis(app, ctx);
        }
    } else if has_input {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label("ML analysis not yet run");
            ui.add_space(20.0);
            let enabled = !is_processing;
            if ui.add_enabled(enabled, egui::Button::new("▶ Run ML Analysis")).clicked() {
                super::processing::run_ml_analysis(app, ctx);
            }
        });
    } else {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label("Load an image first");
        });
    }
}

fn output_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let has_output = app.output_image.is_some();
    
    if has_output {
        let mut reprocess = false;
        
        ui.vertical(|ui| {
            if let Some(output) = &app.output_image {
                let (img_w, img_h) = output.image.dimensions();
                let available = ui.available_size();
                let base_scale = calculate_fit_scale(img_w, img_h, available);
                let pixel_perfect_scale = base_scale.floor().max(1.0);
                let display_w = (img_w as f32 * pixel_perfect_scale * app.preview_zoom) as u32;

                ui.horizontal(|ui| {
                    ui.label(format!("{}×{} pixels", img_w, img_h));
                    ui.label(format!("{} colors", output.palette.len()));
                    ui.label(format!("Scale: {:.0}×", pixel_perfect_scale));
                });

                ui.horizontal(|ui| {
                    let spacer = (available.x - display_w as f32).max(0.0) / 2.0;
                    ui.add_space(spacer);
                    ui.image(&output.texture);
                });
            }

            if ui.button("🔄 Re-process").clicked() {
                reprocess = true;
            }
        });
        
        if reprocess {
            super::processing::process_image(app, ctx);
        }
    } else {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label("Output will appear here after processing");
            
            if app.input_image.is_some() {
                ui.add_space(20.0);
                if ui.button("▶ Process Now").clicked() {
                    super::processing::process_image(app, ctx);
                }
            }
        });
    }
}

fn drop_zone(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    let available = ui.available_size();
    
    ui.allocate_ui_with_layout(
        [available.x, available.y].into(),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            ui.add_space(60.0);
            ui.label(egui::RichText::new("📷").size(64.0));
            ui.add_space(20.0);
            ui.label(egui::RichText::new("Drop image here").size(20.0));
            ui.label(egui::RichText::new("or").size(14.0).weak());
            ui.add_space(10.0);
            if ui.button(egui::RichText::new("Browse...").size(16.0)).clicked() {
                super::processing::open_file_dialog(app);
            }
            ui.add_space(20.0);
            ui.label(egui::RichText::new("Supported: PNG, JPG, JPEG, WebP, BMP").small().weak());
        }
    );
}

/// Calculate scale to fit image in available space
pub fn calculate_fit_scale(image_width: u32, image_height: u32, available: egui::Vec2) -> f32 {
    let img_aspect = image_width as f32 / image_height as f32;
    let avail_aspect = available.x / available.y;
    
    if img_aspect > avail_aspect {
        available.x / image_width as f32
    } else {
        available.y / image_height as f32
    }
}
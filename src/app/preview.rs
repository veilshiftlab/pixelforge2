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
        let display_w = (img_w as f32 * scale * app.preview_zoom) as f32;
        let display_h = (img_h as f32 * scale * app.preview_zoom) as f32;

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{}×{} pixels", img_w, img_h));
                ui.separator();
                ui.label(format!("Zoom: {:.1}×", app.preview_zoom));
                ui.separator();
                if ui.button("Fit").clicked() {
                    app.preview_zoom = 1.0;
                }
                if ui.button("2×").clicked() {
                    app.preview_zoom = 2.0;
                }
                if ui.button("4×").clicked() {
                    app.preview_zoom = 4.0;
                }
            });
            
            ui.separator();
            
            let remaining = ui.available_size();
            let spacer_x = (remaining.x - display_w).max(0.0) / 2.0;
            ui.add_space(spacer_x);
            
            let texture = &input.texture;
            ui.image(egui::load::SizedTexture::new(texture.id(), egui::Vec2::new(display_w, display_h)));
        });
    } else {
        drop_zone(app, ui);
    }
}

fn ml_preview_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    // Clone data BEFORE using in closure to avoid borrow conflicts
    let ml_results = app.ml_results.clone();
    let has_input = app.input_image.is_some();
    let input_dims = app.input_image.as_ref().map(|i| i.image.dimensions());
    let is_processing = matches!(app.processing, crate::processing::ProcessingState::Running(_));
    
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
                    }
                } else {
                    ui.label(egui::RichText::new("⚠ No face detected").color(egui::Color32::YELLOW));
                }
            });

            // Depth Map Section
            ui.collapsing("📐 Depth Estimation", |ui| {
                if let Some(ref depth) = results.depth_map {
                    ui.label(egui::RichText::new("✅ Depth map generated").color(egui::Color32::GREEN));
                    
                    if let Some((w, h)) = input_dims {
                        ui.label(format!("Resolution: {}×{}", w, h));
                        
                        let min = depth.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = depth.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let avg = depth.iter().sum::<f32>() / depth.len() as f32;
                        
                        ui.label(format!("Range: {:.3} - {:.3} | Avg: {:.3}", min, max, avg));
                        
                        ui.add_space(8.0);
                        ui.label("Depth Map Visualization:");
                        
                        if let Some(depth_texture) = create_depth_texture(depth, w, h, ctx) {
                            let available = ui.available_width();
                            let display_scale = (available / w as f32).min(1.0);
                            let display_w = w as f32 * display_scale;
                            let display_h = h as f32 * display_scale;
                            
                            ui.image(egui::load::SizedTexture::new(depth_texture.id(), egui::Vec2::new(display_w, display_h)));
                        }
                    }
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
                    
                    ui.add_space(8.0);
                    ui.label("Segmentation Map Visualization:");
                    
                    if let Some((w, h)) = input_dims {
                        if let Some(seg_texture) = create_segmentation_texture(&seg.regions, w, h, ctx) {
                            let available = ui.available_width();
                            let display_scale = (available / w as f32).min(1.0);
                            let display_w = w as f32 * display_scale;
                            let display_h = h as f32 * display_scale;
                            
                            ui.image(egui::load::SizedTexture::new(seg_texture.id(), egui::Vec2::new(display_w, display_h)));
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
        
        // Call outside closure to avoid borrow conflict
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
    let mut reprocess_clicked = false;
    let has_input = app.input_image.is_some();
    
    let output_data = app.output_image.as_ref().map(|output| {
        let (img_w, img_h) = output.image.dimensions();
        let texture_id = output.texture.id();
        let palette_len = output.palette.len();
        (img_w, img_h, texture_id, palette_len)
    });
    
    if let Some((img_w, img_h, texture_id, palette_len)) = output_data {
        let available = ui.available_size();
        
        let base_scale = calculate_fit_scale(img_w, img_h, available);
        let scale = base_scale * app.preview_zoom;
        let display_w = (img_w as f32 * scale).max(8.0);
        let display_h = (img_h as f32 * scale).max(8.0);

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{}×{} pixels", img_w, img_h));
                ui.separator();
                ui.label(format!("{} colors", palette_len));
                ui.separator();
                ui.label(format!("Display: {:.1}×", scale));
            });
            
            ui.separator();
            
            let remaining = ui.available_size();
            let spacer_x = (remaining.x - display_w).max(0.0) / 2.0;
            ui.add_space(spacer_x);
            
            ui.image(egui::load::SizedTexture::new(texture_id, egui::Vec2::new(display_w, display_h)));
            
            ui.separator();
            
            if ui.button("🔄 Re-process").clicked() {
                reprocess_clicked = true;
            }
        });
    } else {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label("Output will appear here after processing");
            
            if has_input {
                ui.add_space(20.0);
                if ui.button("▶ Process Now").clicked() {
                    reprocess_clicked = true;
                }
            }
        });
    }
    
    // Call outside closure
    if reprocess_clicked {
        super::processing::process_image(app, ctx);
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
    let avail_aspect = if available.y > 0.0 { available.x / available.y } else { 1.0 };
    
    let scale = if img_aspect > avail_aspect {
        available.x / image_width as f32
    } else {
        available.y / image_height as f32
    };
    
    scale * 0.95
}

/// Create a texture from depth map for visualization
fn create_depth_texture(depth: &[f32], width: u32, height: u32, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let mut pixels = Vec::with_capacity(depth.len());
    
    for &d in depth {
        let val = (d.clamp(0.0, 1.0) * 255.0) as u8;
        pixels.push(egui::Color32::from_rgb(val, val, val));
    }
    
    let color_image = egui::ColorImage {
        size: [width as usize, height as usize],
        pixels,
    };
    
    Some(ctx.load_texture("depth_map", color_image, egui::TextureOptions::default()))
}

/// Create a texture from segmentation map for visualization
fn create_segmentation_texture(
    regions: &std::collections::HashMap<u32, crate::ml::SegmentationRegion>,
    width: u32,
    height: u32,
    ctx: &egui::Context,
) -> Option<egui::TextureHandle> {
    use crate::ml::SegmentationRegion;
    
    let region_colors = |region: SegmentationRegion| -> egui::Color32 {
        match region {
            SegmentationRegion::Background => egui::Color32::from_rgb(30, 30, 30),
            SegmentationRegion::Face => egui::Color32::from_rgb(255, 200, 150),
            SegmentationRegion::Eyes => egui::Color32::from_rgb(100, 150, 255),
            SegmentationRegion::Nose => egui::Color32::from_rgb(200, 150, 100),
            SegmentationRegion::Lips => egui::Color32::from_rgb(255, 100, 100),
            SegmentationRegion::Hair => egui::Color32::from_rgb(139, 90, 43),
            SegmentationRegion::Ears => egui::Color32::from_rgb(230, 180, 140),
            SegmentationRegion::Neck => egui::Color32::from_rgb(245, 195, 155),
            SegmentationRegion::Clothing => egui::Color32::from_rgb(100, 150, 100),
        }
    };
    
    let mut pixels = Vec::with_capacity((width * height) as usize);
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as u32;
            let color = regions.get(&idx)
                .map(|&r| region_colors(r))
                .unwrap_or(egui::Color32::from_rgb(30, 30, 30));
            pixels.push(color);
        }
    }
    
    let color_image = egui::ColorImage {
        size: [width as usize, height as usize],
        pixels,
    };
    
    Some(ctx.load_texture("segmentation_map", color_image, egui::TextureOptions::default()))
}
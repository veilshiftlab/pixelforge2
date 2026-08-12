//! Central preview panel

use super::state::{PixelForgeApp, PreviewTab};
use eframe::egui;
use image::GenericImageView;

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn draw(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            for (tab, label) in [
                (PreviewTab::Original, "📷 Original"),
                (PreviewTab::MLMaps,   "🤖 ML Maps"),
                (PreviewTab::Output,   "🎨 Output"),
            ] {
                if ui.selectable_label(app.preview_tab == tab, label).clicked() {
                    app.preview_tab = tab;
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Slider::new(&mut app.preview_zoom, 0.25..=8.0).text("Zoom").logarithmic(true));
            });
        });

        ui.separator();

        match app.preview_tab {
            PreviewTab::Original => original_panel(app, ui),
            PreviewTab::MLMaps   => ml_panel(app, ui, ctx),
            PreviewTab::Output   => output_panel(app, ui, ctx),
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Original tab
// ─────────────────────────────────────────────────────────────────────────────

fn original_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    if app.input_image.is_none() {
        drop_zone(app, ui);
        return;
    }
    let input = app.input_image.as_ref().unwrap();
    let (img_w, img_h) = input.image.dimensions();
    let available = ui.available_size();
    let scale = fit_scale(img_w, img_h, available) * app.preview_zoom;
    let dw = img_w as f32 * scale;
    let dh = img_h as f32 * scale;
    let tex_id = input.texture.id();

    // Extract path info before closure to avoid borrow conflict
    let path_name = input.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string());

    // Drop the borrow on input before the horizontal closure
    let _ = input;

    // Info row
    ui.horizontal(|ui| {
        ui.label(format!("{}×{}", img_w, img_h));
        if let Some(name) = path_name {
            ui.separator();
            ui.label(egui::RichText::new(name).weak());
        }
        ui.separator();
        for (label, zoom) in [("Fit", 1.0f32), ("2×", 2.0), ("4×", 4.0)] {
            if ui.small_button(label).clicked() { app.preview_zoom = zoom; }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("🔄 Load").on_hover_text("Load another image").clicked() {
                super::processing::open_file_dialog(app);
            }
            if ui.small_button("✕ Clear").on_hover_text("Clear current image").clicked() {
                super::processing::clear_image(app);
            }
        });
    });

    ui.separator();

    egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
        // Center the image when smaller than the scroll area
        let avail = ui.available_size();
        ui.vertical(|ui| {
            if dh < avail.y {
                ui.add_space((avail.y - dh) * 0.5);
            }
            ui.horizontal(|ui| {
                if dw < avail.x {
                    ui.add_space((avail.x - dw) * 0.5);
                }
                ui.image(egui::load::SizedTexture::new(tex_id, egui::Vec2::new(dw, dh)));
            });
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Maps tab
// ─────────────────────────────────────────────────────────────────────────────

fn ml_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let has_input = app.input_image.is_some();
    let is_running = matches!(app.processing, crate::processing::ProcessingState::Running(_));

    if !has_input {
        centered_hint(ui, "Load an image first");
        return;
    }

    if app.ml_results.is_none() {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);
            ui.label("ML analysis has not been run yet.");
            ui.add_space(12.0);
            if ui.add_enabled(!is_running, egui::Button::new("▶ Run Analysis")).clicked() {
                super::processing::run_ml_analysis(app, ctx);
            }
        });
        return;
    }

    let (img_w, img_h) = app.input_image.as_ref().unwrap().image.dimensions();

    // Snapshot the lightweight parts before entering the collapsing closures
    // so we don't hold a borrow on app.ml_results through the whole scroll area.
    let has_depth = app.ml_results.as_ref().unwrap().depth_map.is_some();
    let has_edge  = app.ml_results.as_ref().unwrap().edge_map.is_some();

    let depth_stats = app.ml_results.as_ref()
        .and_then(|r| r.depth_map.as_deref())
        .map(|d| {
            let mn = d.iter().cloned().fold(f32::INFINITY,     f32::min);
            let mx = d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let av = d.iter().sum::<f32>() / d.len().max(1) as f32;
            (mn, mx, av)
        });

    // Build textures now (before the scroll area closure captures app)
    let depth_tex = app.ml_results.as_ref()
        .and_then(|r| r.depth_map.as_deref())
        .and_then(|d| depth_texture(d, img_w, img_h, ctx));

    let edge_tex = app.ml_results.as_ref()
        .and_then(|r| r.edge_map.as_deref())
        .and_then(|e| edge_texture(e, img_w, img_h, ctx));

    // SLIC label map visualization — shows the superpixel regions so the user
    // can see how the image is being segmented and tune K / spatial_weight.
    let slic_tex = app.ml_results.as_ref()
        .and_then(|r| r.slic_labels.as_ref())
        .and_then(|labels| slic_label_texture(labels, img_w, img_h, ctx));

    let slic_info = app.ml_results.as_ref()
        .and_then(|r| r.slic_labels.as_ref())
        .map(|labels| {
            let mut s: Vec<u32> = labels.iter().copied().collect();
            s.sort_unstable();
            s.dedup();
            (labels.len(), s.len())
        });

    let mut rerun = false;

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {

        // ── Depth map ───────────────────────────────────────────────────────
        ui.collapsing("📐 Depth Map", |ui| {
            if has_depth {
                ui.colored_label(egui::Color32::GREEN, "✅ Generated");
                if let Some((mn, mx, av)) = depth_stats {
                    ui.label(format!("Range {:.3}–{:.3}  avg {:.3}", mn, mx, av));
                }
                if let Some(ref tex) = depth_tex {
                    map_image(ui, tex.id(), img_w, img_h);
                }
            } else {
                ui.label("Not run");
            }
        });

        // ── Edges ───────────────────────────────────────────────────────────
        ui.collapsing("✏ Edge Map (DexiNed)", |ui| {
            if has_edge {
                ui.colored_label(egui::Color32::GREEN, "✅ Generated");
                if let Some(ref tex) = edge_tex {
                    map_image(ui, tex.id(), img_w, img_h);
                }
            } else {
                ui.label("Not run");
                ui.label(egui::RichText::new("Enable DexiNed in ML Analysis and re-run.").small().weak());
            }
        });

        // ── SLIC regions ───────────────────────────────────────────────────
        ui.collapsing("🧩 SLIC Regions", |ui| {
            if let Some(ref tex) = slic_tex {
                if let Some((n_pixels, n_clusters)) = slic_info {
                    ui.label(format!("{} pixels, {} active clusters", n_pixels, n_clusters));
                }
                ui.label(egui::RichText::new(
                    "Each color = one superpixel. Adjust K and Spatial weight in the left panel, then Re-process."
                ).small().weak());
                map_image(ui, tex.id(), img_w, img_h);
            } else {
                ui.label("Not computed — run Process once first.");
            }
        });

        ui.add_space(8.0);
        if ui.add_enabled(!is_running, egui::Button::new("🔄 Re-run Analysis")).clicked() {
            rerun = true;
        }
    });

    if rerun {
        super::processing::run_ml_analysis(app, ctx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output tab
// ─────────────────────────────────────────────────────────────────────────────

fn output_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let has_input  = app.input_image.is_some();
    let is_running = matches!(app.processing, crate::processing::ProcessingState::Running(_));

    let output_info = app.output_image.as_ref().map(|o| {
        let (w, h) = o.image.dimensions();
        (w, h, o.texture.id(), o.palette.len())
    });

    let mut reprocess = false;

    if let Some((img_w, img_h, tex_id, pal_len)) = output_info {
        let available = ui.available_size();
        let scale = fit_scale(img_w, img_h, available) * app.preview_zoom;
        let dw = img_w as f32 * scale;
        let dh = img_h as f32 * scale;

        ui.horizontal(|ui| {
            ui.label(format!("{}×{}  ·  {} colors  ·  {:.1}×", img_w, img_h, pal_len, scale));
            for (label, zoom) in [("Fit", 1.0f32), ("2×", 2.0), ("4×", 4.0)] {
                if ui.small_button(label).clicked() { app.preview_zoom = zoom; }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!is_running, egui::Button::new("🔄 Re-process")).clicked() {
                    reprocess = true;
                }
            });
        });

        ui.separator();

        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            let avail = ui.available_size();
            ui.vertical(|ui| {
                if dh < avail.y {
                    ui.add_space((avail.y - dh) * 0.5);
                }
                ui.horizontal(|ui| {
                    if dw < avail.x {
                        ui.add_space((avail.x - dw) * 0.5);
                    }
                    ui.image(egui::load::SizedTexture::new(tex_id, egui::Vec2::new(dw, dh)));
                });
            });
        });
    } else {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);
            ui.label("Output will appear here after processing.");
            if has_input {
                ui.add_space(12.0);
                if ui.add_enabled(!is_running, egui::Button::new("▶ Process Now")).clicked() {
                    reprocess = true;
                }
            }
        });
    }

    if reprocess {
        super::processing::process_image(app, ctx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Drop zone
// ─────────────────────────────────────────────────────────────────────────────

fn drop_zone(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    let hovering = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());

    // Draw a highlight rect when something is being dragged over
    let rect = ui.available_rect_before_wrap();
    if hovering {
        ui.painter().rect_filled(
            rect,
            8.0,
            egui::Color32::from_rgba_unmultiplied(100, 160, 255, 25),
        );
        ui.painter().rect_stroke(
            rect,
            8.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 160, 255)),
        );
    }

    ui.vertical_centered(|ui| {
        ui.add_space(60.0);
        ui.label(egui::RichText::new("📷").size(64.0));
        ui.add_space(12.0);
        ui.label(egui::RichText::new(if hovering { "Release to open" } else { "Drop an image here" }).size(20.0));
        ui.add_space(4.0);
        ui.label(egui::RichText::new("or").size(13.0).weak());
        ui.add_space(8.0);
        if ui.button(egui::RichText::new("Browse…").size(15.0)).clicked() {
            super::processing::open_file_dialog(app);
        }
        ui.add_space(16.0);
        ui.label(egui::RichText::new("PNG · JPG · WebP · BMP").small().weak());
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Texture builders — called once per frame while the maps tab is open.
// egui caches textures by name and pixel data, so repeated calls with the
// same data are a no-op after the first upload.
// ─────────────────────────────────────────────────────────────────────────────

fn depth_texture(depth: &[f32], w: u32, h: u32, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    if depth.len() != (w * h) as usize { return None; }
    let pixels: Vec<egui::Color32> = depth.iter()
        .map(|&d| { let (r,g,b) = turbo(d.clamp(0.0,1.0)); egui::Color32::from_rgb(r,g,b) })
        .collect();
    Some(ctx.load_texture("ml_depth", egui::ColorImage { size: [w as usize, h as usize], pixels }, egui::TextureOptions::default()))
}

fn edge_texture(edges: &[f32], w: u32, h: u32, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    if edges.len() != (w * h) as usize { return None; }
    let pixels: Vec<egui::Color32> = edges.iter()
        .map(|&e| { let v = (e.clamp(0.0,1.0)*255.0) as u8; egui::Color32::from_rgb(v,v,v) })
        .collect();
    Some(ctx.load_texture("ml_edges", egui::ColorImage { size: [w as usize, h as usize], pixels }, egui::TextureOptions::default()))
}

/// SLIC label map → colored texture. Each cluster ID gets a distinct color
/// so the user can see how the image is being segmented.
fn slic_label_texture(labels: &[u32], w: u32, h: u32, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    if labels.len() != (w * h) as usize { return None; }

    // Distinct colors for up to 16 cluster IDs (cycled if more)
    let palette: [egui::Color32; 16] = [
        egui::Color32::from_rgb(230,  25,  75),  // red
        egui::Color32::from_rgb( 60, 180,  75),  // green
        egui::Color32::from_rgb(255, 225,  25),  // yellow
        egui::Color32::from_rgb(  0, 130, 200),  // blue
        egui::Color32::from_rgb(245, 130,  48),  // orange
        egui::Color32::from_rgb(145,  30, 180),  // purple
        egui::Color32::from_rgb( 70, 240, 240),  // cyan
        egui::Color32::from_rgb(240,  50, 230),  // magenta
        egui::Color32::from_rgb(210, 245,  60),  // lime
        egui::Color32::from_rgb(250, 190, 190),  // pink
        egui::Color32::from_rgb(  0, 128, 128),  // teal
        egui::Color32::from_rgb(230, 190, 255),  // lavender
        egui::Color32::from_rgb(170, 110,  40),  // brown
        egui::Color32::from_rgb(255, 250, 200),  // beige
        egui::Color32::from_rgb(128, 128, 128),  // gray
        egui::Color32::from_rgb(  0,   0,   0),  // black
    ];

    let pixels: Vec<egui::Color32> = labels.iter()
        .map(|&label| palette[(label as usize) % palette.len()])
        .collect();

    Some(ctx.load_texture("ml_slic", egui::ColorImage { size: [w as usize, h as usize], pixels }, egui::TextureOptions::default()))
}

/// Display a map scaled to fit the available panel width (never upscale)
fn map_image(ui: &mut egui::Ui, id: egui::TextureId, map_w: u32, map_h: u32) {
    let avail = ui.available_width();
    let scale = (avail / map_w as f32).min(1.0);
    let dw = map_w as f32 * scale;
    let dh = map_h as f32 * scale;
    ui.image(egui::load::SizedTexture::new(id, egui::Vec2::new(dw, dh)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Colormap & utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Turbo colormap (Mikhailov 2019) — better perceptual range than grayscale for depth.
/// t=0 → dark, t=0.5 → bright green, t=1 → deep red.
fn turbo(t: f32) -> (u8, u8, u8) {
    let r = (0.13572138 + t*(4.61539260 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943))))) * 255.0;
    let g = (0.09140261 + t*(2.19418839 + t*(4.84296658  + t*(-14.18503333 + t*(  4.27729857  + t* 2.82956604))))) * 255.0;
    let b = (0.10667330 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))))) * 255.0;
    (r.clamp(0.0, 255.0) as u8, g.clamp(0.0, 255.0) as u8, b.clamp(0.0, 255.0) as u8)
}

/// Scale factor to fit an image inside an egui available_size rect.
/// Uses conservative calculation with 4px margin to prevent scrollbars.
pub fn fit_scale(img_w: u32, img_h: u32, available: egui::Vec2) -> f32 {
    let margin = 4.0; // Conservative pixel margin
    let sx = (available.x - margin) / img_w as f32;
    let sy = if available.y > margin { (available.y - margin) / img_h as f32 } else { sx };
    sx.min(sy).max(0.1) // Ensure minimum scale of 0.1
}

fn centered_hint(ui: &mut egui::Ui, text: &str) {
    ui.vertical_centered(|ui| { ui.add_space(80.0); ui.label(text); });
}

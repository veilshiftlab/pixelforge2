//! Central preview panel

use super::state::{PixelForgeApp, PreviewTab};
use crate::ml::SegmentationRegion;
use eframe::egui;
use image::GenericImageView;
use std::cmp::Reverse;
use std::collections::HashMap;

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

    // Info row
    ui.horizontal(|ui| {
        ui.label(format!("{}×{}", img_w, img_h));
        if let Some(path) = &input.path {
            if let Some(name) = path.file_name() {
                ui.separator();
                ui.label(egui::RichText::new(name.to_string_lossy()).weak());
            }
        }
        ui.separator();
        for (label, zoom) in [("Fit", 1.0f32), ("2×", 2.0), ("4×", 4.0)] {
            if ui.small_button(label).clicked() { app.preview_zoom = zoom; }
        }
    });

    ui.separator();

    let tex_id = input.texture.id();
    egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
        // Centre the image when smaller than the scroll area
        let avail = ui.available_size();
        if dw < avail.x { ui.add_space((avail.x - dw) * 0.5); }
        ui.image(egui::load::SizedTexture::new(tex_id, egui::Vec2::new(dw, dh)));
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
    let has_face  = app.ml_results.as_ref().unwrap().face_bounds.is_some();
    let has_depth = app.ml_results.as_ref().unwrap().depth_map.is_some();
    let has_seg   = app.ml_results.as_ref().unwrap().segmentation.is_some();
    let has_edge  = app.ml_results.as_ref().unwrap().edge_map.is_some();

    let face_conf_str = app.ml_results.as_ref()
        .and_then(|r| r.face.as_ref())
        .map(|f| format!("{:.1}%", f.confidence * 100.0));

    let face_bounds_str = app.ml_results.as_ref()
        .and_then(|r| r.face_bounds)
        .map(|b| format!("{:.1}%×{:.1}% at ({:.1}%, {:.1}%)",
            b.width*100.0, b.height*100.0, b.x*100.0, b.y*100.0));

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

    let seg_tex = app.ml_results.as_ref()
        .and_then(|r| r.segmentation.as_ref())
        .and_then(|s| seg_texture(&s.regions, img_w, img_h, ctx));

    let edge_tex = app.ml_results.as_ref()
        .and_then(|r| r.edge_map.as_deref())
        .and_then(|e| edge_texture(e, img_w, img_h, ctx));

    // Region breakdown is the only thing that needs cloning (it's a HashMap)
    let region_summary: Vec<(SegmentationRegion, f32)> = app.ml_results.as_ref()
        .and_then(|r| r.segmentation.as_ref())
        .map(|s| {
            let total = s.regions.len().max(1);
            let mut counts: HashMap<SegmentationRegion, usize> = HashMap::new();
            for &r in s.regions.values() { *counts.entry(r).or_insert(0) += 1; }
            let mut pairs: Vec<_> = counts.into_iter()
                .map(|(r, n)| (r, n as f32 / total as f32 * 100.0))
                .collect();
            pairs.sort_by_key(|(_, pct)| Reverse((*pct * 1000.0) as u32));
            pairs.truncate(8);
            pairs
        })
        .unwrap_or_default();

    let mut rerun = false;

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {

        // ── Face detection ──────────────────────────────────────────────────
        ui.collapsing("👤 Face Detection", |ui| {
            if has_face {
                ui.colored_label(egui::Color32::GREEN, "✅ Detected");
                if let Some(ref s) = face_conf_str   { ui.label(format!("Confidence: {}", s)); }
                if let Some(ref s) = face_bounds_str  { ui.label(s); }
                if let Some(ref r) = app.ml_results {
                    if let Some(ref lm) = r.landmarks {
                        ui.label(format!("{} landmark points", lm.points.len()));
                    }
                }
            } else {
                ui.colored_label(egui::Color32::YELLOW, "⚠ No face detected");
                ui.label(egui::RichText::new("Lower the confidence threshold and re-run.").small().weak());
            }
        });

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

        // ── Segmentation ────────────────────────────────────────────────────
        ui.collapsing("🎨 Face Parsing", |ui| {
            if has_seg {
                ui.colored_label(egui::Color32::GREEN, "✅ Complete");
                for (region, pct) in &region_summary {
                    let (r, g, b, _) = region.rgba();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("■").color(egui::Color32::from_rgb(r, g, b)));
                        ui.label(format!("{}: {:.1}%", region.display_name(), pct));
                    });
                }
                if let Some(ref tex) = seg_tex {
                    map_image(ui, tex.id(), img_w, img_h);
                }
            } else {
                ui.label("Not run");
            }
        });

        // ── Edges ───────────────────────────────────────────────────────────
        ui.collapsing("✏ Edge Map (TEED)", |ui| {
            if has_edge {
                ui.colored_label(egui::Color32::GREEN, "✅ Generated");
                if let Some(ref tex) = edge_tex {
                    map_image(ui, tex.id(), img_w, img_h);
                }
            } else {
                ui.label("Not run");
                ui.label(egui::RichText::new("Enable TEED in ML Analysis and re-run.").small().weak());
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
            if dw < avail.x { ui.add_space((avail.x - dw) * 0.5); }
            ui.image(egui::load::SizedTexture::new(tex_id, egui::Vec2::new(dw, dh)));
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

fn seg_texture(regions: &HashMap<u32, SegmentationRegion>, w: u32, h: u32, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let total = (w * h) as usize;
    if total == 0 { return None; }
    let mut pixels = vec![egui::Color32::from_rgb(30, 30, 30); total];
    for (&idx, &region) in regions {
        if (idx as usize) < total {
            let (r,g,b,_) = region.rgba();
            pixels[idx as usize] = egui::Color32::from_rgb(r,g,b);
        }
    }
    Some(ctx.load_texture("ml_seg", egui::ColorImage { size: [w as usize, h as usize], pixels }, egui::TextureOptions::default()))
}

fn edge_texture(edges: &[f32], w: u32, h: u32, ctx: &egui::Context) -> Option<egui::TextureHandle> {
    if edges.len() != (w * h) as usize { return None; }
    let pixels: Vec<egui::Color32> = edges.iter()
        .map(|&e| { let v = (e.clamp(0.0,1.0)*255.0) as u8; egui::Color32::from_rgb(v,v,v) })
        .collect();
    Some(ctx.load_texture("ml_edges", egui::ColorImage { size: [w as usize, h as usize], pixels }, egui::TextureOptions::default()))
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

/// Scale factor to fit an image inside an egui available_size rect (with 1% margin).
pub fn fit_scale(img_w: u32, img_h: u32, available: egui::Vec2) -> f32 {
    let sx = available.x / img_w as f32;
    let sy = if available.y > 0.0 { available.y / img_h as f32 } else { sx };
    sx.min(sy) * 0.99
}

fn centered_hint(ui: &mut egui::Ui, text: &str) {
    ui.vertical_centered(|ui| { ui.add_space(80.0); ui.label(text); });
}

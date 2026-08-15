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
                (PreviewTab::Original,    "📷 Original"),
                (PreviewTab::MLMaps,      "🤖 ML Maps"),
                (PreviewTab::Flat,        "🎨 Flat (post-DTF)"),
                (PreviewTab::Preprocessed, "📐 Preprocessed"),
                (PreviewTab::Output,      "✨ Output"),
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
            PreviewTab::Original    => original_panel(app, ui),
            PreviewTab::MLMaps      => ml_panel(app, ui, ctx),
            PreviewTab::Flat        => flat_panel(app, ui, ctx),
            PreviewTab::Preprocessed => preprocessed_panel(app, ui, ctx),
            PreviewTab::Output      => output_panel(app, ui, ctx),
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
// ML Maps tab — Phase 1 fixes:
//   • C3: NEAREST texture filtering (was LINEAR → gradient pixels)
//   • P6: cached TextureHandles in app state (rebuilt only on ML change)
//   • C4: 1:1 native zoom toggle for pixel-accurate inspection
//   • U5: source resolution label next to each map
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

    // ── Phase 1 — P6: rebuild cached textures only if missing ──────────────────
    // The ML maps tab is the only consumer; we keep handles in app state and
    // invalidate them in `processing::load_image` / `clear_image` / ML result
    // arrival / SLIC recompute.
    let needs_depth_tex = app.ml_depth_texture.is_none()
        && app.ml_results.as_ref().and_then(|r| r.depth_map.as_deref()).is_some();
    if needs_depth_tex {
        if let Some(d) = app.ml_results.as_ref().and_then(|r| r.depth_map.as_deref()) {
            app.ml_depth_texture = Some(build_map_texture(
                ctx, "ml_depth_cached", d, img_w, img_h, MapKind::Turbo,
            ));
        }
    }

    let needs_edge_tex = app.ml_edge_texture.is_none()
        && app.ml_results.as_ref().and_then(|r| r.edge_map.as_deref()).is_some();
    if needs_edge_tex {
        if let Some(e) = app.ml_results.as_ref().and_then(|r| r.edge_map.as_deref()) {
            app.ml_edge_texture = Some(build_map_texture(
                ctx, "ml_edge_cached", e, img_w, img_h, MapKind::Gray,
            ));
        }
    }

    let needs_slic_tex = app.ml_slic_texture.is_none()
        && app.ml_results.as_ref().and_then(|r| r.slic_labels.as_deref()).is_some();
    if needs_slic_tex {
        if let Some(labels) = app.ml_results.as_ref().and_then(|r| r.slic_labels.as_deref()) {
            app.ml_slic_texture = Some(build_slic_texture(
                ctx, "ml_slic_cached", labels, img_w, img_h,
            ));
        }
    }

    // Snapshot lightweight state for the closures
    let has_depth = app.ml_results.as_ref().unwrap().depth_map.is_some();
    let has_edge  = app.ml_results.as_ref().unwrap().edge_map.is_some();
    let has_slic  = app.ml_results.as_ref().unwrap().slic_labels.is_some();

    let depth_stats = app.ml_results.as_ref()
        .and_then(|r| r.depth_map.as_deref())
        .map(|d| {
            let mn = d.iter().cloned().fold(f32::INFINITY,     f32::min);
            let mx = d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let av = d.iter().sum::<f32>() / d.len().max(1) as f32;
            (mn, mx, av)
        });

    let slic_info = app.ml_results.as_ref()
        .and_then(|r| r.slic_labels.as_ref())
        .map(|labels| {
            let mut s: Vec<u32> = labels.iter().copied().collect();
            s.sort_unstable();
            s.dedup();
            (labels.len(), s.len())
        });

    // Clone the Option<TextureHandle> (cheap Arc) so we can move into closures
    // without borrowing app.
    let depth_tex = app.ml_depth_texture.clone();
    let edge_tex  = app.ml_edge_texture.clone();
    let slic_tex  = app.ml_slic_texture.clone();
    let native    = app.ml_maps_native;

    let mut rerun = false;

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        // ── Top toolbar: 1:1 toggle + zoom presets ───────────────────────────
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.ml_maps_native, "1:1 (native resolution)")
                .on_hover_text(
                    "Display each map at 1 source pixel = 1 screen pixel.\n\
                     Disable to fit-to-width. Useful for inspecting exact ML output.");
            ui.separator();
            ui.label(egui::RichText::new(format!("Source: {}×{}", img_w, img_h)).small().weak());
        });
        ui.separator();

        // ── Depth map ───────────────────────────────────────────────────────
        ui.collapsing("📐 Depth Map", |ui| {
            if has_depth {
                ui.colored_label(egui::Color32::GREEN, "✅ Generated");
                if let Some((mn, mx, av)) = depth_stats {
                    ui.label(egui::RichText::new(
                        format!("Range {:.3}–{:.3}  avg {:.3}", mn, mx, av)
                    ).small().weak());
                }
                ui.label(egui::RichText::new(
                    format!("{}×{} (Depth-Anything V2)", img_w, img_h)
                ).small().weak());
                if let Some(ref tex) = depth_tex {
                    map_image(ui, tex.id(), img_w, img_h, native);
                }
            } else {
                ui.label("Not run");
            }
        });

        // ── Edges ───────────────────────────────────────────────────────────
        ui.collapsing("✏ Edge Map (DexiNed)", |ui| {
            if has_edge {
                ui.colored_label(egui::Color32::GREEN, "✅ Generated");
                ui.label(egui::RichText::new(
                    format!("{}×{} (DexiNed, ImageNet norm)", img_w, img_h)
                ).small().weak());
                if let Some(ref tex) = edge_tex {
                    map_image(ui, tex.id(), img_w, img_h, native);
                }
            } else {
                ui.label("Not run");
                ui.label(egui::RichText::new("Enable DexiNed in ML Analysis and re-run.").small().weak());
            }
        });

        // ── SLIC regions ───────────────────────────────────────────────────
        ui.collapsing("🧩 SLIC Regions", |ui| {
            if has_slic {
                if let Some((n_pixels, n_clusters)) = slic_info {
                    ui.label(egui::RichText::new(
                        format!("{}×{}  ·  {} pixels  ·  {} active clusters",
                            img_w, img_h, n_pixels, n_clusters)
                    ).small().weak());
                }
                ui.label(egui::RichText::new(
                    "Each color = one superpixel. Adjust K and Spatial weight in the left panel, then Re-process."
                ).small().weak());
                if let Some(ref tex) = slic_tex {
                    map_image(ui, tex.id(), img_w, img_h, native);
                }
            } else {
                ui.label("Not computed — run Process once first.");
            }
        });

        ui.add_space(8.0);
        if ui.add_enabled(!is_running, egui::Button::new("🔄 Re-run Analysis")).clicked() {
            rerun = true;
        }
        let _ = native; // suppress unused warning
    });

    if rerun {
        super::processing::run_ml_analysis(app, ctx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat (post-depth-to-flat) tab — Phase 1 — U6
// ─────────────────────────────────────────────────────────────────────────────

fn flat_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let is_running = matches!(app.processing, crate::processing::ProcessingState::Running(_));

    let info = app.flat_color_image.as_ref().map(|img| {
        let (w, h) = img.dimensions();
        let tex = super::processing::upload_texture(ctx, "flat_preview", img);
        (w, h, tex.id(), img as *const _ as usize)
    });

    if let Some((w, h, tex_id, _)) = info {
        let available = ui.available_size();
        let scale = fit_scale(w, h, available) * app.preview_zoom;
        let dw = w as f32 * scale;
        let dh = h as f32 * scale;

        ui.horizontal(|ui| {
            ui.label(format!("{}×{}  ·  post depth-to-flat  ·  {:.1}×", w, h, scale));
            for (label, zoom) in [("Fit", 1.0f32), ("2×", 2.0), ("4×", 4.0)] {
                if ui.small_button(label).clicked() { app.preview_zoom = zoom; }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!is_running, egui::Button::new("🔄 Re-process")).clicked() {
                    super::processing::process_image(app, ctx);
                }
            });
        });

        ui.separator();

        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            let avail = ui.available_size();
            ui.vertical(|ui| {
                if dh < avail.y { ui.add_space((avail.y - dh) * 0.5); }
                ui.horizontal(|ui| {
                    if dw < avail.x { ui.add_space((avail.x - dw) * 0.5); }
                    ui.image(egui::load::SizedTexture::new(tex_id, egui::Vec2::new(dw, dh)));
                });
            });
        });
    } else {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);
            ui.label("Post-depth-to-flat image will appear here after processing.");
            ui.add_space(12.0);
            if ui.add_enabled(app.input_image.is_some() && !is_running, egui::Button::new("▶ Process Now")).clicked() {
                super::processing::process_image(app, ctx);
            }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessed (post-transform) tab — Phase 1 — U7
// ─────────────────────────────────────────────────────────────────────────────

fn preprocessed_panel(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let is_running = matches!(app.processing, crate::processing::ProcessingState::Running(_));

    let info = app.preprocessed_image.as_ref().map(|img| {
        let (w, h) = img.dimensions();
        let tex = super::processing::upload_texture(ctx, "preprocessed_preview", img);
        (w, h, tex.id())
    });

    if let Some((w, h, tex_id)) = info {
        let available = ui.available_size();
        let scale = fit_scale(w, h, available) * app.preview_zoom;
        let dw = w as f32 * scale;
        let dh = h as f32 * scale;

        ui.horizontal(|ui| {
            ui.label(format!("{}×{}  ·  post-transform  ·  {:.1}×", w, h, scale));
            for (label, zoom) in [("Fit", 1.0f32), ("2×", 2.0), ("4×", 4.0)] {
                if ui.small_button(label).clicked() { app.preview_zoom = zoom; }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!is_running, egui::Button::new("🔄 Re-process")).clicked() {
                    super::processing::process_image(app, ctx);
                }
            });
        });

        ui.separator();

        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            let avail = ui.available_size();
            ui.vertical(|ui| {
                if dh < avail.y { ui.add_space((avail.y - dh) * 0.5); }
                ui.horizontal(|ui| {
                    if dw < avail.x { ui.add_space((avail.x - dw) * 0.5); }
                    ui.image(egui::load::SizedTexture::new(tex_id, egui::Vec2::new(dw, dh)));
                });
            });
        });
    } else {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);
            ui.label("Post-transform image will appear here after processing.");
            ui.add_space(12.0);
            if ui.add_enabled(app.input_image.is_some() && !is_running, egui::Button::new("▶ Process Now")).clicked() {
                super::processing::process_image(app, ctx);
            }
        });
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
// Texture builders — Phase 1 fixes:
//   • C3: NEAREST filtering (was LINEAR → gradient pixels along map edges)
//   • P6: cached in app state, rebuilt only when ML results change
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum MapKind {
    Turbo,  // depth
    Gray,   // edges
}

/// Build a single egui texture from a flat f32 map. NEAREST filtering is
/// critical — linear would interpolate between adjacent depth/edge values
/// and produce the gradient pixels the user reported in the preview.
fn build_map_texture(
    ctx: &egui::Context,
    name: &str,
    data: &[f32],
    w: u32,
    h: u32,
    kind: MapKind,
) -> egui::TextureHandle {
    let pixels: Vec<egui::Color32> = data.iter().map(|&v| {
        let v = v.clamp(0.0, 1.0);
        match kind {
            MapKind::Turbo => {
                let (r, g, b) = turbo(v);
                egui::Color32::from_rgb(r, g, b)
            }
            MapKind::Gray => {
                let g8 = (v * 255.0) as u8;
                egui::Color32::from_rgb(g8, g8, g8)
            }
        }
    }).collect();

    ctx.load_texture(
        name,
        egui::ColorImage { size: [w as usize, h as usize], pixels },
        egui::TextureOptions::NEAREST,
    )
}

/// Build a colored texture from SLIC cluster labels. One color per cluster ID.
fn build_slic_texture(
    ctx: &egui::Context,
    name: &str,
    labels: &[u32],
    w: u32,
    h: u32,
) -> egui::TextureHandle {
    const PALETTE: [egui::Color32; 16] = [
        egui::Color32::from_rgb(230,  25,  75),
        egui::Color32::from_rgb( 60, 180,  75),
        egui::Color32::from_rgb(255, 225,  25),
        egui::Color32::from_rgb(  0, 130, 200),
        egui::Color32::from_rgb(245, 130,  48),
        egui::Color32::from_rgb(145,  30, 180),
        egui::Color32::from_rgb( 70, 240, 240),
        egui::Color32::from_rgb(240,  50, 230),
        egui::Color32::from_rgb(210, 245,  60),
        egui::Color32::from_rgb(250, 190, 190),
        egui::Color32::from_rgb(  0, 128, 128),
        egui::Color32::from_rgb(230, 190, 255),
        egui::Color32::from_rgb(170, 110,  40),
        egui::Color32::from_rgb(255, 250, 200),
        egui::Color32::from_rgb(128, 128, 128),
        egui::Color32::from_rgb(  0,   0,   0),
    ];
    let pixels: Vec<egui::Color32> = labels.iter()
        .map(|&label| PALETTE[(label as usize) % PALETTE.len()])
        .collect();
    ctx.load_texture(
        name,
        egui::ColorImage { size: [w as usize, h as usize], pixels },
        egui::TextureOptions::NEAREST,
    )
}

/// Display a map. When `native` is true, render at 1:1 (1 source pixel = 1
/// screen pixel); otherwise fit-to-width. Both modes use NEAREST filtering
/// because the texture itself was uploaded with NEAREST.
fn map_image(ui: &mut egui::Ui, id: egui::TextureId, map_w: u32, map_h: u32, native: bool) {
    let (dw, dh) = if native {
        (map_w as f32, map_h as f32)
    } else {
        let avail = ui.available_width();
        let scale = (avail / map_w as f32).min(1.0);
        (map_w as f32 * scale, map_h as f32 * scale)
    };
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

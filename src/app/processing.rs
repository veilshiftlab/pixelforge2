//! App-level processing callbacks
//!
//! This module handles only egui-aware concerns:
//! - File dialogs and image loading/exporting
//! - Triggering ML analysis
//! - Calling the pure pipeline (`processing::pipeline::run`) and
//!   converting the result into egui textures + app state
//!
//! Heavy pixel-art pipeline logic lives in `crate::processing::pipeline`.

use super::state::{AspectRatioMode, OutputImage, PixelForgeApp};
use crate::image::{ExportConfig, ImageExporter};
use crate::processing::{ProcessingState, ProcessingStatus};
use crate::processing::pipeline::{PipelineInput};
use eframe::egui::{self, ColorImage};
use image::GenericImageView;

// ─────────────────────────────────────────────────────────────────────────────
// Image I/O
// ─────────────────────────────────────────────────────────────────────────────

pub fn load_image(app: &mut PixelForgeApp, path: &std::path::Path, ctx: &egui::Context) {
    match crate::image::ImageLoader::load_with_processor(path, 1024) {
        Ok(processor) => {
            let img = processor.get_processed_image()
                .unwrap_or_else(|_| processor.get_original_image().clone());
            let texture = upload_texture(ctx, "input_image", &img);

            app.custom_output_width  = img.width().min(512);
            app.custom_output_height = img.height().min(512);

            app.input_image = Some(super::state::InputImage {
                image: img,
                texture,
                path: Some(path.to_path_buf()),
            });
            app.image_processor = Some(processor);

            app.preprocessed_image = None;
            app.flat_color_image   = None;
            app.ml_results         = None;
            app.output_image       = None;
            app.processing         = ProcessingState::Idle;
            // Phase 1 — P6: invalidate cached ML map textures
            app.ml_depth_texture = None;
            app.ml_edge_texture  = None;
            app.ml_slic_texture  = None;
            app.ml_seg_texture   = None;
            // Phase 8 — clear stale warnings from any previous image
            app.pipeline_warnings.clear();

            log::info!("Loaded: {}", path.display());
        }
        Err(e) => log::error!("Failed to load {}: {}", path.display(), e),
    }
}

pub fn clear_image(app: &mut PixelForgeApp) {
    app.input_image        = None;
    app.image_processor    = None;
    app.preprocessed_image = None;
    app.flat_color_image   = None;
    app.ml_results         = None;
    app.output_image       = None;
    app.processing         = ProcessingState::Idle;
    // Phase 1 — P6: invalidate cached ML map textures
    app.ml_depth_texture = None;
    app.ml_edge_texture  = None;
    app.ml_slic_texture  = None;
    app.ml_seg_texture   = None;
    // Phase 8 — clear stale warnings
    app.pipeline_warnings.clear();
}

/// Refresh the input image texture to reflect current processor transformations
/// Call this after any flip/rotate operation to update the preview
pub fn refresh_input_preview(app: &mut PixelForgeApp, ctx: &egui::Context) {
    if let Some(ref mut input) = app.input_image {
        if let Some(ref processor) = app.image_processor {
            match processor.get_processed_image() {
                Ok(processed_img) => {
                    let texture = upload_texture(ctx, "input_image", &processed_img);
                    input.image = processed_img;
                    input.texture = texture;
                    ctx.request_repaint();
                }
                Err(e) => log::error!("Failed to refresh preview: {}", e),
            }
        }
    }
}

pub fn open_file_dialog(app: &mut PixelForgeApp) {
    // Phase 9 / Task 6 follow-up: use persisted `last_image_dir` as the
    // dialog's starting directory. Reads `app.config.directories` so the
    // `config` field is no longer dead code.
    let mut dialog = rfd::FileDialog::new()
        .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp"]);
    if let Some(dir) = &app.config.directories.last_image_dir {
        if dir.is_dir() {
            dialog = dialog.set_directory(dir);
        }
    }
    if let Some(path) = dialog.pick_file() {
        // Persist the directory for next time.
        if let Some(parent) = path.parent() {
            app.config.directories.last_image_dir = Some(parent.to_path_buf());
            let _ = app.config.save();
        }
        app.pending_file_load = Some(path);
    }
}

pub fn export_image(app: &PixelForgeApp) {
    let Some(output) = &app.output_image else { return };
    // Read persisted `last_export_dir` (Task 6 follow-up: makes `app.config`
    // actually used). Persistence-on-save would require `&mut self`; skipped
    // to avoid changing the function signature.
    let mut dialog = rfd::FileDialog::new()
        .add_filter("PNG",  &["png"])
        .add_filter("JPEG", &["jpg", "jpeg"]);
    if let Some(dir) = &app.config.directories.last_export_dir {
        if dir.is_dir() {
            dialog = dialog.set_directory(dir);
        }
    }
    if let Some(path) = dialog.save_file() {
        let cfg = ExportConfig { scale: app.transform_config.export_scale, ..Default::default() };
        match ImageExporter::export(&output.image, &path, &cfg) {
            Ok(_)  => log::info!("Exported to {}", path.display()),
            Err(e) => log::error!("Export failed: {}", e),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Contact-sheet export (Phase 1 — U1)
// ─────────────────────────────────────────────────────────────────────────────

/// Compose a single PNG showing: input → depth map → edge map → SLIC labels →
/// post-depth-to-flat → output. Each row labeled with source dimensions.
///
/// Missing intermediates (e.g. ML not yet run, or pipeline not yet processed)
/// render as a labeled placeholder row so the contact sheet is always usable.
pub fn export_contact_sheet(app: &PixelForgeApp) {
    let Some(input) = &app.input_image else {
        log::warn!("Contact sheet: no input image loaded");
        return;
    };

    let (in_w, in_h) = input.image.dimensions();

    // ── Build rows ───────────────────────────────────────────────────────────
    let mut rows: Vec<crate::image::ContactSheetRow<'_>> = Vec::with_capacity(8);

    // Row 1: Input image
    rows.push(crate::image::ContactSheetRow {
        label:    "INPUT",
        image:    Some(input.image.clone()),
        subtitle: Some(Box::leak(
            format!("{in_w}x{in_h}  source").into_boxed_str(),
        )),
        settings: None,
    });

    // Row 2: Depth map
    let depth_img = app.ml_results.as_ref()
        .and_then(|r| r.depth_map.as_deref())
        .map(|d| ml_depth_to_image(d, in_w, in_h));
    let depth_sub = if depth_img.is_some() {
        Some(Box::leak(format!("{in_w}x{in_h}  depth-anything v2").into_boxed_str()) as &str)
    } else {
        Some("not run — enable depth estimation and run ML analysis")
    };
    rows.push(crate::image::ContactSheetRow {
        label:    "DEPTH MAP",
        image:    depth_img,
        subtitle: depth_sub,
        settings: None,
    });

    // Row 3: Edge map
    let edge_img = app.ml_results.as_ref()
        .and_then(|r| r.edge_map.as_deref())
        .map(|e| ml_edge_to_image(e, in_w, in_h));
    let edge_sub = if edge_img.is_some() {
        Some(Box::leak(format!("{in_w}x{in_h}  dexined").into_boxed_str()) as &str)
    } else {
        Some("not run — enable edge detection and run ML analysis")
    };
    rows.push(crate::image::ContactSheetRow {
        label:    "EDGE MAP",
        image:    edge_img,
        subtitle: edge_sub,
        settings: None,
    });

    // Row 4: SLIC labels
    let slic_img = app.ml_results.as_ref()
        .and_then(|r| r.slic_labels.as_deref())
        .map(|l| ml_slic_to_image(l, in_w, in_h));
    let slic_sub: Option<&str> = if slic_img.is_some() {
        let n_clusters = {
            let mut s: Vec<u32> = app.ml_results.as_ref()
                .and_then(|r| r.slic_labels.as_ref())
                .map(|l| l.iter().copied().collect())
                .unwrap_or_default();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        Some(Box::leak(
            format!("{in_w}x{in_h}  k={}  {} regions", app.slic_config.k, n_clusters).into_boxed_str()
        ) as &str)
    } else {
        Some("not computed — run Process once")
    };
    rows.push(crate::image::ContactSheetRow {
        label:    "SLIC REGIONS",
        image:    slic_img,
        subtitle: slic_sub,
        settings: None,
    });

    // Row 4b: Segmentation mask (AnimeSegment)
    let seg_img = app.ml_results.as_ref()
        .and_then(|r| r.segmentation_mask.as_deref())
        .map(|m| ml_edge_to_image(m, in_w, in_h));  // reuse grayscale converter
    let seg_sub: Option<&str> = if seg_img.is_some() {
        Some(Box::leak(format!("{in_w}x{in_h}  anime-segment").into_boxed_str()) as &str)
    } else {
        Some("not run — enable segmentation and run ML analysis")
    };
    rows.push(crate::image::ContactSheetRow {
        label:    "SEGMENTATION",
        image:    seg_img,
        subtitle: seg_sub,
        settings: None,
    });

    // Row 5: Post depth-to-flat
    let flat_img = app.flat_color_image.as_ref().map(|i| i.clone());
    let flat_sub = if flat_img.is_some() {
        Some(Box::leak(
            format!("{in_w}x{in_h}  post depth-to-flat").into_boxed_str()
        ) as &str)
    } else {
        Some("run Process to populate")
    };
    rows.push(crate::image::ContactSheetRow {
        label:    "FLAT (POST-DTF)",
        image:    flat_img,
        subtitle: flat_sub,
        settings: None,
    });

    // Row 6: Output
    let out_img = app.output_image.as_ref().map(|o| o.image.clone());
    let out_sub = if let Some(o) = &app.output_image {
        let (ow, oh) = o.image.dimensions();
        Some(Box::leak(
            format!("{ow}x{oh}  {} colors", o.palette.len()).into_boxed_str()
        ) as &str)
    } else {
        Some("run Process to populate")
    };
    rows.push(crate::image::ContactSheetRow {
        label:    "OUTPUT",
        image:    out_img,
        subtitle: out_sub,
        settings: None,
    });

    // ── Row 7: Settings (ET-1) ──────────────────────────────────────────────
    // All active config values so the contact sheet is reproducible.
    let dtf = &app.depth_to_flat_config;
    let slc = &app.slic_config;
    let edg = &app.edge_config;
    let pal = &app.palette_config;
    let trn = &app.transform_config;
    let mlc = &app.ml_config;
    let src_name = input.path.as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "(untitled)".to_string());

    let settings_pairs: Vec<(String, String)> = vec![
        // Source
        ("SOURCE".into(), src_name),
        ("INPUT DIMS".into(), format!("{}x{}", in_w, in_h)),
        // Depth → Flat
        ("--- DEPTH->FLAT ---".into(), "".into()),
        ("strength".into(), format!("{:.2}", dtf.strength)),
        ("gamma".into(), format!("{:.2}", dtf.gamma)),
        ("mad_threshold".into(), format!("{:.3}", dtf.mad_threshold)),
        ("global_depth_weight".into(), format!("{:.2}", dtf.global_depth_weight)),
        ("use_otsu".into(), format!("{}", dtf.use_otsu_threshold)),
        ("bg_depth_threshold".into(), format!("{:.2}", dtf.bg_depth_threshold)),
        ("bg_desaturation".into(), format!("{:.2}", dtf.bg_desaturation)),
        ("bg_lightness_shift".into(), format!("{:.2}", dtf.bg_lightness_shift)),
        ("bg_cluster_size_pct".into(), format!("{:.2}", dtf.bg_cluster_size_pct)),
        // SLIC
        ("--- SLIC ---".into(), "".into()),
        ("k".into(), format!("{}", slc.k)),
        ("spatial_weight".into(), format!("{:.2}", slc.spatial_weight)),
        // Edges
        ("--- EDGES ---".into(), "".into()),
        ("edge_mode".into(), format!("{:?}", edg.edge_mode)),
        ("thickness".into(), format!("{}", edg.thickness)),
        ("edge_darkener_strength".into(), format!("{:.2}", edg.edge_darkener_strength)),
        ("anti_alias".into(), format!("{}", edg.anti_alias_edges)),
        ("outline_style".into(), format!("{:?}", edg.outline_style)),
        ("teed_threshold".into(), format!("{:.2}", edg.teed_threshold)),
        // Palette
        ("--- PALETTE ---".into(), "".into()),
        ("mode".into(), format!("{:?}", pal.mode)),
        ("max_colors".into(), format!("{}", pal.max_colors)),
        ("preset".into(), format!("{:?}", pal.preset)),
        // Transform / Output
        ("--- TRANSFORM ---".into(), "".into()),
        ("output_size".into(), format!("{}", trn.output_size)),
        ("downsampling_method".into(), format!("{:?}", trn.downsampling_method)),
        ("scale".into(), format!("{:.2}", trn.scale)),
        ("rotation".into(), format!("{:.1}", trn.rotation)),
        ("offset_x".into(), format!("{:.2}", trn.offset_x)),
        ("offset_y".into(), format!("{:.2}", trn.offset_y)),
        ("flip_h".into(), format!("{}", trn.flip_horizontal)),
        ("flip_v".into(), format!("{}", trn.flip_vertical)),
        ("export_scale".into(), format!("{}x", trn.export_scale)),
        // ML
        ("--- ML ---".into(), "".into()),
        ("depth_estimation".into(), format!("{}", mlc.depth_estimation_enabled)),
        ("edge_detection".into(), format!("{}", mlc.edge_detection_enabled)),
    ];
    rows.push(crate::image::ContactSheetRow {
        label:    "SETTINGS",
        image:    None,
        subtitle:  None,
        settings: Some(settings_pairs),
    });

    // ── Compose and save ────────────────────────────────────────────────────
    // ET-1: widened to 768px to fit longer setting names + values
    let row_width = 768;
    match crate::image::compose_contact_sheet(&rows, row_width) {
        Ok(sheet) => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .set_file_name("pixelforge_contact_sheet.png")
                .save_file()
            {
                match ImageExporter::export_png(&sheet, &path, None) {
                    Ok(_)  => log::info!("Contact sheet saved: {}", path.display()),
                    Err(e) => log::error!("Contact sheet save failed: {}", e),
                }
            }
        }
        Err(e) => log::error!("Contact sheet compose failed: {}", e),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML map → DynamicImage converters (for contact sheet + preview tab)
// ─────────────────────────────────────────────────────────────────────────────

/// Turbo colormap — perceptually uniform, better than grayscale for depth.
fn turbo_colormap(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let r = (0.13572138 + t*(4.61539260 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943))))) * 255.0;
    let g = (0.09140261 + t*(2.19418839 + t*(4.84296658  + t*(-14.18503333 + t*(  4.27729857  + t* 2.82956604))))) * 255.0;
    let b = (0.10667330 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))))) * 255.0;
    (r.clamp(0.0, 255.0) as u8, g.clamp(0.0, 255.0) as u8, b.clamp(0.0, 255.0) as u8)
}

/// Convert a depth map (`Vec<f32>`, [0,1], row-major) to a turbo-colored
/// RGBA image at the given dimensions.
pub fn ml_depth_to_image(depth: &[f32], w: u32, h: u32) -> image::DynamicImage {
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let v = depth.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let (r, g, b) = turbo_colormap(v);
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }
    image::DynamicImage::ImageRgba8(img)
}

/// Convert an edge map (`Vec<f32>`, [0,1], row-major) to a grayscale RGBA image.
pub fn ml_edge_to_image(edges: &[f32], w: u32, h: u32) -> image::DynamicImage {
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let v = (edges.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    image::DynamicImage::ImageRgba8(img)
}

/// Convert SLIC cluster labels to a colored RGBA image (one color per cluster).
pub fn ml_slic_to_image(labels: &[u32], w: u32, h: u32) -> image::DynamicImage {
    const PALETTE: [[u8; 3]; 16] = [
        [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
        [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
        [210, 245,  60], [250, 190, 190], [  0, 128, 128], [230, 190, 255],
        [170, 110,  40], [255, 250, 200], [128, 128, 128], [  0,   0,   0],
    ];
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let label = labels.get(idx).copied().unwrap_or(0);
            // BG_LABEL (u32::MAX) = background pixel (not clustered).
            // Render as dark gray to distinguish from foreground clusters.
            if label == crate::processing::slic::BG_LABEL {
                img.put_pixel(x, y, image::Rgba([30, 30, 30, 255]));
            } else {
                let c = PALETTE[(label as usize) % PALETTE.len()];
                img.put_pixel(x, y, image::Rgba([c[0], c[1], c[2], 255]));
            }
        }
    }
    image::DynamicImage::ImageRgba8(img)
}

// ─────────────────────────────────────────────────────────────────────────────
// Presets
// ─────────────────────────────────────────────────────────────────────────────

pub fn save_preset(app: &PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Preset", &["pixelforge"])
        .save_file()
    {
        let preset = crate::preset::Preset {
            name:          path.file_stem().and_then(|s| s.to_str()).unwrap_or("Custom").to_string(),
            transform:     app.transform_config.clone(),
            depth_to_flat: app.depth_to_flat_config.clone(),
            edges:         app.edge_config.clone(),
            palette:       app.palette_config.clone(),
        };
        match preset.save(&path) {
            Ok(_)  => log::info!("Preset saved: {}", path.display()),
            Err(e) => log::error!("Preset save failed: {}", e),
        }
    }
}

pub fn load_preset(app: &mut PixelForgeApp) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Preset", &["pixelforge"])
        .pick_file()
    {
        match crate::preset::Preset::load(&path) {
            Ok(preset) => {
                app.transform_config     = preset.transform;
                app.depth_to_flat_config = preset.depth_to_flat;
                app.edge_config          = preset.edges;
                app.palette_config       = preset.palette;
                app.current_preset       = Some(preset.name);
            }
            Err(e) => log::error!("Preset load failed: {}", e),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output dimensions
// ─────────────────────────────────────────────────────────────────────────────

pub fn get_output_dimensions(app: &PixelForgeApp) -> (u32, u32) {
    match app.aspect_mode {
        AspectRatioMode::Square => {
            let s = app.transform_config.output_size;
            (s, s)
        }
        AspectRatioMode::Preserve => {
            let Some(input) = &app.input_image else {
                let s = app.transform_config.output_size;
                return (s, s);
            };
            let (w, h) = input.image.dimensions();
            let target = app.transform_config.output_size as f32;
            let aspect = w as f32 / h as f32;
            if aspect >= 1.0 {
                let ow = target as u32;
                let oh = (target / aspect).round().max(8.0) as u32;
                (ow, oh)
            } else {
                let oh = target as u32;
                let ow = (target * aspect).round().max(8.0) as u32;
                (ow, oh)
            }
        }
        AspectRatioMode::Custom => {
            (app.custom_output_width.max(8), app.custom_output_height.max(8))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML Analysis
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_ml_analysis(app: &mut PixelForgeApp, ctx: &egui::Context) {
    let Some(_input) = &app.input_image else { return };
    let Some(ref processor) = app.image_processor else { return };

    // Use the processed image (with flips/rotations applied) from the ImageProcessor
    let image = match processor.get_processed_image() {
        Ok(img) => img,
        Err(e) => {
            log::error!("Failed to get processed image for ML: {}", e);
            app.processing = ProcessingState::Error(e.to_string());
            ctx.request_repaint();
            return;
        }
    };

    let ml_config = app.ml_config.clone();
    let mgr       = app.model_manager.clone();

    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.05,
        stage: "Running ML analysis…".into(),
    });
    ctx.request_repaint();

    // Spawn a background thread so the UI doesn't freeze during ML inference.
    // Results are sent back via an mpsc channel, polled in `update()`.
    let (tx, rx) = std::sync::mpsc::channel();
    app.ml_analysis_receiver = Some(rx);

    let ctx_clone = ctx.clone();
    std::thread::spawn(move || {
        let result = crate::ml::MLAnalysis::analyze(&image, &ml_config, &mgr);
        let _ = tx.send(result);
        // Wake the UI so it picks up the result immediately
        ctx_clone.request_repaint();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Processing pipeline
// ─────────────────────────────────────────────────────────────────────────────

pub fn process_image(app: &mut PixelForgeApp, ctx: &egui::Context) {
    if app.input_image.is_none() || app.image_processor.is_none() { return; }

    // Phase 8 — clear stale warnings from the previous run. New warnings
    // will be populated by `pipeline::run` and surfaced in the preview banner.
    app.pipeline_warnings.clear();

    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.05,
        stage: "Starting…".into(),
    });
    ctx.request_repaint();

    let (ow, oh) = get_output_dimensions(app);
    
    // Use the processed image (with flips/rotations applied) from the ImageProcessor
    let input = if let Some(ref processor) = app.image_processor {
        match processor.get_processed_image() {
            Ok(img) => img,
            Err(e) => {
                log::error!("Failed to get processed image: {}", e);
                app.processing = ProcessingState::Error(e.to_string());
                ctx.request_repaint();
                return;
            }
        }
    } else {
        app.input_image.as_ref().unwrap().image.clone()
    };

    // ── Compute SLIC superpixels (cached in ml_results) ───────────────────────
    // SLIC runs on the same processed image the ML analysis used. The label
    // map is cached in ml_results.slic_labels so palette/edge tweaks don't
    // trigger a re-cluster. Recompute when depth_map is available and either
    // the labels are missing or the config changed.
    if let Some(ref mut ml) = app.ml_results {
        let need_recompute = ml.slic_labels.is_none()
            || ml.slic_labels_k != Some(app.slic_config.k)
            || ml.slic_labels_s != Some(app.slic_config.spatial_weight);
        if need_recompute {
            // Phase 1 — P6: invalidate cached SLIC texture before recompute
            app.ml_slic_texture = None;
            // Pass segmentation mask to SLIC so it only clusters foreground
            // pixels (character area). Background gets BG_LABEL sentinel.
            let seg_mask = ml.segmentation_mask.as_deref();
            if let Some(ref depth) = ml.depth_map {
                match crate::processing::slic::slic(&input, Some(depth), seg_mask, &app.slic_config) {
                    Ok(labels) => {
                        ml.slic_labels = Some(labels);
                        ml.slic_labels_k = Some(app.slic_config.k);
                        ml.slic_labels_s = Some(app.slic_config.spatial_weight);
                    }
                    Err(e) => {
                        log::warn!("SLIC failed: {e}");
                        ml.slic_labels = None;
                    }
                }
            } else {
                // No depth map — still run SLIC with depth=0.5 fallback
                match crate::processing::slic::slic(&input, None, seg_mask, &app.slic_config) {
                    Ok(labels) => {
                        ml.slic_labels = Some(labels);
                        ml.slic_labels_k = Some(app.slic_config.k);
                        ml.slic_labels_s = Some(app.slic_config.spatial_weight);
                    }
                    Err(e) => {
                        log::warn!("SLIC (no depth) failed: {e}");
                        ml.slic_labels = None;
                    }
                }
            }
        }
    }

    let pipeline_input = PipelineInput {
        image:         &input,
        ml_results:    app.ml_results.as_ref(),
        transform:     &app.transform_config,
        depth_to_flat: &app.depth_to_flat_config,
        edges:         &app.edge_config,
        palette:       &app.palette_config,
        output_width:  ow,
        output_height: oh,
    };

    // ── Progress updates are reported before and after the pipeline.
    // For more granular in-pipeline updates, PipelineInput could carry
    // a progress callback, but that adds complexity for marginal UX gain
    // on a synchronous pipeline.
    app.processing = ProcessingState::Running(ProcessingStatus {
        progress: 0.10,
        stage: "Processing…".into(),
    });
    ctx.request_repaint();

    let output = crate::processing::pipeline::run(&pipeline_input);

    // Store intermediates for debugging
    app.preprocessed_image = output.preprocessed;
    app.flat_color_image   = output.flat;

    // Phase 8 — surface pipeline warnings to the UI.
    app.pipeline_warnings = output.warnings;

    // Upload final texture
    let texture = upload_texture(ctx, "output_image", &output.image);
    let palette_colors: Vec<egui::Color32> = output.palette_colors.iter()
        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
        .collect();

    app.output_image = Some(OutputImage {
        image:   output.image,
        texture,
        palette: palette_colors,
    });
    app.processing = ProcessingState::Complete;
    ctx.request_repaint();
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn upload_texture(ctx: &egui::Context, name: &str, img: &image::DynamicImage) -> egui::TextureHandle {
    let rgba = img.to_rgba8();
    let pixels: Vec<egui::Color32> = rgba.pixels()
        .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
        .collect();
    // NEAREST filtering is critical for pixel art — linear filtering would
    // interpolate between pixels, producing the "gradient" artifacts the user
    // reported in the preview. This applies to output AND intermediate previews.
    ctx.load_texture(
        name,
        ColorImage { size: [rgba.width() as _, rgba.height() as _], pixels },
        egui::TextureOptions::NEAREST,
    )
}
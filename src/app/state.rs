//! Application state

use crate::config::AppConfig;
use crate::ml::{MLConfig, MLResults};
use crate::models::ModelManager;
use crate::processing::{
    DepthToFlatConfig, EdgeConfig, PaletteConfig,
    ProcessingState, SlicConfig, TransformConfig,
};
use eframe::egui::{self, TextureHandle};
use parking_lot::RwLock;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Sub-types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AspectRatioMode {
    #[default]
    Square,
    Preserve,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewTab {
    #[default]
    Original,
    MLMaps,
    /// Post-depth-to-flat intermediate (shading applied, pre-transform)
    Flat,
    /// Post-transform, pre-downsample (what the downsample/palette stages see)
    Preprocessed,
    Output,
}

pub struct InputImage {
    pub image: image::DynamicImage,
    pub texture: TextureHandle,
    pub path: Option<std::path::PathBuf>,
}

pub struct OutputImage {
    pub image: image::DynamicImage,
    pub texture: TextureHandle,
    pub palette: Vec<egui::Color32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────────────────────

pub struct PixelForgeApp {
    // ── Core ─────────────────────────────────────────────────────────────────
    pub config: AppConfig,
    pub model_manager: Arc<RwLock<ModelManager>>,

    // ── Images ───────────────────────────────────────────────────────────────
    pub input_image: Option<InputImage>,
    /// Image processor for tracking transforms (flip/rotate/resize)
    pub image_processor: Option<crate::image::ImageProcessor>,
    /// Post-transform, pre-downsample (for debugging)
    pub preprocessed_image: Option<image::DynamicImage>,
    /// Post depth-to-flat intermediate
    pub flat_color_image: Option<image::DynamicImage>,
    pub output_image: Option<OutputImage>,

    // ── ML ───────────────────────────────────────────────────────────────────
    pub ml_results: Option<MLResults>,

    // ── Pipeline state ───────────────────────────────────────────────────────
    pub processing: ProcessingState,

    /// Phase 8 — Warnings collected by the last pipeline run. Each silent
    /// fallback (depth_to_flat failure, palette empty, edge render failure,
    /// EdgeMode::Internal SLIC fallback, etc.) pushes a human-readable
    /// message here. The UI surfaces them in a dismissible yellow banner
    /// below the preview tabs. Cleared at the start of each `process_image`.
    pub pipeline_warnings: Vec<String>,

    // ── Configs ──────────────────────────────────────────────────────────────
    pub ml_config: MLConfig,
    pub transform_config: TransformConfig,
    pub depth_to_flat_config: DepthToFlatConfig,
    pub edge_config: EdgeConfig,
    pub palette_config: PaletteConfig,
    pub slic_config: SlicConfig,

    // ── Output sizing ─────────────────────────────────────────────────────────
    pub aspect_mode: AspectRatioMode,
    pub custom_output_width: u32,
    pub custom_output_height: u32,

    // ── Preview ──────────────────────────────────────────────────────────────
    pub preview_tab: PreviewTab,
    pub preview_zoom: f32,
    /// ML map texture cache (Phase 1 — P6). Built lazily when the ML Maps
    /// tab is opened; invalidated when `ml_results` changes. Prevents the
    /// per-frame rebuild that previously happened in `app/preview.rs`.
    pub ml_depth_texture: Option<TextureHandle>,
    pub ml_edge_texture:  Option<TextureHandle>,
    pub ml_slic_texture:  Option<TextureHandle>,
    /// 1:1 zoom toggle for the ML Maps tab (Phase 1 — C4). When true, maps
    /// display at native resolution (1 source pixel = 1 screen pixel) and
    /// the user can scroll/pan inside the scroll area.
    pub ml_maps_native: bool,

    // ── Presets ──────────────────────────────────────────────────────────────
    pub current_preset: Option<String>,

    // ── Dialogs ──────────────────────────────────────────────────────────────
    pub show_model_dialog: bool,
    pub show_about_dialog: bool,
    pub show_token_dialog: bool,
    pub token_input: String,

    // ── Auth ─────────────────────────────────────────────────────────────────
    pub huggingface_token: String,

    // ── System info ──────────────────────────────────────────────────────────
    pub vram_usage: u64,
    pub total_vram: u64,

    // ── Async helpers ────────────────────────────────────────────────────────
    pub pending_file_load: Option<std::path::PathBuf>,

    /// Receiver for background ML analysis results. `Some` while a background
    /// ML thread is running; polled each frame in `update()`.
    pub ml_analysis_receiver: Option<std::sync::mpsc::Receiver<anyhow::Result<crate::ml::MLResults>>>,
}

impl PixelForgeApp {
    pub fn new(cc: &eframe::CreationContext<'_>, config: AppConfig) -> Self {
        super::theme::apply_theme(&cc.egui_ctx, &config.ui.theme);

        let model_manager = Arc::new(RwLock::new(
            ModelManager::new().unwrap_or_default(),
        ));

        // Apply persisted config defaults to the runtime configs so user
        // preferences survive across app restarts. Previously the `config`
        // field was stored but never read after construction (dead-code
        // warning). Now we apply:
        //  - `processing.default_output_size` → transform_config.output_size
        //  - `processing.default_edge_thickness` → edge_config.thickness
        //  - `ui.default_zoom` → preview_zoom
        // The `config` field itself is retained so future code can read
        // other persisted prefs (e.g. directories, model_quality) and so
        // `AppConfig::save()` can be called on exit to persist changes.
        let mut transform_config = TransformConfig::default();
        transform_config.output_size = config.processing.default_output_size.max(16);

        let mut edge_config = EdgeConfig::default();
        edge_config.thickness = config.processing.default_edge_thickness.clamp(1, 4);

        let preview_zoom = config.ui.default_zoom.max(0.25);

        Self {
            config,
            model_manager,
            input_image: None,
            image_processor: None,
            preprocessed_image: None,
            flat_color_image: None,
            output_image: None,
            ml_results: None,
            processing: ProcessingState::Idle,
            pipeline_warnings: Vec::new(),
            ml_config: MLConfig::default(),
            transform_config,
            depth_to_flat_config: DepthToFlatConfig::default(),
            edge_config,
            palette_config: PaletteConfig::default(),
            slic_config: SlicConfig::default(),
            aspect_mode: AspectRatioMode::Square,
            custom_output_width: 64,
            custom_output_height: 64,
            preview_tab: PreviewTab::Original,
            preview_zoom,
            ml_depth_texture: None,
            ml_edge_texture: None,
            ml_slic_texture: None,
            ml_maps_native: false,
            current_preset: None,
            show_model_dialog: false,
            show_about_dialog: false,
            show_token_dialog: false,
            token_input: String::new(),
            huggingface_token: String::new(),
            vram_usage: 0,
            total_vram: 4096,
            pending_file_load: None,
            ml_analysis_receiver: None,
        }
    }
}

impl eframe::App for PixelForgeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(path) = self.pending_file_load.take() {
            super::processing::load_image(self, &path, ctx);
        }

        ctx.input(|i| {
            if let Some(file) = i.raw.dropped_files.first() {
                if let Some(path) = &file.path {
                    self.pending_file_load = Some(path.clone());
                }
            }
        });

        if matches!(self.processing, ProcessingState::Running(_)) {
            ctx.request_repaint();
        }

        // ── Poll background ML analysis ───────────────────────────────────────
        if let Some(rx) = self.ml_analysis_receiver.take() {
            match rx.try_recv() {
                Ok(Ok(results)) => {
                    self.vram_usage = self.model_manager.read().estimated_vram_usage() / (1024 * 1024);
                    self.ml_results = Some(results);
                    // Phase 1 — P6: invalidate cached ML map textures so they
                    // get rebuilt from the new data on next preview frame.
                    self.ml_depth_texture = None;
                    self.ml_edge_texture  = None;
                    self.ml_slic_texture  = None;
                    self.processing = ProcessingState::Complete;
                }
                Ok(Err(e)) => {
                    log::error!("ML analysis failed: {}", e);
                    self.processing = ProcessingState::Error(e.to_string());
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still running — put the receiver back and request a repaint
                    self.ml_analysis_receiver = Some(rx);
                    ctx.request_repaint();
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    log::error!("ML analysis thread disconnected (panicked?)");
                    self.processing = ProcessingState::Error("ML analysis thread crashed".into());
                }
            }
        }

        super::menu::draw(self, ctx);
        super::panels::draw_left(self, ctx);
        super::panels::draw_right(self, ctx);
        super::panels::draw_bottom(self, ctx);
        super::panels::draw_status(self, ctx);
        super::preview::draw(self, ctx);

        if self.show_model_dialog { super::dialogs::model_dialog(self, ctx); }
        if self.show_about_dialog { super::dialogs::about_dialog(self, ctx); }
        if self.show_token_dialog { super::dialogs::token_dialog(self, ctx); }
    }
}

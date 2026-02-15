//! Application state and types

use crate::config::AppConfig;
use crate::processing::{
    DepthToFlatConfig, EdgeConfig, FeaturePreserveConfig, PaletteConfig,
    ProcessingState, TransformConfig,
};
use crate::ml::{MLConfig, MLResults};
use crate::models::ModelManager;
use eframe::egui;
use egui::TextureHandle;
use parking_lot::RwLock;
use std::sync::Arc;

/// Output aspect ratio mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AspectRatioMode {
    #[default]
    Square,
    Preserve,
    Custom,
}

/// Preview tab
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewTab {
    #[default]
    Original,
    MLAnalysis,
    Output,
}

/// Wrapper for input image with texture
pub struct InputImage {
    pub image: image::DynamicImage,
    pub texture: TextureHandle,
    pub path: Option<std::path::PathBuf>,
}

/// Wrapper for output image with texture
pub struct OutputImage {
    pub image: image::DynamicImage,
    pub texture: TextureHandle,
    pub palette: Vec<egui::Color32>,
}

/// Main application state
pub struct PixelForgeApp {
    /// Application configuration
    pub config: AppConfig,

    /// Model manager
    pub model_manager: Arc<RwLock<ModelManager>>,

    /// Currently loaded image
    pub input_image: Option<InputImage>,

    /// Intermediate processing results
    pub preprocessed_image: Option<image::DynamicImage>,
    pub flat_color_image: Option<image::DynamicImage>,

    /// ML analysis results
    pub ml_results: Option<MLResults>,

    /// Processing output
    pub output_image: Option<OutputImage>,

    /// Processing state
    pub processing: ProcessingState,

    /// Control configurations
    pub ml_config: MLConfig,
    pub transform_config: TransformConfig,
    pub depth_to_flat_config: DepthToFlatConfig,
    pub feature_config: FeaturePreserveConfig,
    pub edge_config: EdgeConfig,
    pub palette_config: PaletteConfig,

    /// Aspect ratio mode
    pub aspect_mode: AspectRatioMode,
    pub custom_output_width: u32,
    pub custom_output_height: u32,

    /// Preview toggles
    pub show_landmarks: bool,
    pub show_depth_heatmap: bool,
    pub show_segmentation: bool,

    /// Preview tab
    pub preview_tab: PreviewTab,

    /// Currently loaded preset name
    pub current_preset: Option<String>,

    /// System info
    pub vram_usage: u64,
    pub total_vram: u64,
    
    /// Preview zoom
    pub preview_zoom: f32,

    /// UI state
    pub show_model_dialog: bool,
    pub show_about_dialog: bool,

    /// File to load
    pub pending_file_load: Option<std::path::PathBuf>,

    pub show_token_dialog: bool,
    pub huggingface_token: String,
    pub token_input: String,
}

impl PixelForgeApp {
    /// Create a new application instance
    pub fn new(cc: &eframe::CreationContext<'_>, config: AppConfig) -> Self {
        super::theme::apply_theme(&cc.egui_ctx, &config.ui.theme);

        let model_manager = Arc::new(RwLock::new(
            ModelManager::new().unwrap_or_default()
        ));

        {
            let mut mgr = model_manager.write();
            let _ = mgr.set_quality(config.model_quality);
            mgr.set_sequential_mode(config.sequential_processing);
        }

        Self {
            config,
            model_manager,
            input_image: None,
            preprocessed_image: None,
            flat_color_image: None,
            ml_results: None,
            output_image: None,
            processing: ProcessingState::Idle,
            ml_config: MLConfig::default(),
            transform_config: TransformConfig::default(),
            depth_to_flat_config: DepthToFlatConfig::default(),
            feature_config: FeaturePreserveConfig::default(),
            edge_config: EdgeConfig::default(),
            palette_config: PaletteConfig::default(),
            aspect_mode: AspectRatioMode::Square,
            custom_output_width: 64,
            custom_output_height: 64,
            show_landmarks: true,
            show_depth_heatmap: false,
            show_segmentation: false,
            preview_tab: PreviewTab::Original,
            current_preset: None,
            vram_usage: 0,
            total_vram: 4096,
            preview_zoom: 1.0,
            show_model_dialog: false,
            show_about_dialog: false,
            pending_file_load: None,
            show_token_dialog: false,
            huggingface_token: String::new(),
            token_input: String::new(),
        }
    }
}

impl eframe::App for PixelForgeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle pending file load
        if let Some(path) = self.pending_file_load.take() {
            super::processing::load_image(self, &path, ctx);
        }

        // Handle drag and drop
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(file) = i.raw.dropped_files.first() {
                    if let Some(path) = &file.path {
                        super::processing::load_image(self, path, ctx);
                    }
                }
            }
        });

        // Request repaint while processing
        if matches!(self.processing, ProcessingState::Running(_)) {
            ctx.request_repaint();
        }

        // Draw UI
        super::menu::draw(self, ctx);
        super::panels::draw_left(self, ctx);
        super::panels::draw_right(self, ctx);
        super::panels::draw_bottom(self, ctx);
        super::panels::draw_status(self, ctx);
        super::preview::draw(self, ctx);

        // Dialogs
        if self.show_model_dialog {
            super::dialogs::model_dialog(self, ctx);
        }
        if self.show_about_dialog {
            super::dialogs::about_dialog(self, ctx);
        }
        if self.show_token_dialog {
            super::dialogs::token_dialog(self, ctx);
        }
    }
}

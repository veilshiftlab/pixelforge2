//! Main application state and logic

use crate::config::{AppConfig, ModelQuality};
use crate::processing::{
    DepthToFlatConfig, EdgeConfig, FeaturePreserveConfig, PaletteConfig,
    ProcessingState, ProcessingStatus, TransformConfig,
};
use crate::ml::{MLAnalysis, MLConfig, MLResults};
use crate::models::ModelManager;
use crate::preset::Preset;
use eframe::egui;
use egui::{ColorImage, TextureHandle};
use image::GenericImageView;
use parking_lot::RwLock;
use std::sync::Arc;

/// Main application state
pub struct PixelForgeApp {
    /// Application configuration
    config: AppConfig,

    /// Model manager
    model_manager: Arc<RwLock<ModelManager>>,

    /// Currently loaded image
    input_image: Option<InputImage>,

    /// ML analysis results
    ml_results: Option<MLResults>,

    /// Processing output
    output_image: Option<OutputImage>,

    /// Processing state
    processing: ProcessingState,

    /// Control configurations
    ml_config: MLConfig,
    transform_config: TransformConfig,
    depth_to_flat_config: DepthToFlatConfig,
    feature_config: FeaturePreserveConfig,
    edge_config: EdgeConfig,
    palette_config: PaletteConfig,

    /// Preview toggles
    show_landmarks: bool,
    show_depth_heatmap: bool,
    show_segmentation: bool,

    /// Currently loaded preset name
    current_preset: Option<String>,

    /// System info
    vram_usage: u64,
    total_vram: u64,
    
    /// Preview scale for output
    preview_zoom: f32,
}

/// Wrapper for input image with texture
struct InputImage {
    /// Original image data
    image: image::DynamicImage,

    /// GPU texture for display
    texture: TextureHandle,

    /// File path if loaded from disk
    path: Option<std::path::PathBuf>,
}

/// Wrapper for output image with texture
struct OutputImage {
    /// Processed pixel art image
    image: image::DynamicImage,

    /// GPU texture for display
    texture: TextureHandle,

    /// Palette used
    palette: Vec<egui::Color32>,
}

impl PixelForgeApp {
    /// Create a new application instance
    pub fn new(cc: &eframe::CreationContext<'_>, config: AppConfig) -> Self {
        // Apply theme
        apply_theme(&cc.egui_ctx, &config.ui.theme);

        // Initialize model manager
        let model_manager = Arc::new(RwLock::new(
            ModelManager::new().unwrap_or_default()
        ));

        Self {
            config,
            model_manager,
            input_image: None,
            ml_results: None,
            output_image: None,
            processing: ProcessingState::Idle,
            ml_config: MLConfig::default(),
            transform_config: TransformConfig::default(),
            depth_to_flat_config: DepthToFlatConfig::default(),
            feature_config: FeaturePreserveConfig::default(),
            edge_config: EdgeConfig::default(),
            palette_config: PaletteConfig::default(),
            show_landmarks: true,
            show_depth_heatmap: false,
            show_segmentation: false,
            current_preset: None,
            vram_usage: 0,
            total_vram: 4096,
            preview_zoom: 2.0,
        }
    }

    /// Load an image from a file
    fn load_image(&mut self, path: &std::path::Path, ctx: &egui::Context) {
        match image::open(path) {
            Ok(img) => {
                // Create texture
                let rgba = img.to_rgba8();
                let size = [rgba.width() as _, rgba.height() as _];
                let pixels: Vec<egui::Color32> = rgba
                    .pixels()
                    .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
                    .collect();

                let color_image = ColorImage { size, pixels };
                let texture = ctx.load_texture("input_image", color_image, egui::TextureOptions::default());

                self.input_image = Some(InputImage {
                    image: img,
                    texture,
                    path: Some(path.to_path_buf()),
                });

                // Clear previous results
                self.ml_results = None;
                self.output_image = None;

                log::info!("Loaded image: {}", path.display());
            }
            Err(e) => {
                log::error!("Failed to load image: {}", e);
            }
        }
    }

    /// Run ML analysis on the current image
    fn run_ml_analysis(&mut self) {
        if self.input_image.is_none() {
            return;
        }

        let image = self.input_image.as_ref().unwrap().image.clone();
        let model_manager = self.model_manager.clone();
        let ml_config = self.ml_config.clone();

        // Start processing
        self.processing = ProcessingState::Running(ProcessingStatus {
            progress: 0.0,
            stage: "Starting ML analysis...".to_string(),
        });

        // Run analysis (synchronous for now)
        match MLAnalysis::analyze(&image, &ml_config, &model_manager) {
            Ok(results) => {
                self.ml_results = Some(results);
                self.processing = ProcessingState::Complete;
                
                // Update VRAM usage
                let manager = self.model_manager.read();
                self.vram_usage = manager.estimated_vram_usage() / (1024 * 1024);
            }
            Err(e) => {
                log::error!("ML analysis failed: {}", e);
                self.processing = ProcessingState::Error(e.to_string());
            }
        }
    }

    /// Process the image to pixel art
    fn process_image(&mut self, ctx: &egui::Context) {
        if self.input_image.is_none() {
            return;
        }

        self.processing = ProcessingState::Running(ProcessingStatus {
            progress: 0.0,
            stage: "Processing...".to_string(),
        });

        // Create output texture
        self.create_output_texture(ctx);
    }

    fn create_output_texture(&mut self, ctx: &egui::Context) {
        if self.input_image.is_none() {
            return;
        }

        let input = &self.input_image.as_ref().unwrap().image;
        let output_size = self.transform_config.output_size;

        // Create a simple downsampled version
        let resized = input.resize(output_size, output_size, image::imageops::FilterType::Nearest);
        let rgba = resized.to_rgba8();
        let size = [rgba.width() as _, rgba.height() as _];
        let pixels: Vec<egui::Color32> = rgba
            .pixels()
            .map(|p| egui::Color32::from_rgba_unmultiplied(p.0[0], p.0[1], p.0[2], p.0[3]))
            .collect();

        let color_image = ColorImage { size, pixels };
        let texture = ctx.load_texture("output_image", color_image, egui::TextureOptions::default());

        self.output_image = Some(OutputImage {
            image: resized,
            texture,
            palette: vec![
                egui::Color32::from_rgb(0, 0, 0),
                egui::Color32::from_rgb(85, 85, 85),
                egui::Color32::from_rgb(170, 170, 170),
                egui::Color32::from_rgb(255, 255, 255),
            ],
        });

        self.processing = ProcessingState::Complete;
    }

    /// Export the output image
    fn export_image(&self) {
        if let Some(output) = &self.output_image {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .add_filter("JPEG", &["jpg", "jpeg"])
                .save_file()
            {
                match output.image.save(&path) {
                    Ok(_) => log::info!("Exported to: {}", path.display()),
                    Err(e) => log::error!("Export failed: {}", e),
                }
            }
        }
    }

    /// Save current settings as a preset
    fn save_preset(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Preset", &["pixelforge"])
            .save_file()
        {
            let preset = Preset {
                name: path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Custom")
                    .to_string(),
                transform: self.transform_config.clone(),
                depth_to_flat: self.depth_to_flat_config.clone(),
                features: self.feature_config.clone(),
                edges: self.edge_config.clone(),
                palette: self.palette_config.clone(),
            };

            match preset.save(&path) {
                Ok(_) => log::info!("Saved preset: {}", path.display()),
                Err(e) => log::error!("Failed to save preset: {}", e),
            }
        }
    }

    /// Load a preset
    fn load_preset(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Preset", &["pixelforge"])
            .pick_file()
        {
            match Preset::load(&path) {
                Ok(preset) => {
                    self.transform_config = preset.transform;
                    self.depth_to_flat_config = preset.depth_to_flat;
                    self.feature_config = preset.features;
                    self.edge_config = preset.edges;
                    self.palette_config = preset.palette;
                    self.current_preset = Some(preset.name);
                    log::info!("Loaded preset: {}", path.display());
                }
                Err(e) => log::error!("Failed to load preset: {}", e),
            }
        }
    }
}

impl eframe::App for PixelForgeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle drag and drop
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(file) = i.raw.dropped_files.first() {
                    if let Some(path) = &file.path {
                        self.load_image(path, ctx);
                    }
                }
            }
        });

        // Draw UI
        self.menu_bar(ctx);
        self.left_panel(ctx);
        self.right_panel(ctx);
        self.bottom_panel(ctx);
        self.status_bar(ctx);
        self.central_panel(ctx);
    }
}

// ============================================================================
// UI PANEL IMPLEMENTATIONS
// ============================================================================

impl PixelForgeApp {
    /// Draw the menu bar
    pub fn menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Image...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["png", "jpg", "jpeg", "webp"])
                            .pick_file()
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
                        self.transform_config = TransformConfig::default();
                        self.depth_to_flat_config = DepthToFlatConfig::default();
                        self.feature_config = FeaturePreserveConfig::default();
                        self.edge_config = EdgeConfig::default();
                        self.palette_config = PaletteConfig::default();
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_landmarks, "Show Landmarks");
                    ui.checkbox(&mut self.show_depth_heatmap, "Show Depth Heatmap");
                    ui.checkbox(&mut self.show_segmentation, "Show Segmentation");
                    ui.separator();
                    ui.label(format!("Preview Zoom: {:.1}x", self.preview_zoom));
                    ui.add(egui::Slider::new(&mut self.preview_zoom, 1.0..=8.0).text(""));
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

                    if ui.button("Portrait - Minimal").clicked() {
                        self.apply_minimal_preset();
                        ui.close_menu();
                    }
                    if ui.button("Portrait - Detailed").clicked() {
                        self.apply_detailed_preset();
                        ui.close_menu();
                    }
                    if ui.button("Game Boy Style").clicked() {
                        self.apply_gameboy_preset();
                        ui.close_menu();
                    }
                });

                ui.menu_button("Settings", |ui| {
                    ui.label("Model Quality:");
                    if ui.radio_value(&mut self.config.model_quality, ModelQuality::Minimal, "Minimal (Fast)").clicked() {
                        self.update_model_quality();
                        ui.close_menu();
                    }
                    if ui.radio_value(&mut self.config.model_quality, ModelQuality::Standard, "Standard").clicked() {
                        self.update_model_quality();
                        ui.close_menu();
                    }
                    if ui.radio_value(&mut self.config.model_quality, ModelQuality::High, "High Quality").clicked() {
                        self.update_model_quality();
                        ui.close_menu();
                    }
                    if ui.radio_value(&mut self.config.model_quality, ModelQuality::Sequential, "Sequential (Low VRAM)").clicked() {
                        self.update_model_quality();
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("About PixelForge").clicked() {
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn apply_minimal_preset(&mut self) {
        self.transform_config = TransformConfig {
            output_size: 32,
            ..Default::default()
        };
        self.depth_to_flat_config = DepthToFlatConfig {
            skin_tone_bands: 3,
            hair_bands: 2,
            clothing_bands: 2,
            background_bands: 1,
            ..Default::default()
        };
        self.feature_config = FeaturePreserveConfig {
            eye_size: crate::processing::EyeSize::Small,
            eye_detail: crate::processing::DetailLevel::Minimal,
            ..Default::default()
        };
        self.palette_config = PaletteConfig {
            max_colors: 12,
            ..Default::default()
        };
        self.current_preset = Some("Portrait - Minimal".to_string());
    }

    fn apply_detailed_preset(&mut self) {
        self.transform_config = TransformConfig {
            output_size: 64,
            ..Default::default()
        };
        self.depth_to_flat_config = DepthToFlatConfig {
            skin_tone_bands: 5,
            hair_bands: 4,
            clothing_bands: 3,
            background_bands: 2,
            ..Default::default()
        };
        self.feature_config = FeaturePreserveConfig {
            eye_size: crate::processing::EyeSize::Medium,
            eye_detail: crate::processing::DetailLevel::Standard,
            lip_detail: crate::processing::DetailLevel::Standard,
            nose_detail: crate::processing::DetailLevel::Standard,
            distinct_nostrils: true,
            ..Default::default()
        };
        self.palette_config = PaletteConfig {
            max_colors: 32,
            ..Default::default()
        };
        self.current_preset = Some("Portrait - Detailed".to_string());
    }

    fn apply_gameboy_preset(&mut self) {
        self.transform_config = TransformConfig {
            output_size: 48,
            export_scale: 2,
            ..Default::default()
        };
        self.depth_to_flat_config = DepthToFlatConfig {
            skin_tone_bands: 2,
            hair_bands: 2,
            clothing_bands: 2,
            background_bands: 1,
            ..Default::default()
        };
        self.palette_config = PaletteConfig {
            mode: crate::processing::PaletteMode::Preset,
            preset: crate::processing::PresetPalette::GameBoy,
            max_colors: 4,
            ..Default::default()
        };
        self.current_preset = Some("Game Boy Style".to_string());
    }

    fn update_model_quality(&mut self) {
        let mut manager = self.model_manager.write();
        if let Err(e) = manager.set_quality(self.config.model_quality) {
            log::error!("Failed to update model quality: {}", e);
        }
    }

    /// Draw the left controls panel
    pub fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls_panel")
            .default_width(280.0)
            .min_width(240.0)
            .max_width(360.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;

                    ui.collapsing("ML Analysis", |ui| {
                        self.ml_analysis_controls(ui);
                    });

                    ui.add_space(8.0);

                    ui.collapsing("Depth-to-Flat", |ui| {
                        self.depth_to_flat_controls(ui);
                    });

                    ui.add_space(8.0);

                    ui.collapsing("Feature Preservation", |ui| {
                        self.feature_preserve_controls(ui);
                    });

                    ui.add_space(8.0);

                    ui.collapsing("Edge Controls", |ui| {
                        self.edge_controls(ui);
                    });
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
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Output Settings");
                    ui.separator();

                    ui.label("Output Size:");
                    egui::ComboBox::from_id_salt("output_size")
                        .selected_text(format!("{} px", self.transform_config.output_size))
                        .show_ui(ui, |ui| {
                            let sizes = [16, 24, 32, 48, 64, 96, 128, 192, 256];
                            for size in sizes {
                                ui.selectable_value(
                                    &mut self.transform_config.output_size,
                                    size,
                                    format!("{} px", size),
                                );
                            }
                        });

                    ui.add_space(8.0);

                    ui.label("Input Scale:");
                    ui.add(egui::Slider::new(&mut self.transform_config.scale, 0.1..=4.0).text("x").step_by(0.1));

                    ui.label("Rotation:");
                    ui.add(egui::Slider::new(&mut self.transform_config.rotation, -180.0..=180.0).text("°").step_by(1.0));

                    ui.label("Offset X:");
                    ui.add(egui::Slider::new(&mut self.transform_config.offset_x, -1.0..=1.0).text("").step_by(0.05));

                    ui.label("Offset Y:");
                    ui.add(egui::Slider::new(&mut self.transform_config.offset_y, -1.0..=1.0).text("").step_by(0.05));

                    ui.checkbox(&mut self.transform_config.clip_to_face, "Clip to face region");
                    if self.transform_config.clip_to_face {
                        ui.add(egui::Slider::new(&mut self.transform_config.clip_padding, 0.0..=0.5).text("Padding"));
                    }

                    ui.separator();

                    ui.label("Export Format:");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Auto, "PNG");
                        ui.selectable_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Preset, "JPG");
                    });

                    ui.add_space(8.0);

                    ui.label("Export Scale:");
                    ui.horizontal(|ui| {
                        let scales = [1, 2, 4, 8];
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

                    let export_enabled = self.output_image.is_some();
                    if ui.add_enabled(export_enabled, egui::Button::new("Export Image...")).clicked() {
                        self.export_image();
                    }

                    if ui.button("Batch Process...").clicked() {
                    }

                    ui.separator();

                    ui.heading("Presets");
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            self.save_preset();
                        }
                        if ui.button("Load").clicked() {
                            self.load_preset();
                        }
                    });
                    
                    if let Some(ref preset_name) = self.current_preset {
                        ui.label(format!("Current: {}", preset_name));
                    }

                    ui.separator();

                    ui.heading("Palette");
                    ui.label("Mode:");
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Auto, "Auto");
                        ui.selectable_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Preset, "Preset");
                        ui.selectable_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Custom, "Custom");
                        ui.selectable_value(&mut self.palette_config.mode, crate::processing::PaletteMode::Hybrid, "Hybrid");
                    });

                    if matches!(self.palette_config.mode, crate::processing::PaletteMode::Auto) {
                        ui.label("Max Colors:");
                        egui::ComboBox::from_id_salt("max_colors")
                            .selected_text(format!("{}", self.palette_config.max_colors))
                            .show_ui(ui, |ui| {
                                for count in [4, 8, 12, 16, 24, 32, 48, 64, 128, 256] {
                                    ui.selectable_value(&mut self.palette_config.max_colors, count, format!("{}", count));
                                }
                            });

                        ui.checkbox(&mut self.palette_config.per_region_limit, "Per-region limit");
                    }

                    if matches!(self.palette_config.mode, crate::processing::PaletteMode::Preset) {
                        ui.label("Preset:");
                        egui::ComboBox::from_id_salt("palette_preset")
                            .selected_text(format!("{:?}", self.palette_config.preset))
                            .show_ui(ui, |ui| {
                                use crate::processing::PresetPalette;
                                ui.selectable_value(&mut self.palette_config.preset, PresetPalette::GameBoy, "Game Boy");
                                ui.selectable_value(&mut self.palette_config.preset, PresetPalette::PICO8, "PICO-8");
                                ui.selectable_value(&mut self.palette_config.preset, PresetPalette::NES, "NES");
                                ui.selectable_value(&mut self.palette_config.preset, PresetPalette::DawnBringer32, "DawnBringer32");
                                ui.selectable_value(&mut self.palette_config.preset, PresetPalette::AAP64, "AAP-64");
                            });
                    }

                    if let Some(output) = &self.output_image {
                        ui.add_space(8.0);
                        ui.label("Generated Palette:");
                        ui.horizontal_wrapped(|ui| {
                            for &color in &output.palette {
                                let (r, g, b) = (color.r(), color.g(), color.b());
                                ui.add(egui::Label::new(
                                    egui::RichText::new("██").color(egui::Color32::from_rgb(r, g, b))
                                ));
                            }
                        });
                    }
                });
            });
    }

    /// Draw the bottom processing panel
    pub fn bottom_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("processing_panel")
            .default_height(60.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let progress = match &self.processing {
                        ProcessingState::Idle => 0.0,
                        ProcessingState::Running(status) => status.progress,
                        ProcessingState::Complete => 1.0,
                        ProcessingState::Error(_) => 0.0,
                    };

                    let status_text = match &self.processing {
                        ProcessingState::Idle => "Ready".to_string(),
                        ProcessingState::Running(status) => status.stage.clone(),
                        ProcessingState::Complete => "Complete!".to_string(),
                        ProcessingState::Error(e) => format!("Error: {}", e),
                    };

                    ui.vertical(|ui| {
                        ui.label(&status_text);
                        ui.add(egui::ProgressBar::new(progress).text(format!("{:.0}%", progress * 100.0)));
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let process_enabled = self.input_image.is_some() && 
                            !matches!(self.processing, ProcessingState::Running(_));

                        if ui.add_enabled(process_enabled, egui::Button::new("Process")).clicked() {
                            self.process_image(ctx);
                        }

                        if matches!(self.processing, ProcessingState::Running(_)) {
                            if ui.button("Cancel").clicked() {
                                self.processing = ProcessingState::Idle;
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
                    let status = match &self.processing {
                        ProcessingState::Idle => "Ready",
                        ProcessingState::Running(_) => "Processing...",
                        ProcessingState::Complete => "Complete",
                        ProcessingState::Error(_) => "Error",
                    };
                    ui.label(status);

                    ui.separator();

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

                    ui.label(format!("Output: {}x{}", self.transform_config.output_size, self.transform_config.output_size));

                    ui.separator();

                    let vram_percent = if self.total_vram > 0 {
                        (self.vram_usage as f64 / self.total_vram as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    ui.label(format!("VRAM: {}MB / {}MB ({}%)", 
                        self.vram_usage, self.total_vram, vram_percent));

                    ui.separator();
                    let quality_text = match self.config.model_quality {
                        ModelQuality::Minimal => "Minimal",
                        ModelQuality::Standard => "Standard",
                        ModelQuality::High => "High",
                        ModelQuality::Sequential => "Sequential",
                    };
                    ui.label(format!("Model: {}", quality_text));
                });
            });
    }

    /// Draw the central preview panel
    pub fn central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Original");
                    ui.separator();

                    if let Some(input) = &self.input_image {
                        ui.image(&input.texture);
                        let (w, h) = input.image.dimensions();
                        ui.label(format!("{}x{} pixels", w, h));
                    } else {
                        let available = ui.available_size();
                        let (_id, rect) = ui.allocate_space(available);

                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
                            ui.vertical_centered(|ui| {
                                ui.add_space(40.0);
                                ui.label(egui::RichText::new("📷").size(48.0));
                                ui.add_space(10.0);
                                ui.label(egui::RichText::new("Drop image here").size(16.0));
                                ui.label(egui::RichText::new("or").size(12.0).weak());
                                if ui.button("Browse...").clicked() {
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter("Image", &["png", "jpg", "jpeg", "webp"])
                                        .pick_file()
                                    {
                                        self.load_image(&path, ctx);
                                    }
                                }
                            });
                        });
                    }
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.heading("ML Analysis");
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.show_landmarks, "Landmarks");
                        ui.checkbox(&mut self.show_depth_heatmap, "Depth");
                        ui.checkbox(&mut self.show_segmentation, "Segment");
                    });

                    if let Some(results) = &self.ml_results {
                        ui.label("✓ ML analysis complete");
                        
                        if results.landmarks.is_some() {
                            ui.label("  • Landmarks detected");
                        }
                        if results.depth_map.is_some() {
                            ui.label("  • Depth map generated");
                        }
                        if results.segmentation.is_some() {
                            ui.label("  • Segmentation complete");
                        }
                        
                        if let Some(ref landmarks) = results.landmarks {
                            ui.label(format!("  • {} landmark points", landmarks.points.len()));
                        }
                    } else if self.input_image.is_some() {
                        let analysis_enabled = !matches!(self.processing, ProcessingState::Running(_));
                        if ui.add_enabled(analysis_enabled, egui::Button::new("Run ML Analysis")).clicked() {
                            self.run_ml_analysis();
                        }
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(50.0);
                            ui.label("Load an image first");
                        });
                    }
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.heading("Output");
                    ui.separator();

                    if let Some(output) = &self.output_image {
                        ui.image(&output.texture);
                        let (w, h) = output.image.dimensions();
                        ui.label(format!("{}x{} pixels", w, h));
                        ui.label(format!("{} colors", output.palette.len()));
                        
                        if ui.button("Re-process").clicked() {
                            self.process_image(ctx);
                        }
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(50.0);
                            ui.label("Output will appear here");
                            ui.label("after processing");
                            
                            if self.input_image.is_some() && self.output_image.is_none() {
                                ui.add_space(10.0);
                                if ui.button("Process Now").clicked() {
                                    self.process_image(ctx);
                                }
                            }
                        });
                    }
                });
            });
        });
    }
}

// ============================================================================
// CONTROL PANEL IMPLEMENTATIONS
// ============================================================================

impl PixelForgeApp {
    pub fn ml_analysis_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.ml_config.face_detection_enabled, "Face Detection");
        });

        if self.ml_config.face_detection_enabled {
            ui.indent("face_indent", |ui| {
                ui.checkbox(&mut self.show_landmarks, "Show landmarks overlay");
            });
        }

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.ml_config.depth_estimation_enabled, "Depth Estimation");
        });

        if self.ml_config.depth_estimation_enabled {
            ui.indent("depth_indent", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Depth Bands:");
                    egui::ComboBox::from_id_salt("depth_bands")
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

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.ml_config.segmentation_enabled, "Segmentation");
        });

        if self.ml_config.segmentation_enabled {
            ui.indent("seg_indent", |ui| {
                ui.checkbox(&mut self.show_segmentation, "Show segmentation mask");
            });
        }

        ui.add_space(8.0);

        ui.label("Model Quality:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.config.model_quality, ModelQuality::Minimal), "Minimal").clicked() {
                self.config.model_quality = ModelQuality::Minimal;
                self.update_model_quality();
            }
            if ui.selectable_label(matches!(self.config.model_quality, ModelQuality::Standard), "Standard").clicked() {
                self.config.model_quality = ModelQuality::Standard;
                self.update_model_quality();
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.config.model_quality, ModelQuality::High), "High").clicked() {
                self.config.model_quality = ModelQuality::High;
                self.update_model_quality();
            }
            if ui.selectable_label(matches!(self.config.model_quality, ModelQuality::Sequential), "Sequential").clicked() {
                self.config.model_quality = ModelQuality::Sequential;
                self.update_model_quality();
            }
        });

        ui.add_space(8.0);

        let analysis_enabled = self.input_image.is_some() && 
            !matches!(self.processing, ProcessingState::Running(_));

        if ui.add_enabled(analysis_enabled, egui::Button::new("Run ML Analysis")).clicked() {
            self.run_ml_analysis();
        }
    }

    pub fn depth_to_flat_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Bands per Region:");

        ui.horizontal(|ui| {
            ui.label("Skin:");
            egui::ComboBox::from_id_salt("skin_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.skin_tone_bands))
                .show_ui(ui, |ui| {
                    for count in 2..=8 {
                        ui.selectable_value(&mut self.depth_to_flat_config.skin_tone_bands, count, count.to_string());
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Hair:");
            egui::ComboBox::from_id_salt("hair_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.hair_bands))
                .show_ui(ui, |ui| {
                    for count in 2..=8 {
                        ui.selectable_value(&mut self.depth_to_flat_config.hair_bands, count, count.to_string());
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Clothes:");
            egui::ComboBox::from_id_salt("clothes_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.clothing_bands))
                .show_ui(ui, |ui| {
                    for count in 2..=8 {
                        ui.selectable_value(&mut self.depth_to_flat_config.clothing_bands, count, count.to_string());
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Background:");
            egui::ComboBox::from_id_salt("bg_bands")
                .selected_text(format!("{}", self.depth_to_flat_config.background_bands))
                .show_ui(ui, |ui| {
                    for count in 1..=4 {
                        ui.selectable_value(&mut self.depth_to_flat_config.background_bands, count, count.to_string());
                    }
                });
        });

        ui.add_space(8.0);

        ui.label("Thresholds:");

        ui.horizontal(|ui| {
            ui.label("Shadow:");
            ui.add(egui::Slider::new(&mut self.depth_to_flat_config.shadow_threshold, 0.0..=1.0).text("").step_by(0.05));
            ui.label(format!("{:.0}%", self.depth_to_flat_config.shadow_threshold * 100.0));
        });

        ui.horizontal(|ui| {
            ui.label("Highlight:");
            ui.add(egui::Slider::new(&mut self.depth_to_flat_config.highlight_threshold, 0.0..=1.0).text("").step_by(0.05));
            ui.label(format!("{:.0}%", self.depth_to_flat_config.highlight_threshold * 100.0));
        });

        ui.add_space(8.0);

        ui.checkbox(&mut self.depth_to_flat_config.preserve_gradients, "Preserve gradients");

        if self.depth_to_flat_config.preserve_gradients {
            ui.indent("gradient_indent", |ui| {
                ui.label("Gradient strength:");
                ui.add(egui::Slider::new(&mut self.depth_to_flat_config.gradient_preservation, 0.0..=1.0).text(""));
            });
        }
    }

    pub fn feature_preserve_controls(&mut self, ui: &mut egui::Ui) {
        use crate::processing::{EyeSize, DetailLevel};

        ui.label("Eye Size:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.feature_config.eye_size, EyeSize::Auto), "Auto").clicked() {
                self.feature_config.eye_size = EyeSize::Auto;
            }
            if ui.selectable_label(matches!(self.feature_config.eye_size, EyeSize::Small), "Small").clicked() {
                self.feature_config.eye_size = EyeSize::Small;
            }
            if ui.selectable_label(matches!(self.feature_config.eye_size, EyeSize::Medium), "Medium").clicked() {
                self.feature_config.eye_size = EyeSize::Medium;
            }
            if ui.selectable_label(matches!(self.feature_config.eye_size, EyeSize::Large), "Large").clicked() {
                self.feature_config.eye_size = EyeSize::Large;
            }
        });

        ui.add_space(4.0);

        ui.label("Eye Detail:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.feature_config.eye_detail, DetailLevel::Minimal), "Min").clicked() {
                self.feature_config.eye_detail = DetailLevel::Minimal;
            }
            if ui.selectable_label(matches!(self.feature_config.eye_detail, DetailLevel::Standard), "Std").clicked() {
                self.feature_config.eye_detail = DetailLevel::Standard;
            }
            if ui.selectable_label(matches!(self.feature_config.eye_detail, DetailLevel::Full), "Full").clicked() {
                self.feature_config.eye_detail = DetailLevel::Full;
            }
        });

        ui.add_space(4.0);

        ui.label("Lip Detail:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.feature_config.lip_detail, DetailLevel::Minimal), "Min").clicked() {
                self.feature_config.lip_detail = DetailLevel::Minimal;
            }
            if ui.selectable_label(matches!(self.feature_config.lip_detail, DetailLevel::Standard), "Std").clicked() {
                self.feature_config.lip_detail = DetailLevel::Standard;
            }
            if ui.selectable_label(matches!(self.feature_config.lip_detail, DetailLevel::Full), "Full").clicked() {
                self.feature_config.lip_detail = DetailLevel::Full;
            }
        });

        ui.add_space(4.0);

        ui.label("Nose Detail:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.feature_config.nose_detail, DetailLevel::Minimal), "Min").clicked() {
                self.feature_config.nose_detail = DetailLevel::Minimal;
            }
            if ui.selectable_label(matches!(self.feature_config.nose_detail, DetailLevel::Standard), "Std").clicked() {
                self.feature_config.nose_detail = DetailLevel::Standard;
            }
            if ui.selectable_label(matches!(self.feature_config.nose_detail, DetailLevel::Full), "Full").clicked() {
                self.feature_config.nose_detail = DetailLevel::Full;
            }
        });

        ui.add_space(8.0);

        ui.checkbox(&mut self.feature_config.force_eye_highlights, "Force eye highlights");
        ui.checkbox(&mut self.feature_config.distinct_nostrils, "Distinct nostrils");

        ui.add_space(8.0);

        ui.label("Feature Sharpening:");
        ui.add(egui::Slider::new(&mut self.feature_config.feature_sharpening, 0.0..=1.0).text(""));
    }

    pub fn edge_controls(&mut self, ui: &mut egui::Ui) {
        use crate::processing::{EdgeMode, EdgeColorMode};

        ui.label("Edge Mode:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.edge_config.edge_mode, EdgeMode::None), "None").clicked() {
                self.edge_config.edge_mode = EdgeMode::None;
            }
            if ui.selectable_label(matches!(self.edge_config.edge_mode, EdgeMode::Outlines), "Outlines").clicked() {
                self.edge_config.edge_mode = EdgeMode::Outlines;
            }
            if ui.selectable_label(matches!(self.edge_config.edge_mode, EdgeMode::Internal), "Internal").clicked() {
                self.edge_config.edge_mode = EdgeMode::Internal;
            }
            if ui.selectable_label(matches!(self.edge_config.edge_mode, EdgeMode::Both), "Both").clicked() {
                self.edge_config.edge_mode = EdgeMode::Both;
            }
        });

        ui.add_space(8.0);

        ui.label("Thickness:");
        ui.add(egui::Slider::new(&mut self.edge_config.thickness, 1..=4).text("px"));

        ui.add_space(8.0);

        ui.label("Edge Color:");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(matches!(self.edge_config.edge_color_mode, EdgeColorMode::Black), "Black").clicked() {
                self.edge_config.edge_color_mode = EdgeColorMode::Black;
            }
            if ui.selectable_label(matches!(self.edge_config.edge_color_mode, EdgeColorMode::DarkestShade), "Dark").clicked() {
                self.edge_config.edge_color_mode = EdgeColorMode::DarkestShade;
            }
        });

        ui.horizontal(|ui| {
            if ui.selectable_label(matches!(self.edge_config.edge_color_mode, EdgeColorMode::Custom), "Custom").clicked() {
                self.edge_config.edge_color_mode = EdgeColorMode::Custom;
            }

            if matches!(self.edge_config.edge_color_mode, EdgeColorMode::Custom) {
                ui.color_edit_button_srgba(&mut self.edge_config.custom_edge_color);
            }
        });

        ui.add_space(8.0);

        ui.label("Edge Darkener:");
        ui.add(egui::Slider::new(&mut self.edge_config.edge_darkener_strength, 0.0..=1.0).text(""));
        ui.label("(darkens adjacent pixels)");

        ui.add_space(4.0);

        ui.checkbox(&mut self.edge_config.anti_alias_edges, "Anti-alias edges");
    }
}

fn apply_theme(ctx: &egui::Context, theme: &crate::config::Theme) {
    match theme {
        crate::config::Theme::Dark => {
            ctx.set_visuals(egui::Visuals::dark());
        }
        crate::config::Theme::Light => {
            ctx.set_visuals(egui::Visuals::light());
        }
        crate::config::Theme::System => {
            ctx.set_visuals(egui::Visuals::dark());
        }
    }
}

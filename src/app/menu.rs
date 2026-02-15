//! Menu bar implementation

use super::state::PixelForgeApp;
use crate::config::ModelQuality;
use eframe::egui;

pub fn draw(app: &mut PixelForgeApp, ctx: &egui::Context) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            file_menu(app, ui, ctx);
            edit_menu(app, ui);
            view_menu(app, ui);
            presets_menu(app, ui);
            models_menu(app, ui);
            help_menu(app, ui);
        });
    });
}

fn file_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    ui.menu_button("File", |ui| {
        if ui.button("📂 Open Image...").clicked() {
            super::processing::open_file_dialog(app);
            ui.close_menu();
        }

        if ui.add_enabled(app.input_image.is_some(), egui::Button::new("✖ Close Image")).clicked() {
            super::processing::clear_image(app);
            ui.close_menu();
        }

        ui.separator();

        if ui.add_enabled(app.output_image.is_some(), egui::Button::new("💾 Export Image...")).clicked() {
            super::processing::export_image(app);
            ui.close_menu();
        }

        ui.separator();

        if ui.button("Exit").clicked() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    });
}

fn edit_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("Edit", |ui| {
        if ui.button("Reset to Defaults").clicked() {
            app.transform_config = Default::default();
            app.depth_to_flat_config = Default::default();
            app.feature_config = Default::default();
            app.edge_config = Default::default();
            app.palette_config = Default::default();
            app.aspect_mode = Default::default();
            app.current_preset = None;
            ui.close_menu();
        }
    });
}

fn view_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("View", |ui| {
        ui.checkbox(&mut app.show_landmarks, "Show Landmarks");
        ui.checkbox(&mut app.show_depth_heatmap, "Show Depth Map");
        ui.checkbox(&mut app.show_segmentation, "Show Segmentation");
        ui.separator();
        ui.add(egui::Slider::new(&mut app.preview_zoom, 0.5..=8.0).text("Zoom"));
    });
}

fn presets_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("Presets", |ui| {
        if ui.button("Save Preset...").clicked() {
            super::processing::save_preset(app);
            ui.close_menu();
        }

        if ui.button("Load Preset...").clicked() {
            super::processing::load_preset(app);
            ui.close_menu();
        }

        ui.separator();

        if ui.button("Portrait - Minimal (32x32)").clicked() {
            apply_preset(app, "minimal");
            ui.close_menu();
        }
        if ui.button("Portrait - Detailed (64x64)").clicked() {
            apply_preset(app, "detailed");
            ui.close_menu();
        }
        if ui.button("Portrait - HD (128x128)").clicked() {
            apply_preset(app, "hd");
            ui.close_menu();
        }
        if ui.button("Game Boy Style (48x48)").clicked() {
            apply_preset(app, "gameboy");
            ui.close_menu();
        }
    });
}

fn models_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("Models", |ui| {
        if ui.button("Manage Models...").clicked() {
            app.show_model_dialog = true;
            ui.close_menu();
        }

        ui.separator();

        for quality in [ModelQuality::Minimal, ModelQuality::Standard, ModelQuality::High] {
            if ui.radio(app.config.model_quality == quality, quality.display_name()).clicked() {
                app.config.model_quality = quality;
                super::processing::update_model_settings(app);
            }
        }

        ui.separator();

        if ui.checkbox(&mut app.config.sequential_processing, "Sequential Mode (Low VRAM)").changed() {
            super::processing::update_model_settings(app);
        }
    });
}

fn help_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("Help", |ui| {
        if ui.button("About PixelForge").clicked() {
            app.show_about_dialog = true;
            ui.close_menu();
        }
    });
}

fn apply_preset(app: &mut PixelForgeApp, preset_name: &str) {
    use crate::processing::{EyeSize, DetailLevel, PaletteMode, PresetPalette};
    
    match preset_name {
        "minimal" => {
            app.transform_config = crate::processing::TransformConfig { 
                output_size: 32, ..Default::default() 
            };
            app.aspect_mode = super::state::AspectRatioMode::Square;
            app.depth_to_flat_config = crate::processing::DepthToFlatConfig { 
                skin_tone_bands: 3, hair_bands: 2, clothing_bands: 2, background_bands: 1, 
                ..Default::default() 
            };
            app.feature_config = crate::processing::FeaturePreserveConfig {
                eye_size: EyeSize::Small,
                eye_detail: DetailLevel::Minimal,
                ..Default::default()
            };
            app.palette_config = crate::processing::PaletteConfig { 
                max_colors: 12, ..Default::default() 
            };
            app.current_preset = Some("Portrait - Minimal".to_string());
        }
        "detailed" => {
            app.transform_config = crate::processing::TransformConfig { 
                output_size: 64, ..Default::default() 
            };
            app.aspect_mode = super::state::AspectRatioMode::Square;
            app.depth_to_flat_config = crate::processing::DepthToFlatConfig { 
                skin_tone_bands: 5, hair_bands: 4, clothing_bands: 3, background_bands: 2, 
                ..Default::default() 
            };
            app.feature_config = crate::processing::FeaturePreserveConfig {
                eye_size: EyeSize::Medium,
                eye_detail: DetailLevel::Standard,
                lip_detail: DetailLevel::Standard,
                nose_detail: DetailLevel::Standard,
                distinct_nostrils: true,
                ..Default::default()
            };
            app.palette_config = crate::processing::PaletteConfig { 
                max_colors: 32, ..Default::default() 
            };
            app.current_preset = Some("Portrait - Detailed".to_string());
        }
        "hd" => {
            app.transform_config = crate::processing::TransformConfig { 
                output_size: 128, ..Default::default() 
            };
            app.aspect_mode = super::state::AspectRatioMode::Preserve;
            app.depth_to_flat_config = crate::processing::DepthToFlatConfig { 
                skin_tone_bands: 6, hair_bands: 5, clothing_bands: 4, background_bands: 3, 
                ..Default::default() 
            };
            app.feature_config = crate::processing::FeaturePreserveConfig {
                eye_size: EyeSize::Large,
                eye_detail: DetailLevel::Full,
                lip_detail: DetailLevel::Full,
                nose_detail: DetailLevel::Standard,
                distinct_nostrils: true,
                force_eye_highlights: true,
                ..Default::default()
            };
            app.palette_config = crate::processing::PaletteConfig { 
                max_colors: 64, ..Default::default() 
            };
            app.current_preset = Some("Portrait - HD".to_string());
        }
        "gameboy" => {
            app.transform_config = crate::processing::TransformConfig { 
                output_size: 48, export_scale: 2, ..Default::default() 
            };
            app.aspect_mode = super::state::AspectRatioMode::Square;
            app.depth_to_flat_config = crate::processing::DepthToFlatConfig { 
                skin_tone_bands: 2, hair_bands: 2, clothing_bands: 2, background_bands: 1, 
                ..Default::default() 
            };
            app.palette_config = crate::processing::PaletteConfig {
                mode: PaletteMode::Preset,
                preset: PresetPalette::GameBoy,
                max_colors: 4,
                ..Default::default()
            };
            app.current_preset = Some("Game Boy Style".to_string());
        }
        _ => {}
    }
}

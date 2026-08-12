//! Menu bar

use super::state::{PixelForgeApp, PreviewTab};
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
        if ui.button("📂 Open…").clicked() {
            super::processing::open_file_dialog(app);
            ui.close_menu();
        }

        let has_image = app.input_image.is_some();

        if ui.add_enabled(has_image, egui::Button::new("✖ Close Image")).clicked() {
            super::processing::clear_image(app);
            ui.close_menu();
        }

        ui.separator();

        if ui.add_enabled(app.output_image.is_some(), egui::Button::new("💾 Export…")).clicked() {
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
            app.transform_config     = Default::default();
            app.depth_to_flat_config = Default::default();
            app.edge_config          = Default::default();
            app.palette_config       = Default::default();
            app.ml_config            = Default::default();
            app.aspect_mode          = Default::default();
            app.current_preset       = None;
            ui.close_menu();
        }
    });
}

fn view_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("View", |ui| {
        for (tab, label) in [
            (PreviewTab::Original, "📷 Original"),
            (PreviewTab::MLMaps,   "🤖 ML Maps"),
            (PreviewTab::Output,   "🎨 Output"),
        ] {
            if ui.selectable_label(app.preview_tab == tab, label).clicked() {
                app.preview_tab = tab;
                ui.close_menu();
            }
        }

        ui.separator();

        ui.add(egui::Slider::new(&mut app.preview_zoom, 0.25..=8.0).text("Zoom"));

        for (label, zoom) in [("Fit", 1.0f32), ("2×", 2.0), ("4×", 4.0)] {
            if ui.small_button(label).clicked() {
                app.preview_zoom = zoom;
            }
        }
    });
}

fn presets_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("Presets", |ui| {
        if ui.button("Save Preset…").clicked() {
            super::processing::save_preset(app);
            ui.close_menu();
        }
        if ui.button("Load Preset…").clicked() {
            super::processing::load_preset(app);
            ui.close_menu();
        }

        ui.separator();
        ui.label(egui::RichText::new("Built-in").small().weak());

        for (label, id) in [
            ("Portrait 32×32 — Minimal",  "minimal"),
            ("Portrait 64×64 — Detailed", "detailed"),
            ("Portrait 128 — HD",         "hd"),
            ("Game Boy 48×48",            "gameboy"),
        ] {
            if ui.button(label).clicked() {
                apply_builtin_preset(app, id);
                ui.close_menu();
            }
        }
    });
}

fn models_menu(app: &mut PixelForgeApp, ui: &mut egui::Ui) {
    ui.menu_button("Models", |ui| {
        if ui.button("Manage Models…").clicked() {
            app.show_model_dialog = true;
            ui.close_menu();
        }

        ui.separator();

        // Execution mode — controls whether models run on CPU, GPU-sequential, or GPU-parallel.
        ui.label(egui::RichText::new("Execution").small().weak());

        use crate::ml::ExecutionMode;
        for (mode, label) in [
            (ExecutionMode::GpuSequential, "GPU — Sequential (default)"),
            (ExecutionMode::GpuParallel,   "GPU — Parallel"),
            (ExecutionMode::CpuOnly,       "CPU Only"),
        ] {
            if ui.radio(app.ml_config.execution.mode == mode, label).clicked() {
                app.ml_config.execution.mode = mode;
            }
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

// ─────────────────────────────────────────────────────────────────────────────
// Built-in preset application
// ─────────────────────────────────────────────────────────────────────────────

fn apply_builtin_preset(app: &mut PixelForgeApp, id: &str) {
    use crate::processing::{PaletteMode, PresetPalette};

    match id {
        "minimal" => {
            app.transform_config.output_size = 32;
            app.aspect_mode = super::state::AspectRatioMode::Square;
            app.depth_to_flat_config = Default::default();
            app.palette_config.mode = PaletteMode::Auto;
            app.palette_config.max_colors = 12;
            app.current_preset = Some("Portrait — Minimal".into());
        }
        "detailed" => {
            app.transform_config.output_size = 64;
            app.aspect_mode = super::state::AspectRatioMode::Square;
            app.depth_to_flat_config = Default::default();
            app.palette_config.mode = PaletteMode::Auto;
            app.palette_config.max_colors = 32;
            app.current_preset = Some("Portrait — Detailed".into());
        }
        "hd" => {
            app.transform_config.output_size = 128;
            app.aspect_mode = super::state::AspectRatioMode::Preserve;
            app.depth_to_flat_config = Default::default();
            app.palette_config.mode = PaletteMode::Auto;
            app.palette_config.max_colors = 64;
            app.current_preset = Some("Portrait — HD".into());
        }
        "gameboy" => {
            app.transform_config.output_size  = 48;
            app.transform_config.export_scale = 2;
            app.aspect_mode = super::state::AspectRatioMode::Square;
            app.depth_to_flat_config = Default::default();
            app.palette_config.mode       = PaletteMode::Preset;
            app.palette_config.preset     = PresetPalette::GameBoy;
            app.palette_config.max_colors = 4;
            app.current_preset = Some("Game Boy Style".into());
        }
        _ => {}
    }
}

//! Theme configuration

use crate::config::Theme;
use eframe::egui;

/// Apply theme to egui context
pub fn apply_theme(ctx: &egui::Context, theme: &Theme) {
    match theme {
        Theme::Dark => ctx.set_visuals(egui::Visuals::dark()),
        Theme::Light => ctx.set_visuals(egui::Visuals::light()),
        Theme::System => ctx.set_visuals(egui::Visuals::dark()),
    }
}

//! PixelForge - ML-enhanced pixel art style transfer for character portraits
//!
//! A high-quality pixel art generator that uses machine learning to preserve
//! facial features and translate 3D depth cues into flat color regions.

mod app;
use eframe::egui;
// Re-export library modules at the binary crate root so `app/` submodules
// can use `crate::processing`, `crate::ml`, etc.
pub use pixelforge::*;

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    log::info!("Starting PixelForge...");

    // Load application config
    let config = config::AppConfig::load().unwrap_or_default();

    // Setup native options
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([1024.0, 700.0])
            .with_title("PixelForge")
            .with_drag_and_drop(true),
        ..Default::default()
    };

    // Run the application
    let result = eframe::run_native(
        "PixelForge",
        native_options,
        Box::new(|cc| Ok(Box::new(app::PixelForgeApp::new(cc, config)))),
    );

    if let Err(e) = result {
        log::error!("Application error: {}", e);
    }
}

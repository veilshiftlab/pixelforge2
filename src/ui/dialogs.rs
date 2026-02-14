//! Dialog implementations

use eframe::egui;

/// About dialog
pub fn about_dialog(ui: &mut egui::Ui) {
    ui.vertical_centered(|ui| {
        ui.heading("PixelForge");
        ui.label("Version 0.1.0");
        ui.add_space(8.0);
        ui.label("ML-enhanced pixel art style transfer");
        ui.label("for character portraits");
        ui.add_space(16.0);
        ui.label("Built with Rust, egui, and ONNX Runtime");
    });
}

/// Model download dialog
pub fn model_download_dialog(ui: &mut egui::Ui, download_progress: Option<f32>) {
    ui.heading("Download Models");
    ui.separator();

    ui.label("Available Models:");
    ui.add_space(8.0);

    // Model list with checkboxes
    let mut download_mediapipe = false;
    let mut download_dpt = false;
    let mut download_sam = false;
    
    ui.checkbox(&mut download_mediapipe, "MediaPipe Face Mesh (~12MB) - Better landmarks");
    ui.checkbox(&mut download_dpt, "DPT-Large (~340MB) - Superior depth estimation");
    ui.checkbox(&mut download_sam, "SAM-ViT-B (~375MB) - Superior segmentation");

    ui.add_space(16.0);

    if let Some(progress) = download_progress {
        ui.add(egui::ProgressBar::new(progress).text(format!("{:.0}%", progress * 100.0)));
        ui.label("Downloading...");
    } else {
        ui.horizontal(|ui| {
            if ui.button("Download Selected").clicked() {
                // Start download
            }
            if ui.button("Close").clicked() {
                // Close dialog
            }
        });
    }
}

/// Batch processing dialog
pub fn batch_dialog(ui: &mut egui::Ui, _images: &[std::path::PathBuf], _processing: bool) {
    ui.heading("Batch Processing");
    ui.separator();

    ui.horizontal(|ui| {
        if ui.button("Add Files...").clicked() {
            // Add files
        }
        if ui.button("Add Folder...").clicked() {
            // Add folder
        }
        if ui.button("Clear All").clicked() {
            // Clear list
        }
    });

    ui.add_space(8.0);

    ui.label("No files added yet.");

    ui.add_space(8.0);

    ui.horizontal(|ui| {
        ui.label("Output Folder:");
        ui.button("Browse...");
    });

    ui.add_space(16.0);

    ui.horizontal(|ui| {
        if ui.button("Process All").clicked() {
            // Start batch processing
        }
        if ui.button("Close").clicked() {
            // Close
        }
    });
}

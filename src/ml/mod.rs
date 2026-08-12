//! ML Analysis Module
//!
//! Provides machine learning map generation for PixelForge:
//!
//! - **Depth Estimation** (`depth_anything`): Depth-Anything V2 — per-pixel relative depth
//! - **Edge Detection** (`teed`): TEED — crisp perceptual edge map
//!
//! YOLOv8n-Face and BiSeNet were removed in the pipeline repurpose (see `plan.md`).
//! Region classification is now handled model-free by SLIC superpixels in
//! `crate::processing::slic`.
//!
//! # Module layout
//!
//! ```text
//! src/ml/
//! ├── mod.rs            — re-exports
//! ├── config.rs         — per-model and combined configuration
//! ├── analysis.rs       — pipeline orchestration (MLAnalysis::analyze)
//! ├── types.rs          — result types (MLResults, …)
//! ├── preprocess.rs     — tensor preparation and bilinear coordinate remapping
//! ├── session.rs        — ONNX session cache + execution-provider selection
//! └── models/
//!     ├── mod.rs
//!     ├── depth_anything.rs
//!     └── teed.rs
//! ```

mod analysis;
mod config;
mod types;
pub mod models;
pub mod preprocess;
pub mod session;

pub use analysis::MLAnalysis;
pub use config::*;
pub use types::*;

//! Processing pipeline and configuration

mod config;
pub mod pipeline;
mod depth_to_flat;
mod downsampling;
mod feature_preserve;
mod edges;
mod palette;

pub use config::*;
pub use depth_to_flat::*;
pub use downsampling::*;
pub use feature_preserve::*;
pub use edges::*;
pub use palette::*;
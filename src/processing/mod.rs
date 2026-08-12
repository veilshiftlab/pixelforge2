//! Processing pipeline and configuration

mod config;
pub mod pipeline;
mod depth_to_flat;
mod downsampling;
mod edges;
mod palette;
pub mod slic;

pub use config::*;
pub use depth_to_flat::*;
pub use downsampling::*;
pub use edges::*;
pub use palette::*;

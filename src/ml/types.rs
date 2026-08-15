//! ML data types and structures
//!
//! After the pipeline repurpose (see `plan.md`), PixelForge ships only two ML
//! models: Depth-Anything V2 (depth) and TEED (edges). Face detection,
//! landmarks, and segmentation were removed because BiSeNet does not segment
//! anime-style images reliably, and YOLOv8n-Face had no consumer once BiSeNet
//! was gone. Region classification is now model-free, via SLIC superpixels
//! (`crate::processing::slic`).

/// Results from ML analysis.
///
/// Both maps are stored at the **original image resolution**, row-major, with
/// values normalized to `[0, 1]`. The pipeline resamples them to the
/// post-transform resolution when needed (see `processing/pipeline.rs`).
#[derive(Debug, Clone, Default)]
pub struct MLResults {
    /// Depth map from Depth-Anything V2 (normalized 0.0–1.0, row-major,
    /// same dims as input image). `0 = nearest, 1 = farthest`.
    pub depth_map: Option<Vec<f32>>,

    /// Phase 6 — P1: 5×5 median-filtered depth map, cached so `depth_to_flat`
    /// doesn't recompute it on every pipeline invocation (every slider tweak).
    /// Computed once in `MLAnalysis::analyze` after depth inference; cleared
    /// when ML results are invalidated.
    pub filtered_depth_map: Option<Vec<f32>>,

    /// Edge probability map from TEED (normalized 0.0–1.0, row-major,
    /// same dims as input image). `1.0 = strong edge`.
    pub edge_map: Option<Vec<f32>>,

    /// SLIC superpixel label map (one cluster ID per pixel, same dims as
    /// input image). Populated lazily by `processing::slic` and cached here so
    /// pipeline re-runs (palette/edge tweaks) reuse the same clustering.
    pub slic_labels: Option<Vec<u32>>,

    /// Config snapshot used to generate `slic_labels`. When the user changes
    /// K or spatial_weight, the pipeline detects the mismatch and re-clusters.
    pub slic_labels_k: Option<u32>,
    pub slic_labels_s: Option<f32>,
}

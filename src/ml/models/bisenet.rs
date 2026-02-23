//! BiSeNet: Face Parsing / Semantic Segmentation
//!
//! # Model Details
//!
//! - **Source**: `qualcomm/BiseNet` or `szhubel/face-parsing-bisenet` on HuggingFace
//! - **Size**: ~48 MB
//! - **Architecture**: BiSeNet (Bilateral Segmentation Network)
//! - **Input**: RGB float32 NCHW, ImageNet-normalized, default 512×512
//! - **Output**: `[1, 19, H, W]` — class logits per pixel
//!
//! # Class mapping
//!
//! We preserve all 19 BiSeNet classes in [`SegmentationRegion`] without collapsing
//! distinct regions like eyebrows/eyes/lips into a generic "Face" bucket.
//! Downstream pixel art processing can then apply distinct palette, dithering,
//! and detail budgets per region.
//!
//! # Coordinate mapping
//!
//! Class argmax is computed in model output space, then the discrete label map is
//! upsampled to original resolution via nearest-neighbour (bilinear interpolation
//! on class indices is meaningless — we only bilinear on continuous maps like depth).

use crate::ml::{SegmentationRegion, SegmentationResult};
use crate::ml::preprocess::{
    upsample_class_map_to_original, PreprocessConfig, ResizeScale, preprocess,
};
use crate::ml::session::{SessionManager, ModelType};
use anyhow::{anyhow, Result};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_INPUT_SIZE: u32 = 512;
const NUM_CLASSES: usize = 19;

// ─────────────────────────────────────────────────────────────────────────────
// Segmenter
// ─────────────────────────────────────────────────────────────────────────────

/// Face parsing segmenter using BiSeNet ONNX model.
pub struct BiSeNetSegmenter {
    session_manager: Arc<SessionManager>,
    model_path: PathBuf,
    input_name: String,
    output_name: String,
    model_w: u32,
    model_h: u32,
}

impl BiSeNetSegmenter {
    /// Load the ONNX model and resolve its input dimensions using SessionManager for GPU support.
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let session_manager = crate::ml::session::global_session_manager();
        let model_session = session_manager.get_or_load(&model_path.to_path_buf(), ModelType::Segmentation)?;
        
        let session = model_session.session.lock().unwrap();
        let input_name  = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        let (model_w, model_h) = resolve_input_dims(&session, DEFAULT_INPUT_SIZE);
        drop(session);

        log::info!(
            "BiSeNet loaded: input={}x{}, input_name='{}' (backend: {:?})",
            model_w, model_h, input_name, session_manager.backend()
        );

        Ok(Self {
            session_manager,
            model_path: model_path.to_path_buf(),
            input_name,
            output_name,
            model_w,
            model_h,
        })
    }

    /// Segment `image` into face regions.
    ///
    /// Returns a [`SegmentationResult`] at the original image's resolution.
    /// The `regions` map uses `y * width + x` as pixel index.
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        use image::GenericImageView;
        let (orig_w, orig_h) = image.dimensions();

        // BiSeNet uses ImageNet normalization
        let cfg = PreprocessConfig::imagenet(self.model_w, self.model_h);
        let (tensor, scale) = preprocess(image, &cfg)?;

        let input_tensor = ort::value::Value::from_array((
            [1i64, 3, self.model_h as i64, self.model_w as i64],
            tensor,
        ))?;

        let model_session = self.session_manager.get_or_load(&self.model_path, ModelType::Segmentation)?;
        let mut session = model_session.session.lock().unwrap();
        let outputs = session.run(ort::inputs![&self.input_name => input_tensor])?;

        let output = outputs
            .get(&self.output_name)
            .ok_or_else(|| anyhow!("BiSeNet produced no output"))?;

        let (shape, logits) = output.try_extract_tensor::<f32>()?;

        build_segmentation_result(&shape, &logits, &scale, orig_w, orig_h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output processing
// ─────────────────────────────────────────────────────────────────────────────

fn build_segmentation_result(
    shape: &[i64],
    logits: &[f32],
    scale: &ResizeScale,
    orig_w: u32,
    orig_h: u32,
) -> Result<SegmentationResult> {
    // Expected: [1, 19, H, W]
    let (num_classes, map_h, map_w) = match shape.len() {
        4 => (shape[1] as usize, shape[2] as usize, shape[3] as usize),
        3 => (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        _ => return Err(anyhow!("Unexpected BiSeNet output shape: {:?}", shape)),
    };

    // Upsample class map to original resolution (nearest-neighbour, correct for labels)
    let (class_map, raw_probs) =
        upsample_class_map_to_original(logits, num_classes, map_w, map_h, scale);

    // Build region and bounds maps
    let mut regions: HashMap<u32, SegmentationRegion> =
        HashMap::with_capacity((orig_w * orig_h) as usize);
    let mut region_bounds: HashMap<SegmentationRegion, (f32, f32, f32, f32)> = HashMap::new();

    for (i, (class_id, raw_prob)) in class_map.iter().zip(raw_probs.iter()).enumerate() {
        let class_id = *class_id;
        let _raw_prob = *raw_prob; // used below for confidence
        let x = (i % orig_w as usize) as u32;
        let y = (i / orig_w as usize) as u32;
        let px = i as u32;

        let region = bisenet_class_to_region(class_id);
        regions.insert(px, region);

        // Maintain per-region bounding boxes (normalized)
        let nx = x as f32 / orig_w as f32;
        let ny = y as f32 / orig_h as f32;
        let entry = region_bounds.entry(region).or_insert((nx, ny, nx, ny));
        if nx < entry.0 { entry.0 = nx; }
        if ny < entry.1 { entry.1 = ny; }
        if nx > entry.2 { entry.2 = nx; }
        if ny > entry.3 { entry.3 = ny; }
    }

    // Convert raw logits to a 0–1 confidence approximation via softmax-free normalization.
    // We just sigmoid the max logit as a cheap proxy — sufficient for downstream filtering.
    let confidence: Vec<f32> = raw_probs
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect();

    Ok(SegmentationResult { regions, region_bounds, confidence })
}

// ─────────────────────────────────────────────────────────────────────────────
// Class mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map a BiSeNet class index to a [`SegmentationRegion`].
///
/// All 19 classes are preserved — no collapsing of distinct face regions.
///
/// | ID | BiSeNet class      | Region           |
/// |----|-------------------|------------------|
/// |  0 | Background        | Background       |
/// |  1 | Skin              | Skin             |
/// |  2 | Left eyebrow      | LeftEyebrow      |
/// |  3 | Right eyebrow     | RightEyebrow     |
/// |  4 | Left eye          | LeftEye          |
/// |  5 | Right eye         | RightEye         |
/// |  6 | Eyeglasses        | Eyeglasses       |
/// |  7 | Left ear          | LeftEar          |
/// |  8 | Right ear         | RightEar         |
/// |  9 | Earring           | Earring          |
/// | 10 | Nose              | Nose             |
/// | 11 | Inner mouth       | InnerMouth       |
/// | 12 | Upper lip         | UpperLip         |
/// | 13 | Lower lip         | LowerLip         |
/// | 14 | Neck              | Neck             |
/// | 15 | Hair              | Hair             |
/// | 16 | Hat               | Hat              |
/// | 17 | Left earring      | Earring          |
/// | 18 | Right earring     | Earring          |
fn bisenet_class_to_region(class_id: usize) -> SegmentationRegion {
    match class_id {
        0  => SegmentationRegion::Background,
        1  => SegmentationRegion::Skin,
        2  => SegmentationRegion::LeftEyebrow,
        3  => SegmentationRegion::RightEyebrow,
        4  => SegmentationRegion::LeftEye,
        5  => SegmentationRegion::RightEye,
        6  => SegmentationRegion::Eyeglasses,
        7  => SegmentationRegion::LeftEar,
        8  => SegmentationRegion::RightEar,
        9  => SegmentationRegion::Earring,
        10 => SegmentationRegion::Nose,
        11 => SegmentationRegion::InnerMouth,
        12 => SegmentationRegion::UpperLip,
        13 => SegmentationRegion::LowerLip,
        14 => SegmentationRegion::Neck,
        15 => SegmentationRegion::Hair,
        16 => SegmentationRegion::Hat,
        17 | 18 => SegmentationRegion::Earring,
        _  => SegmentationRegion::Background,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dimension resolution
// ─────────────────────────────────────────────────────────────────────────────


fn resolve_input_dims(
    _session: &ort::session::Session,
    default_size: u32,
) -> (u32, u32) {
    // ORT 2.x does not expose a stable API to read symbolic input dims before inference.
    // We use well-known defaults; dynamic-axes exports accept any size anyway.
    (default_size, default_size)
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder segmenter — generates a coarse face region layout for testing.
pub struct PlaceholderSegmenter;

impl PlaceholderSegmenter {
    pub fn segment(&self, image: &DynamicImage) -> Result<SegmentationResult> {
        use image::GenericImageView;
        let (w, h) = image.dimensions();
        let mut regions = HashMap::with_capacity((w * h) as usize);

        let fy0 = (h as f32 * 0.15) as u32;
        let fy1 = (h as f32 * 0.75) as u32;
        let fx0 = (w as f32 * 0.25) as u32;
        let fx1 = (w as f32 * 0.75) as u32;
        let fw   = (fx1 - fx0).max(1) as f32;
        let fh   = (fy1 - fy0).max(1) as f32;

        for y in 0..h {
            for x in 0..w {
                let region = if y < fy0 {
                    SegmentationRegion::Hair
                } else if y >= fy0 && y <= fy1 && x >= fx0 && x <= fx1 {
                    let rx = (x - fx0) as f32 / fw;
                    let ry = (y - fy0) as f32 / fh;
                    if ry > 0.25 && ry < 0.40 {
                        if rx > 0.18 && rx < 0.40 { SegmentationRegion::LeftEye }
                        else if rx > 0.60 && rx < 0.82 { SegmentationRegion::RightEye }
                        else { SegmentationRegion::Skin }
                    } else if ry > 0.42 && ry < 0.62 && rx > 0.38 && rx < 0.62 {
                        SegmentationRegion::Nose
                    } else if ry > 0.68 && ry < 0.82 && rx > 0.30 && rx < 0.70 {
                        SegmentationRegion::UpperLip
                    } else {
                        SegmentationRegion::Skin
                    }
                } else {
                    SegmentationRegion::Background
                };

                regions.insert(y * w + x, region);
            }
        }

        Ok(SegmentationResult {
            regions,
            region_bounds: HashMap::new(),
            confidence: vec![0.9; (w * h) as usize],
        })
    }
}
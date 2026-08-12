# PixelForge

ML-enhanced pixel art style transfer for character portraits.

A high-quality pixel art generator that uses machine learning to translate 3D
depth cues into flat color regions and perceptual edges into crisp outlines.

## Features

- **Smart Downsampling**: Content-aware weighted downsampling that preserves important details
- **Depth-to-Flat Conversion**: Translates 3D depth cues into flat color bands via background separation (per-region shading via SLIC superpixels arrives in Phase 2.3)
- **Edge Control**: Adjustable edge thickness, color modes, and darkening effects
- **Palette Management**: Auto-extraction, preset palettes (Game Boy, NES, PICO-8, DawnBringer32, AAP64), and custom color options
- **ML Analysis**: Depth estimation (Depth-Anything V2) and edge detection (TEED)
- **Graceful Degradation**: If a model isn't downloaded or fails to load, the pipeline falls back to Sobel-based placeholders — the app remains fully functional

## Requirements

- Rust 1.70 or later
- ONNX Runtime (automatically downloaded via the `ort` crate)
- 4GB VRAM recommended (2GB minimum with CPU fallback)
- 2GB RAM free space

## Building

```bash
# Clone the repository
cd pixelforge

# Build in release mode
cargo build --release

# Run
cargo run --release
```

## Usage

1. **Load an Image**: Drag and drop an image or click "Browse..." in the preview area
2. **Run ML Analysis**: Click "Run ML Analysis" to generate depth and edge maps
3. **Adjust Settings**: Modify depth-to-flat, edges, and palette settings
4. **Process**: Click "Process" to generate the pixel art
5. **Export**: Save the result as PNG or JPEG

## Models

PixelForge uses two ONNX models. Both are optional — the app falls back to
placeholder algorithms when a model isn't available.

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| **Depth-Anything V2 Small** (ViT-S) | Per-pixel relative depth for background separation and shading | ~94 MB | [SStahan/ONNX-exports-dpt-seg-edge-face](https://huggingface.co/SStahan/ONNX-exports-dpt-seg-edge-face/resolve/main/depth_anything_v2_vits.onnx) |
| **DexiNed** | Perceptual edge detection for the outline pass | ~134 MB | [SStahan/ONNX-exports-dpt-seg-edge-face](https://huggingface.co/SStahan/ONNX-exports-dpt-seg-edge-face/resolve/main/dexined.onnx) |

**Total download: ~228 MB** (down from 1.34 GB with the Large depth model).

> **Why ViT-S instead of ViT-L?** The Small variant is ~14× smaller (94 MB vs
> 1.34 GB) with only slightly worse accuracy. For pixel art — where depth is
> used for shading bands, not precise geometry — the difference is negligible.

> **Why DexiNed instead of TEED?** TEED has no canonical ONNX export on
> HuggingFace. DexiNed is a comparable perceptual edge detector that produces
> crisp, thin edges. The edge detector module auto-detects whether the loaded
> model has fixed (DexiNed: 512×512) or dynamic (TEED) input dimensions and
> handles both cases. If you export TEED to ONNX yourself, the module will
> work with it too — just swap the entry in `models.toml`.

### Additional models (available but unused)

The user's HuggingFace repo also contains three other ONNX exports that are
**not wired into the pipeline** per the repurpose plan:

| Model | Size | Why it's not used |
|-------|------|-------------------|
| `face_parsing_resnet18.onnx` | 50 MB | Plan dropped face parsing (BiSeNet didn't work on anime; "two logic paths for photo vs anime" was rejected) |
| `fashn_human_parser.onnx` | 245 MB | Plan dropped body segmentation (same reasoning) |
| `yolov12m_face.onnx` | 78 MB | Plan dropped face detection (no consumer after BiSeNet removal) |

They are listed as commented entries in `models.toml`. Uncomment and enable
only if you decide to deviate from the plan.

## Pipeline Stages

1. **ML inference** (cached) — Depth-Anything V2 + DexiNed run on the original image
2. **Depth preprocessing** — 5×5 median filter on the depth map (removes speckle)
3. **SLIC superpixels** — model-free region classification on `(L, a, b, depth, x·s, y·s)`
4. **Per-region shading** — MAD-normalized depth → contrast curve → Lab L* bias
5. **Geometric transforms** — scale, rotate, offset, flip
6. **ML map resampling** — depth (bilinear) and edge (nearest) maps resampled to post-transform dimensions
7. **Importance map** — Sobel + depth-gradient weighted
8. **Weighted downsample** — to target pixel-art resolution
9. **DexiNed outline pass** — auto-contrast outlines in Lab space (before palette so strict palettes are maintained)
10. **Palette quantization** — nearest-color in Lab space

## Configuration

### ML Analysis

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Depth Estimation | on | toggle | Enable Depth-Anything V2 |
| Edge Detection | on | toggle | Enable TEED |
| Edge threshold | 0.3 | 0.05–0.95 | TEED inference threshold |

### Depth → Flat

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Strength | 0.4 | 0–1 | Shading intensity (Lab L* bias) |
| Gamma | 1.0 | 0.5–2.0 | Contrast curve exponent |
| MAD threshold | 0.05 | 0.01–0.2 | Low-variance region skip |
| Auto threshold | on | toggle | Otsu background separation |
| BG desaturate | 0.65 | 0–1 | Background saturation reduction |
| BG darken | −0.08 | −0.5–0.2 | Background L* shift |

### Regions (SLIC)

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Clusters (K) | 5 | 3–8 | Number of superpixel regions |
| Spatial weight | 0.5 | 0–1 | Blobbiness vs. detail fidelity |

### Edges

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Mode | Outlines | None/Outlines/Internal/Both | Edge detection mode |
| Thickness | 1 | 1–4 | Outline pixel width |
| Darkener | 0.3 | 0–1 | Adjacent pixel darkening |
| Anti-alias | off | toggle | Edge anti-aliasing |
| Outline style | AutoContrast | AutoContrast/AutoContrastWithHueShift/Black | TEED outline coloring |
| Delta L* | 30 | 10–60 | L* shift for auto-contrast |
| Hue shift | 10° | 0–30° | a*b* plane rotation (with hue-shift style) |
| TEED threshold | 0.3 | 0.1–0.7 | Edge probability threshold |

Settings are automatically saved to:
- **Windows**: `%APPDATA%\pixelforge\config.json`
- **macOS**: `~/Library/Application Support/pixelforge/config.json`
- **Linux**: `~/.config/pixelforge/config.json`

## Presets

Built-in presets include:
- **Portrait - Minimal**: Fast processing, 32×32 output
- **Portrait - Detailed**: Higher quality, 64×64 output
- **Game Boy Style**: 4-color retro aesthetic, 48×48 output

Custom presets can be saved and loaded from the Presets menu.

## Project Structure

```
pixelforge/
├── src/
│   ├── main.rs              # Entry point
│   ├── config.rs            # Configuration management
│   ├── app/                 # egui application
│   │   ├── state.rs         # Application state
│   │   ├── panels.rs        # Side/bottom panels
│   │   ├── preview.rs       # Central preview
│   │   ├── processing.rs    # Pipeline invocation
│   │   ├── menu.rs          # Menu bar
│   │   ├── dialogs.rs       # Modal dialogs
│   │   └── theme.rs         # UI theme
│   ├── processing/          # Image processing pipeline
│   │   ├── pipeline.rs      # Main pipeline orchestration
│   │   ├── depth_to_flat.rs # Depth → flat color conversion
│   │   ├── downsampling.rs  # Weighted / nearest / bilinear downsampling
│   │   ├── edges.rs         # Edge detection and drawing
│   │   ├── palette.rs       # Color quantization
│   │   └── config.rs        # Processing config structs
│   ├── ml/                  # Machine learning
│   │   ├── analysis.rs      # ML orchestration
│   │   ├── types.rs         # MLResults
│   │   ├── config.rs        # ML config structs
│   │   ├── preprocess.rs    # Tensor preparation
│   │   ├── session.rs       # ONNX session cache + backend selection
│   │   └── models/
│   │       ├── depth_anything.rs
│   │       └── teed.rs
│   ├── models/              # Model management
│   │   ├── manager.rs       # Model loading/status
│   │   ├── downloader.rs    # Model download
│   │   └── models.toml      # Model registry
│   ├── image/               # Image utilities
│   └── preset/              # Preset management
└── assets/                  # Application assets
```

## License

MIT License

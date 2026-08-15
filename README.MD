# PixelForge

ML-enhanced pixel art style transfer for character portraits.

A high-quality pixel art generator that uses machine learning to translate 3D
depth cues into flat color regions and perceptual edges into crisp,
single-pixel-wide line segments.

## Features

- **Edge Pipeline Overhaul** (Phase 2): DexiNed edge map normalized via
  percentile (5th/95th) instead of min-max; max-pooled + 3×3 NMS for
  sparse-line preservation; Zhang-Suen skeletonization produces true 1px
  lines; hysteresis band widened (`low = 0.4× threshold`,
  `high = threshold`); local-contrast outline color picks the palette entry
  furthest in Lab ΔE from each edge pixel's 3×3 neighborhood mean; true
  `EdgeMode::Internal` vs `Outlines` distinction via SLIC label span;
  edge-direction-aware AA (no more full-image blur); centered outline
  thickness.
- **Depth Signal Architecture** (Phase 3): hierarchical depth normalization
  — global signal (percentile 2nd/98th) + local signal (per-region MAD),
  summed with separate weights so both survive; region-based background
  classification per SLIC cluster (mean depth > p70 AND touches border OR
  size > 15%); 10D SLIC feature vector with gradient terms
  `(L, a, b, depth, dL/dx, dL/dy, dDepth/dx, dDepth/dy, x·s, y·s)`;
  region-growing post-processing splits disconnected cluster components.
- **Downsampling & Smudge Fix** (Phase 4): bilateral pre-filter before
  palette snap (eliminates noisy alternation on smooth gradients);
  Lab-space averaging in `Weighted` method (preserves hue instead of
  producing muddy purple from red+blue); importance weight cap at 2.0×
  (was 5.0×); new `PerceptualDither` method (area-average + Floyd-Steinberg
  error diffusion in Lab); default method now `PaletteMode` (aligns with
  documentation); stack-allocated `[u32; 256]` counters instead of
  per-pixel `HashMap`.
- **Performance Pass** (Phase 6): cached median filter in `MLResults`
  (1.56×–1.81× speedup alone); SLIC 2S×2S windowed search (~60× reduction
  on assignment loop); k-means subsampling to 16K random samples;
  `apply_palette` uses cached `PaletteLab`; `Vec` instead of `HashMap`
  for per-region shading + bg classification; `Cow<DynamicImage>` skips
  clones when ML absent or transforms are no-op; skips redundant
  `apply_palette` pass for `PaletteMode`/`PerceptualDither`.
- **Smart Downsampling**: 5 methods (PaletteMode, PerceptualDither,
  Weighted, NearestNeighbor, Bilinear)
- **Depth-to-Flat Conversion**: hierarchical global+local depth shading
  with region-based background classification
- **Edge Control**: skeletonized 1px lines, local-contrast color,
  per-mode (`Outlines`/`Internal`/`Both`) classification, directional AA,
  centered thickness 1–4
- **Palette Management**: Auto (k-means with subsampling), preset palettes
  (Game Boy, Game Boy Color, NES, PICO-8, DawnBringer32, AAP64), custom
  colors, hybrid mode
- **ML Analysis**: Depth estimation (Depth-Anything V2) and edge detection
  (DexiNed); both optional with Sobel fallback
- **Graceful Degradation**: If a model isn't downloaded or fails to load,
  the pipeline falls back to Sobel-based placeholders — the app remains
  fully functional
- **Contact Sheet Export**: One-click export of all pipeline stages (input,
  depth, edges, SLIC, post-DTF, output) as a single PNG for review

## Requirements

- Rust 1.70 or later
- ONNX Runtime (automatically downloaded via the `ort` crate — CPU-only by
  default; re-add the `cuda` feature in `Cargo.toml` if GPU acceleration is
  needed and the workspace has a CUDA toolkit install)
- 2GB RAM free space
- VRAM not required (CPU inference is the default)

## Building

```bash
# Clone the repository
cd pixelforge

# Build in release mode
cargo build --release

# Run
cargo run --release
```

> **Note**: The first build downloads the ONNX Runtime shared library
> (~50 MB). Subsequent builds reuse the cached download.

## Usage

1. **Load an Image**: Drag and drop an image or click "Browse..." in the
   preview area
2. **Run ML Analysis**: Click "Run ML Analysis" to generate depth and edge
   maps (cached; reused across all subsequent pipeline runs until the input
   image changes)
3. **Adjust Settings**: Modify depth-to-flat, SLIC, edges, palette, and
   downsampling settings via the side panels. Each section has a "↺ Reset"
   button to restore defaults
4. **Process**: Click "Process" to generate the pixel art. Use the preview
   tabs to inspect each pipeline stage:
   - 📷 **Original** — input image
   - 🤖 **ML Maps** — depth (turbo), edges (grayscale), SLIC regions
     (colored); 1:1 native-zoom toggle for pixel-accurate inspection
   - 🎨 **Flat (post-DTF)** — image after depth-to-flat conversion
   - 📐 **Preprocessed** — image after geometric transforms
   - ✨ **Output** — final pixel art
5. **Export**: Save the result as PNG or JPEG; or use "Export Contact Sheet…"
   to save all 6 stages as a single PNG

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

1. **ML inference** (cached) — Depth-Anything V2 + DexiNed run on the
   original image. Depth map is median-filtered (5×5) once and cached in
   `MLResults.filtered_depth_map` for reuse across pipeline runs.
2. **Depth → flat** — hierarchical depth normalization:
   - Global signal: percentile (2nd/98th) normalization, preserves relative
     depth between all pixels.
   - Local signal: per-SLIC-cluster MAD normalization, preserves fine
     detail within each region.
   - Blend: `shading = global * gw + local * lw`, clamped to [-1, 1].
     Default `gw=0.6, lw=0.4` (global-biased).
   - Background classification: per-SLIC-cluster (mean depth > p70 AND
     touches_border OR size > 15%). Background pixels get desaturated +
     darkened; foreground pixels get contrast curve + Lab L* bias.
3. **Geometric transforms** — scale, rotate, offset, flip. Skipped
   entirely when no transform is configured (returns a borrow).
4. **ML map resampling** — depth (bilinear), edge (nearest), SLIC labels
   (nearest) resampled to post-transform dimensions.
5. **Importance map** — Sobel + depth-gradient weighted (cap 2.0×).
6. **Palette extraction** — k-means in Lab on a 16K-pixel subsample.
7. **Downsample** — one of:
   - `PaletteMode` (default): bilateral pre-filter → palette snap →
     mode-pick per block. Crisp, no smudge.
   - `PerceptualDither`: area-average → Floyd-Steinberg error diffusion
     in Lab. Clean gradients via dithering. Best for small palettes.
   - `Weighted`: Lab-space weighted average using importance map.
   - `NearestNeighbor`: picks center pixel.
   - `Bilinear`: smooth interpolation.
8. **Palette quantization** — skipped for `PaletteMode`/`PerceptualDither`
   (already quantized); applied for other methods.
9. **Edge rendering** — skeletonized 1px lines drawn at correct location
   with local-contrast color. `EdgeMode::Outlines` keeps only silhouette
   edges (span ≥2 SLIC labels in 3×3 neighborhood); `Internal` keeps only
   within-region edges; `Both` keeps all. Optional edge-direction AA.

## Configuration

### ML Analysis

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Depth Estimation | on | toggle | Enable Depth-Anything V2 |
| Edge Detection | on | toggle | Enable DexiNed |
| Edge threshold | 0.3 | 0.05–0.80 | DexiNed inference threshold (post-percentile-norm) |

### Depth → Flat

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Strength | 0.6 | 0–1 | Shading intensity (Lab L* bias multiplier) |
| Gamma | 0.8 | 0.5–2.0 | Contrast curve exponent |
| MAD threshold | 0.02 | 0.01–0.4 | Low-variance region skip |
| Global depth | 0.6 | 0–1 | Blend weight: 0=pure local, 1=pure global |
| Auto threshold | on | toggle | Otsu fallback when SLIC unavailable |
| BG threshold | 0.6 | 0.1–0.95 | Manual depth threshold (when Auto off) |
| BG desaturate | 0.65 | 0–1 | Background saturation reduction |
| BG darken | −0.08 | −0.5–0.2 | Background L* shift |
| BG cluster size | 0.15 | 0–1 | Min cluster size as fraction of image (bg fallback) |

### Regions (SLIC)

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Clusters (K) | 5 | 3–8 | Number of superpixel regions (pre-split) |
| Spatial weight | 0.7 | 0–1 | Blobbiness vs. detail fidelity |

SLIC uses a 10D feature vector `(L, a, b, depth, dL/dx, dL/dy, dDepth/dx,
dDepth/dy, x·s, y·s)` and a 2S×2S windowed search. After k-means converges,
disconnected cluster components are split into separate IDs.

### Edges

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Mode | Outlines | None/Outlines/Internal/Both | Edge detection mode |
| Thickness | 1 | 1–4 | Outline pixel width (centered) |
| Darkener | 0.3 | 0–1 | (Legacy — currently unused in Phase 2+ pipeline) |
| Anti-alias | off | toggle | Edge-direction-aware AA (only along line normal) |
| Outline style | AutoContrast | AutoContrast/Black | Local-contrast or fixed darkest |
| Edge threshold | 0.3 | 0.05–0.8 | Edge probability threshold |

### Output / Transform

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Output size | 32 | 16–256 | Square output dimension (or max dim for Preserve mode) |
| Aspect mode | Square | Square/Preserve/Custom | Output aspect ratio handling |
| Downsampling | PaletteMode | 5 methods (see Pipeline Stages) | Downsample algorithm |
| Scale | 1.0 | 0.1–4.0 | Pre-downsample scale factor |
| X / Y offset | 0.0 / 0.0 | −1–1 | Pre-downsample pan |
| Export scale | 1× | 1×/2×/4×/8× | Output PNG upscaling (nearest-neighbor) |

### Palette

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| Mode | Auto | Auto/Preset/Custom/Hybrid | Palette source |
| Max colors | 16 | 2–128 | Auto-mode k-means cluster count |
| Preset | None | GB/GBC/NES/PICO-8/DB32/AAP64 | Preset palette |

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
│   ├── lib.rs               # Library crate (exposes processing, ml, image, models, preset)
│   ├── config.rs            # Configuration management
│   ├── app/                 # egui application
│   │   ├── state.rs         # Application state (incl. cached ML textures)
│   │   ├── panels.rs        # Side/bottom panels (sliders, combo boxes, presets)
│   │   ├── preview.rs       # Central preview (5 tabs: Original/ML Maps/Flat/Preprocessed/Output)
│   │   ├── processing.rs    # Pipeline invocation + contact sheet export
│   │   ├── menu.rs          # Menu bar (incl. Export Contact Sheet)
│   │   ├── dialogs.rs       # Modal dialogs
│   │   └── theme.rs         # UI theme
│   ├── processing/          # Image processing pipeline
│   │   ├── pipeline.rs      # Main pipeline orchestration (Cow-based, skip-no-op-transforms)
│   │   ├── depth_to_flat.rs # Hierarchical depth norm + region-based bg classification
│   │   ├── downsampling.rs  # 5 methods: PaletteMode / PerceptualDither / Weighted / Nearest / Bilinear
│   │   ├── edges.rs         # Skeletonized 1px lines + local-contrast color + edge-direction AA
│   │   ├── palette.rs       # PaletteLab cache + k-means subsampling
│   │   ├── slic.rs          # 10D features + 2S×2S windowed search + region growing
│   │   └── config.rs        # Processing config structs
│   ├── ml/                  # Machine learning
│   │   ├── analysis.rs      # ML orchestration (caches median filter)
│   │   ├── types.rs         # MLResults (with filtered_depth_map cache)
│   │   ├── config.rs        # ML config structs
│   │   ├── preprocess.rs    # Tensor preparation + percentile normalization
│   │   ├── session.rs       # ONNX session cache + backend selection
│   │   └── models/
│   │       ├── depth_anything.rs
│   │       └── teed.rs      # DexiNed/TEED-compatible
│   ├── models/              # Model management
│   │   ├── manager.rs       # Model loading/status
│   │   ├── downloader.rs    # Model download
│   │   └── models.toml      # Model registry
│   ├── image/               # Image utilities (incl. contact sheet composition)
│   └── preset/              # Preset management
├── src/bin/                  # Test + benchmark binaries
│   ├── pipeline_test.rs     # 15-case pipeline smoke test
│   ├── determinism_test.rs  # Byte-identical-output verification
│   ├── phase2_smoke.rs      # Edge pipeline (every EdgeMode × OutlineStyle × AA × thickness)
│   ├── phase3_smoke.rs       # Region-based bg vs per-pixel fallback
│   ├── phase4_smoke.rs       # All 5 downsampling methods + GameBoy dithering
│   ├── phase5_smoke.rs       # NaN-safety (Palette::nearest + k-means on solid color)
│   ├── phase6_bench.rs       # Full pipeline timing at 256/512/1024 px
│   ├── phase6_baseline.rs   # Baseline (no P1 cache) vs optimized comparison
│   └── verify_diff.rs       # EdgeMode::Outlines vs Internal produces distinct output
└── assets/                  # Application assets
```

## Performance

Debug-mode timings (512×512 input, 32×32 output, 3-run average) on a
CPU-only workspace:

| Method | Baseline (no P1 cache) | Optimized (P1–P7) | Speedup |
|--------|------------------------|-------------------|---------|
| PaletteMode | 1998 ms | 1278 ms | **1.56×** |
| PerceptualDither | 1611 ms | 894 ms | **1.80×** |
| Weighted | 1758 ms | 969 ms | **1.81×** |

The median-filter cache (P1) alone gives 1.56×–1.81× speedup. SLIC 2S×2S
windowing (P2) and k-means subsampling (P3) contribute further but are
always-on, so they don't show in the baseline-vs-optimized delta.

Run `cargo run --bin phase6_bench` for full timings across 256/512/1024 px
and `cargo run --bin phase6_baseline` for the baseline-vs-optimized
comparison.

## License

MIT License

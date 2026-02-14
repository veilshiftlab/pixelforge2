# PixelForge

ML-enhanced pixel art style transfer for character portraits.

A high-quality pixel art generator that uses machine learning to preserve facial features and translate 3D depth cues into flat color regions.

## Features

- **Smart Downsampling**: Content-aware weighted downsampling that preserves important details
- **Depth-to-Flat Conversion**: Translates 3D depth cues into deliberate flat color bands
- **Feature Preservation**: Guarantees minimum pixel allocation for eyes, lips, and other critical features
- **Edge Control**: Adjustable edge thickness, color modes, and darkening effects
- **Palette Management**: Auto-extraction, preset palettes, and custom color options
- **ML Analysis**: Face detection, depth estimation, and segmentation
- **Low VRAM Mode**: Sequential processing for systems with limited GPU memory

## Requirements

- Rust 1.70 or later
- ONNX Runtime (automatically downloaded)
- 4GB VRAM recommended (2GB minimum with sequential mode)
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
2. **Run ML Analysis**: Click "Run ML Analysis" to detect facial features and depth
3. **Adjust Settings**: Modify depth bands, feature preservation, edges, and palette
4. **Process**: Click "Process" to generate the pixel art
5. **Export**: Save the result as PNG or JPEG

## Model Tiers

### Minimal (Bundled)
- YuNet 2023: Face detection (~5MB)
- MiDaS-small: Depth estimation (~20MB)
- SegFormer-B0: Segmentation (~5MB)
- **Total VRAM**: ~400MB

### High Quality (Downloadable)
- MediaPipe Face Mesh: Better landmarks (~12MB)
- DPT-Large: Superior depth (~340MB)
- SAM-ViT-B: Superior segmentation (~375MB)
- **Total VRAM**: ~2GB

## Configuration

Settings are automatically saved to:
- **Windows**: `%APPDATA%\pixelforge\config.json`
- **macOS**: `~/Library/Application Support/pixelforge/config.json`
- **Linux**: `~/.config/pixelforge/config.json`

## Presets

Built-in presets include:
- **Portrait - Minimal**: Fast processing, essential detail preservation
- **Portrait - Detailed**: Higher quality, more color bands
- **Game Boy Style**: 4-color retro aesthetic

Custom presets can be saved and loaded from the Presets menu.

## Project Structure

```
pixelforge/
├── src/
│   ├── main.rs              # Entry point
│   ├── app.rs               # Application state
│   ├── config.rs            # Configuration management
│   ├── ui/                  # User interface
│   │   ├── panels.rs        # Main UI panels
│   │   ├── controls.rs      # Control widgets
│   │   └── dialogs.rs       # Modal dialogs
│   ├── processing/          # Image processing pipeline
│   │   ├── pipeline.rs      # Main processing pipeline
│   │   ├── depth_to_flat.rs # Depth conversion
│   │   ├── downsampling.rs  # Weighted downsampling
│   │   ├── feature_preserve.rs # Feature preservation
│   │   ├── edges.rs         # Edge detection/drawing
│   │   └── palette.rs       # Color quantization
│   ├── ml/                  # Machine learning
│   │   ├── analysis.rs      # ML orchestration
│   │   ├── face_detection.rs # Face/landmark detection
│   │   ├── depth.rs         # Depth estimation
│   │   └── segmentation.rs  # Image segmentation
│   ├── models/              # Model management
│   │   ├── manager.rs       # Model loading/unloading
│   │   └── downloader.rs    # Model download
│   ├── image/               # Image utilities
│   │   ├── loader.rs        # Image loading
│   │   ├── transform.rs     # Transformations
│   │   └── export.rs        # Export functionality
│   └── preset/              # Preset management
│       └── preset.rs        # Save/load presets
├── models/                  # Bundled models
├── assets/                  # Application assets
└── presets/                 # Built-in presets
```

## License

MIT License

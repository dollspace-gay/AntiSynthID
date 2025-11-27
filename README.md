# AntiSynthID

A proof-of-concept tool demonstrating the fundamental limitations of invisible watermarking systems like Google DeepMind's SynthID.

## Overview

SynthID is Google DeepMind's invisible watermarking system designed to identify AI-generated images. This tool demonstrates that such watermarking approaches have an inherent weakness: diffusion models reconstruct images based on learned natural image distributions, not watermark patterns.

**Purpose**: This project exists to demonstrate that invisible watermarking cannot be a reliable verification mechanism. Users deserve to understand these limitations rather than placing false trust in "watermark verification."

## How It Works

AntiSynthID uses **DiffPure** (Diffusion Purification), a zero-shot purification method:

1. **Encode** the watermarked image to latent space using a VAE encoder
2. **Add controlled noise** to the latent representation (strength parameter controls how much)
3. **Denoise** using unconditional diffusion (Stable Diffusion 1.5 UNet) with DDIM sampling
4. **Decode** back to image space using a VAE decoder
5. **Optional edge guidance** to preserve sharp features

The diffusion model projects the watermarked image back onto the clean data manifold by treating the watermark as unnatural noise, effectively removing it while preserving visual content.

## Installation

### Prerequisites

- Rust 1.70+ ([install](https://rustup.rs/))
- ~4GB disk space for models (downloaded automatically on first run)

### Build from Source

```bash
git clone https://github.com/dollspace-gay/AntiSynthID.git
cd antisynthid
cargo build --release
```

The binary will be at `target/release/antisynthid` (or `antisynthid.exe` on Windows).

## Usage

### Basic Usage

```bash
antisynthid input.jpg output.jpg
```

### Advanced Options

```bash
antisynthid input.jpg output.jpg \
  --strength 0.04 \             # Noise strength (0.0-1.0, default: 0.04)
  --steps 50 \                  # Number of diffusion steps (default: 50)
  --quality 95 \                # JPEG output quality (default: 95)
  --no-guidance \               # Disable edge preservation
  --seed 42 \                   # Set random seed for reproducibility
  --verbose                     # Show detailed progress
```

### Parameters

- **`--strength`**: Controls noise level added to the image (research-validated values)
  - `0.04` = **default**, preserves fine details while removing watermarks (recommended)
  - `0.10` = moderate watermark removal with good quality
  - `0.25` = **maximum**, aggressive watermark removal (may introduce artifacts)
  - **Lower values** (0.04-0.10) preserve image quality, slower processing
  - **Higher values** (0.15-0.25) stronger watermark removal, faster but more artifacts

- **`--steps`**: Number of denoising iterations
  - More steps = higher quality but slower
  - Default: 50 (recommended quality/speed balance)
  - Try 25 for faster processing or 100 for maximum quality

- **`--quality`**: JPEG output quality (1-100)
  - Default: 95 (high quality)

- **`--no-guidance`**: Disables edge-based spatial guidance
  - Edge guidance preserves sharp features but adds processing time
  - Use this flag for faster processing if quality isn't critical

### Examples

**Default (recommended)** - preserves image quality:
```bash
antisynthid input.jpg output.jpg
```

**Faster processing** (moderate quality):
```bash
antisynthid input.jpg output.jpg --strength 0.10 --steps 25
```

**Aggressive watermark removal** (may reduce quality):
```bash
antisynthid input.jpg output.jpg --strength 0.25 --steps 50
```

**Reproducible output**:
```bash
antisynthid input.jpg output.jpg --seed 42
```

## Performance

First run will download ~4GB of ONNX models from Hugging Face:
- VAE Encoder: ~140 MB
- VAE Decoder: ~198 MB
- UNet: ~3.5 GB

**Processing time** (after models are cached):
- Model loading: ~8 seconds
- Processing per image: ~60-150 seconds (depends on `--strength` and `--steps`)
  - `strength=0.04`: ~90 seconds (50 steps)
  - `strength=0.10`: ~140 seconds (50 steps)
  - `strength=0.25`: ~20 seconds (50 steps)

**Hardware requirements**:
- CPU: Any modern multi-core processor
- RAM: 8GB minimum, 16GB recommended
- GPU: Not required (CPU-only inference via ONNX Runtime)

## Technical Details

### Architecture

- **Language**: Rust
- **ML Runtime**: ONNX Runtime 2.0
- **Models**: Stable Diffusion 1.5 (ONNX format)
  - Source: [`modularai/stable-diffusion-1.5-onnx`](https://huggingface.co/modularai/stable-diffusion-1.5-onnx)
- **Sampling**: DDIM (Denoising Diffusion Implicit Models)
- **Scheduler**: Linear beta schedule (β_start=0.00085, β_end=0.012)

### Pipeline Components

1. **Image Loading** ([`src/image/load.rs`](src/image/load.rs))
   - Resize to 512×512 (SD 1.5 native resolution)
   - Normalize to [-1, 1] range
   - Convert to NCHW tensor format

2. **VAE Encoding** ([`src/pipeline/vae.rs`](src/pipeline/vae.rs))
   - Encode image to 4-channel latent space (1×4×64×64)
   - Scale by VAE factor (0.18215)

3. **Diffusion Loop** ([`src/pipeline/diffusion.rs`](src/pipeline/diffusion.rs))
   - Add gaussian noise based on strength parameter (strength * 1000 timesteps)
   - Iterative denoising with DDIM sampling using **unconditional** diffusion
   - Proper timestep scheduling (1000-step space, partial traversal)

4. **VAE Decoding** ([`src/pipeline/vae.rs`](src/pipeline/vae.rs))
   - Decode latents back to RGB image
   - Denormalize to [0, 255] range

5. **Edge Guidance** ([`src/pipeline/guidance.rs`](src/pipeline/guidance.rs)) (optional)
   - Canny edge detection on original image
   - Blend regenerated image with original at edge locations
   - Preserves sharp features while removing watermarks

### Model Cache

Models are cached in platform-specific directories:
- **Windows**: `%LOCALAPPDATA%\antisynthid\models\`
- **Linux**: `~/.cache/antisynthid/models/`
- **macOS**: `~/Library/Caches/antisynthid/models/`

## Limitations

1. **Quality vs. Effectiveness Tradeoff**:
   - Lower strength values (0.04-0.10) preserve image quality but may leave traces
   - Higher values (0.15-0.25) remove watermarks more aggressively but can introduce artifacts
   - **Recommended**: Start with default 0.04 and increase only if needed

2. **Processing Time**: Each image takes 60-150 seconds to process depending on strength. Not suitable for real-time applications. Lower strength values require more denoising steps, making them slower.

3. **Image Size**: All images are processed at 512×512 internally (SD 1.5 limitation). Output is resized back to original dimensions.

4. **Not Perfect**: This demonstrates a fundamental weakness in invisible watermarking, but:
   - Results vary depending on watermark strength and implementation
   - Some artifacts may appear, especially at higher strength values
   - This is a proof-of-concept, not production-quality software
   - The goal is demonstrating vulnerability, not creating a flawless attack tool

## Ethical Considerations

This tool is released for **educational and research purposes** to demonstrate that:

1. **Invisible watermarking is not a reliable verification mechanism**
2. **Users deserve to understand these limitations**
3. **Technical solutions to content provenance require different approaches**

**Responsible use**: This tool should be used to understand watermarking limitations, not to enable misinformation or copyright infringement.

## Development

### Project Structure

```
antisynthid/
├── src/
│   ├── lib.rs              # Library exports
│   ├── main.rs             # CLI interface
│   ├── error.rs            # Error types
│   ├── image/              # Image processing
│   │   ├── load.rs         # Loading and normalization
│   │   └── save.rs         # Saving and denormalization
│   ├── model/              # Model management
│   │   └── loader.rs       # Download and caching
│   └── pipeline/           # Core pipeline
│       ├── vae.rs          # VAE encode/decode
│       ├── diffusion.rs    # DDIM sampling
│       └── guidance.rs     # Edge preservation
├── Cargo.toml              # Dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

### Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Check code quality
cargo clippy --all-targets
```

### Code Quality

The project enforces strict code quality standards:
- **No unsafe code** (`unsafe_code = "deny"`)
- **Clippy lints**: pedantic + nursery warnings enabled
- **Error handling**: All errors use `thiserror` with proper context
- **No panics**: Result types throughout, no unwrap() in production code

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Stable Diffusion**: Original SD 1.5 model by RunwayML and Stability AI
- **ONNX Models**: Public ONNX conversion by [`modularai`](https://huggingface.co/modularai)
- **ONNX Runtime**: Microsoft's ONNX Runtime for efficient inference
- **Research**: This work is inspired by academic research on watermark robustness

## References

- [SynthID: Google DeepMind's watermarking system](https://deepmind.google/technologies/synthid/)
- [DiffPure: Diffusion Purification Against Adversarial Attacks](https://arxiv.org/abs/2205.07460)
- [First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge](https://arxiv.org/abs/2508.21072)
- [DDIM Paper: Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)

---

**Disclaimer**: This is a research tool demonstrating fundamental limitations of invisible watermarking. Use responsibly and ethically.

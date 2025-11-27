//! Diffusion-based watermark removal pipeline.

mod diffusion;
mod guidance;
mod vae;

pub use diffusion::{Config, Pipeline};

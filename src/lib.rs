//! # `AntiSynthID`
//!
//! A library for removing `SynthID` watermarks from images using diffusion-based regeneration.
//!
//! `SynthID` is Google `DeepMind`'s invisible watermarking system for AI-generated images.
//! This library exploits the fundamental weakness that diffusion models reconstruct images
//! based on learned natural image distributions, not watermark patterns.
//!
//! ## Example
//!
//! ```no_run
//! use antisynthid::{Pipeline, Config};
//!
//! # fn main() -> antisynthid::Result<()> {
//! let config = Config::default();
//! let mut pipeline = Pipeline::new(config)?;
//!
//! pipeline.process("watermarked.png", "clean.png")?;
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod image;
pub mod model;
pub mod pipeline;

pub use error::{Error, Result};
pub use pipeline::{Config, Pipeline};

//! Custom error types for antisynthid.

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for the antisynthid library.
#[derive(Error, Debug)]
pub enum Error {
    /// Failed to load an image file.
    #[error("failed to load image from {path}: {source}")]
    ImageLoad {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    /// Failed to save an image file.
    #[error("failed to save image to {path}: {source}")]
    ImageSave {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    /// Image dimensions are not supported.
    #[error("unsupported image dimensions {width}x{height}: {reason}")]
    UnsupportedDimensions {
        width: u32,
        height: u32,
        reason: String,
    },

    /// Failed to download a model.
    #[error("failed to download model {name}: {source}")]
    ModelDownload {
        name: String,
        #[source]
        source: reqwest::Error,
    },

    /// Failed to load an ONNX model.
    #[error("failed to load ONNX model {name}: {source}")]
    ModelLoad {
        name: String,
        #[source]
        source: ort::Error,
    },

    /// Model inference failed.
    #[error("model inference failed: {source}")]
    Inference {
        #[source]
        source: ort::Error,
    },

    /// Failed to create cache directory.
    #[error("failed to create cache directory {path}: {source}")]
    CacheDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Invalid parameter value.
    #[error("invalid parameter {name}: {reason}")]
    InvalidParameter { name: String, reason: String },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Shape mismatch in tensor operations.
    #[error("tensor shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

/// Result type alias for antisynthid operations.
pub type Result<T> = std::result::Result<T, Error>;

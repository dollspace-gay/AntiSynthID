//! Model downloading and loading utilities.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use ort::session::Session;

use crate::error::{Error, Result};

/// Types of models used in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// VAE Encoder - encodes images to latent space.
    VaeEncoder,
    /// VAE Decoder - decodes latents to images.
    VaeDecoder,
    /// `UNet` - performs the denoising diffusion.
    Unet,
}

impl ModelType {
    /// Get the filename for this model type.
    #[must_use]
    pub const fn filename(&self) -> &'static str {
        match self {
            Self::VaeEncoder => "vae_encoder.onnx",
            Self::VaeDecoder => "vae_decoder.onnx",
            Self::Unet => "unet.onnx",
        }
    }

    /// Get the download URL for this model type.
    /// Using modularai's ONNX exports of Stable Diffusion 1.5 (publicly available).
    #[must_use]
    pub const fn url(&self) -> &'static str {
        match self {
            Self::VaeEncoder => {
                "https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/vae_encoder/model.onnx"
            }
            Self::VaeDecoder => {
                "https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/vae_decoder/model.onnx"
            }
            Self::Unet => {
                "https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/unet/model.onnx"
            }
        }
    }

    /// Get the approximate size in bytes for progress indication.
    #[must_use]
    pub const fn approx_size(&self) -> u64 {
        match self {
            Self::VaeEncoder => 140_000_000,   // ~140 MB
            Self::VaeDecoder => 100_000_000,   // ~100 MB
            Self::Unet => 1_300_000,           // ~1.3 MB (main file, data file is separate)
        }
    }

    /// Get the external data file URL if this model has one.
    #[must_use]
    pub const fn data_url(&self) -> Option<&'static str> {
        match self {
            Self::VaeEncoder | Self::VaeDecoder => None,
            Self::Unet => Some(
                "https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/unet/model.onnx_data",
            ),
        }
    }

    /// Get the data filename for models with external data.
    /// Note: Must match the reference in the ONNX file (usually `model.onnx_data`).
    #[must_use]
    pub const fn data_filename(&self) -> Option<&'static str> {
        match self {
            Self::VaeEncoder | Self::VaeDecoder => None,
            // ONNX file references this exact name internally
            Self::Unet => Some("model.onnx_data"),
        }
    }

    /// Get the approximate size of the data file in bytes.
    #[must_use]
    pub const fn data_approx_size(&self) -> u64 {
        match self {
            Self::VaeEncoder | Self::VaeDecoder => 0,
            Self::Unet => 3_500_000_000, // ~3.5 GB
        }
    }
}

/// Manages the model cache directory and downloads.
pub struct ModelCache {
    cache_dir: PathBuf,
}

impl ModelCache {
    /// Create a new model cache.
    ///
    /// Uses the platform-appropriate cache directory:
    /// - Windows: `%LOCALAPPDATA%\antisynthid\models`
    /// - Linux: `~/.cache/antisynthid/models`
    /// - macOS: `~/Library/Caches/antisynthid/models`
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be created.
    pub fn new() -> Result<Self> {
        let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
        let cache_dir = base.join("antisynthid").join("models");

        fs::create_dir_all(&cache_dir).map_err(|source| Error::CacheDir {
            path: cache_dir.clone(),
            source,
        })?;

        Ok(Self { cache_dir })
    }

    /// Get the path to a model file, downloading if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded or accessed.
    pub fn get_model_path(&self, model_type: ModelType) -> Result<PathBuf> {
        let path = self.cache_dir.join(model_type.filename());

        if !path.exists() {
            download_model(model_type, &path)?;
        }

        // Download external data file if needed
        if let (Some(data_url), Some(data_filename)) =
            (model_type.data_url(), model_type.data_filename())
        {
            let data_path = self.cache_dir.join(data_filename);
            if !data_path.exists() {
                download_file(
                    data_url,
                    &data_path,
                    data_filename,
                    model_type.data_approx_size(),
                )?;
            }
        }

        Ok(path)
    }

    /// Load an ONNX model session.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn load_session(&self, model_type: ModelType) -> Result<Session> {
        let path = self.get_model_path(model_type)?;

        Session::builder()
            .map_err(|source| Error::ModelLoad {
                name: model_type.filename().to_string(),
                source,
            })?
            .commit_from_file(&path)
            .map_err(|source| Error::ModelLoad {
                name: model_type.filename().to_string(),
                source,
            })
    }
}

/// Download a model to the specified path.
fn download_model(model_type: ModelType, path: &Path) -> Result<()> {
    download_file(
        model_type.url(),
        path,
        model_type.filename(),
        model_type.approx_size(),
    )
}

/// Download a file from a URL to a path with progress indication.
#[allow(clippy::cast_possible_truncation)]
fn download_file(url: &str, path: &Path, name: &str, approx_size: u64) -> Result<()> {
    tracing::info!("Downloading {name} from {url}");

    let client = reqwest::blocking::Client::new();
    let response = client.get(url).send().map_err(|source| Error::ModelDownload {
        name: name.to_string(),
        source,
    })?;

    let total_size = response.content_length().unwrap_or(approx_size);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid template")
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading {name}"));

    // Write to a temporary file first, then rename for atomicity
    let temp_path = path.with_extension("tmp");
    let mut file = fs::File::create(&temp_path)?;

    let mut downloaded = 0u64;
    let mut reader = response;

    loop {
        let mut buffer = [0u8; 8192];
        let bytes_read = std::io::Read::read(&mut reader, &mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message(format!("Downloaded {name}"));

    // Atomic rename
    fs::rename(&temp_path, path)?;

    Ok(())
}

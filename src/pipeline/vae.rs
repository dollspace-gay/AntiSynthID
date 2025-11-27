//! Variational Autoencoder operations for encoding/decoding images.

use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{Error, Result};
use crate::image::ImageTensor;

/// Latent tensor type (1, 4, 64, 64) for 512x512 images.
pub type LatentTensor = Array4<f32>;

/// VAE scaling factor (from Stable Diffusion).
const VAE_SCALE: f32 = 0.18215;

/// Encode an image to latent space using the VAE encoder.
///
/// # Arguments
///
/// * `encoder` - ONNX session for the VAE encoder
/// * `image` - Image tensor in NCHW format with values in [-1, 1]
///
/// # Returns
///
/// Latent tensor in NCHW format (1, 4, 64, 64).
///
/// # Errors
///
/// Returns an error if inference fails.
pub fn encode(encoder: &mut Session, image: &ImageTensor) -> Result<LatentTensor> {
    let input_value =
        Tensor::from_array(image.clone()).map_err(|source| Error::Inference { source })?;

    let outputs = encoder
        .run(ort::inputs![input_value])
        .map_err(|source| Error::Inference { source })?;

    // Get first output
    let output = outputs
        .values()
        .next()
        .ok_or_else(|| Error::ShapeMismatch {
            expected: "latent_sample output".to_string(),
            actual: "no output".to_string(),
        })?;

    let latent = extract_array4(&output)?;

    // Scale latents as per SD convention
    Ok(latent * VAE_SCALE)
}

/// Decode latents back to image space using the VAE decoder.
///
/// # Arguments
///
/// * `decoder` - ONNX session for the VAE decoder
/// * `latent` - Latent tensor in NCHW format (1, 4, 64, 64)
///
/// # Returns
///
/// Image tensor in NCHW format with values in [-1, 1].
///
/// # Errors
///
/// Returns an error if inference fails.
pub fn decode(decoder: &mut Session, latent: &LatentTensor) -> Result<ImageTensor> {
    // Unscale latents
    let unscaled = latent / VAE_SCALE;

    let input_value =
        Tensor::from_array(unscaled).map_err(|source| Error::Inference { source })?;

    let outputs = decoder
        .run(ort::inputs![input_value])
        .map_err(|source| Error::Inference { source })?;

    // Get first output
    let output = outputs
        .values()
        .next()
        .ok_or_else(|| Error::ShapeMismatch {
            expected: "sample output".to_string(),
            actual: "no output".to_string(),
        })?;

    extract_array4(&output)
}

/// Extract a 4D array from an ONNX value.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn extract_array4(value: &ort::value::ValueRef<'_>) -> Result<Array4<f32>> {
    let (shape_info, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|source| Error::Inference { source })?;

    // Safe: tensor dimensions are always non-negative and within bounds
    let dims: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();

    if dims.len() != 4 {
        return Err(Error::ShapeMismatch {
            expected: "4D tensor".to_string(),
            actual: format!("{}D tensor", dims.len()),
        });
    }

    Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data.to_vec()).map_err(|_| {
        Error::ShapeMismatch {
            expected: format!("{dims:?}"),
            actual: "reshape failed".to_string(),
        }
    })
}

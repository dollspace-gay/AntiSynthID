//! Spatial guidance for preserving image structure during regeneration.

use image::{GrayImage, Luma};
use imageproc::edges::canny;
use ndarray::Array4;

use crate::image::{ImageTensor, SD_IMAGE_SIZE};

/// Edge map tensor type.
pub type EdgeTensor = Array4<f32>;

/// Extract edges from an image tensor for spatial guidance.
///
/// Uses Canny edge detection to identify strong edges that should be preserved
/// during the diffusion process.
///
/// # Arguments
///
/// * `image` - Image tensor in NCHW format with values in [-1, 1]
/// * `low_threshold` - Low threshold for Canny edge detection
/// * `high_threshold` - High threshold for Canny edge detection
///
/// # Returns
///
/// Edge map tensor with values in [0, 1].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn extract_edges(image: &ImageTensor, low_threshold: f32, high_threshold: f32) -> EdgeTensor {
    let size = SD_IMAGE_SIZE as usize;

    // Convert to grayscale
    let mut gray = GrayImage::new(SD_IMAGE_SIZE, SD_IMAGE_SIZE);
    for y in 0..size {
        for x in 0..size {
            let r = (image[[0, 0, y, x]] + 1.0).mul_add(127.5, 0.0);
            let g = (image[[0, 1, y, x]] + 1.0).mul_add(127.5, 0.0);
            let b = (image[[0, 2, y, x]] + 1.0).mul_add(127.5, 0.0);

            // Standard luminosity formula: 0.299*R + 0.587*G + 0.114*B
            // Safe: clamped to [0, 255] before casting
            let luma = 0.299_f32.mul_add(r, 0.587_f32.mul_add(g, 0.114 * b)).clamp(0.0, 255.0) as u8;
            gray.put_pixel(x as u32, y as u32, Luma([luma]));
        }
    }

    // Apply Canny edge detection
    let edges = canny(&gray, low_threshold, high_threshold);

    // Convert to tensor
    let mut tensor = Array4::<f32>::zeros((1, 1, size, size));
    for y in 0..size {
        for x in 0..size {
            tensor[[0, 0, y, x]] = f32::from(edges.get_pixel(x as u32, y as u32)[0]) / 255.0;
        }
    }

    tensor
}

/// Blend the regenerated image with the original based on edge guidance.
///
/// Areas with strong edges are preserved more from the original image,
/// while smoother areas take from the regenerated image.
///
/// # Arguments
///
/// * `original` - Original image tensor
/// * `regenerated` - Regenerated image tensor from diffusion
/// * `edges` - Edge map from `extract_edges`
/// * `edge_strength` - How much to preserve edges (0.0-1.0)
///
/// # Returns
///
/// Blended image tensor.
#[allow(clippy::suboptimal_flops)]
pub fn blend_with_edges(
    original: &ImageTensor,
    regenerated: &ImageTensor,
    edges: &EdgeTensor,
    edge_strength: f32,
) -> ImageTensor {
    let size = SD_IMAGE_SIZE as usize;
    let mut result = regenerated.clone();

    for y in 0..size {
        for x in 0..size {
            let edge_weight = edges[[0, 0, y, x]] * edge_strength;
            for c in 0..3 {
                result[[0, c, y, x]] = original[[0, c, y, x]]
                    .mul_add(edge_weight, regenerated[[0, c, y, x]] * (1.0 - edge_weight));
            }
        }
    }

    result
}

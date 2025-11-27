//! Image saving utilities.

use std::path::Path;

use image::{imageops::FilterType, ImageBuffer, Rgb};

use crate::error::{Error, Result};

use super::{ImageTensor, SD_IMAGE_SIZE};

/// Save a tensor as an image file.
///
/// The tensor is:
/// 1. Denormalized from [-1, 1] to [0, 255]
/// 2. Resized to the original dimensions if provided
/// 3. Saved to the specified path (format inferred from extension)
///
/// # Arguments
///
/// * `tensor` - NCHW tensor with values in [-1, 1]
/// * `path` - Output file path
/// * `original_dims` - Optional original dimensions to resize to
/// * `quality` - JPEG quality (1-100), ignored for other formats
///
/// # Errors
///
/// Returns an error if the image cannot be saved.
pub fn save_image<P: AsRef<Path>>(
    tensor: &ImageTensor,
    path: P,
    original_dims: Option<(u32, u32)>,
    quality: u8,
) -> Result<()> {
    let path = path.as_ref();

    let img = tensor_to_image(tensor);

    // Resize back to original dimensions if specified
    let final_img = if let Some((width, height)) = original_dims {
        image::DynamicImage::ImageRgb8(img).resize_exact(width, height, FilterType::Lanczos3)
    } else {
        image::DynamicImage::ImageRgb8(img)
    };

    // Determine format and save
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_lowercase();

    match extension.as_str() {
        "jpg" | "jpeg" => {
            let mut output = std::fs::File::create(path)?;
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, quality);
            final_img
                .write_with_encoder(encoder)
                .map_err(|source| Error::ImageSave {
                    path: path.to_path_buf(),
                    source,
                })?;
        }
        _ => {
            final_img.save(path).map_err(|source| Error::ImageSave {
                path: path.to_path_buf(),
                source,
            })?;
        }
    }

    Ok(())
}

/// Convert a normalized NCHW tensor to an RGB image.
#[allow(clippy::cast_possible_truncation)]
fn tensor_to_image(tensor: &ImageTensor) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let height = SD_IMAGE_SIZE as usize;
    let width = SD_IMAGE_SIZE as usize;

    let mut img = ImageBuffer::new(SD_IMAGE_SIZE, SD_IMAGE_SIZE);

    for y in 0..height {
        for x in 0..width {
            // Denormalize from [-1, 1] to [0, 255]
            let r = denormalize(tensor[[0, 0, y, x]]);
            let g = denormalize(tensor[[0, 1, y, x]]);
            let b = denormalize(tensor[[0, 2, y, x]]);

            // Safe: x and y are bounded by SD_IMAGE_SIZE (512) which fits in u32
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img
}

/// Denormalize a value from [-1, 1] to [0, 255] with clamping.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn denormalize(value: f32) -> u8 {
    // Safe: clamped to [0, 255] range before casting
    let scaled = (value + 1.0) * 127.5;
    scaled.clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denormalize() {
        assert_eq!(denormalize(-1.0), 0);
        assert_eq!(denormalize(0.0), 127);
        assert_eq!(denormalize(1.0), 255);
    }

    #[test]
    fn test_denormalize_clamp() {
        assert_eq!(denormalize(-2.0), 0);
        assert_eq!(denormalize(2.0), 255);
    }
}

//! Image loading utilities.

use std::path::Path;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array4;

use crate::error::{Error, Result};

use super::{ImageTensor, RGB_CHANNELS, SD_IMAGE_SIZE};

/// Load an image from disk and convert to a normalized tensor.
///
/// The image is:
/// 1. Loaded from the specified path
/// 2. Resized to 512x512 (SD 1.5 native resolution)
/// 3. Converted to RGB if necessary
/// 4. Normalized to [-1, 1] range
/// 5. Returned as NCHW tensor (1, 3, 512, 512)
///
/// # Errors
///
/// Returns an error if the image cannot be loaded or processed.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<(ImageTensor, (u32, u32))> {
    let path = path.as_ref();

    let img = image::open(path).map_err(|source| Error::ImageLoad {
        path: path.to_path_buf(),
        source,
    })?;

    let original_dims = img.dimensions();

    let tensor = image_to_tensor(&img);

    Ok((tensor, original_dims))
}

/// Convert a `DynamicImage` to a normalized NCHW tensor.
#[allow(clippy::cast_possible_truncation)]
fn image_to_tensor(img: &DynamicImage) -> ImageTensor {
    // Resize to SD native resolution using Lanczos3 for quality
    let resized = img.resize_exact(SD_IMAGE_SIZE, SD_IMAGE_SIZE, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let (width, height) = (SD_IMAGE_SIZE as usize, SD_IMAGE_SIZE as usize);

    // Create tensor in NCHW format
    let mut tensor = Array4::<f32>::zeros((1, RGB_CHANNELS, height, width));

    for y in 0..height {
        for x in 0..width {
            // Safe: x and y are bounded by SD_IMAGE_SIZE (512) which fits in u32
            let pixel = rgb.get_pixel(x as u32, y as u32);
            // Normalize from [0, 255] to [-1, 1]
            tensor[[0, 0, y, x]] = (f32::from(pixel[0]) / 127.5) - 1.0;
            tensor[[0, 1, y, x]] = (f32::from(pixel[1]) / 127.5) - 1.0;
            tensor[[0, 2, y, x]] = (f32::from(pixel[2]) / 127.5) - 1.0;
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape() {
        // Create a simple test image in memory
        let img = DynamicImage::new_rgb8(100, 100);
        let tensor = image_to_tensor(&img);

        assert_eq!(tensor.shape(), &[1, 3, 512, 512]);
    }

    #[test]
    fn test_normalization_range() {
        let img = DynamicImage::new_rgb8(100, 100);
        let tensor = image_to_tensor(&img);

        let min = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Black image should be all -1.0
        assert!((min - (-1.0)).abs() < 0.01);
        assert!((max - (-1.0)).abs() < 0.01);
    }
}

//! Image loading, processing, and saving utilities.

mod load;
mod save;

pub use load::load_image;
pub use save::save_image;

use ndarray::Array4;

/// Image tensor in NCHW format (batch, channels, height, width).
/// Values are normalized to [-1, 1] range for diffusion model compatibility.
pub type ImageTensor = Array4<f32>;

/// Standard image size for Stable Diffusion 1.5.
pub const SD_IMAGE_SIZE: u32 = 512;

/// Number of channels in RGB images.
pub const RGB_CHANNELS: usize = 3;

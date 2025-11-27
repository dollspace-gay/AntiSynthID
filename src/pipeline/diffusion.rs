//! Main diffusion pipeline for watermark removal.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::image;
use crate::model::{ModelCache, ModelType};

use super::guidance::{blend_with_edges, extract_edges};
use super::vae::{self, LatentTensor};

/// Configuration for the watermark removal pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    /// Denoising strength (0.0-1.0). Higher values remove more watermark but may alter image.
    pub strength: f32,

    /// Number of denoising steps.
    pub num_steps: u32,

    /// Whether to use spatial guidance (edge preservation).
    pub use_guidance: bool,

    /// Edge detection low threshold.
    pub edge_low_threshold: f32,

    /// Edge detection high threshold.
    pub edge_high_threshold: f32,

    /// Edge preservation strength (0.0-1.0).
    pub edge_strength: f32,

    /// Output JPEG quality (1-100).
    pub output_quality: u8,

    /// Random seed for reproducibility. None for random.
    pub seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            strength: 0.4,
            num_steps: 20,
            use_guidance: true,
            edge_low_threshold: 50.0,
            edge_high_threshold: 100.0,
            edge_strength: 0.3,
            output_quality: 95,
            seed: None,
        }
    }
}

impl Config {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of valid range.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(Error::InvalidParameter {
                name: "strength".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        if self.num_steps == 0 {
            return Err(Error::InvalidParameter {
                name: "num_steps".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if !(1..=100).contains(&self.output_quality) {
            return Err(Error::InvalidParameter {
                name: "output_quality".to_string(),
                reason: "must be between 1 and 100".to_string(),
            });
        }

        Ok(())
    }
}

/// Main pipeline for removing `SynthID` watermarks.
pub struct Pipeline {
    config: Config,
    vae_encoder: Session,
    vae_decoder: Session,
    unet: Session,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// This will download models if they are not already cached.
    ///
    /// # Errors
    ///
    /// Returns an error if models cannot be loaded.
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;

        tracing::info!("Initializing pipeline with config: {config:?}");

        let cache = ModelCache::new()?;

        tracing::info!("Loading VAE encoder...");
        let vae_encoder = cache.load_session(ModelType::VaeEncoder)?;

        tracing::info!("Loading VAE decoder...");
        let vae_decoder = cache.load_session(ModelType::VaeDecoder)?;

        tracing::info!("Loading `UNet`...");
        let unet = cache.load_session(ModelType::Unet)?;

        tracing::info!("Pipeline initialized successfully");

        Ok(Self {
            config,
            vae_encoder,
            vae_decoder,
            unet,
        })
    }

    /// Process an image to remove the `SynthID` watermark.
    ///
    /// # Arguments
    ///
    /// * `input_path` - Path to the input image
    /// * `output_path` - Path to save the processed image
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process<P: AsRef<Path>, Q: AsRef<Path>>(
        &mut self,
        input_path: P,
        output_path: Q,
    ) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        tracing::info!("Processing image: {}", input_path.display());

        // Load image
        let (image_tensor, original_dims) = image::load_image(input_path)?;

        // Extract edges for guidance if enabled
        let edges = if self.config.use_guidance {
            Some(extract_edges(
                &image_tensor,
                self.config.edge_low_threshold,
                self.config.edge_high_threshold,
            ))
        } else {
            None
        };

        // Encode to latent space
        tracing::info!("Encoding to latent space...");
        let latents = vae::encode(&mut self.vae_encoder, &image_tensor)?;

        // Add noise and denoise
        tracing::info!("Running diffusion...");
        let denoised_latents = self.diffusion_loop(&latents)?;

        // Decode back to image space
        tracing::info!("Decoding from latent space...");
        let mut output_tensor = vae::decode(&mut self.vae_decoder, &denoised_latents)?;

        // Apply edge guidance if enabled
        if let Some(ref edge_map) = edges {
            output_tensor = blend_with_edges(
                &image_tensor,
                &output_tensor,
                edge_map,
                self.config.edge_strength,
            );
        }

        // Save output
        tracing::info!("Saving output to: {}", output_path.display());
        image::save_image(
            &output_tensor,
            output_path,
            Some(original_dims),
            self.config.output_quality,
        )?;

        tracing::info!("Processing complete");
        Ok(())
    }

    /// Run the diffusion denoising loop.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn diffusion_loop(&mut self, latents: &LatentTensor) -> Result<LatentTensor> {
        let mut rng = self
            .config
            .seed
            .map_or_else(rand::rngs::StdRng::from_os_rng, rand::rngs::StdRng::seed_from_u64);

        // Calculate how much noise to add based on strength
        // Safe: result is clamped to valid step range
        #[allow(clippy::cast_precision_loss)]
        let start_step = ((1.0 - self.config.strength) * self.config.num_steps as f32) as u32;
        let num_inference_steps = self.config.num_steps - start_step;

        if num_inference_steps == 0 {
            return Ok(latents.clone());
        }

        // Add noise proportional to strength
        let noise: Array4<f32> =
            Array4::from_shape_fn(latents.dim(), |_| rng.random::<f32>().mul_add(2.0, -1.0));
        let noise_scale = self.config.strength;
        let mut noisy_latents = latents * (1.0 - noise_scale) + noise * noise_scale;

        // Progress bar for denoising
        let pb = ProgressBar::new(u64::from(num_inference_steps));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} Denoising [{bar:40.cyan/blue}] {pos}/{len}")
                .expect("valid template")
                .progress_chars("#>-"),
        );

        // DDIM-style denoising loop
        // Note: This is a simplified version. Full implementation would use proper
        // DDIM scheduler with timestep embeddings and text conditioning.
        for step in 0..num_inference_steps {
            #[allow(clippy::cast_precision_loss)]
            let t = (num_inference_steps - step) as f32 / num_inference_steps as f32;

            // Predict noise using UNet
            let noise_pred = self.predict_noise(&noisy_latents, t)?;

            // DDIM update step
            let alpha = (-t).mul_add(0.5, 1.0); // Simplified alpha schedule: 1.0 - t * 0.5
            noisy_latents = noisy_latents * alpha
                + (latents.clone() - noise_pred * noise_scale) * (1.0 - alpha);

            pb.inc(1);
        }

        pb.finish_with_message("Denoising complete");
        Ok(noisy_latents)
    }

    /// Predict noise using the `UNet`.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn predict_noise(&mut self, latents: &LatentTensor, _timestep: f32) -> Result<LatentTensor> {
        // For img2img without text, we use unconditional generation
        // The `UNet` expects: sample, timestep, encoder_hidden_states

        let sample_value =
            Tensor::from_array(latents.clone()).map_err(|source| Error::Inference { source })?;

        // Timestep as tensor
        let timestep_arr = Array1::from_vec(vec![500i64]); // Mid-range timestep
        let timestep_value =
            Tensor::from_array(timestep_arr).map_err(|source| Error::Inference { source })?;

        // Empty text embedding (unconditional) - shape is (batch, seq_len, hidden_dim)
        let hidden_states = Array3::<f32>::zeros((1, 77, 768));
        let hidden_value =
            Tensor::from_array(hidden_states).map_err(|source| Error::Inference { source })?;

        let outputs = self
            .unet
            .run(ort::inputs![
                "sample" => sample_value,
                "timestep" => timestep_value,
                "encoder_hidden_states" => hidden_value,
            ])
            .map_err(|source| Error::Inference { source })?;

        let output = outputs
            .values()
            .next()
            .ok_or_else(|| Error::ShapeMismatch {
                expected: "noise prediction output".to_string(),
                actual: "no output".to_string(),
            })?;

        let (shape_info, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|source| Error::Inference { source })?;

        // Safe: tensor dimensions are always non-negative and within bounds
        let dims: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();

        Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data.to_vec()).map_err(|_| {
            Error::ShapeMismatch {
                expected: format!("{dims:?}"),
                actual: "reshape failed".to_string(),
            }
        })
    }
}

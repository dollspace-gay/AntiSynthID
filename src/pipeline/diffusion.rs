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
///
/// Based on `DiffPure` research for optimal `SynthID` watermark removal.
#[derive(Debug, Clone)]
pub struct Config {
    /// Denoising strength (0.0-1.0). Research shows 0.15-0.3 is optimal for watermark removal.
    /// Higher values remove more watermark but may alter image content.
    pub strength: f32,

    /// Number of denoising steps. Research recommends 50 steps for quality/speed balance.
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
            strength: 0.04,        // Research: 0.04 preserves details, 0.25 max for aggressive removal
            num_steps: 50,         // Research: 50-100 steps for quality/speed balance
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
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn diffusion_loop(&mut self, latents: &LatentTensor) -> Result<LatentTensor> {
        // SD 1.5 uses 1000 total timesteps
        const MAX_TIMESTEPS: u32 = 1000;

        let mut rng = self
            .config
            .seed
            .map_or_else(rand::rngs::StdRng::from_os_rng, rand::rngs::StdRng::seed_from_u64);

        // Calculate timestep range based on strength
        // Research shows: lower strength (0.04) = less noise = fine details preserved
        // Higher strength (0.25) = more noise = aggressive watermark removal
        // We add noise UP TO (strength * MAX_TIMESTEPS), not (1-strength)
        let max_timestep = (self.config.strength * MAX_TIMESTEPS as f32) as u32;
        let step_size = (max_timestep.max(1) / self.config.num_steps.min(max_timestep.max(1))) as usize;
        let mut timesteps: Vec<u32> = (0..=max_timestep)
            .step_by(step_size.max(1))
            .collect();
        timesteps.reverse();

        if timesteps.is_empty() {
            return Ok(latents.clone());
        }

        // Add noise to latents based on strength
        let noise: Array4<f32> = Array4::from_shape_fn(latents.dim(), |_| {
            rng.random::<f32>().mul_add(2.0, -1.0)
        });

        // Calculate alpha for initial noise level at max_timestep
        let init_alpha = get_alpha_prod(max_timestep);
        let sqrt_alpha = init_alpha.sqrt();
        let sqrt_one_minus_alpha = (1.0 - init_alpha).sqrt();
        let mut noisy_latents = latents * sqrt_alpha + &noise * sqrt_one_minus_alpha;

        // Progress bar
        let pb = ProgressBar::new(timesteps.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} Denoising [{bar:40.cyan/blue}] {pos}/{len}")
                .expect("valid template")
                .progress_chars("#>-"),
        );

        // DDIM denoising loop
        for (i, &timestep) in timesteps.iter().enumerate() {
            // Predict noise
            let noise_pred = self.predict_noise(&noisy_latents, timestep)?;

            // DDIM step
            let alpha_prod = get_alpha_prod(timestep);
            let alpha_prod_prev = if i < timesteps.len() - 1 {
                get_alpha_prod(timesteps[i + 1])
            } else {
                1.0
            };

            // Predict x0 (original sample)
            let sqrt_alpha_prod = alpha_prod.sqrt();
            let sqrt_one_minus_alpha_prod = (1.0 - alpha_prod).sqrt();
            let pred_original = (&noisy_latents - &noise_pred * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod;

            // Direction pointing to xt
            let sqrt_alpha_prod_prev = alpha_prod_prev.sqrt();
            let sqrt_one_minus_alpha_prod_prev = (1.0 - alpha_prod_prev).sqrt();
            noisy_latents = pred_original * sqrt_alpha_prod_prev + noise_pred * sqrt_one_minus_alpha_prod_prev;

            pb.inc(1);
        }

        pb.finish_with_message("Denoising complete");
        Ok(noisy_latents)
    }

    /// Predict noise using the `UNet` with unconditional diffusion.
    ///
    /// Based on `DiffPure` research: uses pure unconditional generation for watermark removal.
    /// This is a zero-shot purification method that projects watermarked images back onto
    /// the clean data manifold by treating the watermark as unnatural noise.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn predict_noise(&mut self, latents: &LatentTensor, timestep: u32) -> Result<LatentTensor> {
        // The `UNet` expects: sample, timestep, encoder_hidden_states

        let sample_value =
            Tensor::from_array(latents.clone()).map_err(|source| Error::Inference { source })?;

        // Timestep as tensor
        let timestep_arr = Array1::from_vec(vec![i64::from(timestep)]);
        let timestep_value =
            Tensor::from_array(timestep_arr).map_err(|source| Error::Inference { source })?;

        // Unconditional prediction (empty text embedding)
        // This is the correct approach for DiffPure watermark removal
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

/// Get alpha cumulative product for a given timestep.
/// Uses SD 1.5's linear beta schedule: `beta_start=0.00085`, `beta_end=0.012`
#[allow(clippy::cast_precision_loss)]
fn get_alpha_prod(timestep: u32) -> f32 {
    const BETA_START: f32 = 0.000_85;
    const BETA_END: f32 = 0.012;
    const MAX_TIMESTEPS: f32 = 1000.0;

    // Linear interpolation for beta at this timestep
    let t = timestep as f32 / MAX_TIMESTEPS;
    let beta = (BETA_END - BETA_START).mul_add(t, BETA_START);

    // For alpha_prod, we need cumulative product
    // Approximate with exponential for efficiency
    let avg_beta = f32::midpoint(BETA_START, beta);

    (1.0 - avg_beta).powf(timestep as f32)
}

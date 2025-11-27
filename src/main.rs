//! `AntiSynthID` CLI - Remove `SynthID` watermarks from images.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use antisynthid::{Config, Pipeline};

/// Remove `SynthID` watermarks from images using diffusion-based regeneration.
#[derive(Parser, Debug)]
#[command(name = "antisynthid")]
#[command(version, about, long_about = None)]
struct Args {
    /// Input image path.
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output image path.
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,

    /// Noise strength (0.0-1.0). Research shows 0.04 preserves details, 0.25 for aggressive removal.
    #[arg(short, long, default_value = "0.04", value_name = "FLOAT")]
    strength: f32,

    /// Number of denoising steps. More steps = better quality but slower.
    #[arg(long, default_value = "50", value_name = "INT")]
    steps: u32,

    /// Output JPEG quality (1-100).
    #[arg(short, long, default_value = "95", value_name = "INT")]
    quality: u8,

    /// Disable spatial guidance (edge preservation).
    #[arg(long)]
    no_guidance: bool,

    /// Random seed for reproducibility.
    #[arg(long, value_name = "INT")]
    seed: Option<u64>,

    /// Enable verbose output.
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("antisynthid={log_level}").into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    if let Err(err) = run(&args) {
        tracing::error!("{err:#}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run(args: &Args) -> Result<()> {
    // Validate input file exists
    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {}", args.input.display());
    }

    // Build configuration
    let config = Config {
        strength: args.strength,
        num_steps: args.steps,
        use_guidance: !args.no_guidance,
        output_quality: args.quality,
        seed: args.seed,
        ..Config::default()
    };

    // Create and run pipeline
    let mut pipeline = Pipeline::new(config).context("Failed to initialize pipeline")?;

    pipeline
        .process(&args.input, &args.output)
        .context("Failed to process image")?;

    println!(
        "Successfully processed {} -> {}",
        args.input.display(),
        args.output.display()
    );

    Ok(())
}

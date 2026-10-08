use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "qwen3-tts")]
pub struct Args {
    #[arg(short, long, default_value = "model-base")]
    pub model: PathBuf,
    #[arg(short, long)]
    pub text: Option<String>,
    #[arg(short, long)]
    pub voice: Option<PathBuf>,
    #[arg(short, long, default_value = "chinese")]
    pub language: String,
    #[arg(short, long, default_value = "0.8")]
    pub temperature: f32,
    #[arg(short, long)]
    pub stream: bool,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(short, long)]
    pub verbose: bool,
}

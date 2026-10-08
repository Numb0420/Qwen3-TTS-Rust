use clap::{Parser, ValueEnum};
use qwen3_tts::models::onnx::{init_onruntime, AudioEncoder, SpeakerEncoder};
use qwen3_tts::VoiceFile;
use std::path::PathBuf;
use qwen3_tts::speaker_creation::{read_wav, resample_to_24k};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    /// Reference text and audio codes plus the speaker embedding.
    Full,
    /// Speaker embedding only; no reference transcript is needed.
    Xvec,
}

#[derive(Parser, Debug)]
#[command(about = "Extract a reusable speaker from a WAV file")]
struct Args {
    /// Reference WAV (mono or multichannel, 16/24/32-bit PCM or 32-bit float).
    #[arg(long)]
    audio: PathBuf,
    /// Speaker name and default output file stem.
    #[arg(long)]
    name: String,
    /// Full ICL clone or speaker-embedding-only clone.
    #[arg(long, value_enum)]
    mode: Mode,
    /// Exact spoken words in the reference audio; required for full mode.
    #[arg(long, conflicts_with = "ref_text_file")]
    ref_text: Option<String>,
    /// UTF-8 file containing the exact spoken words; required for full mode if --ref-text is absent.
    #[arg(long)]
    ref_text_file: Option<PathBuf>,
    /// Output JSON path (default: speakers/<name>.json).
    #[arg(long)]
    output: Option<PathBuf>,
    /// ONNX model directory.
    #[arg(long, default_value = "models/base-1.7b/onnx")]
    model_dir: PathBuf,
    /// Replace an existing output file.
    #[arg(long)]
    force: bool,
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    if args.name.is_empty()
        || !args
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err("Name must contain only letters, numbers, hyphens or underscores".into());
    }

    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from("speakers").join(format!("{}.json", args.name)));
    if output.exists() && !args.force {
        return Err(format!("Output already exists: {} (use --force to replace)", output.display()));
    }

    let ref_text = match args.mode {
        Mode::Full => {
            let text = if let Some(path) = args.ref_text_file {
                std::fs::read_to_string(&path)
                    .map_err(|e| format!("Cannot read {}: {e}", path.display()))?
            } else {
                args.ref_text.unwrap_or_default()
            };
            let text = text.trim().trim_start_matches('\u{feff}').trim().to_string();
            if text.is_empty() {
                return Err("Full mode requires --ref-text or --ref-text-file".into());
            }
            text
        }
        Mode::Xvec => {
            if args.ref_text.is_some() || args.ref_text_file.is_some() {
                return Err("Xvec mode does not use reference text".into());
            }
            String::new()
        }
    };

    let speaker_path = args.model_dir.join("qwen3_tts_speaker_encoder.onnx");
    if !speaker_path.is_file() {
        return Err(format!("Missing model: {}", speaker_path.display()));
    }
    let codec_path = args.model_dir.join("qwen3_tts_codec_encoder.onnx");
    if matches!(args.mode, Mode::Full) && !codec_path.is_file() {
        return Err(format!("Missing model: {}", codec_path.display()));
    }

    let (audio, source_rate) = read_wav(&args.audio)?;
    let duration = audio.len() as f64 / source_rate as f64;
    if duration < 1.0 {
        return Err("Reference audio should contain at least one second of speech".into());
    }
    if !(3.0..=30.0).contains(&duration) {
        eprintln!("Warning: {:.1}s reference; 3-30s of clean speech usually works best", duration);
    }
    let audio = resample_to_24k(&audio, source_rate);
    println!("Audio: {:.2}s, {} Hz -> 24000 Hz", duration, source_rate);

    init_onruntime().map_err(|e| e.to_string())?;
    let mut speaker_encoder = SpeakerEncoder::load(&speaker_path.to_string_lossy())
        .map_err(|e| format!("Cannot load speaker encoder: {e}"))?;
    let embedding = speaker_encoder
        .encode(&audio)
        .map_err(|e| format!("Speaker extraction failed: {e}"))?;
    if embedding.len() != 2048 || !embedding.iter().all(|value| value.is_finite()) {
        return Err(format!("Invalid speaker embedding ({} dimensions)", embedding.len()));
    }

    let codes = if matches!(args.mode, Mode::Full) {
        let mut codec_encoder = AudioEncoder::load(&codec_path.to_string_lossy())
            .map_err(|e| format!("Cannot load codec encoder: {e}"))?;
        let codes = codec_encoder
            .encode(&audio)
            .map_err(|e| format!("Audio code extraction failed: {e}"))?;
        if codes.is_empty() || codes.len() % 16 != 0 {
            return Err(format!("Invalid audio code count: {}", codes.len()));
        }
        codes
    } else {
        Vec::new()
    };

    let voice = VoiceFile::new(ref_text, codes, embedding).with_metadata(
        Some(args.name),
        None,
        None,
        Some(format!("Extracted from {} ({:?})", args.audio.display(), args.mode)),
    );
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    voice.save(&output)?;
    println!("Saved {}", output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

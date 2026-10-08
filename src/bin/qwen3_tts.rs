use clap::Parser;
use qwen3_tts::SamplerConfig;
use qwen3_tts::TtsEngine;
use qwen3_tts::VoiceFile;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model directory (contains models/assets)
    #[arg(long, default_value = "models/base-1.7b")]
    model_dir: PathBuf,

    /// GGUF quantization for the matched Base bundle.
    #[arg(long, default_value = "q5_k_m")]
    quant: String,

    /// Text to generate
    #[arg(short, long)]
    text: String,

    /// Preset voice file (.json)
    #[arg(short, long)]
    voice_file: Option<PathBuf>,

    /// Reference audio file for feature extraction (.wav)
    #[arg(long)]
    ref_audio: Option<PathBuf>,

    /// Reference text for the record audio
    #[arg(long)]
    ref_text: Option<String>,

    /// Path to save the extracted VoiceFile (.json)
    #[arg(long)]
    save_voice: Option<PathBuf>,

    /// Output audio file
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Maximum generation steps (tokens)
    #[arg(long, default_value_t = 512)]
    max_steps: usize,

    /// Speakers directory
    #[arg(long, default_value = "speakers")]
    speakers_dir: PathBuf,

    /// Cloning speaker name (fallback to 39).
    #[arg(short, long)]
    speaker: Option<String>,

    /// Instruction style (e.g. "Happy", "Sad") - prepended to text
    #[arg(long)]
    instruction: Option<String>,

    /// Temperature for sampling (higher = more random, 0.0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-K sampling (0 = disabled)
    #[arg(long, default_value_t = 40)]
    top_k: i32,

    /// Top-P (nucleus) sampling (1.0 = disabled)
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Random seed for reproducibility (omit for random)
    #[arg(long)]
    seed: Option<u64>,

    /// Number of threads for inference
    #[arg(long, default_value_t = 4)]
    threads: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let start_total = Instant::now();

    println!("=== Qwen3-TTS Example CLI ===");
    println!("Model Dir: {:?}", args.model_dir);
    println!("Quant:     {}", args.quant);
    println!("Threads:   {}", args.threads);
    println!("Voice:     {:?}", args.voice_file);
    println!("Text:      {}", args.text);

    // The cloning CLI must not fill a missing Base file with CustomVoice weights.
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(args.model_dir.join("checkpoint-manifest.json"))
            .map_err(|e| format!("Matched Base checkpoint manifest is required: {e}"))?,
    )?;
    if manifest["source_model"].as_str() != Some("Qwen/Qwen3-TTS-12Hz-1.7B-Base") {
        return Err("This cloning CLI requires the matched 1.7B Base bundle".into());
    }
    let quant_folder = match args.quant.as_str() {
        "q5_k_m" => "gguf_q5_k_m", "q8_0" => "gguf_q8_0", "none" => "gguf",
        _ => return Err("Unsupported quantization".into()),
    };
    for relative in [format!("{quant_folder}/qwen3_tts_talker.gguf"), format!("{quant_folder}/qwen3_tts_predictor.gguf"),
        "gguf/qwen3_assets.gguf".into(), "onnx/qwen3_tts_decoder.onnx".into(), "tokenizer/tokenizer.json".into()] {
        if !args.model_dir.join(&relative).is_file() { return Err(format!("Base file missing: {relative}; automatic checkpoint substitution is disabled").into()); }
    }
    // 0. Runtime check; all required Base synthesis files were verified above.
    println!("Checking models...");
    TtsEngine::download_models(&args.model_dir, &args.quant)
        .await
        .map_err(|e| format!("Model download failed: {}", e))?;

    // 1. Load Engine
    let mut engine = TtsEngine::new(&args.model_dir, &args.quant, args.threads)
        .await
        .map_err(|e| format!("Engine load failed: {}", e))?;
    engine.set_max_steps(args.max_steps);

    // 1.1 Configure sampler
    let sampler_config = SamplerConfig::new(args.temperature, args.top_k, args.top_p, args.seed);
    engine.set_sampler_config(sampler_config);
    println!(
        "Sampler: temp={}, top_k={}, top_p={}, seed={:?}",
        args.temperature, args.top_k, args.top_p, args.seed
    );

    // 1.5 Load custom speakers if dir specified
    if args.speakers_dir.exists() {
        engine.load_speakers(&args.speakers_dir)?;
    }

    // 2. Load or Create Voice File
    let voice = if let Some(audio_path) = args.ref_audio {
        println!("Creating voice from reference: {:?}", audio_path);
        let ref_text = args.ref_text.unwrap_or_default();
        let vf = engine
            .create_voice_file(audio_path, ref_text)
            .map_err(|e| format!("Feature extraction failed: {}", e))?;

        if let Some(save_path) = args.save_voice {
            vf.save(&save_path)
                .map_err(|e| format!("Failed to save voice file: {}", e))?;
            println!("Saved new voice file to: {:?}", save_path);
        }
        vf
    } else if let Some(vf_path) = args.voice_file {
        println!("Loading voice from file: {:?}", vf_path);
        VoiceFile::load(vf_path).map_err(|e| format!("Failed to load voice file: {}", e))?
    } else {
        let spk_id = args.speaker.as_deref().unwrap_or("39");
        println!("Using speaker from cache: {}", spk_id);
        engine.get_speaker(spk_id).clone()
    };

    println!("Voice Name: {}", voice.name.as_deref().unwrap_or("Dynamic"));
    if voice.spk_id.is_some() {
        return Err("Base requires a cloning voice JSON, not a CustomVoice preset ID".into());
    }

    // 3. Generate
    let start_gen = Instant::now();
    println!("Generating...");

    // Generate
    let audio = engine
        .generate_with_voice_streaming(&args.text, &voice, args.instruction.as_deref(), None)
        .map_err(|e| format!("Generation failed: {}", e))?;

    let gen_duration = start_gen.elapsed();
    println!("Generation took: {:.2}s", gen_duration.as_secs_f32());

    let audio_duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    let rtf = gen_duration.as_secs_f32() / audio_duration_secs;
    println!(
        "Audio: {:.2}s ({} samples @ {} Hz), RTF: {:.3}",
        audio_duration_secs,
        audio.samples.len(),
        audio.sample_rate,
        rtf
    );

    // 4. Save
    audio
        .save_wav(&args.output)
        .map_err(|e| format!("Failed to save output: {}", e))?;

    println!("Saved to: {:?}", args.output);
    println!("Total time: {:.2}s", start_total.elapsed().as_secs_f32());

    Ok(())
}

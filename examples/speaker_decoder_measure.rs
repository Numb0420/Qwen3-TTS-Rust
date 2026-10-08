//! Isolated decoder measurements. No live service settings or reference audio changes.
use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder};
use qwen3_tts::VoiceFile;
use std::{path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("QWEN_ONNX_NO_SPIN", "1");
    init_onruntime()?;
    let root = Path::new("diagnostics/speaker-decoder-measure-2026-10-08");
    std::fs::create_dir_all(root)?;
    let mut normal = AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder.onnx")?;
    let mut partial = AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder_state_only.onnx")?;
    let codes = [0i64; 64];
    let mut reports = Vec::new();
    for name in ["39", "xiaolu", "xiaoyan"] {
        let voice = VoiceFile::load(Path::new(&format!("speakers/{name}.json")))?;
        let mut reference = AudioDecoder::create_state();
        let mut emitted = 0usize;
        for chunk in voice.audio_codes.chunks(64) {
            emitted += normal.decode(chunk, &mut reference, false)?.len();
        }
        let pending = voice.audio_codes.len() / 16 * 1920 - emitted;
        let mut rows = Vec::new();
        for repeat in 0..4 {
            let mut state = reference.clone();
            let mut remaining = pending;
            for call in 1..=2 {
                let started = Instant::now();
                let pcm = normal.decode(&codes, &mut state, false)?;
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                let discarded = remaining.min(pcm.len());
                remaining -= discarded;
                rows.push(serde_json::json!({"repeat": repeat, "call": call,
                    "decode_ms": ms, "raw_samples": pcm.len(),
                    "discard_reference_samples": discarded,
                    "target_samples": pcm.len() - discarded}));
            }
        }
        let (mut full_state, mut partial_state) = (reference.clone(), reference.clone());
        let first_pcm = normal.decode(&codes, &mut full_state, false)?;
        let started = Instant::now();
        let partial_result = partial.advance_state(&codes, &mut partial_state);
        let partial_ms = started.elapsed().as_secs_f64() * 1000.0;
        reports.push(serde_json::json!({"speaker": name,
            "reference_frames": voice.audio_codes.len() / 16,
            "reference_latent_shape": reference.latent_buffer.shape(),
            "reference_conv_shape": reference.conv_history.shape(),
            "reference_samples_pending": pending,
            "all_first_waveform_discarded": first_pcm.len() <= pending,
            "state_only_state_equal": partial_result.is_ok() && full_state.bitwise_eq(&partial_state),
            "state_only_error": partial_result.err().map(|e| e.to_string()),
            "state_only_ms": partial_ms, "calls": rows,
            "measurement_codes": "zero codec probe; measures shape/sample counts, not synthesized speech or service TTFA"}));
    }
    std::fs::write(root.join("decoder.json"), serde_json::to_vec_pretty(&reports)?)?;
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}

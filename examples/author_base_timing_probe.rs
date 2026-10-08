//! Raw-text, no-instruction diagnosis. Does not alter the running server.
use qwen3_tts::{TtsEngine, SamplerConfig};
use std::path::Path;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::remove_var("QWEN_OPT_MODE_FILE");
    std::env::remove_var("QWEN_PRONUNCIATION_ASSETS");
    std::env::set_var("QWEN_OPT_FLAGS", "quiet");
    std::env::set_var("QWEN_TTFA_PROFILE", "1");
    let quant = std::env::args().nth(1).unwrap_or_else(|| "q5_k_m".into());
    let device = std::env::args().nth(2).unwrap_or_else(|| "vulkan".into());
    if device.starts_with("cpu") {
        std::env::set_var("QWEN_PRONUNCIATION_DEVICE", "cpu");
        std::env::set_var("QWEN_LLAMA_THREADS", "8");
    } else { std::env::remove_var("QWEN_PRONUNCIATION_DEVICE"); }
    let mut engine = TtsEngine::new(Path::new("diagnostics/author-root/base-rust-models"), &quant, 4).await?;
    engine.prewarm_codec_projections();
    engine.load_speakers(Path::new("speakers"))?;
    engine.prewarm_decoder_contexts()?;
    let text = "本文内容由讯飞AI社区注册用户自发贡献，版权归原作者所有，讯飞AI社区不拥有其著作权，亦不承担相应法律责任。如果您发现本社区中有涉嫌抄袭的内容，填写侵权投诉表单进行举报，一经查实，本社区将立刻删除涉嫌侵权内容。";
    for speaker in ["39"] {
        let voice = engine.get_speaker(speaker).clone();
        let modes: &[&str] = &["baseline"];
        for repetition in 1..=3 {
            let mode = modes[0];
            std::env::set_var("QWEN_PRONUNCIATION_PROBE", mode);
            let dir = format!("diagnostics/author-root/base-rust-timing-{quant}-{device}-{speaker}-{repetition}");
            std::fs::create_dir_all(&dir)?;
            std::env::set_var("QWEN_CODEC_TRACE_DIR", &dir);
            let mut config = SamplerConfig::new(0.7, 50, 1.0, Some(43));
            if mode == "full-history" { config.penalty_last_n = usize::MAX; }
            engine.set_sampler_config(config);
            engine.set_max_steps(2048);
            let audio = engine.generate_with_voice_streaming(text, &voice, None, None)?;
            audio.save_wav(format!("{dir}/audio.wav"))?;
            let pcm: Vec<u8> = audio.samples.iter().flat_map(|value| value.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/audio.f32"), pcm)?;
            println!("[PronunciationProbe] speaker={speaker} mode={mode} samples={}", audio.samples.len());
        }
    }
    Ok(())
}

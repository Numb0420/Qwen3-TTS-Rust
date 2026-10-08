//! Reference audio preparation shared by CLI and web speaker creation.
use std::path::Path;
use crate::models::onnx::{init_onruntime, AudioEncoder, SpeakerEncoder};
use crate::VoiceFile;

pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("WAV read failed: {e}"))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err("WAV has invalid channel count or sample rate".into());
    }

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
        (hound::SampleFormat::Int, 8..=16) => {
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i16>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        }
        (hound::SampleFormat::Int, 17..=32) => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        }
        _ => return Err(format!("Unsupported WAV format: {spec:?}")),
    };
    let channels = spec.channels as usize;
    if samples.len() < channels || samples.len() % channels != 0 {
        return Err("WAV has no complete audio frames".into());
    }
    if !samples.iter().all(|sample| sample.is_finite()) {
        return Err("WAV contains non-finite samples".into());
    }
    let mono = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

pub fn resample_to_24k(input: &[f32], input_rate: u32) -> Vec<f32> {
    if input_rate == 24_000 {
        return input.to_vec();
    }

    // Windowed-sinc resampling preserves consonants better than linear interpolation.
    const RADIUS: isize = 32;
    let output_len = (input.len() as u64 * 24_000 / input_rate as u64) as usize;
    let cutoff = (24_000.0 / input_rate as f64).min(1.0);
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let source_pos = index as f64 * input_rate as f64 / 24_000.0;
        let center = source_pos.floor() as isize;
        let mut weighted_sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for sample_index in center - RADIUS + 1..=center + RADIUS {
            let distance = source_pos - sample_index as f64;
            let phase = std::f64::consts::PI * cutoff * distance;
            let sinc = if phase.abs() < 1e-12 {
                1.0
            } else {
                phase.sin() / phase
            };
            let window_phase = std::f64::consts::PI * distance / RADIUS as f64;
            let window = 0.42 + 0.5 * window_phase.cos() + 0.08 * (2.0 * window_phase).cos();
            let weight = cutoff * sinc * window;
            let clamped = sample_index.clamp(0, input.len() as isize - 1) as usize;
            weighted_sum += input[clamped] as f64 * weight;
            weight_sum += weight;
        }
        output.push((weighted_sum / weight_sum) as f32);
    }
    output
}


pub fn extract_full(audio_path: &Path, model_dir: &Path, name: &str, text: &str) -> Result<VoiceFile, String> {
    let (audio, rate) = read_wav(audio_path)?;
    let duration = audio.len() as f64 / rate as f64;
    if !(1.0..=30.0).contains(&duration) {
        return Err("参考音频须为 1–30 秒，建议使用 3–30 秒清晰单人录音。".into());
    }
    let audio = resample_to_24k(&audio, rate);
    init_onruntime().map_err(|e| e.to_string())?;
    let mut speaker = SpeakerEncoder::load(&model_dir.join("qwen3_tts_speaker_encoder.onnx").to_string_lossy()).map_err(|e| e.to_string())?;
    let embedding = speaker.encode(&audio).map_err(|e| e.to_string())?;
    if embedding.len() != 2048 || !embedding.iter().all(|v| v.is_finite()) {
        return Err("说话人特征无效".into());
    }
    let mut codec = AudioEncoder::load(&model_dir.join("qwen3_tts_codec_encoder.onnx").to_string_lossy()).map_err(|e| e.to_string())?;
    let codes = codec.encode(&audio).map_err(|e| e.to_string())?;
    if codes.is_empty() || codes.len() % 16 != 0 {
        return Err("参考音频编码无效".into());
    }
    Ok(VoiceFile::new(text.to_owned(), codes, embedding).with_metadata(Some(name.to_owned()), None, None, Some("Web upload: full ICL, matched Base encoders".into())))
}

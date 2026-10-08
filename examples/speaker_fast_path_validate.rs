//! Replay real production codec traces; validate every state and retained PCM sample.
use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder, DecoderState};
use qwen3_tts::VoiceFile;
use std::path::Path;

fn bytes(samples: &[f32]) -> Vec<u8> { samples.iter().flat_map(|v| v.to_le_bytes()).collect() }

fn compare(
    normal: &mut AudioDecoder, partial: &mut AudioDecoder, cropped: &mut AudioDecoder,
    reference: &DecoderState, pending: usize, codes: &[i64], combined: bool,
) -> Result<(Vec<u8>, usize), Box<dyn std::error::Error>> {
    let (mut a, mut b) = (reference.clone(), reference.clone());
    let (mut skip_a, mut skip_b) = (pending, pending);
    let mut pcm = Vec::new();
    let chunks: Vec<_> = codes.chunks(64).collect();
    for (index, chunk) in chunks.iter().enumerate() {
        let final_call = index + 1 == chunks.len();
        let baseline = normal.decode(chunk, &mut a, final_call)?;
        let target = if index == 0 && !final_call && chunk.len() == 64 {
            if baseline.len() > skip_b { return Err("First waveform includes target samples".into()); }
            partial.advance_state(chunk, &mut b)?;
            skip_b -= baseline.len();
            Vec::new()
        } else if combined && index == 1 && !final_call && chunk.len() == 64 && skip_b == 0 {
            cropped.decode(chunk, &mut b, false)?
        } else { normal.decode(chunk, &mut b, final_call)? };
        if !a.bitwise_eq(&b) { return Err(format!("State differs at call {}", index+1).into()); }
        let trim_a = skip_a.min(baseline.len()); skip_a -= trim_a;
        let trim_b = skip_b.min(target.len()); skip_b -= trim_b;
        let expected = bytes(&baseline[trim_a..]);
        let actual = bytes(&target[trim_b..]);
        if expected != actual { return Err(format!("PCM differs at call {}", index+1).into()); }
        pcm.extend(actual);
    }
    Ok((pcm, chunks.len()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("QWEN_ONNX_NO_SPIN", "1");
    init_onruntime()?;
    let root = Path::new("diagnostics/speaker-fast-path-2026-10-08");
    let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("before/results.json"))?)?;
    let mut normal = AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder.onnx")?;
    let mut partial = AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder_state_only.onnx")?;
    let mut cropped = AudioDecoder::load("runtime/releases/latency-crop-2026-10-08/decoder-stage3-end1292.onnx")?;
    let mut results = Vec::new();
    for row in rows.as_array().ok_or("Invalid results")? {
        let name = row["speaker"].as_str().ok_or("Missing speaker")?;
        let number = row["trace_number"].as_u64().ok_or("Missing trace")?;
        let voice = VoiceFile::load(Path::new(&format!("speakers/{name}.json")))?;
        let trace: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join(format!("before/codes/codec-{number}.json")))?)?;
        let codes: Vec<i64> = trace["codes"].as_array().ok_or("Missing codes")?.iter().map(|v|v.as_i64().unwrap()).collect();
        let mut references = vec![("original", voice.audio_codes.clone())];
        // Exercise the remaining possible partial history length, without editing any voice file.
        if name == "39" && row["text"] == "news" {
            references.push(("synthetic-history-3", voice.audio_codes[..127*16].to_vec()));
        }
        for (kind, reference_codes) in references {
            let mut reference = AudioDecoder::create_state();
            let mut emitted = 0;
            for chunk in reference_codes.chunks(64) { emitted += normal.decode(chunk, &mut reference, false)?.len(); }
            let pending = reference_codes.len()/16*1920-emitted;
            for combined in [false,true] {
                let (pcm,calls) = compare(&mut normal,&mut partial,&mut cropped,&reference,pending,&codes,combined)?;
                let original = std::fs::read(root.join("before").join(row["pcm_file"].as_str().unwrap()))?;
                let matches_production = kind == "original" && pcm == original;
                if kind == "original" && !matches_production { return Err("Replay differs from production PCM".into()); }
                let report = serde_json::json!({"speaker":name,"text":row["text"],"reference":kind,
                    "mode":if combined{"combined"}else{"state-only"},
                    "conv_history_shape":reference.conv_history.shape(),"all_state_tensors_bitwise_equal":true,
                    "pcm_matches_full_decoder":true,"pcm_matches_production":matches_production,
                    "production_comparison_applicable":kind=="original","calls":calls,"pcm_bytes":pcm.len()});
                println!("VALIDATED {report}"); results.push(report);
                std::fs::write(root.join("replay-validation.json"),serde_json::to_vec_pretty(&results)?)?;
            }
        }
    }
    Ok(())
}

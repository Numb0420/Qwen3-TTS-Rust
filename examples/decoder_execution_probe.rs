use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder};
use std::{path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("diagnostics/latency-stages/decoder")?;
    let operator_profile = std::env::args().any(|arg| arg == "--operators");
    if operator_profile { std::env::set_var("QWEN_ONNX_OP_PROFILE", "diagnostics/latency-stages/decoder/operators"); }
    init_onruntime()?;
    let voice=qwen3_tts::VoiceFile::load(Path::new("speakers/39.json"))?;
    let trace:serde_json::Value=serde_json::from_slice(&std::fs::read("diagnostics/latency-stages/codes/codec-2.json")?)?;
    let codes:Vec<i64>=trace["codes"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let mut decoder=AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder.onnx")?;
    let mut reference=AudioDecoder::create_state();
    for chunk in voice.audio_codes.chunks(64) { decoder.decode(chunk,&mut reference,false)?; }
    let mut rows=Vec::new();
    for repeat in 1..=4 {
        let mut state=reference.clone();let mut pcm=Vec::new();
        for (call,chunk) in codes[..128].chunks(64).enumerate() {
            let started=Instant::now();let samples=decoder.decode(chunk,&mut state,false)?;
            rows.push(serde_json::json!({"repeat":repeat,"call":call+1,"ms":started.elapsed().as_secs_f64()*1000.0,"samples":samples.len()}));
            pcm.extend(samples.iter().flat_map(|v|v.to_le_bytes()));
        }
        let label=if operator_profile {"operators"} else if std::env::var_os("QWEN_ONNX_NO_SPIN").is_some(){"no-spin"}else{"default"};
        std::fs::write(format!("diagnostics/latency-stages/decoder/{label}-{repeat}.f32"),pcm)?;
    }
    let label=if operator_profile {"operators"}else if std::env::var_os("QWEN_ONNX_NO_SPIN").is_some(){"no-spin"}else{"default"};
    std::fs::write(format!("diagnostics/latency-stages/decoder/{label}-timings.json"),serde_json::to_vec_pretty(&rows)?)?;
    if operator_profile { println!("Operator profile: {}",decoder.finish_operator_profile()?); }
    println!("{}",serde_json::to_string(&rows)?);
    Ok(())
}

//! Compare all state tensors and full target PCM before considering deployment.
use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder, DecoderState};
use qwen3_tts::VoiceFile;
use std::{path::Path, time::Instant};

fn same_state(a: &DecoderState, b: &DecoderState) -> bool {
    let eq3 = |x: &ndarray::Array3<f32>, y: &ndarray::Array3<f32>| {
        x.shape() == y.shape() && x.iter().zip(y).all(|(a,b)| a.to_bits() == b.to_bits())
    };
    let eq4 = |x: &ndarray::Array4<f32>, y: &ndarray::Array4<f32>| {
        x.shape() == y.shape() && x.iter().zip(y).all(|(a,b)| a.to_bits() == b.to_bits())
    };
    eq3(&a.pre_conv_history,&b.pre_conv_history) && eq3(&a.latent_buffer,&b.latent_buffer)
        && eq3(&a.conv_history,&b.conv_history)
        && a.kv_cache.iter().zip(&b.kv_cache).all(|((ak,av),(bk,bv))| eq4(ak,bk)&&eq4(av,bv))
}
fn bytes(samples: &[f32]) -> Vec<u8> { samples.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn main() -> Result<(),Box<dyn std::error::Error>> {
    std::env::set_var("QWEN_ONNX_NO_SPIN","1");
    init_onruntime()?;
    let voice=VoiceFile::load(Path::new("speakers/39.json"))?;
    let trace:serde_json::Value=serde_json::from_slice(&std::fs::read("diagnostics/latency-stages/codes/codec-2.json")?)?;
    let codes:Vec<i64>=trace["codes"].as_array().ok_or("codes missing")?.iter().map(|v|v.as_i64().unwrap()).collect();
    let mut normal=AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder.onnx")?;
    let mut state_only=AudioDecoder::load("diagnostics/latency-stages/decoder/qwen3_tts_decoder_state_only.onnx")?;
    let mut reference=AudioDecoder::create_state();
    let mut emitted=0;
    for chunk in voice.audio_codes.chunks(64) {emitted+=normal.decode(chunk,&mut reference,false)?.len();}
    let skip=voice.audio_codes.len()/16*1920-emitted;
    let mut rows=Vec::new();
    for repeat in 1..=4 {
        let (mut a,mut b)=(reference.clone(),reference.clone());
        let started=Instant::now();let raw=normal.decode(&codes[..64],&mut a,false)?;let normal_ms=started.elapsed().as_secs_f64()*1000.0;
        let started=Instant::now();state_only.advance_state(&codes[..64],&mut b)?;let state_only_ms=started.elapsed().as_secs_f64()*1000.0;
        if raw.len()>skip {return Err("First decode includes wanted target audio; cannot omit".into())}
        if !same_state(&a,&b) {return Err("State tensors differ; do not deploy".into())}
        rows.push(serde_json::json!({"repeat":repeat,"normal_ms":normal_ms,"state_only_ms":state_only_ms,"state_bitwise_equal":true,"raw_samples":raw.len(),"reference_samples_pending":skip}));
    }
    let (mut a,mut b)=(reference.clone(),reference.clone());
    let raw=normal.decode(&codes[..64],&mut a,false)?;
    state_only.advance_state(&codes[..64],&mut b)?;
    let (mut pcm_a,mut pcm_b)=(Vec::new(),Vec::new());
    let mut remaining_skip=skip-raw.len();
    let chunks:Vec<_>=codes[64..].chunks(64).collect();
    for (index,chunk) in chunks.iter().enumerate() {
        let final_call=index+1==chunks.len();
        let x=normal.decode(chunk,&mut a,final_call)?;
        let y=normal.decode(chunk,&mut b,final_call)?;
        if bytes(&x)!=bytes(&y)||!same_state(&a,&b){return Err("Subsequent PCM/state differs".into())}
        let trim=remaining_skip.min(x.len());remaining_skip-=trim;
        pcm_a.extend(bytes(&x[trim..]));pcm_b.extend(bytes(&y[trim..]));
    }
    let expected=std::fs::read("diagnostics/author-root/hot-news-1.f32")?;
    if pcm_a!=expected||pcm_b!=expected {return Err("Target PCM does not match accepted baseline".into())}
    let report=serde_json::json!({"first_call":rows,"all_19_state_tensors_bitwise_equal":true,"full_pcm_matches_baseline":true,"pcm_bytes":pcm_a.len(),"reference_latent_shape":reference.latent_buffer.shape(),"reference_conv_shape":reference.conv_history.shape()});
    std::fs::write("diagnostics/latency-stages/decoder/state-only-probe.json",serde_json::to_vec_pretty(&report)?)?;
    std::fs::write("diagnostics/latency-stages/decoder/state-only-news.f32",pcm_b)?;
    println!("{report}");
    Ok(())
}

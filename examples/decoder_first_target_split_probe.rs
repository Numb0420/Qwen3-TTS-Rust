//! Isolated waveform replay. Does not change the live service or codec generator.
use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder, DecoderState};
use qwen3_tts::{VoiceFile, AudioSample};
use std::{path::Path, time::Instant};
fn bytes(x:&[f32])->Vec<u8>{x.iter().flat_map(|v|v.to_le_bytes()).collect()}
fn run(decoder:&mut AudioDecoder, initial:&DecoderState, codes:&[i64], first:usize)
    ->Result<(Vec<f32>,DecoderState,Vec<serde_json::Value>),Box<dyn std::error::Error>> {
    let mut state=initial.clone(); let mut pcm=Vec::new(); let mut offset=4usize;
    let mut at8=None; let mut rows=Vec::new();
    let total=codes.len()/16;
    while offset<total {
        let count=if offset==4 {first} else if offset<8 {8-offset} else {4};
        let end=(offset+count).min(total);
        let begin=Instant::now();
        let audio=decoder.decode(&codes[offset*16..end*16],&mut state,end==total)?;
        let ms=begin.elapsed().as_secs_f64()*1000.0;
        if !audio.iter().all(|v|v.is_finite()){return Err("Nonfinite waveform; stop".into())}
        if rows.len()<3 {rows.push(serde_json::json!({"start_frame":offset+1,"end_frame":end,"samples":audio.len(),"decode_ms":ms,"final":end==total}));}
        pcm.extend(audio); offset=end;
        if offset==8 {at8=Some(state.clone());}
    }
    Ok((pcm,at8.ok_or("Missing frame-eight state")?,rows))
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let root=Path::new("diagnostics/last-output-2026-10-07/split-target");
    if root.exists(){return Err("Preserve previous split evidence".into())}
    std::fs::create_dir_all(root)?;
    std::env::set_var("QWEN_ONNX_NO_SPIN","1");
    init_onruntime()?;
    let mut decoder=AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder.onnx")?;
    let mut partial=AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder_state_only.onnx")?;
    let voice=VoiceFile::load(Path::new("speakers/39.json"))?;
    let mut reference=AudioDecoder::create_state(); let mut emitted=0usize;
    for chunk in voice.audio_codes.chunks(64){emitted+=decoder.decode(chunk,&mut reference,false)?.len();}
    let pending=voice.audio_codes.len()/16*1920-emitted;
    let mut reports=Vec::new();
    for (name,trace,golden) in [
        ("news","diagnostics/latency-stages/codes/codec-2.json","diagnostics/author-root/hot-news-1.f32"),
        ("copyright","diagnostics/latency-stages/2026-10-07/head2-a-before/codes/codec-5.json","diagnostics/author-root/base-rust-q5_k_m-vulkan-39/audio.f32")
    ] {
        let parsed:serde_json::Value=serde_json::from_slice(&std::fs::read(trace)?)?;
        let codes:Vec<i64>=parsed["codes"].as_array().ok_or("Missing codes")?.iter().map(|x|x.as_i64().unwrap()).collect();
        let mut initial=reference.clone(); let mut full=reference.clone();
        let omitted=decoder.decode(&codes[..64],&mut full,false)?;
        partial.advance_state(&codes[..64],&mut initial)?;
        if omitted.len()!=pending || !full.bitwise_eq(&initial){return Err("Initial state-only gate failed".into())}
        let (baseline,baseline8,_) =run(&mut decoder,&initial,&codes,4)?;
        if bytes(&baseline)!=std::fs::read(golden)? {return Err("Baseline replay differs".into())}
        let mut shape_pcm:Vec<(usize,Vec<u8>)>=Vec::new();
        for repeat in 0..3 {
            let shapes=if repeat%2==0 {[4,2,1]} else {[1,2,4]};
            for first in shapes {
                let (pcm,state8,rows)=run(&mut decoder,&initial,&codes,first)?;
                let raw=bytes(&pcm);
                if let Some((_,prior))=shape_pcm.iter().find(|(n,_)|*n==first){
                    if prior!=&raw{return Err("Replay not repeatable".into())}
                } else {shape_pcm.push((first,raw.clone()));}
                let changed=pcm.iter().zip(&baseline).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
                let max=pcm.iter().zip(&baseline).map(|(a,b)|(*a-*b).abs()).fold(0f32,f32::max);
                let rmse=(pcm.iter().zip(&baseline).map(|(a,b)|(*a as f64-*b as f64).powi(2)).sum::<f64>()/baseline.len() as f64).sqrt();
                let row=serde_json::json!({"text":name,"first_target_frames":first,"repeat":repeat,
                    "first_use_for_shape":name=="news" && repeat==0 && first!=4,
                    "calls":rows,"samples":pcm.len(),"same_length":pcm.len()==baseline.len(),
                    "pcm_equal":raw==bytes(&baseline),"changed_samples":changed,"max_abs_difference":max,"rmse":rmse,
                    "state_at_8_equal":state8.bitwise_eq(&baseline8),"codec_unchanged":true,
                    "isolated_decoder_timing_not_service_latency":true});
                println!("SPLIT {row}"); reports.push(row);
                std::fs::write(root.join("results.json"),serde_json::to_vec_pretty(&reports)?)?;
                if repeat==0 {
                    let dir=root.join(format!("{name}-first{first}")); std::fs::create_dir_all(&dir)?;
                    std::fs::write(dir.join("audio.f32"),raw)?;
                    AudioSample{samples:pcm,sample_rate:24000,channels:1}.save_wav(dir.join("audio.wav"))?;
                }
            }
        }
    }
    Ok(())
}

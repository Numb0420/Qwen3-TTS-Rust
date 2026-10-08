//! Replay identical codes and states before a decoder experiment reaches the service.
use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder, DecoderState};
use qwen3_tts::VoiceFile;
use std::{path::{Path,PathBuf}, time::Instant};
fn bytes(x:&[f32])->Vec<u8>{x.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn dump(state:&DecoderState,codes:&[i64],dir:&Path)->Result<(),Box<dyn std::error::Error>>{
    std::fs::create_dir_all(dir)?;
    let mut inputs=serde_json::Map::new();
    let mut save=|name:&str,shape:Vec<usize>,values:Vec<f32>|->Result<(),Box<dyn std::error::Error>>{
        std::fs::write(dir.join(format!("{name}.bin")),bytes(&values))?;
        inputs.insert(name.to_owned(),serde_json::json!({"shape":shape,"dtype":"float32"}));Ok(())
    };
    save("pre_conv_history",state.pre_conv_history.shape().to_vec(),state.pre_conv_history.iter().copied().collect())?;
    save("latent_buffer",state.latent_buffer.shape().to_vec(),state.latent_buffer.iter().copied().collect())?;
    save("conv_history",state.conv_history.shape().to_vec(),state.conv_history.iter().copied().collect())?;
    for(i,(k,v))in state.kv_cache.iter().enumerate(){
        save(&format!("past_key_{i}"),k.shape().to_vec(),k.iter().copied().collect())?;
        save(&format!("past_value_{i}"),v.shape().to_vec(),v.iter().copied().collect())?;
    }
    std::fs::write(dir.join("audio_codes.bin"),codes.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;
    inputs.insert("audio_codes".into(),serde_json::json!({"shape":[1,codes.len()/16,16],"dtype":"int64"}));
    std::fs::write(dir.join("is_last.bin"),0f32.to_le_bytes())?;
    inputs.insert("is_last".into(),serde_json::json!({"shape":[1],"dtype":"float32"}));
    std::fs::write(dir.join("inputs.json"),serde_json::to_vec_pretty(&inputs)?)?;Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>>{
    let root=PathBuf::from(std::env::var("QWEN_PROBE_OUTPUT_DIR").unwrap_or("diagnostics/latency-stages/next-hot".into()));std::fs::create_dir_all(&root)?;
    let inter=std::env::var("QWEN_ONNX_INTER_THREADS").ok();std::env::remove_var("QWEN_ONNX_INTER_THREADS");
    let dynamic=std::env::var("QWEN_ONNX_DYNAMIC_BLOCK_BASE").ok();std::env::remove_var("QWEN_ONNX_DYNAMIC_BLOCK_BASE");
    let affinity=std::env::var("QWEN_ONNX_AFFINITY").ok();std::env::remove_var("QWEN_ONNX_AFFINITY");
    let global=std::env::var_os("QWEN_ONNX_GLOBAL_POOL").is_some();
    if global { std::env::set_var("QWEN_ONNX_INDEPENDENT_POOL","1"); }
    std::env::set_var("QWEN_ONNX_NO_SPIN","1");init_onruntime()?;
    let mut normal=AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder.onnx")?;
    let mut state_only=AudioDecoder::load("models/base-1.7b/onnx/qwen3_tts_decoder_state_only.onnx")?;
    let voice=VoiceFile::load(Path::new("speakers/39.json"))?;
    let trace_path=std::env::var("QWEN_PROBE_CODES").unwrap_or("diagnostics/latency-stages/codes/codec-2.json".into());
    let trace:serde_json::Value=serde_json::from_slice(&std::fs::read(trace_path)?)?;
    let codes:Vec<i64>=trace["codes"].as_array().ok_or("codes missing")?.iter().map(|x|x.as_i64().unwrap()).collect();
    let mut state=AudioDecoder::create_state();for chunk in voice.audio_codes.chunks(64){normal.decode(chunk,&mut state,false)?;}
    state_only.advance_state(&codes[..64],&mut state)?;
    dump(&state,&codes[64..128],&root.join("first-target-input"))?;
    if std::env::var_os("QWEN_CAPTURE_ONLY").is_some(){println!("Captured first target decoder inputs");return Ok(())}
    if let Some(value)=inter{std::env::set_var("QWEN_ONNX_INTER_THREADS",value);}
    if let Some(value)=dynamic{std::env::set_var("QWEN_ONNX_DYNAMIC_BLOCK_BASE",value);}
    if let Some(value)=affinity{std::env::set_var("QWEN_ONNX_AFFINITY",value);}
    if global { std::env::remove_var("QWEN_ONNX_INDEPENDENT_POOL"); }
    let model=std::env::var("QWEN_CANDIDATE_MODEL").unwrap_or("models/base-1.7b/onnx/qwen3_tts_decoder.onnx".into());
    let tag=std::env::var("QWEN_CANDIDATE_TAG").unwrap_or("candidate".into());
    let mut candidate=AudioDecoder::load(&model)?;
    if std::env::var_os("QWEN_CANDIDATE_ALL_CALLS").is_some() {
        let (mut reference_a, mut reference_b)=(AudioDecoder::create_state(),AudioDecoder::create_state());
        for (index, chunk) in voice.audio_codes.chunks(64).enumerate() {
            let a=normal.decode(chunk,&mut reference_a,false)?;
            let b=candidate.decode(chunk,&mut reference_b,false)?;
            if bytes(&a)!=bytes(&b)||!reference_a.bitwise_eq(&reference_b) {
                std::fs::write(root.join(format!("{tag}-probe.json")),serde_json::to_vec_pretty(&serde_json::json!({"status":"reference-output-different-do-not-deploy","reference_call":index+1}))?)?;
                println!("Reference output/state difference; experiment stopped: {tag}");return Ok(())
            }
        }
    }
    let mut rows=Vec::new();
    for repeat in 0..5{
        let(mut a,mut b)=(state.clone(),state.clone());
        let started=Instant::now();let x=normal.decode(&codes[64..128],&mut a,false)?;let a_ms=started.elapsed().as_secs_f64()*1000.0;
        let started=Instant::now();let y=candidate.decode(&codes[64..128],&mut b,false)?;let b_ms=started.elapsed().as_secs_f64()*1000.0;
        let equal=bytes(&x)==bytes(&y)&&a.bitwise_eq(&b);
        let changed=x.iter().zip(&y).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        let max_diff=x.iter().zip(&y).map(|(a,b)|(*a-*b).abs()).fold(0.0f32,f32::max);
        rows.push(serde_json::json!({"repeat":repeat,"normal_ms":a_ms,"candidate_ms":b_ms,"normal_samples":x.len(),"candidate_samples":y.len(),"pcm_equal":bytes(&x)==bytes(&y),"states_equal":a.bitwise_eq(&b),"changed_samples":changed,"max_abs_difference":max_diff}));
        std::fs::write(root.join(format!("{tag}-probe.json")),serde_json::to_vec_pretty(&serde_json::json!({"first_target":rows,"status":if equal{"first-target-equal"}else{"output-different-do-not-deploy"}}))?)?;
        if !equal{println!("Output difference detected; experiment stopped: {tag}");return Ok(())}
    }
    let(mut a,mut b)=(state.clone(),state.clone());let(mut pcm_a,mut pcm_b)=(Vec::new(),Vec::new());
    let chunks:Vec<_>=codes[64..].chunks(64).collect();
    let all_calls=std::env::var_os("QWEN_CANDIDATE_ALL_CALLS").is_some();
    for(index,chunk)in chunks.iter().enumerate(){
        let final_call=index+1==chunks.len();let x=normal.decode(chunk,&mut a,final_call)?;
        let y=if index==0||all_calls{candidate.decode(chunk,&mut b,final_call)?}else{normal.decode(chunk,&mut b,final_call)?};
        if bytes(&x)!=bytes(&y)||!a.bitwise_eq(&b){return Err("Later PCM/state differs; do not deploy".into())}
        pcm_a.extend(bytes(&x));pcm_b.extend(bytes(&y));
    }
    let expected_path=std::env::var("QWEN_PROBE_GOLDEN").unwrap_or("diagnostics/author-root/hot-news-1.f32".into());
    let expected=std::fs::read(expected_path)?;
    if pcm_a!=expected||pcm_b!=expected{return Err("Baseline differs; do not deploy".into())}
    let report=serde_json::json!({"status":"full-pcm-and-all-states-equal","first_target":rows,"bytes":pcm_a.len()});
    std::fs::write(root.join(format!("{tag}-probe.json")),serde_json::to_vec_pretty(&report)?)?;println!("{report}");
    if std::env::var_os("QWEN_DIAG_PROFILE_PREFIX").is_some(){println!("Operator profile: {}",candidate.finish_operator_profile()?);}
    Ok(())
}

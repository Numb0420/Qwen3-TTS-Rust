//! Decode identical generated codes in one call to isolate streaming decoder state.
use qwen3_tts::models::onnx::{init_onruntime,AudioDecoder};
use qwen3_tts::{VoiceFile,AudioSample};
#[tokio::main]
async fn main() -> Result<(),Box<dyn std::error::Error>> {
 init_onruntime()?;
 let mut decoder=AudioDecoder::load("models/onnx/qwen3_tts_decoder.onnx")?;
 for voice in ["39","serena"] {
  let dir=format!("diagnostics/author-root/q5_k_m-{voice}-baseline");
  let path=std::fs::read_dir(&dir)?.filter_map(Result::ok).map(|e|e.path()).find(|p|p.file_name().unwrap().to_string_lossy().starts_with("codec-" )).unwrap();
  let trace:serde_json::Value=serde_json::from_slice(&std::fs::read(path)?)?;
  let generated:Vec<i64>=trace["codes"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
  let reference=VoiceFile::load(format!("speakers/{voice}.json"))?;
  let mut state=AudioDecoder::create_state();
  let mut emitted=0;
  for chunk in reference.audio_codes.chunks(16*4) {
   emitted+=decoder.decode(chunk,&mut state,false)?.len();
  }
  let mut skip=(reference.audio_codes.len()/16*1920).saturating_sub(emitted);
  let mut samples=Vec::new();
  let chunks:Vec<_>=generated.chunks(16*8).collect();
  for (index,chunk) in chunks.iter().enumerate() {
   let raw=decoder.decode(chunk,&mut state,index+1==chunks.len())?;
   if !raw.iter().all(|v|v.is_finite()) {return Err("nonfinite PCM in 8-frame probe".into())}
   let drop=skip.min(raw.len());skip-=drop;samples.extend_from_slice(&raw[drop..]);
  }
  let audio=AudioSample{samples,sample_rate:24000,channels:1};
  audio.save_wav(format!("{dir}/block8-decoder.wav"))?;
  let pcm:Vec<u8>=audio.samples.iter().flat_map(|v|v.to_le_bytes()).collect();
  std::fs::write(format!("{dir}/block8-decoder.f32"),pcm)?;
  println!("[Block8Decoder] voice={voice} target_samples={}",audio.samples.len());
 }
 Ok(())
}

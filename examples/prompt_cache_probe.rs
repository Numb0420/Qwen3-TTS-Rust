use qwen3_tts::{assets_manager::Assets, PromptBuilder, Tokenizer, VoiceFile};
use std::{error::Error,path::Path,time::Instant};
fn main()->Result<(),Box<dyn Error>> {
 std::env::remove_var("QWEN_OPT_MODE_FILE");
 let assets=Assets::load(Path::new("models/gguf"))?;
 let tokenizer=Tokenizer::load(Path::new("models"))?;
 let voice=VoiceFile::load("speakers/39.json")?;
 let reference=voice.audio_codes.iter().map(|&v|v as i32).collect::<Vec<_>>();
 let ids=tokenizer.encode(&voice.ref_text);
 let text="他也提到，去年河北十个市，县获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。";
 std::env::remove_var("QWEN_OPT_FLAGS");
 let expected=PromptBuilder::build_clone_prompt(text,&tokenizer,&assets,&reference,&ids,&voice.speaker_embedding,2055,None);
 for mode in ["baseline","reference","prefix","reference,prefix"] {
  std::env::set_var("QWEN_OPT_FLAGS",mode);
  let actual=PromptBuilder::build_clone_prompt(text,&tokenizer,&assets,&reference,&ids,&voice.speaker_embedding,2055,None);
  assert_eq!(expected.text_ids,actual.text_ids);
  assert_eq!(expected.embd.len(),actual.embd.len());
  assert_eq!(expected.spk_emb,actual.spk_emb);
  assert!(expected.embd.iter().flatten().zip(actual.embd.iter().flatten()).all(|(a,b)|a.to_bits()==b.to_bits()));
  let started=Instant::now();
  for _ in 0..100 {std::hint::black_box(PromptBuilder::build_clone_prompt(text,&tokenizer,&assets,&reference,&ids,&voice.speaker_embedding,2055,None));}
  println!("[PromptCacheProbe] {}",serde_json::json!({"mode":mode,"average_ms":started.elapsed().as_secs_f64()*10.0,"iterations":100,"bitwise_equal":true}));
 }
 Ok(())
}

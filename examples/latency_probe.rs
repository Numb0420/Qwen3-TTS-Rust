//! Standalone measurements; does not change the running server or inference policy.
use qwen3_tts::assets_manager::Assets;
use qwen3_tts::models::llama::{LlamaBatch, LlamaContext, LlamaModel, LlamaSampler};
use qwen3_tts::models::onnx::{init_onruntime, AudioDecoder, DecoderState};
use qwen3_tts::{PromptBuilder, Tokenizer, VoiceFile};
use std::{error::Error, path::Path, time::Instant};

fn positions(start: usize, count: usize) -> Vec<i32> {
    let range: Vec<_> = (start..start + count).map(|x| x as i32).collect();
    let mut result = Vec::with_capacity(count * 4);
    for _ in 0..3 { result.extend_from_slice(&range); }
    result.extend(std::iter::repeat_n(0, count));
    result
}

fn millis(start: Instant) -> f64 { start.elapsed().as_secs_f64() * 1000.0 }

fn decoder_case(
    decoder: &mut AudioDecoder, base: &DecoderState, skip: usize,
    codes: &[i64], first: usize, empty_flush: bool,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut state = base.clone();
    let mut crop = skip;
    let mut output = Vec::new();
    let total = codes.len() / 16;
    let mut offset = 0;
    let mut index = 0;
    let all_started = Instant::now();
    while offset < total {
        let count = (if offset == 0 { first } else { 4 }).min(total - offset);
        let end = offset + count;
        let started = Instant::now();
        let pcm = decoder.decode(&codes[offset * 16..end * 16], &mut state,
            end == total && !empty_flush)?;
        let raw = pcm.len();
        let drop_count = crop.min(raw);
        crop -= drop_count;
        output.extend_from_slice(&pcm[drop_count..]);
        if index < 2 {
            println!("PROBE decoder first={} call={} input_frames={} raw_samples={} retained_samples={} decode_ms={:.3}",
                first, index + 1, count, raw, raw - drop_count, millis(started));
        }
        offset = end;
        index += 1;
    }
    if empty_flush {
        let pcm = decoder.decode(&[], &mut state, true)?;
        let drop_count = crop.min(pcm.len());
        output.extend_from_slice(&pcm[drop_count..]);
    }
    println!("PROBE decoder summary first={} empty_flush={} samples={} total_ms={:.3}",
        first, empty_flush, output.len(), millis(all_started));
    Ok(output)
}

fn main() -> Result<(), Box<dyn Error>> {
    init_onruntime()?;
    let mut decoder = AudioDecoder::load("models/onnx/qwen3_tts_decoder.onnx")?;
    let text = "他也提到，去年河北十个市，县获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。";
    let voice = VoiceFile::load("speakers/39.json")?;
    let assets = Assets::load(Path::new("models/gguf"))?;
    let tokenizer = Tokenizer::load(Path::new("models"))?;
    let codes: Vec<_> = voice.audio_codes.iter().map(|&x| x as i32).collect();
    let prompt_started = Instant::now();
    let prompt = PromptBuilder::build_clone_prompt(text, &tokenizer, &assets, &codes,
        &tokenizer.encode(&voice.ref_text), &voice.speaker_embedding, 2055, None);
    println!("PROBE prompt_ms={:.3} tokens={} reference_frames={}",
        millis(prompt_started), prompt.embd.len(), codes.len() / 16);
    let flat: Vec<_> = prompt.embd.iter().flatten().copied().collect();
    let talker = LlamaModel::load(Path::new("models/gguf_q5_k_m/qwen3_tts_talker.gguf"), 99)?;
    let predictor = LlamaModel::load(Path::new("models/gguf_q5_k_m/qwen3_tts_predictor.gguf"), 99)?;
    let mut talker_ctx = LlamaContext::new(&talker, 4096, 2048, 1, -1)?;
    let mut predictor_ctx = LlamaContext::new(&predictor, 512, 32, 0, 4)?;
    let mut talker_batch = LlamaBatch::new(2048, talker.n_embd, 1, 4);
    let mut predictor_batch = LlamaBatch::new(32, predictor.n_embd, 1, 1);
    let mut trial_codes = Vec::new();
    for trial in 0..4 {
        let all_started = Instant::now();
        let clear_started = Instant::now();
        talker_ctx.clear_memory();
        predictor_ctx.clear_memory();
        let clear_ms = millis(clear_started);
        let prefill_started = Instant::now();
        talker_batch.set_embd(&flat, &positions(0, prompt.embd.len()), 0);
        talker_ctx.decode(&mut talker_batch)?;
        // Force completion, so GPU wait is included in prefill measurement.
        std::hint::black_box(talker_ctx.get_logits_ith(prompt.embd.len() - 1));
        let prefill_ms = millis(prefill_started);
        let talker_sampler = LlamaSampler::new(talker.n_vocab, 0.7, 50, 1.0, 43)
            .with_penalties(0.0, 1.05, 0.0, 0.0, 128);
        let predictor_sampler = LlamaSampler::new(predictor.n_vocab, 0.9, 50, 1.0,
            43 ^ 0x9e37_79b9_7f4a_7c15);
        let mut generated = Vec::new();
        let mut projection_ms = 0.0;
        let mut predictor_ms = 0.0;
        let mut sampling_ms = 0.0;
        let mut talker_step_ms = 0.0;
        for step in 0..5 {
            let index = if step == 0 { prompt.embd.len() - 1 } else { 0 };
            let sampling_started = Instant::now();
            let code0 = talker_sampler.sample_with_allow(&talker_ctx, index as i32,
                Some(0), Some(2048), Some(&[2150, 2148, 2149]));
            sampling_ms += millis(sampling_started);
            if code0 >= 2048 { return Err(format!("unexpected initial code {code0}").into()); }
            generated.push(code0 as i64);
            let projection_started = Instant::now();
            let hidden = talker_ctx.get_embedding_at(index).to_vec();
            let projected = assets.project(&hidden);
            let projected_code0 = assets.get_codec_embedding_1024(0, code0);
            projection_ms += millis(projection_started);
            let mut predictor_input = projected;
            predictor_input.extend_from_slice(&projected_code0);
            let pred_started = Instant::now();
            predictor_ctx.clear_kv_cache();
            predictor_batch.set_embd(&predictor_input, &[0, 1], 0);
            predictor_ctx.decode(&mut predictor_batch)?;
            let mut pred_projection_ms = 0.0;
            let mut feedback = assets.get_codec_embedding(0, code0);
            for q in 1..16 {
                let begin = (q - 1) * 2048;
                let code = predictor_sampler.sample(&predictor_ctx, if q == 1 { 1 } else { 0 },
                    Some(begin), Some(begin + 2048)) - begin as i32;
                generated.push(code as i64);
                let emb = assets.get_codec_embedding(q, code);
                for (sum, value) in feedback.iter_mut().zip(&emb) { *sum += value; }
                if q < 15 {
                    let started = Instant::now();
                    let input = assets.get_codec_embedding_1024(q, code);
                    pred_projection_ms += millis(started);
                    predictor_batch.set_embd(&input, &[(q + 1) as i32], 0);
                    predictor_ctx.decode(&mut predictor_batch)?;
                }
            }
            predictor_ms += millis(pred_started) - pred_projection_ms;
            projection_ms += pred_projection_ms;
            println!("PROBE trial={} frame={} elapsed_ms={:.3} projection_ms={:.3} predictor_ms={:.3} talker_step_ms={:.3}",
                trial, step + 1, millis(all_started), projection_ms, predictor_ms, talker_step_ms);
            if step < 4 {
                for (sum, value) in feedback.iter_mut().zip(&assets.tts_pad) { *sum += value; }
                let started = Instant::now();
                talker_batch.set_embd(&feedback, &positions(prompt.embd.len() + step, 1), 0);
                talker_ctx.decode(&mut talker_batch)?;
                std::hint::black_box(talker_ctx.get_logits_ith(0));
                talker_step_ms += millis(started);
            }
        }
        println!("PROBE trial={} total_ms={:.3} clear_ms={:.3} prefill_ms={:.3} projection_ms={:.3} predictor_ms={:.3} sampling_ms={:.3} talker_step_ms={:.3}",
            trial, millis(all_started), clear_ms, prefill_ms, projection_ms, predictor_ms, sampling_ms, talker_step_ms);
        if trial_codes.is_empty() { trial_codes = generated; }
        else { println!("PROBE repeat_codes_equal={}", trial_codes == generated); }
    }
    let mut state = AudioDecoder::create_state();
    let mut emitted = 0;
    for chunk in voice.audio_codes.chunks(64) { emitted += decoder.decode(chunk, &mut state, false)?.len(); }
    let skip = (voice.audio_codes.len() / 16 * 1920).saturating_sub(emitted);
    println!("PROBE decoder reference_emitted={} pending_reference_samples={}", emitted, skip);
    // Same fixed codec input in every decoder case. Prefix comes from actual inference;
    // remaining frames are a test fixture from the reference recording.
    trial_codes.extend_from_slice(&voice.audio_codes[..16 * 11]);
    let baseline = decoder_case(&mut decoder, &state, skip, &trial_codes, 4, false)?;
    for (first, empty_flush) in [(4, true), (2, false), (1, false)] {
        let audio = match decoder_case(&mut decoder, &state, skip, &trial_codes, first, empty_flush) {
            Ok(audio) => audio,
            Err(error) => {
                println!("PROBE decoder comparison first={} empty_flush={} error={}", first, empty_flush, error);
                continue;
            }
        };
        let max_diff = baseline.iter().zip(&audio).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
        let rms_diff = (baseline.iter().zip(&audio).map(|(a,b)| ((*a-*b) as f64).powi(2)).sum::<f64>()
            / baseline.len().min(audio.len()).max(1) as f64).sqrt();
        println!("PROBE decoder comparison first={} empty_flush={} same_length={} max_abs_diff={:.9} rms_diff={:.9}",
            first, empty_flush, baseline.len() == audio.len(), max_diff, rms_diff);
    }
    Ok(())
}

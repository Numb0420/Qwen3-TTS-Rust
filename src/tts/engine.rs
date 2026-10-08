use crate::assets_manager::Assets;
use crate::models::llama::{LlamaBatch, LlamaContext, LlamaModel, LlamaSampler};
use crate::models::onnx::{init_onruntime, AudioDecoder, AudioEncoder, DecoderState, SpeakerEncoder};
use crate::tts::prompt::PromptBuilder;
use crate::tts::short_phrase::{ShortPhraseAudio, ShortPhrasePolicy, NATURAL_PACE_INSTRUCTION};
use crate::utils::cache;
use crate::utils::tokenizer::Tokenizer;
use crate::utils::voice_file::VoiceFile;
use crate::AudioSample;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Sampler configuration for TTS generation
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for sampling (higher = more random, 0.0 = greedy)
    pub temperature: f32,
    /// Top-K sampling (0 = disabled)
    pub top_k: i32,
    /// Top-P (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Min-P sampling threshold (0.0 = disabled)
    pub min_p: f32,
    /// Repeat penalty (1.0 = disabled)
    pub repeat_penalty: f32,
    /// Frequency penalty (0.0 = disabled)
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled)
    pub presence_penalty: f32,
    /// Number of recent tokens to consider for penalties
    pub penalty_last_n: usize,
    /// Random seed (None = use system entropy)
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            seed: None,
        }
    }
}

impl SamplerConfig {
    pub fn new(temperature: f32, top_k: i32, top_p: f32, seed: Option<u64>) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            repeat_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            seed,
        }
    }

    pub fn with_penalties(
        mut self,
        min_p: f32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        penalty_last_n: usize,
    ) -> Self {
        self.min_p = min_p;
        self.repeat_penalty = repeat_penalty;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
        self.penalty_last_n = penalty_last_n;
        self
    }
}

/// Main TTS Engine Struct
///
/// IMPORTANT: Field ordering matters for Drop!
/// Rust drops fields in declaration order. Contexts MUST be declared before models
/// because context destructors reference model memory. If models are dropped first,
/// context destructors will access freed memory (ACCESS_VIOLATION).
pub struct TtsEngine {
    assets: Assets,
    tokenizer: Tokenizer,
    // ONNX Models
    encoder: Option<AudioEncoder>,
    speaker_encoder: Option<SpeakerEncoder>,
    decoder: Arc<Mutex<AudioDecoder>>,
    state_only_decoder: Option<Arc<Mutex<AudioDecoder>>>,
    // Isolated exact-PCM waveform-prefix experiment, absent in normal startup.
    diagnostic_wave_prefix_decoder: Option<Arc<Mutex<AudioDecoder>>>,
    diagnostic_wave_prefix_control_decoder: Option<Arc<Mutex<AudioDecoder>>>,
    // Llama: Contexts MUST be listed before models for correct drop order
    talker_ctx: LlamaContext,
    predictor_ctx: LlamaContext,
    predictor_head_range_enabled: bool,
    talker_model: LlamaModel,
    predictor_model: LlamaModel,

    // Speakers Cache
    speakers: HashMap<String, VoiceFile>,
    decoder_context_cache: HashMap<Vec<i64>, DecoderContext>,

    // Config
    _model_dir: PathBuf,
    max_steps: usize,
    sampler_config: SamplerConfig,
}

#[derive(Clone)]
struct DecoderContext {
    state: DecoderState,
    skip_prefix_samples: usize,
    state_only_first_samples: Option<usize>,
}

impl TtsEngine {
    /// Same values as `LlamaContext::new` for the talker (prefill must respect both).
    const TALKER_N_CTX: usize = 4096;
    const TALKER_N_BATCH: usize = 2048;

    /// Initialize the TTS Engine from the specified model directory.
    ///
    /// This function loads all necessary models (GGUF, Onnx, Tokenizer) from the given directory.
    /// It ensures that the essential components for inference are present.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to the directory containing model files.
    /// * `quant` - Quantization level (e.g., "none", "q5_k_m", "q8_0").
    /// * `n_threads` - Number of threads to use for generation (default: 4 if <= 0).
    pub async fn new(
        model_dir: impl AsRef<Path>,
        quant: &str,
        _n_threads: i32,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        println!("Loading TtsEngine from: {:?} (quant: {})", model_dir, quant);

        // 0. Auto-download check (Models + Runtimes)
        Self::download_models(model_dir, quant).await?;

        let quant_dir = match quant {
            "q5_k_m" => "gguf_q5_k_m",
            "q8_0" => "gguf_q8_0",
            _ => "gguf",
        };

        // 1. Assets - 总是从未量化的 gguf 目录加载（assets 不应该被量化）
        let assets_path = std::env::var_os("QWEN_PRONUNCIATION_ASSETS")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join("gguf"));
        let assets =
            Assets::load(&assets_path).map_err(|e| format!("Failed to load assets: {}", e))?;

        // 2. Tokenizer
        let tokenizer =
            Tokenizer::load(model_dir).map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // 3. Initialize ONNX Runtime (must be called before any ONNX session)
        println!("Initializing ONNX Runtime...");
        init_onruntime().map_err(|e| format!("Failed to init ONNX Runtime: {}", e))?;

        // 4. ONNX Models (Optional for preset mode, but good to have)
        let onnx_dir = model_dir.join("onnx");
        let encoder = AudioEncoder::load(
            &onnx_dir
                .join("qwen3_tts_codec_encoder.onnx")
                .to_string_lossy(),
        )
        .ok();

        let speaker_encoder = SpeakerEncoder::load(
            &onnx_dir
                .join("qwen3_tts_speaker_encoder.onnx")
                .to_string_lossy(),
        )
        .ok();

        // 5. Load GGUF Models
        let talker_path = model_dir.join(quant_dir).join("qwen3_tts_talker.gguf");
        let predictor_path = model_dir.join(quant_dir).join("qwen3_tts_predictor.gguf");

        let diagnostic_gpu_layers = if std::env::var("QWEN_PRONUNCIATION_DEVICE").as_deref() == Ok("cpu") { 0 } else { 99 };
        let talker_model = LlamaModel::load(&talker_path, diagnostic_gpu_layers)
            .map_err(|e| format!("Failed to load Talker: {}", e))?;

        let predictor_model = LlamaModel::load(&predictor_path, diagnostic_gpu_layers)
            .map_err(|e| format!("Failed to load Predictor: {}", e))?;

        // 5. Create Contexts
        // talker: n_ctx=4096, n_batch=2048, embeddings=1, threads=-1 (auto)
        let diagnostic_threads = std::env::var("QWEN_LLAMA_THREADS").ok().and_then(|v| v.parse::<i32>().ok());
        let talker_ctx = LlamaContext::new(&talker_model, 4096, 2048, 1, diagnostic_threads.unwrap_or(-1))
            .map_err(|e| format!("Failed to create Talker context: {}", e))?;

        // predictor: n_ctx=512, n_batch=32, embeddings=0, threads=4
        let predictor_ctx = LlamaContext::new(&predictor_model, 512, 32, 0, diagnostic_threads.unwrap_or(4))
            .map_err(|e| format!("Failed to create Predictor context: {}", e))?;
        let predictor_head_range_enabled = if std::env::var_os("QWEN_PREDICTOR_HEAD_RANGE").is_some() {
            if let Some(scope) = std::env::var_os("QWEN_PREDICTOR_HEAD_MODEL_DIR") {
                let scope = PathBuf::from(scope).canonicalize().map_err(|e| format!("Predictor model scope: {e}"))?;
                model_dir.canonicalize().map_err(|e| format!("Predictor model directory: {e}"))? == scope
            } else { true }
        } else { false };
        if predictor_head_range_enabled {
            println!("[PredictorHeadRange] enabled for {:?}", model_dir);
        }

        // 6. 预加载 decoder（预热）
        println!("Pre-loading AudioDecoder...");
        let decoder =
            AudioDecoder::load(&onnx_dir.join("qwen3_tts_decoder.onnx").to_string_lossy())
                .map_err(|e| format!("Failed to load AudioDecoder: {}", e))?;
        let decoder = Arc::new(Mutex::new(decoder));
        let diagnostic_wave_prefix_decoder = std::env::var("QWEN_DIAG_WAVE_PREFIX_MODEL").ok()
            .map(|path| AudioDecoder::load(&path).map(|decoder| Arc::new(Mutex::new(decoder))))
            .transpose().map_err(|error| format!("Diagnostic waveform graph: {error}"))?;
        let diagnostic_wave_prefix_control_decoder = std::env::var("QWEN_DIAG_WAVE_PREFIX_CONTROL_MODEL").ok()
            .map(|path| AudioDecoder::load(&path).map(|decoder| Arc::new(Mutex::new(decoder))))
            .transpose().map_err(|error| format!("Diagnostic control waveform graph: {error}"))?;
        println!("AudioDecoder pre-loaded and warmed up.");

        println!("TtsEngine loaded successfully.");

        let mut engine = Self {
            assets,
            tokenizer,
            encoder,
            speaker_encoder,
            decoder,
            state_only_decoder: None,
            diagnostic_wave_prefix_decoder,
            diagnostic_wave_prefix_control_decoder,
            talker_model,
            predictor_model,
            talker_ctx,
            predictor_ctx,
            predictor_head_range_enabled,
            speakers: HashMap::new(),
            decoder_context_cache: HashMap::new(),
            _model_dir: model_dir.to_path_buf(),
            max_steps: 4096,
            sampler_config: SamplerConfig::default(),
        };

        // 6. Load Speakers
        let speakers_dir = model_dir.join("preset_speakers"); // Default to preset directory
        let speakers_dir = if speakers_dir.exists() {
            speakers_dir
        } else {
            PathBuf::from("speakers")
        };

        if speakers_dir.exists() {
            engine.load_speakers(&speakers_dir)?;
        }

        Ok(engine)
    }

    /// Set the maximum number of generation steps (tokens).
    pub fn set_max_steps(&mut self, steps: usize) {
        self.max_steps = steps;
    }

    /// Set the sampler configuration for generation.
    pub fn set_sampler_config(&mut self, config: SamplerConfig) {
        self.sampler_config = config;
    }

    /// Get the current sampler configuration.
    pub fn get_sampler_config(&self) -> &SamplerConfig {
        &self.sampler_config
    }

    /// Get mutable sampler configuration.
    pub fn get_sampler_config_mut(&mut self) -> &mut SamplerConfig {
        &mut self.sampler_config
    }

    /// Get the speakers map.
    pub fn get_speakers_map(&self) -> &HashMap<String, VoiceFile> {
        &self.speakers
    }

    /// Limit a single-checkpoint server to voices explicitly validated for it.
    pub fn retain_speakers(&mut self, names: &[String]) -> Result<(), String> {
        if names.is_empty() { return Err("At least one validated speaker is required".into()); }
        for name in names {
            if !self.speakers.contains_key(name) { return Err(format!("Speaker not found: {name}")); }
        }
        self.speakers.retain(|name, _| names.contains(name));
        println!("Validated speakers enabled: {:?}", names);
        Ok(())
    }

    /// Load all speakers from the specified directory.
    pub fn load_speakers(&mut self, speakers_dir: impl AsRef<Path>) -> Result<(), String> {
        let speakers_dir = speakers_dir.as_ref();
        println!("Loading speakers from: {:?}", speakers_dir);

        let entries = std::fs::read_dir(speakers_dir).map_err(|e| e.to_string())?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(voice) = VoiceFile::load(&path) {
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    self.speakers.insert(id, voice);
                }
            }
        }
        println!("Loaded {} speakers.", self.speakers.len());
        Ok(())
    }

    /// Decode ICL reference audio once at startup so first audio can be sent
    /// without repeating several seconds of CPU decoder work per request.
    pub fn prewarm_decoder_contexts(&mut self) -> Result<(), String> {
        let references: Vec<Vec<i64>> = self
            .speakers
            .values()
            .filter(|voice| !voice.audio_codes.is_empty())
            .map(|voice| voice.audio_codes.clone())
            .collect();
        for reference in references {
            self.decoder_context_for(&reference)?;
        }
        Ok(())
    }

    pub fn prewarm_codec_projections(&self) {
        let started = Instant::now();
        self.assets.prewarm_codec_projections();
        println!("Codec projection cache ready in {} ms (about 120 MiB).", started.elapsed().as_millis());
    }

    /// Load an optional pruned graph for explicitly routed speakers. A real
    /// normal decode calibrates the omitted sample count and verifies every
    /// state tensor. This affects recurring requests, not just startup.
    pub fn enable_state_only_decoder(&mut self, path: &Path, speakers: &[String]) -> Result<(), String> {
        let mut partial = AudioDecoder::load(&path.to_string_lossy()).map_err(|e| e.to_string())?;
        partial.validate_state_only_graph().map_err(|e| e.to_string())?;
        for name in speakers {
            let reference = self.speakers.get(name)
                .ok_or_else(|| format!("State-only speaker not found: {name}"))?.audio_codes.clone();
            if reference.is_empty() { continue; }
            let context = self.decoder_context_for(&reference)?;
            // A partial final reference chunk leaves 1-4 convolution-history
            // frames while retaining the same four-frame lookahead. Real-code
            // replay covers these shapes; still calibrate every loaded voice.
            let history_shape = context.state.conv_history.shape();
            if context.state.latent_buffer.shape() != [1,1024,4]
                || history_shape[0] != 1 || history_shape[1] != 1024
                || !(1..=4).contains(&history_shape[2]) { continue; }
            let (mut normal_state, mut partial_state) = (context.state.clone(), context.state.clone());
            let codes = [0i64;64];
            let raw = self.decoder.lock().unwrap().decode(&codes, &mut normal_state, false)
                .map_err(|e| format!("State-only calibration failed: {e}"))?;
            partial.advance_state(&codes, &mut partial_state).map_err(|e| e.to_string())?;
            if !normal_state.bitwise_eq(&partial_state) {
                return Err(format!("State-only decoder state differs for speaker {name}; not enabled"));
            }
            // Tensor shapes depend on chunk/history sizes, not codec values.
            // Use the measured original output count rather than guessing it.
            if !raw.is_empty() && raw.len() <= context.skip_prefix_samples {
                self.decoder_context_cache.get_mut(&reference).unwrap().state_only_first_samples = Some(raw.len());
                println!("[StateOnlyDecoder] speaker={name} calibrated_reference_samples={} history_frames={}", raw.len(), history_shape[2]);
            }
        }
        self.state_only_decoder = Some(Arc::new(Mutex::new(partial)));
        Ok(())
    }

    fn decoder_context_for(&mut self, reference: &[i64]) -> Result<DecoderContext, String> {
        if let Some(cached) = self.decoder_context_cache.get(reference) {
            return Ok(cached.clone());
        }
        let mut state = AudioDecoder::create_state();
        let mut emitted_samples = 0usize;
        for chunk in reference.chunks(16 * 4) {
            let mut decoder = self.decoder.lock().unwrap();
            let samples = decoder
                .decode(chunk, &mut state, false)
                .map_err(|error| format!("Reference decoder prefill failed: {error}"))?;
            emitted_samples += samples.len();
        }
        let context = DecoderContext {
            state,
            skip_prefix_samples: (reference.len() / 16 * 1920).saturating_sub(emitted_samples),
            state_only_first_samples: None,
        };
        self.decoder_context_cache
            .insert(reference.to_vec(), context.clone());
        Ok(context)
    }

    /// Get a speaker by ID or name, with fallback to "vivian".
    pub fn get_speaker(&self, id_or_name: &str) -> &VoiceFile {
        if let Some(v) = self.speakers.get(id_or_name) {
            return v;
        }
        // Fallback to name match
        for v in self.speakers.values() {
            if let Some(ref name) = v.name {
                if name == id_or_name {
                    return v;
                }
            }
        }
        // Final fallback to vivian
        self.speakers.get("vivian").unwrap_or_else(|| {
            // Panic if even vivian is missing and cache is empty
            self.speakers
                .values()
                .next()
                .expect("No speakers loaded in engine!")
        })
    }

    /// Helper to download necessary files before loading.
    pub async fn download_models(model_dir: impl AsRef<Path>, quant: &str) -> Result<(), String> {
        let downloader = crate::download::Downloader::new().await;
        downloader
            .check_and_download(model_dir.as_ref(), quant)
            .await
            .map_err(|e| format!("Download failed: {}", e))
    }

    /// Generate speech from text using a reference audio.
    pub fn generate(
        &mut self,
        text: &str,
        ref_audio_path: impl AsRef<Path>,
        ref_text: &str,
        instruct: Option<&str>,
    ) -> Result<AudioSample, String> {
        let ref_audio_path = ref_audio_path.as_ref();

        // 1. Process Reference Audio
        let (ref_codes, spk_emb) = self.process_reference(ref_audio_path)?;

        // 2. Build Prompt
        // lang_id = 2055 (Chinese) hardcoded for now or parameterize later
        let ref_text_ids = self.tokenizer.encode(ref_text);
        let ref_codes_i32: Vec<i32> = ref_codes.iter().map(|&c| c as i32).collect();

        let prompt_data = PromptBuilder::build_clone_prompt(
            text,
            &self.tokenizer,
            &self.assets,
            &ref_codes_i32,
            &ref_text_ids,
            &spk_emb,
            2055,
            instruct,
        );

        let decoder_context = self.decoder_context_for(&ref_codes)?;
        self.run_inference_stream(prompt_data, None, None, Some(decoder_context), Instant::now())
    }

    /// Process reference audio to get codes and speaker embedding, using cache if available.
    fn process_reference(&mut self, audio_path: &Path) -> Result<(Vec<i64>, Vec<f32>), String> {
        let cache_path = audio_path.with_extension("cache");
        if cache_path.exists() {
            if let Ok((c, e)) = cache::load_cache(&cache_path) {
                return Ok((c, e));
            }
        }

        let audio = AudioSample::load_wav(audio_path)
            .map_err(|e| format!("Failed to load audio: {}", e))?;

        let ref_codes = self
            .encoder
            .as_mut()
            .ok_or("AudioEncoder not loaded (required for processing raw audio)".to_string())?
            .encode(&audio.samples)
            .map_err(|e| format!("Audio encode failed: {}", e))?;
        let spk_emb = self
            .speaker_encoder
            .as_mut()
            .ok_or("SpeakerEncoder not loaded (required for processing raw audio)".to_string())?
            .encode(&audio.samples)
            .map_err(|e| format!("Speaker extraction failed: {}", e))?;

        let _ = cache::save_cache(&cache_path, &ref_codes, &spk_emb);

        Ok((ref_codes, spk_emb))
    }

    // --- Helpers ---

    fn qwen3_position(start: i32, len: usize) -> Vec<i32> {
        let mut pos = Vec::with_capacity(len * 4);
        let range: Vec<i32> = (start..start + len as i32).collect();
        pos.extend_from_slice(&range); // Temporal
        pos.extend_from_slice(&range); // Height
        pos.extend_from_slice(&range); // Width
        pos.extend(std::iter::repeat_n(0, len)); // Channel
        pos
    }

    fn normal_position(cur_pos: usize, n_tokens: usize) -> Vec<i32> {
        (0..n_tokens).map(|i| (cur_pos + i) as i32).collect()
    }

    /// Create a VoiceFile from a reference audio file and its text.
    ///
    /// Requires that AudioEncoder and SpeakerEncoder are loaded.
    /// The reference audio MUST be 24000Hz.
    pub fn create_voice_file(
        &mut self,
        audio_path: impl AsRef<Path>,
        ref_text: String,
    ) -> Result<crate::utils::voice_file::VoiceFile, String> {
        let encoder = self.encoder.as_mut().ok_or(
            "AudioEncoder not loaded. Please ensure models/onnx/qwen3_tts_codec_encoder.onnx exists.",
        )?;
        let speaker_encoder = self.speaker_encoder.as_mut().ok_or(
            "SpeakerEncoder not loaded. Please ensure models/onnx/qwen3_tts_speaker_encoder.onnx exists.",
        )?;

        // 1. Load Audio
        let mut reader =
            hound::WavReader::open(audio_path).map_err(|e| format!("WAV error: {}", e))?;
        let spec = reader.spec();

        if spec.sample_rate != 24000 {
            return Err(format!(
                "Expected 24000Hz audio, found {}Hz",
                spec.sample_rate
            ));
        }

        let audio: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|s| (s.unwrap_or(0) as f32) / 32768.0)
                .collect(),
            (hound::SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .map(|s| (s.unwrap_or(0) as f32) / 2147483648.0)
                .collect(),
            _ => {
                return Err(format!(
                    "Unsupported WAV format: {:?} {} bits",
                    spec.sample_format, spec.bits_per_sample
                ))
            }
        };

        // If stereo, take channel 1
        let audio = if spec.channels > 1 {
            audio.chunks(spec.channels as usize).map(|c| c[0]).collect()
        } else {
            audio
        };

        // 2. Run Encoders
        println!("Extracting audio codes...");
        let audio_codes = encoder.encode(&audio).map_err(|e| e.to_string())?;

        println!("Extracting speaker embedding...");
        let speaker_embedding = speaker_encoder.encode(&audio).map_err(|e| e.to_string())?;

        Ok(crate::utils::voice_file::VoiceFile::new(
            ref_text,
            audio_codes,
            speaker_embedding,
        ))
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
        instruct: Option<&str>,
    ) -> Result<AudioSample, String> {
        self.generate_with_voice_streaming(text, voice, instruct, None)
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice_streaming(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
        instruct: Option<&str>,
        stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
    ) -> Result<AudioSample, String> {
        let profile = std::env::var_os("QWEN_TTFA_PROFILE").is_some();
        let prompt_started = Instant::now();
        let short_phrase = (voice.audio_codes.is_empty() && instruct.is_none())
            .then(|| ShortPhrasePolicy::for_text(text))
            .flatten();
        let effective_instruct = if short_phrase.is_some() {
            Some(NATURAL_PACE_INSTRUCTION)
        } else {
            instruct
        };
        let prompt_data = if voice.audio_codes.is_empty() && voice.spk_id.is_some() {
            PromptBuilder::build_custom_prompt(
                text,
                &self.tokenizer,
                &self.assets,
                voice.spk_id.unwrap(),
                2055,
                effective_instruct,
            )
        } else if voice.audio_codes.is_empty() {
            PromptBuilder::build_core(
                text,
                &self.tokenizer,
                &self.assets,
                Some(2055),
                None,
                Some(&voice.speaker_embedding),
                effective_instruct,
                None,
            )
        } else {
            let ref_text_ids = self.tokenizer.encode(&voice.ref_text);
            let ref_codes_i32: Vec<i32> = voice.audio_codes.iter().map(|&c| c as i32).collect();

            Ok(PromptBuilder::build_clone_prompt(
                text,
                &self.tokenizer,
                &self.assets,
                &ref_codes_i32,
                &ref_text_ids,
                &voice.speaker_embedding,
                2055,
                effective_instruct,
            ))
        }?;

        if profile {
            eprintln!("[TTFA] prompt_ms={} prompt_tokens={} ref_frames={}", prompt_started.elapsed().as_millis(), prompt_data.embd.len(), voice.audio_codes.len() / 16);
        }

        let context_started = Instant::now();
        let decoder_context = if voice.audio_codes.is_empty() {
            None
        } else {
            Some(self.decoder_context_for(&voice.audio_codes)?)
        };
        if profile {
            eprintln!("[TTFA] context_clone_ms={:.3}", context_started.elapsed().as_secs_f64() * 1000.0);
        }
        self.run_inference_stream(
            prompt_data,
            stream_tx,
            short_phrase,
            decoder_context,
            prompt_started,
        )
    }

    fn run_inference_stream(
        &mut self,
        mut prompt_data: crate::tts::prompt::PromptData,
        stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
        short_phrase: Option<ShortPhrasePolicy>,
        decoder_context: Option<DecoderContext>,
        request_started: Instant,
    ) -> Result<AudioSample, String> {
        let quiet_debug = crate::utils::performance::enabled("quiet");
        let cancellation_tx = stream_tx.clone();
        let is_cancelled = || cancellation_tx.as_ref().is_some_and(|tx| tx.is_closed());
        if is_cancelled() { return Err("Streaming request cancelled".into()); }
        let profile = std::env::var_os("QWEN_TTFA_PROFILE").is_some();
        let inference_started = Instant::now();
        // One read per request for an isolated same-process A/B/A experiment.
        if let Some(path) = std::env::var_os("QWEN_LAST_COPY_MODE_FILE") {
            let mode = std::fs::read_to_string(path).map_err(|e| format!("Last-copy diagnostic mode: {e}"))?;
            if !matches!(mode.trim(), "0" | "1") { return Err("Invalid last-copy diagnostic mode".into()); }
            std::env::set_var("QWEN_TALKER_LAST_COPY", mode.trim());
        }
        crate::models::llama::reset_output_projection_profile();
        let clear_data = !crate::utils::performance::enabled("metadata");
        self.talker_ctx.clear_memory_with_data(clear_data);
        self.predictor_ctx.clear_memory_with_data(clear_data);

        if prompt_data.embd.len() > Self::TALKER_N_CTX {
            eprintln!(
                "WARNING: prompt length {} exceeds talker n_ctx {}. Truncating from the end of the fused prompt.",
                prompt_data.embd.len(),
                Self::TALKER_N_CTX
            );
            prompt_data.embd.truncate(Self::TALKER_N_CTX);
        }

        let n_tokens_prompt = prompt_data.embd.len();
        let prompt_embeds_flat: Vec<f32> = prompt_data.embd.iter().flatten().copied().collect();
        let talker_embd = self.talker_model.n_embd;
        let predictor_embd = self.predictor_model.n_embd;

        let mut talker_batch = LlamaBatch::new(Self::TALKER_N_BATCH, talker_embd, 1, 4);
        let mut talker_prefill_last_idx: i32 = 0;
        let mut prefill_off = 0usize;
        while prefill_off < n_tokens_prompt {
            if is_cancelled() {
                self.talker_ctx.synchronize();
                return Err("Streaming request cancelled".into());
            }
            let chunk_tokens = (n_tokens_prompt - prefill_off).min(Self::TALKER_N_BATCH);
            let slice_start = prefill_off * talker_embd;
            let slice_end = (prefill_off + chunk_tokens) * talker_embd;
            let chunk_flat = &prompt_embeds_flat[slice_start..slice_end];
            let pos_arr = Self::qwen3_position(prefill_off as i32, chunk_tokens);
            talker_batch.set_embd(chunk_flat, &pos_arr, 0);
            self.talker_ctx
                .decode(&mut talker_batch)
                .map_err(|e| format!("Talker prefill failed: {}", e))?;
            talker_prefill_last_idx = (chunk_tokens.saturating_sub(1)) as i32;
            prefill_off += chunk_tokens;
        }
        if profile {
            eprintln!("[TTFA] talker_prefill_ms={} tokens={}", inference_started.elapsed().as_millis(), n_tokens_prompt);
        }
        let prefill_finished_ms = request_started.elapsed().as_secs_f64() * 1000.0;
        let mut frame_profile = Vec::new();
        let mut dispatch_profile = Vec::new();

        let mut all_codes: Vec<i32> = Vec::new();
        let mut cur_pos = n_tokens_prompt;

        let mut predictor_batch = LlamaBatch::new(32, predictor_embd, 1, 1);

        let seed = self.sampler_config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or_else(|_| rand::random())
        });
        // Match the official sub-talker sampling defaults. A separate seeded
        // stream keeps repeated requests reproducible without forcing the
        // fine-code predictor into greedy loops.
        let predictor_sampler =
            LlamaSampler::new(self.predictor_model.n_vocab, 0.9, 50, 1.0, seed ^ 0x9e37_79b9_7f4a_7c15);
        let predictor_legacy = if let Some(path) = std::env::var_os("QWEN_PREDICTOR_SAMPLER_MODE_FILE") {
            let mode = std::fs::read_to_string(path)
                .map_err(|error| format!("Sampler diagnostic mode file failed: {error}"))?;
            match mode.trim() {
                "legacy" => true,
                "range" => false,
                _ => return Err("Sampler diagnostic mode must be legacy or range".into()),
            }
        } else {
            std::env::var_os("QWEN_PREDICTOR_SAMPLER_LEGACY").is_some()
        };
        let predictor_head_range = self.predictor_head_range_enabled;
        let predictor_logits_trace = std::env::var_os("QWEN_PREDICTOR_LOGITS_TRACE_DIR")
            .map(std::path::PathBuf::from);
        if let Some(path) = &predictor_logits_trace {
            std::fs::create_dir_all(path).map_err(|e| format!("Predictor trace directory: {e}"))?;
        }
        let talker_sampler = LlamaSampler::new(
            self.talker_model.n_vocab,
            self.sampler_config.temperature,
            self.sampler_config.top_k,
            self.sampler_config.top_p,
            seed,
        )
        .with_penalties(
            self.sampler_config.min_p,
            self.sampler_config.repeat_penalty,
            self.sampler_config.frequency_penalty,
            self.sampler_config.presence_penalty,
            self.sampler_config.penalty_last_n,
        );

        let (tx, rx) = std::sync::mpsc::channel::<(Vec<i64>, bool, Option<std::sync::mpsc::Receiver<bool>>) >();
        let eos_overlap_enabled = stream_tx.is_some() && short_phrase.is_none()
            && std::env::var("QWEN_DIAG_EOS_MODE_FILE").ok()
                .and_then(|path| std::fs::read_to_string(path).ok())
                .is_some_and(|mode| mode.trim()=="1");
        let mut pending_decode_confirmation: Option<std::sync::mpsc::Sender<bool>> = None;
        // Isolated scheduling experiment: preserve the same codes and 4-frame
        // calls, but stop subsequent generation while the first wanted PCM runs.
        let first_pcm_priority = crate::utils::performance::enabled("first-pcm-priority")
            && stream_tx.is_some() && short_phrase.is_none()
            && decoder_context.as_ref().is_some_and(|context|
                context.state_only_first_samples.is_some_and(|count|
                    count > 0 && count == context.skip_prefix_samples));
        let (first_pcm_ready_tx, first_pcm_ready_rx) = if first_pcm_priority {
            let (sender, receiver) = std::sync::mpsc::channel::<()>();
            (Some(sender), Some(receiver))
        } else { (None, None) };

        let tts_pad = self.assets.tts_pad.clone();
        let decoder_arc = self.decoder.clone();
        let state_only_arc = self.state_only_decoder.clone();
        let state_only_enabled = crate::utils::performance::enabled("state-only");
        let wave_prefix_mode = std::env::var("QWEN_DIAG_WAVE_PREFIX_MODE_FILE").ok()
            .and_then(|path| std::fs::read_to_string(path).ok());
        let wave_prefix_arc = match wave_prefix_mode.as_deref().map(str::trim) {
            Some("1")=>self.diagnostic_wave_prefix_decoder.clone(),
            Some("2")=>self.diagnostic_wave_prefix_control_decoder.clone(),
            _=>None,
        };
        let short_done = Arc::new(AtomicBool::new(false));
        let decoder_short_done = short_done.clone();

        let decoder_handle = std::thread::spawn(move || -> Result<Vec<f32>, String> {
            let mut first_pcm_ready_tx = first_pcm_ready_tx;
            let decoder_started = Instant::now();
            let mut first_pcm_logged = false;
            let mut first_packet_logged = false;
            let mut full_audio = Vec::new();
            let mut decoder_profile = Vec::new();
            let mut decode_number = 0usize;
            let (mut state, mut skip_prefix_samples, state_only_first_samples) = decoder_context
                .map(|context| (context.state, context.skip_prefix_samples, context.state_only_first_samples))
                .unwrap_or_else(|| (AudioDecoder::create_state(), 0, None));
            let mut short_audio = short_phrase.map(|_| ShortPhraseAudio::default());
            if profile {
                eprintln!("[TTFA] decoder_context_ready_ms={}", decoder_started.elapsed().as_millis());
            }

            while let Ok((codes, is_final, confirmation)) = rx.recv() {
                if stream_tx.as_ref().is_some_and(|tx| tx.is_closed()) { break; }
                let n_frames = codes.len() / 16;

                if n_frames == 0 {
                    if is_final {
                        let mut local_decoder = decoder_arc.lock().unwrap();
                        if let Ok(mut samples) = local_decoder.decode(&[], &mut state, true) {
                            let drop_count = skip_prefix_samples.min(samples.len());
                            if drop_count > 0 {
                                samples.drain(..drop_count);
                            }
                            if let Some(filter) = short_audio.as_mut() {
                                samples = filter.filter(&samples);
                                if filter.is_done() {
                                    decoder_short_done.store(true, Ordering::Relaxed);
                                }
                            }
                            if !samples.is_empty() {
                                if let Some(ref stx) = stream_tx {
                                    if !first_packet_logged {
                                        eprintln!(
                                            "[首包延迟] {} ms",
                                            request_started.elapsed().as_millis()
                                        );
                                    }
                                    let _ = stx.send(samples.clone());
                                }
                                full_audio.extend(samples);
                            }
                        }
                        break;
                    }
                    continue;
                }

                let safe_codes: Vec<i64> = codes.iter().map(|&c| c.clamp(0, 2047)).collect();
                let checkpoint = confirmation.as_ref().map(|_| (state.clone(), skip_prefix_samples));

                let received_ms = request_started.elapsed().as_secs_f64() * 1000.0;
                let decode_started_ms = request_started.elapsed().as_secs_f64() * 1000.0;
                let omitted_reference_samples = if state_only_enabled && decode_number == 0
                    && !is_final && n_frames == 4 && state_only_arc.is_some() {
                    state_only_first_samples.filter(|&count| count <= skip_prefix_samples).unwrap_or(0)
                } else { 0 };
                let decoded = if omitted_reference_samples > 0 {
                    state_only_arc.as_ref().unwrap().lock().unwrap()
                        .advance_state(&safe_codes, &mut state)
                        .map_err(|error| format!("State-only decoder failed: {error}"))?;
                    skip_prefix_samples -= omitted_reference_samples;
                    Ok(Vec::new())
                } else if decode_number == 1 && state_only_first_samples == Some(7680)
                    && skip_prefix_samples == 0 && !is_final && n_frames == 4 && wave_prefix_arc.is_some() {
                    wave_prefix_arc.as_ref().unwrap().lock().unwrap().decode(&safe_codes, &mut state, false)
                } else {
                    decoder_arc.lock().unwrap().decode(&safe_codes, &mut state, is_final)
                };
                if let Some(confirmation) = confirmation {
                    // No PCM/state commitment until the producer confirms this
                    // exact block is non-final. EOS/error restores the checkpoint.
                    if !confirmation.recv().unwrap_or(false) {
                        let (previous_state,previous_skip) = checkpoint.unwrap();
                        state=previous_state;skip_prefix_samples=previous_skip;
                        continue;
                    }
                }
                if let Ok(mut samples) = decoded {
                    let decode_finished_ms = request_started.elapsed().as_secs_f64() * 1000.0;
                    let raw_samples = samples.len();
                    let drop_count = skip_prefix_samples.min(samples.len());
                    if drop_count > 0 {
                        samples.drain(..drop_count);
                        skip_prefix_samples -= drop_count;
                    }
                    if let Some(filter) = short_audio.as_mut() {
                        samples = filter.filter(&samples);
                        if filter.is_done() {
                            decoder_short_done.store(true, Ordering::Relaxed);
                        }
                    }
                    decode_number += 1;
                    if profile && decode_number <= 3 {
                        decoder_profile.push(serde_json::json!({
                            "call": decode_number, "frames": n_frames, "received_ms": received_ms,
                            "start_ms": decode_started_ms, "finish_ms": decode_finished_ms,
                            "raw_samples": raw_samples, "target_samples": samples.len(),
                            "omitted_reference_samples": omitted_reference_samples,
                        }));
                    }
                    if !samples.is_empty() {
                        if profile && !first_pcm_logged {
                            eprintln!("[TTFA] first_pcm_from_inference_start_ms={}", inference_started.elapsed().as_millis());
                            first_pcm_logged = true;
                        }
                        if let Some(ref stx) = stream_tx {
                            if !first_packet_logged {
                                first_packet_logged = true;
                                eprintln!(
                                    "[首包延迟] {} ms",
                                    request_started.elapsed().as_millis()
                                );
                            }
                            let _ = stx.send(samples.clone());
                        }
                        full_audio.extend(samples);
                        if let Some(ready) = first_pcm_ready_tx.take() {
                            let _ = ready.send(());
                        }
                    }
                    if short_audio.as_ref().is_some_and(ShortPhraseAudio::is_done) {
                        break;
                    }
                }
            }
            if profile {
                eprintln!("[TTFA:decoder-timeline] {}", serde_json::json!(decoder_profile));
            }
            Ok(full_audio)
        });

        // 连续静音检测参数 (12.5Hz = 80ms/帧)
        const MAX_SILENT_FRAMES: usize = 15; // 1.2秒强制停止
        const SILENT_PENALTY_THRESHOLD: usize = 4; // 0.3秒(4帧)后开始惩罚
        const SILENT_PENALTY_MAX_FRAMES: usize = 8; // 0.6秒(8帧)后极大惩罚
        const SILENT_PENALTY_VALUE: f32 = 2.0;

        // Isolated pronunciation probes only; absent in normal server startup.
        let pronunciation_probe = std::env::var("QWEN_PRONUNCIATION_PROBE").unwrap_or_default();
        let probe_no_silence = pronunciation_probe == "no-silence";
        let probe_official_mask = pronunciation_probe == "official-mask";
        let mut silence_penalty_calls = 0usize;

        // 流式发送参数
        const SEND_INTERVAL: usize = 4;
        let mut consecutive_silent_frames: usize = 0;
        let mut last_sent_frame: usize = 0;
        let mut previous_code_0: Option<i32> = None;
        let mut repeated_code_0 = 0usize;
        let mut stalled_code_0: Option<i32> = None;
        let mut generation_finished = false;
        let mut end_code = None;

        let generation_limit = short_phrase
            .map(|policy| self.max_steps.min(policy.hard_limit))
            .unwrap_or(self.max_steps);
        let mut cancelled = false;
        for step in 0..generation_limit {
            if is_cancelled() { cancelled = true; break; }
            let frame_started = Instant::now();
            if short_done.load(Ordering::Relaxed) {
                println!("\n    Short phrase ended after audible speech");
                generation_finished = true;
                break;
            }
            if !quiet_debug {
                print!("\r    Generation Step {}/{}...", step + 1, self.max_steps);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }

            // Talker
            let sample_idx = if cur_pos == n_tokens_prompt {
                talker_prefill_last_idx
            } else {
                0
            };

            let allow_tokens: Vec<i32> = if probe_official_mask { vec![2150] } else { vec![2150, 2148, 2149] };

            let code_0 = if !probe_no_silence && consecutive_silent_frames >= SILENT_PENALTY_THRESHOLD {
                silence_penalty_calls += 1;
                let penalty = if consecutive_silent_frames >= SILENT_PENALTY_MAX_FRAMES {
                    100.0 // 极大惩罚，强制退出静音
                } else {
                    SILENT_PENALTY_VALUE
                        * (1.0
                            + (consecutive_silent_frames - SILENT_PENALTY_THRESHOLD) as f32 * 0.5)
                };
                if !quiet_debug { eprintln!("[Debug] silent {} frames, penalty: {:.1}", consecutive_silent_frames, penalty); }
                talker_sampler.sample_with_silent_penalty(
                    &self.talker_ctx,
                    sample_idx,
                    Some(0),
                    Some(2048),
                    Some(&allow_tokens),
                    penalty,
                    SILENT_PENALTY_MAX_FRAMES, // 只对前8帧应用惩罚
                )
            } else {
                talker_sampler.sample_with_allow(
                    &self.talker_ctx,
                    sample_idx,
                    Some(0),
                    Some(2048),
                    Some(&allow_tokens),
                )
            };
            if !quiet_debug { eprintln!("[Debug] Sampled code_0={}", code_0); }
            let sampled_ms = frame_started.elapsed().as_secs_f64() * 1000.0;

            if code_0 == 2150 || code_0 == 151673 {
                println!("\n    EOS detected at step {} (code_0={})", step, code_0);
                generation_finished = true;
                end_code = Some(code_0);
                break;
            }

            repeated_code_0 = if previous_code_0 == Some(code_0) {
                repeated_code_0 + 1
            } else {
                1
            };
            previous_code_0 = Some(code_0);
            if repeated_code_0 >= 16 {
                eprintln!("[Generation] Stopping repeated codec token {} after {} frames", code_0, repeated_code_0);
                stalled_code_0 = Some(code_0);
                break;
            }

            let is_silent_frame = code_0 < 100;
            if is_silent_frame {
                consecutive_silent_frames += 1;
                if !quiet_debug { eprintln!("[Debug] silent frame {}, code_0={}", consecutive_silent_frames, code_0); }
                if !probe_no_silence && consecutive_silent_frames >= MAX_SILENT_FRAMES {
                    eprintln!(
                        "[Debug] Early stop: {} consecutive silent frames",
                        consecutive_silent_frames
                    );
                    println!(
                        "\n    Early stop: {} consecutive silent frames detected",
                        consecutive_silent_frames
                    );
                    generation_finished = true;
                    break;
                }
            } else {
                consecutive_silent_frames = 0;
            }

            // Release complete chunks after the next talker token establishes
            // that this is not EOS, but before its 15-step predictor. Keeping
            // the final chunk for EOS preserves the decoder's final flag and
            // avoids an unsupported empty-input flush at multiples of four.
            let current_frame = all_codes.len() / 16;
            if current_frame >= last_sent_frame + SEND_INTERVAL {
                let start_idx = last_sent_frame * 16;
                let end_idx = (last_sent_frame + SEND_INTERVAL) * 16;
                let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                    .iter().map(|&c| c as i64).collect();
                if let Some(confirmation) = pending_decode_confirmation.take() {
                    let _ = confirmation.send(true);
                } else {
                    let _ = tx.send((frame_codes, false, None));
                }
                last_sent_frame += SEND_INTERVAL;
                if profile && dispatch_profile.len() < 3 {
                    dispatch_profile.push(serde_json::json!({"frames": last_sent_frame, "at_ms": request_started.elapsed().as_secs_f64() * 1000.0}));
                }
                if first_pcm_priority && last_sent_frame == 8 {
                    let wait_started = Instant::now();
                    let ready = first_pcm_ready_rx.as_ref().unwrap()
                        .recv_timeout(std::time::Duration::from_secs(3));
                    if profile {
                        eprintln!("[FirstPcmPriority] wait_ms={:.4} ready={}",
                            wait_started.elapsed().as_secs_f64() * 1000.0, ready.is_ok());
                    }
                    if ready.is_err() {
                        eprintln!("[FirstPcmPriority] no acknowledgement; resume normal generation");
                    }
                }
            }

            all_codes.push(code_0);

            // Predictor
            let emb_idx = if step == 0 {
                talker_prefill_last_idx as usize
            } else {
                0
            };
            let m_hidden = self.talker_ctx.get_embedding_at(emb_idx).to_vec();

            // Isolated equivalence gate; disabled in normal synthesis and timing.
            if step < 8 {
                if let Some(directory) = std::env::var_os("QWEN_TALKER_TRACE_DIR") {
                    let directory = PathBuf::from(directory);
                    std::fs::create_dir_all(&directory).map_err(|e| format!("Talker trace directory: {e}"))?;
                    let hidden: Vec<u8> = m_hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let logits: Vec<u8> = self.talker_ctx.get_logits_ith(emb_idx)
                        .iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(directory.join(format!("frame-{step:03}-hidden.f32")), hidden)
                        .map_err(|e| format!("Talker hidden trace: {e}"))?;
                    std::fs::write(directory.join(format!("frame-{step:03}-logits.f32")), logits)
                        .map_err(|e| format!("Talker logits trace: {e}"))?;
                }
            }

            let m_h_1024 = self.assets.project(&m_hidden);
            let code_0_1024 = self.assets.get_codec_embedding_1024(0, code_0);
            let projection_finished = frame_started.elapsed().as_secs_f64() * 1000.0;

            let mut predictor_input = Vec::with_capacity(2 * predictor_embd);
            predictor_input.extend_from_slice(&m_h_1024);
            predictor_input.extend_from_slice(&code_0_1024);

            self.predictor_ctx.clear_kv_cache();
            predictor_batch.clear();
            let pred_pos = Self::normal_position(0, 2);
            predictor_batch.set_embd(&predictor_input, &pred_pos, 0);

            if predictor_head_range { self.predictor_ctx.set_predictor_output_range(0)?; }
            self.predictor_ctx
                .decode(&mut predictor_batch)
                .map_err(|e| format!("Predictor prefill failed: {}", e))?;
            let predictor_prefill_finished = frame_started.elapsed().as_secs_f64() * 1000.0;
            let mut predictor_sample_ms = 0.0;
            let mut predictor_decode_ms = 0.0;

            let mut step_embeds_2048: Vec<Vec<f32>> = Vec::new();
            step_embeds_2048.push(self.assets.get_codec_embedding(0, code_0));

            for q in 1..16 {
                let start_offset = (q - 1) * 2048;
                let end_offset = q * 2048;
                if step < 8 {
                    if let Some(path) = &predictor_logits_trace {
                        let values = &self.predictor_ctx.get_logits_ith(if q == 1 { 1 } else { 0 })[start_offset..end_offset];
                        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
                        std::fs::write(path.join(format!("frame-{step:03}-head-{q:02}.f32")), bytes)
                            .map_err(|e| format!("Predictor logits trace: {e}"))?;
                    }
                }
                let sampling_started = (profile && step < 9).then(Instant::now);
                let sampled = if predictor_legacy {
                    predictor_sampler.sample(
                        &self.predictor_ctx, if q == 1 { 1 } else { 0 },
                        Some(start_offset), Some(end_offset),
                    )
                } else {
                    predictor_sampler.sample_predictor_range(
                        &self.predictor_ctx, if q == 1 { 1 } else { 0 },
                        start_offset, end_offset,
                    )
                };
                if let Some(started) = sampling_started { predictor_sample_ms += started.elapsed().as_secs_f64() * 1000.0; }
                let code_q = sampled - start_offset as i32;
                all_codes.push(code_q);

                let emb = self.assets.get_codec_embedding(q, code_q);
                step_embeds_2048.push(emb.to_vec());

                if q < 15 {
                    let next_embed_1024 = self.assets.get_codec_embedding_1024(q, code_q);
                    let next_pos = Self::normal_position(q + 1, 1);
                    predictor_batch.clear();
                    predictor_batch.set_embd(&next_embed_1024, &next_pos, 0);
                    let decoding_started = (profile && step < 9).then(Instant::now);
                    if predictor_head_range { self.predictor_ctx.set_predictor_output_range(q * 2048)?; }
                    self.predictor_ctx
                        .decode(&mut predictor_batch)
                        .map_err(|e| format!("Predictor decode failed: {}", e))?;
                    if let Some(started) = decoding_started { predictor_decode_ms += started.elapsed().as_secs_f64() * 1000.0; }
                }
            }
            let predictor_finished = frame_started.elapsed().as_secs_f64() * 1000.0;
            // Isolated trial: overlap only the first two blocks with the next
            // talker decode/sampling. The existing EOS branch owns confirmation.
            if eos_overlap_enabled && (step==3 || step==7) && pending_decode_confirmation.is_none()
                && all_codes.len()/16 == last_sent_frame+SEND_INTERVAL {
                let start_idx=last_sent_frame*16;
                let end_idx=(last_sent_frame+SEND_INTERVAL)*16;
                let frame_codes=all_codes[start_idx..end_idx].iter().map(|&c|c as i64).collect();
                let (confirmation,receiver)=std::sync::mpsc::channel();
                let _=tx.send((frame_codes,false,Some(receiver)));
                pending_decode_confirmation=Some(confirmation);
            }

            if step == 7 {
                if let Some(profile) = predictor_sampler.sampling_profile() {
                    eprintln!("[SamplerProfile:first8] {}", serde_json::to_string(&profile).unwrap());
                }
            }

            let mut feedback = vec![0.0f32; 2048];
            for embed in &step_embeds_2048 {
                for (i, val) in embed.iter().enumerate() {
                    feedback[i] += val;
                }
            }

            let text_vec = &tts_pad;
            for (i, val) in text_vec.iter().enumerate() {
                if i < feedback.len() {
                    feedback[i] += val;
                }
            }
            feedback.resize(talker_embd, 0.0);

            let talker_pos = Self::qwen3_position(cur_pos as i32, 1);
            talker_batch.clear();
            talker_batch.set_embd(&feedback, &talker_pos, 0);

            let talker_started = Instant::now();
            self.talker_ctx
                .decode(&mut talker_batch)
                .map_err(|e| format!("Talker step failed: {}", e))?;
            if profile && step < 9 {
                frame_profile.push(serde_json::json!({
                    "frame": step + 1, "finish_ms": request_started.elapsed().as_secs_f64() * 1000.0,
                    "sample0_ms": sampled_ms, "projection_ms": projection_finished - sampled_ms,
                    "predictor_prefill_ms": predictor_prefill_finished - projection_finished,
                    "predictor_sampling_ms": predictor_sample_ms, "predictor_decode14_ms": predictor_decode_ms,
                    "predictor_total_ms": predictor_finished - projection_finished,
                    "talker_decode_ms": talker_started.elapsed().as_secs_f64() * 1000.0,
                    "frame_total_ms": frame_started.elapsed().as_secs_f64() * 1000.0,
                }));
            }

            cur_pos += 1;
        }

        // 发送剩余的帧
        if let Some(confirmation)=pending_decode_confirmation.take() {
            let _=confirmation.send(false);
        }
        let total_frames = all_codes.len() / 16;

        if cancelled {
            // llama_decode can return with GPU work still pending. Normal
            // sampling synchronizes it, but cancellation skips that sampling.
            // Keep batch/input storage alive until both contexts finish.
            self.talker_ctx.synchronize();
            self.predictor_ctx.synchronize();
            eprintln!("[Stream] Inference cancelled after {} complete frames", total_frames);
        } else if total_frames > last_sent_frame {
            let start_frame = last_sent_frame;
            let end_frame = total_frames;

            let start_idx = start_frame * 16;
            let end_idx = end_frame * 16;
            let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                .iter()
                .map(|&c| c as i64)
                .collect();

            let _ = tx.send((frame_codes, true, None));
        } else {
            let _ = tx.send((Vec::new(), true, None));
        }
        drop(tx);

        let audio_samples = decoder_handle
            .join()
            .map_err(|_| "Decoder thread panicked".to_string())??;
        if cancelled || is_cancelled() { return Err("Streaming request cancelled".into()); }
        if profile {
            eprintln!("[TTFA:generator-timeline] {}", serde_json::json!({"prefill_finished_ms": prefill_finished_ms, "frames": frame_profile, "dispatch": dispatch_profile}));
        }

        if let Some(profile) = predictor_sampler.sampling_profile() {
            eprintln!("[SamplerProfile:total] {}", serde_json::to_string(&profile).unwrap());
        }
        // Optional diagnostics only; normal requests perform no trace I/O.
        if let Some(directory) = std::env::var_os("QWEN_CODEC_TRACE_DIR") {
            static TRACE_NUMBER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let directory = PathBuf::from(directory);
            let number = TRACE_NUMBER.fetch_add(1, Ordering::Relaxed) + 1;
            let trace = serde_json::json!({
                "codes": &all_codes, "frame_count": total_frames,
                "end_code": end_code, "finished": generation_finished,
                "stalled_code": stalled_code_0,
                "silence_penalty_calls": silence_penalty_calls,
                "pronunciation_probe": pronunciation_probe,
            });
            let write_trace = || -> Result<(), Box<dyn std::error::Error>> {
                std::fs::create_dir_all(&directory)?;
                std::fs::write(directory.join(format!("codec-{number}.json")), serde_json::to_vec(&trace)?)?;
                Ok(())
            };
            write_trace().map_err(|error| format!("Codec diagnostic trace failed: {error}"))?;
        }

        if let Some(code) = stalled_code_0 {
            return Err(format!(
                "语音合成未完成（模型重复编码 {}）。请联系服务维护者。",
                code
            ));
        }
        if !generation_finished && short_phrase.is_none() {
            return Err(format!(
                "已达到最大生成步数 {}，音频可能未合成完整。",
                generation_limit
            ));
        }

        Ok(AudioSample {
            samples: audio_samples,
            sample_rate: 24000,
            channels: 1,
        })
    }
}

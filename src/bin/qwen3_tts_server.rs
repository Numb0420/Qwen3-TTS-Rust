use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        DefaultBodyLimit, State,
    },
    http::header,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use clap::Parser;
use qwen3_tts::{
    audio_postprocess::{self, Processor, Settings},
    pronunciation::prepare_text,
    TtsEngine, VoiceFile,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

#[path = "server/speaker_upload.rs"]
mod speaker_upload;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "models")]
    model_dir: PathBuf,

    /// Matched Base checkpoint for explicitly validated cloning speakers.
    #[arg(long)]
    base_model_dir: Option<PathBuf>,

    /// Load only the matched Base engine and expose its validated cloning voices.
    #[arg(long, requires = "base_model_dir")]
    base_only: bool,

    /// Optional state-only graph, used only by the validated Base route.
    #[arg(long)]
    base_state_decoder: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', default_value = "39")]
    base_speakers: Vec<String>,

    #[arg(long, default_value = "q5_k_m")]
    quant: String,

    #[arg(long, default_value = "speakers")]
    speakers_dir: PathBuf,

    #[arg(long, default_value = "3000")]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 4)]
    threads: i32,
}

struct AppState {
    engine: Arc<Mutex<EngineRouter>>,
    speakers: HashMap<String, VoiceFile>,
    default_speaker: String,
    speakers_dir: PathBuf,
    upload_model_dir: Option<PathBuf>,
    upload_lock: Arc<Mutex<()>>,
}

// One lock preserves serialization across both GPU engines.
struct EngineRouter {
    legacy: Option<TtsEngine>,
    base: Option<TtsEngine>,
    base_speakers: Vec<String>,
}

impl EngineRouter {
    fn for_speaker(&mut self, name: &str) -> &mut TtsEngine {
        if self.base_speakers.iter().any(|speaker| speaker == name) {
            if let Some(base) = self.base.as_mut() {
                return base;
            }
        }
        self.legacy
            .as_mut()
            .expect("Available voices must have an engine route")
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TTSRequest {
    text: String,
    speaker: Option<String>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    instruction: Option<String>,
    max_steps: Option<usize>,
    volume: Option<i32>,
    speed: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TTSResponse {
    success: bool,
    message: Option<String>,
    audio_base64: Option<String>,
    sample_rate: Option<u32>,
    duration_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpeakersResponse {
    speakers: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();
    // Uploaded full ICL voices use the matched Base route after restart.
    if args.base_model_dir.is_some() {
        let registry = args.speakers_dir.join("custom-speakers.txt");
        if registry.exists() {
            for name in std::fs::read_to_string(registry)?.lines().map(str::trim).filter(|s| !s.is_empty()) {
                if name.len() > 64 || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_') {
                    return Err(format!("Invalid uploaded speaker name: {name}").into());
                }
                if !args.base_speakers.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                    args.base_speakers.push(name.to_owned());
                }
            }
        }
    }

    println!("=== Qwen3-TTS Server ===");
    println!("Model Dir: {:?}", args.model_dir);
    println!("Quant:     {}", args.quant);
    println!("Port:      {}", args.port);
    println!("Threads:   {}", args.threads);

    println!(
        "Engine mode: {}",
        if args.base_only {
            "Base only"
        } else {
            "Legacy with optional Base route"
        }
    );
    let engine = if args.base_only {
        None
    } else {
        println!("Checking models...");
        TtsEngine::download_models(&args.model_dir, &args.quant)
            .await
            .map_err(|e| format!("Model download failed: {}", e))?;

        println!("Loading engine...");
        let mut engine = TtsEngine::new(&args.model_dir, &args.quant, args.threads)
            .await
            .map_err(|e| format!("Engine load failed: {}", e))?;

        engine.prewarm_codec_projections();

        if args.speakers_dir.exists() {
            engine.load_speakers(&args.speakers_dir)?;
            engine.prewarm_decoder_contexts()?;
        }

        Some(engine)
    };
    let mut speakers = engine
        .as_ref()
        .map(|engine| engine.get_speakers_map().clone())
        .unwrap_or_default();
    let base = if let Some(root) = args.base_model_dir.as_ref() {
        let quant_folder = match args.quant.as_str() {
            "q5_k_m" => "gguf_q5_k_m",
            "q8_0" => "gguf_q8_0",
            "none" => "gguf",
            other => return Err(format!("Unsupported Base quantization: {other}").into()),
        };
        // Fail before the generic downloader can fill a Base directory with CustomVoice files.
        for relative in [
            format!("{quant_folder}/qwen3_tts_talker.gguf"),
            format!("{quant_folder}/qwen3_tts_predictor.gguf"),
            "gguf/qwen3_assets.gguf".into(),
            "onnx/qwen3_tts_decoder.onnx".into(),
            "onnx/qwen3_tts_codec_encoder.onnx".into(),
            "onnx/qwen3_tts_speaker_encoder.onnx".into(),
            "tokenizer/tokenizer.json".into(),
            "checkpoint-manifest.json".into(),
        ] {
            if !root.join(&relative).is_file() {
                return Err(format!("Base bundle incomplete: {}. Install matched Base files; automatic checkpoint substitution is disabled.", root.join(relative).display()).into());
            }
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("checkpoint-manifest.json"))?)?;
        if manifest["source_model"].as_str() != Some("Qwen/Qwen3-TTS-12Hz-1.7B-Base") {
            return Err("Base bundle manifest must identify Qwen/Qwen3-TTS-12Hz-1.7B-Base".into());
        }
        println!("Base routing: {:?} -> {:?}", args.base_speakers, root);
        let mut base = TtsEngine::new(root, &args.quant, args.threads).await?;
        base.load_speakers(&args.speakers_dir)?;
        if args.base_only {
            base.retain_speakers(&args.base_speakers)?;
            speakers = base.get_speakers_map().clone();
        }
        for name in &args.base_speakers {
            let voice = base
                .get_speakers_map()
                .get(name)
                .ok_or_else(|| format!("Base speaker not found: {name}"))?;
            if voice.spk_id.is_some() {
                return Err(format!(
                    "Base routing requires a cloning voice, not a preset ID: {name}"
                )
                .into());
            }
        }
        base.prewarm_codec_projections();
        base.prewarm_decoder_contexts()?;
        if let Some(path) = args.base_state_decoder.as_ref() {
            base.enable_state_only_decoder(path, &args.base_speakers)?;
        }
        Some(base)
    } else {
        None
    };
    let default_speaker = if args.base_only {
        args.base_speakers
            .first()
            .cloned()
            .ok_or("Base-only mode requires a validated speaker")?
    } else {
        "vivian".to_string()
    };
    let engine = Arc::new(Mutex::new(EngineRouter {
        legacy: engine,
        base,
        base_speakers: args.base_speakers,
    }));

    let state = AppState {
        engine,
        speakers,
        default_speaker,
        speakers_dir: args.speakers_dir,
        upload_model_dir: args.base_model_dir.map(|root| root.join("onnx")),
        upload_lock: Arc::new(Mutex::new(())),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/audio-processor.js", get(audio_processor))
        .route("/api/tts", post(tts_handler))
        .route("/api/tts/stream", get(tts_stream_handler))
        .route("/api/speakers", get(speakers_handler))
        .route("/api/speakers/upload", post(speaker_upload::upload))
        .route("/health", get(health))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(Arc::new(state));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    println!("Server running at http://{}", addr);
    println!("API endpoints:");
    println!("  POST /api/tts        - Generate TTS audio (non-streaming)");
    println!("  GET  /api/tts/stream - WebSocket streaming TTS");
    println!("  GET  /api/speakers   - List available speakers");
    println!("  GET  /health         - Health check");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    if addr.ip().is_loopback() || addr.ip().is_unspecified() {
        axum::serve(listener, app).await?;
    } else {
        let loopback_addr = SocketAddr::new(
            if addr.is_ipv4() { "127.0.0.1".parse()? } else { "::1".parse()? },
            addr.port(),
        );
        let loopback_listener = tokio::net::TcpListener::bind(loopback_addr).await?;
        println!("Also serving local requests at http://{}", loopback_addr);
        tokio::select! {
            result = axum::serve(listener, app.clone()) => result?,
            result = axum::serve(loopback_listener, app) => result?,
        }
    }

    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../../webui/index.html"))
}

async fn audio_processor() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../../webui/audio-processor.js"),
    )
}

async fn health() -> &'static str {
    "OK"
}

async fn speakers_handler(State(state): State<Arc<AppState>>) -> Json<SpeakersResponse> {
    let names: Vec<String> = state.speakers.keys().cloned().collect();
    Json(SpeakersResponse { speakers: names })
}

async fn tts_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TTSRequest>,
) -> Json<TTSResponse> {
    let start = Instant::now();
    let model_text = match prepare_text(&req.text) {
        Ok(text) => text,
        Err(message) => {
            return Json(TTSResponse {
                success: false,
                message: Some(message),
                audio_base64: None,
                sample_rate: None,
                duration_ms: None,
            })
        }
    };
    let postprocess = match Settings::new(req.volume, req.speed) {
        Ok(settings) => settings,
        Err(message) => {
            return Json(TTSResponse {
                success: false,
                message: Some(message),
                audio_base64: None,
                sample_rate: None,
                duration_ms: None,
            })
        }
    };
    let speaker_name = req.speaker.unwrap_or_else(|| state.default_speaker.clone());

    let voice = state.speakers.get(&speaker_name).cloned();

    let voice = match voice {
        Some(v) => v,
        None => {
            return Json(TTSResponse {
                success: false,
                message: Some(format!("Speaker '{}' not found", speaker_name)),
                audio_base64: None,
                sample_rate: None,
                duration_ms: None,
            });
        }
    };

    let mut engines = state.engine.lock().await;
    let engine = engines.for_speaker(&speaker_name);

    // 一次性修改采样器配置
    {
        let config = engine.get_sampler_config().clone();
        let mut new_config = config;
        new_config.temperature = req.temperature.unwrap_or(0.0);
        if let Some(top_k) = req.top_k {
            new_config.top_k = top_k;
        }
        if let Some(top_p) = req.top_p {
            new_config.top_p = top_p;
        }
        new_config.seed = req.seed;
        engine.set_sampler_config(new_config);
    }

    // 设置 max_steps
    if let Some(max_steps) = req.max_steps {
        engine.set_max_steps(max_steps);
    }

    let result =
        engine.generate_with_voice_streaming(&model_text, &voice, req.instruction.as_deref(), None);

    match result {
        Ok(mut audio) => {
            audio_postprocess::process_complete(&mut audio.samples, postprocess);
            let duration_ms = (audio.samples.len() as f64 / audio.sample_rate as f64) * 1000.0;
            let audio_duration_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
            let rtf = start.elapsed().as_secs_f64() / audio_duration_secs;
            println!(
                "RTF: {:.3} (gen: {:.2}s, audio: {:.2}s)",
                rtf,
                start.elapsed().as_secs_f64(),
                audio_duration_secs
            );

            let wav_data = audio.to_wav_bytes();

            Json(TTSResponse {
                success: true,
                message: None,
                audio_base64: Some(BASE64.encode(&wav_data)),
                sample_rate: Some(audio.sample_rate),
                duration_ms: Some(duration_ms),
            })
        }
        Err(e) => Json(TTSResponse {
            success: false,
            message: Some(e),
            audio_base64: None,
            sample_rate: None,
            duration_ms: None,
        }),
    }
}

// WebSocket 流式 TTS 处理器
async fn tts_stream_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_tts_stream(socket, state))
}

async fn handle_tts_stream(mut socket: WebSocket, state: Arc<AppState>) {
    let (text_tx, text_rx) = std::sync::mpsc::channel::<Option<(TTSRequest, Instant)>>();
    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let (error_tx, mut error_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let engine = state.engine.clone();
    let speakers = state.speakers.clone();
    let default_speaker = state.default_speaker.clone();

    let mut total_samples: usize = 0;
    let stream_start = std::time::Instant::now();
    let worker_audio_tx = audio_tx.clone();

    std::thread::spawn(move || {
        while let Ok(Some((req, accepted_at))) = text_rx.recv() {
            if worker_audio_tx.is_closed() { break; }
            // An idle WebSocket must not reserve the engine. Cancelled requests
            // waiting for another synthesis must also leave without inference.
            let mut engines = engine.blocking_lock();
            if worker_audio_tx.is_closed() { break; }
            let spk_name = req.speaker.as_deref().unwrap_or(&default_speaker);
            let Some(voice) = speakers.get(spk_name) else {
                let _ = error_tx.send(format!("Speaker '{spk_name}' is not available in this server configuration"));
                break;
            };

            let engine = engines.for_speaker(spk_name);

            let mut config = engine.get_sampler_config().clone();
            config.temperature = req.temperature.unwrap_or(0.0);
            if let Some(top_k) = req.top_k {
                config.top_k = top_k;
            }
            if let Some(top_p) = req.top_p {
                config.top_p = top_p;
            }
            config.seed = req.seed;
            engine.set_sampler_config(config);
            if let Some(max_steps) = req.max_steps {
                engine.set_max_steps(max_steps);
            }

            if std::env::var_os("QWEN_TTFA_PROFILE").is_some() {
                eprintln!(
                    "[TTFA:server] speaker={} before_inference_ms={:.3}",
                    spk_name,
                    accepted_at.elapsed().as_secs_f64() * 1000.0
                );
            }
            if let Err(message) = engine.generate_with_voice_streaming(
                &req.text,
                voice,
                req.instruction.as_deref(),
                Some(worker_audio_tx.clone()),
            ) {
                if worker_audio_tx.is_closed() { break; }
                let _ = error_tx.send(message);
                break;
            }

            let _ = worker_audio_tx.send(vec![]);
        }
    });
    // The worker owns the only sender; recv() must return None after its final chunk.
    drop(audio_tx);

    let mut end_requested = false;
    let mut stream_failed = false;
    let mut error_channel_closed = false;
    let mut queued_settings = VecDeque::<(Settings, Instant)>::new();
    let mut current_settings: Option<Settings> = None;
    let mut processor: Option<Processor> = None;

    loop {
        tokio::select! {
            biased;
            error = error_rx.recv(), if !error_channel_closed => {
                if let Some(message) = error {
                    let payload = serde_json::json!({"error": message});
                    let _ = socket.send(Message::Text(payload.to_string())).await;
                    stream_failed = true;
                    break;
                }
                error_channel_closed = true;
            }
            maybe_samples = audio_rx.recv() => {
                let Some(samples) = maybe_samples else { break; };
                if samples.is_empty() {
                    if let Some(active) = processor.as_mut() {
                        let tail = active.finish();
                        if !tail.is_empty() && send_pcm(&mut socket, &tail).await.is_err() { break; }
                        total_samples += tail.len();
                    }
                    processor = None;
                    current_settings = None;
                    queued_settings.pop_front();
                    let gen_time = stream_start.elapsed().as_secs_f64();
                    let audio_duration = total_samples as f64 / 24000.0;
                    let rtf = if audio_duration > 0.0 { gen_time / audio_duration } else { 0.0 };
                    eprintln!("[Stream] RTF: {:.3} (gen: {:.2}s, audio: {:.2}s)", rtf, gen_time, audio_duration);
                    if socket.send(Message::Text("segment_done".to_string())).await.is_err() { break; }
                } else {
                    if current_settings.is_none() {
                        let (settings, accepted_at) = queued_settings.front().copied().unwrap_or_else(|| (Settings::new(None, None).unwrap(), Instant::now()));
                        if std::env::var_os("QWEN_TTFA_PROFILE").is_some() {
                            eprintln!("[TTFA:server] first_pcm_to_socket_ms={:.3}", accepted_at.elapsed().as_secs_f64() * 1000.0);
                        }
                        current_settings = Some(settings);
                        if !settings.is_default() { processor = Some(Processor::new(settings)); }
                    }
                    if let Some(active) = processor.as_mut() {
                        let processed = active.push(&samples);
                        total_samples += processed.len();
                        if !processed.is_empty() && send_pcm(&mut socket, &processed).await.is_err() { break; }
                    } else {
                        total_samples += samples.len();
                        if send_pcm(&mut socket, &samples).await.is_err() { break; }
                    }
                }
            }
            msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            // `end` finishes text input, not connection monitoring.
                            if end_requested { continue; }
                            if text == "end" {
                                let _ = text_tx.send(None);
                                end_requested = true;
                            } else {
                                let accepted_at = Instant::now();
                                let mut req: TTSRequest = match serde_json::from_str(&text) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        let _ = socket.send(Message::Text(format!("{{\"error\": \"Invalid request: {}\"}}", e))).await;
                                        continue;
                                    }
                                };

                                if !req.text.trim().is_empty() {
                                    let name = req.speaker.as_deref().unwrap_or(&state.default_speaker);
                                    if !state.speakers.contains_key(name) {
                                        let error = serde_json::json!({"error": format!("Speaker '{name}' is not available in this server configuration")});
                                        let _ = socket.send(Message::Text(error.to_string())).await;
                                        continue;
                                    }
                                    req.text = match prepare_text(&req.text) {
                                        Ok(text) => text,
                                        Err(message) => {
                                            let error = serde_json::json!({"error": message});
                                            let _ = socket.send(Message::Text(error.to_string())).await;
                                            continue;
                                        }
                                    };
                                    let settings = match Settings::new(req.volume, req.speed) {
                                        Ok(settings) => settings,
                                        Err(message) => {
                                            let error = serde_json::json!({"error": message});
                                            let _ = socket.send(Message::Text(error.to_string())).await;
                                            continue;
                                        }
                                    };
                                    queued_settings.push_back((settings, accepted_at));
                                    let _ = text_tx.send(Some((req, accepted_at)));
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) => break,
                        Some(Ok(Message::Binary(_))) => continue,
                        Some(Ok(Message::Ping(_))) => continue,
                        Some(Ok(Message::Pong(_))) => continue,
                        Some(Err(_)) => break,
                        None => break,
                    }
            }
        }
    }

    // Closing the receiver is the cancellation signal observed by inference
    // and decoding, including when the client disconnects after sending `end`.
    audio_rx.close();
    let _ = text_tx.send(None);
    if !stream_failed {
        let _ = socket.send(Message::Text("done".to_string())).await;
    }
}

async fn send_pcm(socket: &mut WebSocket, samples: &[f32]) -> Result<(), axum::Error> {
    let bytes: Vec<u8> = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    socket.send(Message::Binary(bytes)).await
}

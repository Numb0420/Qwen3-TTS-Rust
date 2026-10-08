//! Upload and persistence are separate from synthesis and playback.
use super::AppState;
use axum::{extract::{Multipart, State}, http::StatusCode, Json};
use serde_json::{json, Value};
use std::{fs, path::Path, sync::Arc};

type UploadResult = Result<Json<Value>, (StatusCode, Json<Value>)>;
fn error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"success": false, "message": message.into()})))
}

fn valid_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    !name.is_empty() && name.len() <= 64
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        && !matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$")
        && !(upper.len() == 4 && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'0'..=b'9'))
}

pub async fn upload(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> UploadResult {
    // Try-lock rejects duplicate submissions without queuing large uploads.
    let guard = state.upload_lock.clone().try_lock_owned()
        .map_err(|_| error(StatusCode::CONFLICT, "正在生成另一个说话人，请稍后再试。"))?;
    let models = state.upload_model_dir.clone().ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "未配置匹配的 Base 提取模型。"))?;
    let mut name = None;
    let mut text = None;
    let mut audio = None;
    while let Some(field) = multipart.next_field().await.map_err(|_| error(StatusCode::BAD_REQUEST, "上传格式无效或文件超过 50 MB。"))? {
        match field.name().unwrap_or_default() {
            "name" if name.is_none() => name = Some(field.text().await.map_err(|_| error(StatusCode::BAD_REQUEST, "名称无效"))?),
            "ref_text" if text.is_none() => text = Some(field.text().await.map_err(|_| error(StatusCode::BAD_REQUEST, "参考文本无效"))?),
            "audio" if audio.is_none() => {
                let filename = field.file_name().unwrap_or_default().to_owned();
                if !filename.to_ascii_lowercase().ends_with(".wav") {
                    return Err(error(StatusCode::BAD_REQUEST, "请上传 WAV 音频文件。"));
                }
                audio = Some(field.bytes().await.map_err(|_| error(StatusCode::BAD_REQUEST, "音频读取失败或文件超过 50 MB。"))?);
            }
            _ => return Err(error(StatusCode::BAD_REQUEST, "上传包含未知或重复字段。")),
        }
    }
    let name = name.unwrap_or_default().trim().to_owned();
    let text = text.unwrap_or_default().trim().to_owned();
    if !valid_name(&name) {
        return Err(error(StatusCode::BAD_REQUEST, "名称须为 1–64 个英文字母、数字、连字符或下划线，不能使用 Windows 保留名称。"));
    }
    if text.is_empty() || text.chars().count() > 2000 {
        return Err(error(StatusCode::BAD_REQUEST, "请填写与录音逐字对应的文本（最多 2000 字）。"));
    }
    let audio = audio.filter(|bytes| !bytes.is_empty()).ok_or_else(|| error(StatusCode::BAD_REQUEST, "请选择非空 WAV 文件。"))?;
    let dir = state.speakers_dir.clone();
    let job = tokio::task::spawn_blocking(move || {
        // Keep the upload lock inside the worker even if the HTTP client disconnects.
        let _guard = guard;
        create(&dir, &models, &name, &text, &audio)
    }).await.map_err(|e| {
        eprintln!("Speaker extraction worker failed: {e}");
        error(StatusCode::INTERNAL_SERVER_ERROR, "说话人生成任务失败，请查看服务日志。")
    })?;
    job.map(Json)
}

fn create(dir: &Path, models: &Path, name: &str, text: &str, audio: &[u8]) -> Result<Value, (StatusCode, Json<Value>)> {
    fs::create_dir_all(dir).map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let wav = dir.join(format!("{name}.wav"));
    let voice_path = dir.join(format!("{name}.json"));
    if wav.exists() || voice_path.exists() {
        return Err(error(StatusCode::CONFLICT, "该名称的 WAV 或 JSON 已存在，请使用其他名称。"));
    }
    let registry = dir.join("custom-speakers.txt");
    let mut names = match fs::read_to_string(&registry) {
        Ok(s) => s.lines().map(str::to_owned).collect::<Vec<_>>(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    };
    let pending = dir.join(format!(".{name}.upload.tmp"));
    let json_pending = dir.join(format!(".{name}.json.tmp"));
    let registry_pending = dir.join(".custom-speakers.tmp");
    let result = (|| -> Result<(), String> {
        use std::io::Write;
        let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&pending).map_err(|e| e.to_string())?;
        file.write_all(audio).map_err(|e| e.to_string())?;
        drop(file);
        let voice = qwen3_tts::speaker_creation::extract_full(&pending, models, name, text)?;
        voice.save(&json_pending)?;
        if !names.iter().any(|n| n.eq_ignore_ascii_case(name)) { names.push(name.to_owned()); }
        fs::write(&registry_pending, format!("{}\n", names.join("\n"))).map_err(|e| e.to_string())?;
        // Hard links create destination files without replacing existing voices.
        fs::hard_link(&pending, &wav).map_err(|e| e.to_string())?;
        if let Err(e) = fs::hard_link(&json_pending, &voice_path) {
            let _ = fs::remove_file(&wav);
            return Err(e.to_string());
        }
        if let Err(e) = fs::rename(&registry_pending, &registry) {
            let _ = fs::remove_file(&wav);
            let _ = fs::remove_file(&voice_path);
            return Err(e.to_string());
        }
        Ok(())
    })();
    for path in [pending, json_pending, registry_pending] { let _ = fs::remove_file(path); }
    result.map_err(|e| {
        eprintln!("Speaker upload {name} failed: {e}");
        error(StatusCode::UNPROCESSABLE_ENTITY, format!("生成失败：{e}"))
    })?;
    Ok(json!({"success": true, "name": name, "restart_required": true,
        "message": format!("已保存 speakers/{name}.wav 和 speakers/{name}.json。请重启服务，刷新网页后即可选择该说话人。")}))
}

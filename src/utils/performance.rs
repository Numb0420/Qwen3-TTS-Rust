//! Internal diagnostic switches; no web API or sampling configuration changes.
pub fn enabled(name: &str) -> bool {
    if let Some(path) = std::env::var_os("QWEN_OPT_MODE_FILE") {
        if let Ok(mode) = std::fs::read_to_string(path) {
            return mode.trim().split(',').any(|value| value == name);
        }
    }
    std::env::var("QWEN_OPT_FLAGS").unwrap_or_default().split(',').any(|value| value == name)
}

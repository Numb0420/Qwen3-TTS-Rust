#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod tray;
use std::{fs, io::{Read, Write}, net::{TcpStream, ToSocketAddrs}, path::PathBuf, process::{Command, Stdio, Child}, thread, time::{Duration, Instant}};

fn run() -> Result<(), String> {
    let root = std::env::current_exe().map_err(|e| e.to_string())?.parent().ok_or("启动器路径无效")?.to_owned();
    std::env::set_current_dir(&root).map_err(|e| e.to_string())?;
    let config: serde_json::Value = serde_json::from_slice(&fs::read(root.join("server-config.json")).map_err(|e| format!("无法读取配置：{e}"))?).map_err(|e| format!("配置 JSON 无效：{e}"))?;
    let device = config["inference_device"].as_str().unwrap_or("vulkan");
    if !["cpu", "vulkan"].contains(&device) { return Err("inference_device 必须为 cpu 或 vulkan".into()); }
    let host = config["host"].as_str().unwrap_or("0.0.0.0");
    host.parse::<std::net::Ipv4Addr>().map_err(|_| "host 必须是 IPv4 地址")?;
    let mut port = config["port"].as_u64().unwrap_or(3000);
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if let Some(index) = arguments.iter().position(|a| a == "--port") {
        port = arguments.get(index + 1).ok_or("--port 缺少值")?.parse().map_err(|_| "端口无效")?;
    }
    if port == 0 || port > 65535 { return Err("端口须为 1–65535".into()); }
    let voices = config["speakers"].as_array().ok_or("配置须包含 speakers 数组")?;
    let mut names = Vec::new();
    for value in voices {
        let name = value.as_str().ok_or("说话人名称无效")?;
        if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_') { return Err("说话人名称无效".into()); }
        names.push(name);
    }
    if names.is_empty() { return Err("至少需要一个说话人".into()); }
    let model = root.join("models/base-1.7b");
    for file in ["qwen3_tts_server.exe", "runtime/onnxruntime.dll", "runtime/llama.dll", "runtime/decoder-stage3-end1292.onnx", "models/base-1.7b/checkpoint-manifest.json"] {
        if !root.join(file).is_file() { return Err(format!("缺少文件：{file}，请完整解压绿色版。")); }
    }
    for name in &names {
        if !root.join(format!("speakers/{name}.json")).is_file() { return Err(format!("缺少说话人：{name}")); }
    }
    if arguments.iter().any(|a| a == "--check") { log(&format!("配置和启动文件检查通过：{device}，IP {host}，端口 {port}")); return Ok(()); }
    if arguments.iter().any(|a| a == "--stop") { tray::stop_existing(&root, port)?; log("已发送停止请求"); return Ok(()); }
    let address = format!("127.0.0.1:{port}");
    let endpoint = address.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("地址无效")?;
    if TcpStream::connect_timeout(&endpoint, Duration::from_millis(300)).is_ok() { return Err(format!("端口 {port} 已被占用，请关闭已有服务或修改 server-config.json。")); }
    fs::create_dir_all(root.join("logs")).map_err(|e| format!("目录不可写，请解压到可写文件夹：{e}"))?;
    let stdout = fs::File::create(root.join("logs/server.stdout.log")).map_err(|e| e.to_string())?;
    let stderr = fs::File::create(root.join("logs/server.stderr.log")).map_err(|e| e.to_string())?;
    let runtime = root.join("runtime");
    let mut command = Command::new(root.join("qwen3_tts_server.exe"));
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("QWEN_") || name.starts_with("GGML_") || name.starts_with("ORT_") { command.env_remove(key); }
    }
    let system = PathBuf::from(std::env::var_os("SystemRoot").unwrap_or("C:\\Windows".into()));
    let path = std::env::join_paths([root.clone(), runtime.clone(), system.join("System32"), system]).map_err(|e| e.to_string())?;
    command.current_dir(&root).env("PATH", path)
        .env("QWEN_PRONUNCIATION_DEVICE", device)
        .env("QWEN_OPT_FLAGS", "quiet,state-only,compact-sampler")
        .env("QWEN_ONNX_NO_SPIN", "1")
        .env("QWEN_LLAMA_RUNTIME_DIR", &runtime)
        .env("QWEN_PREDICTOR_HEAD_RANGE", "1")
        .env("QWEN_PREDICTOR_HEAD_MODEL_DIR", &model)
        .env("QWEN_DIAG_WAVE_PREFIX_MODEL", runtime.join("decoder-stage3-end1292.onnx"))
        .env("QWEN_DIAG_WAVE_PREFIX_MODE_FILE", runtime.join("enabled.txt"))
        .env("QWEN_DIAG_EOS_MODE_FILE", runtime.join("enabled.txt"))
        .env("QWEN_DIAG_PROJECTION_MODE_FILE", runtime.join("enabled.txt"))
        .env("GGML_VK_DIAG_CONFIG_FILE", runtime.join("vulkan-config.txt"))
        .args(["--host",host,"--port",&port.to_string(),"--base-only","--model-dir"])
        .arg(&model).arg("--base-model-dir").arg(&model)
        .arg("--base-state-decoder").arg(model.join("onnx/qwen3_tts_decoder_state_only.onnx"))
        .arg("--base-speakers").arg(names.join(","))
        .arg("--speakers-dir").arg(root.join("speakers"))
        .stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr));
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    let mut tray = tray::Tray::new(&root, port)?;
    let mut child = ManagedChild::new(command.spawn().map_err(|e| format!("无法启动服务：{e}"))?)?;
    log(&format!("系统托盘已创建，设备 {device}，监听 {host}:{port}，正在加载模型。"));
    let began = Instant::now();
    let mut ready = false;
    loop {
        if tray.exiting() { log("托盘退出：正在停止服务"); return Ok(()); }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            if status.success() { return Ok(()); }
            return Err(format!("服务退出（{status}）。请查看 logs/server.stderr.log。"));
        }
        if !ready {
            if let Ok(mut stream) = TcpStream::connect_timeout(&endpoint, Duration::from_millis(200)) {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
                let _ = stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
                let mut response = String::new();
                if stream.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.1 200") {
                    ready = true;
                    tray.ready(port);
                    log(&format!("服务已就绪：http://127.0.0.1:{port}/"));
                    if config["open_browser"].as_bool().unwrap_or(true) && !arguments.iter().any(|a| a == "--no-browser") {
                        tray::open(&format!("http://127.0.0.1:{port}/"));
                    }
                }
            }
            if began.elapsed() > Duration::from_secs(300) {
                let _ = child.kill(); let _ = child.wait();
                return Err("启动超过 5 分钟，请检查日志和可用内存。".into());
            }
        }
        thread::sleep(Duration::from_millis(300));
    }
}

fn log(message: &str) {
    let _ = fs::create_dir_all("logs");
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open("logs/launcher.log") { let _ = writeln!(file, "{message}"); }
}
// A kill-on-close Windows Job prevents an orphan server if the launcher is terminated.
struct ManagedChild { process: Child, job: windows_sys::Win32::Foundation::HANDLE }
impl ManagedChild {
    fn new(mut process: Child) -> Result<Self, String> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::{Foundation::CloseHandle, System::JobObjects::*};
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if job.is_null() || SetInformationJobObject(job, JobObjectExtendedLimitInformation, (&limits as *const _) as *const _, std::mem::size_of_val(&limits) as u32) == 0 || AssignProcessToJobObject(job, process.as_raw_handle() as _) == 0 {
                let error = std::io::Error::last_os_error();
                let _ = process.kill(); let _ = process.wait();
                if !job.is_null() { CloseHandle(job); }
                return Err(format!("无法创建服务进程保护：{error}"));
            }
            Ok(Self { process, job })
        }
    }
}
impl std::ops::Deref for ManagedChild { type Target = Child; fn deref(&self) -> &Child { &self.process } }
impl std::ops::DerefMut for ManagedChild { fn deref_mut(&mut self) -> &mut Child { &mut self.process } }
impl Drop for ManagedChild {
    fn drop(&mut self) { let _ = self.process.kill(); let _ = self.process.wait(); unsafe { windows_sys::Win32::Foundation::CloseHandle(self.job); } }
}
fn main() {
    if let Err(message) = run() {
        log(&format!("错误：{message}"));
        if !std::env::args().any(|a| a == "--check" || a == "--stop") { tray::show_error(&message); }
        std::process::exit(1);
    }
}

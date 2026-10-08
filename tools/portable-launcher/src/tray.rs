use std::{path::{Path, PathBuf}, sync::atomic::{AtomicBool, AtomicU32, Ordering}};
use windows_sys::Win32::{Foundation::*, UI::{Shell::*, WindowsAndMessaging::*}, System::LibraryLoader::*};

static EXIT: AtomicBool = AtomicBool::new(false);
static TASKBAR_CREATED: AtomicU32 = AtomicU32::new(0);
const CALLBACK: u32 = WM_APP + 1;
const STOP: u32 = WM_APP + 2;
const OPEN: usize = 101;
const CONFIG: usize = 102;
const LOGS: usize = 103;
const QUIT: usize = 104;
fn wide(value: &str) -> Vec<u16> { value.encode_utf16().chain(Some(0)).collect() }
fn class() -> Vec<u16> { wide("Qwen3TTSPortableTray") }
fn title(root: &Path, port: u64) -> Vec<u16> { wide(&format!("Qwen3TTS:{}:{port}", root.display())) }

pub fn open(target: &str) {
    unsafe { ShellExecuteW(std::ptr::null_mut(), wide("open").as_ptr(), wide(target).as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL); }
}
pub fn show_error(message: &str) {
    unsafe { MessageBoxW(std::ptr::null_mut(), wide(message).as_ptr(), wide("Qwen3-TTS 启动错误").as_ptr(), MB_OK | MB_ICONERROR); }
}
pub fn stop_existing(root: &Path, port: u64) -> Result<(), String> {
    unsafe {
        let window = FindWindowW(class().as_ptr(), title(root, port).as_ptr());
        if window.is_null() { return Err("没有找到该目录、该端口的托盘服务".into()); }
        if PostMessageW(window, STOP, 0, 0) == 0 { return Err(std::io::Error::last_os_error().to_string()); }
    }
    Ok(())
}

pub struct Tray { window: HWND, root: PathBuf, url: String, ready: bool }
unsafe extern "system" fn procedure(window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if message != 0 && message == TASKBAR_CREATED.load(Ordering::SeqCst) {
        let pointer = GetWindowLongPtrW(window, GWLP_USERDATA) as *mut Tray;
        if !pointer.is_null() {
            let tray = &*pointer;
            Shell_NotifyIconW(NIM_ADD, &mut tray.data(if tray.ready { "Qwen3-TTS：运行中" } else { "Qwen3-TTS：正在加载模型…" }));
        }
        return 0;
    }
    match message {
        STOP | WM_CLOSE => { EXIT.store(true, Ordering::SeqCst); return 0; }
        CALLBACK => {
            let pointer = GetWindowLongPtrW(window, GWLP_USERDATA) as *mut Tray;
            if pointer.is_null() { return 0; }
            let tray = &*pointer;
            if lparam as u32 == WM_LBUTTONDBLCLK { open(&tray.url); }
            if lparam as u32 == WM_RBUTTONUP {
                let menu = CreatePopupMenu();
                AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, wide(if tray.ready { "服务运行中" } else { "正在加载模型…" }).as_ptr());
                AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
                AppendMenuW(menu, MF_STRING, OPEN, wide("打开网页").as_ptr());
                AppendMenuW(menu, MF_STRING, CONFIG, wide("打开配置文件").as_ptr());
                AppendMenuW(menu, MF_STRING, LOGS, wide("打开日志目录").as_ptr());
                AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
                AppendMenuW(menu, MF_STRING, QUIT, wide("退出并停止服务").as_ptr());
                let mut point: POINT = std::mem::zeroed(); GetCursorPos(&mut point);
                SetForegroundWindow(window);
                TrackPopupMenu(menu, TPM_RIGHTBUTTON, point.x, point.y, 0, window, std::ptr::null());
                PostMessageW(window, WM_NULL, 0, 0);
                DestroyMenu(menu);
            }
            return 0;
        }
        WM_COMMAND => {
            let pointer = GetWindowLongPtrW(window, GWLP_USERDATA) as *mut Tray;
            if pointer.is_null() { return 0; }
            let tray = &*pointer;
            match wparam & 0xffff {
                OPEN => open(&tray.url),
                CONFIG => {
                    let system = std::path::PathBuf::from(std::env::var_os("SystemRoot").unwrap_or("C:\\Windows".into()));
                    let _ = std::process::Command::new(system.join("System32/notepad.exe")).arg(tray.root.join("server-config.json")).spawn();
                }
                LOGS => open(&tray.root.join("logs").to_string_lossy()),
                QUIT => { EXIT.store(true, Ordering::SeqCst); }
                _ => {}
            }
            return 0;
        }
        _ => {}
    }
    DefWindowProcW(window, message, wparam, lparam)
}
impl Tray {
    pub fn new(root: &Path, port: u64) -> Result<Box<Self>, String> {
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            TASKBAR_CREATED.store(RegisterWindowMessageW(wide("TaskbarCreated").as_ptr()), Ordering::SeqCst);
            let name = class();
            let mut definition: WNDCLASSW = std::mem::zeroed();
            definition.lpfnWndProc = Some(procedure); definition.hInstance = instance; definition.lpszClassName = name.as_ptr();
            if RegisterClassW(&definition) == 0 { return Err(std::io::Error::last_os_error().to_string()); }
            let window = CreateWindowExW(0, name.as_ptr(), title(root, port).as_ptr(), 0, 0, 0, 0, 0, std::ptr::null_mut(), std::ptr::null_mut(), instance, std::ptr::null());
            if window.is_null() { return Err(std::io::Error::last_os_error().to_string()); }
            let mut tray = Box::new(Self { window, root: root.to_owned(), url: format!("http://127.0.0.1:{port}/"), ready: false });
            SetWindowLongPtrW(window, GWLP_USERDATA, (&mut *tray as *mut Self) as isize);
            let mut data = tray.data("Qwen3-TTS：正在加载模型…");
            if Shell_NotifyIconW(NIM_ADD, &mut data) == 0 { return Err("无法创建系统托盘图标，请检查 Windows Explorer 是否运行".into()); }
            Ok(tray)
        }
    }
    fn data(&self, tip: &str) -> NOTIFYICONDATAW {
        unsafe {
            let mut data: NOTIFYICONDATAW = std::mem::zeroed();
            data.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
            data.hWnd = self.window; data.uID = 1; data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP; data.uCallbackMessage = CALLBACK;
            data.hIcon = LoadIconW(std::ptr::null_mut(), IDI_APPLICATION);
            for (slot, value) in data.szTip.iter_mut().take(127).zip(tip.encode_utf16()) { *slot = value; }
            data
        }
    }
    pub fn ready(&mut self, port: u64) { self.ready = true; unsafe { Shell_NotifyIconW(NIM_MODIFY, &mut self.data(&format!("Qwen3-TTS：运行中，端口 {port}"))); } }
    pub fn exiting(&self) -> bool {
        unsafe {
            let mut message: MSG = std::mem::zeroed();
            while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 { TranslateMessage(&message); DispatchMessageW(&message); }
        }
        EXIT.load(Ordering::SeqCst)
    }
}
impl Drop for Tray { fn drop(&mut self) { unsafe { Shell_NotifyIconW(NIM_DELETE, &mut self.data("")); SetWindowLongPtrW(self.window, GWLP_USERDATA, 0); DestroyWindow(self.window); } } }

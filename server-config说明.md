# 服务设备配置

`start-webui.ps1` 启动时读取项目根目录的 `server-config.json`。

```json
{
  "inference_device": "vulkan",
  "host": "0.0.0.0",
  "port": 3000
}
```

- `cpu`：Talker 和 Predictor 的 GPU 层数为 0，使用 CPU 推理。
- `vulkan`：Talker 和 Predictor 请求将模型层放到 Vulkan GPU 上，需安装可用的 Vulkan 驱动。
- ONNX 声码器在两种模式下都继续使用 CPU。
- `host`：监听 IPv4 地址。`0.0.0.0` 监听所有网卡；`127.0.0.1` 仅允许本机访问。
- `port`：服务端口，整数 1–65535，HTTP 和 WebSocket 共用。
- `start-webui.ps1` 显式传入的 `-ListenAddress`、`-Port` 优先于配置文件。
- 修改后必须重启服务；不支持运行期间切换。文件不存在时默认使用 Vulkan；无效配置会阻止启动。
- 此配置由启动脚本读取。直接启动 EXE 时不会读取 JSON。

绿色版应双击 `启动TTS.exe`，它读取绿色版目录内的配置文件，并在系统托盘中运行。右键托盘可以打开网页、配置、日志目录，或“退出并停止服务”。修改配置后退出服务并重新启动才生效。

绿色版还支持 `speakers` 初始说话人数组和 `open_browser` 是否自动打开网页。

启动并打印服务端首包分阶段耗时：

```powershell
pwsh -NoProfile -File .\start-webui.ps1 -ProfileLatency
```

测试时保持相同文本、说话人、采样参数，分别比较首次请求和后续热运行。CPU 和 Vulkan 的浮点计算可能存在差异，切换设备后应同时试听合成效果。

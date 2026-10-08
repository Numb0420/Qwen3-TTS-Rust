# Qwen3-TTS Rust

[中文](README.md) | [English](docs/README_EN.md) | [日本語](docs/README_JA.md) | [한국어](docs/README_KO.md) | [Français](docs/README_FR.md) | [Español](docs/README_ES.md) | [Italiano](docs/README_IT.md) | [Deutsch](docs/README_DE.md) | [Русский](docs/README_RU.md) | [Português](docs/README_PT.md)

## 当前版本

本仓库维护 Windows 上的 Qwen3-TTS Rust 服务，当前部署为 **Qwen3-TTS-12Hz-1.7B-Base、Q5_K_M、单引擎**。Talker 和 Predictor 使用 llama.cpp Vulkan，声码器使用 CPU ONNX Runtime。本机验证硬件为 **AMD Radeon 8060S**；本文只列本项目实测结果，不提供 CUDA、Metal 或其他机器的性能结论。更新日期：2026-10-08。

## 已实现功能

- WebUI：完整音频生成、WebSocket 流式播放、停止、首音时间及分阶段时间显示。
- WAV 上传与对应文本输入，提取完整 ICL 克隆说话人；上传后需重启服务加载。
- 独立音频后处理：语速、音量范围 0–100，默认 50 时跳过处理。
- 指定读音：`<phoneme py="lù">六</phoneme>` 在进入引擎前替换为拼音加空格，并非模型原生 SSML。
- Windows 绿色版托盘启动器，配置监听 IP、端口及 CPU/Vulkan 模式；提供重新打包脚本。
- 默认使用说话人 39，部署列表包含 xiaolu、xiaoyan；音色文件不随源码提交。Base 服务不使用 Serena/Vivian 等 CustomVoice 预设 ID。

## 本项目首包时间

以下为 **Windows / Radeon 8060S / 1.7B Base Q5 / Vulkan + CPU 声码器** 的热运行结果。新闻文本、空指令；Temperature 0.7、Top-K 50、Top-P 1、种子 43、语速/音量 50。一次完整合成后，每种方式连续测 4 次。计时从已连接 WebSocket 发送请求至收到第一包目标 PCM，不是浏览器首音，也不是整段合成耗时。

| 启动方式 / 说话人 | 平均首包 | 范围 |
| --- | ---: | ---: |
| 本地常规启动 / 39 | 322.47 ms | 313.31–333.28 ms |
| 绿色版托盘启动 / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

同日另一组热运行新闻测试：39 为 337.61 ms、xiaolu 为 319.95 ms、xiaoyan 为 328.89 ms（各 3 次，使用此前新闻文本，不能与上表直接计算优化收益）。打包对照的 20 份完整 PCM 逐字节一致；仅证明该测试覆盖的版本与输入一致。

首包会随文本、说话人、并发排队及机器负载变化。用户网页曾测到请求首包 411 ms、首音 431 ms，本轮接口测试未复现，原因尚未确定。最新本地常规服务重启后的第一次请求约 2.47 秒，未计入热结果；当前没有为缩短首次合成而自动生成预热音频。**尚未证明每次首包 ≤300 ms。** 优化包括固定投影缓存、采样范围、4 帧提前发送、参考尾音仅更新状态及第二次波形裁剪；新音色快路径由启动校准决定，并非按名称开启。详见优化记录第 34 节及打包对照文档。

## 运行与配置

完整绿色版解压后双击 `启动TTS.exe`，托盘菜单可打开网页、配置、日志或退出服务。用户无需安装 Rust/Python/Visual Studio 构建工具；Vulkan 模式需要可用的显卡驱动。源码开发与重新打包需要构建工具。配置修改后重启：

`server-config.json`

```json
{
  "inference_device": "vulkan",
  "host": "0.0.0.0",
  "port": 3000,
  "speakers": ["39", "xiaolu", "xiaoyan"],
  "open_browser": true
}
```

[http://127.0.0.1:3000/](http://127.0.0.1:3000/)

**源码仓库不等于可直接运行的绿色版。** `models/` 权重、`runtime/`、发布 EXE 和说话人 JSON 都不在 Git 中；部分本地转换模型、优化 ONNX 和后端还没有公开下载包。必须先准备匹配的文件，当前脚本依赖已经构建并校验的发布目录，不能只克隆仓库就运行。已准备好本机发布文件后启动：

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## 自定义说话人

使用清晰、单人、无背景音乐的 WAV，提供逐字对应的参考文本。网页上传接受 1–30 秒；提取时转换为 24 kHz，原音频保留。Full 保存参考编码、文本和说话人向量；Xvec 仅保存向量，不是完整 ICL 的等价替代。网页上传会登记新音色，重启后加载；手动生成的 JSON 可通过 `-BaseSpeakers` 加入启动名单。

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP 与 WebSocket 共用配置端口（默认 3000）。`POST /api/tts` 返回 JSON，包含 Base64 WAV；`GET /api/speakers` 返回说话人列表，`GET /health` 用于健康检查。流式接口 `ws://127.0.0.1:3000/api/tts/stream` 接收下列 JSON；API 调用应显式传入参数以匹配网页设置，省略 Temperature 时服务端使用 0。

```json
{
  "text": "你好，今天天气很好。",
  "speaker": "39",
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 1,
  "seed": 43,
  "instruction": "",
  "speed": 50,
  "volume": 50
}
```

二进制消息为 24 kHz、单声道、Float32 小端 PCM；`segment_done` 表示当前文本完成，发送文本消息 `end` 结束输入，服务端以 `done` 结束。停止通过关闭当前 WebSocket 实现；`end` 不取消合成，没有独立 HTTP 停止接口。客户端收到 `done` 后仍需播放完已收到的音频。Temperature 为 0 时 Talker 使用最高概率选择，Top-K/Top-P 对该路径不起作用；Predictor 保留独立配置（0.9 / 50 / 1）。

## 文档与来源

模型文件、依赖与完整测试过程见以下文档（详细记录目前为中文）。本项目基于上游 Rust 实现，使用 Qwen3-TTS、llama.cpp 与 ONNX Runtime；保留各组件的许可证及归属。

- [Base 1.7B — models / 模型](models/base-1.7b/README.md)
- [Configuration / 配置](server-config说明.md)
- [Portable package / 绿色版](绿色版打包说明.md)
- [Optimization measurements / 优化方法](优化方法.md)
- [Preserved baseline / 基线](基线版本.md)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
## 许可证与致谢

本项目基于 [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) 继续开发。感谢原作者提供 Rust 实现与项目基础。

- [Component licenses / 组件许可证](tools/portable-licenses/)

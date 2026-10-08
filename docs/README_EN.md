# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## Current deployment

This repository maintains a Windows Qwen3-TTS service in Rust. The current deployment uses **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, one engine**. Talker and Predictor run through llama.cpp Vulkan; the vocoder runs through ONNX Runtime on the CPU. Tests used **AMD Radeon 8060S**. Only this project’s measurements are reported; CUDA, Metal and other hardware have no verified benchmark here. Updated: 2026-10-08.

## Implemented features

- WebUI: complete audio generation, WebSocket streaming, stop, time to first audible output and timing breakdown.
- WAV upload with matching transcript; full ICL voice extraction and loading after a service restart.
- Separate speed/volume postprocessing, 0–100; both bypassed at the default 50.
- Pronunciation markup: `<phoneme py="lù">六</phoneme>` becomes pinyin plus a space before inference; this is not native SSML.
- Windows portable tray launcher, configurable IP/port and CPU/Vulkan mode, and repackaging scripts.
- Deployment voices: 39 (default), xiaolu, xiaoyan. Voice files are excluded from Git. This Base service does not use CustomVoice preset IDs such as Serena/Vivian.

## First-packet measurements

Hot measurements on **Windows / Radeon 8060S / 1.7B Base Q5 / Vulkan + CPU vocoder**. Chinese news text, empty instruction; Temperature 0.7, Top-K 50, Top-P 1, seed 43, speed/volume 50. Four complete requests per mode after one completed request. Timing starts when an already connected WebSocket sends the request and ends at the first target PCM packet; it is neither browser playback latency nor total synthesis time.

| Startup / voice | Mean first packet | Range |
| --- | ---: | ---: |
| Normal local startup / 39 | 322.47 ms | 313.31–333.28 ms |
| Portable tray startup / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

A separate same-day news test measured 337.61 ms for 39, 319.95 ms for xiaolu and 328.89 ms for xiaoyan (three hot runs each, an earlier news text). Do not subtract results across these tests to claim an optimization gain. All 20 complete PCM outputs in the packaging comparison were byte-identical; that evidence covers the tested versions and inputs.

Text, voice, request queues and system load affect latency. A user measured 411 ms to the packet and 431 ms to sound in the browser; the direct API test did not reproduce that result and its cause remains unknown. The latest normal local startup’s first request took about 2.47 s and is excluded; no automatic synthesis warmup is enabled. **A ≤300 ms guarantee has not been demonstrated.** Optimizations include fixed projections, restricted sampling, earlier four-frame dispatch, state-only processing of discarded reference audio and second-call waveform pruning. New voices require startup calibration for the fast path. See optimization section 34 and the packaging comparison.

## Running and configuration

Extract the complete portable package and run `启动TTS.exe`. Its tray menu opens the page, configuration and logs, or stops the service. Users do not need Rust/Python/Visual Studio build tools; Vulkan requires a working graphics driver. Source development and repackaging require build tools. Restart after editing configuration:

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

**The source checkout is not a complete portable distribution.** Model weights, runtime files, release executables and voice JSON files are excluded. Some converted models, optimized ONNX files and backends have no public download package yet. Prepare matching assets first: the current script requires a built, verified release directory. With these files available:

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## Custom voices

Use a clear, single-speaker WAV without background music and an exact transcript. Web upload accepts 1–30 seconds; extraction converts to 24 kHz and preserves the original file. Full stores reference codes, transcript and speaker embedding; Xvec stores only the embedding and is not equivalent to full ICL. Uploaded voices are registered and load after restarting. Manually created JSON files can be included with `-BaseSpeakers`.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP and WebSocket share the configured port (3000 by default). `POST /api/tts` returns JSON with Base64 WAV; `GET /api/speakers` lists voices; `GET /health` checks readiness. Send the following JSON to `ws://127.0.0.1:3000/api/tts/stream`. Pass parameters explicitly to match the WebUI; omitted Temperature defaults to 0 on the server.

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

Binary messages contain 24 kHz mono Float32 little-endian PCM. `segment_done` ends one text segment; send the text message `end` to finish input and receive `done`. Close the WebSocket to cancel; `end` is not cancellation and there is no separate HTTP stop endpoint. Play already received audio to completion after `done`. With Temperature 0, Talker selects the highest probability and its Top-K/Top-P do not apply; Predictor retains separate settings (0.9 / 50 / 1).

## Documentation and origins

The following documents cover assets, dependencies and full experiments (detailed records are currently in Chinese). This project builds on the upstream Rust implementation and uses Qwen3-TTS, llama.cpp and ONNX Runtime; retain each component’s license and attribution.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Upstream Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) · [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Component licenses / 组件许可证](../tools/portable-licenses/)

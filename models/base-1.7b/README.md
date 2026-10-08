# 1.7B Base 模型文件准备

当前服务使用 **Qwen3-TTS-12Hz-1.7B-Base + Q5_K_M**，模型放在本目录。模型权重不提交到 GitHub；此说明、checkpoint-manifest.json 和 state-only 校验元数据保留在源码仓库。

## 目录与文件

```text
models/base-1.7b/
├── checkpoint-manifest.json
├── gguf/
│   └── qwen3_assets.gguf
├── gguf_q5_k_m/
│   ├── qwen3_tts_talker.gguf
│   └── qwen3_tts_predictor.gguf
├── onnx/
│   ├── qwen3_tts_codec_encoder.onnx
│   ├── qwen3_tts_speaker_encoder.onnx
│   ├── qwen3_tts_decoder.onnx
│   ├── qwen3_tts_decoder_state_only.onnx
│   └── qwen3_tts_decoder_state_only.json
└── tokenizer/
    └── tokenizer.json
```

| 相对本目录的路径 | 文件大小 | 下载地址／来源 |
|---|---:|---|
| `gguf/qwen3_assets.gguf` | 1459.0 MiB | 本项目本地转换；尚未发布公开下载地址，见下文 |
| `gguf_q5_k_m/qwen3_tts_predictor.gguf` | 98.5 MiB | 本项目本地转换；尚未发布公开下载地址，见下文 |
| `gguf_q5_k_m/qwen3_tts_talker.gguf` | 959.6 MiB | 本项目本地转换；尚未发布公开下载地址，见下文 |
| `onnx/qwen3_tts_codec_encoder.onnx` | 215.5 MiB | [下载](https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main/onnx/qwen3_tts_codec_encoder.onnx?download=true) |
| `onnx/qwen3_tts_decoder.onnx` | 435.6 MiB | [下载](https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main/onnx/qwen3_tts_decoder.onnx?download=true) |
| `onnx/qwen3_tts_speaker_encoder.onnx` | 46.2 MiB | [下载](https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main/onnx/qwen3_tts_speaker_encoder.onnx?download=true) |
| `tokenizer/tokenizer.json` | 10.9 MiB | [下载](https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main/tokenizer/tokenizer.json?download=true) |
| `onnx/qwen3_tts_decoder_state_only.onnx` | 155.4 MiB | 从本项目匹配的完整 decoder 裁剪生成；尚未发布独立下载地址 |
| `checkpoint-manifest.json` | 约 1.4 KiB | 随源码提供，记录本版本模型 SHA256 |
| `onnx/qwen3_tts_decoder_state_only.json` | 约 1 KiB | 随源码提供，记录裁剪图的源文件和输出 SHA256 |

上表的三个 ONNX 文件和 tokenizer 下载链接来自公开转换仓库。已于 2026-10-08 核对其 Hugging Face 文件元数据 SHA256，与本目录 checkpoint-manifest.json 中的相应值完全一致。仓库名称包含 CustomVoice，但这里只复用已核对一致的声码器、编码器和分词器；不能下载该仓库的 Talker、Predictor 或 assets 来代替本项目的 Base 文件。

## 官方原始权重下载地址

- [官方 1.7B Base 模型目录](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/main)
- [model.safetensors 原始权重](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/model.safetensors?download=true)
- [config.json](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/config.json?download=true)
- [官方 speech_tokenizer 目录](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/main/speech_tokenizer)

官方原始模型是转换来源，不能改名为 GGUF 后直接放进上述目录。当前 GGUF、资产投影及 Q5 量化在本地制作，不是从上述公开仓库直接下载的同名文件。不要用名称相似的其他项目 GGUF 替换；其张量布局、资产、量化和运行库可能不同。

## 获取本项目已验收的完整模型包

当前本地绿色版 `dist/Qwen3-TTS-Portable-win-x64.zip` 内包含完整的 `models/base-1.7b`，还包含匹配的运行库和裁剪图。可从绿色版提取整个目录供源码部署使用。

**截至本次整理，本项目配套 Base GGUF 和 state-only 图尚未发布到公开网站，因此没有可提供的有效公网直链。** 仓库维护者发布时应将配套模型文件放到 GitHub Releases 或模型托管仓库，再把真实链接补到本节。当前绿色版 ZIP 约 3.42 GiB，而 [GitHub Releases 单文件必须小于 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)，因此不能直接上传整个 ZIP；可以分开发布模型文件、制作分卷包，或使用支持此大小的模型托管服务。不要将本地路径或不存在的 Release 地址当作下载链接。

## 当前优化版本还需要的运行文件

`start-webui.ps1` 还会校验 `runtime/releases/latency-crop-2026-10-08` 的服务 EXE、优化 backend DLL、`decoder-stage3-end1292.onnx`、enabled.txt、vulkan-config.txt 及发布清单，以及 `runtime/predictor-head-b7885-v2` 配套库。这些不是上述官方原始模型下载内容；目前完整绿色版提供其运行等价文件，但内部目录布局不同。仅下载四个公开文件不能运行当前优化版源码启动脚本，需准备对应发布目录。

## 校验

文件下载或复制后，使用 checkpoint-manifest.json 中的 SHA256 校验：

```powershell
Get-FileHash .\models\base-1.7b\gguf_q5_k_m\qwen3_tts_talker.gguf -Algorithm SHA256
```

原始 Base model.safetensors 的本版本 SHA256 为 `38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c`。请同时保留完整模型清单；不能只替换其中一个 GGUF 或 assets。

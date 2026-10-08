# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## 현재 구성

Windows용 Qwen3-TTS Rust 서비스입니다. 현재 **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, 단일 엔진**을 사용합니다. Talker/Predictor는 llama.cpp Vulkan, 보코더는 CPU ONNX Runtime으로 실행합니다. **AMD Radeon 8060S**에서 검증했으며 본 프로젝트의 측정만 기재합니다. CUDA, Metal 및 다른 장비의 성능은 검증하지 않았습니다. 갱신: 2026-10-08.

## 구현된 기능

- WebUI: 전체 음성 생성, WebSocket 스트리밍, 중지, 첫 소리 지연과 단계별 시간.
- WAV와 정확한 전사문 업로드로 전체 ICL 음성 생성. 재시작 후 로드.
- 별도 속도·음량 후처리(0–100). 기본값 50에서는 생략.
- `<phoneme py="lù">六</phoneme>`는 추론 전 병음과 공백으로 치환. 모델 고유 SSML은 아닙니다.
- Windows 트레이 실행기, IP/포트와 CPU/Vulkan 설정, 재패키징 스크립트.
- 현재 음성: 39(기본), xiaolu, xiaoyan. 음성 파일은 Git에서 제외되며 Base는 Serena/Vivian 같은 CustomVoice ID를 사용하지 않습니다.

## 첫 패킷 측정

**Windows / Radeon 8060S / 1.7B Base Q5 / Vulkan + CPU 보코더**. 중국어 뉴스, 지시문 없음, Temperature 0.7, Top-K 50, Top-P 1, seed 43, 속도/음량 50. 한 번 합성 완료 후 방식별 4회 측정. 연결된 WebSocket의 요청 전송부터 첫 목표 PCM 수신까지이며 브라우저 첫 소리나 전체 합성 시간은 아닙니다.

| 실행 / 음성 | 평균 첫 패킷 | 범위 |
| --- | ---: | ---: |
| 일반 로컬 실행 / 39 | 322.47 ms | 313.31–333.28 ms |
| 포터블 트레이 실행 / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

같은 날 별도 뉴스 테스트: 39 337.61 ms, xiaolu 319.95 ms, xiaoyan 328.89 ms(각 3회, 이전 뉴스 텍스트). 서로 다른 테스트의 차이를 최적화 이득으로 계산하지 않습니다. 패키징 비교의 전체 PCM 20개는 바이트 단위로 동일했으며 검증한 버전과 입력에 한정된 결과입니다.

텍스트·음성·대기열·시스템 부하에 따라 지연이 달라집니다. 사용자의 브라우저 측정은 패킷 411 ms, 첫 소리 431 ms였지만 직접 API 테스트에서 재현하지 못했고 원인은 미확정입니다. 일반 로컬 재시작 후 첫 요청은 약 2.47초로 표에서 제외했습니다. 자동 합성 예열은 하지 않습니다. **항상 300 ms 이하를 보장하지 않습니다.** 고정 투영, 샘플링 범위, 4프레임 조기 전송, 참조 오디오 상태만 갱신, 두 번째 파형 가지치기를 적용했습니다. 새 음성의 빠른 경로는 시작 시 교정 결과에 따릅니다. 최적화 문서 34절과 패키징 비교를 참조하세요.

## 실행 및 설정

전체 포터블 패키지를 풀고 `启动TTS.exe`를 실행합니다. 트레이에서 웹·설정·로그를 열거나 종료합니다. 사용자는 Rust/Python/Visual Studio 빌드 도구가 필요 없지만 Vulkan 드라이버는 필요합니다. 소스 개발·재패키징에는 빌드 도구가 필요합니다. 설정 변경 후 재시작:

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

**소스 저장소만으로 포터블 서비스를 바로 실행할 수 없습니다.** 모델, runtime, 릴리스 EXE, 음성 JSON은 제외되어 있습니다. 일부 변환 모델·최적화 ONNX·백엔드의 공개 다운로드는 아직 없습니다. 일치하는 파일과 빌드·검증된 릴리스 디렉터리를 준비한 다음 실행하세요:

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## 사용자 음성

배경 음악 없이 한 사람이 명확히 말한 WAV와 정확한 전사문을 사용합니다. 웹 업로드는 1–30초, 추출은 24 kHz 변환이며 원본을 보존합니다. Full은 참조 코드·텍스트·화자 벡터, Xvec은 벡터만 저장하므로 전체 ICL과 동등하지 않습니다. 업로드 음성은 등록 후 재시작하여 로드하고 수동 JSON은 `-BaseSpeakers`로 추가합니다.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP와 WebSocket은 같은 포트(기본 3000)입니다. `POST /api/tts`는 Base64 WAV를 담은 JSON, `GET /api/speakers`는 음성 목록, `GET /health`는 상태 확인입니다. 다음 JSON을 `ws://127.0.0.1:3000/api/tts/stream`으로 보내세요. WebUI와 맞추려면 값을 명시해야 하며 Temperature 생략 시 서버는 0을 사용합니다.

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

바이너리는 24 kHz 모노 Float32 리틀 엔디언 PCM입니다. `segment_done`은 텍스트 구간 완료, 문자열 `end`는 입력 종료, `done`은 처리 완료입니다. 중지는 WebSocket을 닫아서 수행하며 `end`는 취소가 아닙니다. 별도 HTTP 중지 API는 없습니다. `done` 후에도 수신한 오디오는 끝까지 재생하세요. Temperature 0이면 Talker는 최고 확률을 선택하므로 해당 Top-K/Top-P는 적용되지 않습니다. Predictor는 별도 설정(0.9 / 50 / 1)을 유지합니다.

## 문서와 출처

모델·의존성·실험 상세 문서는 아래에 있습니다(현재 상세 기록은 중국어). 상위 Rust 구현을 기반으로 Qwen3-TTS, llama.cpp, ONNX Runtime을 사용하며 각 구성 요소의 라이선스와 출처를 유지합니다.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
## 라이선스 및 감사

이 프로젝트는 [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust)를 기반으로 개발되었습니다. Rust 구현과 프로젝트 기반을 제공한 원저자에게 감사드립니다.

- [Component licenses / 组件许可证](../tools/portable-licenses/)

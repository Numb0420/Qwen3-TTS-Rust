# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## Aktueller Betrieb

Qwen3-TTS-Dienst in Rust für Windows. Aktuell: **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, eine Engine**. Talker/Predictor nutzen llama.cpp Vulkan, der Vocoder ONNX Runtime auf der CPU. Geprüft auf **AMD Radeon 8060S**. Nur eigene Messungen werden genannt; CUDA, Metal und andere Rechner wurden hier nicht vermessen. Stand: 2026-10-08.

## Implementierte Funktionen

- WebUI: vollständige Audiodatei, WebSocket-Streaming, Stopp und Zeit bis zum ersten Ton mit Aufschlüsselung.
- WAV mit genauem Transkript hochladen, vollständige ICL-Stimme extrahieren, nach Neustart laden.
- Separate Nachbearbeitung für Tempo/Lautstärke (0–100), bei 50 übersprungen.
- `<phoneme py="lù">六</phoneme>` wird vor der Inferenz durch Pinyin plus Leerzeichen ersetzt; kein natives SSML.
- Windows-Tray-Start, IP/Port und CPU/Vulkan konfigurierbar, Skripte zum Paketbau.
- Stimmen: 39 (Standard), xiaolu, xiaoyan. Dateien nicht in Git; Base verwendet keine CustomVoice-IDs wie Serena/Vivian.

## Messungen des ersten Pakets

**Windows / Radeon 8060S / Base 1.7B Q5 / Vulkan + CPU-Vocoder**. Chinesischer Nachrichtentext, leere Anweisung; Temperature 0.7, Top-K 50, Top-P 1, Seed 43, Tempo/Lautstärke 50. Vier vollständige Anfragen nach einer abgeschlossenen Synthese. Vom Senden über verbundenes WebSocket bis zum ersten Ziel-PCM-Paket; weder erster Browserton noch gesamte Synthesezeit.

| Start / Stimme | Mittelwert | Bereich |
| --- | ---: | ---: |
| Normaler lokaler Start / 39 | 322.47 ms | 313.31–333.28 ms |
| Portabler Tray-Start / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

Separater Test am selben Tag: 39 337.61 ms, xiaolu 319.95 ms, xiaoyan 328.89 ms (je drei warme Läufe, früherer Nachrichtentext). Differenzen verschiedener Tests sind kein Optimierungsnachweis. Alle 20 vollständigen PCM-Ausgaben des Paketvergleichs waren bytegleich; dies gilt für geprüfte Versionen und Eingaben.

Text, Stimme, Warteschlangen und Last beeinflussen die Zeit. Ein Nutzer meldete 411 ms bis zum Paket und 431 ms bis zum Ton; per API nicht reproduziert, Ursache offen. Erste Anfrage nach normalem lokalen Neustart: etwa 2.47 s, nicht in der Tabelle. Keine automatische Synthese zum Aufwärmen. **Stets ≤300 ms ist nicht nachgewiesen.** Optimierungen: feste Projektionen, Samplingbereich, frühere Vier-Frame-Ausgabe, reine Zustandsaktualisierung für verworfene Referenzwellen und weniger Wellenberechnung im zweiten Aufruf. Neue Stimmen müssen die Startkalibrierung bestehen. Details in Abschnitt 34 und Paketvergleich.

## Start und Konfiguration

Vollständiges Paket entpacken, `启动TTS.exe` starten. Tray-Menü öffnet Seite, Einstellungen und Logs oder beendet den Dienst. Nutzer brauchen keine Rust/Python/Visual-Studio-Buildtools; Vulkan benötigt einen Grafiktreiber. Entwicklung und Paketbau benötigen Buildtools. Nach Änderungen neu starten:

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

**Der Quellcode allein ist kein lauffähiges portables Paket.** Modelle, Runtime, Release-EXE und Stimmen-JSON fehlen in Git. Einige konvertierte Modelle, optimierte ONNX-Dateien und Backends sind noch nicht öffentlich herunterladbar. Passende Dateien und gebautes, geprüftes Release-Verzeichnis vorbereiten, danach starten:

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## Eigene Stimmen

Klare WAV einer Person ohne Musik mit genauem Transkript. Web-Upload: 1–30 s; Umwandlung in 24 kHz, Original bleibt erhalten. Full speichert Referenzcodes, Text und Vektor; Xvec nur den Vektor, nicht gleichwertig zu vollständigem ICL. Nach Upload neu starten; manuelle JSON mit `-BaseSpeakers` aufnehmen.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP und WebSocket nutzen denselben Port (Standard 3000). `POST /api/tts`: JSON mit Base64-WAV; `GET /api/speakers`: Stimmen; `GET /health`: Status. Folgendes JSON an `ws://127.0.0.1:3000/api/tts/stream` senden. Für WebUI-Werte Parameter explizit setzen; ohne Temperature verwendet der Server 0.

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

Binärdaten: 24 kHz Mono Float32 Little-Endian PCM. `segment_done` beendet ein Segment; Textnachricht `end` beendet Eingabe, danach folgt `done`. Zum Abbrechen WebSocket schließen; `end` bricht nicht ab, eine separate HTTP-Stopp-API gibt es nicht. Nach `done` empfangenes Audio fertig abspielen. Temperature 0 wählt im Talker das Maximum; dessen Top-K/Top-P greifen nicht. Predictor behält separate Werte (0.9 / 50 / 1).

## Dokumentation und Herkunft

Ausführliche Dokumente derzeit auf Chinesisch. Basiert auf der ursprünglichen Rust-Implementierung und verwendet Qwen3-TTS, llama.cpp und ONNX Runtime. Lizenzen und Urheberangaben der Komponenten beibehalten.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Upstream Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) · [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Component licenses / 组件许可证](../tools/portable-licenses/)

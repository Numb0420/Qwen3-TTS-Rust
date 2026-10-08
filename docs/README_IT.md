# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## Configurazione attuale

Servizio Qwen3-TTS Rust per Windows: **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, un motore**. Talker/Predictor usano llama.cpp Vulkan; il vocoder usa ONNX Runtime su CPU. Verificato su **AMD Radeon 8060S**. Sono riportati solo test del progetto; CUDA, Metal e altri sistemi non hanno misure verificate qui. Aggiornato: 2026-10-08.

## Funzioni implementate

- WebUI: audio completo, streaming WebSocket, arresto e tempo al primo suono con dettaglio.
- WAV e trascrizione esatta, estrazione ICL completa, caricamento dopo riavvio.
- Post-elaborazione separata di velocità/volume (0–100), saltata al valore 50.
- `<phoneme py="lù">六</phoneme>` diventa pinyin e spazio prima dell’inferenza; non è SSML nativo.
- Avvio Windows nell’area di notifica, IP/porta e CPU/Vulkan configurabili, script di packaging.
- Voci: 39 (predefinita), xiaolu, xiaoyan. File esclusi da Git; Base non usa ID CustomVoice Serena/Vivian.

## Misure del primo pacchetto

**Windows / Radeon 8060S / Base 1.7B Q5 / Vulkan + vocoder CPU**. Notizie cinesi, istruzione vuota; Temperature 0.7, Top-K 50, Top-P 1, seed 43, velocità/volume 50. Quattro richieste complete dopo una sintesi iniziale. Dall’invio su WebSocket già connesso al primo PCM destinato alla riproduzione, non al primo suono nel browser o alla fine della sintesi.

| Avvio / voce | Media | Intervallo |
| --- | ---: | ---: |
| Avvio locale normale / 39 | 322.47 ms | 313.31–333.28 ms |
| Avvio portatile dall’area di notifica / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

Altro test dello stesso giorno: 39 337.61 ms, xiaolu 319.95 ms, xiaoyan 328.89 ms (tre prove ciascuno, testo precedente). Non ricavare guadagni sottraendo test diversi. Tutte le 20 uscite PCM complete del confronto portatile sono identiche byte per byte; la prova riguarda versioni e input verificati.

Testo, voce, code e carico influenzano la latenza. Un utente ha misurato 411 ms al pacchetto e 431 ms al suono; l’API non ha riprodotto il risultato, causa non identificata. Prima richiesta locale dopo riavvio: circa 2.47 s, esclusa; nessuna sintesi automatica di riscaldamento. **Non è dimostrato un limite ≤300 ms per ogni richiesta.** Ottimizzazioni: proiezioni fisse, intervallo di campionamento, invio anticipato di quattro frame, soli stati per audio di riferimento scartato e riduzione dell’onda alla seconda chiamata. Le nuove voci richiedono calibrazione all’avvio. Vedere sezione 34 e confronto packaging.

## Avvio e configurazione

Estrarre il pacchetto completo e avviare `启动TTS.exe`. Il menu apre pagina, configurazione e log oppure arresta il servizio. All’utente non servono Rust/Python/Visual Studio; Vulkan richiede un driver grafico. Sviluppo e packaging richiedono strumenti di compilazione. Riavviare dopo modifiche:

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

**Il repository sorgente non è un pacchetto portatile completo.** Modelli, runtime, EXE e JSON delle voci sono esclusi. Alcuni modelli convertiti, ONNX ottimizzati e backend non hanno ancora download pubblici. Preparare file corrispondenti e directory di rilascio compilata e verificata prima dell’avvio:

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## Voci personalizzate

WAV chiaro, singolo parlante, senza musica, con trascrizione esatta. Upload web: 1–30 s; conversione a 24 kHz con originale conservato. Full salva codici, testo e vettore; Xvec solo il vettore e non equivale a ICL completo. Riavviare dopo upload; JSON manuali aggiungibili con `-BaseSpeakers`.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP e WebSocket condividono la porta (3000). `POST /api/tts`: JSON con WAV Base64; `GET /api/speakers`: voci; `GET /health`: stato. Inviare questo JSON a `ws://127.0.0.1:3000/api/tts/stream`. Specificare i parametri per replicare il WebUI; Temperature omessa vale 0 sul server.

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

Binario: PCM mono 24 kHz Float32 little-endian. `segment_done` termina un segmento; il testo `end` chiude l’input, seguito da `done`. Chiudere WebSocket per annullare; `end` non annulla e non esiste endpoint HTTP separato di arresto. Completare la riproduzione dopo `done`. Temperature 0 seleziona il massimo in Talker e disattiva i suoi Top-K/Top-P. Predictor mantiene configurazione separata (0.9 / 50 / 1).

## Documentazione e origini

Documentazione dettagliata attualmente in cinese. Basato sull’implementazione Rust originale, usa Qwen3-TTS, llama.cpp e ONNX Runtime. Conservare licenze e attribuzioni dei componenti.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Upstream Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) · [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Component licenses / 组件许可证](../tools/portable-licenses/)

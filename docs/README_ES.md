# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## Despliegue actual

Servicio Qwen3-TTS en Rust para Windows. Usa **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, un motor**. Talker/Predictor ejecutan llama.cpp Vulkan; el vocoder usa ONNX Runtime en CPU. Validado en **AMD Radeon 8060S**. Solo se publican mediciones de este proyecto; CUDA, Metal y otros equipos no tienen resultados verificados aquí. Actualizado: 2026-10-08.

## Funciones implementadas

- WebUI: generación completa, streaming WebSocket, detener y tiempo al primer sonido con desglose.
- Carga WAV y transcripción exacta, extracción ICL completa y carga al reiniciar.
- Posprocesado separado de velocidad/volumen (0–100), omitido en 50.
- `<phoneme py="lù">六</phoneme>` se sustituye por pinyin y un espacio antes de inferir; no es SSML nativo.
- Lanzador portátil Windows con bandeja, IP/puerto y CPU/Vulkan configurables, scripts de empaquetado.
- Voces: 39 (predeterminada), xiaolu, xiaoyan. Sus archivos no están en Git. Base no utiliza IDs CustomVoice como Serena/Vivian.

## Tiempo al primer paquete

**Windows / Radeon 8060S / Base 1.7B Q5 / Vulkan + vocoder CPU**. Noticias en chino, instrucción vacía; Temperature 0.7, Top-K 50, Top-P 1, semilla 43, velocidad/volumen 50. Cuatro solicitudes completas tras una síntesis inicial. Desde enviar por un WebSocket conectado hasta el primer PCM objetivo: no mide el sonido del navegador ni la síntesis total.

| Inicio / voz | Media | Intervalo |
| --- | ---: | ---: |
| Inicio local normal / 39 | 322.47 ms | 313.31–333.28 ms |
| Inicio portátil desde bandeja / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

Otro ensayo del mismo día: 39 337.61 ms, xiaolu 319.95 ms, xiaoyan 328.89 ms (tres por voz, texto anterior). No restar ensayos distintos para atribuir mejoras. Las 20 salidas PCM completas del comparativo portátil fueron idénticas byte a byte; esto cubre las versiones y entradas probadas.

Texto, voz, colas y carga afectan al tiempo. Un usuario midió 411 ms al paquete y 431 ms al sonido; la API no lo reprodujo y la causa no está determinada. La primera petición local tras reiniciar tardó unos 2.47 s, excluidos de la tabla; no hay precalentamiento mediante síntesis automática. **No se ha demostrado ≤300 ms en cada petición.** Optimizaciones: proyecciones fijas, rango de muestreo, envío adelantado de cuatro cuadros, actualización de estados sin onda de referencia y recorte del cálculo de onda en la segunda llamada. Las nuevas voces pasan una calibración al inicio. Véanse sección 34 y comparativo del paquete.

## Ejecución y configuración

Descomprimir el paquete completo y ejecutar `启动TTS.exe`. La bandeja abre web, configuración y registros, o detiene el servicio. No requiere herramientas Rust/Python/Visual Studio para el usuario; Vulkan necesita controlador gráfico. Desarrollo y empaquetado sí requieren herramientas de compilación. Reiniciar tras cambios:

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

**El repositorio fuente no es un paquete portátil completo.** Se excluyen modelos, runtime, EXE y JSON de voces. Algunos modelos convertidos, ONNX optimizados y backends aún no tienen descarga pública. Preparar los archivos correspondientes y el directorio de publicación compilado y verificado antes de ejecutar:

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## Voces personalizadas

WAV claro de un hablante, sin música, con transcripción exacta. Carga web de 1–30 s; conversión a 24 kHz conservando el original. Full almacena códigos, texto y vector; Xvec solo el vector y no equivale a ICL completo. Reiniciar tras la carga; añadir JSON manuales con `-BaseSpeakers`.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP y WebSocket comparten puerto (3000 predeterminado). `POST /api/tts`: JSON con WAV Base64; `GET /api/speakers`: voces; `GET /health`: estado. Enviar este JSON a `ws://127.0.0.1:3000/api/tts/stream`. Indicar los parámetros para coincidir con la web; omitir Temperature implica 0 en el servidor.

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

Binario: PCM mono 24 kHz Float32 little-endian. `segment_done` termina un segmento; enviar el texto `end` termina la entrada y se recibe `done`. Cerrar WebSocket cancela; `end` no cancela y no hay API HTTP de parada separada. Reproducir hasta el final lo recibido después de `done`. Con Temperature 0, Talker elige máxima probabilidad y sus Top-K/Top-P no actúan. Predictor mantiene ajustes independientes (0.9 / 50 / 1).

## Documentación y origen

Documentos detallados actualmente en chino. Basado en la implementación Rust original; utiliza Qwen3-TTS, llama.cpp y ONNX Runtime. Conservar sus licencias y atribuciones.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
## Licencias y agradecimientos

Este proyecto se desarrolla a partir de [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust). Gracias al autor original por la implementación en Rust y la base del proyecto.

- [Component licenses / 组件许可证](../tools/portable-licenses/)

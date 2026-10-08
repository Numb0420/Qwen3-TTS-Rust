# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## Configuração atual

Serviço Qwen3-TTS em Rust para Windows: **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, um motor**. Talker/Predictor usam llama.cpp Vulkan; o vocoder usa ONNX Runtime na CPU. Validado em **AMD Radeon 8060S**. Apenas medições deste projeto; CUDA, Metal e outros equipamentos não têm resultados verificados aqui. Atualizado: 2026-10-08.

## Funcionalidades implementadas

- WebUI: áudio completo, streaming WebSocket, parar e tempo até ao primeiro som com detalhes.
- WAV com transcrição exata, extração ICL completa, carregamento após reinício.
- Pós-processamento separado de velocidade/volume (0–100), ignorado em 50.
- `<phoneme py="lù">六</phoneme>` vira pinyin e espaço antes da inferência; não é SSML nativo.
- Iniciador Windows na bandeja, IP/porta e CPU/Vulkan configuráveis, scripts de empacotamento.
- Vozes: 39 (padrão), xiaolu, xiaoyan. Arquivos excluídos do Git; Base não usa IDs CustomVoice Serena/Vivian.

## Medições do primeiro pacote

**Windows / Radeon 8060S / Base 1.7B Q5 / Vulkan + vocoder CPU**. Notícias em chinês, instrução vazia; Temperature 0.7, Top-K 50, Top-P 1, seed 43, velocidade/volume 50. Quatro pedidos completos após uma síntese inicial. Do envio num WebSocket conectado até ao primeiro PCM de destino; não mede o primeiro som do navegador nem a síntese total.

| Inicialização / voz | Média | Intervalo |
| --- | ---: | ---: |
| Inicialização local normal / 39 | 322.47 ms | 313.31–333.28 ms |
| Inicialização portátil na bandeja / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

Outro teste no mesmo dia: 39 337.61 ms, xiaolu 319.95 ms, xiaoyan 328.89 ms (três por voz, texto anterior). Não subtrair testes diferentes para calcular ganhos. Os 20 PCM completos do comparativo portátil foram idênticos byte a byte, para as versões e entradas verificadas.

Texto, voz, filas e carga afetam a latência. Um utilizador mediu 411 ms até ao pacote e 431 ms até ao som; a API não reproduziu o resultado e a causa continua desconhecida. Primeiro pedido local após reinício: cerca de 2.47 s, excluído da tabela. Sem aquecimento por síntese automática. **Não se demonstrou ≤300 ms em todos os pedidos.** Otimizações: projeções fixas, intervalo de amostragem, envio antecipado de quatro quadros, atualização apenas de estados para áudio de referência descartado e redução da onda na segunda chamada. Novas vozes dependem da calibração na inicialização. Ver secção 34 e comparação do pacote.

## Execução e configuração

Extrair o pacote completo e executar `启动TTS.exe`. O menu abre página, configuração e logs ou encerra o serviço. Utilizadores não precisam das ferramentas Rust/Python/Visual Studio; Vulkan exige driver gráfico. Desenvolvimento e empacotamento precisam de ferramentas de compilação. Reiniciar após alterações:

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

**O repositório de código não é um pacote portátil completo.** Modelos, runtime, EXE e JSON de vozes são excluídos. Alguns modelos convertidos, ONNX otimizados e backends ainda não têm download público. Preparar arquivos compatíveis e o diretório compilado e verificado da versão antes de executar:

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## Vozes personalizadas

WAV claro de um falante sem música e transcrição exata. Upload web: 1–30 s; conversão para 24 kHz mantendo o original. Full guarda códigos, texto e vetor; Xvec só o vetor, não equivalente a ICL completo. Reiniciar após upload; incluir JSON manuais com `-BaseSpeakers`.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP/WebSocket partilham a porta (3000). `POST /api/tts`: JSON com WAV Base64; `GET /api/speakers`: vozes; `GET /health`: estado. Enviar este JSON a `ws://127.0.0.1:3000/api/tts/stream`. Definir parâmetros explicitamente para coincidir com o WebUI; Temperature omitida vale 0 no servidor.

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

Binário: PCM mono 24 kHz Float32 little-endian. `segment_done` termina um segmento; o texto `end` termina a entrada, seguido de `done`. Fechar WebSocket para cancelar; `end` não cancela e não há endpoint HTTP separado de paragem. Reproduzir o áudio recebido até ao fim após `done`. Temperature 0 escolhe o máximo em Talker e os seus Top-K/Top-P não atuam. Predictor mantém ajustes separados (0.9 / 50 / 1).

## Documentação e origem

Documentação detalhada atualmente em chinês. Baseado na implementação Rust original e em Qwen3-TTS, llama.cpp e ONNX Runtime. Preservar licenças e atribuições dos componentes.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
## Licenças e agradecimentos

Este projeto é desenvolvido a partir de [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust). Agradecemos ao autor original pela implementação em Rust e pela base do projeto.

- [Component licenses / 组件许可证](../tools/portable-licenses/)

# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## 現在の構成

Windows 向けの Qwen3-TTS Rust サービスです。現在は **Qwen3-TTS-12Hz-1.7B-Base、Q5_K_M、単一エンジン**を使用。Talker/Predictor は llama.cpp Vulkan、ボコーダーは CPU の ONNX Runtime で実行します。検証機は **AMD Radeon 8060S**。本プロジェクトの測定結果のみを掲載し、CUDA・Metal・他の機種の性能は未検証です。更新日：2026-10-08。

## 実装済み機能

- WebUI：音声生成、WebSocket ストリーミング、停止、最初の発声までの時間と内訳。
- WAV と一致する文字起こしをアップロードして ICL 音声を作成。読み込みには再起動が必要。
- 独立した速度・音量の後処理（0–100）。既定値 50 では処理を省略。
- `<phoneme py="lù">六</phoneme>` を推論前にピンインと空白に変換。ネイティブ SSML ではありません。
- Windows トレイ起動、IP・ポート・CPU/Vulkan 設定、再パッケージ用スクリプト。
- 現在の音声：39（既定）、xiaolu、xiaoyan。音声ファイルは Git に含まれず、Serena/Vivian の CustomVoice ID はこの Base サービスでは使用しません。

## 初回パケットの測定

**Windows / Radeon 8060S / 1.7B Base Q5 / Vulkan + CPU ボコーダー**で測定。中国語ニュース、指示なし、Temperature 0.7、Top-K 50、Top-P 1、seed 43、速度・音量 50。1 回の合成完了後に各方式を 4 回測定。接続済み WebSocket の送信から最初の対象 PCM 受信までの時間であり、ブラウザーの発声時刻や全体の合成時間ではありません。

| 起動方式 / 音声 | 平均 | 範囲 |
| --- | ---: | ---: |
| 通常のローカル起動 / 39 | 322.47 ms | 313.31–333.28 ms |
| ポータブル版のトレイ起動 / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

同日の別テストでは 39：337.61 ms、xiaolu：319.95 ms、xiaoyan：328.89 ms（各 3 回、以前のニュース文）。テスト間の差を最適化の効果として計算しないでください。パッケージ比較の全 20 回の PCM はバイト単位で一致しました。これは検証した入力・版に限る結果です。

文章、音声、待ち行列、負荷により時間は変わります。利用者のブラウザーではパケット 411 ms、発声 431 ms が報告されましたが、API 測定では再現せず原因は未確定です。通常ローカル起動後の初回は約 2.47 秒で、表から除外。自動合成によるウォームアップは行いません。**常に 300 ms 以下という保証は未達成です。** 固定投影、サンプリング範囲、4 フレームの早期送信、参照音声の状態のみ更新、2 回目の波形裁剪を最適化。新音声は起動時の校正を通過すると高速経路を使用します。詳細は最適化記録第 34 節とパッケージ比較を参照。

## 起動と設定

完全なポータブル版を展開して `启动TTS.exe` を起動。トレイからページ・設定・ログを開き、サービスを終了できます。利用者に Rust/Python/Visual Studio ビルドツールは不要です。Vulkan には対応ドライバーが必要。開発・再パッケージにはビルドツールが必要です。設定変更後は再起動：

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

**ソースだけではポータブル版として実行できません。** モデル、runtime、公開用 EXE、音声 JSON は Git に含まれません。変換モデル・最適化 ONNX・バックエンドの一部はまだ公開ダウンロードがなく、対応ファイルの準備が必要です。スクリプトは構築・検証済みのリリースディレクトリを前提とします。準備後の起動：

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## カスタム音声

雑音や音楽のない一人の明瞭な WAV と正確な文字起こしを使用。Web は 1–30 秒を受け付け、24 kHz に変換し原本を保持します。Full は参照コード・文章・話者ベクトル、Xvec はベクトルのみで、完全 ICL と同等ではありません。アップロードは登録後再起動で読み込み。手動 JSON は `-BaseSpeakers` で追加できます。

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

HTTP と WebSocket は同じポート（既定 3000）。`POST /api/tts` は Base64 WAV を含む JSON、`GET /api/speakers` は音声一覧、`GET /health` は状態確認。以下の JSON を `ws://127.0.0.1:3000/api/tts/stream` に送信。WebUI と合わせるには明示的に設定し、Temperature 省略時はサーバーで 0 になります。

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

バイナリは 24 kHz・モノラル・Float32 リトルエンディアン PCM。`segment_done` は一区間の完了、文字列 `end` は入力終了、`done` は全処理完了です。停止は WebSocket 切断で実行。`end` は停止ではなく、専用 HTTP 停止 API もありません。`done` 後も受信済み音声を最後まで再生してください。Temperature 0 では Talker が最高確率を選び Top-K/Top-P は適用されません。Predictor は独立設定（0.9 / 50 / 1）です。

## 関連資料と出典

モデル・依存関係・実験の詳細は以下を参照（詳細資料は現在中国語）。上流 Rust 実装を基に Qwen3-TTS、llama.cpp、ONNX Runtime を使用し、各ライセンスと帰属を保持します。

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
## ライセンスと謝辞

本プロジェクトは [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) を基に開発しています。Rust 実装とプロジェクトの基盤を提供した原作者に感謝します。

- [Component licenses / 组件许可证](../tools/portable-licenses/)

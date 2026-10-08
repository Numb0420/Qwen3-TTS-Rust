param(
    [ValidateSet('q5_k_m', 'q8_0', 'none')]
    [string]$Quant = 'q5_k_m',
    [int]$Port = 3000,
    [string]$ListenAddress = '0.0.0.0',
    [string[]]$BaseSpeakers = @('39', 'xiaolu', 'xiaoyan'),
    [switch]$VerboseInference,
    [switch]$ProfileLatency,
    [switch]$KeepOnnxSpinning,
    [switch]$DisableStateOnlyDecoder,
    [switch]$DisableCompactSampler,
    [switch]$DisablePredictorHeadRange,
    [switch]$Baseline,
    [switch]$PreviousOptimization
)

$ErrorActionPreference = 'Stop'
if ($Baseline -and $PreviousOptimization) { throw 'Choose either -Baseline or -PreviousOptimization.' }
Set-Location -LiteralPath $PSScriptRoot

# Startup configuration: applies to Talker and Predictor, not the CPU ONNX decoder.
$taskConfigPath = Join-Path $PSScriptRoot 'server-config.json'
$taskInferenceDevice = 'vulkan'
if (Test-Path -LiteralPath $taskConfigPath -PathType Leaf) {
    $taskServerConfig = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
    if ($taskServerConfig.inference_device -isnot [string] -or $taskServerConfig.inference_device -cnotin @('cpu', 'vulkan')) {
        throw 'server-config.json: inference_device must be "cpu" or "vulkan".'
    }
    $taskInferenceDevice = $taskServerConfig.inference_device
    if ($taskServerConfig.PSObject.Properties.Name -contains 'host') {
        $taskParsedIP = $null
        if ($taskServerConfig.host -isnot [string] -or -not [Net.IPAddress]::TryParse($taskServerConfig.host, [ref]$taskParsedIP) -or $taskParsedIP.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) {
            throw 'server-config.json: host must be an IPv4 address.'
        }
        if (-not $PSBoundParameters.ContainsKey('ListenAddress')) { $ListenAddress = $taskServerConfig.host }
    }
    if ($taskServerConfig.PSObject.Properties.Name -contains 'port') {
        if ($taskServerConfig.port -isnot [long] -and $taskServerConfig.port -isnot [int]) { throw 'server-config.json: port must be an integer.' }
        if ($taskServerConfig.port -lt 1 -or $taskServerConfig.port -gt 65535) { throw 'server-config.json: port must be 1-65535.' }
        if (-not $PSBoundParameters.ContainsKey('Port')) { $Port = [int]$taskServerConfig.port }
    }
}

$quantFolder = switch ($Quant) {
    'q5_k_m' { 'gguf_q5_k_m' }
    'q8_0' { 'gguf_q8_0' }
    default { 'gguf' }
}

$baseRoot = Join-Path $PSScriptRoot 'models/base-1.7b'
$required = @(
    "$baseRoot/gguf/qwen3_assets.gguf"
    "$baseRoot/onnx/qwen3_tts_decoder.onnx"
    "$baseRoot/onnx/qwen3_tts_codec_encoder.onnx"
    "$baseRoot/onnx/qwen3_tts_speaker_encoder.onnx"
    "$baseRoot/tokenizer/tokenizer.json"
)
# Clone voices exposed by the Base service. Speaker 39 has passed listening
# validation; xiaolu and xiaoyan were added for user testing on this same bundle.
if ($BaseSpeakers.Count -eq 0) { throw 'At least one validated Base cloning speaker is required.' }
$baseArguments = @('--host', $ListenAddress, '--base-only', '--model-dir', $baseRoot, '--base-model-dir', $baseRoot, '--base-speakers', ($BaseSpeakers -join ','))
if (-not $DisableStateOnlyDecoder) {
    $taskStateDecoder = Join-Path $baseRoot 'onnx/qwen3_tts_decoder_state_only.onnx'
    $taskStateManifest = [System.IO.Path]::ChangeExtension($taskStateDecoder, '.json')
    $baseArguments += @('--base-state-decoder', $taskStateDecoder)
    $required += @($taskStateDecoder, $taskStateManifest)
}
$required += @(
    "$baseRoot/checkpoint-manifest.json"
    "$baseRoot/$quantFolder/qwen3_tts_talker.gguf"
    "$baseRoot/$quantFolder/qwen3_tts_predictor.gguf"
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    Write-Host 'Missing model files:' -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $_" }
    throw 'Install the model files listed above. For the generated state-only graph, restore the validated files or use -DisableStateOnlyDecoder.'
}
if (-not $DisableStateOnlyDecoder) {
    # A pruned graph must be regenerated and revalidated if the original decoder changes.
    $taskStateInfo = Get-Content -LiteralPath $taskStateManifest -Raw | ConvertFrom-Json
    $taskOriginalHash = (Get-FileHash -LiteralPath (Join-Path $baseRoot 'onnx/qwen3_tts_decoder.onnx') -Algorithm SHA256).Hash
    $taskStateHash = (Get-FileHash -LiteralPath $taskStateDecoder -Algorithm SHA256).Hash
    if ($taskStateInfo.sourceSha256 -ne $taskOriginalHash -or $taskStateInfo.outputSha256 -ne $taskStateHash) {
        throw 'State-only decoder checksum mismatch. Restore the validated files or start with -DisableStateOnlyDecoder; do not use a graph from another decoder.'
    }
}

# Keep recurring hot-path improvements. Startup-only optimizations were withdrawn; see 优化方法.md section 14.
foreach ($taskFlag in @('QWEN_PRONUNCIATION_ASSETS','QWEN_PRONUNCIATION_DEVICE','QWEN_PRONUNCIATION_PROBE','QWEN_FULL_WARMUP','QWEN_PROJECTION_FILE','QWEN_OPT_MODE_FILE','QWEN_ONNX_THREADS','QWEN_ONNX_INTER_THREADS','QWEN_LLAMA_THREADS','QWEN_ONNX_NO_SPIN','QWEN_ONNX_OP_PROFILE','QWEN_DECODER_PROFILE','QWEN_SAMPLER_PROFILE','QWEN_CODEC_TRACE_DIR','QWEN_PREDICTOR_SAMPLER_MODE_FILE','QWEN_PREDICTOR_SAMPLER_LEGACY','QWEN_TTFA_PROFILE','GGML_VK_PERF_LOGGER','GGML_VK_PERF_LOGGER_FREQUENCY','GGML_VK_PERF_LOGGER_CONCURRENT')) {
    Remove-Item -LiteralPath "Env:$taskFlag" -ErrorAction SilentlyContinue
}
# Apply after clearing diagnostic environment variables so the setting survives.
$env:QWEN_PRONUNCIATION_DEVICE = $taskInferenceDevice
Write-Host "Inference device: $taskInferenceDevice (Talker / Predictor); ONNX decoder: CPU."
$env:QWEN_OPT_FLAGS = if ($VerboseInference) { 'baseline' } else { 'quiet' }
Remove-Item -LiteralPath Env:QWEN_OUTPUT_PROJECTION_PROFILE -ErrorAction SilentlyContinue
foreach ($taskExperimentalFlag in @('QWEN_ONNX_DYNAMIC_BLOCK_BASE','QWEN_ONNX_GLOBAL_POOL','QWEN_ONNX_INDEPENDENT_POOL','QWEN_ONNX_AFFINITY','QWEN_LLAMA_RUNTIME_DIR','QWEN_PREDICTOR_HEAD_RANGE','QWEN_PREDICTOR_HEAD_MODEL_DIR','QWEN_PREDICTOR_LOGITS_TRACE_DIR','QWEN_TALKER_TRACE_DIR','QWEN_TALKER_LAST_OUTPUT','QWEN_TALKER_LAST_COPY','QWEN_LAST_COPY_MODE_FILE','QWEN_DIAG_WAVE_PREFIX_MODEL','QWEN_DIAG_WAVE_PREFIX_MODE_FILE','QWEN_DIAG_ACTIVATION','QWEN_DIAG_EOS_MODE_FILE','QWEN_DIAG_ACTIVATION_PARALLEL','QWEN_DIAG_PROJECTION_MODE_FILE','QWEN_DIAG_PROFILE_MODEL','QWEN_DIAG_PROFILE_PREFIX')) {
    Remove-Item -LiteralPath "Env:$taskExperimentalFlag" -ErrorAction SilentlyContinue
}
foreach ($taskExperimentalFlag in @('QWEN_DIAG_WAVE_PREFIX_CONTROL_MODEL','GGML_VK_DIAG_CONFIG_FILE','GGML_VK_DIAG_TRACE','GGML_VK_PERF_LOGGER','GGML_VK_PERF_LOGGER_CONCURRENT','QWEN_EXPANDED_VK_CONFIG_FILE','GGML_VK_DIAG_REUSE_MODE_FILE','GGML_VK_DIAG_REUSE_TRACE')) {
    Remove-Item -LiteralPath "Env:$taskExperimentalFlag" -ErrorAction SilentlyContinue
}
if (-not $DisableStateOnlyDecoder) { $env:QWEN_OPT_FLAGS += ',state-only' }
if (-not $DisableCompactSampler) { $env:QWEN_OPT_FLAGS += ',compact-sampler' }
if (-not $KeepOnnxSpinning) { $env:QWEN_ONNX_NO_SPIN = '1' }
if (-not $DisablePredictorHeadRange) {
    $taskHeadRuntime = Join-Path $PSScriptRoot 'runtime/predictor-head-b7885-v2'
    $taskHeadManifest = Join-Path $taskHeadRuntime 'backend-manifest.json'
    if (-not (Test-Path -LiteralPath $taskHeadManifest)) {
        throw 'Validated predictor backend is missing. Restore runtime/predictor-head-b7885-v2 or use -DisablePredictorHeadRange.'
    }
    $taskHeadInfo = Get-Content -LiteralPath $taskHeadManifest -Raw | ConvertFrom-Json
    foreach ($taskBackendFile in $taskHeadInfo.files) {
        if ([System.IO.Path]::GetFileName($taskBackendFile.name) -ne $taskBackendFile.name) { throw 'Invalid backend manifest path' }
        $taskBackendHash = (Get-FileHash -LiteralPath (Join-Path $taskHeadRuntime $taskBackendFile.name) -Algorithm SHA256).Hash
        if ($taskBackendHash -ne $taskBackendFile.sha256) { throw 'Predictor backend checksum mismatch; use -DisablePredictorHeadRange to restore the original backend.' }
    }
    $env:QWEN_LLAMA_RUNTIME_DIR = $taskHeadRuntime
    $env:QWEN_PREDICTOR_HEAD_RANGE = '1'
    $env:QWEN_PREDICTOR_HEAD_MODEL_DIR = $baseRoot
    Write-Host 'Predictor output range enabled for the verified Base model directory (currently speaker 39).'
}
$taskOptimizedRelease = if ($PreviousOptimization) {
    Join-Path $PSScriptRoot 'runtime/releases/latency-2026-10-07'
} else {
    Join-Path $PSScriptRoot 'runtime/releases/latency-crop-2026-10-08'
}
if (-not $Baseline) {
    if ($DisableStateOnlyDecoder -or $DisablePredictorHeadRange) {
        throw 'The validated latency release requires state-only decoding and predictor output range. Use -Baseline for alternate settings.'
    }
    $taskReleaseInfo = Get-Content -LiteralPath (Join-Path $taskOptimizedRelease 'release-manifest.json') -Raw | ConvertFrom-Json
    foreach ($taskReleaseFile in $taskReleaseInfo.files) {
        if ([IO.Path]::GetFileName($taskReleaseFile.name) -ne $taskReleaseFile.name) { throw 'Invalid release manifest path' }
        if ((Get-FileHash -LiteralPath (Join-Path $taskOptimizedRelease $taskReleaseFile.name) -Algorithm SHA256).Hash -ne $taskReleaseFile.sha256) { throw 'Latency release checksum mismatch' }
    }
    $taskOptimizedBackend = Join-Path $taskOptimizedRelease 'backend'
    $taskOptimizedBackendInfo = Get-Content -LiteralPath (Join-Path $taskOptimizedBackend 'backend-manifest.json') -Raw | ConvertFrom-Json
    foreach ($taskBackendFile in $taskOptimizedBackendInfo.files) {
        if ([IO.Path]::GetFileName($taskBackendFile.name) -ne $taskBackendFile.name) { throw 'Invalid backend manifest path' }
        if ((Get-FileHash -LiteralPath (Join-Path $taskOptimizedBackend $taskBackendFile.name) -Algorithm SHA256).Hash -ne $taskBackendFile.sha256) { throw 'Latency backend checksum mismatch' }
    }
    $env:QWEN_LLAMA_RUNTIME_DIR = $taskOptimizedBackend
    $taskWaveModel = if ($taskReleaseInfo.wave_model) { $taskReleaseInfo.wave_model } else { 'decoder-prefix9-block0-v2.onnx' }
    if ([IO.Path]::GetFileName($taskWaveModel) -ne $taskWaveModel) { throw 'Invalid waveform model path' }
    $env:QWEN_DIAG_WAVE_PREFIX_MODEL = Join-Path $taskOptimizedRelease $taskWaveModel
    $env:QWEN_DIAG_WAVE_PREFIX_MODE_FILE = Join-Path $taskOptimizedRelease 'enabled.txt'
    $env:QWEN_DIAG_EOS_MODE_FILE = Join-Path $taskOptimizedRelease 'enabled.txt'
    $env:QWEN_DIAG_PROJECTION_MODE_FILE = Join-Path $taskOptimizedRelease 'enabled.txt'
    $env:GGML_VK_DIAG_CONFIG_FILE = Join-Path $taskOptimizedRelease 'vulkan-config.txt'
    Write-Host "Validated latency release $($taskReleaseInfo.version) enabled; use -Baseline or -PreviousOptimization to restore a preserved release."
}
if ($ProfileLatency) {
    $env:QWEN_TTFA_PROFILE = '1'
    $env:QWEN_DECODER_PROFILE = '1'
}
Write-Host "Starting Qwen3-TTS on $ListenAddress`:$Port (local: http://127.0.0.1:$Port)"
$releaseExe = if ($Baseline) {
    Join-Path $PSScriptRoot 'runtime/releases/baseline-2026-10-07/qwen3_tts_server.exe'
} else {
    Join-Path $taskOptimizedRelease 'qwen3_tts_server.exe'
}
if (-not (Test-Path -LiteralPath $releaseExe -PathType Leaf)) { throw 'Selected release executable is missing; restore the preserved release.' }
if ($Baseline -and (Get-FileHash -LiteralPath $releaseExe -Algorithm SHA256).Hash -ne 'E6B8A3DEA2B933B62DE76B0A926C1AC5C3EC372D298598483D4124FBA1164D99') {
    throw 'Preserved baseline executable checksum mismatch'
}
$debugExe = Join-Path $PSScriptRoot 'target\debug\qwen3_tts_server.exe'
if (Test-Path -LiteralPath $releaseExe -PathType Leaf) {
    & $releaseExe --quant $Quant --port $Port @baseArguments
} elseif (Test-Path -LiteralPath $debugExe -PathType Leaf) {
    & $debugExe --quant $Quant --port $Port @baseArguments
} else {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'Rust/Cargo is not installed or is not on PATH. Install it from https://rustup.rs/ and reopen PowerShell.'
    }
    & cargo run --release --bin qwen3_tts_server -- --quant $Quant --port $Port @baseArguments
}
if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }

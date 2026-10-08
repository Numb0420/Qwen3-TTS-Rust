<#
.SYNOPSIS
Create a speaker JSON from a reference WAV.
.EXAMPLE
.\create-speaker.ps1 -AudioPath .\speakers\40.wav -Name my-voice -Mode Xvec
.EXAMPLE
.\create-speaker.ps1 -AudioPath .\speakers\40.wav -Name my-voice-full -Mode Full -RefText '参考音频中实际说出的文字'
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$AudioPath,

    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [ValidateSet('Full', 'Xvec')]
    [string]$Mode,

    [string]$RefText,
    [string]$RefTextFile,
    [string]$Output,
    [string]$ModelDir = 'models/base-1.7b/onnx',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$audioFile = (Resolve-Path -LiteralPath $AudioPath -ErrorAction Stop).Path
$textFile = if ($RefTextFile) { (Resolve-Path -LiteralPath $RefTextFile -ErrorAction Stop).Path } else { $null }

if ($Mode -eq 'Full' -and -not $RefText -and -not $textFile) {
    throw 'Full mode requires -RefText or -RefTextFile with the exact words spoken in the WAV.'
}
if ($Mode -eq 'Xvec' -and ($RefText -or $textFile)) {
    throw 'Xvec mode uses only the speaker embedding; omit -RefText and -RefTextFile.'
}
if ($RefText -and $textFile) {
    throw 'Choose either -RefText or -RefTextFile.'
}

Push-Location -LiteralPath $PSScriptRoot
try {
$exe = Join-Path $PSScriptRoot 'target\release\qwen3_tts_extract.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'Extractor is missing and Cargo is not available. Build with: cargo build --release --bin qwen3_tts_extract'
    }
    if (Test-Path -LiteralPath '.cargo-home') {
        $env:CARGO_HOME = Join-Path $PSScriptRoot '.cargo-home'
        & cargo --offline --config 'source.crates-io.replace-with="local"' --config 'source.local.registry="sparse+http://127.0.0.1:38657/"' build --release --bin qwen3_tts_extract
    } else {
        & cargo build --release --bin qwen3_tts_extract
    }
    if ($LASTEXITCODE -ne 0) { throw "Extractor build failed with exit code $LASTEXITCODE" }
}

$modelRoot = (Resolve-Path -LiteralPath $ModelDir -ErrorAction Stop).Path
$arguments = @('--audio', $audioFile, '--name', $Name, '--mode', $Mode.ToLowerInvariant(), '--model-dir', $modelRoot)
if ($RefText) { $arguments += @('--ref-text', $RefText) }
if ($textFile) { $arguments += @('--ref-text-file', $textFile) }
if ($Output) { $arguments += @('--output', $Output) }
if ($Force) { $arguments += '--force' }

& $exe @arguments
if ($LASTEXITCODE -ne 0) { throw "Speaker extraction failed with exit code $LASTEXITCODE" }
Write-Host 'Speaker JSON saved. Validate it on Base, then add its name to start-webui.ps1 -BaseSpeakers and restart the server.'
} finally {
    Pop-Location
}

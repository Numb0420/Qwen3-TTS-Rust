param(
    [string]$VcRedistPath,
    [string]$PythonPath = 'python',
    [switch]$NoZip
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPackagedExe = Join-Path $PSScriptRoot 'dist/Qwen3-TTS-Portable/qwen3_tts_server.exe'
if (Get-Process qwen3_tts_server -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $taskPackagedExe }) {
    throw 'The portable service is running. Exit it from the system tray before packaging.'
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'Rust/Cargo is required on the packaging computer.' }
if (-not (Get-Command $PythonPath -ErrorAction SilentlyContinue)) { throw 'Python 3.11+ is required on the packaging computer. Use -PythonPath to select it.' }
if (-not $VcRedistPath) {
    $taskSearchRoots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    $taskCandidates = foreach ($taskSearchRoot in $taskSearchRoots) {
        Get-ChildItem -Path (Join-Path $taskSearchRoot 'Microsoft Visual Studio/*/*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT') -Directory -ErrorAction SilentlyContinue
    }
    $taskChosen = $taskCandidates | Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $taskChosen) { throw 'Cannot find x64 VC runtime redist. Use -VcRedistPath with the Microsoft.VC143.CRT directory.' }
    $VcRedistPath = $taskChosen.FullName
}
foreach ($taskDll in @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $VcRedistPath $taskDll) -PathType Leaf)) { throw "VC runtime incomplete: $taskDll" }
}
$taskPreviousCargoHome = $env:CARGO_HOME
$taskPreviousRustFlags = $env:RUSTFLAGS
try {
    $env:RUSTFLAGS = '-C target-feature=+crt-static'
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.cargo-home/registry')) {
        $env:CARGO_HOME = Join-Path $PSScriptRoot '.cargo-home'
        & cargo --config 'source.crates-io.replace-with="tts-local-cache"' --config 'source.tts-local-cache.registry="sparse+http://127.0.0.1:38657/"' build --manifest-path tools/portable-launcher/Cargo.toml --release --offline
    } else {
        & cargo build --manifest-path tools/portable-launcher/Cargo.toml --release
    }
    if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
} finally {
    if ($null -eq $taskPreviousCargoHome) { Remove-Item Env:CARGO_HOME -ErrorAction SilentlyContinue } else { $env:CARGO_HOME=$taskPreviousCargoHome }
    if ($null -eq $taskPreviousRustFlags) { Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue } else { $env:RUSTFLAGS=$taskPreviousRustFlags }
}
$taskPackArgs = @('tools/build-portable.py','--vc-redist',$VcRedistPath)
if (-not $NoZip) { $taskPackArgs += '--zip' }
& $PythonPath @taskPackArgs
if ($LASTEXITCODE -ne 0) { throw 'Portable packaging failed.' }
Write-Host 'Done: dist/Qwen3-TTS-Portable; ZIP: dist/Qwen3-TTS-Portable-win-x64.zip'

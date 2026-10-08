"""Developer packaging tool. The distributed application does not need Python."""
from pathlib import Path
import argparse, hashlib, json, shutil, time, tomllib, zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--vc-redist', type=Path, required=True)
parser.add_argument('--zip', action='store_true')
args = parser.parse_args()
destination = root / 'dist/Qwen3-TTS-Portable'
destination.mkdir(parents=True, exist_ok=True)
release = root / 'runtime/releases/latency-crop-2026-10-08'

def copy(source, relative):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

copy(root / 'tools/portable-launcher/target/release/tts-portable-launcher.exe', '启动TTS.exe')
copy(release / 'qwen3_tts_server.exe', 'qwen3_tts_server.exe')
for path in (root / 'models/base-1.7b').rglob('*'):
    if path.is_file(): copy(path, path.relative_to(root))
for path in (release / 'backend').glob('*.dll'):
    if path.name.startswith('ggml-cpu-') or path.name in {
        'ggml.dll', 'ggml-base.dll', 'ggml-vulkan.dll', 'llama.dll',
        'libomp.dll', 'libomp140.x86_64.dll', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll'
    }: copy(path, Path('runtime') / path.name)
for name in ['decoder-stage3-end1292.onnx', 'enabled.txt', 'vulkan-config.txt']:
    copy(release / name, Path('runtime') / name)
for path in args.vc_redist.glob('*.dll'):
    copy(path, path.name)
    copy(path, Path('runtime') / path.name)
source_config = json.loads((root / 'server-config.json').read_text(encoding='utf-8-sig'))
speakers = list(source_config.get('speakers', ['39', 'xiaolu', 'xiaoyan']))
registry = root / 'speakers/custom-speakers.txt'
registered = registry.read_text(encoding='utf-8-sig').splitlines() if registry.exists() else []
for speaker in registered:
    if speaker.strip() and speaker.strip() not in speakers: speakers.append(speaker.strip())
for speaker in speakers:
    if not isinstance(speaker, str) or not speaker or not all(c.isascii() and (c.isalnum() or c in '-_') for c in speaker):
        raise ValueError('Invalid speaker name in configuration')
    copy(root / f'speakers/{speaker}.json', f'speakers/{speaker}.json')
    if (root / f'speakers/{speaker}.wav').exists(): copy(root / f'speakers/{speaker}.wav', f'speakers/{speaker}.wav')
copy(root / 'target/release/qwen3_tts_extract.exe', 'tools/qwen3_tts_extract.exe')
for path in (root / 'tools/portable-licenses').iterdir():
    if path.is_file(): copy(path, Path('licenses') / path.name)
# Preserve dependency license texts from the local source cache.
locked = tomllib.loads((root / 'Cargo.lock').read_text(encoding='utf-8'))['package']
locked += tomllib.loads((root / 'tools/portable-launcher/Cargo.lock').read_text(encoding='utf-8'))['package']
for package in locked:
    for source in (root / '.cargo-home/registry/src').glob(f"*/{package['name']}-{package['version']}"):
        for path in source.iterdir():
            if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
                copy(path, Path('licenses/rust-dependencies') / source.name / path.name)
config = {'inference_device': source_config.get('inference_device', 'vulkan'), 'host': source_config.get('host', '0.0.0.0'), 'port': source_config.get('port', 3000),
          'speakers': speakers, 'open_browser': source_config.get('open_browser', True)}
(destination / 'server-config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(destination / 'speakers/custom-speakers.txt').write_text('\n'.join(registered), encoding='utf-8')
(destination / '使用说明.txt').write_text('''Qwen3-TTS Windows x64 绿色版

1. 完整解压到有写入权限的文件夹，建议本地 SSD。不要在压缩包里直接运行。
2. 双击“启动TTS.exe”。等待模型加载，浏览器将自动打开 http://127.0.0.1:3000/。
3. 启动后服务在系统托盘中运行；双击托盘图标打开网页，右键菜单可打开配置、日志目录，或“退出并停止服务”。Windows 可能将图标折叠在任务栏的小箭头内。
4. 网页支持普通合成、流式播放和上传自定义说话人。新说话人生成后，右键托盘“退出并停止服务”，再双击启动TTS.exe，刷新网页。

本包已包含 1.7B Base Q5 模型、39/xiaolu/xiaoyan 说话人、ONNX 和 llama/Vulkan/CPU 后端，以及应用本地 VC++ 运行库。
不需要安装 Rust、Python、Node.js、PowerShell 7 或 Visual Studio，不需要联网下载模型。
目标系统：Windows 10/11 64 位。当前包仅在本机验证，其他硬件尚需实际测试。
Vulkan 加速需要可用的显卡驱动，显卡驱动不包含在本包中。
默认使用 Vulkan；使用 CPU 时，将 server-config.json 中 inference_device 改为 cpu，再重启。
建议至少 16 GB 内存；速度取决于硬件，首次加载和第一次合成通常较慢。

配置文件 server-config.json：
- inference_device：vulkan 或 cpu；声码器一直使用 CPU。
- host：监听 IP，默认 0.0.0.0。
- port：默认 3000。
- speakers：初始加载的完整 ICL 克隆说话人。
- open_browser：是否启动后自动打开网页。
网页上传创建的说话人会自动登记在 speakers/custom-speakers.txt，重启后加载。

接口：
普通合成 POST /api/tts；流式 ws://127.0.0.1:3000/api/tts/stream。
发送 JSON 请求后发送纯文本 end；接收 24 kHz 单声道 float32 小端 PCM。
segment_done 是单段结束，done 是全部发送结束。停止时关闭 WebSocket 并停止客户端播放器。
详细 API 使用说明见网页。

日志：logs/server.stdout.log、logs/server.stderr.log，重启时覆盖；启动器记录位于 logs/launcher.log；需要保留旧日志请先复制。
端口占用时请关闭旧服务或修改 port。不要同时启动两个相同端口的实例。
局域网访问需网络可达且防火墙允许服务端口。局域网 HTTP 页面会使用浏览器兼容播放器。

本包模型和推理引擎与当前验收版本相同。基线、旧实验模型及编译缓存没有打包。
许可证和组件来源见 licenses 与 THIRD-PARTY-NOTICES.txt。
package-manifest.json 是初始分发文件的校验清单；上传的说话人、修改后的配置及运行日志不在初始校验范围内。
''', encoding='utf-8-sig')
(destination / 'THIRD-PARTY-NOTICES.txt').write_text('''Components:
- Qwen3-TTS / Qwen3-TTS-12Hz-1.7B-Base: https://github.com/QwenLM/Qwen3-TTS and https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0)
- Rust project: https://github.com/cgisky1980/Qwen3-TTS-Rust ; local Cargo.toml declares MIT OR Apache-2.0. This distribution uses Apache-2.0 terms; the Apache license text is included.
- llama.cpp / ggml: https://github.com/ggml-org/llama.cpp (MIT), bundled modified backend commit and optimizations are recorded in build-provenance.json.
- ONNX Runtime: https://github.com/microsoft/onnxruntime (MIT).
- LLVM OpenMP: https://github.com/llvm/llvm-project/tree/main/openmp (LLVM license text included).
- Microsoft Visual C++ runtime: application-local DLLs copied from the installed Visual Studio VC Redist x64 directory; Microsoft-VC-Redist-list.txt is included.
- Rust dependency license texts from the local Cargo source cache are under licenses/rust-dependencies.
Voice references 39, xiaolu and xiaoyan were supplied by the project owner.
''', encoding='utf-8')
(destination / 'build-provenance.json').write_text(json.dumps({
    'release': json.loads((release / 'release-manifest.json').read_text(encoding='utf-8-sig')),
    'backend': json.loads((release / 'backend/backend-manifest.json').read_text(encoding='utf-8-sig')),
    'packaged': time.strftime('%Y-%m-%d %H:%M:%S'),
    'vc_redist_source': str(args.vc_redist),
    'notes': 'Same server EXE and models as current release; native system tray launcher with kill-on-close job, static CRT.'
}, ensure_ascii=False, indent=2), encoding='utf-8')
manifest=[]
for path in sorted(destination.rglob('*')):
    if not path.is_file() or path.name=='package-manifest.json' or 'logs' in path.relative_to(destination).parts: continue
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024**2), b''): digest.update(chunk)
    manifest.append({'path':path.relative_to(destination).as_posix(),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
(destination / 'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Package ready: {destination}; {sum(x["bytes"] for x in manifest)/1024**3:.3f} GiB; {len(manifest)} files', flush=True)
if args.zip:
    archive=root/'dist/Qwen3-TTS-Portable-win-x64.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True,strict_timestamps=False) as output:
        for path in sorted(destination.rglob('*')):
            if path.is_file() and 'logs' not in path.relative_to(destination).parts:
                output.write(path,path.relative_to(destination.parent).as_posix())
    archive_hash=hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024**2),b''): archive_hash.update(chunk)
    archive.with_suffix('.zip.sha256').write_text(f'{archive_hash.hexdigest()}  {archive.name}\n',encoding='utf-8')
    print(f'ZIP ready: {archive}; {archive.stat().st_size/1024**3:.3f} GiB',flush=True)

import fs from 'node:fs';

const speaker = process.argv[2] || 'vivian';
const phrase = process.argv[3] || '你好，今天天气很好。';
const maxSteps = Number(process.argv[4] || 96);
const ws = new WebSocket('ws://127.0.0.1:3000/api/tts/stream');
ws.binaryType = 'arraybuffer';
const chunks = [];
let segmentDone = 0;
let completed = false;
const timer = setTimeout(() => {
  console.error(`timeout speaker=${speaker} chunks=${chunks.length} segmentDone=${segmentDone}`);
  ws.close();
  process.exit(1);
}, 120000);

ws.onopen = () => {
  ws.send(JSON.stringify({ text: phrase, speaker, max_steps: maxSteps, seed: 42 }));
  ws.send('end');
};
ws.onmessage = event => {
  if (event.data instanceof ArrayBuffer) {
    chunks.push(Buffer.from(event.data));
  } else if (event.data === 'segment_done') {
    segmentDone++;
  } else if (event.data === 'done') {
    completed = true;
    clearTimeout(timer);
    const pcm = Buffer.concat(chunks);
    const samples = pcm.length / 4;
    const wav = Buffer.alloc(44 + samples * 2);
    wav.write('RIFF', 0);
    wav.writeUInt32LE(wav.length - 8, 4);
    wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16);
    wav.writeUInt16LE(1, 20);
    wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(24000, 24);
    wav.writeUInt32LE(48000, 28);
    wav.writeUInt16LE(2, 32);
    wav.writeUInt16LE(16, 34);
    wav.write('data', 36);
    wav.writeUInt32LE(samples * 2, 40);
    for (let i = 0; i < samples; i++) {
      const value = Math.max(-1, Math.min(1, pcm.readFloatLE(i * 4)));
      wav.writeInt16LE(Math.round(value * 32767), 44 + i * 2);
    }
    fs.writeFileSync(`stream-${speaker}.wav`, wav);
    console.log(JSON.stringify({ speaker, chunks: chunks.length, segmentDone, samples, durationSeconds: samples / 24000 }));
    ws.close();
  } else {
    console.error(`unexpected message: ${event.data}`);
  }
};
ws.onerror = error => {
  if (completed) return;
  clearTimeout(timer);
  console.error(error);
  process.exit(1);
};

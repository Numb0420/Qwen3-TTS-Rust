// Audio Worklet Processor for real-time PCM playback
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 使用环形缓冲区，更高效
        this.bufferSize = 24000 * 10; // 10秒缓冲
        this.buffer = new Float32Array(this.bufferSize);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.available = 0;
        
        // 缓冲区状态
        this.totalReceived = 0;
        this.totalPlayed = 0;
        this.isDraining = false;
        this.firstAudioReported = false;
        
        this.port.onmessage = (event) => {
            const data = event.data;
            if (data instanceof Float32Array) {
                // Generation can outrun playback for long text. Grow the queue
                // instead of silently discarding audio after ten seconds.
                if (this.available + data.length > this.bufferSize) {
                    let newSize = this.bufferSize;
                    while (newSize < this.available + data.length) newSize *= 2;
                    const grown = new Float32Array(newSize);
                    for (let i = 0; i < this.available; i++) {
                        grown[i] = this.buffer[(this.readIndex + i) % this.bufferSize];
                    }
                    this.buffer = grown;
                    this.bufferSize = newSize;
                    this.readIndex = 0;
                    this.writeIndex = this.available;
                }
                // Preserve the engine's Float32 PCM samples exactly. Packet
                // boundaries are transport details, not audio edit points.
                for (let i = 0; i < data.length; i++) {
                    this.buffer[this.writeIndex] = data[i];
                    this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
                    this.available++;
                }
                this.totalReceived += data.length;
            } else if (data === 'drain') {
                // 标记为正在排空
                this.isDraining = true;
            }
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        for (let i = 0; i < channel.length; i++) {
            if (this.available > 0) {
                channel[i] = this.buffer[this.readIndex];
                this.readIndex = (this.readIndex + 1) % this.bufferSize;
                this.available--;
                this.totalPlayed++;
                if (!this.firstAudioReported && Math.abs(channel[i]) > 0.0001) {
                    this.firstAudioReported = true;
                    this.port.postMessage({ type: 'first_audio_played' });
                }
            } else {
                channel[i] = 0;
            }
        }

        // 如果正在排空且缓冲区为空，通知主线程
        if (this.isDraining && this.available === 0) {
            this.port.postMessage({ type: 'buffer_drained', played: this.totalPlayed, received: this.totalReceived });
            this.isDraining = false;
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);

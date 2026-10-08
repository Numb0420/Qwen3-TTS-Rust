//! Optional post-synthesis PCM processing for the server.
//! This module has no dependency on the TTS engine and can be removed independently.

const FRAME: usize = 960; // 40 ms at 24 kHz
const HOP: usize = FRAME / 4; // 75% overlap softens frame joins at high speed
const OVERLAP: usize = FRAME - HOP;
const SEARCH: usize = 120; // 5 ms

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    volume: u8,
    speed: u8,
}

impl Settings {
    pub fn new(volume: Option<i32>, speed: Option<i32>) -> Result<Self, String> {
        fn check(name: &str, value: Option<i32>) -> Result<u8, String> {
            let value = value.unwrap_or(50);
            if !(0..=100).contains(&value) {
                return Err(format!("{} must be an integer from 0 to 100", name));
            }
            Ok(value as u8)
        }
        Ok(Self {
            volume: check("volume", volume)?,
            speed: check("speed", speed)?,
        })
    }

    pub fn is_default(self) -> bool {
        self.volume == 50 && self.speed == 50
    }

    fn gain(self) -> f32 {
        self.volume as f32 / 50.0
    }

    fn rate(self) -> f64 {
        0.5 + self.speed as f64 / 100.0
    }
}

pub fn process_complete(samples: &mut Vec<f32>, settings: Settings) {
    if settings.is_default() {
        return;
    }
    let mut processor = Processor::new(settings);
    let mut output = processor.push(samples);
    output.extend(processor.finish());
    *samples = output;
}

/// One instance per stream segment. `push` returns ready PCM; `finish` drains its tail.
pub struct Processor {
    settings: Settings,
    stretch: Option<Wsola>,
}

impl Processor {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            stretch: (settings.speed != 50).then(|| Wsola::new(settings.rate())),
        }
    }

    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = match &mut self.stretch {
            Some(stretch) => stretch.push(input),
            None => input.to_vec(),
        };
        self.apply_gain(&mut output);
        output
    }

    pub fn finish(&mut self) -> Vec<f32> {
        let mut output = self.stretch.as_mut().map_or_else(Vec::new, Wsola::finish);
        self.apply_gain(&mut output);
        output
    }

    fn apply_gain(&self, samples: &mut [f32]) {
        if self.settings.volume == 50 {
            return;
        }
        let gain = self.settings.gain();
        for sample in samples {
            *sample = (*sample * gain).clamp(-1.0, 1.0);
        }
    }
}

// Waveform similarity overlap-add: advance through the source at the requested
// rate, align nearby frames by waveform similarity, and crossfade the overlap.
// Unlike resampling, this changes duration without systematically shifting pitch.
struct Wsola {
    rate: f64,
    input: Vec<f32>,
    pending_sum: Vec<f32>,
    pending_weight: Vec<f32>,
    window: Vec<f32>,
    nominal: f64,
    emitted: usize,
    started: bool,
}

impl Wsola {
    fn new(rate: f64) -> Self {
        let window = (0..FRAME)
            .map(|i| {
                let phase = std::f32::consts::TAU * (i as f32 + 0.5) / FRAME as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        Self {
            rate,
            input: Vec::new(),
            pending_sum: vec![0.0; FRAME],
            pending_weight: vec![0.0; FRAME],
            window,
            nominal: HOP as f64 * rate,
            emitted: 0,
            started: false,
        }
    }

    fn push(&mut self, input: &[f32]) -> Vec<f32> {
        self.input.extend_from_slice(input);
        let mut output = Vec::new();
        if !self.started && self.input.len() >= FRAME {
            self.add_frame(0);
            self.emit_hop(&mut output);
            self.started = true;
        }
        while self.started
            && (self.nominal.ceil() as usize).saturating_add(SEARCH + FRAME) <= self.input.len()
        {
            self.advance(&mut output);
        }
        output
    }

    fn finish(&mut self) -> Vec<f32> {
        let target = (self.input.len() as f64 / self.rate).round() as usize;
        if !self.started {
            // A sub-40 ms segment cannot provide a complete WSOLA frame.
            return (0..target)
                .map(|i| {
                    let index =
                        ((i as f64 * self.rate) as usize).min(self.input.len().saturating_sub(1));
                    self.input.get(index).copied().unwrap_or(0.0)
                })
                .collect();
        }
        let mut output = Vec::new();
        while self.emitted + HOP < target {
            self.advance(&mut output);
        }
        let remaining = target.saturating_sub(self.emitted);
        for i in 0..remaining.min(HOP) {
            output.push(self.pending_sum[i] / self.pending_weight[i].max(1e-8));
        }
        output
    }

    fn add_frame(&mut self, start: usize) {
        for i in 0..FRAME {
            let weight = self.window[i];
            self.pending_sum[i] += self.input[start + i] * weight;
            self.pending_weight[i] += weight;
        }
    }

    fn emit_hop(&mut self, output: &mut Vec<f32>) {
        for i in 0..HOP {
            output.push(self.pending_sum[i] / self.pending_weight[i].max(1e-8));
        }
        self.pending_sum.copy_within(HOP..FRAME, 0);
        self.pending_sum[OVERLAP..].fill(0.0);
        self.pending_weight.copy_within(HOP..FRAME, 0);
        self.pending_weight[OVERLAP..].fill(0.0);
        self.emitted += HOP;
    }

    fn advance(&mut self, output: &mut Vec<f32>) {
        let max_start = self.input.len() - FRAME;
        let center = (self.nominal.round() as usize).min(max_start);
        let begin = center.saturating_sub(SEARCH);
        let end = center.saturating_add(SEARCH).min(max_start);
        let tail_energy: f32 = (0..OVERLAP)
            .map(|i| {
                let value = self.pending_sum[i] / self.pending_weight[i].max(1e-8);
                value * value
            })
            .sum();
        let mut best = center;
        let mut best_score = f32::NEG_INFINITY;
        // A 4-sample coarse search is enough to locate the pitch period. Refine
        // around the winner so chunk boundaries cannot alter the chosen frame.
        for candidate in (begin..=end).step_by(4).chain(std::iter::once(center)) {
            let score = self.score(candidate, tail_energy, center);
            if score > best_score {
                best_score = score;
                best = candidate;
            }
        }
        for candidate in best.saturating_sub(3).max(begin)..=best.saturating_add(3).min(end) {
            let score = self.score(candidate, tail_energy, center);
            if score > best_score {
                best_score = score;
                best = candidate;
            }
        }
        self.add_frame(best);
        self.emit_hop(output);
        self.nominal += HOP as f64 * self.rate;
    }

    fn score(&self, start: usize, tail_energy: f32, center: usize) -> f32 {
        let candidate = &self.input[start..start + OVERLAP];
        let mut dot = 0.0;
        let mut energy = 0.0;
        for (i, &b) in candidate.iter().enumerate() {
            let a = self.pending_sum[i] / self.pending_weight[i].max(1e-8);
            dot += a * b;
            energy += b * b;
        }
        if tail_energy < 1e-8 || energy < 1e-8 {
            return -(start.abs_diff(center) as f32);
        }
        // In noisy or unvoiced spans, correlation is weak. Prefer the nominal
        // position there instead of following random waveform matches.
        dot / (tail_energy * energy).sqrt() - 0.05 * start.abs_diff(center) as f32 / SEARCH as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_exact_bypass() {
        let mut samples = vec![0.1, -0.2, 0.9];
        process_complete(&mut samples, Settings::new(None, None).unwrap());
        assert_eq!(samples, vec![0.1, -0.2, 0.9]);
    }

    #[test]
    fn volume_changes_amplitude_and_speed_changes_duration() {
        let source: Vec<f32> = (0..24_000)
            .map(|i| (i as f32 * std::f32::consts::TAU * 220.0 / 24_000.0).sin() * 0.2)
            .collect();
        let mut louder = source.clone();
        process_complete(&mut louder, Settings::new(Some(100), None).unwrap());
        assert_eq!(louder.len(), source.len());
        assert!((louder[100] - source[100] * 2.0).abs() < 1e-6);

        for (speed, expected) in [(0, 48_000), (100, 16_000)] {
            let mut output = source.clone();
            process_complete(&mut output, Settings::new(None, Some(speed)).unwrap());
            assert_eq!(output.len(), expected);
            let largest_step = output
                .windows(2)
                .map(|pair| (pair[1] - pair[0]).abs())
                .fold(0.0_f32, f32::max);
            assert!(largest_step < 0.04, "speed={speed}, step={largest_step}");
            // The central voiced section should retain roughly the source pitch.
            let offset = expected / 3;
            let crossings = output[offset..offset + 2_400]
                .windows(2)
                .filter(|v| v[0] <= 0.0 && v[1] > 0.0)
                .count();
            assert!(
                (18..=26).contains(&crossings),
                "speed={speed}, crossings={crossings}"
            );
        }
    }

    #[test]
    fn streaming_matches_complete_processing() {
        let source: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.07).sin() * 0.2).collect();
        let settings = Settings::new(Some(80), Some(65)).unwrap();
        let mut complete = source.clone();
        process_complete(&mut complete, settings);
        let mut processor = Processor::new(settings);
        let mut streamed = Vec::new();
        for chunk in source.chunks(1777) {
            streamed.extend(processor.push(chunk));
        }
        streamed.extend(processor.finish());
        assert_eq!(streamed, complete);
    }
}

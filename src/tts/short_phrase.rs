//! Guard against an x-vector prompt continuing far beyond a short Chinese phrase.
//! The audio gate works as samples arrive, so streaming does not wait for a
//! client-side buffer or for the whole utterance to finish.

const WINDOW: usize = 480; // 20 ms at 24 kHz
const VOICE_RMS: f32 = 0.003;
const MAX_TRAILING_SILENCE_WINDOWS: usize = 8; // 160 ms
const STOP_SILENCE_WINDOWS: usize = 12; // 240 ms
pub const NATURAL_PACE_INSTRUCTION: &str = "用自然的语速说话。";

#[derive(Clone, Copy)]
pub struct ShortPhrasePolicy {
    pub hard_limit: usize,
}

impl ShortPhrasePolicy {
    pub fn for_text(text: &str) -> Option<Self> {
        let letters: Vec<char> = text.chars().filter(|c| c.is_alphanumeric()).collect();
        if letters.is_empty()
            || letters.len() > 6
            || !letters
                .iter()
                .all(|&c| matches!(c as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff))
        {
            return None;
        }
        Some(Self {
            hard_limit: 44 + 4 * letters.len(),
        })
    }
}

#[derive(Default)]
pub struct ShortPhraseAudio {
    started: bool,
    silence_windows: usize,
    done: bool,
}

impl ShortPhraseAudio {
    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn filter(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(samples.len());
        for window in samples.chunks(WINDOW) {
            if self.done {
                break;
            }
            let rms = (window.iter().map(|v| v * v).sum::<f32>() / window.len() as f32).sqrt();
            if rms >= VOICE_RMS {
                self.started = true;
                self.silence_windows = 0;
                output.extend_from_slice(window);
            } else if self.started {
                self.silence_windows += 1;
                if self.silence_windows <= MAX_TRAILING_SILENCE_WINDOWS {
                    output.extend_from_slice(window);
                }
                if self.silence_windows >= STOP_SILENCE_WINDOWS {
                    self.done = true;
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_short_chinese_phrases_are_limited() {
        assert_eq!(ShortPhrasePolicy::for_text("你好").unwrap().hard_limit, 52);
        assert_eq!(
            ShortPhrasePolicy::for_text("已完成。").unwrap().hard_limit,
            56
        );
        assert!(ShortPhrasePolicy::for_text("今天我们开始讲解语音合成").is_none());
        assert!(ShortPhrasePolicy::for_text("AI 语音").is_none());
    }

    #[test]
    fn removes_long_initial_and_final_silence_without_changing_speech() {
        let mut gate = ShortPhraseAudio::default();
        assert!(gate.filter(&vec![0.0; 9_600]).is_empty());
        assert_eq!(gate.filter(&vec![0.1; 4_800]), vec![0.1; 4_800]);
        assert_eq!(gate.filter(&vec![0.0; 9_600]).len(), 3_840);
        assert!(gate.is_done());
        assert!(gate.filter(&vec![0.1; 4_800]).is_empty());
    }
}

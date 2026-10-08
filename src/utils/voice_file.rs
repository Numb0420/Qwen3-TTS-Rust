use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceFile {
    /// Reference text corresponding to the audio features
    #[serde(default)]
    pub ref_text: String,
    /// Extracted semantic codes from the Audio Encoder
    #[serde(default)]
    pub audio_codes: Vec<i64>,
    /// Speaker embedding vector from the Speaker Encoder
    #[serde(alias = "spk_emb")]
    pub speaker_embedding: Vec<f32>,
    /// Built-in speaker token, when the voice file comes from a preset.
    #[serde(default)]
    pub spk_id: Option<usize>,

    // Metadata
    pub name: Option<String>,
    pub gender: Option<String>,
    pub age: Option<String>,
    pub description: Option<String>,
}

impl VoiceFile {
    pub fn new(ref_text: String, audio_codes: Vec<i64>, speaker_embedding: Vec<f32>) -> Self {
        Self {
            ref_text,
            audio_codes,
            speaker_embedding,
            spk_id: None,
            name: None,
            gender: None,
            age: None,
            description: None,
        }
    }

    pub fn with_metadata(
        mut self,
        name: Option<String>,
        gender: Option<String>,
        age: Option<String>,
        description: Option<String>,
    ) -> Self {
        self.name = name;
        self.gender = gender;
        self.age = age;
        self.description = description;
        self
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| e.to_string())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(|e| e.to_string())
    }
}

//! Assets for Qwen3-TTS - loads numpy files and provides projections

use std::io::{Read, Seek};
use std::sync::{OnceLock, Mutex};
use std::collections::HashMap;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn projection_avx2(hidden: &[f32], bias: &[f32], packed: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;
    let mut output = vec![0.0; bias.len()];
    for group in (0..bias.len()).step_by(8) {
        let mut sum = _mm256_loadu_ps(bias.as_ptr().add(group));
        let weights = packed.as_ptr().add(group * hidden.len());
        for (input, value) in hidden.iter().enumerate() {
            let weight = _mm256_loadu_ps(weights.add(input * 8));
            let product = _mm256_mul_ps(_mm256_set1_ps(*value), weight);
            sum = _mm256_add_ps(sum, product);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(group), sum);
    }
    output
}

pub struct Assets {
    pub tts_pad: Vec<f32>,
    pub proj_weight: Vec<f32>,
    pub proj_bias: Vec<f32>,
    pub codec_embeddings: Vec<Vec<f32>>,
    pub text_table: Vec<f32>,
    codec_projection_cache: Vec<Vec<OnceLock<Vec<f32>>>>,
    prompt_parts: Mutex<HashMap<Vec<u32>, Vec<Vec<f32>>>>,
    diagnostic_projection_weights: OnceLock<Vec<f32>>,
}

impl Assets {
    pub fn load(model_dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading assets from: {}", model_dir.display());
        let gguf_path = model_dir.join("qwen3_assets.gguf");

        if gguf_path.exists() {
            println!("    Found GGUF assets: {}", gguf_path.display());
            return Self::load_gguf(&gguf_path);
        }

        println!("    GGUF not found, falling back to NPY...");
        // Fallback to legacy NPY loading
        Self::load_npy(model_dir)
    }

    fn load_gguf(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Custom minimal GGUF reader to avoid dependencies
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        // 1. Header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err("Not a GGUF file".into());
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version < 2 {
            return Err(format!("Unsupported GGUF version: {}", version).into());
        }

        let mut tensor_count_bytes = [0u8; 8];
        reader.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        let mut kv_count_bytes = [0u8; 8];
        reader.read_exact(&mut kv_count_bytes)?;
        let kv_count = u64::from_le_bytes(kv_count_bytes);

        // Sub-helper to read string
        let read_string = |r: &mut std::io::BufReader<std::fs::File>| -> Result<String, Box<dyn std::error::Error>> {
            let mut len_bytes = [0u8; 8];
            r.read_exact(&mut len_bytes)?;
            let len = u64::from_le_bytes(len_bytes) as usize;
            let mut bytes = vec![0u8; len];
            r.read_exact(&mut bytes)?;
            Ok(String::from_utf8(bytes)?)
        };

        // 2. Metadata KV (Skip)
        for _ in 0..kv_count {
            let _key = read_string(&mut reader)?;
            let mut type_bytes = [0u8; 4];
            reader.read_exact(&mut type_bytes)?;
            let val_type = u32::from_le_bytes(type_bytes);

            match val_type {
                0..=7 => {
                    // Simplified skipping logic
                    let size = match val_type {
                        0 | 1 | 7 => 1,
                        2 | 3 => 2,
                        4..=6 => 4,
                        _ => 0,
                    };
                    if size > 0 {
                        let mut b = vec![0u8; size];
                        reader.read_exact(&mut b)?;
                    }
                }
                8 => {
                    let _s = read_string(&mut reader)?;
                } // String
                10..=12 => {
                    let mut b = [0u8; 8];
                    reader.read_exact(&mut b)?;
                } // 64-bit
                9 => {
                    return Err(
                        "Array type in GGUF metadata not implemented in simple reader".into(),
                    );
                }
                _ => return Err(format!("Unknown GGUF value type: {}", val_type).into()),
            }
        }

        // 3. Tensors Info
        struct TensorInfo {
            name: String,
            offset: u64,
            shape: Vec<usize>,
            _type: u32,
        }
        let mut tensors = std::collections::HashMap::new();

        for _ in 0..tensor_count {
            let name = read_string(&mut reader)?;
            let mut ndim_b = [0u8; 4];
            reader.read_exact(&mut ndim_b)?;
            let n_dims = u32::from_le_bytes(ndim_b) as usize;

            let mut shape = Vec::new();
            for _ in 0..n_dims {
                let mut dim_b = [0u8; 8];
                reader.read_exact(&mut dim_b)?;
                shape.push(u64::from_le_bytes(dim_b) as usize);
            }

            let mut type_b = [0u8; 4];
            reader.read_exact(&mut type_b)?;
            let type_id = u32::from_le_bytes(type_b);

            let mut offset_b = [0u8; 8];
            reader.read_exact(&mut offset_b)?;
            let offset = u64::from_le_bytes(offset_b);

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    offset,
                    shape,
                    _type: type_id,
                },
            );
        }

        // Padding alignment
        let current_pos = reader.stream_position()?;
        let alignment = 32;
        let padding = (alignment - (current_pos % alignment)) % alignment;
        reader.seek(std::io::SeekFrom::Current(padding as i64))?;
        let data_start = reader.stream_position()?;

        let path_clone = path.to_path_buf();
        let read_tensor_data = |info: &TensorInfo| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let mut f = std::fs::File::open(&path_clone)?;
            f.seek(std::io::SeekFrom::Start(data_start + info.offset))?;

            let num_elems: usize = info.shape.iter().product();
            println!(
                "    Debug: Reading tensor '{}' shape={:?}, elems={}, bytes={}",
                info.name,
                info.shape,
                num_elems,
                num_elems * 4
            );
            if info._type != 0 {
                return Err(
                    format!("Unsupported tensor type: {} (expected F32)", info._type).into(),
                );
            }

            if num_elems * 4 != info.shape.iter().product::<usize>() * 4 {
                // Paranoia check
            }

            // Allocate Vec<f32> directly
            let mut floats = Vec::new();
            if let Err(e) = floats.try_reserve_exact(num_elems) {
                return Err(format!(
                    "Failed to reserve memory for tensor '{}' (f32): {}",
                    info.name, e
                )
                .into());
            }
            floats.resize(num_elems, 0.0);

            // Read directly into the f32 buffer as bytes
            // SAFETY: f32 and u8 have same layout size ratio (1:4).
            // We just treat the slice of f32s as a mutable slice of u8s.
            let byte_slice = unsafe {
                std::slice::from_raw_parts_mut(floats.as_mut_ptr() as *mut u8, num_elems * 4)
            };

            // Read in chunks
            let chunk_size = 100 * 1024 * 1024;
            let total_bytes = num_elems * 4;
            let mut current_read = 0;

            while current_read < total_bytes {
                let remaining = total_bytes - current_read;
                let to_read = remaining.min(chunk_size);

                f.read_exact(&mut byte_slice[current_read..current_read + to_read])?;
                current_read += to_read;
            }

            println!(
                "    Debug: Tensor '{}' read complete (optimized).",
                info.name
            );
            Ok(floats)
        };

        // Load specific tensors
        let proj_weight = read_tensor_data(
            tensors
                .get("proj.weight")
                .ok_or("proj.weight (tensor) missing")?,
        )?;
        println!("    Debug: proj.weight loaded.");

        let proj_bias = read_tensor_data(
            tensors
                .get("proj.bias")
                .ok_or("proj.bias (tensor) missing")?,
        )?;
        println!("    Debug: proj.bias loaded.");

        let text_table = if let Some(info) = tensors.get("text_embd") {
            let data = read_tensor_data(info)?;
            println!("    Debug: text_embd loaded.");
            data
        } else {
            println!("    WARNING: text_embd not found in GGUF!");
            Vec::new()
        };

        let mut codec_embeddings = Vec::new();
        for i in 0..16 {
            let name = format!("codec_embd.{}", i);
            if let Some(info) = tensors.get(&name) {
                codec_embeddings.push(read_tensor_data(info)?);
            }
        }
        println!("    Debug: codec_embeddings loaded.");

        let tts_pad = if text_table.len() >= (151671 + 1) * 2048 {
            let start = 151671 * 2048;
            text_table[start..start + 2048].to_vec()
        } else {
            vec![0.0; 2048]
        };
        println!("    Debug: tts_pad loaded.");

        println!(
            "    Loaded via GGUF (Custom Reader): ProjW={}, TextTbl={}, Codec={}",
            proj_weight.len(),
            text_table.len(),
            codec_embeddings.len()
        );

        Ok(Self {
            tts_pad,
            proj_weight,
            proj_bias,
            codec_projection_cache: Self::projection_slots(&codec_embeddings),
            prompt_parts: Mutex::new(HashMap::new()),
            diagnostic_projection_weights: OnceLock::new(),
            codec_embeddings,
            text_table,
        })
    }

    fn load_npy(model_dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let proj_weight = Self::load_npy_f32(&model_dir.join("proj_weight.npy"))?;
        let proj_bias = Self::load_npy_f32_1d(&model_dir.join("proj_bias.npy"))?;

        let text_table_path = model_dir.join("text_embedding_projected.npy");
        let text_table = if text_table_path.exists() {
            Self::load_npy_f32(&text_table_path)?
        } else {
            Vec::new()
        };

        let mut codec_embeddings = Vec::new();
        for i in 0..16 {
            let emb_path = model_dir.join(format!("codec_embedding_{}.npy", i));
            if emb_path.exists() {
                codec_embeddings.push(Self::load_npy_f32(&emb_path)?);
            }
        }

        let tts_pad = if text_table.len() >= (151671 + 1) * 2048 {
            let start = 151671 * 2048;
            text_table[start..start + 2048].to_vec()
        } else {
            vec![0.0; 2048]
        };

        Ok(Self {
            tts_pad,
            proj_weight,
            proj_bias,
            codec_projection_cache: Self::projection_slots(&codec_embeddings),
            prompt_parts: Mutex::new(HashMap::new()),
            diagnostic_projection_weights: OnceLock::new(),
            codec_embeddings,
            text_table,
        })
    }

    fn load_npy_f32(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 10];
        file.read_exact(&mut magic)?;

        if &magic[..6] != b"\x93NUMPY" {
            return Err("Not a numpy file".into());
        }

        let version = (magic[6], magic[7]);
        let header_len = match version {
            (1, _) => {
                file.seek(std::io::SeekFrom::Start(8))?;
                let mut len = [0u8; 2];
                file.read_exact(&mut len)?;
                u16::from_le_bytes(len) as usize
            }
            (2, _) => {
                file.seek(std::io::SeekFrom::Start(8))?;
                let mut len = [0u8; 4];
                file.read_exact(&mut len)?;
                u32::from_le_bytes(len) as usize
            }
            _ => return Err("Unsupported numpy version".into()),
        };

        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)?;

        let shape_start = header
            .windows(6)
            .position(|w| std::str::from_utf8(w).unwrap_or("").starts_with("shape"));

        let shape_str = if let Some(pos) = shape_start {
            let mut start = pos + 6;
            while start < header.len() && header[start] != b'(' {
                start += 1;
            }
            if start < header.len() {
                let mut end = start + 1;
                while end < header.len() && header[end] != b')' {
                    end += 1;
                }
                String::from_utf8_lossy(&header[start + 1..end]).to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let shape: Vec<usize> = shape_str
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .filter_map(|s| {
                s.trim()
                    .trim_matches(|c| c == '(' || c == ')' || c == '[' || c == ']')
                    .parse()
                    .ok()
            })
            .collect();

        let total_elements: usize = shape.iter().product();
        let elem_size = 4;

        let mut data = vec![0u8; total_elements * elem_size];
        file.read_exact(&mut data)?;

        let fdata: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(fdata)
    }

    fn load_npy_f32_1d(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Self::load_npy_f32(path)
    }

    pub fn project(&self, hidden: &[f32]) -> Vec<f32> {
        if std::env::var("QWEN_DIAG_PROJECTION_MODE_FILE").ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .is_some_and(|mode| mode.trim() == "1") {
            return self.project_diagnostic_avx2(hidden);
        }
        let n_in = hidden.len();
        let n_out = self.proj_bias.len();
        let mut result = vec![0.0; n_out];

        for (out_idx, res) in result.iter_mut().enumerate().take(n_out) {
            let mut sum = self.proj_bias[out_idx];
            for (in_idx, h) in hidden.iter().enumerate().take(n_in) {
                // PyTorch Linear weights are [out_features, in_features]
                // Correct index: out_idx * n_in + in_idx
                sum += h * self.proj_weight[out_idx * n_in + in_idx];
            }
            *res = sum;
        }

        result
    }

    /// Isolated bit-preserving experiment: each lane keeps its scalar input order.
    pub fn project_diagnostic_avx2(&self, hidden: &[f32]) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && self.proj_bias.len() % 8 == 0
            && self.proj_weight.len() == hidden.len() * self.proj_bias.len() {
            let packed = self.diagnostic_projection_weights.get_or_init(|| {
                let mut weights = Vec::with_capacity(self.proj_weight.len());
                for group in (0..self.proj_bias.len()).step_by(8) {
                    for input in 0..hidden.len() {
                        for lane in 0..8 {
                            weights.push(self.proj_weight[(group+lane)*hidden.len()+input]);
                        }
                    }
                }
                weights
            });
            // SAFETY: runtime feature check, validated sizes, and unaligned loads.
            return unsafe { projection_avx2(hidden, &self.proj_bias, packed) };
        }
        let mut result = self.proj_bias.clone();
        for (output, sum) in result.iter_mut().enumerate() {
            for (input, value) in hidden.iter().enumerate() {
                *sum += value * self.proj_weight[output * hidden.len() + input];
            }
        }
        result
    }

    pub fn project_embedding(&self, emb_2048: &[f32]) -> Vec<f32> {
        let n_in = 2048;
        let n_out = self.proj_bias.len();
        let mut result = vec![0.0; n_out];

        for (out_idx, res) in result.iter_mut().enumerate().take(n_out) {
            let mut sum = self.proj_bias[out_idx];
            for (in_idx, val) in emb_2048.iter().enumerate().take(n_in) {
                // PyTorch Linear weights are [out_features, in_features]
                // Correct index: out_idx * n_in + in_idx
                sum += val * self.proj_weight[out_idx * n_in + in_idx];
            }
            *res = sum;
        }

        result
    }

    pub fn get_codec_embedding(&self, q: usize, code: i32) -> Vec<f32> {
        if q < self.codec_embeddings.len() {
            let emb = &self.codec_embeddings[q];
            let code = code.max(0) as usize;
            let start = code * 2048;
            if start + 2048 <= emb.len() {
                return emb[start..start + 2048].to_vec();
            } else {
                eprintln!(
                    "WARNING: OOB Code Access: q={}, code={}, start={}, len={}",
                    q,
                    code,
                    start,
                    emb.len()
                );
            }
        }
        vec![0.0; 2048]
    }

    pub fn get_codec_embedding_1024(&self, q: usize, code: i32) -> Vec<f32> {
        if let Some(slot) = self.codec_projection_cache.get(q)
            .and_then(|table| table.get(code.max(0) as usize)) {
            return slot.get_or_init(|| {
                self.project_embedding(&self.get_codec_embedding(q, code))
            }).clone();
        }
        self.project_embedding(&self.get_codec_embedding(q, code))
    }

    fn projection_slots(embeddings: &[Vec<f32>]) -> Vec<Vec<OnceLock<Vec<f32>>>> {
        embeddings.iter().map(|table| {
            (0..table.len() / 2048).map(|_| OnceLock::new()).collect()
        }).collect()
    }

    /// Fill the fixed predictor inputs before accepting requests. Each entry
    /// uses the original scalar projection, with the same accumulation order.
    pub fn prewarm_codec_projections(&self) {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(1).min(4);
        // q15 is sampled last and is never fed back into the predictor.
        std::thread::scope(|scope| {
            for worker in 0..workers {
                scope.spawn(move || {
                    for q in 0..self.codec_embeddings.len().min(15) {
                        let count = self.codec_projection_cache[q].len().min(2048);
                        for code in (worker..count).step_by(workers) {
                            let slot = &self.codec_projection_cache[q][code];
                            slot.get_or_init(|| self.project_embedding(
                                &self.get_codec_embedding(q, code as i32)));
                        }
                    }
                });
            }
        });
    }

    /// CPU embedding cache only; it never reuses model KV across target texts.
    pub fn cached_prompt_part(&self, key: Vec<u32>, build: impl FnOnce() -> Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut cache = self.prompt_parts.lock().unwrap();
        cache.entry(key).or_insert_with(build).clone()
    }

    pub fn get_text_embedding(&self, token_id: usize) -> Vec<f32> {
        let start = token_id * 2048;
        if start + 2048 <= self.text_table.len() {
            self.text_table[start..start + 2048].to_vec()
        } else {
            // Fallback for OOB tokens (should not happen usually)
            self.get_text_embedding_fallback(token_id)
        }
    }

    pub fn get_text_embedding_fallback(&self, token_id: usize) -> Vec<f32> {
        let mut result = vec![0.0; 2048];
        for i in 0..2048.min(result.len()) {
            result[i] = ((token_id * 17 + i) as f32 % 2.0) - 1.0;
        }
        result
    }
}

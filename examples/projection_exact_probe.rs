use qwen3_tts::assets_manager::Assets;
use std::{path::Path, time::Instant, hint::black_box};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::remove_var("QWEN_DIAG_PROJECTION_MODE_FILE");
    let assets=Assets::load(Path::new("models/base-1.7b/gguf"))?;
    let mut state=43u64;
    let mut inputs=Vec::new();
    for _ in 0..128 {
        inputs.push((0..2048).map(|_| {
            state=state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state>>40) as i32-8388608) as f32 / 1048576.0
        }).collect::<Vec<_>>());
    }
    inputs.push(vec![0.0;2048]); inputs.push(vec![-0.0;2048]);
    for (i,input) in inputs.iter().enumerate() {
        let a=assets.project(input);let b=assets.project_diagnostic_avx2(input);
        if a.iter().zip(&b).any(|(a,b)|a.to_bits()!=b.to_bits()) { return Err(format!("projection bits differ at input {i}").into()); }
    }
    let mut rows=Vec::new();
    for round in 0..5 {
        let now=Instant::now();for input in &inputs {black_box(assets.project(black_box(input)));}let scalar=now.elapsed().as_secs_f64()*1000.0;
        let now=Instant::now();for input in &inputs {black_box(assets.project_diagnostic_avx2(black_box(input)));}let simd=now.elapsed().as_secs_f64()*1000.0;
        rows.push(serde_json::json!({"round":round,"scalar_ms":scalar,"simd_ms":simd,"inputs":inputs.len()}));
    }
    let report=serde_json::json!({"all_vectors_bitwise_equal":true,"input_count":inputs.len(),"rows":rows});
    std::fs::create_dir_all("diagnostics/projection-simd-2026-10-07")?;
    std::fs::write("diagnostics/projection-simd-2026-10-07/micro.json",serde_json::to_vec_pretty(&report)?)?;
    println!("{report}"); Ok(())
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GEMM isolation microbench at exact production shapes, for a design-target
//! comparison against other engines measured out of tree. NOT a regression gate, #[ignore]d.
//!
//! Run with:
//!   cargo test -p ffai-kernels-std --test prod_shape_gemm_isolation_bench \
//!     --release -- --ignored --nocapture
//!
//! Dense family: `ffai_qmm_mma` at M in {512, 2048, 4096} x N in {512, 8192},
//! K=2048, int4-affine group_size=64 (the attention/GDN projection shapes).
//!
//! MoE-gather family: `ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan` at
//! mTotal in {4096, 16384, 32768} rows (T=512/2048/4096 tokens, topk=8,
//! 256 experts, Zipf-skewed routing), gate/up (K=2048,N=512) and down
//! (K=512,N=2048) - the real per-expert widths from Qwen3.6-35B-A3B
//! (hidden=2048, moe_intermediate=512), matching
//! `smallm_occupancy_microbench.rs`'s established production fixture rather
//! than a flat K=2048 for both, since down-proj's real K is the intermediate
//! width, not hidden.

#![cfg(target_os = "macos")]

mod common;

use common::{Dt, gpu_lock};
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::{
    gemm::{
        quantized::ffai_qmm_mma,
        quantized_coop::ffai_qmm_coop,
        quantized_mpp::ffai_qmm_mma_mpp,
    },
    moe::moe_mpp_tileplan::{build_tile_plan, ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan},
};

fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    weights
        .chunks_exact(8)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xf) << (i * 4);
            }
            packed
        })
        .collect()
}

fn pack_f32(vals: &[f32], dt: Dt) -> Vec<u8> {
    match dt {
        Dt::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
        Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
    }
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) }
}

/// Same Zipf(s=1.0) skewed-routing model as `smallm_occupancy_microbench.rs`
/// - returns per-expert counts (sorted-ascending layout every gather-GEMM
/// kernel expects is implicit in `build_tile_plan`, which only needs counts).
fn skewed_counts(t: usize, topk: usize, n_experts: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64(seed);
    let weights: Vec<f64> = (0..n_experts).map(|e| 1.0 / (e as f64 + 1.0)).collect();
    let total_w: f64 = weights.iter().sum();
    let mut counts = vec![0usize; n_experts];
    for _ in 0..t {
        let mut chosen = [false; 1024];
        let mut picked = 0;
        while picked < topk {
            let r = rng.next_f64() * total_w;
            let mut acc = 0.0;
            let mut e_sel = n_experts - 1;
            for (e, w) in weights.iter().enumerate() {
                acc += w;
                if r <= acc {
                    e_sel = e;
                    break;
                }
            }
            if !chosen[e_sel] {
                chosen[e_sel] = true;
                counts[e_sel] += 1;
                picked += 1;
            }
        }
    }
    counts
}

fn gflops(flops: f64, us: f64) -> f64 { flops / us / 1e3 }

#[allow(clippy::too_many_arguments)]
fn time_qmm_mma(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
    let gs_per_row = k / group_size;
    let w_unpacked: Vec<u32> = (0..n * k).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let w_packed: Vec<u32> = w_unpacked.chunks_exact(k).flat_map(pack_int4_row).collect();
    let groups_total = n * gs_per_row;
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.0001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.0005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let mut buffers: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    buffers.insert("w".into(), w_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_f32(&scales, dt));
    buffers.insert("biases".into(), pack_f32(&biases, dt));
    buffers.insert("x".into(), pack_f32(&x, dt));
    buffers.insert("out".into(), pack_f32(&vec![0.0f32; m * n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    // ffai_qmm_mma requires M % 32 == 0 - Grid: [N/32, M/32, 1], TG: 128
    // threads (4 SG x 32 lanes), fixed regardless of M/N/K.
    assert_eq!(m % 32, 0, "ffai_qmm_mma requires M % 32 == 0");
    assert_eq!(n % 32, 0, "ffai_qmm_mma requires N % 32 == 0");
    let mut kernel = ffai_qmm_mma::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let grid = [n / 32, m / 32, 1];
    let tg = [128usize, 1, 1];

    let _ =
        ctx.dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg).expect("dispatch");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx
            .dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg)
            .expect("dispatch");
        total_us += r.elapsed_us;
    }
    total_us / iters as f64
}

/// Same buffer layout / dispatch shape as `time_qmm_mma`, but for the
/// already-landed MPP (`mpp::tensor_ops::matmul2d`, M5-class tensor-unit)
/// sibling `ffai_qmm_mma_mpp` - isolated ahead of writing a brand new
/// cooperative kernel, to see how much of the incumbent's ~23%-of-peak
/// plateau the existing tensor-unit path already closes on its own.
#[allow(clippy::too_many_arguments)]
fn time_qmm_mma_mpp(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
    let gs_per_row = k / group_size;
    let w_unpacked: Vec<u32> = (0..n * k).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let w_packed: Vec<u32> = w_unpacked.chunks_exact(k).flat_map(pack_int4_row).collect();
    let groups_total = n * gs_per_row;
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.0001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.0005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let mut buffers: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    buffers.insert("w".into(), w_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_f32(&scales, dt));
    buffers.insert("biases".into(), pack_f32(&biases, dt));
    buffers.insert("x".into(), pack_f32(&x, dt));
    buffers.insert("out".into(), pack_f32(&vec![0.0f32; m * n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    assert_eq!(m % 32, 0, "ffai_qmm_mma_mpp requires M % 32 == 0");
    assert_eq!(n % 32, 0, "ffai_qmm_mma_mpp requires N % 32 == 0");
    let mut kernel = ffai_qmm_mma_mpp::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let grid = [n / 32, m / 32, 1];
    let tg = [128usize, 1, 1];

    let _ =
        ctx.dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg).expect("dispatch");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx
            .dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg)
            .expect("dispatch");
        total_us += r.elapsed_us;
    }
    total_us / iters as f64
}

/// `ffai_qmm_coop` - BN=64 wide-N variant. Grid divides N by 64, not 32.
#[allow(clippy::too_many_arguments)]
fn time_qmm_coop(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
    let gs_per_row = k / group_size;
    let w_unpacked: Vec<u32> = (0..n * k).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let w_packed: Vec<u32> = w_unpacked.chunks_exact(k).flat_map(pack_int4_row).collect();
    let groups_total = n * gs_per_row;
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.0001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.0005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    let mut buffers: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    buffers.insert("w".into(), w_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_f32(&scales, dt));
    buffers.insert("biases".into(), pack_f32(&biases, dt));
    buffers.insert("x".into(), pack_f32(&x, dt));
    buffers.insert("out".into(), pack_f32(&vec![0.0f32; m * n], dt));
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    assert_eq!(m % 32, 0, "ffai_qmm_coop requires M % 32 == 0");
    assert_eq!(n % 64, 0, "ffai_qmm_coop requires N % 64 == 0");
    let mut kernel = ffai_qmm_coop::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let grid = [n / 64, m / 32, 1];
    let tg = [128usize, 1, 1];

    let _ =
        ctx.dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg).expect("dispatch");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx
            .dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg)
            .expect("dispatch");
        total_us += r.elapsed_us;
    }
    total_us / iters as f64
}

#[allow(clippy::too_many_arguments)]
fn time_moe_gather_tileplan(
    ctx: &Context,
    counts: &[usize],
    k_in: usize,
    n_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> (f64, usize) {
    let m_total: usize = counts.iter().sum();
    let (tile_expert, tile_row_start, tile_row_count) = build_tile_plan(counts);
    let num_tiles = tile_expert.len();
    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.0001 * (i as f32 * 0.03).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.0005 * (i as f32 * 0.07).cos()).collect();
    let x: Vec<f32> = (0..m_total * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();
    let x_rows: Vec<u32> = (0..m_total as u32).collect();

    let mut buffers: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    buffers.insert("x".into(), pack_f32(&x, dt));
    buffers.insert("w".into(), weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect());
    buffers.insert("scales".into(), pack_f32(&scales, dt));
    buffers.insert("biases".into(), pack_f32(&biases, dt));
    buffers
        .insert("tile_expert".into(), tile_expert.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert(
        "tile_row_start".into(),
        tile_row_start.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    buffers.insert(
        "tile_row_count".into(),
        tile_row_count.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    buffers.insert("x_rows".into(), x_rows.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert("out".into(), pack_f32(&vec![0.0f32; m_total * n_out], dt));
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let mut kernel = ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let grid = [n_out / 32, num_tiles.max(1), 1];
    let tg = [32usize, 1, 1];

    let _ =
        ctx.dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg).expect("dispatch");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx
            .dispatch_with_grid(&kernel, &buffers, &Default::default(), grid, tg)
            .expect("dispatch");
        total_us += r.elapsed_us;
    }
    (total_us / iters as f64, num_tiles)
}

#[ignore]
#[test]
fn isolation_dense_qmm_mma_prod_shapes() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let k = 2048usize;
    let group_size = 64usize;
    let dt = Dt::F16;

    eprintln!("\n=== ISOLATION: ffai_qmm_mma dense, K={k} GS={group_size} dtype={dt:?} ===");
    for &m in &[512usize, 2048, 4096] {
        for &n in &[512usize, 8192] {
            let iters = if m * n <= 512 * 8192 { 30 } else { 15 };
            let us = time_qmm_mma(&ctx, m, n, k, group_size, iters, dt);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let gf = gflops(flops, us);
            eprintln!(
                "  M={m:>5} N={n:>5} K={k:>5}: {us:>10.2} us  {gf:>9.1} GFLOP/s  pct57T={:>5.2}%",
                100.0 * gf / 57000.0
            );
        }
    }
}

#[ignore]
#[test]
fn isolation_dense_qmm_mma_mpp_prod_shapes() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let k = 2048usize;
    let group_size = 64usize;
    let dt = Dt::F16;

    eprintln!(
        "\n=== ISOLATION: ffai_qmm_mma_mpp dense (tensor-unit matmul2d), K={k} GS={group_size} dtype={dt:?} ==="
    );
    for &m in &[512usize, 2048, 4096] {
        for &n in &[512usize, 8192] {
            let iters = if m * n <= 512 * 8192 { 30 } else { 15 };
            let us = time_qmm_mma_mpp(&ctx, m, n, k, group_size, iters, dt);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let gf = gflops(flops, us);
            eprintln!(
                "  M={m:>5} N={n:>5} K={k:>5}: {us:>10.2} us  {gf:>9.1} GFLOP/s  pct57T={:>5.2}%",
                100.0 * gf / 57000.0
            );
        }
    }
}

#[ignore]
#[test]
fn isolation_dense_qmm_coop_prod_shapes() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let k = 2048usize;
    let group_size = 64usize;
    let dt = Dt::F16;

    eprintln!(
        "\n=== ISOLATION: ffai_qmm_coop dense (BN=64 wide tensor-unit matmul2d), K={k} GS={group_size} dtype={dt:?} ==="
    );
    for &m in &[512usize, 2048, 4096] {
        for &n in &[512usize, 8192] {
            let iters = if m * n <= 512 * 8192 { 30 } else { 15 };
            let us = time_qmm_coop(&ctx, m, n, k, group_size, iters, dt);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let gf = gflops(flops, us);
            eprintln!(
                "  M={m:>5} N={n:>5} K={k:>5}: {us:>10.2} us  {gf:>9.1} GFLOP/s  pct57T={:>5.2}%",
                100.0 * gf / 57000.0
            );
        }
    }
}

#[ignore]
#[test]
fn isolation_moe_gather_tileplan_prod_shapes() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let n_experts = 256usize;
    let topk = 8usize;
    let hidden = 2048usize;
    let moe_intermediate = 512usize;
    let group_size = 64usize;
    let dt = Dt::F16;

    eprintln!(
        "\n=== ISOLATION: ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan, {n_experts} experts top-{topk}, Zipf-skewed, dtype={dt:?} ==="
    );
    for &t in &[512usize, 2048, 4096] {
        let m_total = t * topk;
        let iters = if m_total <= 4096 {
            30
        } else if m_total <= 16384 {
            12
        } else {
            6
        };
        for (k_in, n_out, label) in
            [(hidden, moe_intermediate, "gate/up"), (moe_intermediate, hidden, "down")]
        {
            let flops = 2.0 * m_total as f64 * n_out as f64 * k_in as f64;
            let mut us_sum = 0.0;
            let mut tiles_sum = 0usize;
            const SEEDS: usize = 3;
            for seed_i in 0..SEEDS as u64 {
                let seed = 0xF85_5EED_u64 ^ (seed_i.wrapping_mul(0x9E3779B97F4A7C15));
                let counts = skewed_counts(t, topk, n_experts, seed);
                let (us, tiles) = time_moe_gather_tileplan(
                    &ctx, &counts, k_in, n_out, n_experts, group_size, iters, dt,
                );
                us_sum += us;
                tiles_sum += tiles;
            }
            let us = us_sum / SEEDS as f64;
            let tiles = tiles_sum / SEEDS;
            let gf = gflops(flops, us);
            eprintln!(
                "  T={t:>5} mTotal={m_total:>6} {label:<8} K={k_in:>4} N={n_out:>4}: {us:>10.2} us  {gf:>9.1} GFLOP/s  pct57T={:>5.2}%  tiles={tiles}",
                100.0 * gf / 57000.0
            );
        }
    }
}

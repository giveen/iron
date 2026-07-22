//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! F-85 low-context prefill: per-kernel small-M occupancy microbench.
//!
//! `#[ignore]`-gated diagnostic, not a regression gate - run with:
//!   cargo test -p ffai-kernels-std --test smallm_occupancy_microbench --release -- --ignored --nocapture
//!
//! Measures achieved throughput of the two prefill-hot kernel families the
//! F-85 low-context investigation flagged, at the real problem sizes each
//! sees across the T=512..16384 prefill range for Qwen3.6-35B-A3B (hidden
//! 2048, moe_intermediate 512, 256 experts, top-8, group_size 64):
//!
//!   1. The dense int4 MPP GEMM (`ffai_qmm_mma_mpp`) at the GDN in-projection
//!      shapes - `a_raw`/`b_raw` project to `n_out = num_value_heads = 32`,
//!      a single BN=32 tile column regardless of M. Grid = `[n/32, m/32, 1]`
//!      so a narrow `n_out` collapses the whole kernel onto the M axis alone.
//!   2. The MoE gather GEMM (`ffai_moe_gather_qmm_mma_int4_bm16_mpp`, the
//!      production default per `MoELayer.swift`) under REALISTIC skewed
//!      top-8-of-256 routing, not the degenerate all-expert-0 fixture the
//!      shared `int4_mma_bench` harness uses (that fixture hides the
//!      fragmentation cost entirely - every BM=16 tile is single-expert).

#![cfg(target_os = "macos")]

mod common;

use common::{Dt, SplitMix64, gpu_lock};
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::{
    gemm::quantized_mpp::ffai_qmm_mma_mpp,
    moe::{
        moe_mpp::ffai_moe_gather_qmm_mma_int4_bm16_mpp,
        moe_mpp_tileplan::{build_tile_plan, ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan},
        moe_tile_plan_builder::ffai_moe_build_tile_plan,
    },
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

/// Build a realistic skewed top-`topk`-of-`n_experts` routing fixture:
/// Zipf(s=1.0) per-expert popularity, `topk` distinct experts drawn per
/// token via weighted sampling without replacement, output sorted ascending
/// by expert id (the post-permute layout every gather-GEMM kernel expects).
/// Returns `(indices[m_total], run_lengths[n_experts])`.
fn skewed_routing(t: usize, topk: usize, n_experts: usize, seed: u64) -> (Vec<u32>, Vec<usize>) {
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
    let m_total: usize = counts.iter().sum();
    let mut indices = Vec::with_capacity(m_total);
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            indices.push(e as u32);
        }
    }
    (indices, counts)
}

/// Evenly-divided routing (the ceiling reference): every expert gets
/// exactly `t_total/n_experts` contiguous rows, tile-aligned when that
/// count is a multiple of the BM=16 tile height. Near-zero fragmentation.
fn even_routing(t_total: usize, n_experts: usize) -> Vec<u32> {
    let rows_per_expert = t_total / n_experts;
    let mut indices = Vec::with_capacity(t_total);
    for e in 0..n_experts {
        for _ in 0..rows_per_expert {
            indices.push(e as u32);
        }
    }
    while indices.len() < t_total {
        indices.push((n_experts - 1) as u32);
    }
    indices
}

/// Degenerate single-expert routing - the `int4_mma_bench` harness's
/// fixture (`indices` all zero). Zero fragmentation, an absolute ceiling:
/// every BM=16 tile is always a single sub-run over the *entire* K loop.
/// NOT representative of production; included only to bound the range.
fn degenerate_routing(t_total: usize) -> Vec<u32> { vec![0u32; t_total] }

#[allow(clippy::too_many_arguments)]
fn time_moe_gather_bm16_mpp(
    ctx: &Context,
    indices: &[u32],
    k_in: usize,
    n_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
    let m_total = indices.len();
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
    buffers.insert("indices".into(), indices.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert("x_rows".into(), x_rows.iter().flat_map(|i| i.to_le_bytes()).collect());
    buffers.insert("out".into(), pack_f32(&vec![0.0f32; m_total * n_out], dt));
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let mut kernel = ffai_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let grid = [n_out / 32, m_total.div_ceil(16), 1];
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
    total_us / iters as f64
}

#[allow(clippy::too_many_arguments)]
fn time_dense_qmm_mpp(
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

#[allow(clippy::too_many_arguments)]
fn time_moe_gather_bm16_mpp_tileplan(
    ctx: &Context,
    counts: &[usize],
    k_in: usize,
    n_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
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
    total_us / iters as f64
}

fn gflops(flops: f64, us: f64) -> f64 { flops / us / 1e3 }

/// Times the device-side `ffai_moe_build_tile_plan` plan builder alone, and
/// separately the tileplan GEMM dispatched at the host WORST-CASE tile
/// capacity (`ceil(m_total/16) + n_experts`, see that kernel's dispatch
/// invariants) instead of the true tile count - this is what the sync-free
/// Swift wiring actually pays per MoE layer, not the isolated
/// true-tile-count number `bench_moe_gather_tileplan_vs_bm16_mpp_skewed`
/// reports. Both numbers matter: the plan builder must stay small next to
/// the GEMM it unblocks, and the worst-case-capacity dispatch must not
/// erode the GEMM win enough to lose to the production bm16_mpp default.
#[allow(clippy::too_many_arguments)]
fn time_build_tile_plan(
    ctx: &Context,
    sorted_experts: &[u32],
    n_experts: usize,
    capacity: usize,
    iters: usize,
) -> f64 {
    let m_total = sorted_experts.len();
    let mut buffers: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    buffers.insert(
        "sorted_experts".into(),
        sorted_experts.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    buffers.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());

    let mut kernel = ffai_moe_build_tile_plan::kernel_ir();
    kernel.mode = KernelMode::Reduction;
    let grid = [1usize, 1, 1];
    let tg = [n_experts, 1, 1];

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

#[ignore]
#[test]
fn bench_build_tile_plan_and_worstcase_capacity_dispatch() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let n_experts = 256usize;
    let topk = 8usize;
    let hidden = 2048usize;
    let moe_intermediate = 512usize;
    let group_size = 64usize;

    eprintln!(
        "\n=== F-85 sync-free wiring cost: plan-builder + worst-case-capacity tileplan dispatch, realistic skewed routing ==="
    );
    for &t in &[512usize, 4096, 16384] {
        let m_total = t * topk;
        let iters = if t <= 512 {
            60
        } else if t <= 4096 {
            15
        } else {
            5
        };
        let seed = 0xF85_5EED_u64;
        let counts = skewed_counts(t, topk, n_experts, seed);
        let (skewed_idx, _) = skewed_routing(t, topk, n_experts, seed);
        let real_tiles = build_tile_plan(&counts).0.len();
        // Host worst-case capacity, computable from m_total/n_experts alone
        // - the number the Swift wiring actually dispatches with.
        let capacity = m_total.div_ceil(16) + n_experts;

        let plan_us = time_build_tile_plan(&ctx, &skewed_idx, n_experts, capacity, iters);

        for (k_in, n_out, label) in
            [(hidden, moe_intermediate, "gate/up"), (moe_intermediate, hidden, "down")]
        {
            let flops = 2.0 * m_total as f64 * n_out as f64 * k_in as f64;
            let us_old = time_moe_gather_bm16_mpp(
                &ctx,
                &skewed_idx,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            // Worst-case-capacity dispatch: same tileplan kernel, same
            // build_tile_plan-derived counts, but grid.y = capacity (not
            // real_tiles) - the padding tiles past real_tiles have
            // tile_row_count = 0 (zero-filled by the harness below, same
            // as the Swift wiring's pre-dispatch zero-fill).
            let us_new_worstcase = time_moe_gather_bm16_mpp_tileplan_capacity(
                &ctx,
                &counts,
                capacity,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            eprintln!(
                "  T={t:>6} {label:<8} K={k_in:>4} N={n_out:>4}: old={us_old:>9.1}us ({:>6.1} GF/s)  new(worst-case cap, {capacity} tiles vs {real_tiles} real)={us_new_worstcase:>9.1}us ({:>6.1} GF/s)  speedup={:.3}x",
                gflops(flops, us_old),
                gflops(flops, us_new_worstcase),
                us_old / us_new_worstcase,
            );
        }
        eprintln!(
            "  T={t:>6} plan-builder alone: {plan_us:>9.1}us (capacity={capacity}, real_tiles={real_tiles}, m_total={m_total})"
        );
    }
}

/// Like `time_moe_gather_bm16_mpp_tileplan` but dispatches at a FIXED
/// `capacity` (host worst-case bound) instead of the real tile count -
/// entries past `build_tile_plan`'s real tiles are zero-filled padding,
/// mirroring the sync-free Swift wiring's dispatch contract.
#[allow(clippy::too_many_arguments)]
fn time_moe_gather_bm16_mpp_tileplan_capacity(
    ctx: &Context,
    counts: &[usize],
    capacity: usize,
    k_in: usize,
    n_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
    let m_total: usize = counts.iter().sum();
    let (te, trs, trc) = build_tile_plan(counts);
    let mut tile_expert = vec![0u32; capacity];
    let mut tile_row_start = vec![0u32; capacity];
    let mut tile_row_count = vec![0u32; capacity];
    tile_expert[..te.len()].copy_from_slice(&te);
    tile_row_start[..trs.len()].copy_from_slice(&trs);
    tile_row_count[..trc.len()].copy_from_slice(&trc);

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
    let grid = [n_out / 32, capacity, 1];
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
    total_us / iters as f64
}

#[ignore]
#[test]
fn bench_gdn_inproj_dense_qmm_narrow_n() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let k = 2048usize; // hidden
    let group_size = 64usize;
    let iters = 20;

    eprintln!("\n=== dense ffai_qmm_mma_mpp: GDN in-proj shapes (K={k}, GS={group_size}) ===");
    for &t in &[512usize, 4096, 16384] {
        for (n, label) in
            [(32usize, "a/b-gate (N=32)"), (4096, "z-proj (N=4096)"), (8192, "qkv-proj (N=8192)")]
        {
            let us = time_dense_qmm_mpp(&ctx, t, n, k, group_size, iters, Dt::F32);
            let flops = 2.0 * t as f64 * n as f64 * k as f64;
            let gf = gflops(flops, us);
            let n_tiles = n / 32;
            let m_tiles = t / 32;
            let tgs = n_tiles * m_tiles;
            eprintln!(
                "  T={t:>6} {label:<20}: {us:>9.1}us  {gf:>7.1} GFLOP/s  TGs={tgs:>7} (n_tiles={n_tiles} x m_tiles={m_tiles})"
            );
        }
    }
}

/// Counts-based skewed routing (same Zipf model as `skewed_routing`, but
/// returns per-expert counts directly so the tileplan bench doesn't need
/// to re-derive them from a flat `indices` array).
fn skewed_counts(t: usize, topk: usize, n_experts: usize, seed: u64) -> Vec<usize> {
    let (_, counts) = skewed_routing(t, topk, n_experts, seed);
    counts
}

#[ignore]
#[test]
fn bench_moe_gather_tileplan_vs_bm16_mpp_skewed() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let n_experts = 256usize;
    let topk = 8usize;
    let hidden = 2048usize;
    let moe_intermediate = 512usize;
    let group_size = 64usize;

    eprintln!(
        "\n=== F-85 fix: tileplan (expert-aligned tiles) vs production bm16_mpp (sub-run probe), realistic skewed routing ==="
    );
    for &t in &[512usize, 4096, 16384] {
        let m_total = t * topk;
        let iters = if t <= 512 {
            60
        } else if t <= 4096 {
            15
        } else {
            5
        };
        // Average over 3 independent skewed-routing draws per T to damp
        // per-dispatch timing noise AND routing-instance variance - a
        // single seed's fragmentation pattern is not representative.
        for (k_in, n_out, label) in
            [(hidden, moe_intermediate, "gate/up"), (moe_intermediate, hidden, "down")]
        {
            let flops = 2.0 * m_total as f64 * n_out as f64 * k_in as f64;
            let mut us_old_sum = 0.0;
            let mut us_new_sum = 0.0;
            let mut num_tiles_sum = 0usize;
            for seed_i in 0..3u64 {
                let seed = 0xF85_5EED_u64 ^ (seed_i.wrapping_mul(0x9E3779B97F4A7C15));
                let counts = skewed_counts(t, topk, n_experts, seed);
                let (skewed_idx, _) = skewed_routing(t, topk, n_experts, seed);
                let num_tiles = build_tile_plan(&counts).0.len();
                num_tiles_sum += num_tiles;
                us_old_sum += time_moe_gather_bm16_mpp(
                    &ctx,
                    &skewed_idx,
                    k_in,
                    n_out,
                    n_experts,
                    group_size,
                    iters,
                    Dt::F32,
                );
                us_new_sum += time_moe_gather_bm16_mpp_tileplan(
                    &ctx,
                    &counts,
                    k_in,
                    n_out,
                    n_experts,
                    group_size,
                    iters,
                    Dt::F32,
                );
            }
            let us_old = us_old_sum / 3.0;
            let us_new = us_new_sum / 3.0;
            let old_tiles = m_total.div_ceil(16);
            let num_tiles = num_tiles_sum / 3;
            eprintln!(
                "  T={t:>6} {label:<8} K={k_in:>4} N={n_out:>4}: old(sub-run)={us_old:>9.1}us ({:>6.1} GF/s, {old_tiles} tiles)  new(tileplan)={us_new:>9.1}us ({:>6.1} GF/s, {num_tiles} tiles avg)  speedup={:.3}x",
                gflops(flops, us_old),
                gflops(flops, us_new),
                us_old / us_new,
            );
        }
    }
}

#[ignore]
#[test]
fn bench_moe_gather_bm16_mpp_skewed_vs_even_vs_degenerate() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let n_experts = 256usize;
    let topk = 8usize;
    let hidden = 2048usize;
    let moe_intermediate = 512usize;
    let group_size = 64usize;

    eprintln!(
        "\n=== ffai_moe_gather_qmm_mma_int4_bm16_mpp: skewed vs even vs degenerate routing ==="
    );
    for &t in &[512usize, 4096, 16384] {
        let m_total = t * topk;
        let iters = if t <= 4096 { 10 } else { 3 };
        let (skewed_idx, counts) = skewed_routing(t, topk, n_experts, 0xF85_5EED);
        let even_idx = even_routing(m_total, n_experts);
        let degenerate_idx = degenerate_routing(m_total);

        // Fragmentation stats for the skewed fixture: how many BM=16
        // tiles contain >1 distinct expert (a "sub-run" cost multiplier).
        let mut multi_expert_tiles = 0usize;
        let mut total_sub_runs = 0usize;
        let n_tiles_m = m_total.div_ceil(16);
        for tile in 0..n_tiles_m {
            let start = tile * 16;
            let end = (start + 16).min(m_total);
            let mut last = u32::MAX;
            let mut runs_here = 0;
            for &e in &skewed_idx[start..end] {
                if e != last {
                    runs_here += 1;
                    last = e;
                }
            }
            total_sub_runs += runs_here;
            if runs_here > 1 {
                multi_expert_tiles += 1;
            }
        }
        let nonzero_experts = counts.iter().filter(|&&c| c > 0).count();
        let max_expert = counts.iter().max().copied().unwrap_or(0);

        eprintln!(
            "  T={t:>6} mTotal={m_total:>7} skew-stats: nonzero_experts={nonzero_experts}/{n_experts} max_run={max_expert} avg_run={:.1} multi_expert_tiles={multi_expert_tiles}/{n_tiles_m} ({:.1}%) avg_sub_runs/tile={:.2}",
            m_total as f64 / n_experts as f64,
            100.0 * multi_expert_tiles as f64 / n_tiles_m as f64,
            total_sub_runs as f64 / n_tiles_m as f64,
        );

        for (k_in, n_out, label) in
            [(hidden, moe_intermediate, "gate/up"), (moe_intermediate, hidden, "down")]
        {
            let flops = 2.0 * m_total as f64 * n_out as f64 * k_in as f64;
            let us_skewed = time_moe_gather_bm16_mpp(
                &ctx,
                &skewed_idx,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            let us_even = time_moe_gather_bm16_mpp(
                &ctx,
                &even_idx,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            let us_degen = time_moe_gather_bm16_mpp(
                &ctx,
                &degenerate_idx,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            eprintln!(
                "    {label:<8} K={k_in:>4} N={n_out:>4}: skewed={us_skewed:>9.1}us ({:>6.1} GF/s)  even={us_even:>9.1}us ({:>6.1} GF/s)  degenerate={us_degen:>9.1}us ({:>6.1} GF/s)  skewed/degenerate={:.3}  skewed/even={:.3}",
                gflops(flops, us_skewed),
                gflops(flops, us_even),
                gflops(flops, us_degen),
                us_degen / us_skewed,
                us_even / us_skewed,
            );
        }
    }
}

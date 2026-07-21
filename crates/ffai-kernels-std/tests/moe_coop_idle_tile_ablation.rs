//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! F-85 low-context prefill decisive round: idle-tile cost ablation for the
//! PRODUCTION MoE coop gather GEMM (`ffai_moe_gather_qmm_coop`, BM=32 tile
//! plan, wired as the prefill default in FFAI's `MoELayer.swift`).
//!
//! `#[ignore]`-gated diagnostic, not a regression gate - run with:
//!   cargo test -p ffai-kernels-std --test moe_coop_idle_tile_ablation \
//!     --release -- --ignored --nocapture
//!
//! The kernel's guard structure (`moe_gather_qmm_coop.rs`) has NO early
//! return: every dispatched threadgroup, including ones whose
//! `tile_row_count == 0` (the host worst-case-capacity padding past the
//! real tile count - see `moeTilePlanCapacityBm32`'s dispatch contract),
//! runs the FULL K-loop - unconditional W dequant (`Ws` store) and
//! `coop_tile_run` matmul for every `kb` slab. Only the X zero-fill
//! (`select` into threadgroup memory) and the final output store
//! (`out_in_run` guard) are masked. So a "does the idle tile pay for the
//! K-loop" question cannot be answered by reading the guard alone - this
//! measures it directly: same tile-plan buffers, same weights, same grid.x,
//! only grid.y (the tile axis) differs between the production
//! worst-case-capacity dispatch and a hypothetical dispatch clamped to the
//! real tile count (host-computed here for the experiment only - DEVPLAN2
//! forbids this readback in production; a real fix needs indirect dispatch,
//! see `moe_tile_plan_builder_bm32.rs`'s `tile_count_gateup`/
//! `tile_count_down` outputs).

#![cfg(target_os = "macos")]

mod common;

use common::{Dt, gpu_lock};
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::moe::{
    moe_gather_qmm_coop::ffai_moe_gather_qmm_coop,
    moe_mpp_shared::zipfish_counts,
    moe_mpp_tileplan::build_tile_plan_with_bm,
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

fn gflops(flops: f64, us: f64) -> f64 { flops / us / 1e3 }

/// Times `ffai_moe_gather_qmm_coop` at a fixed dispatch tile-axis height
/// (`dispatch_tiles`), independent of the tile plan's real tile count
/// (`real_tiles`, from `build_tile_plan_with_bm`). Passing
/// `dispatch_tiles == capacity` reproduces the production sync-free wiring
/// (`Ops.moeGatherDequantGemmInt4CoopTilePlan`'s worst-case-capacity grid);
/// passing `dispatch_tiles == real_tiles` is the ablation arm - same tile
/// plan buffers (still sized/zero-filled to `capacity` so the kernel's
/// per-tile buffer reads never go out of bounds), just a smaller grid.y so
/// the padding tiles past `real_tiles` are never dispatched at all.
#[allow(clippy::too_many_arguments)]
fn time_moe_coop_at_height(
    ctx: &Context,
    counts: &[usize],
    capacity: usize,
    dispatch_tiles: usize,
    k_in: usize,
    n_out: usize,
    n_experts: usize,
    group_size: usize,
    iters: usize,
    dt: Dt,
) -> f64 {
    let m_total: usize = counts.iter().sum();
    let (te, trs, trc) = build_tile_plan_with_bm(counts, 32);
    assert!(dispatch_tiles <= capacity);
    assert!(te.len() <= capacity, "real tiles must fit host worst-case capacity");
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

    let mut kernel = ffai_moe_gather_qmm_coop::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let grid = [n_out / 64, dispatch_tiles, 1];
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

/// The decisive ablation: production worst-case-capacity dispatch (grid.y =
/// `capacity`) vs the same tile plan clamped to `real_tiles` (grid.y =
/// `real_tiles`, host-computed here for the experiment only). The delta IS
/// the idle-tile cost the F-85 round is gating on - under ~3% means the
/// "near-free" claim holds and utilization is a red herring; 5%+ means the
/// full-K-loop guard structure is really costing material time and the
/// indirect-dispatch fix is warranted.
#[ignore]
#[test]
fn ablation_moe_coop_capacity_vs_real_tile_count() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let n_experts = 256usize;
    let hidden = 2048usize;
    let moe_intermediate = 512usize;
    let group_size = 64usize;

    eprintln!(
        "\n=== F-85 decisive round: ffai_moe_gather_qmm_coop idle-tile-cost ablation (capacity dispatch vs real-tile-count dispatch) ==="
    );
    for &m_total in &[4096usize, 16384, 32768] {
        let iters = if m_total <= 4096 {
            60
        } else if m_total <= 16384 {
            20
        } else {
            10
        };
        let seed = 0x9E37_0001u64;
        let counts = zipfish_counts(m_total, n_experts, seed);
        let (te, ..) = build_tile_plan_with_bm(&counts, 32);
        let real_tiles = te.len();
        let capacity = m_total.div_ceil(32) + n_experts;
        let utilization = 100.0 * real_tiles as f64 / capacity as f64;

        for (k_in, n_out, label) in
            [(hidden, moe_intermediate, "gate/up"), (moe_intermediate, hidden, "down")]
        {
            let flops = 2.0 * m_total as f64 * n_out as f64 * k_in as f64;
            let us_capacity = time_moe_coop_at_height(
                &ctx,
                &counts,
                capacity,
                capacity,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            let us_real = time_moe_coop_at_height(
                &ctx,
                &counts,
                capacity,
                real_tiles,
                k_in,
                n_out,
                n_experts,
                group_size,
                iters,
                Dt::F32,
            );
            let idle_cost_pct = 100.0 * (us_capacity - us_real) / us_capacity;
            eprintln!(
                "  mTotal={m_total:>6} {label:<8} K={k_in:>4} N={n_out:>4}: capacity({capacity} tiles, {utilization:.1}% util)={us_capacity:>9.1}us ({:>6.1} GF/s)  real({real_tiles} tiles)={us_real:>9.1}us ({:>6.1} GF/s)  idle-tile-cost={idle_cost_pct:>6.2}%",
                gflops(flops, us_capacity),
                gflops(flops, us_real),
            );
        }
    }
}

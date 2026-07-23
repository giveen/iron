//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Isolated old-vs-new wall-clock comparison for the parallelized MoE
//! tile-plan builders (F-85 small-M follow-up): the original
//! single-threadgroup builder (one dispatch) vs
//! `iron_moe_tile_plan_expert_counts` chained into the matching
//! parallel phase-2 kernel (two dispatches, fused onto one command
//! buffer via `dispatch_chain`, matching how the Swift wiring calls
//! it) - same shape as `moe_sort_plan_counting_isolated_bench.rs`.
//!
//! `#[ignore]`-gated diagnostic, not a regression gate - run with:
//!   cargo test -p wh-iron-std --release --test \
//!     moe_tile_plan_parallel_isolated_bench -- --ignored --nocapture

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use wh_iron::{Context, DispatchSpec, core::ir::KernelMode};
use wh_iron_std::kernels::moe::{
    moe_mpp_shared::zipfish_counts,
    moe_tile_plan_builder::iron_moe_build_tile_plan,
    moe_tile_plan_builder_bm32::iron_moe_build_tile_plan_bm32,
    moe_tile_plan_builder_bm32_own::iron_moe_build_tile_plan_bm32_own,
    moe_tile_plan_builder_bm32_own_parallel::iron_moe_build_tile_plan_bm32_own_parallel,
    moe_tile_plan_builder_bm32_parallel::iron_moe_build_tile_plan_bm32_parallel,
    moe_tile_plan_builder_parallel::iron_moe_build_tile_plan_parallel,
    moe_tile_plan_expert_counts::iron_moe_tile_plan_expert_counts,
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn sorted_experts_from_counts(counts: &[usize]) -> Vec<u32> {
    let mut out = Vec::new();
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            out.push(e as u32);
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Bm16,
    Bm32,
    Bm32Own,
}

impl Variant {
    fn bm(self) -> usize {
        match self {
            Variant::Bm16 => 16,
            Variant::Bm32 | Variant::Bm32Own => 32,
        }
    }
    fn has_indirect_count(self) -> bool { matches!(self, Variant::Bm32 | Variant::Bm32Own) }
    fn label(self) -> &'static str {
        match self {
            Variant::Bm16 => "bm16",
            Variant::Bm32 => "bm32",
            Variant::Bm32Own => "bm32_own",
        }
    }
}

fn time_original(
    ctx: &Context,
    variant: Variant,
    sorted_experts: &[u32],
    n_experts: usize,
    iters: usize,
) -> f64 {
    let m_total = sorted_experts.len();
    let capacity = m_total.div_ceil(variant.bm()) + n_experts;
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("sorted_experts".into(), u32_bytes(sorted_experts));
    buffers.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    if variant.has_indirect_count() {
        buffers.insert("tile_count_gateup".into(), vec![0u8; 4]);
        buffers.insert("tile_count_down".into(), vec![0u8; 4]);
    }

    let mut k = match variant {
        Variant::Bm16 => iron_moe_build_tile_plan::kernel_ir(),
        Variant::Bm32 => iron_moe_build_tile_plan_bm32::kernel_ir(),
        Variant::Bm32Own => iron_moe_build_tile_plan_bm32_own::kernel_ir(),
    };
    k.mode = KernelMode::Reduction;

    let _ = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("warmup");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx
            .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
            .expect("dispatch");
        total_us += r.elapsed_us;
    }
    total_us / iters as f64
}

fn time_parallel(
    ctx: &Context,
    variant: Variant,
    sorted_experts: &[u32],
    n_experts: usize,
    iters: usize,
) -> f64 {
    let m_total = sorted_experts.len();
    let capacity = m_total.div_ceil(variant.bm()) + n_experts;
    let ids_bytes = u32_bytes(sorted_experts);

    let mut b1: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b1.insert("sorted_experts".into(), ids_bytes);
    b1.insert("expert_row_base".into(), vec![0u8; n_experts * 4]);
    b1.insert("expert_count".into(), vec![0u8; n_experts * 4]);
    b1.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    let mut k1 = iron_moe_tile_plan_expert_counts::kernel_ir();
    k1.mode = KernelMode::Reduction;

    let mut b2: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b2.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    b2.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    b2.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    b2.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    if variant.has_indirect_count() {
        b2.insert("tile_count_gateup".into(), vec![0u8; 4]);
        b2.insert("tile_count_down".into(), vec![0u8; 4]);
    }
    let mut k2 = match variant {
        Variant::Bm16 => iron_moe_build_tile_plan_parallel::kernel_ir(),
        Variant::Bm32 => iron_moe_build_tile_plan_bm32_parallel::kernel_ir(),
        Variant::Bm32Own => iron_moe_build_tile_plan_bm32_own_parallel::kernel_ir(),
    };
    k2.mode = KernelMode::Reduction;

    let empty_resident = BTreeMap::new();
    let empty_fn_consts = BTreeMap::new();
    let build_specs = || {
        [
            DispatchSpec {
                kernel: &k1,
                buffers: &b1,
                fn_consts: &empty_fn_consts,
                grid_groups: [n_experts, 1, 1],
                threads_per_group: [1, 1, 1],
                resident: &empty_resident,
            },
            DispatchSpec {
                kernel: &k2,
                buffers: &b2,
                fn_consts: &empty_fn_consts,
                grid_groups: [1, 1, 1],
                threads_per_group: [n_experts, 1, 1],
                resident: &empty_resident,
            },
        ]
    };

    let _ = ctx.dispatch_chain(&build_specs()).expect("warmup chain");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx.dispatch_chain(&build_specs()).expect("chain");
        // Chained passes share one command buffer - GPU time attributed
        // to the first result, same convention as the sort-plan bench.
        total_us += r[0].elapsed_us;
    }
    total_us / iters as f64
}

/// Interleaves old/new measurement in short bursts (rather than one long
/// `time_original` block followed by one long `time_parallel` block) so a
/// transient system event (background disk activity, thermal/frequency
/// step) skews both arms roughly equally instead of landing entirely
/// inside one arm's measurement window. Returns `(median_us_old,
/// median_us_new)` over `reps` bursts of `burst_iters` each.
fn interleaved_median(
    ctx: &Context,
    variant: Variant,
    sorted_experts: &[u32],
    n_experts: usize,
    reps: usize,
    burst_iters: usize,
) -> (f64, f64) {
    let mut old_samples = Vec::with_capacity(reps);
    let mut new_samples = Vec::with_capacity(reps);
    for i in 0..reps {
        if i % 2 == 0 {
            old_samples.push(time_original(ctx, variant, sorted_experts, n_experts, burst_iters));
            new_samples.push(time_parallel(ctx, variant, sorted_experts, n_experts, burst_iters));
        } else {
            new_samples.push(time_parallel(ctx, variant, sorted_experts, n_experts, burst_iters));
            old_samples.push(time_original(ctx, variant, sorted_experts, n_experts, burst_iters));
        }
    }
    old_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    new_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (old_samples[reps / 2], new_samples[reps / 2])
}

#[ignore]
#[test]
fn isolated_bench_old_vs_parallel() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let n_experts = 256usize;

    eprintln!(
        "\n=== MoE tile-plan builders: single-threadgroup original vs per-expert-parallel, isolated ==="
    );
    eprintln!(
        "(median over 11 interleaved bursts of 20 dispatches each, old/new order alternated per burst)"
    );
    for variant in [Variant::Bm16, Variant::Bm32, Variant::Bm32Own] {
        eprintln!("--- {} ---", variant.label());
        for &m_total in &[4096usize, 8192, 32768] {
            let counts =
                zipfish_counts(m_total, n_experts, 0x5EED_0002u64.wrapping_add(m_total as u64));
            let sorted_experts = sorted_experts_from_counts(&counts);

            let (us_old, us_new) =
                interleaved_median(&ctx, variant, &sorted_experts, n_experts, 11, 20);
            let speedup = us_old / us_new;
            eprintln!(
                "  mTotal={m_total:>6}: original={us_old:>10.1}us   parallel={us_new:>10.1}us   speedup={speedup:>5.2}x"
            );
        }
    }
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Isolated old-vs-new wall-clock comparison for the `ffai_moe_sort_plan`
//! counting-sort replacement, at production-relevant `mTotal`. The O(n^2)
//! original does full-width parallel compares, so this asks the honest
//! question directly: does the O(n*n_experts) three-pass pipeline (fused
//! onto one command buffer via `dispatch_chain`, matching how the Swift
//! DEVPLAN2 path will call it) actually win on wall-clock, not just
//! asymptotically.
//!
//! `#[ignore]`-gated diagnostic, not a regression gate - run with:
//!   cargo test -p ffai-kernels-std --release --test \
//!     moe_sort_plan_counting_isolated_bench -- --ignored --nocapture

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use ffai_kernels::{Context, DispatchSpec, core::ir::KernelMode};
use ffai_kernels_std::kernels::moe::{
    moe_permute::ffai_moe_sort_plan,
    moe_sort_plan_counting::{
        ffai_moe_sort_plan_hist,
        ffai_moe_sort_plan_offsets,
        ffai_moe_sort_plan_scatter,
    },
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn zipf_counts(m_total: usize, n_experts: usize, seed: u64) -> Vec<usize> {
    let mut weights: Vec<f64> = (0..n_experts)
        .map(|e| {
            let jitter = 1.0
                + 0.4
                    * (((e as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f64
                        / 1000.0
                        - 0.5);
            jitter / ((e + 1) as f64)
        })
        .collect();
    let total_w: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total_w;
    }
    let mut counts: Vec<usize> = weights.iter().map(|w| (w * m_total as f64) as usize).collect();
    let assigned: usize = counts.iter().sum();
    if assigned < m_total {
        counts[0] += m_total - assigned;
    } else if assigned > m_total {
        let over = assigned - m_total;
        counts[0] = counts[0].saturating_sub(over);
    }
    counts
}

fn shuffled_ids(counts: &[usize], seed: u64) -> Vec<u32> {
    let mut ids = Vec::new();
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            ids.push(e as u32);
        }
    }
    let mut state = seed | 1;
    let mut next_u64 = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = ids.len();
    for i in (1..n).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    ids
}

fn time_original(ctx: &Context, ids: &[u32], k: usize, iters: usize) -> f64 {
    let m_total = ids.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("topk_ids".into(), u32_bytes(ids));
    buffers.insert("sorted_experts".into(), vec![0u8; m_total * 4]);
    buffers.insert("source_tokens".into(), vec![0u8; m_total * 4]);
    buffers.insert("inv_perm".into(), vec![0u8; m_total * 4]);
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let k_ir = ffai_moe_sort_plan::kernel_ir();
    let tg = m_total.min(256);
    let groups = m_total.div_ceil(tg.max(1));

    let _ = ctx
        .dispatch_with_grid(&k_ir, &buffers, &BTreeMap::new(), [groups.max(1), 1, 1], [
            tg.max(1),
            1,
            1,
        ])
        .expect("warmup");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx
            .dispatch_with_grid(&k_ir, &buffers, &BTreeMap::new(), [groups.max(1), 1, 1], [
                tg.max(1),
                1,
                1,
            ])
            .expect("dispatch");
        total_us += r.elapsed_us;
    }
    total_us / iters as f64
}

#[allow(clippy::too_many_arguments)]
fn time_counting_sort(
    ctx: &Context,
    ids: &[u32],
    k: usize,
    n_experts: usize,
    block_size: usize,
    iters: usize,
) -> f64 {
    let m_total = ids.len();
    let n_blocks = m_total.div_ceil(block_size);

    let ids_bytes = u32_bytes(ids);
    let mut b1: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b1.insert("topk_ids".into(), ids_bytes.clone());
    b1.insert("block_counts".into(), vec![0u8; n_blocks * n_experts * 4]);
    b1.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    b1.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    b1.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    let mut k1 = ffai_moe_sort_plan_hist::kernel_ir();
    k1.mode = KernelMode::Reduction;

    let mut b2: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b2.insert("block_offsets".into(), vec![0u8; n_blocks * n_experts * 4]);
    b2.insert("n_blocks".into(), (n_blocks as u32).to_le_bytes().to_vec());
    b2.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    let mut k2 = ffai_moe_sort_plan_offsets::kernel_ir();
    k2.mode = KernelMode::Reduction;

    let mut b3: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b3.insert("topk_ids".into(), ids_bytes);
    b3.insert("sorted_experts".into(), vec![0u8; m_total * 4]);
    b3.insert("source_tokens".into(), vec![0u8; m_total * 4]);
    b3.insert("inv_perm".into(), vec![0u8; m_total * 4]);
    b3.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    b3.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    b3.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    b3.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    let k3 = ffai_moe_sort_plan_scatter::kernel_ir();
    let tg3 = m_total.min(256);
    let groups3 = m_total.div_ceil(tg3.max(1));

    let empty_resident = BTreeMap::new();
    let empty_fn_consts = BTreeMap::new();
    let build_specs = || {
        [
            DispatchSpec {
                kernel: &k1,
                buffers: &b1,
                fn_consts: &empty_fn_consts,
                grid_groups: [n_blocks, 1, 1],
                threads_per_group: [n_experts, 1, 1],
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
            DispatchSpec {
                kernel: &k3,
                buffers: &b3,
                fn_consts: &empty_fn_consts,
                grid_groups: [groups3.max(1), 1, 1],
                threads_per_group: [tg3.max(1), 1, 1],
                resident: &empty_resident,
            },
        ]
    };

    let _ = ctx.dispatch_chain(&build_specs()).expect("warmup chain");
    let mut total_us = 0.0;
    for _ in 0..iters {
        let r = ctx.dispatch_chain(&build_specs()).expect("chain");
        // Chained passes share one command buffer - GPU time is
        // attributed to the first result (see `dispatch_chain`'s doc).
        total_us += r[0].elapsed_us;
    }
    total_us / iters as f64
}

#[ignore]
#[test]
fn isolated_bench_old_vs_counting_sort() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let k = 8usize;
    let n_experts = 256usize;
    let block_size = 256usize;

    eprintln!(
        "\n=== ffai_moe_sort_plan: O(n^2) original vs O(n*n_experts) counting-sort, isolated ==="
    );
    for &m_total in &[4096usize, 8192, 16384, 32768] {
        let iters = if m_total <= 4096 {
            60
        } else if m_total <= 8192 {
            40
        } else if m_total <= 16384 {
            20
        } else {
            10
        };
        let counts = zipf_counts(m_total, n_experts, 0x5EED_0002u64.wrapping_add(m_total as u64));
        let ids = shuffled_ids(&counts, 0x1357_9BDFu64.wrapping_add(m_total as u64));

        let us_old = time_original(&ctx, &ids, k, iters);
        let us_new = time_counting_sort(&ctx, &ids, k, n_experts, block_size, iters);
        let speedup = us_old / us_new;
        eprintln!(
            "  mTotal={m_total:>6}: original={us_old:>10.1}us   counting-sort={us_new:>10.1}us   speedup={speedup:>5.2}x"
        );
    }
}

/// Sweeps `block_size` at a fixed `mTotal` to find a reasonable default
/// before wiring the Swift side.
#[ignore]
#[test]
fn block_size_sweep_at_production_scale() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let k = 8usize;
    let n_experts = 256usize;
    let m_total = 4096usize;
    let iters = 40;
    let counts = zipf_counts(m_total, n_experts, 0xC0FF_EE01);
    let ids = shuffled_ids(&counts, 0xFACE_B00C);

    eprintln!("\n=== block_size sweep at mTotal={m_total} ===");
    for &block_size in &[64usize, 128, 256, 512, 1024] {
        let us_new = time_counting_sort(&ctx, &ids, k, n_experts, block_size, iters);
        eprintln!("  block_size={block_size:>5}: counting-sort={us_new:>10.1}us");
    }
}

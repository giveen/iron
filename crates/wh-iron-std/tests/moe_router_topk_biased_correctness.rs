//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `kernels::moe::iron_moe_router_topk_biased` — top-K by biased
//! score, weights = unbiased[chosen] renormalised to sum 1.
//!
//! ## Tie-break contract
//!
//! When two or more experts share the same `score_biased` value, which one
//! wins the last top-K slot is an implementation detail of a GPU parallel
//! argmax — not obvious from the spec alone. Reading the kernel
//! (`moe_router_topk_biased.rs`): each of the `k` masked-argmax passes
//! computes `global_best_val = simd_max(best_val)`, then breaks ties among
//! lanes holding that value via `global_best_idx = simd_min(my_idx_or_max)`
//! — i.e. **on a tie, the SMALLEST expert index wins**. This file's CPU
//! reference (`cpu_topk`, below) codifies that explicitly via
//! `.then(a.cmp(&b))` in the sort comparator, rather than relying on
//! `Vec::sort_by`'s stability as an implicit (and easy-to-accidentally-
//! break) proxy for the same rule.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes, unpack_u32_bytes};
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::moe::moe_router_topk_biased::{
    iron_moe_router_topk_biased,
    iron_remap_u32,
};

/// CPU reference for `iron_moe_router_topk_biased`: top-K indices by
/// `biased` descending, ties broken by SMALLEST index (matches the
/// kernel's `simd_min` tie-break — see the tie-break contract doc comment
/// above), weights = `unbiased[chosen]` renormalised to sum to 1.
fn cpu_topk(biased: &[f32], unbiased: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let n = biased.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap().then(a.cmp(&b)));
    let chosen: Vec<usize> = order.iter().take(k).copied().collect();
    let wsum: f32 = chosen.iter().map(|&e| unbiased[e]).sum();
    let want_w: Vec<f32> = chosen.iter().map(|&e| unbiased[e] / wsum).collect();
    (chosen, want_w)
}

/// Dispatch `iron_moe_router_topk_biased` (f32) and assert its
/// (indices, weights) output matches `cpu_topk`.
fn run_and_check(label: &str, biased: &[f32], unbiased: &[f32], k: usize) {
    let n = biased.len();
    let (chosen, want_w) = cpu_topk(biased, unbiased, k);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("score_biased".into(), pack_bytes(biased, Dt::F32));
    buffers.insert("score_unbiased".into(), pack_bytes(unbiased, Dt::F32));
    buffers.insert("indices_out".into(), vec![0u8; k * 4]);
    buffers.insert("weights_out".into(), pack_bytes(&vec![0.0f32; k], Dt::F32));
    buffers.insert("n_experts".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut kernel = iron_moe_router_topk_biased::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [32, 1, 1])
        .expect("dispatch");
    let got_idx = unpack_u32_bytes(result.outputs.get("indices_out").expect("idx"));
    let got_w = unpack_bytes(result.outputs.get("weights_out").expect("w"), Dt::F32);

    eprintln!("[{label}] chosen={chosen:?} got_idx={got_idx:?}");
    assert_eq!(
        got_idx.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        chosen,
        "[{label}] indices (tie-break: smallest index wins)"
    );
    for (i, (g, w)) in got_w.iter().zip(want_w.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "[{label}] weight {i}: got={g} want={w}");
    }
}

#[test]
fn moe_router_topk_biased_f32() {
    let _g = gpu_lock();
    let n = 256usize;
    let k = 6usize;
    // Deterministic distinct biased scores so the top-K is unambiguous.
    let biased: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 251) as f32 * 0.1).collect();
    let unbiased: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 199) as f32 * 0.01 + 0.05).collect();
    run_and_check("distinct", &biased, &unbiased, k);
}

/// Duplicate scores AT the k boundary: experts 4..12 (8 candidates, well
/// above and below them are strictly separated) all share the same biased
/// score, and the top-K cut (k=6) lands inside that tied block. The tied
/// experts occupy exactly rank positions 4..12 pre-tie-break, so slots 4
/// and 5 of the top-6 are contested — exactly the scenario the original
/// "scores deliberately distinct" test structurally could not exercise.
#[test]
fn moe_router_topk_biased_tie_at_k_boundary() {
    let _g = gpu_lock();
    let n = 32usize;
    let k = 6usize;
    let tie_val = 500.0f32;
    let mut biased: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 10.0).collect();
    // Experts 0..4 score strictly above tie_val, experts 4..12 (8 of them)
    // all tie AT tie_val, experts 12..n score strictly below it.
    for (i, v) in biased.iter_mut().enumerate().take(4) {
        *v = 1000.0 - (i as f32); // 1000, 999, 998, 997 — strictly above tie_val
    }
    for &i in &[4, 5, 6, 7, 8, 9, 10, 11] {
        biased[i] = tie_val; // 8-way tie for the last 2 of the top-6
    }
    for (i, v) in biased.iter_mut().enumerate().take(n).skip(12) {
        *v = 100.0 - (i as f32) * 0.1; // strictly below tie_val
    }
    let unbiased: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 199) as f32 * 0.01 + 0.05).collect();
    run_and_check("tie_at_k_boundary", &biased, &unbiased, k);
}

/// All-equal scores: every expert ties. The kernel must fall back entirely
/// to the index-order tie-break (experts 0..k win), and — separately —
/// this is the sharpest test of whether the "already chosen" mask
/// (`chosen_mask` in the kernel) correctly excludes prior picks even when
/// every remaining candidate has the identical value each iteration.
#[test]
fn moe_router_topk_biased_all_equal() {
    let _g = gpu_lock();
    let n = 16usize;
    let k = 6usize;
    let biased = vec![7.0f32; n];
    let unbiased: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 199) as f32 * 0.01 + 0.05).collect();
    run_and_check("all_equal", &biased, &unbiased, k);
}

/// k == n_experts: every expert is selected, exercising the boundary where
/// the final masked-argmax pass has exactly one unmasked candidate left
/// (no tie-break freedom at all — a good check that the mask logic doesn't
/// e.g. off-by-one and leave two experts unmasked on the last pass).
#[test]
fn moe_router_topk_biased_k_equals_n() {
    let _g = gpu_lock();
    let n = 8usize;
    let k = 8usize;
    let biased: Vec<f32> = vec![3.0, 3.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0];
    let unbiased: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 199) as f32 * 0.01 + 0.05).collect();
    run_and_check("k_equals_n", &biased, &unbiased, k);
}

/// k == 1 with a tie at the top: the simplest possible tie-break case,
/// isolated from any masking/multi-pass interaction.
#[test]
fn moe_router_topk_biased_k1_tie() {
    let _g = gpu_lock();
    let n = 10usize;
    let k = 1usize;
    let mut biased = vec![1.0f32; n];
    biased[3] = 9.0;
    biased[6] = 9.0; // two-way tie for the single top slot
    let unbiased: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 199) as f32 * 0.01 + 0.05).collect();
    run_and_check("k1_tie", &biased, &unbiased, k);
}

/// `iron_remap_u32`: out[i] = table[idx[i]] — a plain u32 gather over `n`
/// elements. Non-generic Grid3D kernel (one thread per output), so it
/// dispatches via `kernel_ir()` and a [n,1,1] grid with [1,1,1] tg.
#[test]
fn remap_u32_matches_cpu() {
    let _g = gpu_lock();
    let n = 6usize;
    let table_len = 256usize;
    let table: Vec<u32> = (0..table_len).map(|e| ((e * 37 + 11) % table_len) as u32).collect();
    let idx: Vec<u32> = vec![3, 200, 0, 255, 128, 64];
    let want: Vec<u32> = idx.iter().map(|&e| table[e as usize]).collect();

    let to_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("table".into(), to_bytes(&table));
    buffers.insert("idx".into(), to_bytes(&idx));
    buffers.insert("out".into(), vec![0u8; n * 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let kernel = iron_remap_u32::kernel_ir();
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n, 1, 1], [1, 1, 1])
        .expect("dispatch");
    let got = unpack_u32_bytes(result.outputs.get("out").expect("out"));

    eprintln!("want={want:?} got={got:?}");
    assert_eq!(got, want, "remap_u32 gather mismatch");
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `iron_moe_build_tile_plan` (F-85 device-side tile-plan
//! builder) against the `build_tile_plan` host reference, on the same kind
//! of skewed/ragged routing fixture `moe_gather_qmm_tileplan_correctness.rs`
//! uses for the GEMM kernel this feeds: several zero-count experts, one
//! expert spanning many BM=16 tiles, several short runs under 16 rows, no
//! run boundary aligned to 16.
//!
//! Unlike the GEMM correctness test, this kernel takes `sorted_experts`
//! directly (no separate `counts` input) and must derive the same tile
//! plan `build_tile_plan(counts)` would, purely from the sorted array -
//! that is the whole point (the production caller never has `counts` on
//! the host; it only has the device-sorted expert-id array).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::moe::{
    moe_mpp_tileplan::build_tile_plan,
    moe_tile_plan_builder::iron_moe_build_tile_plan,
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Runs the kernel over a `counts`-derived `sorted_experts` fixture and
/// returns the produced `(tile_expert, tile_row_start, tile_row_count)`,
/// truncated to the REAL tile count (`build_tile_plan(counts).0.len()`) -
/// entries past that are host-owned worst-case padding, checked separately
/// for the zero-fill contract rather than compared to the oracle's
/// (shorter) vectors.
fn run_case(counts: &[usize], n_experts: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, usize) {
    let _g = gpu_lock();
    let m_total: usize = counts.iter().sum();
    let mut sorted_experts = Vec::with_capacity(m_total);
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            sorted_experts.push(e as u32);
        }
    }

    // Host worst-case capacity - see the kernel's dispatch-invariants doc:
    // ceil(m_total/16) + n_experts, computable from m_total/n_experts alone.
    let capacity = m_total.div_ceil(16) + n_experts;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("sorted_experts".into(), u32_bytes(&sorted_experts));
    buffers.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = iron_moe_build_tile_plan::kernel_ir();
    k.mode = KernelMode::Reduction;
    // ONE threadgroup, n_experts lanes - see the kernel's dispatch invariants.
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("build_tile_plan dispatch");

    let got_expert = unpack_u32(r.outputs.get("tile_expert").unwrap());
    let got_start = unpack_u32(r.outputs.get("tile_row_start").unwrap());
    let got_count = unpack_u32(r.outputs.get("tile_row_count").unwrap());
    (got_expert, got_start, got_count, capacity)
}

fn check_case(counts: &[usize], n_experts: usize) {
    let (expected_expert, expected_start, expected_count) = build_tile_plan(counts);
    let real_tiles = expected_expert.len();
    let (got_expert, got_start, got_count, capacity) = run_case(counts, n_experts);

    assert_eq!(
        &got_expert[..real_tiles],
        &expected_expert[..],
        "tile_expert diverges from build_tile_plan oracle, counts={counts:?}"
    );
    assert_eq!(
        &got_start[..real_tiles],
        &expected_start[..],
        "tile_row_start diverges from build_tile_plan oracle, counts={counts:?}"
    );
    assert_eq!(
        &got_count[..real_tiles],
        &expected_count[..],
        "tile_row_count diverges from build_tile_plan oracle, counts={counts:?}"
    );

    // Worst-case padding past the real tile count: the kernel never
    // touches these (host pre-zeros them before dispatch, verified here
    // by construction since the input buffers started zeroed) - this
    // assertion is really documenting the contract the GEMM kernel relies
    // on, not exercising new kernel behavior.
    assert!(
        got_count[real_tiles..capacity].iter().all(|&c| c == 0),
        "padding past real_tiles={real_tiles} (capacity={capacity}) must stay zero-filled, counts={counts:?}"
    );
}

#[test]
fn matches_oracle_on_skewed_ragged_routing() {
    // Same fixture shape as moe_gather_qmm_tileplan_correctness.rs: zero
    // counts, one expert spanning 3 tiles, several short/misaligned runs.
    check_case(&[1, 0, 47, 0, 5, 22, 0, 3, 16, 9], 10);
}

#[test]
fn matches_oracle_on_all_zero_but_one_expert() {
    // Degenerate: every row lands on a single expert (the worst case for
    // the kernel's own binary search - count = m_total for one lane).
    check_case(&[0, 0, 130, 0, 0], 5);
}

#[test]
fn matches_oracle_on_every_expert_touched_once() {
    // Every expert gets exactly one row - max tile-count fragmentation
    // relative to m_total, exercises the prefix-sum across many nonzero,
    // all-size-1 experts.
    check_case(&[1; 32], 32);
}

#[test]
fn matches_oracle_on_tile_aligned_even_counts() {
    // Every expert's count is an exact multiple of 16 - no partial tiles,
    // sanity check against off-by-one in the ceiling division.
    check_case(&[16, 32, 0, 48, 16], 5);
}

#[test]
fn matches_oracle_at_realistic_scale() {
    // Larger n_experts / m_total closer to the production shape (still far
    // below the 22-iteration binary search's 2^22 exactness bound), mixed
    // zero/short/long runs.
    let mut counts = vec![0usize; 256];
    let mut rng_state = 0x5EEDu64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as u32
    };
    for (e, c) in counts.iter_mut().enumerate() {
        // Skew: most experts light, a handful heavy, several exactly zero.
        *c = match e % 7 {
            0 => 0,
            1 => (next() % 8) as usize,
            2 => (next() % 40) as usize,
            _ => (next() % 400) as usize,
        };
    }
    check_case(&counts, 256);
}

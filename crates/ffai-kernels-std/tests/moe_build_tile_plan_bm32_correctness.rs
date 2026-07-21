//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai_moe_build_tile_plan_bm32` (the production
//! device tile-plan builder feeding `ffai_moe_gather_qmm_coop`), covering
//! both the pre-existing tile-emission contract (against the host
//! `build_tile_plan_with_bm(counts, 32)` reference) and the F-85
//! idle-tile-cost follow-up's new `tile_count_gateup`/`tile_count_down`
//! indirect-dispatch outputs.
//!
//! This kernel had no dedicated GPU correctness test before this round -
//! only a unit test asserting the IR has no inline MSL. The device tile
//! plan itself was previously only exercised indirectly, through
//! `ffai_moe_gather_qmm_coop`'s own kernel_tests, which build the tile plan
//! on the HOST (`build_tile_plan_with_bm`), not the device. This file
//! closes that gap for the device builder directly, since it is now being
//! modified.

#![cfg(target_os = "macos")]

mod common;

use common::gpu_lock;
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::moe::{
    moe_mpp_tileplan::build_tile_plan_with_bm,
    moe_tile_plan_builder_bm32::ffai_moe_build_tile_plan_bm32,
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Dispatches the device builder against a per-expert row-count fixture
/// and returns `(tile_expert, tile_row_start, tile_row_count, capacity,
/// tile_count_gateup, tile_count_down)`. `tile_count_gateup`/`tile_count_down`
/// are seeded with a poison value (`0xDEADBEEF`) before dispatch so a
/// builder that silently skips the last-lane write fails loudly instead of
/// passing by coincidence.
#[allow(clippy::type_complexity)]
fn run_case(counts: &[usize], n_experts: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, usize, u32, u32) {
    let _g = gpu_lock();
    let m_total: usize = counts.iter().sum();
    let mut sorted_experts = Vec::with_capacity(m_total);
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            sorted_experts.push(e as u32);
        }
    }

    let capacity = m_total.div_ceil(32) + n_experts;

    let mut buffers = std::collections::BTreeMap::new();
    buffers.insert("sorted_experts".to_string(), u32_bytes(&sorted_experts));
    buffers.insert("tile_expert".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_count_gateup".to_string(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
    buffers.insert("tile_count_down".to_string(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
    buffers.insert("m_total".to_string(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".to_string(), (n_experts as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = ffai_moe_build_tile_plan_bm32::kernel_ir();
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &std::collections::BTreeMap::new(), [1, 1, 1], [
            n_experts, 1, 1,
        ])
        .expect("bm32 tile-plan dispatch");

    let tile_count_gateup = unpack_u32(r.outputs.get("tile_count_gateup").unwrap())[0];
    let tile_count_down = unpack_u32(r.outputs.get("tile_count_down").unwrap())[0];

    (
        unpack_u32(r.outputs.get("tile_expert").unwrap()),
        unpack_u32(r.outputs.get("tile_row_start").unwrap()),
        unpack_u32(r.outputs.get("tile_row_count").unwrap()),
        capacity,
        tile_count_gateup,
        tile_count_down,
    )
}

/// Pre-existing contract (unchanged by this round): the device builder's
/// real entries (`0..real_tiles`) must match the host reference
/// `build_tile_plan_with_bm(counts, 32)` exactly (same tile boundaries,
/// same ascending-by-expert concatenation order); entries past the real
/// tile count are zero-fill padding.
fn assert_matches_host_reference(counts: &[usize], n_experts: usize) {
    let (te, trs, trc, capacity, _gu, _dn) = run_case(counts, n_experts);
    let (host_te, host_trs, host_trc) = build_tile_plan_with_bm(counts, 32);
    let real_tiles = host_te.len();
    assert!(real_tiles <= capacity, "counts={counts:?}");
    assert_eq!(&te[..real_tiles], &host_te[..], "tile_expert mismatch; counts={counts:?}");
    assert_eq!(&trs[..real_tiles], &host_trs[..], "tile_row_start mismatch; counts={counts:?}");
    assert_eq!(&trc[..real_tiles], &host_trc[..], "tile_row_count mismatch; counts={counts:?}");
    for &c in &trc[real_tiles..] {
        assert_eq!(c, 0, "padding past real_tiles must stay zero-filled; counts={counts:?}");
    }
}

/// The F-85 idle-tile-cost follow-up's new contract: `tile_count_gateup`/
/// `tile_count_down` must equal the REAL tile count, not the worst-case
/// `capacity` this dispatch was sized to - that gap is exactly the
/// padding the indirect-dispatch fix is meant to eliminate. Both outputs
/// must agree (they carry the same value into two differently-shaped
/// indirect-args triples for the gate/up and down legs).
fn assert_tile_count_matches_real_tiles(counts: &[usize], n_experts: usize) {
    let (_te, _trs, _trc, capacity, gateup, down) = run_case(counts, n_experts);
    let real_tiles = build_tile_plan_with_bm(counts, 32).0.len() as u32;
    assert_eq!(gateup, real_tiles, "tile_count_gateup vs real tile count; counts={counts:?}");
    assert_eq!(down, real_tiles, "tile_count_down vs real tile count; counts={counts:?}");
    assert!(
        (real_tiles as usize) <= capacity,
        "real tile count must never exceed the worst-case capacity; counts={counts:?}"
    );
}

#[test]
fn matches_host_reference_on_skewed_ragged_routing() {
    let counts = [1usize, 0, 200, 0, 5, 22, 0, 3, 32, 9, 0, 47];
    assert_matches_host_reference(&counts, counts.len());
}

#[test]
fn matches_host_reference_on_all_zero_but_one_expert() {
    let counts = [0usize, 0, 130, 0, 0];
    assert_matches_host_reference(&counts, counts.len());
}

#[test]
fn matches_host_reference_on_every_expert_touched_once() {
    let counts = [1usize; 32];
    assert_matches_host_reference(&counts, counts.len());
}

#[test]
fn matches_host_reference_on_exact_bm_multiples() {
    let counts = [32usize, 64, 0, 96, 32];
    assert_matches_host_reference(&counts, counts.len());
}

#[test]
fn matches_host_reference_at_realistic_scale() {
    let mut counts = vec![0usize; 256];
    let mut rng_state = 0x5EEDu64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as u32
    };
    for (e, c) in counts.iter_mut().enumerate() {
        *c = match e % 7 {
            0 => 0,
            1 => (next() % 8) as usize,
            2 => (next() % 40) as usize,
            _ => (next() % 400) as usize,
        };
    }
    assert_matches_host_reference(&counts, 256);
}

#[test]
fn matches_host_reference_on_zipfish_realistic_fixture() {
    let counts =
        ffai_kernels_std::kernels::moe::moe_mpp_shared::zipfish_counts(4096, 256, 0x9E37_0001);
    assert_matches_host_reference(&counts, 256);
}

#[test]
fn tile_count_zero_when_m_total_zero() { assert_tile_count_matches_real_tiles(&[0usize, 0, 0], 3); }

#[test]
fn tile_count_matches_on_skewed_ragged_routing() {
    let counts = [1usize, 0, 200, 0, 5, 22, 0, 3, 32, 9, 0, 47];
    assert_tile_count_matches_real_tiles(&counts, counts.len());
}

#[test]
fn tile_count_matches_on_every_expert_touched_once() {
    let counts = [1usize; 32];
    assert_tile_count_matches_real_tiles(&counts, counts.len());
}

#[test]
fn tile_count_matches_on_exact_bm_multiples() {
    let counts = [32usize, 64, 0, 96, 32];
    assert_tile_count_matches_real_tiles(&counts, counts.len());
}

#[test]
fn tile_count_matches_at_realistic_scale() {
    let mut counts = vec![0usize; 256];
    let mut rng_state = 0x5EEDu64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as u32
    };
    for (e, c) in counts.iter_mut().enumerate() {
        *c = match e % 7 {
            0 => 0,
            1 => (next() % 8) as usize,
            2 => (next() % 40) as usize,
            _ => (next() % 400) as usize,
        };
    }
    assert_tile_count_matches_real_tiles(&counts, 256);
}

#[test]
fn tile_count_matches_on_zipfish_realistic_fixture() {
    let counts =
        ffai_kernels_std::kernels::moe::moe_mpp_shared::zipfish_counts(4096, 256, 0x9E37_0001);
    assert_tile_count_matches_real_tiles(&counts, 256);
}

#[test]
fn tile_count_matches_on_zipfish_at_all_three_prod_sizes() {
    for &m_total in &[4096usize, 16384, 32768] {
        let counts = ffai_kernels_std::kernels::moe::moe_mpp_shared::zipfish_counts(
            m_total,
            256,
            0x9E37_0001,
        );
        assert_tile_count_matches_real_tiles(&counts, 256);
    }
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai_moe_build_tile_plan_bm32_own` (the "own" leg
//! of the F-85 tile-pairing lever's two-dispatch split) against the
//! `build_own_tile_plan_bm32` host reference - same skewed/ragged fixture
//! style as `moe_build_tile_plan_correctness.rs` (this kernel's unpaired
//! BM=32 sibling), plus fixtures specifically targeting the ONE thing that
//! differs: a `1..=16`-row remainder must produce NO tile here (it belongs
//! to `ffai_moe_build_tile_plan_bm32_paired` instead), while every other
//! shape (exact multiples of 32, `17..=31`-row remainders) must match the
//! unpaired sibling's own tile-for-tile output exactly.
//!
//! Also covers the F-85 idle-tile-cost follow-up's `tile_count_gateup`/
//! `tile_count_down` indirect-dispatch outputs (poisoned-buffer oracle,
//! mirroring `moe_build_tile_plan_bm32_paired_correctness.rs`'s pair-count
//! coverage): both must equal the real "own" tile count, not the
//! worst-case capacity.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::moe::{
    moe_tile_plan_builder_bm32_own::ffai_moe_build_tile_plan_bm32_own,
    moe_tile_plan_builder_paired::build_own_tile_plan_bm32,
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

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

    // Same worst-case capacity bound as the unpaired BM=32 sibling - this
    // kernel never emits more tiles than that one does for the same
    // fixture (it only ever emits FEWER, by excluding 1..=16 remainders).
    let capacity = m_total.div_ceil(32) + n_experts;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("sorted_experts".into(), u32_bytes(&sorted_experts));
    buffers.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    // Non-zero seed values so a kernel that silently skips the last-lane
    // write fails loudly instead of coincidentally reading back as 0.
    buffers.insert("tile_count_gateup".into(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
    buffers.insert("tile_count_down".into(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = ffai_moe_build_tile_plan_bm32_own::kernel_ir();
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("build_tile_plan_bm32_own dispatch");

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

fn check_case(counts: &[usize], n_experts: usize) {
    let (expected_expert, expected_start, expected_count) = build_own_tile_plan_bm32(counts);
    let real_tiles = expected_expert.len();
    let (got_expert, got_start, got_count, capacity, _gu, _dn) = run_case(counts, n_experts);

    assert_eq!(
        &got_expert[..real_tiles],
        &expected_expert[..],
        "tile_expert diverges from build_own_tile_plan_bm32 oracle, counts={counts:?}"
    );
    assert_eq!(
        &got_start[..real_tiles],
        &expected_start[..],
        "tile_row_start diverges from build_own_tile_plan_bm32 oracle, counts={counts:?}"
    );
    assert_eq!(
        &got_count[..real_tiles],
        &expected_count[..],
        "tile_row_count diverges from build_own_tile_plan_bm32 oracle, counts={counts:?}"
    );
    assert!(
        got_count[real_tiles..capacity].iter().all(|&c| c == 0),
        "padding past real_tiles={real_tiles} (capacity={capacity}) must stay zero-filled, counts={counts:?}"
    );
    // Every real tile's row_count is either exactly 32 (a full run) or
    // 17..=31 (a solo remainder) - NEVER 1..=16 (that would be a pairing
    // candidate this kernel must not touch).
    for &c in &got_count[..real_tiles] {
        assert!(
            c == 32 || (17..=31).contains(&c),
            "unexpected own-tile row_count {c}, counts={counts:?}"
        );
    }
}

#[test]
fn matches_oracle_on_skewed_ragged_routing() { check_case(&[1, 0, 47, 0, 5, 22, 0, 3, 16, 9], 10); }

#[test]
fn matches_oracle_on_all_zero_but_one_expert() { check_case(&[0, 0, 130, 0, 0], 5); }

#[test]
fn matches_oracle_on_every_expert_touched_once() {
    // Every expert's count is 1 - EVERY expert is a pure pairing
    // candidate, so this kernel must emit ZERO tiles.
    let counts = [1usize; 32];
    let (got_expert, _got_start, got_count, _capacity, _gu, _dn) = run_case(&counts, 32);
    let real_tiles = got_expert.len().min(got_count.iter().filter(|&&c| c > 0).count());
    assert_eq!(
        real_tiles, 0,
        "every count=1 expert is a pairing candidate, own plan must be empty"
    );
}

#[test]
fn matches_oracle_on_tile_aligned_even_counts() { check_case(&[16, 32, 0, 48, 16], 5); }

/// The property this kernel exists for: a `1..=16`-row remainder produces
/// NO tile, while a `17..=31`-row remainder produces its usual split.
#[test]
fn short_remainder_produces_no_tile_long_remainder_does() {
    // Expert 0: count=16 (pure candidate, remainder=16) -> 0 own tiles.
    // Expert 1: count=48 (full=1, remainder=16) -> 1 own tile (the full
    //   32-row run only; its 16-row remainder is a candidate elsewhere).
    // Expert 2: count=49 (full=1, remainder=17) -> 2 own tiles (32 + 17).
    let counts = [16usize, 48usize, 49usize];
    let (got_expert, got_start, got_count, _capacity, _gu, _dn) = run_case(&counts, 3);
    let real_tiles = got_expert.iter().zip(&got_count).filter(|&(_, &c)| c > 0).count();
    assert_eq!(real_tiles, 3, "counts={counts:?}");
    let mut got: Vec<(u32, u32, u32)> =
        (0..real_tiles).map(|i| (got_expert[i], got_start[i], got_count[i])).collect();
    got.sort_unstable();
    // Expert 0: row_base=0, no tile.
    // Expert 1: row_base=16, one 32-row tile at row 16.
    // Expert 2: row_base=64, one 32-row tile at row 64, one 17-row tile at row 96.
    let mut want = vec![(1, 16, 32), (2, 64, 32), (2, 96, 17)];
    want.sort_unstable();
    assert_eq!(got, want, "counts={counts:?}");
}

#[test]
fn matches_oracle_at_realistic_scale() {
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
    check_case(&counts, 256);
}

#[test]
fn matches_oracle_on_zipfish_realistic_fixture() {
    let counts =
        ffai_kernels_std::kernels::moe::moe_mpp_shared::zipfish_counts(512, 256, 0x9E37_0001);
    check_case(&counts, 256);
}

/// F-85 idle-tile-cost follow-up: `tile_count_gateup`/`tile_count_down`
/// must equal the real "own" tile count (`build_own_tile_plan_bm32`'s
/// output length), not the worst-case `capacity` this dispatch was sized
/// to. Mirrors `moe_build_tile_plan_bm32_paired_correctness.rs`'s
/// pair-count oracle coverage.
fn assert_tile_count_matches_real_tiles(counts: &[usize], n_experts: usize) {
    let (_te, _trs, _trc, capacity, gateup, down) = run_case(counts, n_experts);
    let real_tiles = build_own_tile_plan_bm32(counts).0.len() as u32;
    assert_eq!(gateup, real_tiles, "tile_count_gateup vs real own-tile count; counts={counts:?}");
    assert_eq!(down, real_tiles, "tile_count_down vs real own-tile count; counts={counts:?}");
    assert!(
        (real_tiles as usize) <= capacity,
        "real own-tile count must never exceed the worst-case capacity; counts={counts:?}"
    );
}

#[test]
fn tile_count_zero_when_every_expert_is_a_pairing_candidate() {
    // Every count=1 -> every expert is a pure pairing candidate, so the
    // real "own" tile count must be 0, not the worst-case capacity.
    assert_tile_count_matches_real_tiles(&[1usize; 32], 32);
}

#[test]
fn tile_count_matches_on_skewed_ragged_routing() {
    assert_tile_count_matches_real_tiles(&[1, 0, 47, 0, 5, 22, 0, 3, 16, 9], 10);
}

#[test]
fn tile_count_matches_on_all_zero_but_one_expert() {
    assert_tile_count_matches_real_tiles(&[0, 0, 130, 0, 0], 5);
}

#[test]
fn tile_count_matches_on_tile_aligned_even_counts() {
    assert_tile_count_matches_real_tiles(&[16, 32, 0, 48, 16], 5);
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

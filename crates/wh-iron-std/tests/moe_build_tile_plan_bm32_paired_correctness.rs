//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `iron_moe_build_tile_plan_bm32_paired` (the device
//! PAIR-ONLY tile-plan builder feeding `iron_moe_gather_qmm_coop_paired`).
//!
//! This kernel only emits tiles for the `1..=16`-row pairing candidates
//! (see `moe_tile_plan_builder_paired.rs`'s module doc for why full BM=32
//! runs and `17..=31`-row remainders are handled by the UNCHANGED existing
//! `iron_moe_build_tile_plan_bm32` / `iron_moe_gather_qmm_coop` path
//! instead). This test does NOT assert list equality against the host
//! reference (`split_for_pairing`): the host sorts candidates
//! descending-by-count before pairing, while the device builder pairs
//! them in ascending-expert-compaction order (see the kernel's module
//! doc) - a deliberate, harmless divergence documented in both builders.
//! What must match is the property that actually matters for the
//! consuming GEMM kernel: every candidate row (a `1..=16`-row remainder)
//! appears in exactly one (tile, half) slot, every half's row count stays
//! within its `1..=16` / `0..=16` bound, and non-candidate rows (full
//! runs, `17..=31`-row remainders, zero-count experts) contribute NO
//! tiles from this kernel at all.

#![cfg(target_os = "macos")]

mod common;

use common::gpu_lock;
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::moe::{
    moe_tile_plan_builder_bm32_paired::iron_moe_build_tile_plan_bm32_paired,
    moe_tile_plan_builder_paired::paired_tile_plan_worst_case_capacity,
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[allow(clippy::type_complexity)]
fn run_case(
    counts: &[usize],
    n_experts: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
    let (ea, sa, ca, eb, sb, cb, capacity, _gu, _dn) = run_case_with_pair_counts(counts, n_experts);
    (ea, sa, ca, eb, sb, cb, capacity)
}

/// Same dispatch as `run_case`, plus the two indirect-dispatch pair-count
/// outputs (F-85 follow-up - see the kernel module doc).
#[allow(clippy::type_complexity)]
fn run_case_with_pair_counts(
    counts: &[usize],
    n_experts: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize, u32, u32) {
    let _g = gpu_lock();
    let m_total: usize = counts.iter().sum();
    let mut sorted_experts = Vec::with_capacity(m_total);
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            sorted_experts.push(e as u32);
        }
    }

    let capacity = paired_tile_plan_worst_case_capacity(n_experts);

    let mut buffers = std::collections::BTreeMap::new();
    buffers.insert("sorted_experts".to_string(), u32_bytes(&sorted_experts));
    buffers.insert("tile_expert_a".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start_a".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count_a".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_expert_b".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start_b".to_string(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count_b".to_string(), vec![0u8; capacity * 4]);
    // Non-zero seed values so a kernel that silently skips the write (e.g.
    // the last-lane guard never firing) fails loudly instead of coincidentally
    // reading back as 0.
    buffers.insert("pair_count_gateup".to_string(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
    buffers.insert("pair_count_down".to_string(), 0xDEAD_BEEFu32.to_le_bytes().to_vec());
    buffers.insert("m_total".to_string(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".to_string(), (n_experts as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = iron_moe_build_tile_plan_bm32_paired::kernel_ir();
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &std::collections::BTreeMap::new(), [1, 1, 1], [
            n_experts, 1, 1,
        ])
        .expect("paired tile-plan dispatch");

    let pair_count_gateup = unpack_u32(r.outputs.get("pair_count_gateup").unwrap())[0];
    let pair_count_down = unpack_u32(r.outputs.get("pair_count_down").unwrap())[0];

    (
        unpack_u32(r.outputs.get("tile_expert_a").unwrap()),
        unpack_u32(r.outputs.get("tile_row_start_a").unwrap()),
        unpack_u32(r.outputs.get("tile_row_count_a").unwrap()),
        unpack_u32(r.outputs.get("tile_expert_b").unwrap()),
        unpack_u32(r.outputs.get("tile_row_start_b").unwrap()),
        unpack_u32(r.outputs.get("tile_row_count_b").unwrap()),
        capacity,
        pair_count_gateup,
        pair_count_down,
    )
}

fn covered_rows(
    ea: &[u32],
    sa: &[u32],
    ca: &[u32],
    eb: &[u32],
    sb: &[u32],
    cb: &[u32],
) -> Vec<(u32, u32)> {
    let mut rows = Vec::new();
    for i in 0..ca.len() {
        for r in 0..ca[i] {
            rows.push((sa[i] + r, ea[i]));
        }
        for r in 0..cb[i] {
            rows.push((sb[i] + r, eb[i]));
        }
    }
    rows
}

/// Expected candidate rows only: each expert's `1..=16`-row remainder
/// (the tail after its full BM=32 runs), NOT its full row range - full
/// runs and `17..=31`-row remainders are out of this kernel's scope.
fn expected_candidate_rows(counts: &[usize]) -> Vec<(u32, u32)> {
    let mut expected = Vec::new();
    let mut row_base = 0u32;
    for (e, &c) in counts.iter().enumerate() {
        if c > 0 {
            let full = c / 32;
            let remainder = c % 32;
            if (1..=16).contains(&remainder) {
                let start = row_base + (full * 32) as u32;
                for r in 0..remainder as u32 {
                    expected.push((start + r, e as u32));
                }
            }
        }
        row_base += c as u32;
    }
    expected
}

fn assert_matches_candidates(counts: &[usize], n_experts: usize) {
    let (ea, sa, ca, eb, sb, cb, capacity) = run_case(counts, n_experts);
    // Real tiles (`ca > 0`) must carry 1..=16 real A-half rows; entries
    // past the real tile count are host-zero-filled padding (`ca == 0`).
    for &c in ca.iter().filter(|&&c| c > 0) {
        assert!((1..=16).contains(&c), "A half must carry 1..=16 rows, got {c}, counts={counts:?}");
    }
    for &c in &cb {
        assert!(c <= 16, "B half must carry 0..=16 rows, got {c}, counts={counts:?}");
    }
    let mut got = covered_rows(&ea, &sa, &ca, &eb, &sb, &cb);
    let mut want = expected_candidate_rows(counts);
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want, "counts={counts:?}");

    let real_tiles = ca.iter().filter(|&&c| c > 0).count();
    assert!(real_tiles <= capacity, "counts={counts:?}");
}

#[test]
fn matches_candidates_on_skewed_ragged_routing() {
    assert_matches_candidates(&[1, 0, 47, 0, 5, 22, 0, 3, 16, 9], 10);
}

#[test]
fn matches_candidates_on_all_zero_but_one_expert() {
    // count=130: full=4, remainder=2 -> ONE candidate (the 2-row tail).
    assert_matches_candidates(&[0, 0, 130, 0, 0], 5);
}

#[test]
fn matches_candidates_on_every_expert_touched_once() {
    // Every expert has count=1 -> every one is a candidate.
    assert_matches_candidates(&[1; 32], 32);
}

#[test]
fn full_bm32_runs_and_exact_multiples_produce_no_candidates() {
    // 16, 32, 0, 48, 16: remainders are 16, 0, 0, 16, 16 - three
    // candidates (the two 16-counts and 48's remainder is 16), one
    // exact-multiple (32) and one zero contribute nothing.
    assert_matches_candidates(&[16, 32, 0, 48, 16], 5);
}

/// The core pairing hazard fixture: two experts, each exactly at the
/// 16-row pairing cap, must merge into ONE device tile with two DIFFERENT
/// expert ids across its two halves.
#[test]
fn device_pairs_two_complementary_16_row_experts_into_one_tile() {
    let (ea, _sa, ca, eb, _sb, cb, _cap) = run_case(&[16, 16], 2);
    let real: Vec<usize> = (0..ea.len()).filter(|&i| ca[i] > 0).collect();
    assert_eq!(real.len(), 1, "two complementary 16-row remainders must produce exactly one tile");
    let i = real[0];
    assert_eq!(ca[i], 16);
    assert_eq!(cb[i], 16);
    assert_ne!(ea[i], eb[i], "the paired tile's two halves must be DIFFERENT experts");
}

#[test]
fn device_handles_unbalanced_pairs_and_odd_leftover() { assert_matches_candidates(&[3, 13, 1], 3); }

#[test]
fn seventeen_to_thirty_one_remainder_produces_no_candidate() {
    // 25-row remainder (17..=31) is OUT OF SCOPE for this kernel - the
    // existing unpaired builder/kernel handles it, unchanged, via
    // own_counts. Only the second expert's 5-row remainder is a
    // candidate here (odd leftover, no partner).
    let (ea, _sa, ca, _eb, _sb, cb, _cap) = run_case(&[25, 5], 2);
    let real: Vec<usize> = (0..ea.len()).filter(|&i| ca[i] > 0).collect();
    assert_eq!(real.len(), 1, "only expert 1's 5-row remainder is a candidate");
    assert_eq!(ea[real[0]], 1);
    assert_eq!(ca[real[0]], 5);
    assert_eq!(cb[real[0]], 0, "odd leftover: inert B half");
}

#[test]
fn matches_candidates_at_realistic_scale() {
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
    assert_matches_candidates(&counts, 256);
}

#[test]
fn matches_candidates_on_zipfish_realistic_fixture() {
    let counts = wh_iron_std::kernels::moe::moe_mpp_shared::zipfish_counts(512, 256, 0x9E37_0001);
    assert_matches_candidates(&counts, 256);
}

/// Oracle for the indirect-dispatch pair count (F-85 follow-up): the
/// device-emitted `pair_count_gateup`/`pair_count_down` must equal the
/// REAL tile count (`ca.iter().filter(|&&c| c > 0).count()`), not the
/// worst-case `capacity` this dispatch was sized to - that gap is exactly
/// the padding the indirect-dispatch fix is meant to eliminate. Both
/// outputs must agree with each other (they carry the same value into two
/// differently-shaped indirect-args triples).
fn assert_pair_count_matches_real_tiles(counts: &[usize], n_experts: usize) {
    let (_ea, _sa, ca, _eb, _sb, _cb, capacity, gateup, down) =
        run_case_with_pair_counts(counts, n_experts);
    let real_tiles = ca.iter().filter(|&&c| c > 0).count() as u32;
    assert_eq!(gateup, real_tiles, "pair_count_gateup vs real tile count; counts={counts:?}");
    assert_eq!(down, real_tiles, "pair_count_down vs real tile count; counts={counts:?}");
    assert!(
        (real_tiles as usize) <= capacity,
        "real tile count must never exceed the worst-case capacity; counts={counts:?}"
    );
}

#[test]
fn pair_count_zero_when_no_candidates() {
    // Every count is either 0 or an exact multiple of 32 - no 1..=16
    // remainder anywhere, so the real paired-tile count must be 0 (not the
    // worst-case capacity every prior direct-dispatch call used to pay for).
    assert_pair_count_matches_real_tiles(&[0, 32, 64, 0, 96], 5);
}

#[test]
fn pair_count_matches_on_skewed_ragged_routing() {
    assert_pair_count_matches_real_tiles(&[1, 0, 47, 0, 5, 22, 0, 3, 16, 9], 10);
}

#[test]
fn pair_count_matches_on_odd_leftover() { assert_pair_count_matches_real_tiles(&[3, 13, 1], 3); }

#[test]
fn pair_count_matches_on_every_expert_a_candidate() {
    assert_pair_count_matches_real_tiles(&[1; 32], 32);
}

#[test]
fn pair_count_matches_at_realistic_scale() {
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
    assert_pair_count_matches_real_tiles(&counts, 256);
}

#[test]
fn pair_count_matches_on_zipfish_realistic_fixture() {
    let counts = wh_iron_std::kernels::moe::moe_mpp_shared::zipfish_counts(512, 256, 0x9E37_0001);
    assert_pair_count_matches_real_tiles(&counts, 256);
}

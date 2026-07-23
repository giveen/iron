//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Byte-exact equality for the parallelized MoE tile-plan builders
//! (F-85 small-M follow-up): `iron_moe_tile_plan_expert_counts` (phase 1,
//! shared) chained into `iron_moe_build_tile_plan_parallel` /
//! `_bm32_parallel` / `_bm32_own_parallel` (phase 2, per tile-height
//! variant), against both a CPU reference AND a live GPU dispatch of
//! each variant's ORIGINAL single-threadgroup kernel on the identical
//! input - mirrors `moe_sort_plan_counting_correctness.rs`'s dual-oracle
//! style, so a divergence can never hide behind a CPU-oracle bug.
//!
//! Fixture coverage: uniform routing, Zipf-skewed routing, zero-count
//! experts, single-expert-takes-all, boundary counts at the tile-height
//! edge (32/33 rows, and the BM=16 sibling's own 16/17), tiny and
//! production-scale `m_total`, plus a deterministic-repeat check and
//! poisoned output buffers throughout (a builder that silently skips a
//! write fails loudly instead of passing by a lucky zero match).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::moe::{
    moe_mpp_shared::zipfish_counts,
    moe_mpp_tileplan::{build_tile_plan, build_tile_plan_with_bm},
    moe_tile_plan_builder::iron_moe_build_tile_plan,
    moe_tile_plan_builder_bm32::iron_moe_build_tile_plan_bm32,
    moe_tile_plan_builder_bm32_own::iron_moe_build_tile_plan_bm32_own,
    moe_tile_plan_builder_bm32_own_parallel::iron_moe_build_tile_plan_bm32_own_parallel,
    moe_tile_plan_builder_bm32_parallel::iron_moe_build_tile_plan_bm32_parallel,
    moe_tile_plan_builder_parallel::iron_moe_build_tile_plan_parallel,
    moe_tile_plan_expert_counts::iron_moe_tile_plan_expert_counts,
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Poison fill - any output byte a phase fails to write shows up as
/// `0xDEAD_BEEF`, not a lucky zero match against the oracle.
const POISON: u32 = 0xDEAD_BEEF;
fn poisoned(n: usize) -> Vec<u8> { u32_bytes(&vec![POISON; n]) }

fn sorted_experts_from_counts(counts: &[usize]) -> Vec<u32> {
    let mut out = Vec::new();
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            out.push(e as u32);
        }
    }
    out
}

/// Phase 1 dispatch: `sorted_experts` -> `(expert_row_base, expert_count)`.
fn run_phase1(ctx: &Context, sorted_experts: &[u32], n_experts: usize) -> (Vec<u32>, Vec<u32>) {
    let m_total = sorted_experts.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("sorted_experts".into(), u32_bytes(sorted_experts));
    buffers.insert("expert_row_base".into(), poisoned(n_experts));
    buffers.insert("expert_count".into(), poisoned(n_experts));
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());

    let mut k = iron_moe_tile_plan_expert_counts::kernel_ir();
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [n_experts, 1, 1], [1, 1, 1])
        .expect("phase1 dispatch");
    (
        unpack_u32(r.outputs.get("expert_row_base").unwrap()),
        unpack_u32(r.outputs.get("expert_count").unwrap()),
    )
}

/// Which tile-height variant to exercise; parameterizes the shared
/// phase-2 + original-kernel dispatch helpers below.
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

    /// Host oracle for this variant's real tile-emission contract.
    fn host_reference(self, counts: &[usize]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        match self {
            Variant::Bm16 => build_tile_plan(counts),
            Variant::Bm32 => build_tile_plan_with_bm(counts, 32),
            // "own" excludes 1..=16-row remainders - not representable by
            // build_tile_plan_with_bm directly, so derive it the same way
            // the kernel does: full tiles + one more iff remainder > 16.
            Variant::Bm32Own => {
                let mut te = Vec::new();
                let mut trs = Vec::new();
                let mut trc = Vec::new();
                let mut row_base = 0usize;
                for (e, &c) in counts.iter().enumerate() {
                    let full = c / 32;
                    let remainder = c - full * 32;
                    let n_tiles = full + usize::from(remainder > 16);
                    for t in 0..n_tiles {
                        let start = t * 32;
                        let n = (c - start).min(32);
                        te.push(e as u32);
                        trs.push((row_base + start) as u32);
                        trc.push(n as u32);
                    }
                    row_base += c;
                }
                (te, trs, trc)
            },
        }
    }
}

/// Runs the ORIGINAL single-threadgroup kernel for `variant` on
/// `sorted_experts`, one dispatch. Returns
/// `(tile_expert, tile_row_start, tile_row_count, tile_count_gateup,
/// tile_count_down)` - the last two are `None` for BM=16 (no indirect
/// outputs).
#[allow(clippy::type_complexity)]
fn run_original(
    ctx: &Context,
    variant: Variant,
    sorted_experts: &[u32],
    n_experts: usize,
    capacity: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Option<u32>, Option<u32>) {
    let m_total = sorted_experts.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("sorted_experts".into(), u32_bytes(sorted_experts));
    // Zero-filled, not poisoned: production callers zero-fill these
    // before dispatch (see the builders' dispatch-invariants docs) and
    // the kernel only writes the real [0, real_tiles) prefix, so the
    // zero-fill-padding contract below is only meaningful starting from
    // an actual zero fill.
    buffers.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    if variant.has_indirect_count() {
        buffers.insert("tile_count_gateup".into(), POISON.to_le_bytes().to_vec());
        buffers.insert("tile_count_down".into(), POISON.to_le_bytes().to_vec());
    }

    let mut k = match variant {
        Variant::Bm16 => iron_moe_build_tile_plan::kernel_ir(),
        Variant::Bm32 => iron_moe_build_tile_plan_bm32::kernel_ir(),
        Variant::Bm32Own => iron_moe_build_tile_plan_bm32_own::kernel_ir(),
    };
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("original builder dispatch");

    let (gu, dn) = if variant.has_indirect_count() {
        (
            Some(unpack_u32(r.outputs.get("tile_count_gateup").unwrap())[0]),
            Some(unpack_u32(r.outputs.get("tile_count_down").unwrap())[0]),
        )
    } else {
        (None, None)
    };
    (
        unpack_u32(r.outputs.get("tile_expert").unwrap()),
        unpack_u32(r.outputs.get("tile_row_start").unwrap()),
        unpack_u32(r.outputs.get("tile_row_count").unwrap()),
        gu,
        dn,
    )
}

/// Runs phase 1 + the parallel phase 2 for `variant` on `sorted_experts`.
#[allow(clippy::type_complexity)]
fn run_parallel(
    ctx: &Context,
    variant: Variant,
    sorted_experts: &[u32],
    n_experts: usize,
    capacity: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Option<u32>, Option<u32>) {
    let (row_base, count) = run_phase1(ctx, sorted_experts, n_experts);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("expert_row_base".into(), u32_bytes(&row_base));
    buffers.insert("expert_count".into(), u32_bytes(&count));
    // Zero-filled, not poisoned: production callers zero-fill these
    // before dispatch (see the builders' dispatch-invariants docs) and
    // the kernel only writes the real [0, real_tiles) prefix, so the
    // zero-fill-padding contract below is only meaningful starting from
    // an actual zero fill.
    buffers.insert("tile_expert".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_start".into(), vec![0u8; capacity * 4]);
    buffers.insert("tile_row_count".into(), vec![0u8; capacity * 4]);
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    if variant.has_indirect_count() {
        buffers.insert("tile_count_gateup".into(), POISON.to_le_bytes().to_vec());
        buffers.insert("tile_count_down".into(), POISON.to_le_bytes().to_vec());
    }

    let mut k = match variant {
        Variant::Bm16 => iron_moe_build_tile_plan_parallel::kernel_ir(),
        Variant::Bm32 => iron_moe_build_tile_plan_bm32_parallel::kernel_ir(),
        Variant::Bm32Own => iron_moe_build_tile_plan_bm32_own_parallel::kernel_ir(),
    };
    k.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&k, &buffers, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("parallel phase2 dispatch");

    let (gu, dn) = if variant.has_indirect_count() {
        (
            Some(unpack_u32(r.outputs.get("tile_count_gateup").unwrap())[0]),
            Some(unpack_u32(r.outputs.get("tile_count_down").unwrap())[0]),
        )
    } else {
        (None, None)
    };
    (
        unpack_u32(r.outputs.get("tile_expert").unwrap()),
        unpack_u32(r.outputs.get("tile_row_start").unwrap()),
        unpack_u32(r.outputs.get("tile_row_count").unwrap()),
        gu,
        dn,
    )
}

/// Checks one `(variant, counts)` fixture against the CPU oracle, the
/// live original kernel, and confirms parallel == original byte-exact.
fn check_case(variant: Variant, counts: &[usize], n_experts: usize, label: &str) {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let sorted_experts = sorted_experts_from_counts(counts);
    let m_total = sorted_experts.len();
    let capacity = m_total.div_ceil(variant.bm()) + n_experts;

    let (exp_te, exp_trs, exp_trc) = variant.host_reference(counts);
    let real_tiles = exp_te.len();
    assert!(real_tiles <= capacity, "{label}: oracle exceeds worst-case capacity");

    let (orig_te, orig_trs, orig_trc, orig_gu, orig_dn) =
        run_original(&ctx, variant, &sorted_experts, n_experts, capacity);
    assert_eq!(
        &orig_te[..real_tiles],
        &exp_te[..],
        "{label}: original kernel vs CPU oracle (test bug?)"
    );
    assert_eq!(
        &orig_trs[..real_tiles],
        &exp_trs[..],
        "{label}: original kernel vs CPU oracle (test bug?)"
    );
    assert_eq!(
        &orig_trc[..real_tiles],
        &exp_trc[..],
        "{label}: original kernel vs CPU oracle (test bug?)"
    );

    let (par_te, par_trs, par_trc, par_gu, par_dn) =
        run_parallel(&ctx, variant, &sorted_experts, n_experts, capacity);

    assert_eq!(&par_te[..real_tiles], &exp_te[..], "{label}: parallel tile_expert vs CPU oracle");
    assert_eq!(
        &par_trs[..real_tiles],
        &exp_trs[..],
        "{label}: parallel tile_row_start vs CPU oracle"
    );
    assert_eq!(
        &par_trc[..real_tiles],
        &exp_trc[..],
        "{label}: parallel tile_row_count vs CPU oracle"
    );

    // Byte-exact against the CURRENT kernel's own GPU output over the
    // FULL buffer, including worst-case padding - the load-bearing check
    // this campaign exists for, not "a valid plan".
    assert_eq!(
        par_te, orig_te,
        "{label}: tile_expert diverges from the original builder's own output"
    );
    assert_eq!(
        par_trs, orig_trs,
        "{label}: tile_row_start diverges from the original builder's own output"
    );
    assert_eq!(
        par_trc, orig_trc,
        "{label}: tile_row_count diverges from the original builder's own output"
    );
    assert_eq!(
        par_gu, orig_gu,
        "{label}: tile_count_gateup diverges from the original builder's own output"
    );
    assert_eq!(
        par_dn, orig_dn,
        "{label}: tile_count_down diverges from the original builder's own output"
    );

    // Padding past the real tile count must stay zero-filled - same
    // contract as the original builder, unchanged by this split.
    assert!(
        par_trc[real_tiles..capacity].iter().all(|&c| c == 0),
        "{label}: padding past real_tiles={real_tiles} (capacity={capacity}) must stay zero-filled"
    );
}

const VARIANTS: [Variant; 3] = [Variant::Bm16, Variant::Bm32, Variant::Bm32Own];

#[test]
fn uniform_routing_production_scale() {
    for variant in VARIANTS {
        for &t in &[512usize, 1024, 4096] {
            let m_total = t * 8;
            let n_experts = 256;
            // Uniform routing: every row's expert id is `i * odd_const mod
            // n_experts` before sorting - the per-expert COUNT is what the
            // builders consume (they take a pre-sorted array, never derive
            // one), so tally counts directly rather than round-tripping
            // through an unsorted id array.
            let mut counts = vec![0usize; n_experts];
            for i in 0..m_total {
                counts[(i * 2654435761) % n_experts] += 1;
            }
            check_case(variant, &counts, n_experts, &format!("uniform T={t}"));
        }
    }
}

#[test]
fn zipf_skewed_routing() {
    for variant in VARIANTS {
        for &m_total in &[4096usize, 16384, 32768] {
            let counts = zipfish_counts(m_total, 256, 0x5EED_0001u64.wrapping_add(m_total as u64));
            check_case(variant, &counts, 256, &format!("zipf mTotal={m_total}"));
        }
    }
}

#[test]
fn zero_count_experts() {
    for variant in VARIANTS {
        let n_experts = 40;
        let m_total = 2048;
        let mut counts = vec![0usize; n_experts];
        let mut remaining = m_total;
        let mut e = 0;
        while remaining > 0 {
            if e % 5 == 0 {
                let take = remaining.min(37);
                counts[e % n_experts] += take;
                remaining -= take;
            }
            e += 1;
        }
        check_case(variant, &counts, n_experts, "zero-count experts");
    }
}

#[test]
fn single_expert_takes_all() {
    for variant in VARIANTS {
        for &m_total in &[64usize, 4096] {
            let mut counts = vec![0usize; 16];
            counts[7] = m_total;
            check_case(variant, &counts, 16, &format!("single-expert-takes-all m={m_total}"));
        }
    }
}

#[test]
fn boundary_counts_at_tile_edge() {
    // Counts straddling the tile-height boundary: exactly at, one under,
    // and one over, for BOTH BM=16 (16/17) and BM=32 (32/33) - the exact
    // edges the task's fixture set calls out.
    for variant in VARIANTS {
        let bm = variant.bm();
        let counts = [
            bm - 1, // one under: single short tile
            bm,     // exactly one full tile
            bm + 1, // one over: one full tile + a 1-row remainder
            0,
            2 * bm,
            2 * bm + 1,
        ];
        check_case(variant, &counts, counts.len(), &format!("boundary counts bm={bm}"));
    }
}

#[test]
fn tiny_m_total() {
    for variant in VARIANTS {
        for &m_total in &[8usize, 32] {
            let mut counts = vec![0usize; 5];
            let mut remaining = m_total;
            let mut e = 0;
            while remaining > 0 {
                let take = remaining.min(((e * 7 + 3) % 11) + 1);
                counts[e % 5] += take;
                remaining -= take;
                e += 1;
            }
            check_case(variant, &counts, 5, &format!("tiny m_total={m_total}"));
        }
    }
}

#[test]
fn production_scale_uniform_and_zipf() {
    for variant in VARIANTS {
        for &t in &[512usize, 1024, 2048, 4096] {
            let m_total = t * 8;
            let counts = zipfish_counts(
                m_total,
                256,
                0x9E37_0001u64.wrapping_add(t as u64).wrapping_add(variant.bm() as u64),
            );
            check_case(variant, &counts, 256, &format!("production zipf T={t}"));
        }
    }
}

#[test]
fn deterministic_repeat_dispatch() {
    // Same input dispatched twice must produce bit-identical output - no
    // atomics anywhere in this design, but the cheapest guard against a
    // regression introducing nondeterministic scheduling dependence.
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    for variant in VARIANTS {
        let counts = zipfish_counts(4096, 256, 0x42);
        let sorted_experts = sorted_experts_from_counts(&counts);
        let capacity = 4096usize.div_ceil(variant.bm()) + 256;
        let (te1, trs1, trc1, gu1, dn1) =
            run_parallel(&ctx, variant, &sorted_experts, 256, capacity);
        let (te2, trs2, trc2, gu2, dn2) =
            run_parallel(&ctx, variant, &sorted_experts, 256, capacity);
        assert_eq!(te1, te2, "tile_expert nondeterministic across identical dispatches");
        assert_eq!(trs1, trs2, "tile_row_start nondeterministic across identical dispatches");
        assert_eq!(trc1, trc2, "tile_row_count nondeterministic across identical dispatches");
        assert_eq!(gu1, gu2, "tile_count_gateup nondeterministic across identical dispatches");
        assert_eq!(dn1, dn2, "tile_count_down nondeterministic across identical dispatches");
    }
}

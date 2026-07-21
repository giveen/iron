//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Device-side PAIR-ONLY tile-plan builder for `ffai_moe_gather_qmm_coop_paired`
//! (`moe_gather_qmm_coop_paired.rs`). `ffai_moe_build_tile_plan_bm32_paired`.
//!
//! Device mirror of `split_for_pairing` in `moe_tile_plan_builder_paired.rs`;
//! see that module's doc comment for why this is a SEPARATE dispatch
//! (feeding only the `ffai_moe_gather_qmm_coop_paired` kernel) rather than
//! folding into the existing `ffai_moe_build_tile_plan_bm32` device
//! builder: the two kernels serve disjoint row ranges (this one only the
//! `1..=16`-row pairing candidates; the existing builder's consumer
//! handles everything else via `own_counts`, computed on the host from
//! the SAME routing decision this kernel reads from `sorted_experts`; see
//! the dispatch invariants below for how the two stay consistent without
//! a host readback). Same one-threadgroup/one-lane-per-expert binary
//! search design as the unpaired sibling, with a single prefix-sum pass
//! (rather than that kernel's tile-offset scan) to compact and pair the
//! candidates: still ONE threadgroup, ONE barrier, no host readback of
//! routing counts.
//!
//! ## Algorithm
//!
//! Per lane (`tid = e`):
//!
//! 1. Binary-search `sorted_experts` for `count` and `row_base` -
//!    identical to the unpaired sibling.
//! 2. `remainder = count % 32`. `is_candidate = 1` iff `1 <= remainder <=
//!    16`. Write `is_candidate` to threadgroup memory, barrier.
//! 3. Scan `is_candidate` over all experts (`n_experts`-length, same
//!    O(n_experts)-per-lane style the sibling's prefix sum uses):
//!    `short_prefix` = exclusive prefix sum over experts `< e`
//!    (ascending-expert-order compaction position); `short_total` =
//!    ungated grand total (same value every lane, unused directly but
//!    documents the codomain of `short_prefix`).
//! 4. If `is_candidate`: `pair_idx = short_prefix / 2`, writes into slot A
//!    (`short_prefix` even) or slot B (`short_prefix` odd) of tile
//!    `pair_idx`. TWO different lanes write the two halves of the same
//!    paired tile; since they write disjoint fields (`_a` vs `_b`) this
//!    is race-free with no extra barrier. An odd-sized candidate list's
//!    last candidate always lands in slot A with no B partner - see the
//!    zero-fill note below for why that is safe. Flattened to two
//!    SEPARATE top-level `if`s gated by pre-multiplied 0/1 flags
//!    (`write_a`/`write_b`), NOT one `if is_candidate { if a_slot {...}
//!    else {...} }` - nesting an if/else whose two arms store to
//!    DIFFERENT tensors directly inside another `if` was found
//!    (empirically, via this kernel's own device oracle while developing
//!    it) to drop the inner branch's guard in codegen, executing BOTH
//!    arms' stores unconditionally instead of the intended one. This is a
//!    real codegen limitation worth flagging on its own, independent of
//!    this kernel - see the PR description for a minimal repro.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256`, same scope as the unpaired sibling.
//! - Caller pre-sizes all six tile buffers to
//!   `paired_tile_plan_worst_case_capacity(n_experts)`
//!   (`moe_tile_plan_builder_paired.rs`: `ceil(n_experts / 2)`) and
//!   ZERO-FILLS them before this dispatch - required for the
//!   odd-leftover-candidate case above (`tile_expert_b = 0`,
//!   `tile_row_start_b = 0`, `tile_row_count_b = 0` is a valid,
//!   fully-masked B half), not merely a courtesy for unused tail
//!   capacity.
//! - The companion "own" dispatch (`ffai_moe_gather_qmm_coop` fed by
//!   `ffai_moe_build_tile_plan_bm32` over `own_counts`) MUST use the SAME
//!   `sorted_experts` array this kernel reads - the two kernels partition
//!   one routing decision's rows between them without either side reading
//!   the other's output, so a mismatched `sorted_experts` between the two
//!   dispatches would silently double-count or drop rows.
//!
//! ## Indirect-dispatch pair count (F-85 follow-up)
//!
//! Two extra scalar outputs, `pair_count_gateup` / `pair_count_down`, feed
//! `MTLDispatchThreadgroupsIndirectArguments` for the paired GEMM's
//! gate/up and down dispatches respectively (the two legs differ only in
//! `n_out`, so each needs its own indirect-args triple; the caller writes
//! the `n_out_tiles`/`z=1` words of each triple once, host-side, at
//! buffer-alloc time - only the middle word, the real paired-tile count,
//! is device-computed here). Both receive the SAME value
//! (`ceil(total_candidates / 2)`), written by exactly one lane
//! (`e == n_experts - 1`) using that lane's already-computed
//! `short_prefix` (an EXCLUSIVE prefix sum over experts `< e`) plus its own
//! `is_candidate`: at the last lane this sum equals the INCLUSIVE total
//! over every expert, with no extra scan needed. A single flat `if`
//! (not nested inside the `write_a`/`write_b` gates above, and not an
//! if/else) storing to two DIFFERENT tensors - same shape as those two
//! gates' own workaround for the nested-if-else codegen bug documented
//! above, so it does not reintroduce it.

use ffai_kernels::kernel;

/// Device-side PAIR-ONLY MoE tile-plan builder, BM=32. See module docs
/// for the algorithm and the zero-fill dispatch contract this pairs with.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_build_tile_plan_bm32_paired(
    sorted_experts: Tensor<u32>,
    mut tile_expert_a: Tensor<u32>,
    mut tile_row_start_a: Tensor<u32>,
    mut tile_row_count_a: Tensor<u32>,
    mut tile_expert_b: Tensor<u32>,
    mut tile_row_start_b: Tensor<u32>,
    mut tile_row_count_b: Tensor<u32>,
    mut pair_count_gateup: Tensor<u32>,
    mut pair_count_down: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] n_experts: u32,
) {
    let e = tid;
    // Capacity fixed at 256: Qwen3.6-35B-A3B's expert count, same as the
    // unpaired sibling.
    threadgroup_alloc("is_candidate_pe", 256u32, u32);

    // lower_bound(e) - first index in sorted_experts whose value is >= e.
    let mut lo0 = 0u32;
    let mut hi0 = m_total;
    for _it in range(0u32, 22u32, 1u32) {
        if lo0 < hi0 {
            let mid = (lo0 + hi0) / 2u32;
            let v = load(sorted_experts[mid]);
            if v < e {
                lo0 = mid + 1u32;
            } else {
                hi0 = mid;
            }
        }
    }
    let row_base = lo0;

    // lower_bound(e + 1) == upper_bound(e) - same search, shifted target.
    let e_next = e + 1u32;
    let mut lo1 = 0u32;
    let mut hi1 = m_total;
    for _it in range(0u32, 22u32, 1u32) {
        if lo1 < hi1 {
            let mid = (lo1 + hi1) / 2u32;
            let v = load(sorted_experts[mid]);
            if v < e_next {
                lo1 = mid + 1u32;
            } else {
                hi1 = mid;
            }
        }
    }
    let row_end = lo1;

    let count = row_end - row_base;
    let full = count / 32u32;
    let remainder = count - full * 32u32; // 0..=31
    let is_candidate = select(remainder == 0u32, 0u32, select(remainder > 16u32, 0u32, 1u32)); // 1..=16

    threadgroup_store("is_candidate_pe", e, is_candidate);
    threadgroup_barrier();

    // Single n_experts-length scan: exclusive prefix sum of is_candidate
    // over experts < e - this lane's compaction position.
    let mut short_prefix = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        let sc = threadgroup_load("is_candidate_pe", e2);
        if e2 < e {
            short_prefix = short_prefix + sc;
        }
    }

    // This expert's 1..16-row pairing candidate, if any - lands in slot A
    // or B of a tile shared with a DIFFERENT expert's candidate (or, for
    // an odd leftover, alone in slot A - see module doc step 4). Two flat
    // top-level `if`s, not a nested if/else - see module doc for why.
    let start = row_base + full * 32u32;
    let idx = short_prefix / 2u32;
    let a_slot = select((short_prefix & 1u32) == 0u32, 1u32, 0u32);
    let write_a = is_candidate * a_slot;
    let write_b = is_candidate * (1u32 - a_slot);
    if write_a == 1u32 {
        store(tile_expert_a[idx], e);
        store(tile_row_start_a[idx], start);
        store(tile_row_count_a[idx], remainder);
    }
    if write_b == 1u32 {
        store(tile_expert_b[idx], e);
        store(tile_row_start_b[idx], start);
        store(tile_row_count_b[idx], remainder);
    }

    // Indirect-dispatch pair count - see module doc. `short_prefix` at the
    // LAST lane is the exclusive sum over all experts < (n_experts - 1);
    // adding this lane's own `is_candidate` makes it the inclusive total
    // over every expert, i.e. `total_candidates`.
    let total_candidates = short_prefix + is_candidate;
    let is_last_lane = select(e == (n_experts - 1u32), 1u32, 0u32);
    if is_last_lane == 1u32 {
        let pair_count = (total_candidates + 1u32) / 2u32;
        store(pair_count_gateup[0], pair_count);
        store(pair_count_down[0], pair_count);
    }
}

#[cfg(test)]
mod tests {
    use ffai_kernels::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_without_inline_msl() {
        let k = ffai_moe_build_tile_plan_bm32_paired::kernel_ir();
        assert_eq!(k.name, "ffai_moe_build_tile_plan_bm32_paired");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        let _ = DType::F32;
    }
}

/// Bench registration - required for `ffaik build` to discover this
/// kernel; see the unpaired sibling's identical note.
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_moe_build_tile_plan_bm32_paired;
    use crate::kernels::moe::moe_tile_plan_builder_paired::paired_tile_plan_worst_case_capacity;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Realistic scale (T=512 * top-8, 256 experts) skewed-ish fixture -
    /// same shape as the unpaired sibling's bench.
    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan_bm32_paired(_dt: DType) -> BenchSetup {
        let n_experts = 256usize;
        let m_total = 4096usize;
        let mut sorted_experts = Vec::with_capacity(m_total);
        let mut rows_left = m_total;
        let mut e = 0u32;
        while rows_left > 0 && (e as usize) < n_experts {
            let run = (rows_left).min(((e as usize * 37 + 5) % 64) + 1);
            for _ in 0..run {
                sorted_experts.push(e);
            }
            rows_left -= run;
            e += 1;
        }
        while sorted_experts.len() < m_total {
            sorted_experts.push((n_experts - 1) as u32);
        }
        let capacity = paired_tile_plan_worst_case_capacity(n_experts);

        BenchSetup::new(ffai_moe_build_tile_plan_bm32_paired::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("sorted_experts", u32_bytes(&sorted_experts), DType::U32))
            .buffer(BenchBuffer::zeros("tile_expert_a", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_start_a", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_count_a", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_expert_b", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_start_b", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_count_b", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("pair_count_gateup", 1, DType::U32).output())
            .buffer(BenchBuffer::zeros("pair_count_down", 1, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_experts", n_experts as u32)
            .with_shape_label(format!("M{m_total} E{n_experts} cap{capacity}"))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((m_total * 4 + capacity * 4 * 6) as u64)
    }
}

// GPU correctness for this kernel lives in
// `tests/moe_build_tile_plan_bm32_paired_correctness.rs` (an integration
// test, not a `kernel_tests` submodule here) so it can use the raw
// `Context::dispatch_with_grid` path the unpaired sibling's own
// `tests/moe_build_tile_plan_correctness.rs` uses - this kernel is u32-only
// (no `T` generic), so it is not a `#[test_kernel]` dtype-matrix case.

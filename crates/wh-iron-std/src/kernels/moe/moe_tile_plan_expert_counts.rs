//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Phase 1 of the parallelized MoE tile-plan builders (F-85 small-M
//! follow-up). `iron_moe_tile_plan_expert_counts`.
//!
//! The existing tile-plan builders (`iron_moe_build_tile_plan`,
//! `_bm32`, `_bm32_own` - see their module docs) each dispatch ONE
//! threadgroup for the WHOLE plan: every expert's binary search, the
//! prefix sum over all experts, and every tile emission all run on a
//! single GPU core. That per-layer fixed cost does not shrink with
//! `m_total`, so it is disproportionately expensive at low prefill `T`.
//!
//! This file splits out the one part of that work that is genuinely
//! parallel across experts and independent of the OTHER experts'
//! results: finding each expert's contiguous row range in
//! `sorted_experts`. `iron_moe_tile_plan_expert_counts` dispatches ONE
//! THREADGROUP PER EXPERT (grid `[n_experts, 1, 1]`, one thread per
//! group) so all `n_experts` binary searches run concurrently across
//! the GPU's cores instead of serially on one core - the same
//! one-threadgroup-per-expert idiom `iron_moe_gather_qmm_mma_eg_*`
//! (`moe_mpp_expert_grid.rs`) already uses for its per-expert run scan.
//!
//! The remaining work - the prefix sum over experts and the tile
//! emission loop - stays a single small threadgroup (see
//! `moe_tile_plan_builder_parallel.rs` / `_bm32_parallel.rs` /
//! `_bm32_own_parallel.rs`, this file's phase 2 siblings): genuinely
//! tiny next to the binary search, same two-phase count/emit split
//! `iron_moe_sort_plan_hist` -> `iron_moe_sort_plan_offsets` already
//! uses for the sort-plan kernel (`moe_sort_plan_counting.rs`).
//!
//! Identical binary search to every existing builder's per-lane code -
//! this file only moves WHERE it runs (one lane per threadgroup, not
//! one lane among `n_experts` sharing a threadgroup), never changes
//! the arithmetic, so `expert_row_base[e]`/`expert_count[e]` are
//! exactly what any existing builder's `row_base`/`count` locals would
//! be for that `e`. Shared by ALL THREE tile-height variants (BM=16,
//! BM=32, BM=32 "own") - the row range a sorted-expert array implies
//! for expert `e` does not depend on the tile height the caller will
//! slice it into; only phase 2 differs per variant.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_experts, 1, 1]` (one threadgroup PER
//!   EXPERT); threadgroup `[1, 1, 1]` (one thread per group).
//! - `sorted_experts` length `m_total`, ascending expert-sorted (same
//!   contract as every builder this feeds).
//! - `expert_row_base`/`expert_count` length `n_experts` (<= 256, same
//!   scope limit as the single-threadgroup builders), fully
//!   overwritten by this dispatch - no zero-fill precondition, unlike
//!   the tile-plan output buffers (every entry in `[0, n_experts)` gets
//!   written by exactly one threadgroup).

use wh_iron::kernel;

/// Phase 1: per-expert row range, one threadgroup per expert. See module
/// docs for the algorithm and why this is shared across all tile-height
/// variants.
#[kernel]
pub fn iron_moe_tile_plan_expert_counts(
    sorted_experts: Tensor<u32>,
    mut expert_row_base: Tensor<u32>,
    mut expert_count: Tensor<u32>,
    #[constexpr] m_total: u32,
) {
    let e = tgid_x;

    // lower_bound(e) - first index in sorted_experts whose value is >= e.
    // Identical to every existing builder's per-lane search, just indexed
    // by this threadgroup's grid position instead of its lane id.
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

    store(expert_row_base[e], row_base);
    store(expert_count[e], row_end - row_base);
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_without_inline_msl() {
        let k = iron_moe_tile_plan_expert_counts::kernel_ir();
        assert_eq!(k.name, "iron_moe_tile_plan_expert_counts");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        let _ = DType::F32;
    }
}

/// Bench registration - required for `iron build` to discover this
/// kernel; see the single-threadgroup builders' identical note.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_tile_plan_expert_counts;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[bench(dtypes = [f32])]
    fn bench_moe_tile_plan_expert_counts(_dt: DType) -> BenchSetup {
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

        BenchSetup::new(iron_moe_tile_plan_expert_counts::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("sorted_experts", u32_bytes(&sorted_experts), DType::U32))
            .buffer(BenchBuffer::zeros("expert_row_base", n_experts, DType::U32).output())
            .buffer(BenchBuffer::zeros("expert_count", n_experts, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .with_shape_label(format!("M{m_total} E{n_experts}"))
            .grid_3d(n_experts as u32, 1, 1, [1, 1, 1])
            .bytes_moved((m_total * 4 + n_experts * 4 * 2) as u64)
    }
}

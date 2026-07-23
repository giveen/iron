//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Phase 2 of the parallelized BM=16 tile-plan builder (F-85 small-M
//! follow-up). `iron_moe_build_tile_plan_parallel`.
//!
//! Pairs with `iron_moe_tile_plan_expert_counts`
//! (`moe_tile_plan_expert_counts.rs`, phase 1), which computes each
//! expert's `row_base`/`count` across `n_experts` parallel
//! threadgroups. This phase takes those per-expert results and does
//! the two things that genuinely need one expert's data next to
//! another's: the prefix sum over experts (to find each expert's
//! tile-write offset) and the tile emission itself. One threadgroup,
//! `n_experts` lanes - same shape as `iron_moe_build_tile_plan`
//! (`moe_tile_plan_builder.rs`) had for its WHOLE job, but now this is
//! all that is left in it: no per-lane binary search over
//! `sorted_experts` (`O(m_total)` per lane, gone - moved to phase 1),
//! just an `O(n_experts)` prefix sum and each lane's own
//! `O(ceil(count/16))` emission loop.
//!
//! Output format is IDENTICAL to `iron_moe_build_tile_plan`: same
//! `tile_expert`/`tile_row_start`/`tile_row_count` entries, same
//! ascending-by-expert concatenation order, same worst-case-capacity +
//! zero-fill padding contract. Only the WAY those entries get computed
//! changed (chained two-dispatch, not one) - see
//! `moe_tile_plan_parallel_correctness.rs`, which checks this against
//! both a CPU oracle and a live dispatch of the original
//! single-threadgroup kernel on the same input.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256`, same scope limit as the original builder.
//! - `expert_row_base`/`expert_count` are phase 1's outputs for the
//!   SAME `sorted_experts` array and `n_experts`; this kernel does not
//!   re-derive them.
//! - Caller pre-sizes and zero-fills `tile_expert`/`tile_row_start`/
//!   `tile_row_count` to the same worst-case capacity
//!   (`ceil(m_total/16) + n_experts`) as the original builder - same
//!   padding contract, unchanged by this split.

use wh_iron::kernel;

/// Phase 2: BM=16 prefix sum + tile emission. See module docs.
#[kernel]
pub fn iron_moe_build_tile_plan_parallel(
    expert_row_base: Tensor<u32>,
    expert_count: Tensor<u32>,
    mut tile_expert: Tensor<u32>,
    mut tile_row_start: Tensor<u32>,
    mut tile_row_count: Tensor<u32>,
    #[constexpr] n_experts: u32,
) {
    let e = tid;
    // Capacity fixed at 256: same scope limit as the original builder's
    // threadgroup scratch.
    threadgroup_alloc("num_tiles_pe", 256u32, u32);

    let row_base = load(expert_row_base[e]);
    let count = load(expert_count[e]);
    let num_tiles = (count + 15u32) / 16u32;
    threadgroup_store("num_tiles_pe", e, num_tiles);
    threadgroup_barrier();

    // Exclusive prefix sum over num_tiles_pe[0..n_experts) - this lane's
    // tile-write offset. Ascending-expert-order concatenation, same as
    // the original builder.
    let mut tile_offset = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        if e2 < e {
            tile_offset = tile_offset + threadgroup_load("num_tiles_pe", e2);
        }
    }

    for local in range(0u32, num_tiles, 1u32) {
        let idx = tile_offset + local;
        let start = row_base + local * 16u32;
        let remaining = count - local * 16u32;
        let rc = select(remaining < 16u32, remaining, 16u32);
        store(tile_expert[idx], e);
        store(tile_row_start[idx], start);
        store(tile_row_count[idx], rc);
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_without_inline_msl() {
        let k = iron_moe_build_tile_plan_parallel::kernel_ir();
        assert_eq!(k.name, "iron_moe_build_tile_plan_parallel");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        let _ = DType::F32;
    }
}

/// Bench registration - required for `iron build` to discover this
/// kernel; see the original builder's identical note.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_build_tile_plan_parallel;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan_parallel(_dt: DType) -> BenchSetup {
        let n_experts = 256usize;
        let m_total = 4096usize;
        let mut counts = vec![0u32; n_experts];
        let mut rows_left = m_total;
        let mut e = 0usize;
        let mut row_base = vec![0u32; n_experts];
        let mut base = 0u32;
        while rows_left > 0 && e < n_experts {
            let run = (rows_left).min(((e * 37 + 5) % 64) + 1) as u32;
            row_base[e] = base;
            counts[e] = run;
            base += run;
            rows_left -= run as usize;
            e += 1;
        }
        for row in row_base.iter_mut().skip(e) {
            *row = base;
        }
        let capacity = m_total.div_ceil(16) + n_experts;

        BenchSetup::new(iron_moe_build_tile_plan_parallel::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("expert_row_base", u32_bytes(&row_base), DType::U32))
            .buffer(BenchBuffer::from_vec("expert_count", u32_bytes(&counts), DType::U32))
            .buffer(BenchBuffer::zeros("tile_expert", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_start", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_count", capacity, DType::U32).output())
            .constexpr("n_experts", n_experts as u32)
            .with_shape_label(format!("M{m_total} E{n_experts} cap{capacity}"))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((n_experts * 4 * 2 + capacity * 4 * 3) as u64)
    }
}

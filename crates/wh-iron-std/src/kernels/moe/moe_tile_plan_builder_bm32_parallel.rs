//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Phase 2 of the parallelized BM=32 tile-plan builder (F-85 small-M
//! follow-up). `iron_moe_build_tile_plan_bm32_parallel`.
//!
//! BM=32 sibling of `iron_moe_build_tile_plan_parallel`
//! (`moe_tile_plan_builder_parallel.rs`) - same split, same phase 1
//! (`iron_moe_tile_plan_expert_counts`), only the tile-height constant
//! differs (16 -> 32), matching `iron_moe_build_tile_plan_bm32`'s
//! relationship to `iron_moe_build_tile_plan`. This is the coop-core
//! gather GEMM's default production tile-plan builder
//! (`iron_moe_gather_qmm_coop`, wired via `useCoopTilePlan` in
//! `MoELayer.swift`) - the hot path this whole split exists to speed up.
//!
//! Output format is IDENTICAL to `iron_moe_build_tile_plan_bm32`,
//! INCLUDING the `tile_count_gateup`/`tile_count_down` indirect-dispatch
//! outputs added by the real-tile-count follow-up: same real tile
//! count, written by the same last-lane derivation. Consumers
//! (`iron_moe_gather_qmm_coop`'s indirect dispatch) read these exactly
//! as before.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256`, same scope limit as the original BM=32 builder.
//! - `expert_row_base`/`expert_count` are
//!   `iron_moe_tile_plan_expert_counts`'s outputs for the SAME
//!   `sorted_experts` array and `n_experts`.
//! - Caller pre-sizes and zero-fills `tile_expert`/`tile_row_start`/
//!   `tile_row_count` to the same worst-case capacity
//!   (`ceil(m_total/32) + n_experts`) as the original BM=32 builder -
//!   same padding contract, unchanged by this split.

use wh_iron::kernel;

/// Phase 2: BM=32 prefix sum + tile emission, plus the indirect-dispatch
/// real tile count. See module docs.
#[kernel]
pub fn iron_moe_build_tile_plan_bm32_parallel(
    expert_row_base: Tensor<u32>,
    expert_count: Tensor<u32>,
    mut tile_expert: Tensor<u32>,
    mut tile_row_start: Tensor<u32>,
    mut tile_row_count: Tensor<u32>,
    mut tile_count_gateup: Tensor<u32>,
    mut tile_count_down: Tensor<u32>,
    #[constexpr] n_experts: u32,
) {
    let e = tid;
    threadgroup_alloc("num_tiles_pe", 256u32, u32);

    let row_base = load(expert_row_base[e]);
    let count = load(expert_count[e]);
    let num_tiles = (count + 31u32) / 32u32;
    threadgroup_store("num_tiles_pe", e, num_tiles);
    threadgroup_barrier();

    let mut tile_offset = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        if e2 < e {
            tile_offset = tile_offset + threadgroup_load("num_tiles_pe", e2);
        }
    }

    for local in range(0u32, num_tiles, 1u32) {
        let idx = tile_offset + local;
        let start = row_base + local * 32u32;
        let remaining = count - local * 32u32;
        let rc = select(remaining < 32u32, remaining, 32u32);
        store(tile_expert[idx], e);
        store(tile_row_start[idx], start);
        store(tile_row_count[idx], rc);
    }

    // Indirect-dispatch real tile count - same derivation as
    // `iron_moe_build_tile_plan_bm32`'s: `tile_offset` at the LAST lane is
    // the exclusive prefix sum over all experts < e; adding this lane's
    // own `num_tiles` makes it the inclusive total over every expert.
    let real_tile_count = tile_offset + num_tiles;
    let is_last_lane = select(e == (n_experts - 1u32), 1u32, 0u32);
    if is_last_lane == 1u32 {
        store(tile_count_gateup[0], real_tile_count);
        store(tile_count_down[0], real_tile_count);
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_without_inline_msl() {
        let k = iron_moe_build_tile_plan_bm32_parallel::kernel_ir();
        assert_eq!(k.name, "iron_moe_build_tile_plan_bm32_parallel");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        let _ = DType::F32;
    }
}

/// Bench registration - required for `iron build` to discover this
/// kernel; see the original BM=32 builder's identical note.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_build_tile_plan_bm32_parallel;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan_bm32_parallel(_dt: DType) -> BenchSetup {
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
        let capacity = m_total.div_ceil(32) + n_experts;

        BenchSetup::new(iron_moe_build_tile_plan_bm32_parallel::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("expert_row_base", u32_bytes(&row_base), DType::U32))
            .buffer(BenchBuffer::from_vec("expert_count", u32_bytes(&counts), DType::U32))
            .buffer(BenchBuffer::zeros("tile_expert", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_start", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_count", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_count_gateup", 1, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_count_down", 1, DType::U32).output())
            .constexpr("n_experts", n_experts as u32)
            .with_shape_label(format!("M{m_total} E{n_experts} cap{capacity}"))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((n_experts * 4 * 2 + capacity * 4 * 3) as u64)
    }
}

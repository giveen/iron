//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Phase 2 of the parallelized BM=32 "own" tile-plan builder (F-85
//! small-M follow-up). `iron_moe_build_tile_plan_bm32_own_parallel`.
//!
//! "Own" sibling of `iron_moe_build_tile_plan_bm32_parallel`
//! (`moe_tile_plan_builder_bm32_parallel.rs`) - shares the same phase 1
//! (`iron_moe_tile_plan_expert_counts`); the only difference from that
//! file is the tile-count formula, mirroring
//! `iron_moe_build_tile_plan_bm32_own`'s relationship to
//! `iron_moe_build_tile_plan_bm32`: an expert whose remainder
//! (`count % 32`) is `1..=16` rows does NOT get a tile for that
//! remainder here (those rows are pairing candidates for the separate
//! `iron_moe_build_tile_plan_bm32_paired` dispatch - see
//! `moe_tile_plan_builder_bm32_own.rs`'s module doc).
//!
//! `own_tiles(count) = count / 32 + (1 if count % 32 > 16 else 0)` -
//! identical formula to the single-threadgroup original, just applied
//! to phase 1's `expert_count[e]` instead of a locally-computed
//! `count`.
//!
//! This builder feeds the tile-pairing lever
//! (`IRON_MOE_PAIRED`), which is not yet wired into `MoELayer.swift` -
//! this parallel version is added alongside its single-threadgroup
//! sibling for when that wiring lands, so the pairing lever inherits
//! the same small-M fix the default coop path gets from
//! `iron_moe_build_tile_plan_bm32_parallel`, rather than needing a
//! second follow-up round later.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256`, same scope limit as the original "own" builder.
//! - `expert_row_base`/`expert_count` are
//!   `iron_moe_tile_plan_expert_counts`'s outputs for the SAME
//!   `sorted_experts` array and `n_experts` - must be the SAME array
//!   `iron_moe_build_tile_plan_bm32_paired` reads in the same forward,
//!   same requirement as the original "own" builder.
//! - Caller pre-sizes and zero-fills `tile_expert`/`tile_row_start`/
//!   `tile_row_count` to the SAME worst-case capacity as the unpaired
//!   sibling (`ceil(m_total/32) + n_experts`) - unchanged by this split.

use wh_iron::kernel;

/// Phase 2: BM=32 "own" prefix sum + tile emission (excludes 1..=16-row
/// pairing candidates), plus the indirect-dispatch real tile count. See
/// module docs.
#[kernel]
pub fn iron_moe_build_tile_plan_bm32_own_parallel(
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
    let full = count / 32u32;
    let remainder = count - full * 32u32; // 0..=31
    let is_solo = select(remainder > 16u32, 1u32, 0u32); // 17..=31 only
    let num_tiles = full + is_solo; // excludes 1..=16 (pairing candidates)

    threadgroup_store("num_tiles_pe", e, num_tiles);
    threadgroup_barrier();

    let mut tile_offset = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        if e2 < e {
            tile_offset = tile_offset + threadgroup_load("num_tiles_pe", e2);
        }
    }

    // Same uniform emission loop as the single-threadgroup "own"
    // builder - see that kernel's comment for why this is always a full
    // 32-row tile when `is_solo == 0`.
    for local in range(0u32, num_tiles, 1u32) {
        let idx = tile_offset + local;
        let start = row_base + local * 32u32;
        let remaining = count - local * 32u32;
        let rc = select(remaining < 32u32, remaining, 32u32);
        store(tile_expert[idx], e);
        store(tile_row_start[idx], start);
        store(tile_row_count[idx], rc);
    }

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
        let k = iron_moe_build_tile_plan_bm32_own_parallel::kernel_ir();
        assert_eq!(k.name, "iron_moe_build_tile_plan_bm32_own_parallel");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        let _ = DType::F32;
    }
}

/// Bench registration - required for `iron build` to discover this
/// kernel; see the original "own" builder's identical note.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_build_tile_plan_bm32_own_parallel;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan_bm32_own_parallel(_dt: DType) -> BenchSetup {
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

        BenchSetup::new(iron_moe_build_tile_plan_bm32_own_parallel::kernel_ir())
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

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Device-side tile-plan builder for `ffai_moe_gather_qmm_coop`
//! (`moe_gather_qmm_coop.rs`, BM=32). `ffai_moe_build_tile_plan_bm32`.
//!
//! BM=32 sibling of `ffai_moe_build_tile_plan` (`moe_tile_plan_builder.rs`).
//! Same algorithm, same one-threadgroup/one-lane-per-expert binary-search
//! design, same sync-free on-device contract (no host readback of routing
//! counts) - the only change is the tile-height constant, 16 -> 32,
//! everywhere it appears (`ceil(count/16)` -> `ceil(count/32)`, tile row
//! stride, and the caller's worst-case capacity formula). Kept as a
//! SEPARATE kernel (rather than adding a `bm` constexpr to the existing
//! one) so the production BM=16 default's kernel and its Swift wrapper are
//! untouched - this kernel is reachable only when the coop-core gather
//! path is selected (see `FFAI_MOE_COOP` in the Swift wiring).
//!
//! Originally built for a bounded BM=32 tileplan experiment on the OLD
//! single-simdgroup MPP kernel (`moe_mpp_tileplan_bm32.rs`, not carried
//! into this branch - see `moe_gather_qmm_coop.rs`'s module doc for why
//! that experiment's result does not transfer here); the plan-building
//! algorithm itself is geometry-agnostic and applies unchanged to the
//! 4-simdgroup coop-core gather kernel.
//!
//! ## Algorithm
//!
//! One threadgroup, one lane per expert (`tid = e`). Each lane:
//!
//! 1. Binary-searches `sorted_experts` (ascending) for `lower_bound(e)` and
//!    `lower_bound(e+1)` - fixed 22-iteration branchless binary search,
//!    exact for any `m_total` under 2^22. `count = lower_bound(e+1) -
//!    lower_bound(e)`.
//! 2. Writes `ceil(count/32)` to threadgroup memory, barrier.
//! 3. Computes its exclusive prefix sum over all experts' tile counts to
//!    find its own tile-write offset.
//! 4. Emits its `ceil(count/32)` tiles at that offset - same row/tile
//!    layout and ascending-by-expert concatenation order as
//!    `build_tile_plan_with_bm(counts, 32)` (the Rust host-side reference
//!    in `moe_mpp_tileplan.rs`).
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256` - scoped to Qwen3.6-35B-A3B's 256 experts, same as
//!   the BM=16 sibling.
//! - Caller pre-sizes `tile_expert`/`tile_row_start`/`tile_row_count` to a
//!   host-computed WORST-CASE capacity, `ceil(m_total/32) + n_experts`
//!   (half the BM=16 sibling's `ceil(m_total/16) + n_experts` M-term, same
//!   `+ n_experts` slack for the "every expert gets at least a rounding
//!   tile" bound), and zero-fills them before this dispatch. Entries at or
//!   past the real tile count are left at their zero fill: `tile_expert =
//!   0` (a valid expert id - the consuming GEMM kernel still reads real,
//!   in-range weight memory for these tiles, it just does redundant work)
//!   and `tile_row_count = 0`, which the GEMM kernel's `mr < row_count`
//!   tail mask turns into a dispatched-but-inert tile.

use ffai_kernels::kernel;

/// Device-side MoE tile-plan builder, BM=32. See module docs for the
/// algorithm and the worst-case-capacity dispatch contract this pairs
/// with.
#[kernel]
pub fn ffai_moe_build_tile_plan_bm32(
    sorted_experts: Tensor<u32>,
    mut tile_expert: Tensor<u32>,
    mut tile_row_start: Tensor<u32>,
    mut tile_row_count: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] n_experts: u32,
) {
    let e = tid;
    // Capacity fixed at 256: Qwen3.6-35B-A3B's expert count. See the
    // dispatch-invariants doc comment above.
    threadgroup_alloc("num_tiles_pe", 256u32, u32);

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
    let num_tiles = (count + 31u32) / 32u32;
    threadgroup_store("num_tiles_pe", e, num_tiles);
    threadgroup_barrier();

    // Exclusive prefix sum over num_tiles_pe[0..n_experts) - this lane's
    // tile-write offset. Ascending-expert-order concatenation, matching
    // `build_tile_plan_with_bm(counts, 32)`'s iteration order.
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
}

#[cfg(test)]
mod tests {
    use ffai_kernels::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_without_inline_msl() {
        let k = ffai_moe_build_tile_plan_bm32::kernel_ir();
        assert_eq!(k.name, "ffai_moe_build_tile_plan_bm32");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        // Sanity: this kernel is u32-only (no T generic), one IR regardless
        // of the model's activation dtype.
        let _ = DType::F32;
    }
}

/// Bench registration - required for `ffaik build` to discover this kernel
/// at all (kernel discovery for MSL/metallib/Swift emission walks the bench
/// registry, not the raw `#[kernel]` inventory; see
/// `kernel_registry_consistency.rs`). No `T` generic, so `dtypes = [f32]` is
/// a formality (mirrors the BM=16 sibling's bench).
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_moe_build_tile_plan_bm32;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Realistic scale (T=512 * top-8, 256 experts) skewed-ish fixture -
    /// same shape as the BM=16 sibling's bench.
    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan_bm32(_dt: DType) -> BenchSetup {
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
        let capacity = m_total.div_ceil(32) + n_experts;

        BenchSetup::new(ffai_moe_build_tile_plan_bm32::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("sorted_experts", u32_bytes(&sorted_experts), DType::U32))
            .buffer(BenchBuffer::zeros("tile_expert", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_start", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_count", capacity, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_experts", n_experts as u32)
            .with_shape_label(format!("M{m_total} E{n_experts} cap{capacity}"))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((m_total * 4 + capacity * 4 * 3) as u64)
    }
}

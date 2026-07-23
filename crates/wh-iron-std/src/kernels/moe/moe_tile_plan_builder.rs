//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Device-side tile-plan builder for `iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`
//! (`moe_mpp_tileplan.rs`). `iron_moe_build_tile_plan`.
//!
//! The tileplan GEMM needs its `tile_expert`/`tile_row_start`/`tile_row_count`
//! plan built from the SAME per-expert row counts the sort step already
//! implies. The token-order device routing plan (`iron_moe_sort_plan` in
//! `moe_permute.rs`) produces `sorted_experts[m_total]` entirely on device,
//! with no host readback by design - that is the whole point of the
//! device plan, and reintroducing a host sync just to read counts back
//! would tax every MoE layer's prefill step with a wait the isolated
//! kernel win cannot pay for. This kernel closes that gap on-device: it
//! takes the already-sorted `sorted_experts` buffer directly (no separate
//! counts input) and derives the same plan `build_tile_plan` (the Rust
//! host-side reference in `moe_mpp_tileplan.rs`) would produce from
//! `counts[e] = count of e in sorted_experts`.
//!
//! ## Algorithm
//!
//! One threadgroup, one lane per expert (`tid = e`). Each lane:
//!
//! 1. Binary-searches `sorted_experts` (ascending) for `lower_bound(e)` and
//!    `lower_bound(e+1)` - fixed 22-iteration branchless binary search,
//!    exact for any `m_total` under 2^22 (this model's largest prefill is
//!    far below that). `count = lower_bound(e+1) - lower_bound(e)`.
//! 2. Writes `ceil(count/16)` to threadgroup memory, barrier.
//! 3. Computes its exclusive prefix sum over all experts' tile counts
//!    (`O(n_experts)` per lane, `O(n_experts^2)` total - trivial next to
//!    the GEMM work this unblocks) to find its own tile-write offset.
//! 4. Emits its `ceil(count/16)` tiles at that offset - same row/tile
//!    layout as `build_tile_plan`, same ascending-by-expert concatenation
//!    order (zero-count experts naturally contribute zero tiles, since
//!    the loop bound is their own `num_tiles`).
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256` - the threadgroup scratch array is sized for
//!   Qwen3.6-35B-A3B's 256 experts; this kernel is scoped to that model,
//!   not a general routing-width primitive.
//! - Caller pre-sizes `tile_expert`/`tile_row_start`/`tile_row_count` to a
//!   host-computed WORST-CASE capacity (`ceil(m_total/16) + n_experts` -
//!   provably an upper bound on `sum_e ceil(count[e]/16)`, computable from
//!   `m_total`/`n_experts` alone with no device data) and zero-fills them
//!   before this dispatch. Entries at or past the real tile count are left
//!   at their zero fill: `tile_expert = 0` (a valid expert id - the
//!   consuming GEMM kernel still reads real, in-range weight memory for
//!   these tiles, it just does redundant work) and `tile_row_count = 0`,
//!   which the GEMM kernel's `mr < row_count` tail mask turns into a
//!   dispatched-but-inert tile: no X row is staged, no output row is
//!   written. This trades a bounded number of wasted-but-safe tiles
//!   (`<= n_experts`, shrinking to a negligible fraction of the real tile
//!   count as `m_total` grows) for a fully static, sync-free dispatch grid
//!   on the GEMM kernel that follows - no indirect dispatch needed.

use wh_iron::kernel;

/// Device-side MoE tile-plan builder. See module docs for the algorithm and
/// the worst-case-capacity dispatch contract this pairs with.
#[kernel]
pub fn iron_moe_build_tile_plan(
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
    let num_tiles = (count + 15u32) / 16u32;
    threadgroup_store("num_tiles_pe", e, num_tiles);
    threadgroup_barrier();

    // Exclusive prefix sum over num_tiles_pe[0..n_experts) - this lane's
    // tile-write offset. Ascending-expert-order concatenation, matching
    // `build_tile_plan`'s iteration order.
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
        let k = iron_moe_build_tile_plan::kernel_ir();
        assert_eq!(k.name, "iron_moe_build_tile_plan");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        // Sanity: this kernel is u32-only (no T generic), one IR regardless
        // of the model's activation dtype.
        let _ = DType::F32;
    }
}

/// Bench registration - required for `iron build` to discover this kernel
/// at all (kernel discovery for MSL/metallib/Swift emission walks the bench
/// registry, not the raw `#[kernel]` inventory; see
/// `kernel_registry_consistency.rs`). No `T` generic, so `dtypes = [f32]` is
/// a formality (mirrors `iron_moe_sort_plan`'s bench in `moe_permute.rs`,
/// the same u32-only-kernel shape).
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_build_tile_plan;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Realistic scale (T=512 * top-8, 256 experts) skewed-ish fixture -
    /// enough nonzero/zero experts mixed in that the prefix-sum + per-lane
    /// emission loops aren't degenerate no-ops.
    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan(_dt: DType) -> BenchSetup {
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
        let capacity = m_total.div_ceil(16) + n_experts;

        BenchSetup::new(iron_moe_build_tile_plan::kernel_ir())
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

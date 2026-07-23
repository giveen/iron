//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Device-side "own" tile-plan builder for the F-85 tile-pairing lever's
//! `iron_moe_gather_qmm_coop` (UNCHANGED, existing kernel) leg.
//! `iron_moe_build_tile_plan_bm32_own`.
//!
//! Near-identical to `moe_tile_plan_builder_bm32.rs`'s
//! `iron_moe_build_tile_plan_bm32` - same one-threadgroup/one-lane-per-
//! expert binary-search + prefix-sum design, same tile row/layout and
//! ascending-expert-order concatenation - with exactly ONE change: an
//! expert whose remainder (`count % 32`) is `1..=16` rows does NOT get a
//! tile for that remainder here. Those rows are pairing CANDIDATES,
//! handled by the separate `iron_moe_build_tile_plan_bm32_paired` /
//! `iron_moe_gather_qmm_coop_paired` dispatch instead (see
//! `moe_tile_plan_builder_paired.rs`'s module doc for why this is two
//! dispatches, not one).
//!
//! This exists as its OWN kernel (rather than reusing the production
//! `iron_moe_build_tile_plan_bm32` unmodified) because that kernel emits a
//! tile for EVERY nonzero remainder unconditionally - reusing it as-is
//! for the "own" dispatch would re-emit a tile for a `1..=16`-row
//! remainder that the paired dispatch ALSO covers, double-processing (and
//! double-writing `out[...]` for) those rows. The tile-pairing lever is
//! entirely opt-in (`IRON_MOE_PAIRED`), so this new kernel changes nothing
//! about the production BM=16/BM=32 default's existing builder or its
//! Swift wrapper.
//!
//! `own_tiles(count) = count / 32 + (1 if count % 32 > 16 else 0)` -
//! `full` complete BM=32 tiles, plus one more IF the remainder is
//! `17..=31` (no pairing hazard on that case - see the paired module's
//! doc). Compare to the unmodified sibling's `ceil(count / 32)`: the two
//! formulas agree everywhere except when the remainder is `1..=16`, where
//! this one is exactly one tile smaller.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[1, 1, 1]` (ONE threadgroup); threadgroup
//!   `[n_experts, 1, 1]`.
//! - `n_experts <= 256`, same scope as the unpaired sibling.
//! - Caller pre-sizes `tile_expert`/`tile_row_start`/`tile_row_count` to
//!   the SAME worst-case capacity as the unpaired sibling
//!   (`ceil(m_total/32) + n_experts` - this kernel never emits MORE tiles
//!   than that one does for the same fixture) and zero-fills them before
//!   dispatch; unused tail entries stay zero-filled, same contract.
//! - Must be dispatched against the SAME `sorted_experts` array
//!   `iron_moe_build_tile_plan_bm32_paired` reads in the same forward -
//!   see that kernel's module doc.
//!
//! ## Indirect-dispatch real tile count (F-85 idle-tile-cost follow-up)
//!
//! Two extra scalar outputs, `tile_count_gateup` / `tile_count_down`, same
//! contract as the unpaired sibling's (`moe_tile_plan_builder_bm32.rs`)
//! own indirect-dispatch outputs: the SAME real tile count value, written
//! by exactly one lane (`e == n_experts - 1`) as `tile_offset + num_tiles`,
//! feeding two independently-offset `MTLDispatchThreadgroupsIndirectArguments`
//! triples (gate/up and down legs differ in `n_out`). This "own" builder's
//! utilization against the shared worst-case capacity is typically WORSE
//! than the unpaired sibling's own utilization at the same `m_total` -
//! stripping the `1..=16`-row pairing candidates out to the paired
//! dispatch shrinks `num_tiles` well below the unpaired sibling's count
//! while the capacity bound stays the same - so the indirect-dispatch fix
//! matters more here, not less, whenever the pairing lever is opted into.

use wh_iron::kernel;

/// Device-side "own" MoE tile-plan builder, BM=32. See module docs.
#[kernel]
pub fn iron_moe_build_tile_plan_bm32_own(
    sorted_experts: Tensor<u32>,
    mut tile_expert: Tensor<u32>,
    mut tile_row_start: Tensor<u32>,
    mut tile_row_count: Tensor<u32>,
    mut tile_count_gateup: Tensor<u32>,
    mut tile_count_down: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] n_experts: u32,
) {
    let e = tid;
    // Capacity fixed at 256: Qwen3.6-35B-A3B's expert count, same as the
    // unpaired sibling.
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
    let full = count / 32u32;
    let remainder = count - full * 32u32; // 0..=31
    let is_solo = select(remainder > 16u32, 1u32, 0u32); // 17..=31 only
    let num_tiles = full + is_solo; // excludes 1..=16 (pairing candidates)

    threadgroup_store("num_tiles_pe", e, num_tiles);
    threadgroup_barrier();

    // Exclusive prefix sum over num_tiles_pe[0..n_experts) - this lane's
    // tile-write offset. Ascending-expert-order concatenation, same as
    // the unpaired sibling.
    let mut tile_offset = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        if e2 < e {
            tile_offset = tile_offset + threadgroup_load("num_tiles_pe", e2);
        }
    }

    // Same uniform emission loop as the unpaired sibling - correct here
    // too because `num_tiles` already excludes the case where the last
    // sub-tile would be a 1..=16-row remainder: when `is_solo == 0`,
    // `local` only ever reaches `full - 1`, whose `remaining` is always
    // `> 32` (this expert has at least one more full 32-row chunk beyond
    // it, or `remainder <= 16 < 32`), so `rc` is always exactly 32 for
    // every emitted tile in that case - never a short one.
    for local in range(0u32, num_tiles, 1u32) {
        let idx = tile_offset + local;
        let start = row_base + local * 32u32;
        let remaining = count - local * 32u32;
        let rc = select(remaining < 32u32, remaining, 32u32);
        store(tile_expert[idx], e);
        store(tile_row_start[idx], start);
        store(tile_row_count[idx], rc);
    }

    // Indirect-dispatch real tile count - see module doc. Same derivation
    // as the unpaired sibling: `tile_offset` at the LAST lane is the
    // exclusive prefix sum over all experts < e; adding this lane's own
    // `num_tiles` makes it the inclusive total over every expert.
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
        let k = iron_moe_build_tile_plan_bm32_own::kernel_ir();
        assert_eq!(k.name, "iron_moe_build_tile_plan_bm32_own");
        let all_ops =
            || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
        assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
        let _ = DType::F32;
    }
}

/// Bench registration - required for `iron build` to discover this
/// kernel; see the unpaired sibling's identical note.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_build_tile_plan_bm32_own;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[bench(dtypes = [f32])]
    fn bench_moe_build_tile_plan_bm32_own(_dt: DType) -> BenchSetup {
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

        BenchSetup::new(iron_moe_build_tile_plan_bm32_own::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("sorted_experts", u32_bytes(&sorted_experts), DType::U32))
            .buffer(BenchBuffer::zeros("tile_expert", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_start", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_row_count", capacity, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_count_gateup", 1, DType::U32).output())
            .buffer(BenchBuffer::zeros("tile_count_down", 1, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_experts", n_experts as u32)
            .with_shape_label(format!("M{m_total} E{n_experts} cap{capacity}"))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((m_total * 4 + capacity * 4 * 3) as u64)
    }
}

// GPU correctness for this kernel lives in
// `tests/moe_build_tile_plan_bm32_own_correctness.rs`, mirroring
// `tests/moe_build_tile_plan_correctness.rs`'s raw
// `Context::dispatch_with_grid` pattern.

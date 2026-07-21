//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! `ffai_moe_gather_qmm_coop_paired` - two-expert PAIRED tile variant of
//! the coop-core MoE gather GEMM (`ffai_moe_gather_qmm_coop`,
//! `moe_gather_qmm_coop.rs`). F-85 follow-up lever: at low-context
//! prefill (T=512-ish, top-8 of 256 experts, average 16 rows/expert),
//! most experts finish with a `1..=16`-row remainder well under the
//! BM=32 tile height, so a large share of the unpaired kernel's dispatched
//! tiles carry far fewer than 32 real rows - the rest of the tile's
//! K-loop work is spent on masked-out padding that produces no output.
//! This kernel packs TWO such short remainders (from generally different,
//! unrelated experts) into ONE tile dispatch instead of two, eliminating
//! that padding for whichever rows fit.
//!
//! This is a SEPARATE kernel and a SEPARATE dispatch from
//! `ffai_moe_gather_qmm_coop`, not a generalization of it - see
//! `moe_tile_plan_builder_paired.rs`'s module doc for why: a design that
//! routed every tile (including ordinary full-32-row and 17-31-row
//! remainder tiles) through one uniform two-expert format would roughly
//! DOUBLE the per-tile weight-dequant cost of every tile to fix padding on
//! a much smaller subset. This kernel is dispatched ONLY over the
//! `1..=16`-row pairing-candidate tiles `ffai_moe_build_tile_plan_bm32_paired`
//! produces; every other row is handled by the completely unmodified
//! `ffai_moe_gather_qmm_coop` dispatch, at its existing cost.
//!
//! ## The fragment-alignment safety argument
//!
//! A prior scout flagged tile-pairing as correctness-risky and deferred:
//! the hazard is a cooperative MMA fragment reading rows from TWO
//! different experts' weights within the SAME 16x16 tensor-op, silently
//! mixing two experts' math into one accumulator. This kernel avoids that
//! by construction rather than by masking it after the fact:
//!
//! - The coop-core GEMM already partitions its BM=32 M-tile into two
//!   16-row halves along the SIMDGROUP axis - `sm = sg / 2` assigns
//!   simdgroups 0-1 to M-rows 0..16 (the tile's "A half") and simdgroups
//!   2-3 to M-rows 16..32 ("B half") - see `ffai_moe_gather_qmm_coop`'s
//!   module doc. Each simdgroup's cooperative matmul fragments are
//!   therefore ALREADY confined to one 16-row half.
//! - This kernel gives the A half and B half INDEPENDENT expert
//!   indirection (`tile_expert_a`/`tile_row_start_a`/`tile_row_count_a`
//!   and the `_b` sibling). A simdgroup with `sm == 0` reads ONLY
//!   expert A's dequantized weights for its entire K-loop; a simdgroup
//!   with `sm == 1` reads ONLY expert B's. No fragment ever spans the
//!   16-row boundary (the tile-plan builder never emits a tile whose
//!   halves are misaligned with it - see `moe_tile_plan_builder_paired.rs`),
//!   so no fragment ever mixes two experts' weights.
//! - The expert-A/B split is a pure THREADGROUP-MEMORY ADDRESS OFFSET
//!   (`ws_expert_off`, added to the existing per-simdgroup N-column
//!   offset before `coop_tile_load_b`), not a runtime-selected buffer
//!   NAME or a branch around the `coop_tile_*` intrinsics themselves -
//!   `Ws` is sized for BOTH experts (`2 * BN * (BK+4)`, expert A's
//!   dequant in the first half, expert B's in the second) and every lane
//!   stages BOTH halves unconditionally, same as the unpaired kernel
//!   stages its one expert's W tile. This sidesteps a real codegen
//!   limitation found (and worked around) while building this kernel's
//!   tile-plan builder: an `if candidate { if slot_a {...} else {...} }`
//!   shape whose two inner arms store to DIFFERENT device tensors was
//!   observed to drop the inner branch's guard, executing BOTH arms
//!   unconditionally (see `moe_tile_plan_builder_bm32_paired.rs`'s module
//!   doc) - this kernel avoids that class of bug entirely by never
//!   needing a runtime-selected STORE TARGET for the W stage, only a
//!   runtime-selected ADDRESS.
//! - `Xs` (the row-gathered activation tile) stays ONE buffer, same
//!   layout as the unpaired kernel - it is not expert-specific, only
//!   row-specific, so a per-row (not per-simdgroup) `select` on
//!   `row_start`/`row_count` is enough to route each of the 32 tile-local
//!   rows to its correct half's source range.
//!
//! Threadgroup memory, the K-loop's 2-barrier structure, and the coop
//! matmul geometry (TPG=128, 4 SG in a 2x2 warp grid, BN=64, gs64
//! step-counter) are otherwise identical to `ffai_moe_gather_qmm_coop` -
//! see that kernel's module doc for the full staging walkthrough. `Ws` is
//! the only threadgroup buffer that changes size (2304 -> 4608 elements,
//! two experts' worth); `Xs` and `OutScratch` are unchanged.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/64, num_pair_tiles, 1]`; threadgroup
//!   `[128, 1, 1]` (4 simdgroups, 2x2 warp grid) - same shape as the
//!   unpaired sibling, `num_pair_tiles` in place of that kernel's
//!   `num_tiles`.
//! - `k_in % 32 == 0`, `n_out % 64 == 0`, `group_size` divides `k_in`.
//! - `tile_row_count_a[i]` in `1..=16` for every `i` (the paired tile-plan
//!   builder never emits an empty A half); `tile_row_count_b[i]` in
//!   `0..=16` (`0` marks an odd-leftover tile with an inert B half - see
//!   `moe_tile_plan_builder_paired.rs`). Built by `build_paired_tile_plan_bm32`
//!   on the host or `ffai_moe_build_tile_plan_bm32_paired` on device.
//! - int4 only, scope-matched to the unpaired sibling this is an
//!   opt-in ADDITION to (not a replacement for).
//! - macOS 26+ / Metal 4, same as the sibling coop kernels.

use ffai_kernels::kernel;

/// Paired variant of `ffai_moe_gather_qmm_coop`. Params: `x [n_x_rows,
/// k_in]`, `w [n_experts, n_out, k_in/8]` int4 packed, `scales`/`biases
/// [n_experts, n_out, k_in/group_size]`, `tile_expert_a`/`tile_row_start_a`/
/// `tile_row_count_a` and the `_b` sibling `[num_pair_tiles]` (paired tile
/// plan, see module docs), `x_rows [m_total]` (gather indirection into
/// `x`), `out [m_total, n_out]` (sorted-row space, same index space as
/// `x_rows` - shared with the unpaired sibling's dispatch, since the two
/// kernels partition one routing decision's rows between them).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gather_qmm_coop_paired<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    tile_expert_a: Tensor<u32>,
    tile_row_start_a: Tensor<u32>,
    tile_row_count_a: Tensor<u32>,
    tile_expert_b: Tensor<u32>,
    tile_row_start_b: Tensor<u32>,
    tile_row_count_b: Tensor<u32>,
    x_rows: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    // 2x2 warp grid; sm selects the M-half (0 = rows 0..16 = expert A,
    // 1 = rows 16..32 = expert B) - see module doc for why this is the
    // safe fragment-alignment boundary.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 32u32;
    let w_n_base = tgid_x * 64u32;
    let tile_idx = tgid_y;

    let m_tile_base_a = load(tile_row_start_a[tile_idx]);
    let row_count_a = load(tile_row_count_a[tile_idx]);
    let expert_a = load(tile_expert_a[tile_idx]);
    let m_tile_base_b = load(tile_row_start_b[tile_idx]);
    let row_count_b = load(tile_row_count_b[tile_idx]);
    let expert_b = load(tile_expert_b[tile_idx]);

    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    let w_expert_base_a = expert_a * n_out * packs_per_row;
    let sb_expert_base_a = expert_a * n_out * groups_per_row;
    let w_expert_base_b = expert_b * n_out * packs_per_row;
    let sb_expert_base_b = expert_b * n_out * groups_per_row;

    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // BM=32 x (BK+4)=36
    threadgroup_alloc("Ws", 4608u32, coop_stage(T)); // 2 experts x BN=64 x (BK+4)=36
    threadgroup_alloc("OutScratch", 1024u32, f32); // 4 SG x 16 x 16
    coop_tile_setup(
        "gemm0",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_setup(
        "gemm1",
        16u32,
        16u32,
        32u32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    coop_tile_zero("gemm0");
    coop_tile_zero("gemm1");

    // X coop-load lane mapping: 128 lanes x 8 contiguous K-elems = 1024 =
    // BM*BK, identical to the unpaired sibling - ALL 128 lanes stage the
    // full 32-row tile cooperatively, spanning BOTH halves. `x_m_row` is
    // the TILE-LOCAL row (0..32); which half (and therefore which
    // row_start/row_count/gather source) it belongs to is a per-row
    // `select`, resolved once outside the K-loop since it does not
    // depend on `kb` - same pattern the unpaired sibling uses for its
    // single `row_count` mask.
    let x_m_row = lane_in_tg / 4u32; // 0..32
    let x_k_quad = lane_in_tg & 3u32; // 0..4
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let x_row_in_half_a = x_m_row < 16u32;
    let x_local_in_half = select(x_row_in_half_a, x_m_row, x_m_row - 16u32);
    let x_half_row_start = select(x_row_in_half_a, m_tile_base_a, m_tile_base_b);
    let x_half_row_count = select(x_row_in_half_a, row_count_a, row_count_b);
    let x_in_run = x_local_in_half < x_half_row_count;
    let x_global_row = x_half_row_start + x_local_in_half;
    let x_safe_row = select(x_in_run, x_global_row, 0u32);
    let x_src_row = load(x_rows[x_safe_row]);

    // W coop-dequant lane mapping for the low half (w_row 0..32) and high
    // half (w_row 32..64) of the BN=64 tile - identical to the unpaired
    // sibling, computed ONCE and reused for BOTH experts' dequant below
    // (the (n, k) addressing is expert-independent; only the expert base
    // offset differs).
    let w_row_lo = lane_in_tg / 4u32; // 0..32
    let w_row_hi = 32u32 + w_row_lo; // 32..64
    let w_pack_in_row = lane_in_tg & 3u32; // 0..4
    let wn_lo = w_n_base + w_row_lo;
    let wn_hi = w_n_base + w_row_hi;
    let ws_base_lo = w_row_lo * 36u32 + w_pack_in_row * 8u32;
    let ws_base_hi = w_row_hi * 36u32 + w_pack_in_row * 8u32;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off0 = sg_n_base * 36u32; // gemm0: cols sg_n_base .. sg_n_base+16
    let ws_sg_off1 = (sg_n_base + 16u32) * 36u32; // gemm1: cols +16 .. +32
    let sg_scratch_off = sg * 256u32;

    // Expert A/B base addresses for scale/bias/weight-pack - independent
    // of `kb`, computed once.
    let sb_base_lo_a = sb_expert_base_a + wn_lo * groups_per_row;
    let sb_base_hi_a = sb_expert_base_a + wn_hi * groups_per_row;
    let w_pack_row_base_lo_a = w_expert_base_a + wn_lo * packs_per_row;
    let w_pack_row_base_hi_a = w_expert_base_a + wn_hi * packs_per_row;
    let sb_base_lo_b = sb_expert_base_b + wn_lo * groups_per_row;
    let sb_base_hi_b = sb_expert_base_b + wn_hi * groups_per_row;
    let w_pack_row_base_lo_b = w_expert_base_b + wn_lo * packs_per_row;
    let w_pack_row_base_hi_b = w_expert_base_b + wn_hi * packs_per_row;

    // `Ws` block layout: expert A's dequant occupies [0, 2304), expert
    // B's occupies [2304, 4608) - a fixed ADDRESS offset, not a separate
    // buffer name, so selecting between them at matmul time (below) is a
    // `select` on a plain integer, never a branch around the
    // `coop_tile_*` intrinsics.
    let ws_block_b_off = 2304u32;

    for kb in range(0u32, k_in, 32u32) {
        // 1. Stage X[x_src_row, kb + x_k_base..+8] -> Xs, zero-filled for
        //    tile-local rows past their half's row_count.
        let x_row_dev_base = x_src_row * k_in + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, select(x_in_run, xv, 0.0f32));
        }

        // 2. W dequant, expert A - low half then high half of the BN=64
        //    tile, written into Ws block A.
        let k_off = kb + w_pack_in_row * 8u32;
        let g = k_off / group_size;
        let pack_dev_lo_a = w_pack_row_base_lo_a + kb / 8u32 + w_pack_in_row;
        let packed_lo_a = load(w[pack_dev_lo_a]);
        let scale_lo_a = load(scales[sb_base_lo_a + g]).cast::<f32>();
        let bias_lo_a = load(biases[sb_base_lo_a + g]).cast::<f32>();
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = ((packed_lo_a >> (_ni * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", ws_base_lo + _ni, scale_lo_a * nib + bias_lo_a);
        }
        let pack_dev_hi_a = w_pack_row_base_hi_a + kb / 8u32 + w_pack_in_row;
        let packed_hi_a = load(w[pack_dev_hi_a]);
        let scale_hi_a = load(scales[sb_base_hi_a + g]).cast::<f32>();
        let bias_hi_a = load(biases[sb_base_hi_a + g]).cast::<f32>();
        for _nj in range(0u32, 8u32, 1u32) {
            let nib = ((packed_hi_a >> (_nj * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", ws_base_hi + _nj, scale_hi_a * nib + bias_hi_a);
        }

        // 3. W dequant, expert B - same (n, k) addressing, DIFFERENT
        //    expert base, written into Ws block B (offset by
        //    ws_block_b_off). Unconditional, same as block A - every lane
        //    stages BOTH experts' weights every K-slab, the deliberate
        //    2x-dequant cost this kernel accepts (see module doc) in
        //    exchange for one fewer dispatched tile.
        let pack_dev_lo_b = w_pack_row_base_lo_b + kb / 8u32 + w_pack_in_row;
        let packed_lo_b = load(w[pack_dev_lo_b]);
        let scale_lo_b = load(scales[sb_base_lo_b + g]).cast::<f32>();
        let bias_lo_b = load(biases[sb_base_lo_b + g]).cast::<f32>();
        for _nk in range(0u32, 8u32, 1u32) {
            let nib = ((packed_lo_b >> (_nk * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store(
                "Ws",
                ws_block_b_off + ws_base_lo + _nk,
                scale_lo_b * nib + bias_lo_b,
            );
        }
        let pack_dev_hi_b = w_pack_row_base_hi_b + kb / 8u32 + w_pack_in_row;
        let packed_hi_b = load(w[pack_dev_hi_b]);
        let scale_hi_b = load(scales[sb_base_hi_b + g]).cast::<f32>();
        let bias_hi_b = load(biases[sb_base_hi_b + g]).cast::<f32>();
        for _nl in range(0u32, 8u32, 1u32) {
            let nib = ((packed_hi_b >> (_nl * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store(
                "Ws",
                ws_block_b_off + ws_base_hi + _nl,
                scale_hi_b * nib + bias_hi_b,
            );
        }

        threadgroup_barrier();

        // 4. Per-SG cooperative matmul. `ws_expert_off` picks WHICH of
        //    the two dequant blocks this simdgroup's matmul reads - an
        //    address offset added to the existing per-simdgroup N-column
        //    offset, uniform across the whole simdgroup (every lane in
        //    an SG shares the same `sm`), never a per-lane divergent
        //    value.
        let ws_expert_off = select(sm == 0u32, 0u32, ws_block_b_off);
        coop_tile_load_a("gemm0", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b(
            "gemm0",
            "Ws",
            true,
            coop_stage(T),
            36u32,
            16u32,
            ws_expert_off + ws_sg_off0,
        );
        coop_tile_run("gemm0");
        coop_tile_load_a("gemm1", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b(
            "gemm1",
            "Ws",
            true,
            coop_stage(T),
            36u32,
            16u32,
            ws_expert_off + ws_sg_off1,
        );
        coop_tile_run("gemm1");
        threadgroup_barrier();
    }

    let out_n_base0 = w_n_base + sg_n_base;
    let out_n_base1 = out_n_base0 + 16u32;
    let o_row = lane / 2u32; // 0..16, ALREADY relative to this SG's half.
    let o_col_base = (lane & 1u32) * 8u32;
    // This simdgroup's half is fixed by `sm` (uniform per-SG) - reuse it
    // directly rather than recomputing a per-row comparison, since
    // `sg_m_base + o_row` is by construction always in `[0, 16)` when
    // `sm == 0` and `[16, 32)` when `sm == 1`.
    let out_in_half_a = sm == 0u32;
    let out_half_row_start = select(out_in_half_a, m_tile_base_a, m_tile_base_b);
    let out_half_row_count = select(out_in_half_a, row_count_a, row_count_b);
    let out_in_run = o_row < out_half_row_count;
    let out_row = out_half_row_start + o_row;
    coop_tile_store_c("gemm0", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        if out_in_run {
            let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
            store(out[out_row * n_out + (out_n_base0 + col)], v.cast::<T>());
        }
    }
    threadgroup_barrier();
    coop_tile_store_c("gemm1", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    for _j in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _j;
        if out_in_run {
            let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
            store(out[out_row * n_out + (out_n_base1 + col)], v.cast::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use ffai_kernels::core::{DType, ir::KernelMode};

    use super::*;

    #[test]
    fn kernel_ir_uses_two_coop_tile_ops_no_inline_msl() {
        use ffai_kernels::core::ir::Op;
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt);
            assert_eq!(k.name, "ffai_moe_gather_qmm_coop_paired");
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            let setups = all_ops().filter(|op| matches!(op, Op::CoopTileSetup { .. })).count();
            assert_eq!(setups, 2, "expected gemm0 + gemm1 CoopTileSetup ops");
            let runs = all_ops().filter(|op| matches!(op, Op::CoopTileRun { .. })).count();
            assert_eq!(runs, 2, "expected one CoopTileRun per N-half per K-block-body");
        }
    }

    /// bf16 must stage through `half` - same requirement as every other
    /// `coop_tile_*` kernel in this family.
    #[test]
    fn bf16_stages_through_half() {
        use ffai_kernels::core::ir::Op;
        let k = ffai_moe_gather_qmm_coop_paired::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half for matmul2d");
    }

    /// Developer aid: dump the generated MSL, per the F-85 constexpr-
    /// codegen audit recipe. This kernel's `moe_tile_plan_builder_bm32_paired`
    /// sibling caught a real codegen bug this way (a nested if/else with
    /// different device-tensor stores per arm silently dropped the inner
    /// guard) - useful to re-check here too if the kernel's structure ever
    /// grows a similar shape.
    /// `cargo test -p ffai-kernels-std --lib --release -- \
    ///   kernels::moe::moe_gather_qmm_coop_paired::tests::dump --nocapture --test-threads=1`
    #[test]
    fn dump() {
        use ffai_kernels::codegen::msl::MslGenerator;
        let mut k = ffai_moe_gather_qmm_coop_paired::kernel_ir_for(DType::F16);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

/// New-syntax correctness for `ffai_moe_gather_qmm_coop_paired` against a
/// CPU dequant-then-matmul oracle that walks the paired tile plan
/// per-half, using each half's OWN expert - the exact property the
/// fragment-alignment safety argument in the module doc depends on. The
/// attack fixtures below specifically target the hazard a prior scout
/// flagged: paired experts with DIFFERENT weight values (so a wrong-
/// expert read produces a wrong, not merely close, number), the exact
/// 16/16 boundary, unbalanced pairs, and zero-count experts.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_moe_gather_qmm_coop_paired;
    use crate::{
        kernels::moe::moe_tile_plan_builder_paired::build_paired_tile_plan_bm32,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
        weights
            .chunks_exact(8)
            .map(|chunk| {
                let mut packed = 0u32;
                for (i, &q) in chunk.iter().enumerate() {
                    packed |= (q & 0xf) << (i * 4);
                }
                packed
            })
            .collect()
    }

    /// Dequant-then-matmul a single half's real rows into `out`, using
    /// THAT half's expert - the function this whole test suite exists to
    /// pin down: a bug that read the wrong half's expert for some rows
    /// would show up here as a wrong-but-plausible number, not a crash.
    #[allow(clippy::too_many_arguments)]
    fn compute_half(
        out: &mut [f32],
        x: &[f32],
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        x_rows: &[u32],
        expert: u32,
        row_start: u32,
        row_count: u32,
        k_in: usize,
        n_out: usize,
        group_size: usize,
    ) {
        let weight_stride_m = k_in / 8;
        let groups_per_row = k_in / group_size;
        for local in 0..row_count {
            let global_row = (row_start + local) as usize;
            let src_row = x_rows[global_row] as usize;
            for n in 0..n_out {
                let weight_row_base =
                    expert as usize * n_out * weight_stride_m + n * weight_stride_m;
                let scale_row_base = expert as usize * n_out * groups_per_row + n * groups_per_row;
                let x_row_base = src_row * k_in;
                let mut acc = 0.0f32;
                for pack_idx in 0..(k_in / 8) {
                    let packed = weight_packed[weight_row_base + pack_idx];
                    let k_first = pack_idx * 8;
                    let g = k_first / group_size;
                    let scale = scales[scale_row_base + g];
                    let bias = biases[scale_row_base + g];
                    for bit in 0..8 {
                        let q = ((packed >> (bit * 4)) & 0xf) as f32;
                        acc += (q * scale + bias) * x[x_row_base + k_first + bit];
                    }
                }
                out[global_row * n_out + n] = acc;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_oracle(
        x: &[f32],
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        x_rows: &[u32],
        tile_expert_a: &[u32],
        tile_row_start_a: &[u32],
        tile_row_count_a: &[u32],
        tile_expert_b: &[u32],
        tile_row_start_b: &[u32],
        tile_row_count_b: &[u32],
        m_total: usize,
        k_in: usize,
        n_out: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m_total * n_out];
        for i in 0..tile_expert_a.len() {
            compute_half(
                &mut out,
                x,
                weight_packed,
                scales,
                biases,
                x_rows,
                tile_expert_a[i],
                tile_row_start_a[i],
                tile_row_count_a[i],
                k_in,
                n_out,
                group_size,
            );
            compute_half(
                &mut out,
                x,
                weight_packed,
                scales,
                biases,
                x_rows,
                tile_expert_b[i],
                tile_row_start_b[i],
                tile_row_count_b[i],
                k_in,
                n_out,
                group_size,
            );
        }
        out
    }

    /// Build a `TestSetup` from EXPLICIT paired-tile specs
    /// `(expert_a, row_start_a, count_a, expert_b, row_start_b, count_b)`,
    /// giving full control over the exact scenarios under test (16/16
    /// boundary, unbalanced pairs, zero-count experts) rather than
    /// deriving them indirectly from a routing fixture. `n_experts`' worth
    /// of weight/scale/bias tables are filled with a position-dependent
    /// pattern (`(i * 7 + 3) & 0xf` for weights, sinusoidal for
    /// scales/biases) that differs PER EXPERT (via the `i` index spanning
    /// the full `[n_experts, n_out, k_in]` tensor), so two different
    /// experts NEVER share identical weights: a wrong-expert read is
    /// always numerically wrong, not accidentally correct.
    #[allow(clippy::too_many_arguments)]
    fn setup(
        kernel: ffai_kernels::core::ir::Kernel,
        tiles: &[(u32, u32, u32, u32, u32, u32)],
        n_experts: usize,
        m_total: usize,
        x_table_rows: usize,
        n_out: usize,
        k_in: usize,
        group_size: usize,
        dt: DType,
    ) -> TestSetup {
        let num_tiles = tiles.len();
        let tile_expert_a: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        let tile_row_start_a: Vec<u32> = tiles.iter().map(|t| t.1).collect();
        let tile_row_count_a: Vec<u32> = tiles.iter().map(|t| t.2).collect();
        let tile_expert_b: Vec<u32> = tiles.iter().map(|t| t.3).collect();
        let tile_row_start_b: Vec<u32> = tiles.iter().map(|t| t.4).collect();
        let tile_row_count_b: Vec<u32> = tiles.iter().map(|t| t.5).collect();

        let mut weight_unpacked = vec![0u32; n_experts * n_out * k_in];
        for (i, wv) in weight_unpacked.iter_mut().enumerate() {
            *wv = ((i as u32) * 7 + 3) & 0xf;
        }
        let weight_packed: Vec<u32> =
            weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

        let n_groups = k_in / group_size;
        let scales_f: Vec<f32> = (0..n_experts * n_out * n_groups)
            .map(|i| 0.005 + 0.001 * (i as f32 * 0.031).sin())
            .collect();
        let biases_f: Vec<f32> = (0..n_experts * n_out * n_groups)
            .map(|i| -0.02 + 0.004 * (i as f32 * 0.071).cos())
            .collect();
        let x_f: Vec<f32> =
            (0..x_table_rows * k_in).map(|i| 0.05 * (i as f32 * 0.017).sin()).collect();
        let x_rows: Vec<u32> = (0..m_total as u32).collect(); // identity gather.

        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = cpu_oracle(
            &x,
            &weight_packed,
            &s,
            &b,
            &x_rows,
            &tile_expert_a,
            &tile_row_start_a,
            &tile_row_count_a,
            &tile_expert_b,
            &tile_row_start_b,
            &tile_row_count_b,
            m_total,
            k_in,
            n_out,
            group_size,
        );

        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", u32_bytes(&weight_packed), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("tile_expert_a", u32_bytes(&tile_expert_a), DType::U32))
            .input(TestBuffer::from_vec(
                "tile_row_start_a",
                u32_bytes(&tile_row_start_a),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "tile_row_count_a",
                u32_bytes(&tile_row_count_a),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("tile_expert_b", u32_bytes(&tile_expert_b), DType::U32))
            .input(TestBuffer::from_vec(
                "tile_row_start_b",
                u32_bytes(&tile_row_start_b),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "tile_row_count_b",
                u32_bytes(&tile_row_count_b),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("x_rows", u32_bytes(&x_rows), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_out as u32 / 64, num_tiles as u32, 1, [128, 1, 1])
    }

    /// THE core hazard fixture: two experts at the exact 16/16 pairing
    /// cap, DIFFERENT weight patterns (guaranteed by `setup`'s
    /// per-expert-varying fill) - if the kernel ever let a simdgroup read
    /// the wrong half's expert, this fails loudly (wrong numbers on
    /// exactly the rows that crossed), not merely imprecisely.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_paired_exact_16_16_boundary_different_experts(dt: DType) -> TestSetup {
        // expert 0 rows [0,16), expert 1 rows [16,32) in the row space -
        // one paired tile, m_total=32.
        let tiles = [(0u32, 0u32, 16u32, 1u32, 16u32, 16u32)];
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            2,
            32,
            32,
            128,
            128,
            64,
            dt,
        )
    }

    /// Unbalanced pair: 3 + 13 rows, still summing to 16 real rows with
    /// 16 rows of padding.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_paired_unbalanced_3_and_13(dt: DType) -> TestSetup {
        let tiles = [(0u32, 0u32, 3u32, 1u32, 3u32, 13u32)];
        // n_out must be a multiple of BN=64 - see the kernel's dispatch
        // invariants.
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            2,
            16,
            16,
            128,
            128,
            64,
            dt,
        )
    }

    /// Maximally unbalanced pair: 16 + 1.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_paired_unbalanced_16_and_1(dt: DType) -> TestSetup {
        let tiles = [(0u32, 0u32, 16u32, 1u32, 16u32, 1u32)];
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            2,
            17,
            17,
            128,
            128,
            64,
            dt,
        )
    }

    /// Odd leftover: a paired tile with an INERT B half
    /// (`tile_row_count_b == 0`) - the B half's rows must never be
    /// written (verified implicitly: `m_total` covers only the A half's
    /// real rows, so any spurious B-half write would touch memory the
    /// oracle never assigned an expected value for the wrong row - since
    /// this uses the same `m_total`-sized `out` buffer as everything
    /// else, an out-of-run write from a buggy mask would corrupt a
    /// DIFFERENT row's expected output rather than silently vanish).
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_paired_odd_leftover_inert_b_half(dt: DType) -> TestSetup {
        let tiles = [(0u32, 0u32, 9u32, 0u32, 0u32, 0u32)];
        setup(ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt), &tiles, 1, 9, 9, 64, 128, 64, dt)
    }

    /// A paired tile immediately followed, in the SAME dispatch, by
    /// another paired tile that reuses ONE of the same experts (adjacent-
    /// tile expert reuse) - checks the per-tile expert/row_start/row_count
    /// loads are correctly re-read per tile, not stale from the previous
    /// tile in the grid.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_paired_adjacent_tiles_reuse_expert(dt: DType) -> TestSetup {
        let tiles = [(0u32, 0u32, 8u32, 1u32, 8u32, 8u32), (0u32, 16u32, 6u32, 2u32, 22u32, 10u32)];
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            3,
            32,
            32,
            64,
            128,
            64,
            dt,
        )
    }

    /// Realistic skewed/ragged fixture, via the actual host paired-plan
    /// builder (`build_paired_tile_plan_bm32`) rather than hand-crafted
    /// tiles - end-to-end check that the builder's output feeds this
    /// kernel correctly, several zero-count experts included (they
    /// contribute no candidates, so no tile references them - verified
    /// implicitly by the oracle matching).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_paired_skewed_realistic_fixture(dt: DType) -> TestSetup {
        let counts = [1usize, 0, 16, 0, 5, 11, 0, 3, 16, 9, 0, 6, 2, 14, 8, 8, 8, 1, 1, 1];
        let m_total: usize = counts.iter().sum();
        let plan = build_paired_tile_plan_bm32(&counts);
        let tiles: Vec<(u32, u32, u32, u32, u32, u32)> = (0..plan.num_tiles())
            .map(|i| {
                (
                    plan.tile_expert_a[i],
                    plan.tile_row_start_a[i],
                    plan.tile_row_count_a[i],
                    plan.tile_expert_b[i],
                    plan.tile_row_start_b[i],
                    plan.tile_row_count_b[i],
                )
            })
            .collect();
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            counts.len(),
            m_total,
            m_total,
            128,
            128,
            64,
            dt,
        )
    }

    /// Qwen3.6-35B-A3B gate/up leg shape: K=2048, N=512 (BN=64 -> 8 column
    /// tiles), paired tiles built from a realistic skewed fixture.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_paired_gate_up_shape(dt: DType) -> TestSetup {
        let counts = [3usize, 0, 14, 12, 0, 8, 16, 0, 5, 9, 1, 0, 16, 16, 16, 4, 2];
        let m_total: usize = counts.iter().sum();
        let plan = build_paired_tile_plan_bm32(&counts);
        let tiles: Vec<(u32, u32, u32, u32, u32, u32)> = (0..plan.num_tiles())
            .map(|i| {
                (
                    plan.tile_expert_a[i],
                    plan.tile_row_start_a[i],
                    plan.tile_row_count_a[i],
                    plan.tile_expert_b[i],
                    plan.tile_row_start_b[i],
                    plan.tile_row_count_b[i],
                )
            })
            .collect();
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            counts.len(),
            m_total,
            m_total,
            512,
            2048,
            64,
            dt,
        )
    }

    /// Qwen3.6-35B-A3B down leg shape: K=512, N=2048.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_paired_down_shape(dt: DType) -> TestSetup {
        let counts = [5usize, 0, 12, 9, 0, 6, 16, 0, 3, 15, 1, 0, 16, 16, 10, 2, 1];
        let m_total: usize = counts.iter().sum();
        let plan = build_paired_tile_plan_bm32(&counts);
        let tiles: Vec<(u32, u32, u32, u32, u32, u32)> = (0..plan.num_tiles())
            .map(|i| {
                (
                    plan.tile_expert_a[i],
                    plan.tile_row_start_a[i],
                    plan.tile_row_count_a[i],
                    plan.tile_expert_b[i],
                    plan.tile_row_start_b[i],
                    plan.tile_row_count_b[i],
                )
            })
            .collect();
        setup(
            ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt),
            &tiles,
            counts.len(),
            m_total,
            m_total,
            2048,
            512,
            64,
            dt,
        )
    }

    /// Non-identity gather: reversed `x_rows` over a table twice the size
    /// of `m_total`, so a bug that fell back to identity gather (row ==
    /// source row) fails loudly.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_paired_xrows_gather(dt: DType) -> TestSetup {
        let tiles = [(0u32, 0u32, 10u32, 1u32, 10u32, 6u32)];
        let m_total = 16usize;
        let x_table_rows = m_total * 2;
        let n_out = 64usize;
        let k_in = 128usize;
        let group_size = 64usize;
        let n_experts = 2usize;
        let num_tiles = tiles.len();

        let mut weight_unpacked = vec![0u32; n_experts * n_out * k_in];
        for (i, wv) in weight_unpacked.iter_mut().enumerate() {
            *wv = ((i as u32) * 7 + 3) & 0xf;
        }
        let weight_packed: Vec<u32> =
            weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
        let n_groups = k_in / group_size;
        let scales_f: Vec<f32> = (0..n_experts * n_out * n_groups)
            .map(|i| 0.005 + 0.001 * (i as f32 * 0.031).sin())
            .collect();
        let biases_f: Vec<f32> = (0..n_experts * n_out * n_groups)
            .map(|i| -0.02 + 0.004 * (i as f32 * 0.071).cos())
            .collect();
        let x_f: Vec<f32> =
            (0..x_table_rows * k_in).map(|i| 0.05 * (i as f32 * 0.017).sin()).collect();
        // Reversed gather, deliberately far from identity.
        let x_rows: Vec<u32> = (0..m_total as u32).map(|i| (x_table_rows as u32 - 1) - i).collect();

        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let tile_expert_a: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        let tile_row_start_a: Vec<u32> = tiles.iter().map(|t| t.1).collect();
        let tile_row_count_a: Vec<u32> = tiles.iter().map(|t| t.2).collect();
        let tile_expert_b: Vec<u32> = tiles.iter().map(|t| t.3).collect();
        let tile_row_start_b: Vec<u32> = tiles.iter().map(|t| t.4).collect();
        let tile_row_count_b: Vec<u32> = tiles.iter().map(|t| t.5).collect();
        let expected = cpu_oracle(
            &x,
            &weight_packed,
            &s,
            &b,
            &x_rows,
            &tile_expert_a,
            &tile_row_start_a,
            &tile_row_count_a,
            &tile_expert_b,
            &tile_row_start_b,
            &tile_row_count_b,
            m_total,
            k_in,
            n_out,
            group_size,
        );

        TestSetup::new(ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", u32_bytes(&weight_packed), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("tile_expert_a", u32_bytes(&tile_expert_a), DType::U32))
            .input(TestBuffer::from_vec(
                "tile_row_start_a",
                u32_bytes(&tile_row_start_a),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "tile_row_count_a",
                u32_bytes(&tile_row_count_a),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("tile_expert_b", u32_bytes(&tile_expert_b), DType::U32))
            .input(TestBuffer::from_vec(
                "tile_row_start_b",
                u32_bytes(&tile_row_start_b),
                DType::U32,
            ))
            .input(TestBuffer::from_vec(
                "tile_row_count_b",
                u32_bytes(&tile_row_count_b),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("x_rows", u32_bytes(&x_rows), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_out as u32 / 64, num_tiles as u32, 1, [128, 1, 1])
    }
}

/// New-syntax benchmarks for `ffai_moe_gather_qmm_coop_paired`. Without a
/// `#[bench]` entry the kernel is invisible to `ffaik build --emit`, same
/// note as the unpaired sibling. Fixtures use
/// `moe_mpp_shared::zipfish_counts` at Qwen3.6-35B-A3B's 256-expert count,
/// paired-tile-plan-built via `build_paired_tile_plan_bm32`, at the SAME
/// mTotal values and seeds the unpaired sibling's isolated bench uses -
/// the isolated GO/NO-GO comparison is `bench_moe_build_own_paired_isolated_*`
/// (own+paired dispatch time) vs `bench_moe_gather_qmm_coop_isolated_*`
/// (unpaired single-dispatch time) at the SAME routing skew.
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_moe_gather_qmm_coop_paired;
    use crate::kernels::moe::{
        moe_mpp_shared::zipfish_counts,
        moe_tile_plan_builder_paired::build_paired_tile_plan_bm32,
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[allow(clippy::too_many_arguments)]
    fn moe_coop_paired_bench(
        dt: DType,
        m_total: usize,
        n_out: usize,
        k_in: usize,
        n_experts: usize,
        group_size: usize,
        seed: u64,
        label: &str,
    ) -> BenchSetup {
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in / 8;
        let sz = dt.size_bytes();
        let counts = zipfish_counts(m_total, n_experts, seed);
        let plan = build_paired_tile_plan_bm32(&counts);
        let num_tiles = plan.num_tiles().max(1); // never dispatch a zero-sized grid.
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;

        BenchSetup::new(ffai_moe_gather_qmm_coop_paired::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::from_vec(
                "tile_expert_a",
                u32_bytes(&plan.tile_expert_a),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "tile_row_start_a",
                u32_bytes(&plan.tile_row_start_a),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "tile_row_count_a",
                u32_bytes(&plan.tile_row_count_a),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "tile_expert_b",
                u32_bytes(&plan.tile_expert_b),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "tile_row_start_b",
                u32_bytes(&plan.tile_row_start_b),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "tile_row_count_b",
                u32_bytes(&plan.tile_row_count_b),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "x_rows",
                u32_bytes(&(0..m_total as u32).collect::<Vec<_>>()),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "{label} M{m_total} N{n_out} K{k_in} E{n_experts} pairtiles{num_tiles} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / 64, num_tiles as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    // ── gate/up leg (K=2048, N=512), 3 seeds per mTotal ──────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t4096_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t4096_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t4096_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t8192_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 8192, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t8192_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 8192, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t8192_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 8192, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t16384_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t16384_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t16384_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t32768_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t32768_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_gate_up_t32768_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }

    // ── down leg (K=512, N=2048), 3 seeds per mTotal ─────────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t4096_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t4096_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t4096_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t8192_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 8192, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t8192_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 8192, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t8192_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 8192, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t16384_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t16384_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t16384_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t32768_s0(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t32768_s1(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_paired_isolated_down_t32768_s2(dt: DType) -> BenchSetup {
        moe_coop_paired_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
}

/// The paired lever's "own" dispatch leg - the UNMODIFIED
/// `ffai_moe_gather_qmm_coop` kernel, fed by `build_own_tile_plan_bm32`
/// (full BM=32 runs and 17..31-row remainders only) instead of the
/// production `build_tile_plan_with_bm` plan. Isolated GO/NO-GO
/// comparison: `bench_moe_own_isolated_*` + `bench_moe_gather_qmm_coop_paired_isolated_*`
/// (this kernel's own two dispatches) vs `bench_moe_gather_qmm_coop_isolated_*`
/// (`moe_gather_qmm_coop.rs`'s single-dispatch baseline) at the SAME
/// `(leg, mTotal, seed)` - all three call the shared `zipfish_counts`
/// helper so the routing skew is identical, not just similarly-shaped.
pub mod kernel_benches_own {
    use ffai_kernels::{bench, test::*};

    use crate::kernels::moe::{
        moe_gather_qmm_coop::ffai_moe_gather_qmm_coop,
        moe_mpp_shared::zipfish_counts,
        moe_tile_plan_builder_paired::build_own_tile_plan_bm32,
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[allow(clippy::too_many_arguments)]
    fn moe_own_bench(
        dt: DType,
        m_total: usize,
        n_out: usize,
        k_in: usize,
        n_experts: usize,
        group_size: usize,
        seed: u64,
        label: &str,
    ) -> BenchSetup {
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in / 8;
        let sz = dt.size_bytes();
        let counts = zipfish_counts(m_total, n_experts, seed);
        let (tile_expert, tile_row_start, tile_row_count) = build_own_tile_plan_bm32(&counts);
        let num_tiles = tile_expert.len().max(1);
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;

        BenchSetup::new(ffai_moe_gather_qmm_coop::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::from_vec("tile_expert", u32_bytes(&tile_expert), DType::U32))
            .buffer(BenchBuffer::from_vec("tile_row_start", u32_bytes(&tile_row_start), DType::U32))
            .buffer(BenchBuffer::from_vec("tile_row_count", u32_bytes(&tile_row_count), DType::U32))
            .buffer(BenchBuffer::from_vec(
                "x_rows",
                u32_bytes(&(0..m_total as u32).collect::<Vec<_>>()),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "{label} M{m_total} N{n_out} K{k_in} E{n_experts} owntiles{num_tiles} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / 64, num_tiles as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    // ── gate/up leg (K=2048, N=512), 3 seeds per mTotal ──────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t4096_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t4096_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t4096_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t8192_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 8192, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t8192_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 8192, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t8192_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 8192, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t16384_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t16384_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t16384_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t32768_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t32768_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_gate_up_t32768_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }

    // ── down leg (K=512, N=2048), 3 seeds per mTotal ─────────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t4096_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t4096_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t4096_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t8192_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 8192, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t8192_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 8192, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t8192_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 8192, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t16384_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t16384_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t16384_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t32768_s0(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t32768_s1(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_own_isolated_down_t32768_s2(dt: DType) -> BenchSetup {
        moe_own_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
}

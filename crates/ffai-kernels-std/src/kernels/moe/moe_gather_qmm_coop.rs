//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! `ffai_moe_gather_qmm_coop` - MoE gather variant of the cooperative-
//! staging wide-N int4 dense GEMM core (`ffai_qmm_coop`,
//! `gemm/quantized_coop.rs`).
//!
//! The dense core closes most of the remaining gap to the tensor-unit
//! roofline target (isolated 57-62% of peak vs the pre-F-85 23%) by
//! doubling the N-tile a threadgroup covers (BN=32 -> BN=64) while keeping
//! the K-loop's barrier count and X-tile size unchanged. MoE expert GEMMs
//! are the low-context prefill hot path this was built to fix
//! (~30-47% of low-T prefill time at only 12-26% of peak) but they need a
//! per-tile expert boundary and a gathered/sorted row space the dense core
//! has no concept of - this kernel adds exactly that on top of the SAME
//! inner loop and staging structure, following
//! `moe_mpp_tileplan.rs`'s `ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`
//! pattern for how a host/device tile plan feeds a `coop_tile_*` GEMM:
//!
//!   1. Per-tile expert indirection: `tile_expert`/`tile_row_start`/
//!      `tile_row_count` (built by `moe_tile_plan_builder.rs`'s
//!      `build_tile_plan_with_bm` on the host, or
//!      `moe_tile_plan_builder_bm32.rs`'s `ffai_moe_build_tile_plan_bm32` on
//!      device) give this tile's expert id and row range. `w`/`scales`/
//!      `biases` addressing adds a per-tile `expert * n_out * (...)` base
//!      offset on top of the dense core's per-column-tile addressing - the
//!      only change to the W-dequant math, not its shape.
//!   2. Row-gather on the X load: each tile-local row reads
//!      `x_rows[tile_row_start + local_row]` to find its source row in the
//!      (unsorted) activation table, the same indirection
//!      `ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`'s X-stage step
//!      uses.
//!   3. Ragged rows-per-tile guard: a plan tile can hold fewer than BM=32
//!      real rows (an expert's short last run). The X load zero-fills
//!      padding rows (`select` on `local_row < row_count`, matching the
//!      BM=16 sibling's `mr < row_count` guard) so the matmul never reads
//!      past `x_rows`' valid range; the writeback only stores rows inside
//!      `row_count`. `out` is written in the SAME sorted-row index space as
//!      `x_rows` (`out[tile_row_start + local_row, ...]`), not scattered
//!      back to the original token order - unsort/scatter is a downstream
//!      kernel's job, matching the BM=16 sibling's writeback contract.
//!
//! BM=32 was chosen (over reusing the production BM=16 tile plan and
//! pairing two tiles per threadgroup) because it matches this core's
//! native M-tile height exactly - the tile plan boundary and the GEMM's
//! M-tile boundary coincide, so no in-kernel tile-pairing logic is needed.
//! The earlier BM=32-vs-BM=16 tileplan experiment
//! (`moe_mpp_tileplan_bm32.rs`, single-simdgroup 32x32x16 `matmul2d`) is a
//! DIFFERENT kernel design and does not bear on this decision - see the
//! isolated bench results this commit's PR description reports for the
//! actual measurement on this 4-SG core.
//!
//! Threadgroup memory, tile geometry (TPG=128, 4 SG in a 2x2 warp grid,
//! `Xs[32 x 36]` + `Ws[64 x 36]` + `OutScratch[4 SG x 16 x 16]`), and the
//! 2-barrier-per-K-slab structure are byte-for-byte identical to
//! `ffai_qmm_coop` - see that kernel's module doc for the full staging
//! walkthrough.
//!
//! `group_size` is a genuine `#[constexpr]` runtime parameter (not a baked
//! literal like the dense core's group-64 divisor) so the scale/bias
//! step-counter generalizes correctly across both production leg shapes
//! this kernel serves - gate/up (`k_in=2048`, `n_out=512`, BN=64 -> 8
//! column tiles) and down (`k_in=512`, `n_out=2048`, group_size=64 -> a
//! 64-wide group still spans exactly 2 BK=32 K-slabs at this K, same as the
//! gate/up leg; verified by the `kernel_tests` module's K=512 case). This
//! mirrors `ffai_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`'s own
//! `group_size` constexpr rather than the dense core's baked-64 approach,
//! since a per-tile expert offset already means this kernel isn't a
//! drop-in ABI match for the dense core anyway.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/64, num_tiles, 1]`; threadgroup
//!   `[128, 1, 1]` (4 simdgroups, 2x2 warp grid).
//! - `k_in % 32 == 0`, `n_out % 64 == 0`, `group_size` divides `k_in`.
//! - `tile_row_count[i]` in `1..=32` for every `i` (the tile plan builder
//!   never emits an empty tile); `tile_row_start[i] + tile_row_count[i] <=
//!   m_total`. Built by `build_tile_plan_with_bm(counts, 32)` or the
//!   device `ffai_moe_build_tile_plan_bm32`.
//! - int4 only, scope-matched to the BM=16 tileplan sibling this is A/B'd
//!   against.
//! - macOS 26+ / Metal 4, same as the sibling MPP/coop kernels.

use ffai_kernels::kernel;

/// MoE gather variant of `ffai_qmm_coop`. Params: `x [n_x_rows, k_in]`,
/// `w [n_experts, n_out, k_in/8]` int4 packed (8 nibbles/u32), `scales`/
/// `biases [n_experts, n_out, k_in/group_size]`, `tile_expert`/
/// `tile_row_start`/`tile_row_count [num_tiles]` (tile plan, see module
/// docs), `x_rows [m_total]` (gather indirection into `x`, identity when
/// `x` is pre-gathered), `out [m_total, n_out]` (sorted-row space, same
/// index space as `x_rows`).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gather_qmm_coop<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    tile_expert: Tensor<u32>,
    tile_row_start: Tensor<u32>,
    tile_row_count: Tensor<u32>,
    x_rows: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    // 2x2 warp grid; each SG's quadrant is 16 rows x 32 cols (BM/2 x BN/2) -
    // unchanged from `ffai_qmm_coop`.
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 32u32;
    let w_n_base = tgid_x * 64u32;
    let tile_idx = tgid_y;

    // Tile plan: this TG's expert and row range in the sorted row space.
    // The plan builder guarantees a real, non-empty, single-expert tile -
    // no sentinel probe needed, same property the BM=16 sibling relies on.
    let m_tile_base = load(tile_row_start[tile_idx]);
    let row_count = load(tile_row_count[tile_idx]);
    let expert = load(tile_expert[tile_idx]);
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    let w_expert_base = expert * n_out * packs_per_row;
    let sb_expert_base = expert * n_out * groups_per_row;

    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // BM=32 x (BK+4)=36
    threadgroup_alloc("Ws", 2304u32, coop_stage(T)); // BN=64 x (BK+4)=36
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
    // BM*BK, identical to `ffai_qmm_coop`. `x_m_row` is the TILE-LOCAL row
    // (0..32); the gather + ragged guard below resolve it to a real source
    // row exactly once (outside the K-loop, since it does not depend on
    // `kb`), same as the dense core's per-lane-fixed M row.
    let x_m_row = lane_in_tg / 4u32; // 0..32
    let x_k_quad = lane_in_tg & 3u32; // 0..4
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let x_in_run = x_m_row < row_count;
    let x_global_row = m_tile_base + x_m_row;
    let x_safe_row = select(x_in_run, x_global_row, 0u32);
    let x_src_row = load(x_rows[x_safe_row]);

    // W coop-dequant lane mapping for the low half (w_row 0..32) and high
    // half (w_row 32..64) of the BN=64 tile - identical to `ffai_qmm_coop`,
    // with a per-tile expert base offset folded into the device addresses.
    let w_row_lo = lane_in_tg / 4u32; // 0..32
    let w_row_hi = 32u32 + w_row_lo; // 32..64
    let w_pack_in_row = lane_in_tg & 3u32; // 0..4
    let wn_lo = w_n_base + w_row_lo;
    let wn_hi = w_n_base + w_row_hi;
    let sb_base_lo = sb_expert_base + wn_lo * groups_per_row;
    let sb_base_hi = sb_expert_base + wn_hi * groups_per_row;
    let w_pack_row_base_lo = w_expert_base + wn_lo * packs_per_row;
    let w_pack_row_base_hi = w_expert_base + wn_hi * packs_per_row;
    let ws_base_lo = w_row_lo * 36u32 + w_pack_in_row * 8u32;
    let ws_base_hi = w_row_hi * 36u32 + w_pack_in_row * 8u32;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off0 = sg_n_base * 36u32; // gemm0: cols sg_n_base .. sg_n_base+16
    let ws_sg_off1 = (sg_n_base + 16u32) * 36u32; // gemm1: cols +16 .. +32
    let sg_scratch_off = sg * 256u32;

    for kb in range(0u32, k_in, 32u32) {
        // 1. Stage X[x_src_row, kb + x_k_base..+8] -> Xs, zero-filled for
        //    tile-local rows past `row_count` (ragged-tail guard).
        let x_row_dev_base = x_src_row * k_in + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, select(x_in_run, xv, 0.0f32));
        }
        // 2. W dequant, low half (N rows w_n_base..w_n_base+32).
        let pack_dev_lo = w_pack_row_base_lo + kb / 8u32 + w_pack_in_row;
        let packed_lo = load(w[pack_dev_lo]);
        let k_off = kb + w_pack_in_row * 8u32;
        let g = k_off / group_size;
        let scale_lo = load(scales[sb_base_lo + g]).cast::<f32>();
        let bias_lo = load(biases[sb_base_lo + g]).cast::<f32>();
        for _ni in range(0u32, 8u32, 1u32) {
            let nib = ((packed_lo >> (_ni * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", ws_base_lo + _ni, scale_lo * nib + bias_lo);
        }
        // 2b. W dequant, high half (N rows w_n_base+32..w_n_base+64).
        let pack_dev_hi = w_pack_row_base_hi + kb / 8u32 + w_pack_in_row;
        let packed_hi = load(w[pack_dev_hi]);
        let scale_hi = load(scales[sb_base_hi + g]).cast::<f32>();
        let bias_hi = load(biases[sb_base_hi + g]).cast::<f32>();
        for _nj in range(0u32, 8u32, 1u32) {
            let nib = ((packed_hi >> (_nj * 4u32)) & 15u32).cast::<f32>();
            threadgroup_store("Ws", ws_base_hi + _nj, scale_hi * nib + bias_hi);
        }
        threadgroup_barrier();
        // 3. Per-SG cooperative matmul, both N-halves of the SG's quadrant.
        coop_tile_load_a("gemm0", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm0", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off0);
        coop_tile_run("gemm0");
        coop_tile_load_a("gemm1", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm1", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off1);
        coop_tile_run("gemm1");
        threadgroup_barrier();
    }

    let out_n_base0 = w_n_base + sg_n_base;
    let out_n_base1 = out_n_base0 + 16u32;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    let out_local_row = sg_m_base + o_row; // tile-local row, 0..32
    let out_in_run = out_local_row < row_count;
    let out_row = m_tile_base + out_local_row; // sorted-row space, same as x_rows
    // Writeback gemm0, then reuse OutScratch for gemm1 - identical
    // sequencing to `ffai_qmm_coop`; only the store is now guarded by the
    // ragged-tail mask (no guard was needed on the dense core, whose M is
    // always an exact BM multiple).
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
            let k = ffai_moe_gather_qmm_coop::kernel_ir_for(dt);
            assert_eq!(k.name, "ffai_moe_gather_qmm_coop");
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            let setups = all_ops().filter(|op| matches!(op, Op::CoopTileSetup { .. })).count();
            assert_eq!(setups, 2, "expected gemm0 + gemm1 CoopTileSetup ops");
            let runs = all_ops().filter(|op| matches!(op, Op::CoopTileRun { .. })).count();
            assert_eq!(
                runs, 2,
                "expected one CoopTileRun per N-half per K-block-body (unrolled once in IR)"
            );
        }
    }

    /// bf16 must stage through `half` - same requirement as every other
    /// `coop_tile_*` kernel in this family (Apple's `matmul2d` mishandles
    /// `bfloat` cooperative tensors).
    #[test]
    fn bf16_stages_through_half() {
        use ffai_kernels::core::ir::Op;
        let k = ffai_moe_gather_qmm_coop::kernel_ir_for(DType::BF16);
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

    /// Developer aid, per the F-85 constexpr-codegen audit recipe: dump the
    /// generated MSL and confirm register/tile-geometry constants (the
    /// `Xs`/`Ws`/`OutScratch` extents, the 8-wide unrolled nibble/coop
    /// loops) are literal, and that the tile-plan/gather indirection reads
    /// (`tile_expert`/`tile_row_start`/`tile_row_count`/`x_rows`) show up as
    /// ordinary device-buffer loads, not something that leaked into the
    /// `#[constexpr]` set alongside the intended `n_out`/`k_in`/
    /// `group_size` runtime scalars.
    /// `cargo test -p ffai-kernels-std --lib --release -- \
    ///   kernels::moe::moe_gather_qmm_coop::tests::dump --nocapture --test-threads=1`
    #[test]
    fn dump() {
        use ffai_kernels::codegen::msl::MslGenerator;
        let mut k = ffai_moe_gather_qmm_coop::kernel_ir_for(DType::F16);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

/// New-syntax correctness for `ffai_moe_gather_qmm_coop` against a CPU
/// dequant-then-matmul oracle that walks the SAME tile plan the kernel
/// consumes (built via `build_tile_plan_with_bm(counts, 32)`), with an
/// explicit gather (`x_rows`) indirection so the oracle exercises the same
/// source-row resolution the kernel does.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_moe_gather_qmm_coop;
    use crate::{
        kernels::moe::moe_mpp_tileplan::build_tile_plan_with_bm,
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

    /// Per-sorted-row dequant-then-matmul oracle. `tile_expert_by_row[row]`
    /// is the expert owning sorted row `row`; `x_rows[row]` is the source
    /// row `x` gathers from for it - the same two indirections the kernel's
    /// tile plan + `x_rows` buffers encode, just pre-expanded per-row for a
    /// simple CPU reference instead of walked via the tile plan.
    #[allow(clippy::too_many_arguments)]
    fn cpu_oracle(
        x: &[f32],
        weight_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        x_rows: &[u32],
        tile_expert_by_row: &[u32],
        m_total: usize,
        k_in: usize,
        n_out: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let weight_stride_m = k_in / 8;
        let groups_per_row = k_in / group_size;
        let mut out = vec![0.0f32; m_total * n_out];
        for row in 0..m_total {
            let expert = tile_expert_by_row[row] as usize;
            let src_row = x_rows[row] as usize;
            for n in 0..n_out {
                let weight_row_base = expert * n_out * weight_stride_m + n * weight_stride_m;
                let scale_row_base = expert * n_out * groups_per_row + n * groups_per_row;
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
                out[row * n_out + n] = acc;
            }
        }
        out
    }

    /// Build a `TestSetup` from a per-expert row-count fixture (skewed,
    /// possibly containing zero-count experts and ragged short/long runs)
    /// and an `x_rows` permutation. `x_table_rows` is the size of the
    /// UNSORTED activation table `x_rows` indexes into - pass `m_total` for
    /// an identity-shaped table, or larger to exercise a real gather.
    #[allow(clippy::too_many_arguments)]
    fn setup(
        kernel: ffai_kernels::core::ir::Kernel,
        counts: &[usize],
        x_rows: Vec<u32>,
        x_table_rows: usize,
        n_out: usize,
        k_in: usize,
        group_size: usize,
        dt: DType,
    ) -> TestSetup {
        let n_experts = counts.len();
        let m_total: usize = counts.iter().sum();
        assert_eq!(x_rows.len(), m_total);

        let (tile_expert, tile_row_start, tile_row_count) = build_tile_plan_with_bm(counts, 32);
        let num_tiles = tile_expert.len();

        let mut tile_expert_by_row = vec![0u32; m_total];
        for (e, &c) in counts.iter().enumerate() {
            let start: usize = counts[..e].iter().sum();
            for slot in tile_expert_by_row.iter_mut().skip(start).take(c) {
                *slot = e as u32;
            }
        }

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

        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = cpu_oracle(
            &x,
            &weight_packed,
            &s,
            &b,
            &x_rows,
            &tile_expert_by_row,
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
            .input(TestBuffer::from_vec("tile_expert", u32_bytes(&tile_expert), DType::U32))
            .input(TestBuffer::from_vec("tile_row_start", u32_bytes(&tile_row_start), DType::U32))
            .input(TestBuffer::from_vec("tile_row_count", u32_bytes(&tile_row_count), DType::U32))
            .input(TestBuffer::from_vec("x_rows", u32_bytes(&x_rows), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_out as u32 / 64, num_tiles as u32, 1, [128, 1, 1])
    }

    /// Skewed/ragged routing: three zero-count experts, one expert
    /// spanning 7 BM=32 tiles (200 rows), one exact-BM tile (32 rows, not
    /// ragged), and several short runs under 32 rows with no boundary
    /// aligned to 32 - the case that stresses both the tile-plan boundary
    /// math and the kernel's `row_count < 32` tail mask, mirroring
    /// `moe_gather_qmm_tileplan_correctness.rs`'s ragged coverage for the
    /// BM=16 sibling. Identity `x_rows` (gather correctness is covered
    /// separately below) so this test isolates the tile-plan/ragged axis.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_coop_skewed_ragged(dt: DType) -> TestSetup {
        let counts = [1usize, 0, 200, 0, 5, 22, 0, 3, 32, 9, 0, 47];
        let m_total: usize = counts.iter().sum();
        setup(
            ffai_moe_gather_qmm_coop::kernel_ir_for(dt),
            &counts,
            (0..m_total as u32).collect(),
            m_total,
            128,
            128,
            64,
            dt,
        )
    }

    /// Non-identity `x_rows`: a reversed gather over a table twice the size
    /// of `m_total`, so a bug that silently fell back to identity
    /// (row == source row) or read outside the intended half of the table
    /// fails loudly. Same skewed/ragged expert-count fixture as above.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_moe_gather_qmm_coop_xrows_gather(dt: DType) -> TestSetup {
        let counts = [4usize, 0, 60, 9, 0, 17];
        let m_total: usize = counts.iter().sum();
        let x_table_rows = m_total * 2;
        // Row i gathers from the table's back half, reversed - deliberately
        // far from the identity mapping.
        let x_rows: Vec<u32> = (0..m_total as u32).map(|i| (x_table_rows as u32 - 1) - i).collect();
        setup(
            ffai_moe_gather_qmm_coop::kernel_ir_for(dt),
            &counts,
            x_rows,
            x_table_rows,
            64,
            128,
            64,
            dt,
        )
    }

    /// Qwen3.6-35B-A3B gate/up leg shape: K=2048, N=512 (BN=64 -> 8 column
    /// tiles), skewed routing across a realistic expert count.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_moe_gather_qmm_coop_gate_up_shape(dt: DType) -> TestSetup {
        let counts = [3usize, 0, 130, 47, 0, 8, 256, 0, 19, 65, 1, 0, 512];
        let m_total: usize = counts.iter().sum();
        setup(
            ffai_moe_gather_qmm_coop::kernel_ir_for(dt),
            &counts,
            (0..m_total as u32).collect(),
            m_total,
            512,
            2048,
            64,
            dt,
        )
    }

    /// Qwen3.6-35B-A3B down leg shape: K=512, N=2048. group_size=64 means a
    /// group spans exactly 2 BK=32 K-slabs at this K (same as the gate/up
    /// leg) - this test is the K=512 scale-step verification the module
    /// docs call out.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_moe_gather_qmm_coop_down_shape(dt: DType) -> TestSetup {
        let counts = [5usize, 0, 96, 33, 0, 12, 128, 0, 27, 41, 1, 0, 256];
        let m_total: usize = counts.iter().sum();
        setup(
            ffai_moe_gather_qmm_coop::kernel_ir_for(dt),
            &counts,
            (0..m_total as u32).collect(),
            m_total,
            2048,
            512,
            64,
            dt,
        )
    }
}

/// New-syntax benchmarks for `ffai_moe_gather_qmm_coop`. Without a
/// `#[bench]` entry the kernel is invisible to `ffaik build --emit` (build
/// discovery walks the bench registry, not the test registry), so this is
/// required for the kernel to ever reach `FFAIKernels.swift`, not just for
/// isolated perf numbers.
///
/// Fixtures use `moe_mpp_shared::zipfish_counts` (a handful of heavy
/// experts, a long tail of light ones, some zero-count) at Qwen3.6-35B-A3B's
/// 256-expert count, at the mTotal values and seeds the F-85 isolated
/// GO/NO-GO comparison uses - see `moe_mpp_tileplan.rs`'s matching
/// `bench_moe_gather_qmm_coop_isolated_cmp_*` sibling entries for the BM=16
/// production incumbent at the SAME shapes/seeds (both sides call the
/// shared `zipfish_counts` helper so the routing skew each kernel sees at a
/// given `(leg, m_total, seed)` is identical, not just similarly-shaped).
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_moe_gather_qmm_coop;
    use crate::kernels::moe::{
        moe_mpp_shared::zipfish_counts,
        moe_mpp_tileplan::build_tile_plan_with_bm,
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[allow(clippy::too_many_arguments)]
    fn moe_coop_bench(
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
        let words_per_row = k_in / 8; // int4: 8 nibbles/u32
        let sz = dt.size_bytes();
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;
        let counts = zipfish_counts(m_total, n_experts, seed);
        let (tile_expert, tile_row_start, tile_row_count) = build_tile_plan_with_bm(&counts, 32);
        let num_tiles = tile_expert.len();

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
                "{label} M{m_total} N{n_out} K{k_in} E{n_experts} tiles{num_tiles} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / 64, num_tiles as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    // ── gate/up leg (K=2048, N=512) ─────────────────────────────────────
    // 3 independently-seeded Zipf draws per F-85 target mTotal.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t4096_s0(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t4096_s1(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t4096_s2(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t16384_s0(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t16384_s1(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t16384_s2(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t32768_s0(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t32768_s1(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_gate_up_t32768_s2(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }

    // ── down leg (K=512, N=2048) ────────────────────────────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t4096_s0(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t4096_s1(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t4096_s2(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t16384_s0(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t16384_s1(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t16384_s2(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t32768_s0(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t32768_s1(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_coop_isolated_down_t32768_s2(dt: DType) -> BenchSetup {
        moe_coop_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
}

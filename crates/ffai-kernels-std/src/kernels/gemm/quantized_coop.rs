//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! `ffai_qmm_coop` - wide-N cooperative-dequant int4 quantized matmul.
//!
//! Isolated benching of `ffai_qmm_mma` (manual 8x8 `simdgroup_matmul`) vs
//! `ffai_qmm_mma_mpp` (single `mpp::tensor_ops::matmul2d` call per SG per
//! K-block, BN=32) on this M5 Max showed the MPP/tensor-unit path already
//! closes most of the gap to the design target on its own (`ffai_qmm_mma`
//! flat ~23% of 57 TFLOP/s peak at every shape; `ffai_qmm_mma_mpp` reaches
//! ~42-44% at M >= 2048, still under it at M=512). This kernel is the
//! remaining lever from that isolation result: keep the tensor-unit MAC
//! path (proven the winner on M5 in the isolation bench, matching the
//! read that hardware tensor units dominate manual simdgroup MMA on
//! M5-class silicon) but double the N-tile each threadgroup covers,
//! BN=32 -> BN=64, so one X-tile device read + threadgroup stage feeds
//! two `matmul2d` calls per simdgroup instead of one. This raises the
//! output-per-load-issued ratio without touching the K-loop's barrier
//! count or growing the X tile (the M side of the tile, and thus the
//! per-K-block X device traffic, is unchanged).
//!
//! ## Geometry
//!
//! - TPG: 128 threads (4 SG x 32 lanes, WM = WN = 2). Fixed.
//! - BM = 32, BN = 64, BK = 32 -> 32x64 output tile (2048 outputs/TG,
//!   2x `ffai_qmm_mma_mpp`'s 1024).
//! - Grid: `[n/64, m/32, 1]`.
//! - Each SG owns a 16-row x 32-col quadrant of the output tile, covered
//!   by two independent 16x16x32 `matmul2d` cooperative-tensor ops
//!   (`gemm0` for the low 16 N-columns of the quadrant, `gemm1` for the
//!   high 16), each with its own persistent fp32 accumulator across the
//!   whole K loop. Both ops load the same X sub-tile (shared threadgroup
//!   read, no extra device traffic) against two different W sub-tiles.
//! - Group size baked at 64 (Qwen3.6-A3B default) - same as
//!   `ffai_qmm_mma_mpp`.
//! - Threadgroup memory: `Xs[32 x 36]` (BM x (BK+4) skew) +
//!   `Ws[64 x 36]` (BN x (BK+4) skew) live for the whole K loop;
//!   `OutScratch[4 SG x 16 x 16]` is reused sequentially for the two
//!   N-halves during the (one-time, post-K-loop) writeback so it does
//!   not have to double in size alongside `Ws`.
//!
//! Per-K-block (all 128 lanes cooperatively):
//!   1. X-tile coop-load -> `Xs[BM x TG_LD=36]` (unchanged from
//!      `ffai_qmm_mma_mpp` - 128 lanes x 8 contiguous K-elems).
//!   2. W-tile coop-dequant int4, twice the row range of
//!      `ffai_qmm_mma_mpp` (`w_row` spans 0..64 via a 2-step loop over
//!      "halves" of 32 rows, each step reusing the same 128-lane x
//!      8-nibble mapping) -> `Ws[BN x TG_LD=36]`.
//!   3. `threadgroup_barrier()`.
//!   4. Each SG: `coop_tile_load_a`/`load_b`/`run` for `gemm0`, then the
//!      same for `gemm1` (`ct_a`/`ct_b`/`ct_c` are per-named-op
//!      registers, so each needs its own load call, but both loads read
//!      the same in-flight `Xs` tile already resident in threadgroup
//!      memory - no extra device bandwidth).
//!   5. `threadgroup_barrier()`.
//!
//! Exactly 2 threadgroup-scope barriers per K-slab, matching
//! `ffai_qmm_mma`/`ffai_qmm_mma_mpp`; `matmul2d`'s cooperative-tensor
//! accumulator registers are simdgroup-scoped hardware state (immune to
//! the `#[constexpr]`-lowers-to-runtime-buffer-scalar codegen trap
//! documented on `ffai_gated_delta_prep_chunk`/`_fast` - see
//! `crates/ffai-kernels-std/src/kernels/ssm/gated_delta_prep_chunk.rs`),
//! same as `ffai_qmm_mma_mpp`'s single accumulator. `k`/`n`/`gs_per_row`
//! stay `#[constexpr]` runtime args (M/N/K themselves are not baked);
//! BM/BN/BK/group_size are genuine Rust `u32` literals in the kernel
//! body, not routed through `#[constexpr]`, so they reach MSL as literal
//! tile/loop-bound constants the same way `ffai_qmm_mma_mpp`'s already do
//! (see the `dump` test below for the MSL-level confirmation).

use ffai_kernels::kernel;

/// MPP int4 quantized matmul, BN=64 wide-N variant. Params:
///   `w [n, k/8]` int4 packed (8 nibbles/u32),
///   `scales`/`biases [n, k/group_size]` (T),
///   `x [m, k]` (T), `out [m, n]` (T). group_size = 64. Requires
///   `m % 32 == 0`, `n % 64 == 0` (no ragged-tile guard - production
///   projection shapes are exact multiples of 64; see kernel_tests for
///   the covered shapes).
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_qmm_coop<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] gs_per_row: u32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    // 2x2 warp grid; each SG's quadrant is 16 rows x 32 cols (BM/2 x BN/2).
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 32u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 64u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T)); // BM=32 x (BK+4)=36
    threadgroup_alloc("Ws", 2304u32, coop_stage(T)); // BN=64 x (BK+4)=36
    threadgroup_alloc("OutScratch", 1024u32, f32); // 4 SG x 16 x 16, reused per N-half
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
    // X coop-load lane mapping: 128 lanes x 8 contiguous K-elems = 1024 = BM*BK.
    let x_m_row = lane_in_tg / 4u32; // 0..32
    let x_k_quad = lane_in_tg & 3u32; // 0..4
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let packs_per_row = k / 8u32;
    // W coop-dequant lane mapping for the low half (w_row 0..32) and high
    // half (w_row 32..64) of the BN=64 tile - same per-lane pack/nibble
    // shape as ffai_qmm_mma_mpp, run twice with a 32-row offset.
    let w_row_lo = lane_in_tg / 4u32; // 0..32
    let w_row_hi = 32u32 + w_row_lo; // 32..64
    let w_pack_in_row = lane_in_tg & 3u32; // 0..4
    let wn_lo = w_n_base + w_row_lo;
    let wn_hi = w_n_base + w_row_hi;
    let sb_base_lo = wn_lo * gs_per_row;
    let sb_base_hi = wn_hi * gs_per_row;
    let w_pack_row_base_lo = wn_lo * packs_per_row;
    let w_pack_row_base_hi = wn_hi * packs_per_row;
    let ws_base_lo = w_row_lo * 36u32 + w_pack_in_row * 8u32;
    let ws_base_hi = w_row_hi * 36u32 + w_pack_in_row * 8u32;
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off0 = sg_n_base * 36u32; // gemm0: cols sg_n_base .. sg_n_base+16
    let ws_sg_off1 = (sg_n_base + 16u32) * 36u32; // gemm1: cols +16 .. +32
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        // 1. Stage X[x_m_base + x_m_row, kb + x_k_base..+8] -> Xs.
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        // 2. W dequant, low half (N rows w_n_base..w_n_base+32).
        let pack_dev_lo = w_pack_row_base_lo + kb / 8u32 + w_pack_in_row;
        let packed_lo = load(w[pack_dev_lo]);
        let k_off = kb + w_pack_in_row * 8u32;
        let g = k_off / 64u32;
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
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base0 = w_n_base + sg_n_base;
    let out_n_base1 = out_n_base0 + 16u32;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    // Writeback gemm0, then reuse OutScratch for gemm1 (keeps TG memory at
    // ffai_qmm_mma_mpp's single-accumulator OutScratch size).
    coop_tile_store_c("gemm0", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base0 + col)], v.cast::<T>());
    }
    threadgroup_barrier();
    coop_tile_store_c("gemm1", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    for _j in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _j;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base1 + col)], v.cast::<T>());
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
            let k = ffai_qmm_coop::kernel_ir_for(dt);
            assert_eq!(k.name, "ffai_qmm_coop");
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

    /// Developer aid, per the F-85 constexpr-codegen audit recipe: dump the
    /// generated MSL and confirm the tile/loop-bound constants (`36`, the
    /// `Xs`/`Ws`/`OutScratch` array extents, the 8-wide unrolled nibble/coop
    /// loops) are literal, and that the only `constant T &name [[buffer]]`
    /// runtime scalars are `k`/`n`/`gs_per_row` (the documented, intended
    /// M/N/K passthrough - not a tile-geometry constant leaking through).
    /// `cargo test -p ffai-kernels-std --lib --release -- \
    ///   kernels::gemm::quantized_coop::tests::dump --nocapture --test-threads=1`
    #[test]
    fn dump() {
        use ffai_kernels::codegen::msl::MslGenerator;
        let mut k = ffai_qmm_coop::kernel_ir_for(DType::F16);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

/// New-syntax correctness for `ffai_qmm_coop`. Shares the affine-int4 CPU
/// oracle (`qmm_oracle`) with the rest of the `quantized`/`quantized_mpp`
/// family via `qx_setup` - same math, only the dispatch geometry (BN=64,
/// grid `[n/64, m/32, 1]`) differs.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_qmm_coop;
    use crate::kernels::gemm::quantized::kernel_tests::qx_setup;

    // Single-TG cell: M=32, N=64 (one gemm0 + one gemm1 tile), K=512.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_qmm_coop(dt: DType) -> TestSetup {
        qx_setup(
            ffai_qmm_coop::kernel_ir_for(dt),
            32,
            64,
            512,
            4,
            64,
            true,
            [1, 1, 1],
            [128, 1, 1],
            dt,
        )
    }

    // Multi-tile: M=128 (4 M-tiles), N=256 (4 N-tiles), K=512.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_qmm_coop_multi_tile(dt: DType) -> TestSetup {
        qx_setup(
            ffai_qmm_coop::kernel_ir_for(dt),
            128,
            256,
            512,
            4,
            64,
            true,
            [4, 4, 1],
            [128, 1, 1],
            dt,
        )
    }

    // Qwen3.6-A3B-class production shape: hidden=2048, a realistic
    // batched-prefill M chunk, N wider than one TG-row of tiles.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_qmm_coop_prod_shape(dt: DType) -> TestSetup {
        qx_setup(
            ffai_qmm_coop::kernel_ir_for(dt),
            128,
            2048,
            2048,
            4,
            64,
            true,
            [32, 4, 1],
            [128, 1, 1],
            dt,
        )
    }
}

/// New-syntax benchmark for `ffai_qmm_coop`. Without a `#[bench]` entry the
/// kernel is invisible to `ffaik build --emit` (build discovery walks the
/// bench registry, not the test registry - see `quantized_mpp.rs`'s
/// `kernel_benches` module for the sibling this mirrors), so this is
/// required for the kernel to ever reach `FFAIKernels.swift`, not just for
/// isolated perf numbers.
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_qmm_coop;
    use crate::kernels::gemm::quantized::kernel_benches::qmb;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_qmm_coop(dt: DType) -> BenchSetup {
        qmb(
            ffai_qmm_coop::kernel_ir_for(dt),
            32,
            4096,
            4096,
            4,
            64,
            true,
            [64, 1, 1],
            [128, 1, 1],
            dt,
        )
    }
}

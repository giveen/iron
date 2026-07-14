//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantized matmul via `mpp::tensor_ops::matmul2d`
//! (NAX tensor engine) — the block-scaled counterpart of
//! `mlx/quantized_nax{,_int8}.rs`. `Out = X · dequant(W)` for the
//! spec-conformant + legacy float-scale + symmetric-int8 formats.
//!
//! This is the **NAX-named twin** of `mlx/block_scaled_qmm_mpp.rs`: the kernel
//! bodies are **byte-identical**, only the `mt_<fmt>_qmm_nax` fn names differ.
//! The two co-exist purely for naming compatibility (NAX = MPP, both dispatch
//! through `mpp::tensor_ops::matmul2d`) — exactly like `quantized_nax.rs`
//! mirrors `quantized_mpp.rs`. Consumers pick the `_nax` vs `_mpp` name in their
//! own dispatch tables.
//!
//! The **dispatch geometry and cooperative-matmul tail are byte-identical** to
//! the proven int4/int8 NAX kernels — TPG 128 (4 SG × 32), BM=BN=BK=32, grid
//! `[n/32, m/32, 1]`, the 2×2 warp grid, `Xs`/`Ws`/`OutScratch` threadgroup
//! tiles, and the `coop_tile_*` ops. Only the **W-dequant staging** differs:
//! `element_decode(code) · block_scale` (no bias) instead of the affine
//! `scale·q + bias`. W is dequantized to `coop_stage(T)` as it lands in `Ws`,
//! so the tensor engine sees the same fp16/fp32 tile in every format.
//!
//! Weight layout (per N-row): 4-bit `w [n, k/8] u32` (8 E2M1 nibbles/word),
//! 8-bit `w [n, k] u8` (one E4M3/E5M2/int8 code per byte). Scales
//! `[n, k/block_size]` are u8 (E8M0/E4M3) or f32 (nvfp8 / legacy fp / int8).
//! `block_size` divides the per-lane 8-K-element stripe (8 ≤ block_size, and
//! the stripe is 8-aligned), so one scale load per lane per K-block is exact.
//! `KernelMode::Reduction`. fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 +
//! f32-scale shape). Codegen-only; correctness pinned by the `#[test_kernel]`s.

use ffai_kernels::kernel;

/// Block-scaled NAX (cooperative-tensor) dequantizing matmul, folded over the
/// 28-format axis (§7). `out = X · dequant(W)` via a 16×16×32 `matmul2d`
/// (2×2 simdgroups). Only the per-format W-staging decode into the `Ws`
/// cooperative tile folds onto `(BITS, WDEC, SKIND)` (buffer types `(WT, ST)`,
/// legend in `block_scaled_matmul`); the X-staging, matmul, and write-back are
/// format-independent. Decodes through `kernels/primitives.rs`. Produces
/// `mt_<FMT>_qmm_nax`.
#[kernel(variants(
    (FMT,          BITS,  WT,  ST,  WDEC, SKIND) = [
        (mxfp4,        4u32, u32, u8,  0u32, 0u32),
        (nvfp4,        4u32, u32, u8,  0u32, 1u32),
        (fp4,          4u32, u32, f32, 0u32, 2u32),
        (fp4_f16,      4u32, u32, f16, 0u32, 2u32),
        (int2,         2u32, u32, f32, 1u32, 2u32),
        (int3,         3u32, u32, f32, 1u32, 2u32),
        (int4,         4u32, u32, f32, 1u32, 2u32),
        (int5,         5u32, u32, f32, 1u32, 2u32),
        (int6,         6u32, u32, f32, 1u32, 2u32),
        (mxint2,       2u32, u32, u8,  1u32, 0u32),
        (mxint3,       3u32, u32, u8,  1u32, 0u32),
        (mxint4,       4u32, u32, u8,  1u32, 0u32),
        (mxint5,       5u32, u32, u8,  1u32, 0u32),
        (mxint6,       6u32, u32, u8,  1u32, 0u32),
        (int2_f16,     2u32, u32, f16, 1u32, 2u32),
        (int3_f16,     3u32, u32, f16, 1u32, 2u32),
        (int4_f16,     4u32, u32, f16, 1u32, 2u32),
        (int5_f16,     5u32, u32, f16, 1u32, 2u32),
        (int6_f16,     6u32, u32, f16, 1u32, 2u32),
        (mxfp8_e4m3,   8u32, u8,  u8,  2u32, 0u32),
        (mxfp8_e5m2,   8u32, u8,  u8,  3u32, 0u32),
        (mxint8,       8u32, u8,  u8,  4u32, 0u32),
        (nvfp8,        8u32, u8,  f32, 2u32, 2u32),
        (fp8_e5m2,     8u32, u8,  f32, 3u32, 2u32),
        (int8,         8u32, u8,  f32, 4u32, 2u32),
        (nvfp8_f16,    8u32, u8,  f16, 2u32, 2u32),
        (fp8_e5m2_f16, 8u32, u8,  f16, 3u32, 2u32),
        (int8_f16,     8u32, u8,  f16, 4u32, 2u32),
    ],
    suffix = "{FMT}_qmm_nax",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    w: Tensor<WT>,
    scales: Tensor<ST>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let sg_m_base = sm * 16u32;
    let sg_n_base = sn * 16u32;
    let x_m_base = tgid_y * 32u32;
    let w_n_base = tgid_x * 32u32;
    threadgroup_alloc("Xs", 1152u32, coop_stage(T));
    threadgroup_alloc("Ws", 1152u32, coop_stage(T));
    threadgroup_alloc("OutScratch", 1024u32, f32);
    coop_tile_setup(
        "gemm",
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
    coop_tile_zero("gemm");
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    let x_ws_base = x_m_row * 36u32 + x_k_base;
    let gs_per_row = k / block_size;
    let wn_plus_wr = w_n_base + x_m_row;
    let sb_base = wn_plus_wr * gs_per_row;
    // Per-WDEC weight row bases (only the matching one is used; others DCE).
    let w_pack_row_base = wn_plus_wr * (k / 8u32); // nibble: 8 codes / u32
    let w_word_row_base = wn_plus_wr * (k * BITS / 32u32); // sub-byte: bit-stream
    let w_row_base = wn_plus_wr * k; // byte: one code / byte
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    let xs_sg_off = sg_m_base * 36u32;
    let ws_sg_off = sg_n_base * 36u32;
    let sg_scratch_off = sg * 256u32;
    for kb in range(0u32, k, 32u32) {
        // Stage this lane's 8 X elements (format-independent).
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        for _i in range(0u32, 8u32, 1u32) {
            let xv = load(x[x_row_dev_base + _i]).cast::<f32>();
            threadgroup_store("Xs", x_ws_base + _i, xv);
        }
        // One block scale for this lane's 8 consecutive (8-aligned) W codes.
        let k_off = kb + x_k_base;
        let sraw = load(scales[sb_base + k_off / block_size]);
        let scale = if SKIND == 0u32 {
            exp2(sraw.cast::<f32>() - 127.0f32)
        } else if SKIND == 1u32 {
            mt_decode_e4m3(sraw.cast::<u32>()) * global
        } else {
            sraw.cast::<f32>()
        };
        // Decode 8 W codes into the Ws cooperative tile.
        if WDEC == 0u32 {
            let packed = load(w[w_pack_row_base + kb / 8u32 + x_k_quad]);
            for _i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (_i * 4u32)) & 15u32;
                threadgroup_store("Ws", x_ws_base + _i, mt_decode_e2m1(nib) * scale);
            }
        } else if WDEC == 1u32 {
            for _i in range(0u32, 8u32, 1u32) {
                let bit_off = (k_off + _i) * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                let spill = BITS - lo_bits;
                let w0 = load(w[w_word_row_base + word_idx]);
                let w1 = load(w[w_word_row_base + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                let qf = q.cast::<f32>();
                let elem = select(q >= half, qf - full, qf);
                threadgroup_store("Ws", x_ws_base + _i, elem * scale);
            }
        } else {
            for _i in range(0u32, 8u32, 1u32) {
                let raw = load(w[w_row_base + k_off + _i]).cast::<u32>();
                let elem = if WDEC == 2u32 {
                    mt_decode_e4m3(raw)
                } else if WDEC == 3u32 {
                    mt_decode_e5m2(raw)
                } else {
                    mt_decode_int8(raw)
                };
                threadgroup_store("Ws", x_ws_base + _i, elem * scale);
            }
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 36u32, 16u32, xs_sg_off);
        coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 36u32, 16u32, ws_sg_off);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "OutScratch", true, f32, 16u32, 16u32, sg_scratch_off);
    threadgroup_barrier();
    let out_m_base = x_m_base + sg_m_base;
    let out_n_base = w_n_base + sg_n_base;
    let o_row = lane / 2u32;
    let o_col_base = (lane & 1u32) * 8u32;
    for _i in range(0u32, 8u32, 1u32) {
        let col = o_col_base + _i;
        let v = threadgroup_load("OutScratch", sg_scratch_off + o_row * 16u32 + col);
        store(out[(out_m_base + o_row) * n + (out_n_base + col)], v.cast::<T>());
    }
}
pub mod kernel_tests {
    use ffai_kernels::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// `out[mr,nc] = Σ_k x[mr,k] · dequant(W)[nc,k]` — W block-scaled `[n,k]`.
    fn nax_setup(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> TestSetup {
        let w: Vec<f32> = (0..n * k)
            .map(|i| {
                let r = (i / k) as f32;
                let c = (i % k) as f32;
                let mag = (0.4 + (r % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15);
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::kernels::quant::format::pack(fmt, &w, n, k);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, n, k);
        let x_f: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let mut expected = vec![0.0f32; m * n];
        for mr in 0..m {
            for nc in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x[mr * k + kk] * wdq[nc * k + kk];
                }
                expected[mr * n + nc] = acc;
            }
        }
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as native half; E8M0/E4M3 scales as
        // one byte. Both axes are driven off the format so new integer + fp16
        // formats pick up the right buffer types (4-bit collapses to the old
        // `== 4` branch, no regression).
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("w", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (n / 32) as u32,
            (m / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // m=32, n=64, k=512 (divisible by 16/32/64) — mirrors the int8 NAX test.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp4_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxfp4_qmm_nax::kernel_ir_for(dt), QFormat::Mxfp4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_nvfp4_qmm_nax::kernel_ir_for(dt), QFormat::Nvfp4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_fp4_qmm_nax::kernel_ir_for(dt), QFormat::Fp4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxfp8_e4m3_qmm_nax::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxfp8_e5m2_qmm_nax::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_fp8_e5m2_qmm_nax::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_nvfp8_qmm_nax::kernel_ir_for(dt), QFormat::Nvfp8, 32, 64, 512, dt)
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_nvfp8_qmm_nax::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int8_qmm_nax::kernel_ir_for(dt), QFormat::Int8, 32, 64, 512, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). k=512 is a multiple of 32, so
    // each W row's bit-stream is word-aligned for every width; the kernel and
    // oracle share the codec, so the GPU output tracks the dequant-then-matmul
    // reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int2_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int2_qmm_nax::kernel_ir_for(dt), QFormat::Int2, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int3_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int3_qmm_nax::kernel_ir_for(dt), QFormat::Int3, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int4_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int4_qmm_nax::kernel_ir_for(dt), QFormat::Int4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int5_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int5_qmm_nax::kernel_ir_for(dt), QFormat::Int5, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int6_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int6_qmm_nax::kernel_ir_for(dt), QFormat::Int6, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint2_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxint2_qmm_nax::kernel_ir_for(dt), QFormat::Mxint2, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint3_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxint3_qmm_nax::kernel_ir_for(dt), QFormat::Mxint3, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint4_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxint4_qmm_nax::kernel_ir_for(dt), QFormat::Mxint4, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint5_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxint5_qmm_nax::kernel_ir_for(dt), QFormat::Mxint5, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint6_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxint6_qmm_nax::kernel_ir_for(dt), QFormat::Mxint6, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxint8_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_mxint8_qmm_nax::kernel_ir_for(dt), QFormat::Mxint8, 32, 64, 512, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest pair with
    // their own kernel. Same dims/geometry as the FP32 twins — only the scale
    // tensor binds as native half.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_nvfp8_f16_qmm_nax::kernel_ir_for(dt), QFormat::Nvfp8F16, 32, 64, 512, dt)
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_nvfp8_f16_qmm_nax::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_fp4_f16_qmm_nax::kernel_ir_for(dt), QFormat::Fp4F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_fp8_e5m2_f16_qmm_nax::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int2_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int2F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int3_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int3F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int4_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int4F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int5_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int5F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int6_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int6F16, 32, 64, 512, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_qmm_nax(dt: DType) -> TestSetup {
        nax_setup(mt_int8_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int8F16, 32, 64, 512, dt)
    }
}

/// NAX tensor-engine matmul benches at a 128×4096×4096 tile shape.
pub mod kernel_benches {
    use ffai_kernels::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn nax_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> BenchSetup {
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        // `bitstream_words` collapses to the old `n*k/8` for 4-bit, so the
        // pre-existing formats are unchanged.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n * k, DType::U8)
        } else {
            (crate::kernels::quant::format::bitstream_words(n * k, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = n * (k / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m * k * sz
            + m * n * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("w", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("x", m * k, dt))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((n / 32) as u32, (m / 32) as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m as u64 * n as u64 * k as u64)
            .with_shape_label(format!("{} m={m} n={n} k={k}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4(dt: DType) -> BenchSetup {
        nax_bench(mt_mxfp4_qmm_nax::kernel_ir_for(dt), QFormat::Mxfp4, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4(dt: DType) -> BenchSetup {
        nax_bench(mt_nvfp4_qmm_nax::kernel_ir_for(dt), QFormat::Nvfp4, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4(dt: DType) -> BenchSetup {
        nax_bench(mt_fp4_qmm_nax::kernel_ir_for(dt), QFormat::Fp4, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3(dt: DType) -> BenchSetup {
        nax_bench(mt_mxfp8_e4m3_qmm_nax::kernel_ir_for(dt), QFormat::Mxfp8E4, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2(dt: DType) -> BenchSetup {
        nax_bench(mt_mxfp8_e5m2_qmm_nax::kernel_ir_for(dt), QFormat::Mxfp8E5, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2(dt: DType) -> BenchSetup {
        nax_bench(mt_fp8_e5m2_qmm_nax::kernel_ir_for(dt), QFormat::Fp8E5m2, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8(dt: DType) -> BenchSetup {
        nax_bench(mt_nvfp8_qmm_nax::kernel_ir_for(dt), QFormat::Nvfp8, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        nax_bench(mt_int8_qmm_nax::kernel_ir_for(dt), QFormat::Int8, 128, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2(dt: DType) -> BenchSetup {
        nax_bench(mt_int2_qmm_nax::kernel_ir_for(dt), QFormat::Int2, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3(dt: DType) -> BenchSetup {
        nax_bench(mt_int3_qmm_nax::kernel_ir_for(dt), QFormat::Int3, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4(dt: DType) -> BenchSetup {
        nax_bench(mt_int4_qmm_nax::kernel_ir_for(dt), QFormat::Int4, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5(dt: DType) -> BenchSetup {
        nax_bench(mt_int5_qmm_nax::kernel_ir_for(dt), QFormat::Int5, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6(dt: DType) -> BenchSetup {
        nax_bench(mt_int6_qmm_nax::kernel_ir_for(dt), QFormat::Int6, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2(dt: DType) -> BenchSetup {
        nax_bench(mt_mxint2_qmm_nax::kernel_ir_for(dt), QFormat::Mxint2, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3(dt: DType) -> BenchSetup {
        nax_bench(mt_mxint3_qmm_nax::kernel_ir_for(dt), QFormat::Mxint3, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4(dt: DType) -> BenchSetup {
        nax_bench(mt_mxint4_qmm_nax::kernel_ir_for(dt), QFormat::Mxint4, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5(dt: DType) -> BenchSetup {
        nax_bench(mt_mxint5_qmm_nax::kernel_ir_for(dt), QFormat::Mxint5, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6(dt: DType) -> BenchSetup {
        nax_bench(mt_mxint6_qmm_nax::kernel_ir_for(dt), QFormat::Mxint6, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8(dt: DType) -> BenchSetup {
        nax_bench(mt_mxint8_qmm_nax::kernel_ir_for(dt), QFormat::Mxint8, 128, 4096, 4096, dt)
    }
    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_nvfp8_f16_qmm_nax::kernel_ir_for(dt), QFormat::Nvfp8F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_nvfp8_f16_qmm_nax::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_fp4_f16_qmm_nax::kernel_ir_for(dt), QFormat::Fp4F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16(dt: DType) -> BenchSetup {
        nax_bench(
            mt_fp8_e5m2_f16_qmm_nax::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            128,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_int2_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int2F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_int3_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int3F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_int4_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int4F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_int5_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int5F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_int6_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int6F16, 128, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16(dt: DType) -> BenchSetup {
        nax_bench(mt_int8_f16_qmm_nax::kernel_ir_for(dt), QFormat::Int8F16, 128, 4096, 4096, dt)
    }
}

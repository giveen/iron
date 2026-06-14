//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **dequantizing GEMM** (multi-row matmul) kernels — the qmm
//! counterpart of the GEMVs in [`super::block_scaled_matmul`]:
//! `output[m, n] = Σ_k dequant(weight[n, k]) · x[m, k]` for the spec-conformant
//! formats (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! Each `(m, n)` output element is one threadgroup that reduces over K — the
//! same proven Reduction geometry as the GEMVs, just flattened into a 1-D grid
//! of `out_dim · m_rows` threadgroups so it depends only on `program_id::<0>()`
//! (no 2-D grid assumptions). `tg → (mr = tg / out_dim, n = tg − mr·out_dim)`,
//! and `output[tg]` is exactly `output[mr·out_dim + n]`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim·m_rows, 1, 1]`, `tpg = [TPG, 1, 1]`
//!   with TPG ≥ 32 & a multiple of 32. One TG per output element.
//! - Weight/scale layouts + the `block_size | 8` packing rule are identical to
//!   the GEMVs (see [`super::block_scaled_matmul`]). `x` is `[m_rows, in_dim]`,
//!   `output` is `[m_rows, out_dim]`, both row-major.

use metaltile::kernel;

/// Block-scaled dequantizing GEMM, folded over the 28-format axis (§7).
///
/// `output[mr, n] = Σ_k dequant(weight[n, k]) · x[mr, k]`. One threadgroup per
/// `(m-row, output-col)` pair; `reduce_sum` folds the K partials (TPG ≥ 32,
/// multiple of 32). Per-element decode + per-block scale are selected by the
/// `(BITS, WDEC, SKIND)` co-vars; weight/scale buffer types by `(WT, ST)`:
///   WDEC 0 = E2M1 nibble (pack-strided), 1 = sub-byte int bit-stream,
///        2 = E4M3 byte, 3 = E5M2 byte, 4 = int8 byte.
///   SKIND 0 = E8M0 pow-2 (u8), 1 = E4M3 micro × global (u8, nvfp4),
///         2 = direct per-block scale (f32 / f16).
/// Decodes through the shared `kernels/primitives.rs`. Produces `mt_<FMT>_qmm`.
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
    suffix = "{FMT}_qmm",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    x: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let tg = program_id::<0>();
    let mr = tg / out_dim;
    let n = tg - mr * out_dim;
    let n_blocks = in_dim / block_size;
    let n_block_off = n * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;

    if WDEC == 0u32 {
        // E2M1 nibble, pack-strided over output-col n's weight row.
        let n_packs_per_row = in_dim / 8u32;
        let packs_per_block = block_size / 8u32;
        let row_pack_off = n * n_packs_per_row;
        let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
        for p_iter in range(0u32, p_iters, 1u32) {
            let pack_idx = p_iter * lsize + tid;
            if pack_idx < n_packs_per_row {
                let blk = pack_idx / packs_per_block;
                let sraw = load(scales[n_block_off + blk]);
                let scale = if SKIND == 0u32 {
                    exp2(sraw.cast::<f32>() - 127.0f32)
                } else if SKIND == 1u32 {
                    mt_decode_e4m3(sraw.cast::<u32>()) * global
                } else {
                    sraw.cast::<f32>()
                };
                let packed = load(weight[row_pack_off + pack_idx]);
                let p_off = pack_idx * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                    acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                }
            }
        }
    } else {
        // Element-strided: one code per thread-iter (sub-byte int or byte float).
        let half = 1u32 << (BITS - 1u32);
        let full = (1u32 << BITS).cast::<f32>();
        let words_per_row = in_dim * BITS / 32u32;
        let row_word_off = n * words_per_row;
        let row_off = n * in_dim;
        let iters = (in_dim + lsize - 1u32) / lsize;
        for it in range(0u32, iters, 1u32) {
            let c = it * lsize + tid;
            if c < in_dim {
                let blk = c / block_size;
                let sraw = load(scales[n_block_off + blk]);
                let scale = if SKIND == 0u32 {
                    exp2(sraw.cast::<f32>() - 127.0f32)
                } else {
                    sraw.cast::<f32>()
                };
                let val = if WDEC == 1u32 {
                    let bit_off = c * BITS;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                    let spill = BITS - lo_bits;
                    let w0 = load(weight[row_word_off + word_idx]);
                    let w1 = load(
                        weight[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf)
                } else {
                    let raw = load(weight[row_off + c]).cast::<u32>();
                    if WDEC == 2u32 {
                        mt_decode_e4m3(raw)
                    } else if WDEC == 3u32 {
                        mt_decode_e5m2(raw)
                    } else {
                        mt_decode_int8(raw)
                    }
                };
                acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[tg], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{block_scaled_qmm_oracle, block_scaled_weights, pack_f32, unpack_f32},
    };

    /// Reduction-contract threadgroup width (≥ 32, multiple of 32).
    const TPG: u32 = 64;

    fn qmm_setup(
        kernel: Kernel,
        fmt: QFormat,
        m_rows: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = block_scaled_weights(out_dim, in_dim);
        let p = crate::kernels::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_dim, in_dim);
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = block_scaled_qmm_oracle(&wdq, &x, m_rows, in_dim, out_dim);
        // 8-bit codes bind as one uchar each; everything sub-byte (E2M1 nibbles
        // + int2-6 bit-streams) binds as `DType::U32`. FP32 scales bind as f32,
        // FP16 scales as f16; E8M0/E4M3 scales as one byte.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("output", m_rows * out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            (out_dim * m_rows) as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // m_rows 3, out_dim 4, in_dim 256 (divisible by both block sizes).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp4_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp4_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp8_e4m3_qmm::kernel_ir_for(dt), QFormat::Mxfp8E4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxfp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Mxfp8E5, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 3, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // in_dim 256 is a multiple of int8's block_size (64), so all formats fit.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp4_qmm::kernel_ir_for(dt), QFormat::Fp4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int8_qmm::kernel_ir_for(dt), QFormat::Int8, 3, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0, block 32). The kernel and oracle
    // share the codec, so the GPU output matches the oracle to float precision
    // regardless of how coarse the quantization is. in_dim 256 is a multiple of
    // 32, so `in_dim * bits` is u32-aligned for every sub-byte width and is also
    // divisible by both block sizes (int 64, mxint 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int2_qmm::kernel_ir_for(dt), QFormat::Int2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int3_qmm::kernel_ir_for(dt), QFormat::Int3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int4_qmm::kernel_ir_for(dt), QFormat::Int4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int5_qmm::kernel_ir_for(dt), QFormat::Int5, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int6_qmm::kernel_ir_for(dt), QFormat::Int6, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint2_qmm::kernel_ir_for(dt), QFormat::Mxint2, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint3_qmm::kernel_ir_for(dt), QFormat::Mxint3, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint4_qmm::kernel_ir_for(dt), QFormat::Mxint4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint5_qmm::kernel_ir_for(dt), QFormat::Mxint5, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint6_qmm::kernel_ir_for(dt), QFormat::Mxint6, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_mxint8_qmm::kernel_ir_for(dt), QFormat::Mxint8, 3, 4, 256, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Nvfp8F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp4_f16_qmm::kernel_ir_for(dt), QFormat::Fp4F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_fp8_e5m2_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int2_f16_qmm::kernel_ir_for(dt), QFormat::Int2F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int3_f16_qmm::kernel_ir_for(dt), QFormat::Int3F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int4_f16_qmm::kernel_ir_for(dt), QFormat::Int4F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int5_f16_qmm::kernel_ir_for(dt), QFormat::Int5F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int6_f16_qmm::kernel_ir_for(dt), QFormat::Int6F16, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_qmm(dt: DType) -> TestSetup {
        qmm_setup(mt_int8_f16_qmm::kernel_ir_for(dt), QFormat::Int8F16, 3, 4, 256, dt)
    }
}

/// Batched-decode (m=32) GEMM benches at N=K=4096 — the compute-throughput
/// precision ranking. Random packed buffers (throughput is data-independent).
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn qmm_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = out_dim * (in_dim / fmt.block_size());
        // 8-bit codes are one uchar each; sub-byte codes (E2M1 nibbles + int2-6
        // bit-streams) tight-bit-pack into u32 words (with a guard word for
        // straddling 3/5/6-bit reads).
        let n_elems = out_dim * in_dim;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_elems, DType::U8)
        } else {
            (
                crate::kernels::quant::format::bitstream_words(n_elems, fmt.element_bits()),
                DType::U32,
            )
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m * in_dim * sz
            + m * out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::zeros("output", m * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_dim * m) as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m as u64 * out_dim as u64 * in_dim as u64) // GEMM: 2·M·N·K
            .with_shape_label(format!("{} m={m} n={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp4_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp4_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp8_e4m3_qmm::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxfp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp4_qmm::kernel_ir_for(dt), QFormat::Fp4, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp8_e5m2_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int8_qmm::kernel_ir_for(dt), QFormat::Int8, 32, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int2_qmm::kernel_ir_for(dt), QFormat::Int2, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int3_qmm::kernel_ir_for(dt), QFormat::Int3, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int4_qmm::kernel_ir_for(dt), QFormat::Int4, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int5_qmm::kernel_ir_for(dt), QFormat::Int5, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int6_qmm::kernel_ir_for(dt), QFormat::Int6, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint2_qmm::kernel_ir_for(dt), QFormat::Mxint2, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint3_qmm::kernel_ir_for(dt), QFormat::Mxint3, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint4_qmm::kernel_ir_for(dt), QFormat::Mxint4, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint5_qmm::kernel_ir_for(dt), QFormat::Mxint5, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint6_qmm::kernel_ir_for(dt), QFormat::Mxint6, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_mxint8_qmm::kernel_ir_for(dt), QFormat::Mxint8, 32, 4096, 4096, dt)
    }
    // FP16-scale twins. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Nvfp8F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_nvfp8_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp4_f16_qmm::kernel_ir_for(dt), QFormat::Fp4F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_fp8_e5m2_f16_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int2_f16_qmm::kernel_ir_for(dt), QFormat::Int2F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int3_f16_qmm::kernel_ir_for(dt), QFormat::Int3F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int4_f16_qmm::kernel_ir_for(dt), QFormat::Int4F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int5_f16_qmm::kernel_ir_for(dt), QFormat::Int5F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int6_f16_qmm::kernel_ir_for(dt), QFormat::Int6F16, 32, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_qmm(dt: DType) -> BenchSetup {
        qmm_bench(mt_int8_f16_qmm::kernel_ir_for(dt), QFormat::Int8F16, 32, 4096, 4096, dt)
    }
}

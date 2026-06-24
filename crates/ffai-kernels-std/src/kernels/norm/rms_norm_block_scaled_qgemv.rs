//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **RMSNorm + block-scaled dequantizing GEMV** for decode (single-token),
//! for the spec-conformant float formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 /
//! nvfp8), the legacy float-scale fp4 / fp8 + symmetric int8, and the symmetric
//! integer family (int2/3/4/5/6 + MXINT2..6 + MXINT8).
//!
//! `y = qmatmul(rms_norm(x) · norm_weight, W_q)` in one dispatch — the int4
//! fusion of `kernels/norm/rms_norm_qgemv.rs` (`mt_rms_norm_qgemv`, the simple
//! one-row-per-TG variant) with the block-scaled weight decode of
//! `mlx/block_scaled_matmul.rs`. The normalized activation never leaves
//! registers between the RMSNorm reduce and the matvec.
//!
//! ## DISPATCH INVARIANTS (identical to the proven int4 + block-scaled GEMVs)
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [TPG, 1, 1]` with
//!   TPG ≥ 32 and a multiple of 32 (tests/benches use 64). One TG per row.
//! - Phase 1: per-thread Σx², `mt_rms_inv_scalar` does the TG reduce + rsqrt.
//! - Phase 2: the pack-/element-strided block-scaled GEMV of
//!   `block_scaled_matmul.rs`, feeding on `normed[i] = x[i]·norm_weight[i]·inv_rms`.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8,
//!   and `in_dim · bits` a multiple of 32 for the sub-byte int widths so each
//!   row's tight bit-stream is u32-word-aligned.
//! - weight `[out_dim, in_dim]` u8 (8-bit) or, for every sub-byte width
//!   (4-bit nibble packs + int2/3/5/6 tight bit-streams), `bitstream_words` u32
//!   words per the row's LSB-first bit-stream; scales
//!   `[out_dim, in_dim/block_size]` (u8 E8M0/E4M3, or f32 for nvfp8 / int*).
//!
//! Block-scaled formats carry **no bias** (the int affine scale+bias path lives
//! in `rms_norm_qgemv.rs`); the accumulation is `dequant(W)·normed`.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use ffai_kernels::kernel;

/// Fused RMSNorm + block-scaled dequantizing GEMV, folded over the 28-format
/// axis (§7). Phase 1 computes `inv_rms` over `x`; phase 2 does the GEMV on the
/// normalized activation `normed[i] = x[i]·norm_weight[i]·inv_rms`. Per-element
/// weight decode + per-block scale by the `(BITS, WDEC, SKIND)` co-vars; buffer
/// types by `(WT, ST)` — see `gemm/block_scaled_matmul` for the legend. Decodes
/// through `kernels/primitives.rs`. Produces `mt_<FMT>_rms_norm_qgemv`.
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
    suffix = "{FMT}_rms_norm_qgemv",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    x: Tensor<T>,
    norm_weight: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    output: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let row = program_id::<0>();
    // Phase 1: RMSNorm — per-thread Σx², TG reduce + rsqrt via cross-kernel call.
    let mut ssq = 0.0f32;
    let n_iters = (in_dim + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let d = _iter * lsize + tid;
        if d < in_dim {
            let v = load(x[d]).cast::<f32>();
            ssq = ssq + v * v;
        }
    }
    let inv_rms = mt_rms_inv_scalar(ssq, eps_buf, in_dim);
    // Phase 2: block-scaled GEMV over `normed[i] = x[i]·norm_weight[i]·inv_rms`.
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;

    if WDEC == 0u32 {
        let n_packs_per_row = in_dim / 8u32;
        let packs_per_block = block_size / 8u32;
        let row_pack_off = row * n_packs_per_row;
        let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
        for p_iter in range(0u32, p_iters, 1u32) {
            let pack_idx = p_iter * lsize + tid;
            if pack_idx < n_packs_per_row {
                let blk = pack_idx / packs_per_block;
                let sraw = load(scales[row_block_off + blk]);
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
                    let d = p_off + i;
                    let normed =
                        load(x[d]).cast::<f32>() * load(norm_weight[d]).cast::<f32>() * inv_rms;
                    acc = acc + (val * scale) * normed;
                }
            }
        }
    } else {
        let half = 1u32 << (BITS - 1u32);
        let full = (1u32 << BITS).cast::<f32>();
        let words_per_row = in_dim * BITS / 32u32;
        let row_word_off = row * words_per_row;
        let row_off = row * in_dim;
        let iters = (in_dim + lsize - 1u32) / lsize;
        for it in range(0u32, iters, 1u32) {
            let c = it * lsize + tid;
            if c < in_dim {
                let blk = c / block_size;
                let sraw = load(scales[row_block_off + blk]);
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
                let normed =
                    load(x[c]).cast::<f32>() * load(norm_weight[c]).cast::<f32>() * inv_rms;
                acc = acc + (val * scale) * normed;
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use ffai_kernels::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{block_scaled_weights, pack_f32, unpack_f32},
    };

    /// One TG-row's lanes; ≥ 32 and a multiple of 32 (the Reduction contract).
    const TPG: u32 = 64;
    const EPS: f32 = 1e-5;

    /// Fused RMSNorm + dequant-dot reference:
    /// `inv_rms = 1/√(mean(x²)+eps)`, `out[r] = Σ_c W_dq[r,c]·(x[c]·nw[c]·inv_rms)`.
    fn rms_qgemv_oracle(
        wdq: &[f32],
        x: &[f32],
        nw: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let ssq: f32 = x.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / (ssq / in_dim as f32 + EPS).sqrt();
        (0..out_dim)
            .map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * (x[c] * nw[c] * inv_rms)).sum())
            .collect()
    }

    fn rms_qgemv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = block_scaled_weights(out_dim, in_dim);
        let p = crate::kernels::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_dim, in_dim);
        // Round x / norm_weight through `dt` so the oracle sees what the GPU sees.
        let x_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.05 + 0.1).collect();
        let nw_f: Vec<f32> = (0..in_dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let nw = unpack_f32(&pack_f32(&nw_f, dt), dt);
        let expected = rms_qgemv_oracle(&wdq, &x, &nw, out_dim, in_dim);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as native half; E8M0/E4M3 scales as one
        // byte. Both axes are driven off the format so new integer / fp16 formats
        // pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&nw_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .input(TestBuffer::from_vec("eps_buf", EPS.to_le_bytes().to_vec(), DType::F32))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            out_dim as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // out_dim 4, in_dim 256 (divisible by both block sizes 16 / 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_nvfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_mxfp8_e4m3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_mxfp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // in_dim 256 is a multiple of int8's block_size 64 (= 4 × 64).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_fp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_fp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each row's bit-stream is
    // word-aligned. The kernel and oracle share the codec, so the fused output
    // tracks the RMSNorm + dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int3_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int5_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int6_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint3_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint5_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint6_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_mxint8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxint8, 4, 256, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode in
    // their own clone. Same dims (in_dim 256) satisfy every block / bit-stream
    // alignment, and the codec round-trip keeps the fused output on the oracle.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_fp4_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(
            mt_fp8_e5m2_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int2_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int3_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int4_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int5_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int6_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_rms_norm_qgemv(dt: DType) -> TestSetup {
        rms_qgemv_setup(mt_int8_f16_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8F16, 4, 256, dt)
    }
}

/// Decode-shape (single-token) benches at the canonical hidden = out = 4096 so
/// the GFLOP/s + roofline columns rank precisions side by side. Throughput is
/// data-independent, so packed weight/scale buffers are random bytes.
pub mod kernel_benches {
    use ffai_kernels::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn rms_qgemv_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = out_dim * (in_dim / fmt.block_size());
        let n = out_dim * in_dim;
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n, DType::U8)
        } else {
            (crate::kernels::quant::format::bitstream_words(n, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + in_dim * sz   // x
            + in_dim * sz   // norm_weight
            + out_dim * sz; // output
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", in_dim, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .buffer(BenchBuffer::random("eps_buf", 1, DType::F32))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64) // GEMV (B=1): 2·N·K
            .with_shape_label(format!("{} m={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_mxfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_nvfp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxfp8_e4m3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxfp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_fp4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_nvfp8_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_fp8_e5m2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int8_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int2_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int3_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int4_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int5_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(mt_int6_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint2_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint3_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint4_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint5_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint6_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_mxint8_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            4096,
            4096,
            dt,
        )
    }
    // FP16-scale twins (fp8_e4m3_f16 reuses the nvfp8_f16 kernel).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_nvfp8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_fp4_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_fp8_e5m2_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int2_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int3_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int4_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int5_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int6_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_rms_norm_qgemv(dt: DType) -> BenchSetup {
        rms_qgemv_bench(
            mt_int8_f16_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            4096,
            4096,
            dt,
        )
    }
}

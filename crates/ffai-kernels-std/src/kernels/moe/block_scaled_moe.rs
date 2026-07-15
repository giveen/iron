//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **MoE gather-GEMM** kernels — per-token expert-routed matmul:
//! `output[m, n] = Σ_k dequant(weight[expert_ids[m], n, k]) · x[m, k]` for the
//! spec-conformant formats (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! Identical to [`super::block_scaled_qmm`] except the weight row is selected by
//! the per-token expert id: the expert stack is one `[E·out_dim, in_dim]` packed
//! tensor, so row `expert_ids[m]·out_dim + n` addresses expert `e`'s output row
//! `n`. Packing the whole stack in one call keeps nvfp4's single global FP32
//! valid across experts (no per-expert scale bookkeeping).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim·m_rows, 1, 1]`, `tpg = [TPG, 1, 1]`
//!   (TPG ≥ 32 & multiple of 32) — same as qmm; only the weight/scale row offset
//!   gains the `expert·out_dim` term.
//! - `weight` is the `[E·out_dim, …]` packed stack; `scales` likewise; layouts +
//!   the `block_size | 8` rule match the GEMV/GEMM kernels. `expert_ids` is
//!   `[m_rows]` u32, `x` is `[m_rows, in_dim]`, `output` is `[m_rows, out_dim]`.

use ffai_kernels::kernel;

/// Block-scaled MoE gather-GEMM, folded over the 28-format axis (§7).
///
/// `output[mr, n] = Σ_k dequant(weight[expert_ids[mr], n, k]) · x[mr, k]` — one
/// threadgroup per `(token-row, output-col)`; the weight row is gathered by the
/// token's expert. Per-element decode + per-block scale by the
/// `(BITS, WDEC, SKIND)` co-vars; buffer types by `(WT, ST)` — see
/// `gemm/block_scaled_matmul` for the legend. Decodes through
/// `kernels/primitives.rs`. Produces `ffai_<FMT>_gather_qmm`.
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
    suffix = "{FMT}_gather_qmm",
))]
#[allow(clippy::too_many_arguments)]
pub fn ffai<T>(
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    expert_ids: Tensor<u32>,
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
    let wrow = load(expert_ids[mr]) * out_dim + n;
    let n_blocks = in_dim / block_size;
    let row_block_off = wrow * n_blocks;
    let x_row_off = mr * in_dim;
    let mut acc = 0.0f32;

    if WDEC == 0u32 {
        let n_packs_per_row = in_dim / 8u32;
        let packs_per_block = block_size / 8u32;
        let row_pack_off = wrow * n_packs_per_row;
        let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
        for p_iter in range(0u32, p_iters, 1u32) {
            let pack_idx = p_iter * lsize + tid;
            if pack_idx < n_packs_per_row {
                let blk = pack_idx / packs_per_block;
                let sraw = load(scales[row_block_off + blk]);
                let scale = if SKIND == 0u32 {
                    exp2(sraw.cast::<f32>() - 127.0f32)
                } else if SKIND == 1u32 {
                    ffai_decode_e4m3(sraw.cast::<u32>()) * global
                } else {
                    sraw.cast::<f32>()
                };
                let packed = load(weight[row_pack_off + pack_idx]);
                let p_off = pack_idx * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    let val = ffai_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                    acc = acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                }
            }
        }
    } else {
        let half = 1u32 << (BITS - 1u32);
        let full = (1u32 << BITS).cast::<f32>();
        let words_per_row = in_dim * BITS / 32u32;
        let row_word_off = wrow * words_per_row;
        let row_off = wrow * in_dim;
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
                    let q = ffai_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf)
                } else {
                    let raw = load(weight[row_off + c]).cast::<u32>();
                    if WDEC == 2u32 {
                        ffai_decode_e4m3(raw)
                    } else if WDEC == 3u32 {
                        ffai_decode_e5m2(raw)
                    } else {
                        ffai_decode_int8(raw)
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
    use ffai_kernels::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{block_scaled_weights, pack_f32, unpack_f32},
    };

    const TPG: u32 = 64;

    /// `out[m, n] = Σ_k dequant(W)[expert_ids[m]·out_dim + n, k] · x[m, k]`.
    #[allow(clippy::too_many_arguments)]
    fn gather_oracle(
        wdq: &[f32],
        x: &[f32],
        eids: &[u32],
        m_rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m_rows * out_dim];
        for mr in 0..m_rows {
            let base = eids[mr] as usize * out_dim;
            for n in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += wdq[(base + n) * in_dim + k] * x[mr * in_dim + k];
                }
                out[mr * out_dim + n] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_setup(
        kernel: Kernel,
        fmt: QFormat,
        n_experts: usize,
        m_rows: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let stack_rows = n_experts * out_dim;
        // Build the FULL `[E·out_dim, in_dim]` stacked weight matrix (all experts
        // stacked along rows) and pack it in ONE call — never per-expert packing
        // + byte concatenation. For sub-byte widths (3/5/6-bit) `pack` appends a
        // single guard word at the very end of the contiguous bit-stream;
        // concatenating per-expert buffers would instead inject a guard word
        // mid-stream and misalign every expert after the first. One stacked pack
        // is byte-identical to the old per-expert concat for the 4-bit/8-bit
        // formats (those widths divide 32 ⇒ exact word count, no guard word) and
        // correct for every sub-byte width. `in_dim` is a multiple of 32, so each
        // row's bit-stream is word-aligned for every width.
        let w = block_scaled_weights(stack_rows, in_dim);
        let p = crate::kernels::quant::format::pack(fmt, &w, stack_rows, in_dim);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, stack_rows, in_dim);
        // Deterministic per-token expert routing.
        let eids: Vec<u32> = (0..m_rows).map(|m| (m * 2 + 1) as u32 % n_experts as u32).collect();
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = gather_oracle(&wdq, &x, &eids, m_rows, in_dim, out_dim);
        let eid_bytes: Vec<u8> = eids.iter().flat_map(|e| e.to_le_bytes()).collect();
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as native half; E8M0/E4M3 scales as one
        // byte. Both axes are driven off the format so new integer/fp16 formats
        // pick up the right buffer types.
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
            .input(TestBuffer::from_vec("expert_ids", eid_bytes, DType::U32))
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

    // 4 experts, 3 routed tokens, out_dim 4, in_dim 256.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxfp4_gather_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_nvfp4_gather_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 4, 3, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_mxfp8_e4m3_gather_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            3,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_mxfp8_e5m2_gather_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            3,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_nvfp8_gather_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 4, 3, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    // in_dim 256 is a multiple of int8's group of 64 (256 / 64 = 4).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_fp4_gather_qmm::kernel_ir_for(dt), QFormat::Fp4, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_nvfp8_gather_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_fp8_e5m2_gather_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_int8_gather_qmm::kernel_ir_for(dt), QFormat::Int8, 4, 3, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 is a multiple of 32, so
    // `in_dim*bits % 32 == 0` for every width and each gathered weight row's
    // bit-stream is word-aligned. The whole `[E·out_dim, in_dim]` stack is packed
    // once, so the bit-stream stays contiguous (one guard word at the very end)
    // and the kernel/oracle share the codec — the GPU output tracks the
    // dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_int2_gather_qmm::kernel_ir_for(dt), QFormat::Int2, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_int3_gather_qmm::kernel_ir_for(dt), QFormat::Int3, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_int4_gather_qmm::kernel_ir_for(dt), QFormat::Int4, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_int5_gather_qmm::kernel_ir_for(dt), QFormat::Int5, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_int6_gather_qmm::kernel_ir_for(dt), QFormat::Int6, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxint2_gather_qmm::kernel_ir_for(dt), QFormat::Mxint2, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxint3_gather_qmm::kernel_ir_for(dt), QFormat::Mxint3, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxint4_gather_qmm::kernel_ir_for(dt), QFormat::Mxint4, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxint5_gather_qmm::kernel_ir_for(dt), QFormat::Mxint5, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxint6_gather_qmm::kernel_ir_for(dt), QFormat::Mxint6, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_mxint8_gather_qmm::kernel_ir_for(dt), QFormat::Mxint8, 4, 3, 4, 256, dt)
    }

    // FP16-scale twins of the float-scale + int formats. Same element packing as
    // their FP32 twins (codes dtype unchanged); only the scale buffer is native
    // half. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel (same 8-bit-E4M3 +
    // f16-scale shape). in_dim 256 is a multiple of every block/group size.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_nvfp8_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_nvfp8_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(ffai_fp4_f16_gather_qmm::kernel_ir_for(dt), QFormat::Fp4F16, 4, 3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_fp8_e5m2_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_int2_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_int3_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_int4_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_int5_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_int6_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_gather_qmm(dt: DType) -> TestSetup {
        gather_setup(
            ffai_int8_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            3,
            4,
            256,
            dt,
        )
    }
}

/// Decode-shape (single routed token) gather-GEMM benches at the canonical
/// N=K=4096 so the GFLOP/s + roofline columns rank the precisions side by side
/// (the spec's "which precision is fastest" goal). Throughput is data-
/// independent, so the packed weight/scale buffers are random bytes and the
/// single token routes to expert 0.
pub mod kernel_benches {
    use ffai_kernels::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    /// Experts in the packed stack; throughput is independent of the count, but
    /// it sizes the weight buffer realistically.
    const N_EXPERTS: usize = 8;
    /// One routed token (decode shape) — the GEMV-equivalent of the qmm benches.
    const M_ROWS: usize = 1;

    fn gather_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let stack_rows = N_EXPERTS * out_dim;
        // Full packed stack lengths (what the buffers must hold). The whole
        // `[E·out_dim, in_dim]` stack is one contiguous bit-stream (single pack),
        // so its code length is `bitstream_words` over the *total* element count
        // (one guard word for the whole stack). 8-bit codes are one uchar each;
        // every sub-byte width (4-bit nibble packs + int2/3/5/6 tight bit-streams)
        // tight-bit-packs into u32 words. Both axes are driven off the format.
        let stack_blocks = stack_rows * (in_dim / fmt.block_size());
        let stack_n = stack_rows * in_dim;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (stack_n, DType::U8)
        } else {
            (
                crate::kernels::quant::format::bitstream_words(stack_n, fmt.element_bits()),
                DType::U32,
            )
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        // Bytes touched by the single routed token: only its expert's `out_dim`
        // weight rows + their scales + the expert id + the input row + output
        // row (the rest of the stack is never read for one token). The token's
        // weight rows are `out_dim` contiguous bit-stream rows, so their code
        // length is `bitstream_words` over `out_dim · in_dim` elements.
        let tok_codes = if fmt.element_bits() == 8 {
            out_dim * in_dim
        } else {
            crate::kernels::quant::format::bitstream_words(out_dim * in_dim, fmt.element_bits())
        };
        let tok_blocks = out_dim * (in_dim / fmt.block_size());
        let bytes = tok_codes * codes_dt.size_bytes()
            + tok_blocks * scales_dt.size_bytes()
            + M_ROWS * DType::U32.size_bytes()
            + M_ROWS * in_dim * sz
            + M_ROWS * out_dim * sz;
        let eid_bytes: Vec<u8> = (0..M_ROWS as u32).flat_map(|_| 0u32.to_le_bytes()).collect();
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", stack_blocks, scales_dt))
            .buffer(BenchBuffer::from_vec("expert_ids", eid_bytes, DType::U32))
            .buffer(BenchBuffer::random("x", M_ROWS * in_dim, dt))
            .buffer(BenchBuffer::zeros("output", M_ROWS * out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_dim * M_ROWS) as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * M_ROWS as u64 * out_dim as u64 * in_dim as u64) // gather-GEMV: 2·M·N·K
            .with_shape_label(format!("{} m={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxfp4_gather_qmm::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_nvfp4_gather_qmm::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(
            ffai_mxfp8_e4m3_gather_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(
            ffai_mxfp8_e5m2_gather_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_nvfp8_gather_qmm::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_fp4_gather_qmm::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_nvfp8_gather_qmm::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_fp8_e5m2_gather_qmm::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int8_gather_qmm::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int2_gather_qmm::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int3_gather_qmm::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int4_gather_qmm::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int5_gather_qmm::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int6_gather_qmm::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxint2_gather_qmm::kernel_ir_for(dt), QFormat::Mxint2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxint3_gather_qmm::kernel_ir_for(dt), QFormat::Mxint3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxint4_gather_qmm::kernel_ir_for(dt), QFormat::Mxint4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxint5_gather_qmm::kernel_ir_for(dt), QFormat::Mxint5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxint6_gather_qmm::kernel_ir_for(dt), QFormat::Mxint6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_mxint8_gather_qmm::kernel_ir_for(dt), QFormat::Mxint8, 4096, 4096, dt)
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same
    // 8-bit-E4M3 + f16-scale shape).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(
            ffai_nvfp8_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(
            ffai_nvfp8_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_fp4_f16_gather_qmm::kernel_ir_for(dt), QFormat::Fp4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(
            ffai_fp8_e5m2_f16_gather_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int2_f16_gather_qmm::kernel_ir_for(dt), QFormat::Int2F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int3_f16_gather_qmm::kernel_ir_for(dt), QFormat::Int3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int4_f16_gather_qmm::kernel_ir_for(dt), QFormat::Int4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int5_f16_gather_qmm::kernel_ir_for(dt), QFormat::Int5F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int6_f16_gather_qmm::kernel_ir_for(dt), QFormat::Int6F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_gather_qmm(dt: DType) -> BenchSetup {
        gather_bench(ffai_int8_f16_gather_qmm::kernel_ir_for(dt), QFormat::Int8F16, 4096, 4096, dt)
    }
}

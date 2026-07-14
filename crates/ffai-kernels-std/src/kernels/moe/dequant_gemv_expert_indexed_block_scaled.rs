//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Per-expert-indexed **block-scaled / legacy-fp / int8 dequantizing GEMV**.
//!
//! Block-scaled counterpart of `ffai/dequant_gemv_expert_indexed.rs` (which
//! handles the int4-affine case). For the eight non-int4 quantization formats
//! — mxfp4 / nvfp4 / mxfp8_e4m3 / mxfp8_e5m2 / nvfp8 + legacy fp4 / fp8_e5m2 /
//! int8 — the weight + scale tensors are **stacked across experts** and the
//! kernel reads which expert to index from a GPU-resident
//! `expert_index: Tensor<u32>` at runtime.
//!
//! Each kernel body is the `mlx/block_scaled_matmul.rs` qgemv for that format
//! (same one-TG-per-output-row pack-/element-strided reduction), with two extra
//! per-row offsets — exactly like the int4 expert-indexed kernel:
//!
//!   weight_expert_off = expert · out_dim · n_packs_per_row   (4-bit)
//!   weight_expert_off = expert · out_dim · in_dim            (8-bit)
//!   scale_expert_off  = expert · out_dim · n_blocks
//!
//! computed from `expert_index[0]` loaded once per threadgroup, then folded
//! into the row pack/element/block base offsets. There is no int affine bias —
//! block-scaled / fp / symmetric-int8 carry a scale only.
//!
//! `fp8_e4m3` is **not** a separate kernel: its layout (8-bit E4M3 codes + a
//! per-group FP32 scale) is identical to `nvfp8`, so the `fp8_e4m3` test + bench
//! dispatch `mt_nvfp8_dequant_gemv_expert_indexed` with `QFormat::Fp8E4m3`.
//!
//! ## Memory layout
//!
//! For `n_experts` experts each a `[out_dim, in_dim]` block-scaled slab:
//!
//!   weights_stacked  [n_experts, out_dim, in_dim/8]  u32   (4-bit, 8 nibbles/word)
//!   weights_stacked  [n_experts, out_dim, in_dim]    u8    (8-bit, 1 code/byte)
//!   scales_stacked   [n_experts, out_dim, in_dim/B]  u8|f32 (E8M0/E4M3 byte, or FP32)
//!   input            [in_dim]                         T
//!   expert_index     [1]                              u32
//!   output           [out_dim]                        T
//!
//! ## Dispatch
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [TPG, 1, 1]` with
//!   TPG ≥ 32 and a multiple of 32 (tests/benches use 64). One TG per row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.

use ffai_kernels::kernel;

/// Expert-indexed block-scaled dequantizing GEMV, folded over the 28-format
/// axis (§7). `output[r] = Σ_k dequant(weights_stacked[expert, r, k]) · input[k]`
/// where `expert = expert_index[0]`. One threadgroup per output row. Per-element
/// decode + per-block scale by the `(BITS, WDEC, SKIND)` co-vars; buffer types
/// by `(WT, ST)` — see `gemm/block_scaled_matmul` for the legend. Decodes
/// through `kernels/primitives.rs`. Produces `mt_<FMT>_dequant_gemv_expert_indexed`.
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
    suffix = "{FMT}_dequant_gemv_expert_indexed",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    weights_stacked: Tensor<WT>,
    scales_stacked: Tensor<ST>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let expert = load(expert_index[0u32]);
    let row_block_off = expert * out_dim * n_blocks + row * n_blocks;
    let mut acc = 0.0f32;

    if WDEC == 0u32 {
        let n_packs_per_row = in_dim / 8u32;
        let packs_per_block = block_size / 8u32;
        let row_pack_off = expert * out_dim * n_packs_per_row + row * n_packs_per_row;
        let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
        for p_iter in range(0u32, p_iters, 1u32) {
            let pack_idx = p_iter * lsize + tid;
            if pack_idx < n_packs_per_row {
                let blk = pack_idx / packs_per_block;
                let sraw = load(scales_stacked[row_block_off + blk]);
                let scale = if SKIND == 0u32 {
                    exp2(sraw.cast::<f32>() - 127.0f32)
                } else if SKIND == 1u32 {
                    mt_decode_e4m3(sraw.cast::<u32>()) * global
                } else {
                    sraw.cast::<f32>()
                };
                let packed = load(weights_stacked[row_pack_off + pack_idx]);
                let p_off = pack_idx * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                    acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
                }
            }
        }
    } else {
        let half = 1u32 << (BITS - 1u32);
        let full = (1u32 << BITS).cast::<f32>();
        let words_per_row = in_dim * BITS / 32u32;
        let row_word_off = expert * out_dim * words_per_row + row * words_per_row;
        let row_off = expert * out_dim * in_dim + row * in_dim;
        let iters = (in_dim + lsize - 1u32) / lsize;
        for it in range(0u32, iters, 1u32) {
            let c = it * lsize + tid;
            if c < in_dim {
                let blk = c / block_size;
                let sraw = load(scales_stacked[row_block_off + blk]);
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
                    let w0 = load(weights_stacked[row_word_off + word_idx]);
                    let w1 = load(
                        weights_stacked
                            [row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf)
                } else {
                    let raw = load(weights_stacked[row_off + c]).cast::<u32>();
                    if WDEC == 2u32 {
                        mt_decode_e4m3(raw)
                    } else if WDEC == 3u32 {
                        mt_decode_e5m2(raw)
                    } else {
                        mt_decode_int8(raw)
                    }
                };
                acc = acc + (val * scale) * load(input[c]).cast::<f32>();
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
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// 64 lanes per output row (≥ 32, multiple of 32 — Reduction contract).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` weights for expert `e` — mixed signs +
    /// per-expert + per-row + along-K magnitude variation so the per-block scale
    /// (and the expert stride) are genuinely exercised.
    fn weights(e: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + e as f32 * 0.3 + r * 0.25) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// Dequant-then-dot reference for the selected expert's dequantized slab.
    fn oracle(wdq: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim).map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * input[c]).sum()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn expert_setup(
        kernel: Kernel,
        fmt: QFormat,
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        expert: usize,
        dt: DType,
    ) -> TestSetup {
        // Build the FULL `[n_experts·out_dim, in_dim]` stacked weight and pack it
        // ONCE — `pack` then produces a single contiguous, per-row word-aligned
        // bit-stream for `codes` (and a contiguous `scales` axis). Packing each
        // expert independently and concatenating would misalign experts after the
        // first for the straddling sub-byte widths (3/5/6), which append a guard
        // word; a single stacked pack is byte-identical for the 4/8-bit formats
        // (no regression) and correct for every width. `p.codes`/`p.scales` bind
        // directly with no per-expert concat.
        let stacked_rows = n_experts * out_dim;
        let mut stacked_w: Vec<f32> = Vec::with_capacity(stacked_rows * in_dim);
        for e in 0..n_experts {
            stacked_w.extend_from_slice(&weights(e, out_dim, in_dim));
        }
        let p = crate::kernels::quant::format::pack(fmt, &stacked_w, stacked_rows, in_dim);
        let sel_global = p.global;
        // Dequant the full stack, then slice the selected expert's row band for
        // the oracle (rows `[expert·out_dim, (expert+1)·out_dim)`).
        let wdq_all = crate::kernels::quant::format::dequant(fmt, &p, stacked_rows, in_dim);
        let wdq = &wdq_all[expert * out_dim * in_dim..(expert + 1) * out_dim * in_dim];

        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        // Round-trip the input through `dt` so the oracle sees what the GPU sees.
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = oracle(wdq, &x, out_dim, in_dim);

        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weights_stacked", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_stacked", p.scales, scales_dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("expert_index", u32_bytes(&[expert as u32]), DType::U32))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", sel_global);
        }
        s.expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt)).grid_3d(
            out_dim as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // n_experts 4, out_dim 4, in_dim 256 (divisible by every block/group size —
    // 16/32/64), expert 2 to exercise a non-zero expert stride.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxfp8_e4m3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxfp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode in their own.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // int8 group is 64 → in_dim 256 = 4×64 divides evenly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width (and divides every group/block
    // size), so the single stacked pack's per-row bit-stream is word-aligned and
    // the kernel + oracle share the codec to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_mxint8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // ── FP16-scale twins ────────────────────────────────────────────────────
    // Same geometry as the FP32-scaled formats; only the scale axis is half.
    // `fp8_e4m3_f16` dispatches the nvfp8_f16 kernel (identical 8-bit-E4M3 +
    // f16-scale shape), exactly as `fp8_e4m3` reuses the nvfp8 kernel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_fp8_e5m2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int3_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int5_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int6_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_dequant_gemv_expert_indexed(dt: DType) -> TestSetup {
        expert_setup(
            mt_int8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
}

/// Decode-shape benches: per-expert-indexed dequant GEMV over an 8-expert stack
/// at the canonical out_dim=in_dim=4096 so the GFLOP/s + roofline columns rank
/// the precisions side by side. Active stream = one expert's slab + its scales +
/// input + output. One TG per output row. Throughput is data-independent, so the
/// packed weight/scale buffers are random bytes.
pub mod kernel_benches {
    use ffai_kernels::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn expert_bench(
        kernel: Kernel,
        fmt: QFormat,
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let blocks_per_expert = out_dim * (in_dim / fmt.block_size());
        // The whole stack is one bit-stream packed once, so the `weights_stacked`
        // buffer is `bitstream_words(n_experts·out_dim·in_dim, bits)` u32 words for
        // every sub-byte width (4-bit collapses to the old `total/8`), or one byte
        // per code at 8-bit. Per-expert lengths drive only the active-stream byte
        // accounting (one expert's slab is read per dispatch). Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let codes_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let total_codes = if fmt.element_bits() == 8 {
            n_experts * out_dim * in_dim
        } else {
            crate::kernels::quant::format::bitstream_words(
                n_experts * out_dim * in_dim,
                fmt.element_bits(),
            )
        };
        let codes_per_expert = if fmt.element_bits() == 8 {
            out_dim * in_dim
        } else {
            crate::kernels::quant::format::bitstream_words(out_dim * in_dim, fmt.element_bits())
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        // Active stream: one expert's weight slab + its scales + input + output.
        let bytes = codes_per_expert * codes_dt.size_bytes()
            + blocks_per_expert * scales_dt.size_bytes()
            + in_dim * sz
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weights_stacked", total_codes, codes_dt))
            .buffer(BenchBuffer::random("scales_stacked", n_experts * blocks_per_expert, scales_dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("expert_index", 1, DType::U32))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64) // qgemv expert-indexed (B=1): 2·N·K
            .with_shape_label(format!("{} m={out_dim} k={in_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxfp8_e4m3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxfp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp8_e5m2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8,
            8,
            4096,
            4096,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint2_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint2,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint3_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint3,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint4_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint4,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint5_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint5,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint6_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint6,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_mxint8_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Mxint8,
            8,
            4096,
            4096,
            dt,
        )
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3
    // + f16-scale shape); the rest decode in their own per-element kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_nvfp8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp4F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_fp8_e5m2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int2_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int2F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int3_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int3F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int4_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int4F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int5_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int5F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int6_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int6F16,
            8,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_dequant_gemv_expert_indexed(dt: DType) -> BenchSetup {
        expert_bench(
            mt_int8_f16_dequant_gemv_expert_indexed::kernel_ir_for(dt),
            QFormat::Int8F16,
            8,
            4096,
            4096,
            dt,
        )
    }
}

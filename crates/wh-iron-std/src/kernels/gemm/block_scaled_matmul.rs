//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **dequantizing GEMV** kernels (Phase B of the precision
//! roadmap, `specs/BENCH_METRICS_SPEC.md` Appendix B): `output[row] =
//! Σ_k dequant(weight[row, k]) · input[k]` for the spec-conformant formats.
//!
//! The dispatch geometry is the **proven pack-strided reduction** from
//! `iron/dequant_gemv.rs` — one threadgroup per output row, threads stride over
//! the row's packed words, `reduce_sum` folds the partials. Only the per-element
//! *decode* differs (block-scaled E2M1/E4M3/… instead of int-affine), so no new
//! dispatch shape is introduced (and the reduction freeze hazard — TPG ≥ 32 &
//! multiple of 32 — is handled exactly as the int kernels handle it).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [TPG, 1, 1]` with
//!   TPG ≥ 32 and a multiple of 32 (tests/benches use 64). One TG per row.
//! - `in_dim` a multiple of `block_size`; `block_size` a multiple of 8 (so a
//!   u32 pack of 8 nibbles lies wholly inside one block — one scale load/pack).
//! - weight `[out_dim, in_dim/8]` u32 (8 E2M1 nibbles/word, little-endian);
//!   scales `[out_dim, in_dim/block_size]` u8 (E8M0); input `[in_dim]`,
//!   output `[out_dim]`.

use wh_iron::kernel;

/// Block-scaled dequantizing GEMV, folded over the 28-format axis (§7).
///
/// `output[row] = Σ_k dequant(weight[row,k]) · input[k]`. One threadgroup per
/// output row; `reduce_sum` folds the partials (TPG ≥ 32, multiple of 32 — the
/// Reduction freeze hazard is handled exactly as the int kernels did). The
/// per-element decode and per-block scale are selected by the `(BITS, WDEC,
/// SKIND)` co-vars; the weight/scale buffer types by `(WT, ST)`:
///   WDEC 0 = E2M1 nibble (pack-strided), 1 = sub-byte int bit-stream,
///        2 = E4M3 byte, 3 = E5M2 byte, 4 = int8 byte.
///   SKIND 0 = E8M0 pow-2 (u8), 1 = E4M3 micro × global (u8, nvfp4),
///         2 = direct per-block scale (f32 / f16).
/// Decodes through the shared `kernels/primitives.rs` (mirroring `kernels::quant::codec`)
/// so the kernel and the oracle cannot drift. Produces `iron_<FMT>_qgemv`.
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
    suffix = "{FMT}_qgemv",
))]
#[allow(clippy::too_many_arguments)]
pub fn iron<T>(
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let row = program_id::<0>();
    let n_blocks = in_dim / block_size;
    let row_block_off = row * n_blocks;
    let mut acc = 0.0f32;

    if WDEC == 0u32 {
        // E2M1 nibble, pack-strided: 8 nibbles per u32, one scale load per pack.
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
                    iron_decode_e4m3(sraw.cast::<u32>()) * global
                } else {
                    sraw.cast::<f32>()
                };
                let packed = load(weight[row_pack_off + pack_idx]);
                let p_off = pack_idx * 8u32;
                for i in range(0u32, 8u32, 1u32) {
                    let val = iron_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                    acc = acc + (val * scale) * load(input[p_off + i]).cast::<f32>();
                }
            }
        }
    } else {
        // Element-strided: one code per thread-iter (sub-byte int or byte float).
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
                    // Sub-byte int bit-stream (BITS ∈ {2..6}), straddle-aware.
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
                    let q = iron_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf) // sign-extend
                } else {
                    // Byte format: one code per byte, decoded by WDEC.
                    let raw = load(weight[row_off + c]).cast::<u32>();
                    if WDEC == 2u32 {
                        iron_decode_e4m3(raw)
                    } else if WDEC == 3u32 {
                        iron_decode_e5m2(raw)
                    } else {
                        iron_decode_int8(raw)
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
    use wh_iron::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{block_scaled_qgemv_oracle, block_scaled_weights, pack_f32, unpack_f32},
    };

    /// One TG-row's lanes; ≥ 32 and a multiple of 32 (the Reduction contract).
    const TPG: u32 = 64;

    fn qgemv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_dim: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let w = block_scaled_weights(out_dim, in_dim);
        let p = crate::kernels::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_dim, in_dim);
        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        // Round-trip the input through `dt` so the oracle sees what the GPU sees.
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = block_scaled_qgemv_oracle(&wdq, &x, out_dim, in_dim);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new formats pick up the right
        // buffer types.
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
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
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

    // out_dim 4, in_dim 256 (divisible by both block sizes) — mirrors the int
    // dequant_gemv test shape.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxfp4_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_nvfp4_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxfp8_e4m3_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E4, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxfp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E5, 4, 256, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4, 256, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_fp4_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_fp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int8_qgemv::kernel_ir_for(dt), QFormat::Int8, 4, 256, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each row's bit-stream is
    // word-aligned. The kernel and oracle share the codec, so the GPU output
    // tracks the dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int2_qgemv::kernel_ir_for(dt), QFormat::Int2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int3_qgemv::kernel_ir_for(dt), QFormat::Int3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int4_qgemv::kernel_ir_for(dt), QFormat::Int4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int5_qgemv::kernel_ir_for(dt), QFormat::Int5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int6_qgemv::kernel_ir_for(dt), QFormat::Int6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxint2_qgemv::kernel_ir_for(dt), QFormat::Mxint2, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxint3_qgemv::kernel_ir_for(dt), QFormat::Mxint3, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxint4_qgemv::kernel_ir_for(dt), QFormat::Mxint4, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxint5_qgemv::kernel_ir_for(dt), QFormat::Mxint5, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxint6_qgemv::kernel_ir_for(dt), QFormat::Mxint6, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_mxint8_qgemv::kernel_ir_for(dt), QFormat::Mxint8, 4, 256, dt)
    }

    // FP16-scale twins of the FP32-scaled formats. `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape); the rest decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Nvfp8F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_fp4_f16_qgemv::kernel_ir_for(dt), QFormat::Fp4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_fp8_e5m2_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int2_f16_qgemv::kernel_ir_for(dt), QFormat::Int2F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int3_f16_qgemv::kernel_ir_for(dt), QFormat::Int3F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int4_f16_qgemv::kernel_ir_for(dt), QFormat::Int4F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int5_f16_qgemv::kernel_ir_for(dt), QFormat::Int5F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int6_f16_qgemv::kernel_ir_for(dt), QFormat::Int6F16, 4, 256, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_qgemv(dt: DType) -> TestSetup {
        qgemv_setup(iron_int8_f16_qgemv::kernel_ir_for(dt), QFormat::Int8F16, 4, 256, dt)
    }
}

/// Decode-shape (single-token GEMV) benches at the canonical N=K=4096 so the
/// GFLOP/s + roofline columns rank the precisions side by side (the spec's
/// "which precision is fastest" goal). Throughput is data-independent, so the
/// packed weight/scale buffers are random bytes.
pub mod kernel_benches {
    use wh_iron::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn qgemv_bench(
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
            + in_dim * sz
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
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
    fn bench_mxfp4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxfp4_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_nvfp4_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxfp8_e4m3_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxfp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Mxfp8E5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_fp4_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_nvfp8_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_fp8_e5m2_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int8_qgemv::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int2_qgemv::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int3_qgemv::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int4_qgemv::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int5_qgemv::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int6_qgemv::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxint2_qgemv::kernel_ir_for(dt), QFormat::Mxint2, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxint3_qgemv::kernel_ir_for(dt), QFormat::Mxint3, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxint4_qgemv::kernel_ir_for(dt), QFormat::Mxint4, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxint5_qgemv::kernel_ir_for(dt), QFormat::Mxint5, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxint6_qgemv::kernel_ir_for(dt), QFormat::Mxint6, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_mxint8_qgemv::kernel_ir_for(dt), QFormat::Mxint8, 4096, 4096, dt)
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Nvfp8F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_nvfp8_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_fp4_f16_qgemv::kernel_ir_for(dt), QFormat::Fp4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_fp8_e5m2_f16_qgemv::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int2_f16_qgemv::kernel_ir_for(dt), QFormat::Int2F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int3_f16_qgemv::kernel_ir_for(dt), QFormat::Int3F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int4_f16_qgemv::kernel_ir_for(dt), QFormat::Int4F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int5_f16_qgemv::kernel_ir_for(dt), QFormat::Int5F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int6_f16_qgemv::kernel_ir_for(dt), QFormat::Int6F16, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_qgemv(dt: DType) -> BenchSetup {
        qgemv_bench(iron_int8_f16_qgemv::kernel_ir_for(dt), QFormat::Int8F16, 4096, 4096, dt)
    }
}

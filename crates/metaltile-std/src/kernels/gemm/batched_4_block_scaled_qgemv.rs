//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **batched 4-output block-scaled dequantizing GEMV** for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! One dispatch computes FOUR independent projections that share the same
//! single-token activation `x`: `out = [Wa·x, Wb·x, Wc·x, Wd·x]`. Mirrors the
//! int4 fused variant `ffai_batched_4_qgemv_fast`, but with the simple
//! one-output-row-per-TG Reduction geometry of the block-scaled Q/K/V variant
//! `mt_*_batched_qkv_qgemv` rather than the perf-tuned mask-without-shift path.
//!
//! Unlike the Q/K/V variant — which writes a single concatenated `output`
//! buffer — the four projections here write to FOUR SEPARATE output buffers
//! (`a_out`, `b_out`, `c_out`, `d_out`), each indexed directly by `row` (no
//! offset). Callers may alias all four into one backing allocation; the kernel
//! only sees four base pointers. This matches `ffai_batched_4_qgemv_fast`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [max(out_a,out_b,out_c,out_d), 1, 4]`,
//!   `tpg = [TPG, 1, 1]`, TPG >= 32 & a multiple of 32. One TG per
//!   (matrix, row); rows past a matrix's `out_*` no-op. The grid z-dimension
//!   `program_id::<2>()` selects the matrix (0=A, 1=B, 2=C, 3=D),
//!   `program_id::<0>()` the output row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_*, in_dim/8]` u32 (4-bit) or `[out_*, in_dim]` u8 (8-bit);
//!   scales `[out_*, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!   Each matrix writes its own `out_*`-length output buffer. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// Batched 4-output block-scaled dequantizing GEMV, folded over
/// the 28-format axis (§7). `matrix = program_id::<2>()` selects one of the
/// 4 (a/b/c/d) weight matrices; each runs the shared per-element decode tree
/// over its own `w_*`/`scales_*` row into a single output. `(BITS, WDEC, SKIND)`
/// + `(WT, ST)` as in `block_scaled_matmul`. Decodes through
/// `kernels/primitives.rs`. Produces `mt_<FMT>_batched_4_qgemv`.
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
    suffix = "{FMT}_batched_4_qgemv",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    x: Tensor<T>,
    w_a: Tensor<WT>,
    scales_a: Tensor<ST>,
    w_b: Tensor<WT>,
    scales_b: Tensor<ST>,
    w_c: Tensor<WT>,
    scales_c: Tensor<ST>,
    w_d: Tensor<WT>,
    scales_d: Tensor<ST>,
    mut a_out: Tensor<T>,
    mut b_out: Tensor<T>,
    mut c_out: Tensor<T>,
    mut d_out: Tensor<T>,
    #[constexpr] out_a: u32,
    #[constexpr] out_b: u32,
    #[constexpr] out_c: u32,
    #[constexpr] out_d: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let matrix = program_id::<2>();
    let row = program_id::<0>();
    let n_packs = in_dim / 8u32;
    let n_blocks = in_dim / block_size;
    let packs_per_block = block_size / 8u32;
    let p_iters = (n_packs + lsize - 1u32) / lsize;
    let e_iters = (in_dim + lsize - 1u32) / lsize;
    let row_pack_off = row * n_packs;
    let row_block_off = row * n_blocks;
    let words_per_row = in_dim * BITS / 32u32;
    let row_word_off = row * words_per_row;
    let row_off = row * in_dim;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    let mut acc = 0.0f32;
    if matrix == 0u32 {
        if row < out_a {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_a[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            mt_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_a[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_a[row_block_off + c / block_size]);
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
                            let w0 = load(w_a[row_word_off + word_idx]);
                            let w1 = load(
                                w_a[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_a[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                mt_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                mt_decode_e5m2(raw)
                            } else {
                                mt_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_b {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_b[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            mt_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_b[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_b[row_block_off + c / block_size]);
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
                            let w0 = load(w_b[row_word_off + word_idx]);
                            let w1 = load(
                                w_b[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_b[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                mt_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                mt_decode_e5m2(raw)
                            } else {
                                mt_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_c {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_c[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            mt_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_c[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_c[row_block_off + c / block_size]);
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
                            let w0 = load(w_c[row_word_off + word_idx]);
                            let w1 = load(
                                w_c[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_c[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                mt_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                mt_decode_e5m2(raw)
                            } else {
                                mt_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 3u32 {
        if row < out_d {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_d[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            mt_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_d[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = mt_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc = acc + (val * scale) * load(x[p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_d[row_block_off + c / block_size]);
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
                            let w0 = load(w_d[row_word_off + word_idx]);
                            let w1 = load(
                                w_d[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_d[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                mt_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                mt_decode_e5m2(raw)
                            } else {
                                mt_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[c]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_a {
                store(a_out[row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_b {
                store(b_out[row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_c {
                store(c_out[row], total.cast::<T>());
            }
        }
        if matrix == 3u32 {
            if row < out_d {
                store(d_out[row], total.cast::<T>());
            }
        }
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    const TPG: u32 = 64;

    fn weights(out_dim: usize, in_dim: usize, seed: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + ((r as usize + seed) % 5) as f32 * 0.2) * (0.1 + (c % 13.0) * 0.2);
                if (i + seed).is_multiple_of(3) { -mag } else { mag }
            })
            .collect()
    }

    fn gemv(wdq: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim).map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * x[c]).sum()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_4_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_a: usize,
        out_b: usize,
        out_c: usize,
        out_d: usize,
        in_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let pack_w = |out_dim: usize, seed: usize| {
            let w = weights(out_dim, in_dim, seed);
            let p = crate::kernels::quant::format::pack(fmt, &w, out_dim, in_dim);
            let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_dim, in_dim);
            (p, wdq)
        };
        let (pa, wdq_a) = pack_w(out_a, 0);
        let (pb, wdq_b) = pack_w(out_b, 1);
        let (pc, wdq_c) = pack_w(out_c, 2);
        let (pd, wdq_d) = pack_w(out_d, 3);
        let x_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        // Four SEPARATE expected outputs — each matrix writes its own buffer.
        let ea = gemv(&wdq_a, &x, out_a, in_dim);
        let eb = gemv(&wdq_b, &x, out_b, in_dim);
        let ec = gemv(&wdq_c, &x, out_c, in_dim);
        let ed = gemv(&wdq_d, &x, out_d, in_dim);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32, FP16 scales as f16; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new formats pick up the right
        // buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let max_rows = out_a.max(out_b).max(out_c).max(out_d);
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w_a", pa.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_a", pa.scales, scales_dt))
            .input(TestBuffer::from_vec("w_b", pb.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_b", pb.scales, scales_dt))
            .input(TestBuffer::from_vec("w_c", pc.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_c", pc.scales, scales_dt))
            .input(TestBuffer::from_vec("w_d", pd.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_d", pd.scales, scales_dt))
            .input(TestBuffer::zeros("a_out", out_a, dt))
            .input(TestBuffer::zeros("b_out", out_b, dt))
            .input(TestBuffer::zeros("c_out", out_c, dt))
            .input(TestBuffer::zeros("d_out", out_d, dt))
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", pa.global.max(pb.global).max(pc.global).max(pd.global));
        }
        s.expect(TestBuffer::from_vec("a_out", pack_f32(&ea, dt), dt))
            .expect(TestBuffer::from_vec("b_out", pack_f32(&eb, dt), dt))
            .expect(TestBuffer::from_vec("c_out", pack_f32(&ec, dt), dt))
            .expect(TestBuffer::from_vec("d_out", pack_f32(&ed, dt), dt))
            .grid_3d(max_rows as u32, 1, 4, [TPG, 1, 1])
    }

    // out_a 16, out_b/out_c/out_d 4, in_dim 256 (÷ 16/32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxfp8_e4m3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxfp8_e5m2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode in their
    // own kernels above.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_fp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_fp8_e5m2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int8,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each row's bit-stream is
    // word-aligned. Kernel + oracle share the codec, so the four GPU outputs
    // track the dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int2,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int3,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int5_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int5,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int6_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int6,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxint2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxint3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxint4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxint5_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxint6_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_mxint8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }

    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same
    // 8-bit-E4M3 + f16-scale shape); the rest decode in their own f16 kernels.
    // in_dim 256 keeps every sub-byte bit-stream word-aligned, exactly as the
    // FP32 / E8M0 integer tests above. Kernel + oracle share the codec, so the
    // four GPU outputs track the dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp8_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_nvfp8_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_fp4_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_fp8_e5m2_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int2_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int3_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int4_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int5_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int6_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_batched_4_qgemv(dt: DType) -> TestSetup {
        batched_4_setup(
            mt_int8_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            16,
            4,
            4,
            4,
            256,
            dt,
        )
    }
}

/// Decode-shape benches at a fused 4-projection shape (out_a=out_b=out_c=out_d
/// =4096, in_dim=4096). Throughput is data-independent → random buffers.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn batched_4_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_a: usize,
        out_b: usize,
        out_c: usize,
        out_d: usize,
        in_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let bs = fmt.block_size();
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let codes_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let max_rows = out_a.max(out_b).max(out_c).max(out_d);
        let sz = dt.size_bytes();
        // Per-matrix code-buffer length: 8-bit stays `o·in_dim` bytes, every
        // sub-byte width is the tight bit-stream's u32-word count.
        let elem_bits = fmt.element_bits();
        let codes = |o: usize| {
            if elem_bits == 8 {
                o * in_dim
            } else {
                crate::kernels::quant::format::bitstream_words(o * in_dim, elem_bits)
            }
        };
        let scl = |o: usize| o * (in_dim / bs);
        let bytes = (codes(out_a) + codes(out_b) + codes(out_c) + codes(out_d))
            * codes_dt.size_bytes()
            + (scl(out_a) + scl(out_b) + scl(out_c) + scl(out_d)) * scales_dt.size_bytes()
            + in_dim * sz
            + (out_a + out_b + out_c + out_d) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", in_dim, dt))
            .buffer(BenchBuffer::random("w_a", codes(out_a), codes_dt))
            .buffer(BenchBuffer::random("scales_a", scl(out_a), scales_dt))
            .buffer(BenchBuffer::random("w_b", codes(out_b), codes_dt))
            .buffer(BenchBuffer::random("scales_b", scl(out_b), scales_dt))
            .buffer(BenchBuffer::random("w_c", codes(out_c), codes_dt))
            .buffer(BenchBuffer::random("scales_c", scl(out_c), scales_dt))
            .buffer(BenchBuffer::random("w_d", codes(out_d), codes_dt))
            .buffer(BenchBuffer::random("scales_d", scl(out_d), scales_dt))
            .buffer(BenchBuffer::zeros("a_out", out_a, dt).output())
            .buffer(BenchBuffer::zeros("b_out", out_b, dt).output())
            .buffer(BenchBuffer::zeros("c_out", out_c, dt).output())
            .buffer(BenchBuffer::zeros("d_out", out_d, dt).output())
            .constexpr("out_a", out_a as u32)
            .constexpr("out_b", out_b as u32)
            .constexpr("out_c", out_c as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", bs as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(max_rows as u32, 1, 4, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * (out_a + out_b + out_c + out_d) as u64 * in_dim as u64)
            .with_shape_label(format!(
                "{} a={out_a} b={out_b} c={out_c} d={out_d} in={in_dim}",
                fmt.name()
            ))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxfp8_e4m3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxfp8_e5m2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_fp4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_fp8_e5m2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int8,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int2,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int3,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int5_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int5,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int6_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int6,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxint2_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxint3_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxint4_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxint5_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxint6_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_mxint8_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp8_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_nvfp8_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_fp4_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_fp8_e5m2_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int2_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int3_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int4_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int5_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int6_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_batched_4(dt: DType) -> BenchSetup {
        batched_4_bench(
            mt_int8_f16_batched_4_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            4096,
            4096,
            4096,
            4096,
            4096,
            dt,
        )
    }
}

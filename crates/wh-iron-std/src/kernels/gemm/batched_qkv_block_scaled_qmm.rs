//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Fused **batched Q/K/V block-scaled dequantizing GEMM** (M>1) for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! Multi-token sibling of `batched_qkv_block_scaled_qgemv`: instead of a
//! single activation row it consumes `x: [M, in_dim]` and produces row `m`
//! of THREE separate output tensors —
//!   q_buf: [M, out_q] T
//!   k_buf: [M, out_k] T
//!   v_buf: [M, out_v] T
//!
//! One dispatch computes `out_X[m, n] = Σ_k dequant(W_X[n, k]) · x[m, k]` for
//! all three projections. The block-scaled weight decode of
//! `mlx/block_scaled_qmm.rs` replaces the int affine; the (matrix, token, row)
//! geometry mirrors the int4 `iron_batched_qkv_qmm_fast`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [max(out_q,out_k,out_v), M, 3]`,
//!   `tpg = [TPG, 1, 1]`, TPG ≥ 32 & a multiple of 32. One TG per
//!   `(matrix, m_token, out_row)`; rows past a matrix's `out_*` no-op.
//!   * `program_id::<2>()` selects matrix (0→Q, 1→K, 2→V).
//!   * `program_id::<1>()` selects batched token `mr` (0..M).
//!   * `program_id::<0>()` selects the output row.
//! - `in_dim` a multiple of `block_size`; 4-bit `block_size` a multiple of 8.
//! - weight `[out_*, in_dim/8]` u32 (4-bit) or `[out_*, in_dim]` u8 (8-bit);
//!   scales `[out_*, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!   `x` is `[M, in_dim]`, each `*_buf` is `[M, out_*]`, all row-major. No bias.
//!
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s.

use wh_iron::kernel;

/// Batched QKV 3-output block-scaled dequantizing GEMV, folded over
/// the 28-format axis (§7). `matrix = program_id::<2>()` selects one of the
/// 3 (q/k/v) weight matrices; each runs the shared per-element decode tree
/// over its own `w_*`/`scales_*` row into a single output. `(BITS, WDEC, SKIND)`
/// + `(WT, ST)` as in `block_scaled_matmul`. Decodes through
/// `kernels/primitives.rs`. Produces `iron_<FMT>_batched_qkv_%s`." % op
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
    suffix = "{FMT}_batched_qkv_qmm",
))]
#[allow(clippy::too_many_arguments)]
pub fn iron<T>(
    x: Tensor<T>,
    w_q: Tensor<WT>,
    scales_q: Tensor<ST>,
    w_k: Tensor<WT>,
    scales_k: Tensor<ST>,
    w_v: Tensor<WT>,
    scales_v: Tensor<ST>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let matrix = program_id::<2>();
    let mr = program_id::<1>();
    let row = program_id::<0>();
    let x_row_off = mr * in_dim;
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
        if row < out_q {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_q[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            iron_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_q[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = iron_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc =
                                acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_q[row_block_off + c / block_size]);
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
                            let w0 = load(w_q[row_word_off + word_idx]);
                            let w1 = load(
                                w_q[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = iron_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_q[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                iron_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                iron_decode_e5m2(raw)
                            } else {
                                iron_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 1u32 {
        if row < out_k {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_k[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            iron_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_k[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = iron_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc =
                                acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_k[row_block_off + c / block_size]);
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
                            let w0 = load(w_k[row_word_off + word_idx]);
                            let w1 = load(
                                w_k[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = iron_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_k[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                iron_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                iron_decode_e5m2(raw)
                            } else {
                                iron_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                    }
                }
            }
        }
    }
    if matrix == 2u32 {
        if row < out_v {
            if WDEC == 0u32 {
                for _p in range(0u32, p_iters, 1u32) {
                    let pack_idx = _p * lsize + tid;
                    if pack_idx < n_packs {
                        let blk = pack_idx / packs_per_block;
                        let sraw = load(scales_v[row_block_off + blk]);
                        let scale = if SKIND == 0u32 {
                            exp2(sraw.cast::<f32>() - 127.0f32)
                        } else if SKIND == 1u32 {
                            iron_decode_e4m3(sraw.cast::<u32>()) * global
                        } else {
                            sraw.cast::<f32>()
                        };
                        let packed = load(w_v[row_pack_off + pack_idx]);
                        let p_off = pack_idx * 8u32;
                        for i in range(0u32, 8u32, 1u32) {
                            let val = iron_decode_e2m1((packed >> (i * 4u32)) & 0xFu32);
                            acc =
                                acc + (val * scale) * load(x[x_row_off + p_off + i]).cast::<f32>();
                        }
                    }
                }
            } else {
                for it in range(0u32, e_iters, 1u32) {
                    let c = it * lsize + tid;
                    if c < in_dim {
                        let sraw = load(scales_v[row_block_off + c / block_size]);
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
                            let w0 = load(w_v[row_word_off + word_idx]);
                            let w1 = load(
                                w_v[row_word_off + select(spill > 0u32, word_idx + 1u32, word_idx)],
                            );
                            let q = iron_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            select(q >= half, qf - full, qf)
                        } else {
                            let raw = load(w_v[row_off + c]).cast::<u32>();
                            if WDEC == 2u32 {
                                iron_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                iron_decode_e5m2(raw)
                            } else {
                                iron_decode_int8(raw)
                            }
                        };
                        acc = acc + (val * scale) * load(x[x_row_off + c]).cast::<f32>();
                    }
                }
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        if matrix == 0u32 {
            if row < out_q {
                store(q_buf[mr * out_q + row], total.cast::<T>());
            }
        }
        if matrix == 1u32 {
            if row < out_k {
                store(k_buf[mr * out_k + row], total.cast::<T>());
            }
        }
        if matrix == 2u32 {
            if row < out_v {
                store(v_buf[mr * out_v + row], total.cast::<T>());
            }
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Reduction-contract threadgroup width (≥ 32, multiple of 32).
    const TPG: u32 = 64;

    /// Deterministic `[out_dim, in_dim]` quantized weights (mixed signs).
    /// `seed` decorrelates the Q/K/V matrices.
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

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`, row-major `[M, out_dim]`.
    fn qmm_oracle(
        wdq: &[f32],
        x: &[f32],
        m_rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m_rows * out_dim];
        for mr in 0..m_rows {
            for n in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    acc += wdq[n * in_dim + k] * x[mr * in_dim + k];
                }
                out[mr * out_dim + n] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_setup(
        kernel: Kernel,
        fmt: QFormat,
        out_q: usize,
        out_k: usize,
        out_v: usize,
        in_dim: usize,
        m_rows: usize,
        dt: DType,
    ) -> TestSetup {
        // Pack + dequant each of the three weight matrices (distinct seeds).
        let pack_w = |out_dim: usize, seed: usize| {
            let w = weights(out_dim, in_dim, seed);
            let p = crate::kernels::quant::format::pack(fmt, &w, out_dim, in_dim);
            let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_dim, in_dim);
            (p, wdq)
        };
        let (pq, wdq_q) = pack_w(out_q, 0);
        let (pk, wdq_k) = pack_w(out_k, 1);
        let (pv, wdq_v) = pack_w(out_v, 2);
        // Build x as [m_rows, in_dim] and round it through the storage dtype.
        let x_f: Vec<f32> = (0..m_rows * in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let eq = qmm_oracle(&wdq_q, &x, m_rows, in_dim, out_q);
        let ek = qmm_oracle(&wdq_k, &x, m_rows, in_dim, out_k);
        let ev = qmm_oracle(&wdq_v, &x, m_rows, in_dim, out_v);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; FP16 scales as half; E8M0/E4M3 scales as one byte.
        // Both axes are driven off the format so new formats pick up the right
        // buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let max_rows = out_q.max(out_k).max(out_v);
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w_q", pq.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_q", pq.scales, scales_dt))
            .input(TestBuffer::from_vec("w_k", pk.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_k", pk.scales, scales_dt))
            .input(TestBuffer::from_vec("w_v", pv.codes, weight_dt))
            .input(TestBuffer::from_vec("scales_v", pv.scales, scales_dt))
            .input(TestBuffer::zeros("q_buf", m_rows * out_q, dt))
            .input(TestBuffer::zeros("k_buf", m_rows * out_k, dt))
            .input(TestBuffer::zeros("v_buf", m_rows * out_v, dt))
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", pq.global.max(pk.global).max(pv.global));
        }
        s.expect(TestBuffer::from_vec("q_buf", pack_f32(&eq, dt), dt))
            .expect(TestBuffer::from_vec("k_buf", pack_f32(&ek, dt), dt))
            .expect(TestBuffer::from_vec("v_buf", pack_f32(&ev, dt), dt))
            .grid_3d(max_rows as u32, m_rows as u32, 3, [TPG, 1, 1])
    }

    // out_q 16, out_k/out_v 4, in_dim 256 (÷ 16/32), m 2.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxfp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp4,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_nvfp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp4,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxfp8_e4m3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxfp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the
    // nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_fp4_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Fp4, 16, 4, 4, 256, 2, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_fp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_int8_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int8, 16, 4, 4, 256, 2, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim 256 satisfies
    // `in_dim*bits % 32 == 0` for every width, so each weight row's bit-stream is
    // word-aligned. The kernel and oracle share the codec, so the GPU output
    // tracks the dequant-then-matmul reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_int2_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int2, 16, 4, 4, 256, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_int3_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int3, 16, 4, 4, 256, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_int4_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int4, 16, 4, 4, 256, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_int5_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int5, 16, 4, 4, 256, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(iron_int6_batched_qkv_qmm::kernel_ir_for(dt), QFormat::Int6, 16, 4, 4, 256, 2, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxint2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint2,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxint3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint3,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxint4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint4,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxint5_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint5,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxint6_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint6,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_mxint8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint8,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }

    // FP16-scale twins (nvfp8 / fp4 / fp8_e5m2 + int2..6 + int8). fp8_e4m3_f16
    // reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape); the
    // scales bind as half via the `ScaleKind::F16` arm of `qkv_setup`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_nvfp8_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_nvfp8_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_fp4_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp4F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_fp8_e5m2_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_int2_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int2F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_int3_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int3F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_int4_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int4F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_int5_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int5F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_int6_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int6F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_batched_qkv_qmm(dt: DType) -> TestSetup {
        qkv_setup(
            iron_int8_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int8F16,
            16,
            4,
            4,
            256,
            2,
            dt,
        )
    }
}

/// Small-batch prefill (M=8) benches at a Qwen3-class fused-QKV shape
/// (out_q=4096, out_k=out_v=1024, in_dim=4096). Throughput is
/// data-independent → random packed buffers.
pub mod kernel_benches {
    use wh_iron::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn qkv_bench(
        kernel: Kernel,
        fmt: QFormat,
        out_q: usize,
        out_k: usize,
        out_v: usize,
        in_dim: usize,
        m: usize,
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
        let max_rows = out_q.max(out_k).max(out_v);
        let sz = dt.size_bytes();
        // Per-matrix code-buffer length: 8-bit is one byte per element; sub-byte
        // widths tight-bit-pack `out_dim · in_dim` elements into u32 words.
        let codes = |o: usize| {
            if fmt.element_bits() == 8 {
                o * in_dim
            } else {
                crate::kernels::quant::format::bitstream_words(o * in_dim, fmt.element_bits())
            }
        };
        let scl = |o: usize| o * (in_dim / bs);
        let bytes = (codes(out_q) + codes(out_k) + codes(out_v)) * codes_dt.size_bytes()
            + (scl(out_q) + scl(out_k) + scl(out_v)) * scales_dt.size_bytes()
            + m * in_dim * sz
            + m * (out_q + out_k + out_v) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m * in_dim, dt))
            .buffer(BenchBuffer::random("w_q", codes(out_q), codes_dt))
            .buffer(BenchBuffer::random("scales_q", scl(out_q), scales_dt))
            .buffer(BenchBuffer::random("w_k", codes(out_k), codes_dt))
            .buffer(BenchBuffer::random("scales_k", scl(out_k), scales_dt))
            .buffer(BenchBuffer::random("w_v", codes(out_v), codes_dt))
            .buffer(BenchBuffer::random("scales_v", scl(out_v), scales_dt))
            .buffer(BenchBuffer::zeros("q_buf", m * out_q, dt).output())
            .buffer(BenchBuffer::zeros("k_buf", m * out_k, dt).output())
            .buffer(BenchBuffer::zeros("v_buf", m * out_v, dt).output())
            .constexpr("out_q", out_q as u32)
            .constexpr("out_k", out_k as u32)
            .constexpr("out_v", out_v as u32)
            .constexpr("in_dim", in_dim as u32)
            .constexpr("block_size", bs as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(max_rows as u32, m as u32, 3, [64, 1, 1])
            .bytes_moved(bytes as u64)
            // 3 fused qmms: 2 * m * (out_q + out_k + out_v) * in_dim
            .flops(2 * m as u64 * (out_q + out_k + out_v) as u64 * in_dim as u64)
            .with_shape_label(format!(
                "{} m={m} q={out_q} k={out_k} v={out_v} in={in_dim}",
                fmt.name()
            ))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxfp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_nvfp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxfp8_e4m3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxfp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_fp4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_nvfp8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_fp8_e5m2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int8,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int2,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int3,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int5_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int5,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int6_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int6,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxint2_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint2,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxint3_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint3,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxint4_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint4,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxint5_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint5,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxint6_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint6,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_mxint8_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Mxint8,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    // FP16-scale twins (nvfp8 / fp4 / fp8_e5m2 + int2..6 + int8). fp8_e4m3_f16
    // reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_nvfp8_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_nvfp8_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_fp4_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_fp8_e5m2_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int2_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int2F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int3_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int3F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int4_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int4F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int5_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int5F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int6_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int6F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_qkv(dt: DType) -> BenchSetup {
        qkv_bench(
            iron_int8_f16_batched_qkv_qmm::kernel_ir_for(dt),
            QFormat::Int8F16,
            4096,
            1024,
            1024,
            4096,
            8,
            dt,
        )
    }
}

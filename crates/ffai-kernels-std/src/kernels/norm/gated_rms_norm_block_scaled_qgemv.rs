//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused **gated-RMSNorm + block-scaled dequantizing GEMV** for the
//! spec-conformant formats (mxfp4 / nvfp4 / mxfp8 e4m3+e5m2 / nvfp8).
//!
//! `out = qmatmul(gated_rms_norm(y, z) · norm_weight, W_q)` in one dispatch —
//! the gated-RMSNorm staging of `ffai/gated_rms_norm_qgemv.rs` fused with the
//! block-scaled weight decode of `mlx/block_scaled_matmul.rs`.
//!
//! Phase 1 stages `inner[r,d] = y[r,d] · rsqrt(mean_d(y[r]²)+eps) ·
//! norm_weight[d] · silu(z[r,d])` into a `tg_inner` threadgroup buffer
//! (fp32), via the proven 2-simdgroup per-row scheme (`sg=0` even rows,
//! `sg=1` odd rows; one `simd_sum` per row). Phase 2 is the simple
//! one-output-row-per-TG block-scaled reduction GEMV reading `tg_inner`.
//! The staging is **weight-format-independent**, so it is identical across
//! all five formats — only phase 2's decode differs.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction**, `grid = [out_dim, 1, 1]`, `tpg = [64, 1, 1]`
//!   (2 simdgroups × 32 lanes — required by the phase-1 staging).
//! - `dv` a multiple of 32; `hv` even; `in_dim = hv·dv`; `in_dim` a multiple
//!   of `block_size`; 4-bit `block_size` a multiple of 8.
//! - `y` `[hv,dv]` fp32; `z` `[hv,dv]`; `norm_weight` `[dv]`; weight
//!   `[out_dim, in_dim/8]` u32 (4-bit) or `[out_dim, in_dim]` u8 (8-bit);
//!   scales `[out_dim, in_dim/block_size]` (u8 E8M0/E4M3 or f32 nvfp8).
//!
//! Block-scaled formats carry no bias. Codegen-only; correctness pinned by
//! the in-source `#[test_kernel]`s.

use ffai_kernels::kernel;

/// Fused gated-RMSNorm + block-scaled dequantizing GEMV, folded over the
/// 28-format axis (§7). Phase 1 (gate = silu(z), RMSNorm of `y`, staged into
/// `tg_inner`) is format-independent; phase 2 runs the block-scaled GEMV over
/// `tg_inner`. Per-element weight decode + per-block scale by the
/// `(BITS, WDEC, SKIND)` co-vars; buffer types by `(WT, ST)` — see
/// `gemm/block_scaled_matmul` for the legend. Decodes through
/// `kernels/primitives.rs`. Produces `mt_<FMT>_gated_rms_norm_qgemv`.
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
    suffix = "{FMT}_gated_rms_norm_qgemv",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    threadgroup_alloc("tg_inner", 4096, "f32");
    let sg = simd_id;
    let lane = simd_lane;
    // Phase 1: gated RMSNorm staged into tg_inner (2-simdgroup per-row scheme).
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = hv / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        let row_base = r * dv;
        let lane_base = lane * dv_per_lane;
        let mut partial_ssq = 0.0f32;
        for k in range(0u32, dv_per_lane, 1u32) {
            let yv = load(y[row_base + lane_base + k]);
            partial_ssq = partial_ssq + yv * yv;
        }
        let row_ssq = simd_sum(partial_ssq);
        let inv_rms = rsqrt(row_ssq / dv + eps);
        for k in range(0u32, dv_per_lane, 1u32) {
            let d = lane_base + k;
            let idx = row_base + d;
            let yv = load(y[idx]);
            let zv = load(z[idx]).cast::<f32>();
            let wv = load(norm_weight[d]).cast::<f32>();
            let gate = zv / (1.0f32 + exp(0.0f32 - zv));
            let inner = yv * inv_rms * wv * gate;
            threadgroup_store("tg_inner", idx, inner);
        }
    }
    threadgroup_barrier();
    // Phase 2: block-scaled GEMV over tg_inner (one output row per TG).
    let row = program_id::<0>();
    let in_dim = hv * dv;
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
                    let inner = threadgroup_load("tg_inner", p_off + i);
                    acc = acc + (val * scale) * inner;
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
                let inner = threadgroup_load("tg_inner", c);
                acc = acc + (val * scale) * inner;
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use ffai_kernels::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{kernels::quant::format::QFormat, utils::pack_f32};

    const TPG: u32 = 64;
    const EPS: f32 = 1e-5;

    /// Deterministic xorshift source (matches the int4 gated test generator).
    fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
            })
            .collect()
    }

    fn round(v: &[f32], dt: DType) -> Vec<f32> { crate::utils::unpack_f32(&pack_f32(v, dt), dt) }

    /// Gated-RMSNorm staging (CPU) → `inner[r,d]`, identical to the kernel's
    /// phase 1 (`y` fp32; `z`/`norm_weight` rounded to `dt`).
    fn stage_inner(y: &[f32], z: &[f32], nw: &[f32], hv: usize, dv: usize) -> Vec<f32> {
        let mut inner = vec![0.0f32; hv * dv];
        for r in 0..hv {
            let base = r * dv;
            let ssq: f32 = y[base..base + dv].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (ssq / dv as f32 + EPS).sqrt();
            for d in 0..dv {
                let g = z[base + d] / (1.0 + (-z[base + d]).exp());
                inner[base + d] = y[base + d] * inv_rms * nw[d] * g;
            }
        }
        inner
    }

    fn gated_setup(
        kernel: Kernel,
        fmt: QFormat,
        hv: usize,
        dv: usize,
        out_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let in_dim = hv * dv;
        // Block-scaled weights `[out_dim, in_dim]` via the shared codec.
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let r = (i / in_dim) as f32;
                let c = (i % in_dim) as f32;
                let mag = (0.5 + r * 0.25) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::kernels::quant::format::pack(fmt, &w, out_dim, in_dim);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_dim, in_dim);
        // y fp32; z / norm_weight rounded through dt (kernel loads them as T).
        let y = source(in_dim, 0xA1, 2.0, 0.1);
        let z = round(&source(in_dim, 0xD4, 1.5, 0.0), dt);
        let nw = round(&source(dv, 0xB2, 0.4, 1.0), dt);
        let inner = stage_inner(&y, &z, &nw, hv, dv);
        let expected: Vec<f32> = (0..out_dim)
            .map(|r| (0..in_dim).map(|c| wdq[r * in_dim + c] * inner[c]).sum())
            .collect();
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
            .input(TestBuffer::from_vec("y", pack_f32(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack_f32(&z, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&nw, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", EPS.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", out_dim, dt))
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            out_dim as u32,
            1,
            1,
            [TPG, 1, 1],
        )
    }

    // hv=4, dv=128, in_dim=512 (÷ 16/32), out_dim=4.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_mxfp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Mxfp4, 4, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_nvfp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp4, 4, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxfp8_e4m3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxfp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Nvfp8, 4, 128, 4, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_fp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 4, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_fp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            128,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int8_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int8, 4, 128, 4, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). in_dim = hv·dv = 512 satisfies
    // `in_dim·bits % 32 == 0` for every width, so each row's bit-stream is
    // word-aligned; the kernel and oracle share the codec, so the GPU output
    // tracks the gated-dequant-then-dot reference to float precision.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int2_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int2, 4, 128, 4, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int3_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int3, 4, 128, 4, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int4, 4, 128, 4, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int5_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int5, 4, 128, 4, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(mt_int6_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Int6, 4, 128, 4, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxint2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxint3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxint4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxint5_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxint6_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_mxint8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            128,
            4,
            dt,
        )
    }

    // FP16-scale twins of the FP32-scaled formats. Same element packing as their
    // twin (so codes dtype is unchanged); only the scale tensor binds as f16.
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_nvfp8_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_nvfp8_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_fp4_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_fp8_e5m2_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_int2_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_int3_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_int4_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_int5_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_int6_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            128,
            4,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_gated_rms_norm_qgemv(dt: DType) -> TestSetup {
        gated_setup(
            mt_int8_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            128,
            4,
            dt,
        )
    }
}

/// Decode-shape benches at the Qwen3.6-A3B activation shape (hv=16, dv=128,
/// in_dim=2048, out=2048). Throughput is data-independent → random buffers.
pub mod kernel_benches {
    use ffai_kernels::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn gated_bench(
        kernel: Kernel,
        fmt: QFormat,
        hv: usize,
        dv: usize,
        out_dim: usize,
        dt: DType,
    ) -> BenchSetup {
        let in_dim = hv * dv;
        let n = out_dim * in_dim;
        let n_blocks = out_dim * (in_dim / fmt.block_size());
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
            + in_dim * 4   // y (fp32)
            + in_dim * sz  // z
            + dv * sz      // norm_weight
            + out_dim * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("y", in_dim, DType::F32))
            .buffer(BenchBuffer::random("z", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", dv, dt))
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", out_dim, dt).output())
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64)
            .with_shape_label(format!("{} hv={hv} dv={dv} m={out_dim}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxfp4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxfp8_e4m3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxfp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_gated(dt: DType) -> BenchSetup {
        gated_bench(mt_fp4_gated_rms_norm_qgemv::kernel_ir_for(dt), QFormat::Fp4, 16, 128, 2048, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_fp8_e5m2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int8,
            16,
            128,
            2048,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int2,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int3,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int5_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int5,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int6_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int6,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxint2_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint2,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxint3_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint3,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxint4_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint4,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxint5_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint5,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxint6_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint6,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_mxint8_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Mxint8,
            16,
            128,
            2048,
            dt,
        )
    }
    // FP16-scale twins. fp8_e4m3_f16 reuses the nvfp8_f16 kernel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp8_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_nvfp8_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_fp4_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp4F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_fp8_e5m2_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int2_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int2F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int3_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int3F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int4_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int4F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int5_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int5F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int6_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int6F16,
            16,
            128,
            2048,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_gated(dt: DType) -> BenchSetup {
        gated_bench(
            mt_int8_f16_gated_rms_norm_qgemv::kernel_ir_for(dt),
            QFormat::Int8F16,
            16,
            128,
            2048,
            dt,
        )
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantized-weight variants of the cooperative (simdgroup-matrix
//! MMA) 2D convolution. These are the M≥32 ALU-throughput conv path for the
//! spec-conformant quantized formats — a direct fusion of two GPU-verified
//! templates:
//!
//! - `ffai/conv2d_mma.rs` (dense cooperative conv2d) supplies the **entire
//!   geometry verbatim**: the implicit-im2col A-load (input gather with
//!   in-bounds masking), the threadgroup/simdgroup allocation sizes, the
//!   4-frag × 4-k-inner `simdgroup_matmul` inner loop, the C-store, and the
//!   dispatch grid.
//!
//! - `mlx/block_scaled_mma.rs` (block-scaled MMA qmm) supplies the per-format
//!   **weight dequant staging** — the ONLY thing that changes vs the dense
//!   conv2d_mma is the cooperative B-load into `bs`.
//!
//! ## Implicit im2col as a quantized-weight GEMM
//!
//! Treat the convolution as a GEMM where the B (weight) operand is quantized:
//!
//!   out[BN_pixels, BM_oc] = A[BN_pixels, BK] × dequant(B)[BK, BM_oc]
//!
//! where:
//!
//!   - `BK  = in_ch * kh * kw` (filter taps per output position = contraction)
//!   - `BN  = batch * out_h * out_w` (output positions = "pixels")
//!   - `BM  = out_ch` (output channels)
//!
//! The A matrix is never materialised — each lane computes its `(kh, kw, ic)`
//! → `(h_in, w_in, ic)` gather index on-the-fly (copied verbatim from the
//! dense conv2d_mma). The B matrix is the quantized filter laid out as the
//! 2-D matrix `[out_ch, BK]` (row `oc`, column `kt = ic*kh*kw + ky*kw + kx`).
//!
//! ## Quantized B-load (the only change vs dense conv2d_mma)
//!
//! Each lane stages 8 contiguous taps `kt` for its oc-row `b_oc_row` into
//! `bs`. The dense load reads `weight[w_oc_base + kt]` directly; here the
//! same `(oc, kt)` element is dequantized instead:
//!
//!   - 4-bit (E2M1): weight is `[out_ch, BK/8]` u32 (8 nibbles/word). The
//!     nibble for tap `kt` is word `oc*(BK/8) + kt/8`, shift `(kt%8)*4`.
//!
//!   - 8-bit (E4M3 / E5M2 / int8): weight is `[out_ch, BK]` u8 (1 byte/tap).
//!     The byte for tap `kt` is `oc*BK + kt`.
//!
//!   - The block scale for tap `kt` is `scales[oc*(BK/block_size) + kt/block_size]`
//!     (E8M0 `exp2(b-127)`, E4M3 micro-scale × global FP32, or raw FP32).
//!
//! Decoding is done tap-by-tap so arbitrary `kt` alignment is handled (the
//! K-loop steps by 32 but `BK` need only be a multiple of 32 and of the
//! format's block_size). The dense in-bounds masking is preserved:
//! `select(kt < total_k, decoded, 0.0)`, and the safe (clamped) index keeps
//! the gather in range.
//!
//! ## Dispatch invariants (identical to dense conv2d_mma)
//!
//! - **Mode: Reduction**, grid `[out_ch/32, (batch*out_h*out_w)/32, 1]`,
//!   tpg = 128 (4 simdgroups × 32 lanes, 2×2 warp grid).
//! - BM = BN = 32, output tile 32×32. `out_ch` and `n_pixels` multiples of 32.
//! - `BK = in_ch*kh*kw` a multiple of 32 (the MMA K-tile) and of block_size.
//! - stride = 1, dilation = 1, padding = 0 (vision patch-conv style).
//! - NCHW input, quantized OIHW-flattened weight, pixel-major out. No bias.
//!
//! Codegen-only. Correctness validated by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// Quantized-weight conv2d (simdgroup-MMA), folded over the 28-format axis (§7).
/// Implicit-im2col A-load, the 8×8 simdgroup-fragment MMA, and the write-back are
/// format-independent; only the W-dequant "B-load" folds onto `(BITS, WDEC,
/// SKIND)` (buffer types `(WT, ST)`, legend in `gemm/block_scaled_matmul`),
/// through `kernels/primitives.rs`. Produces `mt_<FMT>_conv2d_mma`.
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
    suffix = "{FMT}_conv2d_mma",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    input: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    // ── Geometry copied verbatim from dense conv2d_mma ──
    let oc_tile = tgid_x;
    let px_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T);
    threadgroup_alloc("bs", 1152, T);
    let c_f00 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f00, 0, 0.0f32);
    simdgroup_elem_store(c_f00, 1, 0.0f32);
    let c_f01 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f01, 0, 0.0f32);
    simdgroup_elem_store(c_f01, 1, 0.0f32);
    let c_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f10, 0, 0.0f32);
    simdgroup_elem_store(c_f10, 1, 0.0f32);
    let c_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(c_f11, 0, 0.0f32);
    simdgroup_elem_store(c_f11, 1, 0.0f32);
    let a_f0 = simdgroup_alloc::<T, 8, 8>();
    let a_f1 = simdgroup_alloc::<T, 8, 8>();
    let b_f0 = simdgroup_alloc::<T, 8, 8>();
    let b_f1 = simdgroup_alloc::<T, 8, 8>();
    let kk = kh * kw;
    let total_k = in_ch * kk;
    let out_hw = out_h * out_w;
    let a_px_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_px = px_tile * 32u32 + a_px_row;
    let n_px = global_px / out_hw;
    let rem_px = global_px - n_px * out_hw;
    let oh_px = rem_px / out_w;
    let ow_px = rem_px - oh_px * out_w;
    let in_n_stride = in_ch * in_h * in_w;
    let px_in_base = n_px * in_n_stride;
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    // Quantized weight layout: 4-bit packs 8 nibbles/u32 word → row stride
    // total_k/8 words; one E8M0 scale byte per block → row stride
    // total_k/block_size.
    let packs_per_row = total_k / 8u32;
    let n_blocks = total_k / block_size;
    let w_pack_row_base = global_oc * packs_per_row;
    let sb_base = global_oc * n_blocks;
    let w_byte_row_base = global_oc * total_k;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit im2col gather) — verbatim from conv2d_mma ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / kk;
            let rem_kt = kt_safe - ic * kk;
            let ky = rem_kt / kw;
            let kx = rem_kt - ky * kw;
            let ih = oh_px + ky;
            let iw = ow_px + kx;
            let in_idx = px_in_base + ic * in_h * in_w + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_px_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (W dequant, folded over the format axis) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let sraw = load(scales[sb_base + kt_safe / block_size]);
            let scale = if SKIND == 0u32 {
                exp2(sraw.cast::<f32>() - 127.0f32)
            } else if SKIND == 1u32 {
                mt_decode_e4m3(sraw.cast::<u32>()) * global
            } else {
                sraw.cast::<f32>()
            };
            let elem = if WDEC == 0u32 {
                let pack = load(weight[w_pack_row_base + kt_safe / 8u32]);
                mt_decode_e2m1((pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32)
            } else if WDEC == 1u32 {
                let bit_off = (w_byte_row_base + kt_safe) * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                let spill = BITS - lo_bits;
                let w0 = load(weight[word_idx]);
                let w1 = load(weight[select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                let qf = q.cast::<f32>();
                select(q >= half, qf - full, qf)
            } else {
                let raw = load(weight[w_byte_row_base + kt_safe]).cast::<u32>();
                if WDEC == 2u32 {
                    mt_decode_e4m3(raw)
                } else if WDEC == 3u32 {
                    mt_decode_e5m2(raw)
                } else {
                    mt_decode_int8(raw)
                }
            };
            let val = select(in_bounds, elem * scale, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_oc_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        // ─ 3. MMA inner loop — copied verbatim from conv2d_mma ─
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 1
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 2
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 3
        simdgroup_elem_store(a_f0, 0, threadgroup_load("as", row_a0 * stride + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("as", row_a0 * stride + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("as", row_a1 * stride + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("as", row_a1 * stride + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("bs", (col_b0 + fn0) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("bs", (col_b0 + fn1) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("bs", (col_b1 + fn0) * stride + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("bs", (col_b1 + fn1) * stride + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    // ── 4. Write 4 C frags to global out — verbatim from conv2d_mma ──
    let out_px_base = px_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_px_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::{QFormat, ScaleKind},
        utils::{pack_f32, unpack_f32},
    };

    /// Bounded zig-zag ramp (keeps f16/bf16 in range), identical to the dense
    /// conv2d_mma helper.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 2D conv oracle, pixel-major output `[n_pixels, out_ch]`, run over
    /// the *dequantized* filter `[out_ch, BK]` (BK = in_ch*kh*kw, col =
    /// (ic*kh + ky)*kw + kx). stride=1, dilation=1, pad=0, no bias. The SAME
    /// dense math as conv2d_mma.rs's `naive_conv2d_mma`.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv2d_mma(
        input: &[f32],
        weight: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let out_hw = out_h * out_w;
        let n_pixels = batch * out_hw;
        let mut out = vec![0.0f32; n_pixels * out_ch];
        for n in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let pixel = n * out_hw + oh * out_w + ow;
                    for oc in 0..out_ch {
                        let mut acc = 0.0f32;
                        for ic in 0..in_ch {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ih = oh + ky;
                                    let iw = ow + kx;
                                    let in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                                    // Quantized filter is the 2-D matrix
                                    // [out_ch, BK]; col = (ic*kh+ky)*kw+kx.
                                    let col = (ic * kh + ky) * kw + kx;
                                    let w_idx = oc * (in_ch * kh * kw) + col;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        out[pixel * out_ch + oc] = acc;
                    }
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the `[out_ch, BK]` filter via the
    /// shared codec, dequantize for the oracle, and run the dense conv2d_mma
    /// math over the dequantized filter. Mirrors conv2d_mma.rs's `mma_setup`
    /// grid + KernelMode, just swapping the dense filter for a quantized one.
    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        assert_eq!(out_ch % 32, 0, "out_ch must be a multiple of 32 for the MMA tile");
        assert_eq!(n_pixels % 32, 0, "n_pixels must be a multiple of 32 for the MMA tile");
        // BK = in_ch*kh*kw — the quantized filter is [out_ch, BK]. Must be a
        // multiple of the format's block_size and of the 32 MMA K-tile.
        let bk = in_ch * kh * kw;
        assert_eq!(bk % 32, 0, "BK (in_ch*kh*kw) must be a multiple of the 32 MMA K-tile");
        assert_eq!(bk % fmt.block_size(), 0, "BK must be a multiple of the format block_size");
        let n_out = n_pixels * out_ch;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 2.0);
        // Quantize the [out_ch, BK] filter via the shared codec.
        let filter_f = ramp(out_ch * bk, 11, 2.0);
        let p = crate::quant::format::pack(fmt, &filter_f, out_ch, bk);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, bk);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = naive_conv2d_mma(&input, &wdq, batch, in_ch, in_h, in_w, out_ch, kh, kw);
        // Axis-driven binding (robust across all element widths/scale kinds):
        //   - 8-bit codes (E4M3/E5M2/int8/mxint8) bind one uchar each; every
        //     sub-byte width (4-bit E2M1 + the 2/3/5/6-bit int bit-streams)
        //     binds as a packed u32 code stream.
        //   - FP32-scaled formats bind raw f32 scales; E8M0/E4M3-scaled formats
        //     bind one byte each.
        // For the pre-existing formats this is identical to the old `== 4` /
        // `matches!` logic (4-bit → U32, 8-bit → U8; the float-scale list maps
        // exactly to ScaleKind::F32), so there is no regression.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            ScaleKind::F32 => DType::F32,
            ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (out_ch / 32) as u32,
            (n_pixels / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // Dims: in_ch=4, kh=kw=4 → BK=64 (2 K-blocks of 32; divisible by 16/32/64).
    // 7×7 input → out 4×4, batch=2 → n_pixels=32; out_ch=32. One 32×32 tile.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp4_conv2d_mma::kernel_ir_for(dt), QFormat::Mxfp4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp4_conv2d_mma::kernel_ir_for(dt), QFormat::Nvfp4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_conv2d_mma::kernel_ir_for(dt), QFormat::Fp4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e4m3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e5m2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_conv2d_mma::kernel_ir_for(dt), QFormat::Nvfp8, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    // fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int8_conv2d_mma::kernel_ir_for(dt), QFormat::Int8, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    // ── Symmetric integer formats (FP32 group scale, group 64) ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int2_conv2d_mma::kernel_ir_for(dt), QFormat::Int2, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int3_conv2d_mma::kernel_ir_for(dt), QFormat::Int3, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int4_conv2d_mma::kernel_ir_for(dt), QFormat::Int4, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int5_conv2d_mma::kernel_ir_for(dt), QFormat::Int5, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int6_conv2d_mma::kernel_ir_for(dt), QFormat::Int6, 2, 4, 7, 7, 32, 4, 4, dt)
    }

    // ── E8M0-scaled symmetric integer formats (block 32) ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint2_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint3_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint4_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint5_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint6_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint8_conv2d_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    // ── FP16-scale twins of the FP32-scaled formats ──
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp4_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_conv2d_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_f16_conv2d_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            2,
            4,
            7,
            7,
            32,
            4,
            4,
            dt,
        )
    }
}

/// Realistic vision-encoder benches — the M≥32 simdgroup-matrix throughput
/// path for quantized-weight conv2d. Random packed buffers (throughput is
/// data-independent). Shape: 1×1 conv on a 32×32 feature map with in_ch=64
/// → BK = 64·1·1 = 64 (divisible by 16/32/64 block sizes and the 32 K-tile),
/// n_pixels = 1024, out_ch = 1024.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::{QFormat, ScaleKind};

    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_pixels = batch * out_h * out_w;
        let n_out = n_pixels * out_ch;
        // BK = in_ch*kh*kw — the quantized filter is [out_ch, BK].
        let bk = in_ch * kh * kw;
        let n_blocks = out_ch * (bk / fmt.block_size());
        // Axis-driven code buffer: 8-bit codes are one byte each; every sub-byte
        // width packs into a tight u32 bit-stream (`bitstream_words` collapses to
        // the old `n/8` for the 4-bit format, so no regression).
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (out_ch * bk, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(out_ch * bk, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            ScaleKind::F32 => DType::F32,
            ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_ch / 32) as u32, (n_pixels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // 2 * N * Co * Ho * Wo * Ci * kh * kw (stride=1, pad=0)
            .flops(
                2 * (batch as u64)
                    * (out_ch as u64)
                    * (out_h as u64)
                    * (out_w as u64)
                    * (in_ch as u64)
                    * (kh as u64)
                    * (kw as u64),
            )
    }

    // One bench per QFormat via the shared `mma_bench` helper — mirrors the
    // `conv3d_mma_bench_fmt!` pattern (and every other block-scaled conv file)
    // instead of 29 hand-written fns. Representative MMA shape: 1×64×32×32 →
    // out_ch 1024, 1×1 kernel.
    macro_rules! conv2d_mma_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                mma_bench($kernel(dt), $fmt, 1, 64, 32, 32, 1024, 1, 1, dt)
            }
        };
    }
    conv2d_mma_bench_fmt!(bench_mxfp4, mt_mxfp4_conv2d_mma::kernel_ir_for, QFormat::Mxfp4);
    conv2d_mma_bench_fmt!(bench_nvfp4, mt_nvfp4_conv2d_mma::kernel_ir_for, QFormat::Nvfp4);
    conv2d_mma_bench_fmt!(bench_fp4, mt_fp4_conv2d_mma::kernel_ir_for, QFormat::Fp4);
    conv2d_mma_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_conv2d_mma::kernel_ir_for,
        QFormat::Mxfp8E4
    );
    conv2d_mma_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_conv2d_mma::kernel_ir_for,
        QFormat::Mxfp8E5
    );
    conv2d_mma_bench_fmt!(bench_nvfp8, mt_nvfp8_conv2d_mma::kernel_ir_for, QFormat::Nvfp8);
    conv2d_mma_bench_fmt!(bench_fp8_e5m2, mt_fp8_e5m2_conv2d_mma::kernel_ir_for, QFormat::Fp8E5m2);
    conv2d_mma_bench_fmt!(bench_int8, mt_int8_conv2d_mma::kernel_ir_for, QFormat::Int8);
    conv2d_mma_bench_fmt!(bench_int2, mt_int2_conv2d_mma::kernel_ir_for, QFormat::Int2);
    conv2d_mma_bench_fmt!(bench_int3, mt_int3_conv2d_mma::kernel_ir_for, QFormat::Int3);
    conv2d_mma_bench_fmt!(bench_int4, mt_int4_conv2d_mma::kernel_ir_for, QFormat::Int4);
    conv2d_mma_bench_fmt!(bench_int5, mt_int5_conv2d_mma::kernel_ir_for, QFormat::Int5);
    conv2d_mma_bench_fmt!(bench_int6, mt_int6_conv2d_mma::kernel_ir_for, QFormat::Int6);
    conv2d_mma_bench_fmt!(bench_mxint2, mt_mxint2_conv2d_mma::kernel_ir_for, QFormat::Mxint2);
    conv2d_mma_bench_fmt!(bench_mxint3, mt_mxint3_conv2d_mma::kernel_ir_for, QFormat::Mxint3);
    conv2d_mma_bench_fmt!(bench_mxint4, mt_mxint4_conv2d_mma::kernel_ir_for, QFormat::Mxint4);
    conv2d_mma_bench_fmt!(bench_mxint5, mt_mxint5_conv2d_mma::kernel_ir_for, QFormat::Mxint5);
    conv2d_mma_bench_fmt!(bench_mxint6, mt_mxint6_conv2d_mma::kernel_ir_for, QFormat::Mxint6);
    conv2d_mma_bench_fmt!(bench_mxint8, mt_mxint8_conv2d_mma::kernel_ir_for, QFormat::Mxint8);
    conv2d_mma_bench_fmt!(
        bench_nvfp8_f16,
        mt_nvfp8_f16_conv2d_mma::kernel_ir_for,
        QFormat::Nvfp8F16
    );
    conv2d_mma_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_conv2d_mma::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    conv2d_mma_bench_fmt!(bench_fp4_f16, mt_fp4_f16_conv2d_mma::kernel_ir_for, QFormat::Fp4F16);
    conv2d_mma_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_conv2d_mma::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    conv2d_mma_bench_fmt!(bench_int2_f16, mt_int2_f16_conv2d_mma::kernel_ir_for, QFormat::Int2F16);
    conv2d_mma_bench_fmt!(bench_int3_f16, mt_int3_f16_conv2d_mma::kernel_ir_for, QFormat::Int3F16);
    conv2d_mma_bench_fmt!(bench_int4_f16, mt_int4_f16_conv2d_mma::kernel_ir_for, QFormat::Int4F16);
    conv2d_mma_bench_fmt!(bench_int5_f16, mt_int5_f16_conv2d_mma::kernel_ir_for, QFormat::Int5F16);
    conv2d_mma_bench_fmt!(bench_int6_f16, mt_int6_f16_conv2d_mma::kernel_ir_for, QFormat::Int6F16);
    conv2d_mma_bench_fmt!(bench_int8_f16, mt_int8_f16_conv2d_mma::kernel_ir_for, QFormat::Int8F16);
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantized-weight variants of the cooperative (simdgroup-matrix
//! MMA) patch embedding. These are the M≥32 ALU-throughput patch-embed path for
//! the spec-conformant quantized formats — a direct fusion of two GPU-verified
//! templates:
//!
//! - `ffai/patch_embed_mma.rs` (dense cooperative patch embed) supplies the
//!   **entire geometry verbatim**: the implicit patch-unfold A-load (input
//!   gather with in-bounds masking), the threadgroup/simdgroup allocation
//!   sizes, the 4-frag × 4-k-inner `simdgroup_matmul` inner loop, the
//!   bias-added C-store (output layout `[num_patches, hidden]`), and the
//!   dispatch grid.
//!
//! - `ffai/conv2d_mma_block_scaled.rs` (block-scaled MMA conv) supplies the
//!   per-format **weight dequant staging** — the ONLY thing that changes vs the
//!   dense patch_embed_mma is the cooperative B-load into `bs`.
//!
//! ## Patch embed as a quantized-weight GEMM
//!
//! The patch embedding is a linear projection, algebraically a conv2d with
//! `stride = patch`, no overlap, no padding, but a `[num_patches, hidden]`
//! output layout. Treat it as a GEMM where the B (weight) operand is quantized:
//!
//!   out[num_patches, hidden] = A[num_patches, patch_dim] × dequant(B)[patch_dim, hidden]
//!
//! where:
//!
//!   - `patch_dim = in_ch * patch_h * patch_w` (taps per patch = contraction K).
//!   - `A` is the implicit patch unfold — each lane computes its `(ic, py, px)`
//!     → `(ic, py0+py, px0+px)` gather index on-the-fly (copied verbatim from
//!     the dense patch_embed_mma).
//!   - `B` is the quantized projection weight laid out as the 2-D matrix
//!     `[hidden, patch_dim]` (row `h` = hidden unit, column `kt = ic*patch_h*patch_w
//!     + py*patch_w + px`). This is the same row-major `[N, K]` weight the
//!     `kernels::quant::format` packer produces.
//!
//! ## Quantized B-load (the only change vs dense patch_embed_mma)
//!
//! Each lane stages 8 contiguous taps `kt` for its hidden-unit row `b_h_row`
//! into `bs`. The dense load reads `weight[w_h_base + kt]` directly; here the
//! same `(h, kt)` element is dequantized instead — identical decode to
//! `conv2d_mma_block_scaled` with the conv's `total_k` replaced by `patch_dim`:
//!
//!   - 4-bit (E2M1): weight is `[hidden, patch_dim/8]` u32 (8 nibbles/word). The
//!     nibble for tap `kt` is word `h*(patch_dim/8) + kt/8`, shift `(kt%8)*4`.
//!
//!   - 8-bit (E4M3 / E5M2 / int8 / MXINT8): weight is `[hidden, patch_dim]` u8
//!     (1 byte/tap). The byte for tap `kt` is `h*patch_dim + kt`.
//!
//!   - sub-byte symmetric int (int2/3/4/5/6 + MXINT2..6): weight is a FLAT
//!     row-major u32 bit-stream, tight-packed LSB-first by `kernels::quant::format::pack`.
//!     The N-bit two's-complement code for tap `kt` of row `h` lives at GLOBAL
//!     bit offset `(h*patch_dim + kt)*N`, read straddle-aware across two words
//!     and float-sign-extended (`code - 2^N` when the top bit is set). `patch_dim`
//!     is a multiple of 32, so every row's bit-stream is word-aligned.
//!
//!   - The block scale for tap `kt` is
//!     `scales[h*(patch_dim/block_size) + kt/block_size]` (E8M0 `exp2(b-127)`,
//!     E4M3 micro-scale × global FP32, or raw FP32).
//!
//! The dense in-bounds masking is preserved: `select(kt < patch_dim, decoded,
//! 0.0)`, and the safe (clamped) index keeps the gather in range.
//!
//! ## Dispatch invariants (identical to dense patch_embed_mma)
//!
//! - **Mode: Reduction**, grid `[hidden/32, num_patches/32, 1]`, tpg = 128
//!   (4 simdgroups × 32 lanes, 2×2 warp grid).
//! - BM = BN = 32, output tile 32×32. `hidden` and `num_patches` multiples of 32.
//! - `patch_dim = in_ch*patch_h*patch_w` a multiple of 32 (the MMA K-tile) and
//!   of the format's block_size.
//! - Single image (no batch — matches `patch_embed.rs` layout). Per-channel
//!   `bias` stays `T` (tiny + precision-sensitive); only the weight is quantized.
//!
//! Codegen-only. Correctness validated by the in-source `#[test_kernel]`s.

use metaltile::kernel;

/// Quantized-weight patch-embed (simdgroup-MMA), folded over the 28-format axis
/// (§7). Implicit-im2col A-load + 8×8 simdgroup MMA + write-back are
/// format-independent; only the W-dequant B-load folds onto `(BITS, WDEC,
/// SKIND)`, through `kernels/primitives.rs`. Sub-byte ints index a global
/// bit-stream `(global_h*patch_dim + tap)*BITS`. Produces `mt_<FMT>_patch_embed_mma`.
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
    suffix = "{FMT}_patch_embed_mma",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    image: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] patch_h: u32,
    #[constexpr] patch_w: u32,
    #[constexpr] hidden: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    // ── Geometry copied verbatim from dense patch_embed_mma ──
    let h_tile = tgid_x;
    let pat_tile = tgid_y;
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
    let phw = patch_h * patch_w;
    let patch_dim = in_ch * phw;
    let patches_w = in_w / patch_w;
    let input_plane = in_h * in_w;
    let a_pat_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pat = pat_tile * 32u32 + a_pat_row;
    let py0 = (global_pat / patches_w) * patch_h;
    let px0 = (global_pat - (global_pat / patches_w) * patches_w) * patch_w;
    let b_h_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_h = h_tile * 32u32 + b_h_row;
    // Quantized weight layout: 4-bit packs 8 nibbles/u32 word → row stride
    // patch_dim/8 words; one E8M0 scale byte per block → row stride
    // patch_dim/block_size.
    let packs_per_row = patch_dim / 8u32;
    let n_blocks = patch_dim / block_size;
    let w_pack_row_base = global_h * packs_per_row;
    let sb_base = global_h * n_blocks;
    let w_byte_base = global_h * patch_dim;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    for kb in range(0u32, patch_dim, 32u32) {
        // ─ 1. Coop A load (implicit patch unfold gather) — verbatim from patch_embed_mma ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < patch_dim;
            let kt_safe = select(in_bounds, kt, 0u32);
            let ic = kt_safe / phw;
            let rem_kt = kt_safe - ic * phw;
            let py = rem_kt / patch_w;
            let px = rem_kt - py * patch_w;
            let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
            let raw = load(image[img_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pat_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (W dequant, folded over the format axis) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < patch_dim;
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
                let bit_off = (w_byte_base + kt_safe) * BITS;
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
                let raw = load(weight[w_byte_base + kt_safe]).cast::<u32>();
                if WDEC == 2u32 {
                    mt_decode_e4m3(raw)
                } else if WDEC == 3u32 {
                    mt_decode_e5m2(raw)
                } else {
                    mt_decode_int8(raw)
                }
            };
            let val = select(in_bounds, elem * scale, 0.0f32).cast::<T>();
            threadgroup_store("bs", b_h_row * stride + b_k_base + i, val);
        }
        threadgroup_barrier();
        // ─ 3. MMA inner loop — copied verbatim from patch_embed_mma ─
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
    // ── 4. Add bias and write 4 C frags to global out — verbatim from patch_embed_mma ──
    // out layout: [num_patches, hidden].
    let out_pat_base = pat_tile * 32u32 + sm * 16u32;
    let out_h_base = h_tile * 32u32 + sn * 16u32;
    let b00 = load(bias[out_h_base + fn0]).cast::<f32>();
    let b01 = load(bias[out_h_base + fn1]).cast::<f32>();
    let b10 = load(bias[out_h_base + 8u32 + fn0]).cast::<f32>();
    let b11 = load(bias[out_h_base + 8u32 + fn1]).cast::<f32>();
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f00, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f00, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f01, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f01, 1) + b11).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn0],
        (simdgroup_elem_load(c_f10, 0) + b00).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + fn1],
        (simdgroup_elem_load(c_f10, 1) + b01).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn0],
        (simdgroup_elem_load(c_f11, 0) + b10).cast::<T>(),
    );
    store(
        out[(out_pat_base + 8u32 + fm) * hidden + out_h_base + 8u32 + fn1],
        (simdgroup_elem_load(c_f11, 1) + b11).cast::<T>(),
    );
}
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Bounded zig-zag ramp (keeps f16/bf16 in range), identical to the dense
    /// patch_embed_mma helper.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Explicit unfold + projection + bias oracle, run over the *dequantized*
    /// weight `[hidden, patch_dim]` (col = ic*patch_h*patch_w + py*patch_w + px).
    /// Output `[num_patches, hidden]`, f32. The SAME dense math as
    /// patch_embed_mma.rs's `naive_patch_embed_mma`.
    #[allow(clippy::too_many_arguments)]
    fn naive_patch_embed(
        image: &[f32],
        weight: &[f32],
        bias: &[f32],
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let patch_dim = in_ch * patch_h * patch_w;
        let input_plane = in_h * in_w;
        let patches_h = in_h / patch_h;
        let patches_w = in_w / patch_w;
        let num_patches = patches_h * patches_w;
        let mut out = vec![0.0f32; num_patches * hidden];
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                let pat = ph * patches_w + pw;
                let py0 = ph * patch_h;
                let px0 = pw * patch_w;
                for h in 0..hidden {
                    let mut acc = bias[h];
                    for ic in 0..in_ch {
                        for py in 0..patch_h {
                            for px in 0..patch_w {
                                let img_idx = ic * input_plane + (py0 + py) * in_w + (px0 + px);
                                let w_idx =
                                    h * patch_dim + ic * patch_h * patch_w + py * patch_w + px;
                                acc += image[img_idx] * weight[w_idx];
                            }
                        }
                    }
                    out[pat * hidden + h] = acc;
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the `[hidden, patch_dim]` projection
    /// weight via the shared codec, dequantize for the oracle, and run the dense
    /// patch_embed_mma math over the dequantized weight. Mirrors
    /// patch_embed_mma.rs's `mma_setup` grid + KernelMode, just swapping the
    /// dense weight for a quantized one (the bias stays `T`).
    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> TestSetup {
        let num_patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        assert_eq!(hidden % 32, 0, "hidden must be a multiple of 32 for the MMA tile");
        assert_eq!(num_patches % 32, 0, "num_patches must be a multiple of 32");
        // patch_dim is the contraction K — must be a multiple of the 32 MMA
        // K-tile and of the format's block_size.
        assert_eq!(patch_dim % 32, 0, "patch_dim must be a multiple of the 32 MMA K-tile");
        assert_eq!(patch_dim % fmt.block_size(), 0, "patch_dim must be a multiple of block_size");
        let n_out = num_patches * hidden;
        let image_f = ramp(in_ch * in_h * in_w, 13, 2.0);
        // Quantize the [hidden, patch_dim] projection weight via the shared codec.
        let weight_f = ramp(hidden * patch_dim, 11, 2.0);
        let p = crate::kernels::quant::format::pack(fmt, &weight_f, hidden, patch_dim);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, hidden, patch_dim);
        let bias_f = ramp(hidden, 5, 1.0);
        let image = unpack_f32(&pack_f32(&image_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected =
            naive_patch_embed(&image, &wdq, &bias, in_ch, in_h, in_w, patch_h, patch_w, hidden);
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types
        // (these are exactly equivalent to the old per-format lists for the
        // pre-existing formats — 4-bit collapses to the u32 branch).
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        // FP32 scales bind as f32; FP16 scales as f16; E8M0/E4M3 scales as one
        // byte. Driven off the format so each new precision picks the right type.
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("image", pack_f32(&image_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (hidden / 32) as u32,
            (num_patches / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // Dims: in_ch=4, patch 4×4 → patch_dim=64 (2 K-blocks of 32; divisible by
    // 16/32/64). 32×32 image → 64 patches; hidden=32. Two 32×32 patch tiles.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_patch_embed_mma::kernel_ir_for(dt), QFormat::Fp4, 4, 32, 32, 4, 4, 32, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e4m3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxfp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). patch_dim=64 is a multiple of 32
    // (the MMA K-tile) and of both block sizes, and `patch_dim*bits % 32 == 0`
    // for every width (64 is a multiple of 32), so each weight row's tight
    // bit-stream is word-aligned. The kernel and oracle share the codec, so the
    // GPU output tracks the dequant-then-projection reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_mxint8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // FP16-scale twins. Same dims / oracle as their FP32 siblings — only the
    // scale tensor binds as f16 (`mma_setup` picks DType::F16 off `scale_kind`).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + f16-scale shape).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_fp8_e5m2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int3_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int5_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int6_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_patch_embed_mma(dt: DType) -> TestSetup {
        mma_setup(
            mt_int8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            32,
            32,
            4,
            4,
            32,
            dt,
        )
    }
}

/// Realistic vision-encoder benches — the M≥32 simdgroup-matrix throughput
/// path for quantized-weight patch embedding. Random packed buffers
/// (throughput is data-independent). Shape: 8×8 patch, 4 channels
/// (patch_dim = 256, divisible by 16/32/64 block sizes and the 32 K-tile),
/// 256×256 image → 32×32 = 1024 patches, hidden = 1024.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        patch_h: usize,
        patch_w: usize,
        hidden: usize,
        dt: DType,
    ) -> BenchSetup {
        let num_patches = (in_h / patch_h) * (in_w / patch_w);
        let patch_dim = in_ch * patch_h * patch_w;
        let n_out = num_patches * hidden;
        let n_blocks = hidden * (patch_dim / fmt.block_size());
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words
        // (4-bit `bitstream_words` collapses to the old `n/8`, so no regression).
        let n_weight = hidden * patch_dim;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_weight, DType::U8)
        } else {
            (crate::kernels::quant::format::bitstream_words(n_weight, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("image", in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", hidden, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("patch_h", patch_h as u32)
            .constexpr("patch_w", patch_w as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((hidden / 32) as u32, (num_patches / 32) as u32, 1, [128, 1, 1])
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            // 2 * num_patches * hidden * in_ch * patch_h * patch_w
            // (conv2d formula: N=1, Co=hidden, Ho*Wo=num_patches)
            .flops(
                2 * (num_patches as u64)
                    * (hidden as u64)
                    * (in_ch as u64)
                    * (patch_h as u64)
                    * (patch_w as u64),
            )
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp8_e4m3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxfp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). patch_dim=256 is a multiple of 32 and every block
    // size, and word-aligned for every bit width.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint2_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint2,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint3_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint3,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint4_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint4,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint5_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint5,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint6_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint6,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_mxint8_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Mxint8,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }

    // FP16-scale twins — same shapes / FLOPs as their FP32 siblings; only the
    // scale buffer binds as f16 (`mma_bench` picks DType::F16 off `scale_kind`).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp4F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int2_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int2F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int3_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int3F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int4_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int4F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int5_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int5F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int6_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int6F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_patch_embed_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_int8_f16_patch_embed_mma::kernel_ir_for(dt),
            QFormat::Int8F16,
            4,
            256,
            256,
            8,
            8,
            1024,
            dt,
        )
    }
}

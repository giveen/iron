//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight cooperative (MMA) 3D convolution** — the
//! simdgroup-matrix counterpart of `ffai/conv3d_mma.rs`, with a quantized
//! filter. It is the 3D analogue of `mlx/block_scaled_mma.rs` and the
//! MMA-throughput sibling of `ffai/conv3d_block_scaled.rs`.
//!
//! The dense `conv3d_mma` treats the conv as a GEMM whose A matrix is an
//! implicit im2col over `(kd, kh, kw, ic)` gather indices and whose B matrix
//! is the filter `[out_ch, total_k]` (`total_k = in_ch · kd · kh · kw`).
//! Here only the **B-load (weight staging)** changes: instead of loading a
//! dense `T` filter element, each lane decodes a block-scaled quantized code
//! and multiplies by its per-block scale. The implicit-im2col **A-load**,
//! threadgroup-memory layout, 8×8 frag mapping, MMA inner loop, C-store,
//! grid, and `KernelMode` are **copied verbatim** from `conv3d_mma.rs`.
//!
//! ## Implicit im2col as a matmul (identical to `conv3d_mma`)
//!
//!   out[BN_voxels, BM_oc] = A[BN_voxels, BK] × B[BK, BM_oc]
//!
//! where `BK = in_ch · kd · kh · kw`, `BN = batch · out_d · out_h · out_w`,
//! `BM = out_ch`. Tile geometry: tpg = 128 = 4 SG × 32 lanes, BM = BN = 32,
//! BK = 32, grid `[out_ch/32, (batch·out_d·out_h·out_w)/32, 1]`. Constraints:
//! stride = 1, dilation = 1, padding = 0; `out_ch` and the voxel count both
//! divisible by 32; NCDHW input, OIDHW filter. **No bias** (the dense kernel
//! has none).
//!
//! ## Quantized filter B-load
//!
//! The filter is the 2-D matrix `[out_ch, total_k]`, block-scaled along
//! `total_k`. For output channel `oc` (= `b_oc_row + oc_tile·32`, reused from
//! the dense kernel) and a tap `kt`, the dense filter element is replaced by
//!
//!   element_decode(code[oc, kt]) · scale[oc, kt / block_size]   (× global for nvfp4)
//!
//! with `kt = ((ic·kd + kz)·kh + ky)·kw + kx`. 4-bit codes pack `[out_ch,
//! total_k/8]` u32 (8 nibbles/word — word `oc·(total_k/8) + kt/8`, shift
//! `(kt%8)·4`); 8-bit codes are `[out_ch, total_k]` u8 (byte `oc·total_k +
//! kt`). Decode is per-tap (kt-by-kt), keeping the dense in-bounds masking and
//! the same `bs` store position. `total_k` is a multiple of `block_size`
//! (4-bit `block_size` a multiple of 8) and of the 32-wide MMA K-tile.
//!
//! fp8_e4m3 reuses the nvfp8 kernel (same 8-bit-E4M3 + f32-scale shape).
//! Codegen-only; correctness pinned by the in-source `#[test_kernel]`s vs a
//! `kernels::quant::format::dequant` oracle running the dense conv3d_mma math.

use metaltile::kernel;

/// Quantized-weight conv3d (simdgroup-MMA), folded over the 28-format axis (§7).
/// Same structure as `conv2d_mma`: implicit-im2col A-load + 8×8 simdgroup MMA +
/// write-back are format-independent; only the W-dequant B-load folds onto
/// `(BITS, WDEC, SKIND)` through `kernels/primitives.rs`. Produces
/// `mt_<FMT>_conv3d_mma`.
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
    suffix = "{FMT}_conv3d_mma",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    input: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    out: Tensor<T>,
    #[constexpr] in_ch: u32,
    #[constexpr] in_d: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_d: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kd: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    // BM (oc-axis) tile = tgid_x * 32, BN (voxel-axis) tile = tgid_y * 32.
    let oc_tile = tgid_x;
    let pv_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // ── 8×8 frag lane mapping (Apple steel_gemm layout) ──────────────────
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    // ── TG memory: A and B tiles, skewed stride = 36 ─────────────────────
    let stride = 36u32;
    threadgroup_alloc("as", 1152, T);
    threadgroup_alloc("bs", 1152, T);
    // ── Accumulator frags ─────────────────────────────────────────────────
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
    // ── Precompute K-space extents ────────────────────────────────────────
    let khw = kh * kw;
    let kdhw = kd * khw; // taps per input channel
    let total_k = in_ch * kdhw; // total tap dimension
    // ── Voxel-axis im2col decode for this TG's A rows ────────────────────
    let out_hw = out_h * out_w;
    let out_dhw = out_d * out_hw;
    // Coop A-load lane assignment: lane_in_tg = pv_row * 4 + k_quad.
    let a_pv_row = lane_in_tg / 4u32;
    let a_k_quad = lane_in_tg & 3u32;
    let a_k_base = a_k_quad * 8u32;
    let global_pv = pv_tile * 32u32 + a_pv_row;
    let n_pv = global_pv / out_dhw;
    let rem_pv = global_pv - n_pv * out_dhw;
    let od_pv = rem_pv / out_hw;
    let rem_hw = rem_pv - od_pv * out_hw;
    let oh_pv = rem_hw / out_w;
    let ow_pv = rem_hw - oh_pv * out_w;
    // Base device offset for this voxel's batch + channel-0 position.
    let in_plane = in_h * in_w;
    let in_vol = in_d * in_plane;
    let in_n_stride = in_ch * in_vol;
    let pv_in_base = n_pv * in_n_stride;
    // Coop B-load (quantized weight). Filter as [out_ch, total_k], block-scaled
    // along total_k; 4-bit codes pack 8 nibbles per u32 word.
    let b_oc_row = lane_in_tg / 4u32;
    let b_k_quad = lane_in_tg & 3u32;
    let b_k_base = b_k_quad * 8u32;
    let global_oc = oc_tile * 32u32 + b_oc_row;
    let packs_per_row = total_k / 8u32;
    let n_blocks_per_row = total_k / block_size;
    let w_oc_pack_base = global_oc * packs_per_row;
    let w_oc_blk_base = global_oc * n_blocks_per_row;
    let w_oc_byte_base = global_oc * total_k;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    // ── K-block loop ──────────────────────────────────────────────────────
    // K-tail handling: `total_k = in_ch * kd * kh * kw` rarely lands on a
    // multiple of 32. The A/B coop loads mask out-of-bound K-taps and clamp
    // the gather/decode index to 0 on OOB so we never read past the buffers;
    // zero contributions leave the partial-K MMA accumulator correct.
    for kb in range(0u32, total_k, 32u32) {
        // ─ 1. Coop A load (implicit 5D im2col gather) ───────────────────
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + a_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            // Decompose kt_safe into (ic, kz, ky, kx).
            let ic = kt_safe / kdhw;
            let rem_kt = kt_safe - ic * kdhw;
            let kz = rem_kt / khw;
            let rem_kh = rem_kt - kz * khw;
            let ky = rem_kh / kw;
            let kx = rem_kh - ky * kw;
            // Gather indices (stride=1, pad=0).
            let id = od_pv + kz;
            let ih = oh_pv + ky;
            let iw = ow_pv + kx;
            let in_idx = pv_in_base + ic * in_vol + id * in_plane + ih * in_w + iw;
            let raw = load(input[in_idx]).cast::<f32>();
            let val = select(in_bounds, raw, 0.0f32).cast::<T>();
            threadgroup_store("as", a_pv_row * stride + a_k_base + i, val);
        }
        // ─ 2. Coop B load (W dequant, folded over the format axis) ─
        for i in range(0u32, 8u32, 1u32) {
            let kt = kb + b_k_base + i;
            let in_bounds = kt < total_k;
            let kt_safe = select(in_bounds, kt, 0u32);
            let sraw = load(scales[w_oc_blk_base + kt_safe / block_size]);
            let scale = if SKIND == 0u32 {
                exp2(sraw.cast::<f32>() - 127.0f32)
            } else if SKIND == 1u32 {
                mt_decode_e4m3(sraw.cast::<u32>()) * global
            } else {
                sraw.cast::<f32>()
            };
            let elem = if WDEC == 0u32 {
                let pack = load(weight[w_oc_pack_base + kt_safe / 8u32]);
                mt_decode_e2m1((pack >> ((kt_safe & 7u32) * 4u32)) & 0xFu32)
            } else if WDEC == 1u32 {
                let bit_off = (w_oc_byte_base + kt_safe) * BITS;
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
                let raw = load(weight[w_oc_byte_base + kt_safe]).cast::<u32>();
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
        // ─ 3. MMA inner loop (4 k-inner × 4 frags = 16 MMAs / SG) ──────
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
    // ── 4. Write 4 C frags to global out ─────────────────────────────────
    // out layout: [batch * out_d * out_h * out_w, out_ch].
    let out_pv_base = pv_tile * 32u32 + sm * 16u32;
    let out_oc_base = oc_tile * 32u32 + sn * 16u32;
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f00, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f00, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_pv_base + 8u32 + fm) * out_ch + out_oc_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Bounded zig-zag ramp identical to the dense conv3d_mma helper.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 3D conv oracle, voxel-major output `[n_voxels, out_ch]`.
    /// stride=1, dilation=1, pad=0, no bias. The SAME dense math as
    /// `conv3d_mma.rs::naive_conv3d_mma`, run over the *dequantized* filter
    /// laid out as the 2-D matrix `[out_ch, C]`, `C = in_ch·kd·kh·kw`, with
    /// `col = ((ic·kd + kz)·kh + ky)·kw + kx`. All f32.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv3d_mma(
        input: &[f32],
        weight: &[f32],
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let out_hw = out_h * out_w;
        let out_dhw = out_d * out_hw;
        let n_voxels = batch * out_dhw;
        let in_plane = in_h * in_w;
        let in_vol = in_d * in_plane;
        let contraction = in_ch * kd * kh * kw;
        let mut out = vec![0.0f32; n_voxels * out_ch];
        for n in 0..batch {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let voxel = n * out_dhw + od * out_hw + oh * out_w + ow;
                        for oc in 0..out_ch {
                            let mut acc = 0.0f32;
                            for ic in 0..in_ch {
                                for kz in 0..kd {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let id = od + kz;
                                            let ih = oh + ky;
                                            let iw = ow + kx;
                                            let in_idx = n * in_ch * in_vol
                                                + ic * in_vol
                                                + id * in_plane
                                                + ih * in_w
                                                + iw;
                                            // Dequantized filter is contiguous over
                                            // col = ((ic*kd+kz)*kh+ky)*kw+kx per oc row.
                                            let col = ((ic * kd + kz) * kh + ky) * kw + kx;
                                            let w_idx = oc * contraction + col;
                                            acc += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                            out[voxel * out_ch + oc] = acc;
                        }
                    }
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the `[out_ch, C]` filter via the
    /// shared codec, dequantize for the oracle, and run the dense conv3d_mma
    /// math. Mirrors conv3d_mma.rs's `mma_setup` grid + KernelMode exactly.
    #[allow(clippy::too_many_arguments)]
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> TestSetup {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_voxels = batch * out_d * out_h * out_w;
        assert_eq!(out_ch % 32, 0, "out_ch must be a multiple of 32 for the MMA tile");
        assert_eq!(n_voxels % 32, 0, "n_voxels must be a multiple of 32 for the MMA tile");
        let n_out = n_voxels * out_ch;
        // Contraction C = in_ch*kd*kh*kw — the quantized filter is [out_ch, C].
        let contraction = in_ch * kd * kh * kw;
        let input_f = ramp(batch * in_ch * in_d * in_h * in_w, 13, 2.0);
        // Quantize the [out_ch, C] filter via the shared codec.
        let w_f = ramp(out_ch * contraction, 11, 2.0);
        let p = crate::kernels::quant::format::pack(fmt, &w_f, out_ch, contraction);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, out_ch, contraction);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        // Oracle: dense conv3d_mma over the dequantized filter row [out_ch, C].
        let expected =
            naive_conv3d_mma(&input, &wdq, batch, in_ch, in_d, in_h, in_w, out_ch, kd, kh, kw);
        // 8-bit codes bind as one uchar each; sub-byte codes pack into a u32
        // bit-stream. F32-scaled formats bind raw f32 scales; F16-scaled bind
        // raw f16 scales; E8M0/E4M3 are one byte. Axis-driven so every
        // int/mxint width routes correctly.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (out_ch / 32) as u32,
            (n_voxels / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // One correctness test per QFormat via the shared `mma_setup` helper —
    // mirrors the `*_bench_fmt!` benches instead of 30 hand-written fns.
    // Shape: in_ch=8, 2×2×2 kernel; 5×5×5 volume, out_ch=32 — MMA tile shape.
    macro_rules! conv3d_mma_test_fmt {
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
            fn $fn(dt: DType) -> TestSetup {
                mma_setup($kernel(dt), $fmt, 1, 8, 5, 5, 5, 32, 2, 2, 2, dt)
            }
        };
    }
    conv3d_mma_test_fmt!(test_mxfp4_conv3d_mma, mt_mxfp4_conv3d_mma::kernel_ir_for, QFormat::Mxfp4);
    conv3d_mma_test_fmt!(test_nvfp4_conv3d_mma, mt_nvfp4_conv3d_mma::kernel_ir_for, QFormat::Nvfp4);
    conv3d_mma_test_fmt!(test_fp4_conv3d_mma, mt_fp4_conv3d_mma::kernel_ir_for, QFormat::Fp4);
    conv3d_mma_test_fmt!(
        test_mxfp8_e4m3_conv3d_mma,
        mt_mxfp8_e4m3_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp8E4
    );
    conv3d_mma_test_fmt!(
        test_mxfp8_e5m2_conv3d_mma,
        mt_mxfp8_e5m2_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp8E5
    );
    conv3d_mma_test_fmt!(
        test_fp8_e5m2_conv3d_mma,
        mt_fp8_e5m2_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E5m2
    );
    conv3d_mma_test_fmt!(test_nvfp8_conv3d_mma, mt_nvfp8_conv3d_mma::kernel_ir_for, QFormat::Nvfp8);
    conv3d_mma_test_fmt!(
        test_fp8_e4m3_conv3d_mma,
        mt_nvfp8_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E4m3
    );
    conv3d_mma_test_fmt!(test_int8_conv3d_mma, mt_int8_conv3d_mma::kernel_ir_for, QFormat::Int8);
    conv3d_mma_test_fmt!(test_int2_conv3d_mma, mt_int2_conv3d_mma::kernel_ir_for, QFormat::Int2);
    conv3d_mma_test_fmt!(test_int3_conv3d_mma, mt_int3_conv3d_mma::kernel_ir_for, QFormat::Int3);
    conv3d_mma_test_fmt!(test_int4_conv3d_mma, mt_int4_conv3d_mma::kernel_ir_for, QFormat::Int4);
    conv3d_mma_test_fmt!(test_int5_conv3d_mma, mt_int5_conv3d_mma::kernel_ir_for, QFormat::Int5);
    conv3d_mma_test_fmt!(test_int6_conv3d_mma, mt_int6_conv3d_mma::kernel_ir_for, QFormat::Int6);
    conv3d_mma_test_fmt!(
        test_mxint2_conv3d_mma,
        mt_mxint2_conv3d_mma::kernel_ir_for,
        QFormat::Mxint2
    );
    conv3d_mma_test_fmt!(
        test_mxint3_conv3d_mma,
        mt_mxint3_conv3d_mma::kernel_ir_for,
        QFormat::Mxint3
    );
    conv3d_mma_test_fmt!(
        test_mxint4_conv3d_mma,
        mt_mxint4_conv3d_mma::kernel_ir_for,
        QFormat::Mxint4
    );
    conv3d_mma_test_fmt!(
        test_mxint5_conv3d_mma,
        mt_mxint5_conv3d_mma::kernel_ir_for,
        QFormat::Mxint5
    );
    conv3d_mma_test_fmt!(
        test_mxint6_conv3d_mma,
        mt_mxint6_conv3d_mma::kernel_ir_for,
        QFormat::Mxint6
    );
    conv3d_mma_test_fmt!(
        test_mxint8_conv3d_mma,
        mt_mxint8_conv3d_mma::kernel_ir_for,
        QFormat::Mxint8
    );
    conv3d_mma_test_fmt!(
        test_nvfp8_f16_conv3d_mma,
        mt_nvfp8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Nvfp8F16
    );
    conv3d_mma_test_fmt!(
        test_fp8_e4m3_f16_conv3d_mma,
        mt_nvfp8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    conv3d_mma_test_fmt!(
        test_fp4_f16_conv3d_mma,
        mt_fp4_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp4F16
    );
    conv3d_mma_test_fmt!(
        test_fp8_e5m2_f16_conv3d_mma,
        mt_fp8_e5m2_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    conv3d_mma_test_fmt!(
        test_int2_f16_conv3d_mma,
        mt_int2_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int2F16
    );
    conv3d_mma_test_fmt!(
        test_int3_f16_conv3d_mma,
        mt_int3_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int3F16
    );
    conv3d_mma_test_fmt!(
        test_int4_f16_conv3d_mma,
        mt_int4_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int4F16
    );
    conv3d_mma_test_fmt!(
        test_int5_f16_conv3d_mma,
        mt_int5_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int5F16
    );
    conv3d_mma_test_fmt!(
        test_int6_f16_conv3d_mma,
        mt_int6_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int6F16
    );
    conv3d_mma_test_fmt!(
        test_int8_f16_conv3d_mma,
        mt_int8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Int8F16
    );
}

/// Decode-shape benches: a realistic conv (in_ch=64, out_ch=256, 2×2×2 kernel →
/// C = 512, divisible by all block sizes 16/32/64). Reduction mode,
/// `grid_3d(out_ch/32, n_voxels/32, 1, [128,1,1])` like the dense conv3d_mma.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_d: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_d = in_d - kd + 1;
        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let n_voxels = batch * out_d * out_h * out_w;
        let n_out = n_voxels * out_ch;
        let contraction = in_ch * kd * kh * kw;
        // 8-bit codes are one byte each; sub-byte codes pack into a tight u32
        // bit-stream of `bitstream_words(total_elems, bits)` words. Axis-driven
        // so every int/mxint width sizes its code buffer correctly.
        let total_elems = out_ch * contraction;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (total_elems, DType::U8)
        } else {
            (
                crate::kernels::quant::format::bitstream_words(total_elems, fmt.element_bits()),
                DType::U32,
            )
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = out_ch * (contraction / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + batch * in_ch * in_d * in_h * in_w * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_d * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_d", in_d as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_d", out_d as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kd", kd as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((out_ch / 32) as u32, (n_voxels / 32) as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            // 2 * n_out * C; C = in_ch*kd*kh*kw is the per-output contraction.
            .flops(2 * n_out as u64 * contraction as u64)
            .with_shape_label(format!(
                "{} co={out_ch} do={out_d} ho={out_h} wo={out_w} C={contraction}",
                fmt.name()
            ))
    }

    macro_rules! conv3d_mma_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                // in_ch=64, out_ch=256, 2×2×2 kernel on a 9×9×9 volume →
                // out 8×8×8, n_voxels=512; C=512 (÷ 16/32/64).
                mma_bench($kernel(dt), $fmt, 1, 64, 9, 9, 9, 256, 2, 2, 2, dt)
            }
        };
    }
    conv3d_mma_bench_fmt!(bench_mxfp4, mt_mxfp4_conv3d_mma::kernel_ir_for, QFormat::Mxfp4);
    conv3d_mma_bench_fmt!(bench_nvfp4, mt_nvfp4_conv3d_mma::kernel_ir_for, QFormat::Nvfp4);
    conv3d_mma_bench_fmt!(
        bench_mxfp8_e4m3,
        mt_mxfp8_e4m3_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp8E4
    );
    conv3d_mma_bench_fmt!(
        bench_mxfp8_e5m2,
        mt_mxfp8_e5m2_conv3d_mma::kernel_ir_for,
        QFormat::Mxfp8E5
    );
    conv3d_mma_bench_fmt!(bench_nvfp8, mt_nvfp8_conv3d_mma::kernel_ir_for, QFormat::Nvfp8);
    conv3d_mma_bench_fmt!(bench_fp4, mt_fp4_conv3d_mma::kernel_ir_for, QFormat::Fp4);
    conv3d_mma_bench_fmt!(bench_fp8_e5m2, mt_fp8_e5m2_conv3d_mma::kernel_ir_for, QFormat::Fp8E5m2);
    conv3d_mma_bench_fmt!(bench_int8, mt_int8_conv3d_mma::kernel_ir_for, QFormat::Int8);
    conv3d_mma_bench_fmt!(bench_int2, mt_int2_conv3d_mma::kernel_ir_for, QFormat::Int2);
    conv3d_mma_bench_fmt!(bench_int3, mt_int3_conv3d_mma::kernel_ir_for, QFormat::Int3);
    conv3d_mma_bench_fmt!(bench_int4, mt_int4_conv3d_mma::kernel_ir_for, QFormat::Int4);
    conv3d_mma_bench_fmt!(bench_int5, mt_int5_conv3d_mma::kernel_ir_for, QFormat::Int5);
    conv3d_mma_bench_fmt!(bench_int6, mt_int6_conv3d_mma::kernel_ir_for, QFormat::Int6);
    conv3d_mma_bench_fmt!(bench_mxint2, mt_mxint2_conv3d_mma::kernel_ir_for, QFormat::Mxint2);
    conv3d_mma_bench_fmt!(bench_mxint3, mt_mxint3_conv3d_mma::kernel_ir_for, QFormat::Mxint3);
    conv3d_mma_bench_fmt!(bench_mxint4, mt_mxint4_conv3d_mma::kernel_ir_for, QFormat::Mxint4);
    conv3d_mma_bench_fmt!(bench_mxint5, mt_mxint5_conv3d_mma::kernel_ir_for, QFormat::Mxint5);
    conv3d_mma_bench_fmt!(bench_mxint6, mt_mxint6_conv3d_mma::kernel_ir_for, QFormat::Mxint6);
    conv3d_mma_bench_fmt!(bench_mxint8, mt_mxint8_conv3d_mma::kernel_ir_for, QFormat::Mxint8);
    // ── FP16-scale twins (fp8_e4m3_f16 reuses the nvfp8_f16 kernel) ──
    conv3d_mma_bench_fmt!(
        bench_nvfp8_f16,
        mt_nvfp8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Nvfp8F16
    );
    conv3d_mma_bench_fmt!(
        bench_fp8_e4m3_f16,
        mt_nvfp8_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    conv3d_mma_bench_fmt!(bench_fp4_f16, mt_fp4_f16_conv3d_mma::kernel_ir_for, QFormat::Fp4F16);
    conv3d_mma_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_conv3d_mma::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    conv3d_mma_bench_fmt!(bench_int2_f16, mt_int2_f16_conv3d_mma::kernel_ir_for, QFormat::Int2F16);
    conv3d_mma_bench_fmt!(bench_int3_f16, mt_int3_f16_conv3d_mma::kernel_ir_for, QFormat::Int3F16);
    conv3d_mma_bench_fmt!(bench_int4_f16, mt_int4_f16_conv3d_mma::kernel_ir_for, QFormat::Int4F16);
    conv3d_mma_bench_fmt!(bench_int5_f16, mt_int5_f16_conv3d_mma::kernel_ir_for, QFormat::Int5F16);
    conv3d_mma_bench_fmt!(bench_int6_f16, mt_int6_f16_conv3d_mma::kernel_ir_for, QFormat::Int6F16);
    conv3d_mma_bench_fmt!(bench_int8_f16, mt_int8_f16_conv3d_mma::kernel_ir_for, QFormat::Int8F16);
}

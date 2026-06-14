//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped block-scaled BGEMM (BM=BN=64 tile) — the
//! block-scaled / legacy-float / symmetric-int8 counterpart of
//! `moe_mpp_bm64` (int4) and `moe_mpp_bm64_int8`.
//!
//! Geometry is **byte-identical** to `mt_moe_gather_qmm_mma_int4_bm64_mpp` /
//! `…_int8_bm64_mpp`: BM=BN=64 / BK=32 MPP cooperative-tensor tiling, 4
//! simdgroups in a 2×2 warp grid, the per-TG contiguous-expert sub-run walk,
//! the X staging, the per-SG `coop_tile_*` ops with their 32×32 offsets, and
//! the masked coop-write tail. The **only** change versus the int templates is
//! the W-dequant staging block — it emits `element_decode(code) · block_scale`
//! (no bias) per `mlx/block_scaled_qmm_mpp` instead of the affine `scale·q +
//! bias`.
//!
//! Eight float/int8 kernels cover all nine "legacy" formats (`fp8_e4m3` reuses
//! the `nvfp8` kernel — both are 8-bit E4M3 + f32 per-block scale):
//!
//! | kernel                                  | element | weight | scale       |
//! |-----------------------------------------|---------|--------|-------------|
//! | `mt_mxfp4_moe_gather_qmm_bm64_mpp`      | E2M1    | u32    | E8M0 (u8)   |
//! | `mt_nvfp4_moe_gather_qmm_bm64_mpp`      | E2M1    | u32    | E4M3 (u8) × global |
//! | `mt_fp4_moe_gather_qmm_bm64_mpp`        | E2M1    | u32    | f32         |
//! | `mt_mxfp8_e4m3_moe_gather_qmm_bm64_mpp` | E4M3    | u8     | E8M0 (u8)   |
//! | `mt_mxfp8_e5m2_moe_gather_qmm_bm64_mpp` | E5M2    | u8     | E8M0 (u8)   |
//! | `mt_fp8_e5m2_moe_gather_qmm_bm64_mpp`   | E5M2    | u8     | f32         |
//! | `mt_nvfp8_moe_gather_qmm_bm64_mpp`      | E4M3    | u8     | f32         |
//! | `mt_int8_moe_gather_qmm_bm64_mpp`       | int8    | u8     | f32         |
//!
//! Eleven more kernels cover the symmetric Track-1 integer formats — five
//! FP32-scaled (`mt_int{2,3,4,5,6}_moe_gather_qmm_bm64_mpp`, group 64), five
//! E8M0-scaled (`mt_mxint{2,3,4,5,6}_moe_gather_qmm_bm64_mpp`, block 32), and
//! `mt_mxint8_moe_gather_qmm_bm64_mpp` (8-bit, E8M0, block 32). The 2/3/4/5/6-bit
//! weights are a tight LSB-first two's-complement bit-stream over the whole
//! stacked `[n_experts·n_out, k_in]` matrix (u32 words, one guard word at the
//! very end); `mxint8` is one signed byte per code (u8). Decode mirrors the
//! `mlx/block_scaled_qmm_mpp` `int_qmm_mma_mpp_*` macros (straddle-aware two-word
//! read + float sign-extend) and stages `code · block_scale` into `Ws` exactly
//! like the float kernels — **same geometry, only the W-stage decode differs**.
//!
//! Weight layout per expert (stacked `[n_experts, …]`): 4-bit `w [n_out, k_in/8]
//! u32` (8 E2M1 nibbles/word, LSB-first), 8-bit `w [n_out, k_in] u8` (one code
//! per byte). Scales `[n_experts, n_out, k_in/block_size]` are u8 (E8M0/E4M3) or
//! f32 (nvfp8 / legacy fp / int8). No `biases` param — block-scaled is
//! scale-only.
//!
//! ## 4-bit / 8-bit lane mapping (BM=64, BK=32)
//!
//! W tile size: BN(64) × BK(32) = 2048 elements.
//!
//! - **4-bit**: 128 lanes × 2 packs/lane × 8 nibbles/pack = 2048 ✓
//!   - `pack_id = lane_in_tg*2 + _pi`; `w_row = pack_id/4`; `pack_in_row =
//!     pack_id%4` (BK=32 → 4 u32 packs of 8 nibbles)
//!   - `k_off = kb + pack_in_row*8`; `ws_base = w_row*32 + pack_in_row*8`
//!   - The 8-element K span per pack is 8-aligned, so it sits inside one block
//!     (`block_size ≥ 16`); one scale load per pack via `g = k_off/block_size`.
//!
//! - **8-bit**: 128 lanes × 4 packs/lane × 4 bytes/pack = 2048 ✓. The int8
//!   template packs 4 u8 codes per u32 word; the block-scaled 8-bit weight is a
//!   plain `u8` tensor (1 byte/elem), so each "pack" instead loads 4 contiguous
//!   `u8` bytes directly and stores the decoded values to the same `Ws`
//!   positions the int8 version wrote to.
//!   - `pack_id = lane_in_tg*4 + _pi`; `w_row = pack_id/8`; `pack_in_row =
//!     pack_id%8` (BK=32 → 8 packs of 4 bytes)
//!   - `k_off = kb + pack_in_row*4`; `ws_base = w_row*32 + pack_in_row*4`
//!   - The 4-element K span per pack is 4-aligned, so it sits inside one block
//!     (`block_size ≥ 16`); one scale load per pack via `g = k_off/block_size`.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(32, 32, 32, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — identical to the int4/int8 bm64 MPP descriptor; only
//! the threadgroup W tile contents differ.
//!
//! ## bf16 staging
//!
//! Same `coop_stage(T)` trick as the int templates: bf16 activations stage
//! through `half` so `mpp::tensor_ops::matmul2d` sees a supported
//! cooperative-tensor dtype. Accumulation is fp32.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/64, ceil(m_total/64), 1]`; threadgroup
//!   `[128, 1, 1]` (4 simdgroups, 2×2 warp grid).
//! - `k_in % 32 == 0`, `n_out % 64 == 0`, `block_size` divides `k_in`, and
//!   `block_size ≥ 16` (so the per-pack K span — 8 elems for 4-bit, 4 elems for
//!   8-bit, both pack-aligned — sits inside one block; one scale load per pack
//!   is exact).
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a linkable stub.

use metaltile::kernel;

/// MPP MoE gather BGEMM (BM=BN=64, 4-simdgroup 2×2 path), folded over the
/// 28-format axis (§7). Expert sub-run walk / X-staging / matmul / write-back
/// are format-independent; the W-dequant into the Ws tile folds onto
/// `(BITS, WDEC, SKIND)` with a unified lane layout (8 codes/pack for nibbles, 4 for byte/sub-byte)
/// (nibble 2 packs/lane×8, byte/sub-byte 4 packs/lane×4) + a per-WDEC decode,
/// through `kernels/primitives.rs`. Produces `mt_<FMT>_moe_gather_qmm_bm64_mpp`.
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
    suffix = "{FMT}_moe_gather_qmm_bm64_mpp",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    x: Tensor<T>,
    w: Tensor<WT>,
    scales: Tensor<ST>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let packs_per_row = k_in / 8u32;
    let words_per_row = k_in * BITS / 32u32;
    let groups_per_row = k_in / block_size;
    // W-staging granularity: 8 for E2M1 nibbles (one u32 = 8 codes), 4 for byte
    // floats (one u32 = 4 bytes) and sub-byte bit-streams (fixed 4-elem groups).
    let vals_per_pack = if WDEC == 0u32 { 8u32 } else { 4u32 };
    let packs_in_row = 32u32 / vals_per_pack;
    let packs_per_lane = 16u32 / vals_per_pack;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    threadgroup_alloc("Xs", 2048, coop_stage(T));
    threadgroup_alloc("Ws", 2048, coop_stage(T));
    threadgroup_alloc("OutScratch", 4096, f32);
    coop_tile_setup(
        "gemm",
        32,
        32,
        32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 64u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 64u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 64u32;
        let mut found = 0u32;
        for _ii in range(0u32, 64u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 64u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 64u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 64u32);
        if cur_valid {
            let w_expert_pack = cur_expert * n_out * packs_per_row;
            let w_expert_word = cur_expert * n_out * words_per_row;
            let w_expert_byte = cur_expert * n_out * k_in;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                let gr_x = m_tile_base + x_m_row;
                let in_run_x = (x_m_row >= sub_offset) & (x_m_row < sub_end) & (gr_x < m_total);
                let safe_gr_x = select(in_run_x, gr_x, 0u32);
                let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
                let x_ws_base = x_m_row * 32u32 + x_k_base;
                for _i in range(0u32, 16u32, 1u32) {
                    let xv = load(x[x_dev_base + _i]).cast::<f32>();
                    threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
                }
                // W-dequant → Ws, folded over the format axis (unified vals_per_pack layout).
                for _pi in range(0u32, packs_per_lane, 1u32) {
                    let pack_id = lane_in_tg * packs_per_lane + _pi;
                    let w_row = pack_id / packs_in_row;
                    let pack_in_row = pack_id % packs_in_row;
                    let g_row = n_tile_base + w_row;
                    let k_off = kb + pack_in_row * vals_per_pack;
                    let sb_off = sb_expert_base + g_row * groups_per_row + k_off / block_size;
                    let sraw = load(scales[sb_off]);
                    let scale = if SKIND == 0u32 {
                        exp2(sraw.cast::<f32>() - 127.0f32)
                    } else if SKIND == 1u32 {
                        mt_decode_e4m3(sraw.cast::<u32>()) * global
                    } else {
                        sraw.cast::<f32>()
                    };
                    let ws_base = w_row * 32u32 + pack_in_row * vals_per_pack;
                    if WDEC == 0u32 {
                        let packed = load(
                            w[w_expert_pack + g_row * packs_per_row + kb / 8u32 + pack_in_row],
                        );
                        for _j in range(0u32, vals_per_pack, 1u32) {
                            let nib = (packed >> (_j * 4u32)) & 15u32;
                            threadgroup_store("Ws", ws_base + _j, mt_decode_e2m1(nib) * scale);
                        }
                    } else if WDEC == 1u32 {
                        let wwb = w_expert_word + g_row * words_per_row;
                        for _j in range(0u32, vals_per_pack, 1u32) {
                            let bit_off = (k_off + _j) * BITS;
                            let word_idx = bit_off / 32u32;
                            let bit_in_w = bit_off & 31u32;
                            let bits_in_w0 = 32u32 - bit_in_w;
                            let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                            let spill = BITS - lo_bits;
                            let w0 = load(w[wwb + word_idx]);
                            let w1 = load(w[wwb + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                            let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                            let qf = q.cast::<f32>();
                            let elem = select(q >= half, qf - full, qf);
                            threadgroup_store("Ws", ws_base + _j, elem * scale);
                        }
                    } else {
                        let w_dev = w_expert_byte + g_row * k_in + k_off;
                        for _j in range(0u32, vals_per_pack, 1u32) {
                            let raw = load(w[w_dev + _j]).cast::<u32>();
                            let elem = if WDEC == 2u32 {
                                mt_decode_e4m3(raw)
                            } else if WDEC == 3u32 {
                                mt_decode_e5m2(raw)
                            } else {
                                mt_decode_int8(raw)
                            };
                            threadgroup_store("Ws", ws_base + _j, elem * scale);
                        }
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
                coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
            threadgroup_barrier();
            for _e in range(0u32, 32u32, 1u32) {
                let flat = lane_in_tg * 32u32 + _e;
                let mr = flat / 64u32;
                let nc = flat & 63u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
                    let v = threadgroup_load(
                        "OutScratch",
                        src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
                    );
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Test shape for a block-scaled MoE variant (all clean multiples).
    struct BlockTestShape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    /// Build a `TestSetup` for a block-scaled indexed-MoE-MPP kernel. Mirrors
    /// `int8_indexed_setup`: per-row expert routing, dtype-rounded x, oracle =
    /// `Σ_k x[t,k] · dequant(W)[expert·n_out + nc, k]`. The whole stacked
    /// `[n_experts·n_out, k_in]` weight is packed in ONE call via
    /// `quant::format::pack` (single contiguous bit-stream with one trailing
    /// guard word — never per-expert concat, which would misalign sub-byte
    /// widths) and the codes/scales are bound directly. No biases buffer; scale
    /// dtype + weight dtype are axis-driven off the format (`element_bits` /
    /// `scale_kind`). `block_size` and the nvfp4 `global` constexpr come from the
    /// packed tensor.
    #[allow(clippy::too_many_arguments)]
    fn block_indexed_setup(
        kernel: Kernel,
        fmt: QFormat,
        shape: BlockTestShape,
        dt: DType,
    ) -> TestSetup {
        let BlockTestShape { n_experts, m_total, n_out, k_in } = shape;
        let block_size = fmt.block_size();

        // Per-row expert indices, sorted (post-permute layout).
        let indices: Vec<u32> = (0..m_total).map(|r| (r / (m_total / n_experts)) as u32).collect();

        // Build the FULL stacked `[n_experts·n_out, k_in]` weight matrix (all
        // experts stacked along rows) and pack it in ONE call — never per-expert
        // packing + byte concatenation. For sub-byte widths (3/5/6-bit) `pack`
        // appends a single guard word at the very end of the contiguous
        // bit-stream; concatenating per-expert buffers would instead inject a
        // guard word mid-stream and misalign every expert after the first. One
        // stacked pack is byte-identical to the old per-expert concat for the
        // 4-bit/8-bit formats (those widths divide 32 ⇒ exact word count, no
        // guard word) and correct for every sub-byte width. `k_in` is a multiple
        // of 32, so each row's bit-stream is word-aligned for every width. The
        // kernels index global W row `g_row = expert·n_out + n`, matching this
        // stacked layout exactly. Magnitude pattern mirrors the non-MoE test.
        let stack_rows = n_experts * n_out;
        let stacked: Vec<f32> = (0..stack_rows * k_in)
            .map(|i| {
                let g = i / k_in; // global row = expert·n_out + n
                let e = (g / n_out) as f32;
                let r = (g % n_out) as f32;
                let c = (i % k_in) as f32;
                let mag = (0.4 + ((r + e) % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15);
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::quant::format::pack(fmt, &stacked, stack_rows, k_in);
        let global = p.global;
        // Dequant the whole stack once; the oracle slices per expert below.
        let wdq = crate::quant::format::dequant(fmt, &p, stack_rows, k_in);

        // Activations: dtype-rounded so the GPU sees exactly the oracle's x.
        let x_f: Vec<f32> = (0..m_total * k_in).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);

        // Oracle: out[t, nc] = Σ_k x[t, k] · dequant(W)[expert(t)·n_out + nc, k].
        let mut expected = vec![0.0f32; m_total * n_out];
        for t in 0..m_total {
            let expert = indices[t] as usize;
            let base = expert * n_out;
            for nc in 0..n_out {
                let mut acc = 0.0f32;
                for kk in 0..k_in {
                    acc += x[t * k_in + kk] * wdq[(base + nc) * k_in + kk];
                }
                expected[t * n_out + nc] = acc;
            }
        }

        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };

        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            n_out as u32 / 64,
            (m_total as u32).div_ceil(64),
            1,
            [128, 1, 1],
        )
    }

    // n_experts=4, m_total=64, n_out=64, k_in=64 (divisible by 16/32/64).
    const SHAPE: BlockTestShape = BlockTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp4_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxfp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Nvfp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp4_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e4m3_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp8_e4m3_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mxfp8_e5m2_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxfp8_e5m2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e5m2_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp8_e5m2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp8_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8,
            SHAPE,
            dt,
        )
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_fp8_e4m3_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_int8_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int8,
            SHAPE,
            dt,
        )
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). k_in=64 is a multiple of 32, so
    // `k_in*bits % 32 == 0` for every width and each stacked weight row's
    // bit-stream is word-aligned. The whole `[n_experts·n_out, k_in]` stack is
    // packed once, so the bit-stream stays contiguous (one guard word at the very
    // end) and the kernel/oracle share the codec.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int3_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int5_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int6_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int6,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint2_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint2,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint3_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint3_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint3,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint4_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint4,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint5_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint5_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint5,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint6_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint6_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint6,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mxint8_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_mxint8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint8,
            SHAPE,
            dt,
        )
    }

    // FP16-scale twins: same element packing as their FP32 twin, scale axis is
    // native `half`. `fp8_e4m3_f16` reuses the `nvfp8_f16` kernel (8-bit E4M3 +
    // f16 scale).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_nvfp8_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            SHAPE,
            dt,
        )
    }
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e4m3_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_nvfp8_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp4_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp4_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp4F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_fp8_e5m2_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_fp8_e5m2_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int2_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int2_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int2F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int3_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int3_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int3F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int4_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int4_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int4F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int5_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int5_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int5F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int6_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int6_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int6F16,
            SHAPE,
            dt,
        )
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_int8_f16_moe_gather_qmm_bm64_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            mt_int8_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int8F16,
            SHAPE,
            dt,
        )
    }
}

/// New-syntax benchmarks for the MPP block-scaled MoE BGEMM (BM=BN=64). Random
/// buffers; `flops = 2·m_total·n_out·k_in` (the gather does a full matmul per
/// row's expert — dense-equivalent FLOPs).
///
/// Grid (Reduction, 4 simdgroups per TG): `grid_3d(n_out/64, m_total.div_ceil(64), 1, [128,1,1])`.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    struct BlockBenchShape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    fn block_bench(kernel: Kernel, fmt: QFormat, shape: BlockBenchShape, dt: DType) -> BenchSetup {
        let BlockBenchShape { n_experts, m_total, n_out, k_in } = shape;
        let block_size = fmt.block_size();
        let groups_per_row = k_in / block_size;
        // Codes: the whole `[n_experts·n_out, k_in]` stack is one contiguous
        // bit-stream (single pack), so its code length is `bitstream_words` over
        // the *total* element count (one guard word for the whole stack). 8-bit
        // codes are one uchar each; every sub-byte width (4-bit nibble packs +
        // int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words. Both axes
        // are driven off the format so new integer formats pick up the right
        // buffer types.
        let stack_n = n_experts * n_out * k_in;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (stack_n, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(stack_n, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = n_experts * n_out * groups_per_row;
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m_total * k_in * sz
            + m_total * n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("indices", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(n_out as u32 / 64, (m_total as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            // MoE gather_qmm indexed: 2 * m_total * n_out * k_in (dense-equivalent).
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
            .with_shape_label(format!(
                "{} M{m_total} N{n_out} K{k_in} E{n_experts} {}",
                fmt.name(),
                crate::utils::dtype_label(dt)
            ))
    }

    // n_experts=8, m_total=512, n_out=4096, k_in=4096.
    const SHAPE: BlockBenchShape =
        BlockBenchShape { n_experts: 8, m_total: 512, n_out: 4096, k_in: 4096 };

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4(dt: DType) -> BenchSetup {
        block_bench(mt_mxfp4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Mxfp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Nvfp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4(dt: DType) -> BenchSetup {
        block_bench(mt_fp4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Fp4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxfp8_e4m3_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E4,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxfp8_e5m2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxfp8E5,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp8_e5m2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8(dt: DType) -> BenchSetup {
        block_bench(mt_nvfp8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Nvfp8, SHAPE, dt)
    }
    // fp8_e4m3 reuses the nvfp8 kernel (8-bit E4M3 + f32 scale, block 32).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        block_bench(mt_int8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Int8, SHAPE, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2(dt: DType) -> BenchSetup {
        block_bench(mt_int2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Int2, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3(dt: DType) -> BenchSetup {
        block_bench(mt_int3_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Int3, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4(dt: DType) -> BenchSetup {
        block_bench(mt_int4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Int4, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5(dt: DType) -> BenchSetup {
        block_bench(mt_int5_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Int5, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6(dt: DType) -> BenchSetup {
        block_bench(mt_int6_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt), QFormat::Int6, SHAPE, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint2_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint2,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint3_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint3,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint4_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint4,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint5_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint5,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint6_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint6,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8(dt: DType) -> BenchSetup {
        block_bench(
            mt_mxint8_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Mxint8,
            SHAPE,
            dt,
        )
    }

    // FP16-scale twins (same element packing as their FP32 twin, half scale).
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (8-bit E4M3 + f16 scale, block 32).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Nvfp8F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_nvfp8_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp4_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp4F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_fp8_e5m2_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int2_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int2F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int3_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int3F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int4_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int4F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int5_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int5F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int6_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int6F16,
            SHAPE,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16(dt: DType) -> BenchSetup {
        block_bench(
            mt_int8_f16_moe_gather_qmm_bm64_mpp::kernel_ir_for(dt),
            QFormat::Int8F16,
            SHAPE,
            dt,
        )
    }
}

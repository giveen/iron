//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled simdgroup-matrix (MMA) dequantizing GEMM — the M ≥ 32
//! ALU-throughput path for the spec-conformant formats. This is a direct
//! adaptation of `mlx/quantized.rs::mt_qmm_mma` (the int4 affine MMA): the
//! **dispatch geometry, threadgroup-memory layout, 8×8 frag mapping, and MMA
//! inner loop are copied verbatim** — only the per-pack weight *dequant*
//! staging changes (E2M1 codebook × E8M0 pow-2 scale instead of int4 affine).
//! Reusing the proven geometry keeps it off the reduction freeze-hazard surface.
//!
//! ## DISPATCH INVARIANTS (identical to `mt_qmm_mma`)
//!
//! - **Mode: Reduction**, `grid = [n/32, m/32, 1]`, `tpg = [128, 1, 1]`
//!   (4 simdgroups × 32 lanes, WM=WN=2). `m`, `n`, `k` all multiples of 32.
//! - BM = BN = BK = 32, output tile 32×32. TG memory `xs`/`ws` are `32×36`
//!   (skew 4 to break bank conflicts; 36 is correct for every dtype).
//! - weight `[n, k/8]` u32 (8 E2M1 nibbles/word); scales `[n, k/block_size]` u8
//!   (E8M0); `block_size` a multiple of 8. x `[m, k]`, out `[m, n]`, row-major.

use ffai_kernels::kernel;

/// Block-scaled simdgroup-MMA dequantizing matmul, folded over the 28-format
/// axis (§7). `out = X · dequant(W)` via 8×8 simdgroup_matmul frags (2×2 SG, 4
/// frags). Only the per-format W-dequant into the `ws` tile folds onto
/// `(BITS, WDEC, SKIND)` (buffer types `(WT, ST)`, legend in
/// `block_scaled_matmul`); X-load, the MMA loop, and write-back are
/// format-independent. Decodes through `kernels/primitives.rs`. Produces
/// `mt_<FMT>_qmm_mma`.
#[kernel(variants(
    (FMT,          BITS,  WT,  ST,  WDEC, SKIND) = [
        (mxfp4,        4u32, u32, u8,  0u32, 0u32),
        (nvfp4,        4u32, u32, u8,  0u32, 1u32),
        (fp4,          4u32, u32, f32, 0u32, 2u32),
        (fp4_float,    4u32, u32, f32, 0u32, 2u32),
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
    suffix = "{FMT}_qmm_mma",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    w: Tensor<WT>,
    scales: Tensor<ST>,
    x: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let n_tile = tgid_x;
    let m_tile = tgid_y;
    let lane = simd_lane;
    let sg = simd_group_id();
    let sm = sg / 2u32;
    let sn = sg & 1u32;
    let lane_in_tg = sg * 32u32 + lane;
    // 8×8 frag lane mapping (Apple steel_gemm layout).
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    threadgroup_alloc("xs", 1152, T);
    threadgroup_alloc("ws", 1152, T);
    // 4 output frags per SG, init to 0.
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
    let w_row = lane_in_tg / 4u32;
    let pack_in_row = lane_in_tg & 3u32;
    let x_m_base = m_tile * 32u32;
    let w_n_base = n_tile * 32u32;
    let packs_per_row = k / 8u32;
    let n_blocks_per_row = k / block_size;
    // Per-lane scale row base (E8M0, one byte per block). Fixed across K-blocks.
    let sb_base = (w_n_base + w_row) * n_blocks_per_row;
    let w_pack_row_base = (w_n_base + w_row) * packs_per_row;
    let xs_ld = 36u32;
    let ws_ld = 36u32;
    // Coop X-load mapping: lane → (m_row, k_quad) reading 8 contiguous K.
    let x_m_row = lane_in_tg / 4u32;
    let x_k_quad = lane_in_tg & 3u32;
    let x_k_base = x_k_quad * 8u32;
    // Per-WDEC weight row bases (only the matching one is used; others DCE).
    let w_word_row_base = (w_n_base + w_row) * (k * BITS / 32u32);
    let w_row_base = (w_n_base + w_row) * k;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    for kb in range(0u32, k, 32u32) {
        // ── 1. Coop X load — 128 lanes × 8 contiguous K elems per lane ──
        let x_row_dev_base = (x_m_base + x_m_row) * k + kb + x_k_base;
        let x_ws_base = x_m_row * xs_ld + x_k_base;
        let xv0 = load(x[x_row_dev_base]).cast::<T>();
        let xv1 = load(x[x_row_dev_base + 1u32]).cast::<T>();
        let xv2 = load(x[x_row_dev_base + 2u32]).cast::<T>();
        let xv3 = load(x[x_row_dev_base + 3u32]).cast::<T>();
        let xv4 = load(x[x_row_dev_base + 4u32]).cast::<T>();
        let xv5 = load(x[x_row_dev_base + 5u32]).cast::<T>();
        let xv6 = load(x[x_row_dev_base + 6u32]).cast::<T>();
        let xv7 = load(x[x_row_dev_base + 7u32]).cast::<T>();
        threadgroup_store("xs", x_ws_base, xv0);
        threadgroup_store("xs", x_ws_base + 1u32, xv1);
        threadgroup_store("xs", x_ws_base + 2u32, xv2);
        threadgroup_store("xs", x_ws_base + 3u32, xv3);
        threadgroup_store("xs", x_ws_base + 4u32, xv4);
        threadgroup_store("xs", x_ws_base + 5u32, xv5);
        threadgroup_store("xs", x_ws_base + 6u32, xv6);
        threadgroup_store("xs", x_ws_base + 7u32, xv7);
        // ── 2. Coop W dequant — folded over (BITS, WDEC, SKIND) ──
        let k_off = kb + x_k_base;
        let sraw = load(scales[sb_base + k_off / block_size]);
        let scale = if SKIND == 0u32 {
            exp2(sraw.cast::<f32>() - 127.0f32)
        } else if SKIND == 1u32 {
            mt_decode_e4m3(sraw.cast::<u32>()) * global
        } else {
            sraw.cast::<f32>()
        };
        let ws_base = w_row * ws_ld + x_k_base;
        if WDEC == 0u32 {
            let pack = load(w[w_pack_row_base + kb / 8u32 + pack_in_row]);
            for i in range(0u32, 8u32, 1u32) {
                let nib = (pack >> (i * 4u32)) & 0xFu32;
                threadgroup_store("ws", ws_base + i, (mt_decode_e2m1(nib) * scale).cast::<T>());
            }
        } else if WDEC == 1u32 {
            for i in range(0u32, 8u32, 1u32) {
                let bit_off = (k_off + i) * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                let spill = BITS - lo_bits;
                let w0 = load(w[w_word_row_base + word_idx]);
                let w1 = load(w[w_word_row_base + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                let qf = q.cast::<f32>();
                let elem = select(q >= half, qf - full, qf);
                threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
            }
        } else {
            let w_dev_base = w_row_base + k_off;
            for i in range(0u32, 8u32, 1u32) {
                let raw = load(w[w_dev_base + i]).cast::<u32>();
                let elem = if WDEC == 2u32 {
                    mt_decode_e4m3(raw)
                } else if WDEC == 3u32 {
                    mt_decode_e5m2(raw)
                } else {
                    mt_decode_int8(raw)
                };
                threadgroup_store("ws", ws_base + i, (elem * scale).cast::<T>());
            }
        }
        threadgroup_barrier();
        // ── 3. MMA inner loop — 4 frags × 4 k-inner = 16 MMAs per SG ──
        let row_a0 = sm * 16u32 + fm;
        let row_a1 = sm * 16u32 + 8u32 + fm;
        let col_b0 = sn * 16u32;
        let col_b1 = sn * 16u32 + 8u32;
        // k_inner = 0
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 1
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 8u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 8u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 8u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 8u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 2
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 16u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 16u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 16u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 16u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        // k_inner = 3
        simdgroup_elem_store(a_f0, 0, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f0, 1, threadgroup_load("xs", row_a0 * xs_ld + 24u32 + fn1));
        simdgroup_elem_store(a_f1, 0, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn0));
        simdgroup_elem_store(a_f1, 1, threadgroup_load("xs", row_a1 * xs_ld + 24u32 + fn1));
        simdgroup_barrier_mem_none();
        simdgroup_elem_store(b_f0, 0, threadgroup_load("ws", (col_b0 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f0, 1, threadgroup_load("ws", (col_b0 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 0, threadgroup_load("ws", (col_b1 + fn0) * ws_ld + 24u32 + fm));
        simdgroup_elem_store(b_f1, 1, threadgroup_load("ws", (col_b1 + fn1) * ws_ld + 24u32 + fm));
        simdgroup_barrier_mem_none();
        simdgroup_matmul(a_f0, b_f0, c_f00);
        simdgroup_matmul(a_f0, b_f1, c_f01);
        simdgroup_matmul(a_f1, b_f1, c_f11);
        simdgroup_matmul(a_f1, b_f0, c_f10);
        simdgroup_barrier_mem_none();
        threadgroup_barrier();
    }
    // ── 4. Write 4 C frags to global out ──
    let out_m_base = m_tile * 32u32 + sm * 16u32;
    let out_n_base = n_tile * 32u32 + sn * 16u32;
    store(out[(out_m_base + fm) * n + out_n_base + fn0], simdgroup_elem_load(c_f00, 0).cast::<T>());
    store(out[(out_m_base + fm) * n + out_n_base + fn1], simdgroup_elem_load(c_f00, 1).cast::<T>());
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f01, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f01, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn0],
        simdgroup_elem_load(c_f10, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + fn1],
        simdgroup_elem_load(c_f10, 1).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn0],
        simdgroup_elem_load(c_f11, 0).cast::<T>(),
    );
    store(
        out[(out_m_base + 8u32 + fm) * n + out_n_base + 8u32 + fn1],
        simdgroup_elem_load(c_f11, 1).cast::<T>(),
    );
}
pub mod kernel_tests {
    use ffai_kernels::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Deterministic `[n, k]` weights (mixed signs, per-block magnitude).
    fn weights(n: usize, k: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| {
                let r = (i / k) as f32;
                let c = (i % k) as f32;
                let mag = (0.4 + (r % 7.0) * 0.15) * (0.1 + (c % 13.0) * 0.2);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// `out[m, n] = Σ_k dequant(W)[n, k] · x[m, k]`.
    fn qmm_oracle(wdq: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for mr in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += wdq[nn * k + kk] * x[mr * k + kk];
                }
                out[mr * n + nn] = acc;
            }
        }
        out
    }

    /// m, n multiples of 32; k a multiple of 32 (and of block_size).
    fn mma_setup(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(n, k);
        let p = crate::kernels::quant::format::pack(fmt, &w, n, k);
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, n, k);
        let x_f: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = qmm_oracle(&wdq, &x, m, k, n);
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
            .input(TestBuffer::from_vec("w", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::zeros("out", m * n, dt))
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            (n / 32) as u32,
            (m / 32) as u32,
            1,
            [128, 1, 1],
        )
    }

    // 32×32 output tile, K=64 (2 K-blocks).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp4_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp4_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_nvfp4_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp4_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp8_e4m3_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp8_e4m3_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxfp8_e5m2_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxfp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E5, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_nvfp8_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8, 32, 32, 64, dt)
    }

    // Legacy float-scale fp4 / fp8 + symmetric int8. fp8_e4m3 reuses the nvfp8
    // kernel (same 8-bit-E4M3 + f32-scale shape); the others decode in their own
    // kernels. int8 has block_size=64, so K=64 is exactly one K-block (and a
    // multiple of the 32 MMA tile) — valid for every variant.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp4_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_float_qmm_mma::kernel_ir_for(dt), QFormat::Fp4, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e4m3_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E4m3, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e5m2_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E5m2, 32, 32, 64, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int8_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int8_qmm_mma::kernel_ir_for(dt), QFormat::Int8, 32, 32, 64, dt)
    }

    // Symmetric sub-byte ints (FP32 group scale, group 64) + MXINT (E8M0 block
    // scale, block 32) + MXINT8 (8-bit, E8M0). K=64 is a multiple of 32 (the MMA
    // tile) and of both block sizes, and `K*bits % 32 == 0` for every width, so
    // each weight row's tight bit-stream is word-aligned. The kernel and oracle
    // share the codec, so the GPU output tracks the dequant-then-matmul reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int2_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int2_qmm_mma::kernel_ir_for(dt), QFormat::Int2, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int3_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int3_qmm_mma::kernel_ir_for(dt), QFormat::Int3, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int4_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int4_qmm_mma::kernel_ir_for(dt), QFormat::Int4, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int5_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int5_qmm_mma::kernel_ir_for(dt), QFormat::Int5, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int6_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int6_qmm_mma::kernel_ir_for(dt), QFormat::Int6, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint2_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint2_qmm_mma::kernel_ir_for(dt), QFormat::Mxint2, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint3_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint3_qmm_mma::kernel_ir_for(dt), QFormat::Mxint3, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint4_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint4_qmm_mma::kernel_ir_for(dt), QFormat::Mxint4, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint5_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint5_qmm_mma::kernel_ir_for(dt), QFormat::Mxint5, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint6_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint6_qmm_mma::kernel_ir_for(dt), QFormat::Mxint6, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_mxint8_mma(dt: DType) -> TestSetup {
        mma_setup(mt_mxint8_qmm_mma::kernel_ir_for(dt), QFormat::Mxint8, 32, 32, 64, dt)
    }

    // FP16-scale twins. Same dims as their FP32 twins (K=64: a multiple of 32 and
    // of every block size; `K*bits % 32 == 0` for every int width, so each weight
    // row's tight bit-stream stays word-aligned). `fp8_e4m3_f16` reuses the
    // `nvfp8_f16` kernel (same 8-bit-E4M3 + scale shape). Tolerances match the
    // FP32 twins — the half-precision scale rounds at pack time, but the kernel
    // and oracle share the codec so the GPU output tracks the dequant reference.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_nvfp8_f16_qmm_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e4m3_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E4m3F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp4_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp4F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_fp8_e5m2_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_fp8_e5m2_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E5m2F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int2_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int2_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int2F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int3_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int3_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int3F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int4_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int4F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int5_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int5_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int5F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int6_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int6_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int6F16, 32, 32, 64, dt)
    }
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 4e-1])]
    fn test_int8_f16_mma(dt: DType) -> TestSetup {
        mma_setup(mt_int8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int8F16, 32, 32, 64, dt)
    }
}

/// Prefill GEMM (m=n=k=4096) benches — the M≥32 simdgroup-matrix throughput
/// path, where GFLOP/s + %FLOP rank the precisions (and the M5 NA story lives).
/// Random packed buffers (throughput is data-independent).
pub mod kernel_benches {
    use ffai_kernels::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn mma_bench(
        kernel: Kernel,
        fmt: QFormat,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> BenchSetup {
        let n_blocks = n * (k / fmt.block_size());
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n * k, DType::U8)
        } else {
            (crate::kernels::quant::format::bitstream_words(n * k, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + (m * k + m * n) * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("w", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("x", m * k, dt))
            .buffer(BenchBuffer::zeros("out", m * n, dt).output())
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d((n / 32) as u32, (m / 32) as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m as u64 * n as u64 * k as u64) // GEMM: 2·M·N·K
            .with_shape_label(format!("{} m={m} n={n} k={k}", fmt.name()))
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxfp4_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp4, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp4_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp4, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e4m3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxfp8_e4m3_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E4, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxfp8_e5m2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxfp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Mxfp8E5, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp4_float_qmm_mma::kernel_ir_for(dt), QFormat::Fp4, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp8_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E4m3, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp8_e5m2_qmm_mma::kernel_ir_for(dt), QFormat::Fp8E5m2, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int8_qmm_mma::kernel_ir_for(dt), QFormat::Int8, 4096, 4096, 4096, dt)
    }
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). K=4096 is a multiple of 32 and every block size.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int2_qmm_mma::kernel_ir_for(dt), QFormat::Int2, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int3_qmm_mma::kernel_ir_for(dt), QFormat::Int3, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int4_qmm_mma::kernel_ir_for(dt), QFormat::Int4, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int5_qmm_mma::kernel_ir_for(dt), QFormat::Int5, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int6_qmm_mma::kernel_ir_for(dt), QFormat::Int6, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint2_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint2_qmm_mma::kernel_ir_for(dt), QFormat::Mxint2, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint3_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint3_qmm_mma::kernel_ir_for(dt), QFormat::Mxint3, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint4_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint4_qmm_mma::kernel_ir_for(dt), QFormat::Mxint4, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint5_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint5_qmm_mma::kernel_ir_for(dt), QFormat::Mxint5, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint6_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint6_qmm_mma::kernel_ir_for(dt), QFormat::Mxint6, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mxint8_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_mxint8_qmm_mma::kernel_ir_for(dt), QFormat::Mxint8, 4096, 4096, 4096, dt)
    }
    // FP16-scale twins. K=4096 is a multiple of 32 and every block size.
    // fp8_e4m3_f16 reuses the nvfp8_f16 kernel (same 8-bit-E4M3 + scale shape).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp8_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Nvfp8F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e4m3_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_nvfp8_f16_qmm_mma::kernel_ir_for(dt),
            QFormat::Fp8E4m3F16,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp4_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_fp4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Fp4F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_fp8_e5m2_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(
            mt_fp8_e5m2_f16_qmm_mma::kernel_ir_for(dt),
            QFormat::Fp8E5m2F16,
            4096,
            4096,
            4096,
            dt,
        )
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int2_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int2_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int2F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int3_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int3_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int3F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int4_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int4_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int4F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int5_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int5_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int5F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int6_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int6_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int6F16, 4096, 4096, 4096, dt)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_int8_f16_mma(dt: DType) -> BenchSetup {
        mma_bench(mt_int8_f16_qmm_mma::kernel_ir_for(dt), QFormat::Int8F16, 4096, 4096, 4096, dt)
    }
}

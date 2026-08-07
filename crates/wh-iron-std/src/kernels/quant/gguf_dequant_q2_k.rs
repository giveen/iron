//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GGUF Q2_K block dequant — k-quant 2-bit-per-weight with two-level scales.
//!
//! Follows the canonical `dequantize_row_q2_K` reference algorithm.
//!
//! ## On-disk block layout (decomposed CPU-side at load time)
//!
//! ```text
//!   struct block_q2_K {
//!     uint8_t  scales[16];   // 16 bytes — low 4 bits = scale, high 4 bits = min
//!     uint8_t  qs[64];       // 64 bytes — 2-bit-packed quants, 4 vals per byte
//!     uint16_t d;            //  2 bytes — fp16 super-scale for scales
//!     uint16_t dmin;         //  2 bytes — fp16 super-scale for mins
//!   };                       // 84 bytes per 256 values (BPW = 2.625)
//! ```
//!
//! Per output value `i ∈ [0, 256)`:
//!
//! ```text
//!   sub        = i / 16           // 0..15, picks the (4-bit scale, 4-bit min) pair
//!   in_sub     = i & 15            // 0..15 inside the sub-block
//!   scale_byte = scales[sub]
//!   scale_4bit = scale_byte & 0xf
//!   min_4bit   = (scale_byte >> 4) & 0xf
//!   qs_byte    = qs[i / 4]
//!   shift      = (i & 3) * 2
//!   q_2bit     = (qs_byte >> shift) & 0x3
//!   out[i]     = d * scale_4bit * q_2bit - dmin * min_4bit
//! ```
//!
//! ## GPU-resident split (the loader produces these from the packed block)
//!
//! 1. `qs_packed [n_blocks * 16]`   — `u32`, the 64 packed-quant bytes
//!    per block re-laid as 16 u32 words. `qs_packed[block*16 + j]`
//!    carries 16 two-bit quants in the lower / upper bytes of each
//!    u32. Output index `i ∈ [0, 256)` → `u32 j = i / 16`, then a
//!    `(i % 16) * 2`-bit shift on the byte that holds it.
//! 2. `scales    [n_blocks * 16]`   — `u8`, the raw scale/min byte
//!    pairs (low nibble = scale, high nibble = min) — kept packed
//!    because both nibbles are used per dequant.
//! 3. `d_f32     [n_blocks]`        — `f32`, host-converted from fp16.
//! 4. `dmin_f32  [n_blocks]`        — `f32`, host-converted from fp16.
//!
//! ## Dispatch
//!
//! 1D grid: one thread per *output value*. ~6 reads (1 qs_packed + 1
//! scales + 1 each of d_f32 / dmin_f32, scales cache-multicast across
//! 16 lanes that share a sub-block) and ~4 arithmetic ops per output —
//! cleanly bandwidth-bound on Apple9.

use wh_iron::kernel;

// Bare `#[kernel]` — see Q8_0 sibling for why; mixed concrete +
// generic param dtype set doesn't fit the legacy `bench(...)` shape.
#[kernel]
pub fn iron_gguf_dequant_q2_k<T>(
    qs_packed: Tensor<u32>,
    scales: Tensor<u8>,
    d_f32: Tensor<f32>,
    dmin_f32: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] n_values: u32,
) {
    let i = tid;
    if i < n_values {
        let block = i / 256u32;
        let in_block = i - block * 256u32;
        // Canonical Q2_K block layout: the 256
        // values are NOT 4-consecutive-per-byte. They split as 2 halves of
        // 128; each half is 4 j-groups of 32; each j-group is two runs of 16
        // values that index 16 CONSECUTIVE qs bytes at a SHARED 2-bit shift
        // (shift = j*2). The naive in_block/4 mapping was wrong.
        let half = in_block / 128u32; // 0..1  → qs byte base half*32
        let yh = in_block - half * 128u32; // 0..127
        let jg = yh / 32u32; // 0..3 → shift = jg*2
        let yg = yh - jg * 32u32; // 0..31
        let sub_half = yg / 16u32; // 0..1
        let l = yg - sub_half * 16u32; // 0..15 → byte within the 16-run
        let shift = jg * 2u32;
        let q_byte = half * 32u32 + sub_half * 16u32 + l; // 0..63
        let scale_idx = half * 8u32 + jg * 2u32 + sub_half; // 0..15
        let word_idx = q_byte / 4u32;
        let byte_in_word = q_byte & 3u32;
        let word = load(qs_packed[block * 16u32 + word_idx]);
        let qs_byte = (word >> (byte_in_word * 8u32)) & 0xffu32;
        let q_2bit = (qs_byte >> shift) & 0x3u32;

        let scale_byte = load(scales[block * 16u32 + scale_idx]).cast::<u32>();
        let scale_4bit = scale_byte & 0xfu32;
        let min_4bit = (scale_byte >> 4u32) & 0xfu32;

        let d = load(d_f32[block]);
        let dmin = load(dmin_f32[block]);

        let scaled =
            d * (scale_4bit.cast::<i32>().cast::<f32>()) * (q_2bit.cast::<i32>().cast::<f32>());
        let offset = dmin * (min_4bit.cast::<i32>().cast::<f32>());
        // Store the f32 result directly: the DSL narrows f32→T implicitly
        // at the Store site (an explicit `.cast::<T>()` would emit a
        // spurious same-type MSL cast).
        store(out[i], scaled - offset);
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_gguf_dequant_q2_k;
    use crate::{kernels::quant::gguf, utils::pack_f32};

    fn setup(n_blocks: usize, dt: DType) -> TestSetup {
        let n = n_blocks * 256;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007 - 0.5).sin() * 1.5).collect();
        // Pack + dequant via the shared GgufFormat oracle (kernels::quant::gguf) — the
        // canonical q2_k_qpos map + two-level decode now live in one place the
        // kernel, the quantizer, and this oracle all share, so the oracle can't
        // drift from the kernel (the bug class this fixes).
        let p = gguf::pack_q2_k(&values);
        let dequantized = gguf::dequant_q2_k(&p);
        // Pack u32 vec as little-endian bytes for the test framework.
        let qs_bytes: Vec<u8> = p.qs_packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        TestSetup::new(iron_gguf_dequant_q2_k::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("qs_packed", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", p.scales, DType::U8))
            .input(TestBuffer::from_vec("d_f32", pack_f32(&p.d, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("dmin_f32", pack_f32(&p.dmin, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n_values", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_single_block(dt: DType) -> TestSetup { setup(1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_many_blocks(dt: DType) -> TestSetup { setup(8, dt) }

    // ── External anchor (2026-08-07) ────────────────────────────────────
    //
    // `setup()` above (and `dequant_q2_k`/`q2_k_qpos` in `kernels::quant::
    // gguf`) is deliberately self-referential: the kernel, the quantizer,
    // and the CPU oracle all share one `q2_k_qpos` index map by design —
    // an earlier *independent* hand-rolled oracle had a wrong 2-bit byte
    // layout and produced a false-positive failure against a correct
    // kernel (see `gguf.rs:26-34`), and retiring independence in favor of
    // one shared definition fixed that. The trade-off: if `q2_k_qpos`
    // itself encodes the wrong permutation relative to the real on-disk
    // GGUF Q2_K format, kernel + quantizer + oracle + the bijection unit
    // test (`gguf.rs::q2_k_qpos_is_a_bijection_over_256`) all agree with
    // each other and ship silently wrong — `q2_k_qpos_is_a_bijection_over_256`
    // only proves the map is *a* permutation of the 256 slots, not *the*
    // correct one.
    //
    // This test breaks that circularity with one external anchor: a
    // single hand-picked `block_q2_K` byte sequence (scales/mins chosen
    // to include the 4-bit boundary values 0x0 and 0xF in both nibbles;
    // `qs` chosen to include the 2-bit boundary codes 0 and 3; `d`/`dmin`
    // exact powers of two so every expected output is an exact f32
    // fraction, no rounding ambiguity) and its dequantized values,
    // computed by an independent external decoder implementing the
    // published GGUF Q2_K on-disk format from `ggml-quants.c`'s
    // `dequantize_row_q2_K` (not this repo, not `kernels::quant::gguf`,
    // not the kernel's DSL body — a from-spec reimplementation on the
    // host side, cross-checked by hand against the block layout doc at
    // the top of this file). `Q2_K_ANCHOR_SCALES` / `_QS` / `_EXPECTED`
    // below are that external decoder's literal output, transcribed once;
    // regenerate from the spec (not from anything in this repo) if the
    // fixture ever needs to change.
    //
    // Per the mission's cost/benefit call: this is the "cheap, <1hr" path
    // the RST sweep asked for — one fixture, one assertion — not a
    // rewrite of the shared oracle, which the in-file rationale above
    // still stands behind.
    #[rustfmt::skip]
    const Q2_K_ANCHOR_SCALES: [u8; 16] = [
        0x00, 0xff, 0xf0, 0x0f, 0x4c, 0x38, 0x31, 0xa7,
        0x80, 0x68, 0xe6, 0x93, 0x2e, 0x93, 0x70, 0x19,
    ];
    #[rustfmt::skip]
    const Q2_K_ANCHOR_QS: [u8; 64] = [
        0x00, 0xff, 0x4d, 0x55, 0x9b, 0xaa, 0xc3, 0x59,
        0x73, 0x69, 0xa7, 0x9a, 0xbc, 0xa2, 0x77, 0xca,
        0xeb, 0x9e, 0x48, 0xa2, 0xbc, 0xf0, 0x97, 0x6a,
        0xd8, 0x75, 0xb4, 0x06, 0x16, 0xe3, 0xa2, 0x63,
        0x7c, 0xd6, 0xb3, 0x79, 0x8e, 0xf7, 0xc7, 0x2b,
        0x8a, 0x4c, 0xdf, 0x14, 0x98, 0xc7, 0xa2, 0x79,
        0xe2, 0x58, 0xd6, 0xc4, 0x82, 0xc6, 0x66, 0xea,
        0x0a, 0x20, 0xaa, 0xe2, 0x7d, 0x75, 0xf4, 0xc8,
    ];
    /// `d = 0.0625`, `dmin = 0.03125` — both exact powers of two, so every
    /// `out[i] = d*scale4*q2 - dmin*min4` term below is an exact f32
    /// fraction (no fp16<->f32 rounding to account for in the tolerance).
    #[rustfmt::skip]
    const Q2_K_ANCHOR_EXPECTED: [f32; 256] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.34375, 1.40625, -0.46875, 1.40625, -0.46875, -0.46875, 2.34375, 1.40625,
        -0.46875, 0.46875, -0.46875, 1.40625, 1.40625, 2.34375, 1.40625, 2.34375,
        -0.46875, -0.46875, -0.46875, -0.46875, -0.46875, -0.46875, -0.46875, -0.46875,
        -0.46875, -0.46875, -0.46875, -0.46875, -0.46875, -0.46875, -0.46875, -0.46875,
        1.875, 2.8125, 1.875, 0.0, 2.8125, 0.0, 0.9375, 1.875,
        1.875, 0.9375, 0.9375, 0.9375, 0.9375, 0.0, 0.0, 0.0,
        -0.125, 2.125, -0.125, 0.625, 0.625, 1.375, -0.125, 0.625,
        2.125, 1.375, 1.375, 0.625, 2.125, 1.375, 2.125, -0.125,
        0.90625, 0.40625, -0.09375, 0.90625, 1.40625, 1.40625, 0.40625, 0.90625,
        0.40625, 1.40625, 1.40625, -0.09375, 0.40625, 0.90625, 0.90625, 0.90625,
        -0.09375, 0.09375, -0.03125, -0.03125, 0.03125, 0.03125, 0.09375, -0.03125,
        -0.03125, -0.03125, 0.03125, 0.03125, 0.03125, 0.03125, -0.03125, 0.09375,
        1.0, 0.5625, 0.125, 0.5625, 0.5625, 1.0, 0.5625, 0.125,
        1.0, 0.125, 0.5625, -0.3125, -0.3125, 1.0, 0.5625, 0.125,
        -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,
        -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,
        0.8125, -0.1875, 0.8125, -0.1875, 0.8125, 0.8125, 0.8125, 0.8125,
        0.8125, -0.1875, 0.8125, 0.8125, 0.3125, 0.3125, -0.1875, -0.1875,
        0.6875, -0.0625, -0.4375, 0.3125, 0.6875, -0.0625, -0.0625, 0.3125,
        0.3125, 0.6875, 0.6875, -0.0625, 0.3125, -0.0625, -0.4375, 0.3125,
        -0.28125, 0.09375, -0.09375, -0.09375, -0.28125, -0.09375, -0.09375, 0.09375,
        0.09375, -0.28125, 0.09375, -0.28125, 0.28125, -0.09375, -0.09375, 0.09375,
        2.5625, 0.8125, 2.5625, 2.5625, -0.0625, 2.5625, -0.0625, 1.6875,
        -0.0625, -0.0625, 0.8125, 0.8125, 0.8125, -0.0625, 1.6875, 2.5625,
        0.09375, -0.09375, -0.09375, -0.28125, -0.28125, -0.28125, 0.09375, 0.09375,
        -0.28125, 0.09375, 0.09375, 0.09375, 0.28125, 0.28125, 0.28125, -0.28125,
        -0.21875, -0.21875, -0.21875, -0.21875, -0.21875, -0.21875, -0.21875, -0.21875,
        -0.21875, -0.21875, -0.21875, -0.21875, -0.21875, -0.21875, -0.21875, -0.21875,
        1.65625, 0.53125, 1.65625, 1.65625, 1.09375, 1.65625, 0.53125, 1.65625,
        -0.03125, -0.03125, 1.09375, 1.65625, 0.53125, 0.53125, 1.65625, 1.65625,
    ];

    fn external_anchor_setup(dt: DType) -> TestSetup {
        // The 64 raw `qs` bytes are already the LE byte layout of 16 u32
        // words (4 consecutive on-disk bytes = 1 word, byte 0 in bits
        // 0..8) — exactly what `qs_packed` expects, so no repacking beyond
        // a `.to_vec()` is needed (contrast `setup()` above, which starts
        // from `Vec<u32>` and has to *produce* that same LE layout).
        TestSetup::new(iron_gguf_dequant_q2_k::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("qs_packed", Q2_K_ANCHOR_QS.to_vec(), DType::U32))
            .input(TestBuffer::from_vec("scales", Q2_K_ANCHOR_SCALES.to_vec(), DType::U8))
            .input(TestBuffer::from_vec("d_f32", pack_f32(&[0.0625f32], DType::F32), DType::F32))
            .input(TestBuffer::from_vec(
                "dmin_f32",
                pack_f32(&[0.03125f32], DType::F32),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", 256, dt))
            .constexpr("n_values", 256u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&Q2_K_ANCHOR_EXPECTED, dt), dt))
            .grid_1d(256, 256)
    }

    /// Anchors the kernel (not `kernels::quant::gguf`'s shared oracle)
    /// against a single real-format `block_q2_K` byte sequence and an
    /// independent external decoder's output — see the block comment
    /// above for why this is needed alongside `setup()`'s self-consistent
    /// coverage. If `q2_k_qpos` ever gets the byte/shift permutation
    /// wrong in a way the bijection unit test can't see, this is the one
    /// test in the suite positioned to catch it.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_gguf_q2_k_external_anchor(dt: DType) -> TestSetup { external_anchor_setup(dt) }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_gguf_dequant_q2_k;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_q2_k(dt: DType) -> BenchSetup {
        // Representative MoE-expert down-proj slab — 4096 × 4096.
        let n = 4096 * 4096usize;
        let n_blocks = n / 256;
        BenchSetup::new(iron_gguf_dequant_q2_k::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("qs_packed", n_blocks * 16, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks * 16, DType::U8))
            .buffer(BenchBuffer::random("d_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::random("dmin_f32", n_blocks, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("n_values", n as u32)
            .grid_1d(n, 256)
            // qs_packed 64 B + scales 16 B + 2*4 B per block + output T
            .bytes_moved(((n_blocks * (64 + 16 + 8)) + n * dt.size_bytes()) as u64)
    }
}

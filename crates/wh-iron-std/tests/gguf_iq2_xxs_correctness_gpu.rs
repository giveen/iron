//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for the three GGUF IQ2_XXS kernels
//! (`iron_gguf_dequant_iq2_xxs`, `iron_gguf_dequant_iq2_xxs_raw`,
//! `iron_gguf_iq2_xxs_extract_qs`) — shipped, dispatched-in-production
//! kernels (confirmed: `w43-mtploop/Sources/IronSwift/Resources/manifest.json`
//! + compiled `.metal` files for all three × all dtypes, plus real call
//!   sites in `Sources/ButterSwift/Ops/OpsGGUF.swift:435-523`) that had
//!   **zero** correctness coverage before this file: every `kernel_tests`
//!   module in `gguf_dequant_iq2_xxs.rs` / `gguf_dequant_iq2_xxs_raw.rs` /
//!   `gguf_iq2_xxs_extract_qs.rs` only IR-shape "smoke tests" the kernel
//!   (`kernel_ir_for` + non-empty body + param-name presence) — none ever
//!   dispatches on GPU or checks a single output value. The dequant files'
//!   own doc comments call this "WIP scaffold... pending the canonical
//!   `iq2xxs_grid` table" — that table has existed in production Swift
//!   (`GGUFIQ2XXSTables.swift`) the whole time; this file ports it into a
//!   Rust test fixture and wires the missing dispatch + assertion.
//!
//! ## Table provenance
//!
//! `IQ2XXS_GRID_U64` / `IQ2XXS_KSIGNS` below are transcribed verbatim
//! (read-only reference, not linked/imported) from
//! `w43-mtploop/Sources/ButterSwift/Loader/GGUF/GGUFIQ2XXSTables.swift`
//! (`gridU64`, 256 entries; `ksigns`, 128 entries) — the same tables
//! Iron's Metal loader uploads at runtime init, both themselves
//! transcriptions of the canonical `iq2xxs_grid` / `ksigns_iq2xs` tables
//! from ggml-quants.c.
//!
//! ## Oracle independence
//!
//! `dequant_iq2_xxs_oracle` / `extract_qs_oracle` below are written
//! directly from the GGUF IQ2_XXS spec (the same algorithm the kernel
//! files' doc comments describe — grid lookup + 7-bit-index sign lookup
//! + composite scale) as fresh Rust functions in THIS file. They do not
//!   call into `iron_gguf_dequant_iq2_xxs` / `iron_gguf_dequant_iq2_xxs_raw`
//!   / `iron_gguf_iq2_xxs_extract_qs` or share code with them — only the
//!   documented on-disk format is shared, which is the point (an
//!   independent implementation of the spec, not a restatement of the
//!   kernel's DSL body).
//!
//! macOS-gated. Serial GPU lock (shared common::gpu_lock).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes, unpack_u32_bytes};
use wh_iron::Context;
use wh_iron_std::kernels::quant::{
    gguf_dequant_iq2_xxs::iron_gguf_dequant_iq2_xxs,
    gguf_dequant_iq2_xxs_raw::iron_gguf_dequant_iq2_xxs_raw,
    gguf_iq2_xxs_extract_qs::iron_gguf_iq2_xxs_extract_qs,
};

// ── Table provenance: transcribed from GGUFIQ2XXSTables.swift ──────────────
// (`w43-mtploop/Sources/ButterSwift/Loader/GGUF/GGUFIQ2XXSTables.swift`,
// `gridU64` + `ksigns`, read 2026-08-07). Do not hand-edit — regenerate from
// that file (itself a transcription of ggml-quants.c's `iq2xxs_grid` /
// `ksigns_iq2xs`) if either ever changes.
#[rustfmt::skip]
const IQ2XXS_GRID_U64: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

#[rustfmt::skip]
const IQ2XXS_KSIGNS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

/// `iq2xxs_grid` as `[256*8]` u8 (row-major, one signed-magnitude octet per
/// byte) — the layout the kernel's `Tensor<u8> grid` expects
/// (`grid[key*8 + lane]`), matching `GGUFIQ2XXSTables.swift`'s own
/// little-endian byte-decomposition of `gridU64`.
fn grid_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(256 * 8);
    for entry in IQ2XXS_GRID_U64 {
        out.extend_from_slice(&entry.to_le_bytes());
    }
    out
}

/// Pack one IQ2_XXS group's `(aux_idx, aux_sgn)` u32 pair from its four
/// per-octet `(grid_key, sign_idx)` values plus the group's shared 4-bit
/// scale, per the documented bit layout (`gguf_dequant_iq2_xxs.rs:22-40`):
/// `aux_idx` = 4 grid-key bytes; `aux_sgn` = 4 packed 7-bit sign indices
/// (bits 0..27) + 4-bit scale (bits 28..31).
fn pack_group_words(grid_keys: [u32; 4], sign_idxs: [u32; 4], scale_4bit: u32) -> (u32, u32) {
    let aux_idx = grid_keys[0] | (grid_keys[1] << 8) | (grid_keys[2] << 16) | (grid_keys[3] << 24);
    let aux_sgn = (sign_idxs[0] & 0x7f)
        | ((sign_idxs[1] & 0x7f) << 7)
        | ((sign_idxs[2] & 0x7f) << 14)
        | ((sign_idxs[3] & 0x7f) << 21)
        | ((scale_4bit & 0xf) << 28);
    (aux_idx, aux_sgn)
}

/// Independent CPU dequant oracle, written directly from the GGUF
/// IQ2_XXS spec (grid lookup + sign application + composite scale) —
/// NOT a call into the kernel. `qs_u32` is `[n_blocks*16]` (2 words per
/// group, 8 groups per block); `d_f32` is `[n_blocks]`.
fn dequant_iq2_xxs_oracle(
    qs_u32: &[u32],
    d_f32: &[f32],
    grid: &[u8],
    signs: &[u8],
    n_blocks: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_blocks * 256];
    for block in 0..n_blocks {
        let d = d_f32[block];
        for group in 0..8usize {
            let aux_idx = qs_u32[block * 16 + group * 2];
            let aux_sgn = qs_u32[block * 16 + group * 2 + 1];
            let scale_4bit = aux_sgn >> 28;
            let db = d * ((scale_4bit as f32) + 0.5) * 0.25;
            for j in 0..4usize {
                let grid_key = ((aux_idx >> (8 * j)) & 0xff) as usize;
                let sign_idx = ((aux_sgn >> (7 * j)) & 0x7f) as usize;
                let sign_mask = signs[sign_idx];
                for k in 0..8usize {
                    let octet = f32::from(grid[grid_key * 8 + k]);
                    let sign = if (sign_mask >> k) & 1 != 0 { -1.0f32 } else { 1.0f32 };
                    out[block * 256 + group * 32 + j * 8 + k] = db * sign * octet;
                }
            }
        }
    }
    out
}

/// Independent oracle for the qs-extract kernel: raw 66-byte-per-block
/// GGUF layout -> packed `[n_blocks*16]` u32 words, per
/// `gguf_iq2_xxs_extract_qs.rs:12-25`'s documented layout. Fresh
/// little-endian byte assembly, no shared code with the kernel.
fn extract_qs_oracle(raw_bytes: &[u8], n_blocks: usize) -> Vec<u32> {
    let mut out = vec![0u32; n_blocks * 16];
    for block in 0..n_blocks {
        for w in 0..16usize {
            let off = block * 66 + 2 + w * 4;
            out[block * 16 + w] = u32::from_le_bytes([
                raw_bytes[off],
                raw_bytes[off + 1],
                raw_bytes[off + 2],
                raw_bytes[off + 3],
            ]);
        }
    }
    out
}

/// Build a 2-block `qs_u32` fixture covering representative + boundary
/// cases:
///   - block 0, group 0, j=0: grid_key=0 (min grid index), sign_idx=0
///     (sign mask 0 — no octets flip), scale_4bit=0 (min scale)
///   - block 0, group 0, j=3: grid_key=255 (max grid index), sign_idx=127
///     (sign mask 255 — every octet flips), scale_4bit=0 (shared per group
///     with j=0..2, see below)
///   - block 0, groups 1..7: deterministic pseudo-random grid_key /
///     sign_idx / scale_4bit (small LCG), scale_4bit=15 (max) on group 1
///     specifically to also cover the opposite scale boundary
///   - block 1: fully pseudo-random representative block, different `d`
///     (1.7 vs block 0's 0.6) so per-block `d` indexing is exercised
///
/// IQ2_XXS has no partial/tail-block concept — every block is a fixed
/// 66 bytes / 256 values (`gguf_dequant_iq2_xxs.rs:16-20`); the loader
/// enforces whole-block alignment the same way it does for Q2_K/Q8_0, so
/// there is no boundary case to add there (mirrors the existing
/// `gguf.rs` Q2_K/Q8_0 coverage note).
fn build_fixture() -> (Vec<u32>, Vec<f32>) {
    let n_blocks = 2usize;
    let mut qs_u32 = vec![0u32; n_blocks * 16];
    let d_f32 = vec![0.6f32, 1.7f32];

    let mut lcg_state: u32 = 0x2463_1741;
    let mut next = |modulus: u32| -> u32 {
        lcg_state = lcg_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        lcg_state % modulus
    };

    for block in 0..n_blocks {
        for group in 0..8usize {
            let (grid_keys, sign_idxs, scale_4bit) = if block == 0 && group == 0 {
                // Boundary case: min grid index + no-flip sign at j=0,
                // max grid index + all-flip sign at j=3, min scale.
                ([0u32, next(256), next(256), 255u32], [0u32, next(128), next(128), 127u32], 0u32)
            } else if block == 0 && group == 1 {
                // Opposite scale boundary (max scale_4bit=15).
                (
                    [next(256), next(256), next(256), next(256)],
                    [next(128), next(128), next(128), next(128)],
                    15u32,
                )
            } else {
                (
                    [next(256), next(256), next(256), next(256)],
                    [next(128), next(128), next(128), next(128)],
                    next(16),
                )
            };
            let (aux_idx, aux_sgn) = pack_group_words(grid_keys, sign_idxs, scale_4bit);
            qs_u32[block * 16 + group * 2] = aux_idx;
            qs_u32[block * 16 + group * 2 + 1] = aux_sgn;
        }
    }
    (qs_u32, d_f32)
}

/// `iron_gguf_dequant_iq2_xxs` (pre-split `qs_u32` variant) vs. the
/// independent oracle above, across all three output dtypes.
#[test]
fn dequant_iq2_xxs_matches_oracle() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (qs_u32, d_f32) = build_fixture();
    let n_blocks = d_f32.len();
    let n = n_blocks * 256;
    let grid = grid_bytes();
    let signs = IQ2XXS_KSIGNS.to_vec();
    let want = dequant_iq2_xxs_oracle(&qs_u32, &d_f32, &grid, &signs, n_blocks);

    for dt in [Dt::F32, Dt::F16, Dt::Bf16] {
        let dtype = dt.to_dtype();
        let mut buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buf.insert("qs_u32".into(), pack_u32_bytes(&qs_u32));
        buf.insert("d_f32".into(), pack_bytes(&d_f32, Dt::F32));
        buf.insert("grid".into(), grid.clone());
        buf.insert("signs".into(), signs.clone());
        buf.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
        buf.insert("n_values".into(), (n as u32).to_le_bytes().to_vec());

        let kernel = iron_gguf_dequant_iq2_xxs::kernel_ir_for(dtype);
        let tpg = 256usize;
        let groups = n.div_ceil(tpg);
        let out = ctx
            .dispatch_with_grid(&kernel, &buf, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("iron_gguf_dequant_iq2_xxs dispatch");
        let got = unpack_bytes(out.outputs.get("out").expect("out"), dt);

        let mut max_abs = 0.0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((a - dt.round(*b)).abs());
        }
        // Grid octets are small integers {8,25,43}; scale is an exact
        // quarter-step; d is a clean decimal. f32 should be exact-ish
        // (float rounding only); f16/bf16 tolerance follows their
        // mantissa width at this magnitude (|out| up to ~1.7*43*3.9≈285).
        let tol = match dt {
            Dt::F32 => 5e-3,
            Dt::F16 => 0.6,
            Dt::Bf16 => 3.0,
        };
        assert!(max_abs <= tol, "{dt:?}: max|Δ|={max_abs:.4} > tol {tol}");
    }
}

/// `iron_gguf_dequant_iq2_xxs_raw` (raw on-disk-bytes variant) vs. the
/// SAME oracle output as the qs_u32 variant above — both kernels
/// implement the identical algorithm over the identical group data, just
/// sourced from a pre-split buffer vs. the raw 66-byte block layout.
#[test]
fn dequant_iq2_xxs_raw_matches_oracle() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (qs_u32, d_f32) = build_fixture();
    let n_blocks = d_f32.len();
    let n = n_blocks * 256;
    let grid = grid_bytes();
    let signs = IQ2XXS_KSIGNS.to_vec();
    let want = dequant_iq2_xxs_oracle(&qs_u32, &d_f32, &grid, &signs, n_blocks);

    // Raw 66-byte-per-block layout: 2 dummy d-header bytes (kernel reads
    // `d` from the separate `d_f32` tensor, never from these — see
    // `gguf_dequant_iq2_xxs_raw.rs:9-14`) + the 64 qs bytes as the
    // little-endian encoding of the same `qs_u32` words used above.
    let mut raw_bytes = vec![0u8; n_blocks * 66];
    for block in 0..n_blocks {
        for w in 0..16usize {
            let word = qs_u32[block * 16 + w];
            let off = block * 66 + 2 + w * 4;
            raw_bytes[off..off + 4].copy_from_slice(&word.to_le_bytes());
        }
    }

    for dt in [Dt::F32, Dt::F16, Dt::Bf16] {
        let dtype = dt.to_dtype();
        let mut buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buf.insert("raw_bytes".into(), raw_bytes.clone());
        buf.insert("d_f32".into(), pack_bytes(&d_f32, Dt::F32));
        buf.insert("grid".into(), grid.clone());
        buf.insert("signs".into(), signs.clone());
        buf.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
        buf.insert("n_values".into(), (n as u32).to_le_bytes().to_vec());

        let kernel = iron_gguf_dequant_iq2_xxs_raw::kernel_ir_for(dtype);
        let tpg = 256usize;
        let groups = n.div_ceil(tpg);
        let out = ctx
            .dispatch_with_grid(&kernel, &buf, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("iron_gguf_dequant_iq2_xxs_raw dispatch");
        let got = unpack_bytes(out.outputs.get("out").expect("out"), dt);

        let mut max_abs = 0.0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((a - dt.round(*b)).abs());
        }
        let tol = match dt {
            Dt::F32 => 5e-3,
            Dt::F16 => 0.6,
            Dt::Bf16 => 3.0,
        };
        assert!(max_abs <= tol, "{dt:?}: max|Δ|={max_abs:.4} > tol {tol}");
    }
}

/// `iron_gguf_iq2_xxs_extract_qs` (raw-bytes -> packed `qs_u32`) vs. the
/// independent byte-assembly oracle. Uses the same raw-bytes fixture as
/// the raw-dequant test above (qs portion only; the extract kernel never
/// touches the d-header bytes either — see `gguf_iq2_xxs_extract_qs.rs`).
#[test]
fn extract_qs_matches_oracle() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (qs_u32, d_f32) = build_fixture();
    let n_blocks = d_f32.len();
    let mut raw_bytes = vec![0u8; n_blocks * 66];
    for block in 0..n_blocks {
        for w in 0..16usize {
            let word = qs_u32[block * 16 + w];
            let off = block * 66 + 2 + w * 4;
            raw_bytes[off..off + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    let want = extract_qs_oracle(&raw_bytes, n_blocks);

    let mut buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buf.insert("raw_bytes".into(), raw_bytes);
    buf.insert("qs_u32".into(), pack_u32_bytes(&vec![0u32; n_blocks * 16]));
    buf.insert("n_blocks".into(), (n_blocks as u32).to_le_bytes().to_vec());

    let kernel = iron_gguf_iq2_xxs_extract_qs::kernel_ir_for();
    let n_words = n_blocks * 16;
    let tpg = 256usize;
    let groups = n_words.div_ceil(tpg);
    let out = ctx
        .dispatch_with_grid(&kernel, &buf, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("iron_gguf_iq2_xxs_extract_qs dispatch");
    let got = unpack_u32_bytes(out.outputs.get("qs_u32").expect("qs_u32"));

    assert_eq!(got, want, "extract_qs output diverges from the independent byte-assembly oracle");
    // Sanity: this should also exactly equal the qs_u32 fixture itself
    // (extract_qs's whole job is reconstructing it from raw bytes).
    assert_eq!(got, qs_u32, "extract_qs did not reproduce the source qs_u32 words");
}

/// Mutation-kill evidence: flip one sign bit in the `signs` (ksigns)
/// fixture table — `signs[0]` (used by block 0 / group 0 / j=0 in the
/// fixture above, sign mask 0 -> 1, which flips octet 0's sign) — and
/// confirm the kernel's output diverges from the clean run at exactly
/// that position. Proves `dequant_iq2_xxs_matches_oracle` has teeth: a
/// wrong sign-table entry would actually be caught, not silently pass.
#[test]
fn dequant_iq2_xxs_sign_bitflip_diverges() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (qs_u32, d_f32) = build_fixture();
    let n_blocks = d_f32.len();
    let n = n_blocks * 256;
    let grid = grid_bytes();
    let dt = Dt::F32;

    let run = |signs: &[u8]| -> Vec<f32> {
        let mut buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buf.insert("qs_u32".into(), pack_u32_bytes(&qs_u32));
        buf.insert("d_f32".into(), pack_bytes(&d_f32, Dt::F32));
        buf.insert("grid".into(), grid.clone());
        buf.insert("signs".into(), signs.to_vec());
        buf.insert("out".into(), pack_bytes(&vec![0.0f32; n], dt));
        buf.insert("n_values".into(), (n as u32).to_le_bytes().to_vec());
        let kernel = iron_gguf_dequant_iq2_xxs::kernel_ir_for(dt.to_dtype());
        let tpg = 256usize;
        let groups = n.div_ceil(tpg);
        let out = ctx
            .dispatch_with_grid(&kernel, &buf, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
            .expect("iron_gguf_dequant_iq2_xxs dispatch");
        unpack_bytes(out.outputs.get("out").expect("out"), dt)
    };

    let clean = run(&IQ2XXS_KSIGNS);
    let mut corrupted_signs = IQ2XXS_KSIGNS;
    corrupted_signs[0] ^= 0x01; // flip bit 0 of the sign_idx=0 mask (was 0 -> now 1)
    let corrupted = run(&corrupted_signs);

    let mut max_abs_diff = 0.0f32;
    for (a, b) in clean.iter().zip(corrupted.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    eprintln!("iq2_xxs sign-bitflip mutation: max|Δ|={max_abs_diff:.4}");
    assert!(
        max_abs_diff > 1e-3,
        "mutation check: expected the corrupted-signs run to diverge from the clean run \
         (max|Δ|={max_abs_diff:.6}) — if this is ~0, the oracle comparison has no teeth",
    );
}

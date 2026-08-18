//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MLX-format dequantizing GEMV kernels for int2 / int3 / int4 / int5 /
//! int6 / int8 weights. Reduction-mode kernels; one threadgroup per output row.
//!
//! Layouts (per dtype, with N = `in_dim`, G = `group_size`):
//!
//!   weight  [out_dim, N * bits / 32]   uint32  (bit-packed)
//!   scales  [out_dim, N / G]           T
//!   biases  [out_dim, N / G]           T
//!   input   [N]                        T
//!   output  [out_dim]                  T
//!
//! Two dispatch strategies, chosen per bit-width via compile-time
//! `if 32u32 % BITS == 0`:
//!
//! **Pack-strided** (BITS ∈ {2, 4, 8}) — threads stride over u32 packs; each
//! pack yields `32/BITS` values. One u32 load amortises across all values in
//! the pack; no extra bit-extraction arithmetic beyond a simple shift+mask.
//! Requires `32 % BITS == 0` (i.e. BITS divides a u32 evenly).
//!
//! **Element-strided** (BITS ∈ {5, 6}) — threads stride over individual
//! elements using the two-word bit-stream formula from `dequant_gather.rs`.
//! Used when BITS does not divide 32 evenly; element-striding is cleaner and
//! achieves the same cache behaviour (adjacent threads share the same u32
//! words → L1 multicast) while avoiding the idle-thread problem of the old
//! group-strided approach.
//!
//! **Block-aligned** (BITS == 3) — threads stride over 32-value blocks
//! instead of individual elements: 32 * 3 bits = 96 bits = exactly 3 u32
//! words, so a block's `w0`/`w1`/`w2` load once and all 32 3-bit fields
//! are extracted via compile-time-constant shifts/masks (no per-element
//! global load, no per-element `bit_off / 32u32` division — see the
//! `BITS == 3u32` arm below for the derivation). Falls back to the
//! per-element formula for a trailing partial block when `in_dim` is not
//! a multiple of 32.
//!
//! ## Variant axis
//!
//! `#[kernel(variants(BITS = [2, 3, 4, 5, 6, 8], suffix = "int{BITS}"))]`
//! produces kernels: `iron_dequant_gemv_int2`, `_int3`, `_int4`, `_int5`,
//! `_int6`, `_int8`. The `iron_dequant_gemv_int2_fast` and
//! `iron_dequant_gemv_int4_fast` kernels are separate perf-tuned variants
//! with a different algorithm.

use wh_iron::kernel;

/// Dequantizing GEMV — variable bit-widths (2, 3, 4, 5, 6, 8).
///
/// Produces: `iron_dequant_gemv_int2`, `_int3`, `_int4`, `_int5`, `_int6`,
/// `_int8`. One threadgroup per output row; `reduce_sum` across lanes.
///
/// `32 % BITS == 0` (BITS ∈ {2,4,8}): pack-strided (one u32 covers `32/BITS` elements).
/// `32 % BITS != 0` (BITS ∈ {3,5,6}): element-strided two-word bit-stream (`lo | hi` formula).
#[kernel(variants(BITS = [2, 3, 4, 5, 6, 8], suffix = "int{BITS}"))]
pub fn iron_dequant_gemv<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let row_group_off = row * n_groups;
    let mut acc = 0.0f32;

    if 32u32 % BITS == 0 {
        // Pack-strided: one u32 load covers `32/BITS` values.
        let vals_per_pack = 32u32 / BITS;
        let mask = (1u32 << BITS) - 1u32;
        let n_packs_per_row = in_dim / vals_per_pack;
        let packs_per_group = group_size / vals_per_pack;
        let row_pack_off = row * n_packs_per_row;
        let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
        for p_iter in range(0u32, p_iters, 1u32) {
            let pack_idx = p_iter * lsize + tid;
            if pack_idx < n_packs_per_row {
                let g = pack_idx / packs_per_group;
                let scale = load(scales[row_group_off + g]).cast::<f32>();
                let bias = load(biases[row_group_off + g]).cast::<f32>();
                let packed = load(weight[row_pack_off + pack_idx]);
                let p_off = pack_idx * vals_per_pack;
                for i in range(0u32, vals_per_pack, 1u32) {
                    let q = (packed >> (i * BITS)) & mask;
                    acc = acc
                        + (q.cast::<f32>() * scale + bias) * load(input[p_off + i]).cast::<f32>();
                }
            }
        }
    } else if BITS == 3u32 {
        // Block-aligned int3 fast path. 32 values * 3 bits = 96 bits =
        // exactly 3 u32 words, so a 32-value block never straddles a
        // word boundary from the outside: each block loads w0/w1/w2
        // exactly once (vs. up to 64 unconditional weight loads for the
        // same 32 values under the old per-element two-word formula),
        // and every 3-bit field's word/shift is a compile-time literal
        // (no per-element `bit_off / 32u32` division). The 32 shift
        // amounts below are the closed-form solution of the old
        // per-element formula (`bit_off = i*3`, `word = bit_off/32`,
        // `bit_in_w = bit_off%32`) evaluated for i in 0..32 — 10 fields
        // land fully inside w0, 1 straddles w0/w1, 10 fully inside w1, 1
        // straddles w1/w2, 10 fully inside w2 (10+1+10+1+10 = 32).
        //
        // Group-size cadence: a block is only ever covered by a single
        // scale/bias pair when `group_size` is a multiple of 32 (true
        // for every group_size used by this kernel today: the
        // element-strided correctness test uses 32, production MoE/attn
        // weights use 64). The tail loop below (for `in_dim` not a
        // multiple of 32) falls back to the original per-element
        // two-word formula, so the kernel stays correct for arbitrary
        // `in_dim` even though the fast path only covers full blocks.
        let mask = 7u32; // (1u32 << 3) - 1u32
        let vals_per_block = 32u32;
        let words_per_block = 3u32; // BITS
        let n_blocks_per_row = in_dim / vals_per_block;
        let aligned_in_dim = n_blocks_per_row * vals_per_block;
        let row_u32_off = row * (in_dim * BITS / 32u32);
        let b_iters = (n_blocks_per_row + lsize - 1u32) / lsize;
        for b_iter in range(0u32, b_iters, 1u32) {
            let blk = b_iter * lsize + tid;
            if blk < n_blocks_per_row {
                let d0 = blk * vals_per_block;
                let g = d0 / group_size;
                let scale = load(scales[row_group_off + g]).cast::<f32>();
                let bias = load(biases[row_group_off + g]).cast::<f32>();
                let word_base = row_u32_off + blk * words_per_block;
                let w0 = load(weight[word_base]);
                let w1 = load(weight[word_base + 1u32]);
                let w2 = load(weight[word_base + 2u32]);

                let q0 = w0 & mask;
                let q1 = (w0 >> 3u32) & mask;
                let q2 = (w0 >> 6u32) & mask;
                let q3 = (w0 >> 9u32) & mask;
                let q4 = (w0 >> 12u32) & mask;
                let q5 = (w0 >> 15u32) & mask;
                let q6 = (w0 >> 18u32) & mask;
                let q7 = (w0 >> 21u32) & mask;
                let q8 = (w0 >> 24u32) & mask;
                let q9 = (w0 >> 27u32) & mask;
                let q10 = ((w0 >> 30u32) & 3u32) | ((w1 & 1u32) << 2u32);
                let q11 = (w1 >> 1u32) & mask;
                let q12 = (w1 >> 4u32) & mask;
                let q13 = (w1 >> 7u32) & mask;
                let q14 = (w1 >> 10u32) & mask;
                let q15 = (w1 >> 13u32) & mask;
                let q16 = (w1 >> 16u32) & mask;
                let q17 = (w1 >> 19u32) & mask;
                let q18 = (w1 >> 22u32) & mask;
                let q19 = (w1 >> 25u32) & mask;
                let q20 = (w1 >> 28u32) & mask;
                let q21 = ((w1 >> 31u32) & 1u32) | ((w2 & 3u32) << 1u32);
                let q22 = (w2 >> 2u32) & mask;
                let q23 = (w2 >> 5u32) & mask;
                let q24 = (w2 >> 8u32) & mask;
                let q25 = (w2 >> 11u32) & mask;
                let q26 = (w2 >> 14u32) & mask;
                let q27 = (w2 >> 17u32) & mask;
                let q28 = (w2 >> 20u32) & mask;
                let q29 = (w2 >> 23u32) & mask;
                let q30 = (w2 >> 26u32) & mask;
                let q31 = (w2 >> 29u32) & mask;

                acc = acc + (q0.cast::<f32>() * scale + bias) * load(input[d0]).cast::<f32>();
                acc =
                    acc + (q1.cast::<f32>() * scale + bias) * load(input[d0 + 1u32]).cast::<f32>();
                acc =
                    acc + (q2.cast::<f32>() * scale + bias) * load(input[d0 + 2u32]).cast::<f32>();
                acc =
                    acc + (q3.cast::<f32>() * scale + bias) * load(input[d0 + 3u32]).cast::<f32>();
                acc =
                    acc + (q4.cast::<f32>() * scale + bias) * load(input[d0 + 4u32]).cast::<f32>();
                acc =
                    acc + (q5.cast::<f32>() * scale + bias) * load(input[d0 + 5u32]).cast::<f32>();
                acc =
                    acc + (q6.cast::<f32>() * scale + bias) * load(input[d0 + 6u32]).cast::<f32>();
                acc =
                    acc + (q7.cast::<f32>() * scale + bias) * load(input[d0 + 7u32]).cast::<f32>();
                acc =
                    acc + (q8.cast::<f32>() * scale + bias) * load(input[d0 + 8u32]).cast::<f32>();
                acc =
                    acc + (q9.cast::<f32>() * scale + bias) * load(input[d0 + 9u32]).cast::<f32>();
                acc = acc
                    + (q10.cast::<f32>() * scale + bias) * load(input[d0 + 10u32]).cast::<f32>();
                acc = acc
                    + (q11.cast::<f32>() * scale + bias) * load(input[d0 + 11u32]).cast::<f32>();
                acc = acc
                    + (q12.cast::<f32>() * scale + bias) * load(input[d0 + 12u32]).cast::<f32>();
                acc = acc
                    + (q13.cast::<f32>() * scale + bias) * load(input[d0 + 13u32]).cast::<f32>();
                acc = acc
                    + (q14.cast::<f32>() * scale + bias) * load(input[d0 + 14u32]).cast::<f32>();
                acc = acc
                    + (q15.cast::<f32>() * scale + bias) * load(input[d0 + 15u32]).cast::<f32>();
                acc = acc
                    + (q16.cast::<f32>() * scale + bias) * load(input[d0 + 16u32]).cast::<f32>();
                acc = acc
                    + (q17.cast::<f32>() * scale + bias) * load(input[d0 + 17u32]).cast::<f32>();
                acc = acc
                    + (q18.cast::<f32>() * scale + bias) * load(input[d0 + 18u32]).cast::<f32>();
                acc = acc
                    + (q19.cast::<f32>() * scale + bias) * load(input[d0 + 19u32]).cast::<f32>();
                acc = acc
                    + (q20.cast::<f32>() * scale + bias) * load(input[d0 + 20u32]).cast::<f32>();
                acc = acc
                    + (q21.cast::<f32>() * scale + bias) * load(input[d0 + 21u32]).cast::<f32>();
                acc = acc
                    + (q22.cast::<f32>() * scale + bias) * load(input[d0 + 22u32]).cast::<f32>();
                acc = acc
                    + (q23.cast::<f32>() * scale + bias) * load(input[d0 + 23u32]).cast::<f32>();
                acc = acc
                    + (q24.cast::<f32>() * scale + bias) * load(input[d0 + 24u32]).cast::<f32>();
                acc = acc
                    + (q25.cast::<f32>() * scale + bias) * load(input[d0 + 25u32]).cast::<f32>();
                acc = acc
                    + (q26.cast::<f32>() * scale + bias) * load(input[d0 + 26u32]).cast::<f32>();
                acc = acc
                    + (q27.cast::<f32>() * scale + bias) * load(input[d0 + 27u32]).cast::<f32>();
                acc = acc
                    + (q28.cast::<f32>() * scale + bias) * load(input[d0 + 28u32]).cast::<f32>();
                acc = acc
                    + (q29.cast::<f32>() * scale + bias) * load(input[d0 + 29u32]).cast::<f32>();
                acc = acc
                    + (q30.cast::<f32>() * scale + bias) * load(input[d0 + 30u32]).cast::<f32>();
                acc = acc
                    + (q31.cast::<f32>() * scale + bias) * load(input[d0 + 31u32]).cast::<f32>();
            }
        }
        // Tail: any elements past the last full 32-value block, for
        // `in_dim` not a multiple of 32 (not exercised by any current
        // caller, kept for general correctness). Same formula as the
        // pre-rewrite element-strided path.
        let n_tail_iters = (in_dim - aligned_in_dim + lsize - 1u32) / lsize;
        for _iter in range(0u32, n_tail_iters, 1u32) {
            let d = aligned_in_dim + _iter * lsize + tid;
            if d < in_dim {
                let g = d / group_size;
                let scale = load(scales[row_group_off + g]).cast::<f32>();
                let bias = load(biases[row_group_off + g]).cast::<f32>();
                let bit_off = d * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                let spill = BITS - lo_bits;
                let w0t = load(weight[row_u32_off + word_idx]);
                let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                let w1t = load(weight[row_u32_off + w1idx]);
                let lo = (w0t >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                let hi = (w1t & ((1u32 << spill) - 1u32)) << lo_bits;
                let q = lo | hi;
                acc = acc + (q.cast::<f32>() * scale + bias) * load(input[d]).cast::<f32>();
            }
        }
    } else {
        // Element-strided: two-word bit-stream for odd widths (int5, int6).
        let mask = (1u32 << BITS) - 1u32;
        let u32_per_row = in_dim * BITS / 32u32;
        let row_u32_off = row * u32_per_row;
        let n_iters = (in_dim + lsize - 1u32) / lsize;
        for _iter in range(0u32, n_iters, 1u32) {
            let d = _iter * lsize + tid;
            if d < in_dim {
                let g = d / group_size;
                let scale = load(scales[row_group_off + g]).cast::<f32>();
                let bias = load(biases[row_group_off + g]).cast::<f32>();
                let bit_off = d * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let bits_in_w0 = 32u32 - bit_in_w;
                let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                let spill = BITS - lo_bits;
                let w0 = load(weight[row_u32_off + word_idx]);
                let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                let w1 = load(weight[row_u32_off + w1idx]);
                let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                let q = lo | hi;
                acc = acc + (q.cast::<f32>() * scale + bias) * load(input[d]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// Perf-tuned int2 GEMV, 8 output rows per TG.
//
// Uses the same 8-row geometry as `iron_dequant_gemv_int4_fast`, but each
// lane consumes one packed u32 holding 16 2-bit values. The input values are
// loaded once and reused across four weight rows in each simdgroup.

/// Perf-tuned int2 dequant GEMV, 8 rows per threadgroup.
///
/// `output[row] = Σ_i (q[row,i]·scale_g + bias_g) · input[i]`
/// for 8 consecutive output rows per dispatch. Grid: `[out_dim/8, 1, 1]`,
/// TPG = 64, group_size = 64, in_dim a multiple of 512.
#[kernel]
pub fn iron_dequant_gemv_int2_fast<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 16u32;
    let lane_x_off = lane * 16u32;
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }

    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        let x0 = load(input[xb]).cast::<f32>();
        let x1 = load(input[xb + 1u32]).cast::<f32>();
        let x2 = load(input[xb + 2u32]).cast::<f32>();
        let x3 = load(input[xb + 3u32]).cast::<f32>();
        let x4 = load(input[xb + 4u32]).cast::<f32>();
        let x5 = load(input[xb + 5u32]).cast::<f32>();
        let x6 = load(input[xb + 6u32]).cast::<f32>();
        let x7 = load(input[xb + 7u32]).cast::<f32>();
        let x8 = load(input[xb + 8u32]).cast::<f32>();
        let x9 = load(input[xb + 9u32]).cast::<f32>();
        let x10 = load(input[xb + 10u32]).cast::<f32>();
        let x11 = load(input[xb + 11u32]).cast::<f32>();
        let x12 = load(input[xb + 12u32]).cast::<f32>();
        let x13 = load(input[xb + 13u32]).cast::<f32>();
        let x14 = load(input[xb + 14u32]).cast::<f32>();
        let x15 = load(input[xb + 15u32]).cast::<f32>();
        let xs =
            x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;
        let g = xb / group_size;
        let pack_off = _b / 16u32 + lane;
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let packed = load(weight[row * packs_per_row + pack_off]);
            let sb = row * gs_per_row + g;
            let s = load(scales[sb]).cast::<f32>();
            let bi = load(biases[sb]).cast::<f32>();
            let qd = (packed & 3u32).cast::<f32>() * x0
                + ((packed >> 2u32) & 3u32).cast::<f32>() * x1
                + ((packed >> 4u32) & 3u32).cast::<f32>() * x2
                + ((packed >> 6u32) & 3u32).cast::<f32>() * x3
                + ((packed >> 8u32) & 3u32).cast::<f32>() * x4
                + ((packed >> 10u32) & 3u32).cast::<f32>() * x5
                + ((packed >> 12u32) & 3u32).cast::<f32>() * x6
                + ((packed >> 14u32) & 3u32).cast::<f32>() * x7
                + ((packed >> 16u32) & 3u32).cast::<f32>() * x8
                + ((packed >> 18u32) & 3u32).cast::<f32>() * x9
                + ((packed >> 20u32) & 3u32).cast::<f32>() * x10
                + ((packed >> 22u32) & 3u32).cast::<f32>() * x11
                + ((packed >> 24u32) & 3u32).cast::<f32>() * x12
                + ((packed >> 26u32) & 3u32).cast::<f32>() * x13
                + ((packed >> 28u32) & 3u32).cast::<f32>() * x14
                + ((packed >> 30u32) & 3u32).cast::<f32>() * x15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * xs);
        }
    }

    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(output[base_row + _r], r.cast::<T>());
        }
    }
}

// ── Perf-tuned int4 GEMV — 8 output rows per TG ─────────────────────────
//
// Mirrors `iron_qmv`'s geometry: tpg = 64 (2 simdgroups × 32 lanes);
// each simdgroup computes 4 output rows (indexed by `simd_id`); each
// lane caches 16 X values per 512-wide K-block. Uses the mask-without-
// shift trick + algebraic-split accumulator `s*q_dot + b*xs` from
// `iron_qmv` / MLX `qdot` (quantized.h:235-244).
//
// Kept separate from `iron_dequant_gemv_int4` (the one-row-per-TG scalar)
// for backward compat — Iron's GPU-router uses the indirect variant of
// the scalar kernel. The fast variant has no indirect consumer today;
// adding one is a one-line edit in `iron_dequant_gemv_wants_indirect`.
//
// Dispatch:
//   Grid: [out_dim/8, 1, 1]  — one TG per 8-row tile.
//   TPG: 64                  — 2 SG × 32 lanes.
//   in_dim: multiple of 512 (block = 16 X × 32 lanes = 512 K elements).
//   out_dim: multiple of 8.
//   group_size: 64.

/// Perf-tuned int4 dequant GEMV — 8 rows per TG, `iron_qmv` geometry.
///
/// `output[row] = Σ_i (q[row,i]·scale_g + bias_g) · input[i]`
/// for 8 consecutive output rows per dispatch. Grid: `[out_dim/8, 1, 1]`,
/// TPG = 64, group_size = 64, in_dim a multiple of 512.
///
/// The existing `iron_dequant_gemv_int4` is kept unchanged for backward compat
/// (Iron's indirect-dispatch router uses that name). This variant is the
/// perf path for new callers that can guarantee the alignment constraints.
///
/// ## Implementation notes
///
/// The 4-rows-per-simdgroup work is expressed as a `range(0u32, 4u32, 1u32)`
/// loop with a `stack_alloc("accs", 4, f32)` for the per-row accumulators.
/// The DSL unrolls constexpr-bounded `range(...)` loops at codegen, so the
/// emitted MSL is identical to the hand-unrolled form — same 4 weight
/// loads, same 16-nibble mask-without-shift dot per row — just expressed
/// in ~30 lines of loop body instead of 4 × ~40 line copy-pasted blocks.
/// `stack_alloc` accumulators are required because the DSL doesn't lower
/// runtime-indexed `let mut [T; N]` arrays (see the `_m{16,32}` notes in
/// `iron/moe.rs` for the same constraint).
#[kernel]
pub fn iron_dequant_gemv_int4_fast<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    // 8 rows per TG: SG 0 → rows 0-3, SG 1 → rows 4-7. `base_row` is
    // the first of the 4 rows this simdgroup owns.
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32; // 8 int4 values per u32
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // `thread`-private array indexable by a runtime loop variable.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    // Mask-without-shift constants — eliminates 56 shifts per block.
    // Matches `iron_qmv` / MLX `qdot` (quantized.h:235-244): instead of
    // shifting each nibble to position 0, multiply x[1/2/3] by 1/16,
    // 1/256, 1/4096 once and keep the nibble in its native bit slot.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        // 16 X loads per K-block, shared by all 4 rows. Slot 0/4/8/12
        // (the first nibble in each u16 half) is unscaled; the others
        // get pre-scaled by 1/16, 1/256, 1/4096 for mask-without-shift.
        let x0 = load(input[xb]).cast::<f32>();
        let x1_raw = load(input[xb + 1u32]).cast::<f32>();
        let x2_raw = load(input[xb + 2u32]).cast::<f32>();
        let x3_raw = load(input[xb + 3u32]).cast::<f32>();
        let x4 = load(input[xb + 4u32]).cast::<f32>();
        let x5_raw = load(input[xb + 5u32]).cast::<f32>();
        let x6_raw = load(input[xb + 6u32]).cast::<f32>();
        let x7_raw = load(input[xb + 7u32]).cast::<f32>();
        let x8 = load(input[xb + 8u32]).cast::<f32>();
        let x9_raw = load(input[xb + 9u32]).cast::<f32>();
        let x10_raw = load(input[xb + 10u32]).cast::<f32>();
        let x11_raw = load(input[xb + 11u32]).cast::<f32>();
        let x12 = load(input[xb + 12u32]).cast::<f32>();
        let x13_raw = load(input[xb + 13u32]).cast::<f32>();
        let x14_raw = load(input[xb + 14u32]).cast::<f32>();
        let x15_raw = load(input[xb + 15u32]).cast::<f32>();
        // Algebraic-split: acc = scale * q_dot + bias * xs, where
        // xs = Σ input[i] over the 16-element block.
        let xs = x0
            + x1_raw
            + x2_raw
            + x3_raw
            + x4
            + x5_raw
            + x6_raw
            + x7_raw
            + x8
            + x9_raw
            + x10_raw
            + x11_raw
            + x12
            + x13_raw
            + x14_raw
            + x15_raw;
        // Pre-scale at nibble positions 1/2/3 (within each u16 half).
        let x1 = x1_raw * s_16;
        let x2 = x2_raw * s_256;
        let x3 = x3_raw * s_4096;
        let x5 = x5_raw * s_16;
        let x6 = x6_raw * s_256;
        let x7 = x7_raw * s_4096;
        let x9 = x9_raw * s_16;
        let x10 = x10_raw * s_256;
        let x11 = x11_raw * s_4096;
        let x13 = x13_raw * s_16;
        let x14 = x14_raw * s_256;
        let x15 = x15_raw * s_4096;
        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;
        // 4 rows × identical work, looped — DSL unrolls at codegen.
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let w_base = row * packs_per_row;
            let sb_base = row * gs_per_row;
            let p_lo = load(weight[w_base + pack_off]);
            let p_hi_word = load(weight[w_base + pack_off + 1u32]);
            let p_lo_hi = p_lo >> 16u32;
            let p_hi_hi = p_hi_word >> 16u32;
            let s = load(scales[sb_base + g]).cast::<f32>();
            let bi = load(biases[sb_base + g]).cast::<f32>();
            // 16-nibble dot, mask-without-shift form. Each u32 carries
            // 8 nibbles split as 4 in the low 16 bits + 4 in the high
            // 16 bits; the four masks `15 / 240 / 3840 / 61440` peel off
            // the nibble at slot 0/1/2/3 of each half.
            let qd = (p_lo & 15u32).cast::<f32>() * x0
                + (p_lo & 240u32).cast::<f32>() * x1
                + (p_lo & 3840u32).cast::<f32>() * x2
                + (p_lo & 61440u32).cast::<f32>() * x3
                + (p_lo_hi & 15u32).cast::<f32>() * x4
                + (p_lo_hi & 240u32).cast::<f32>() * x5
                + (p_lo_hi & 3840u32).cast::<f32>() * x6
                + (p_lo_hi & 61440u32).cast::<f32>() * x7
                + (p_hi_word & 15u32).cast::<f32>() * x8
                + (p_hi_word & 240u32).cast::<f32>() * x9
                + (p_hi_word & 3840u32).cast::<f32>() * x10
                + (p_hi_word & 61440u32).cast::<f32>() * x11
                + (p_hi_hi & 15u32).cast::<f32>() * x12
                + (p_hi_hi & 240u32).cast::<f32>() * x13
                + (p_hi_hi & 3840u32).cast::<f32>() * x14
                + (p_hi_hi & 61440u32).cast::<f32>() * x15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * xs);
        }
    }
    // Cross-lane reduce: one simd_sum per row → one value per simdgroup.
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(output[base_row + _r], r.cast::<T>());
        }
    }
}

/// Perf-tuned int4 dequant GEMV over a gathered list of weight rows.
///
/// `row_indices[out_row]` selects the source matrix row for each output.
/// The output count must be a multiple of 8. Grid, threadgroup, input,
/// group-size, and alignment contracts match `iron_dequant_gemv_int4_fast`.
#[kernel]
pub fn iron_dequant_gemv_int4_gathered<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    row_indices: Tensor<u32>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let base_out_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }

    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        let x0 = load(input[xb]).cast::<f32>();
        let x1_raw = load(input[xb + 1u32]).cast::<f32>();
        let x2_raw = load(input[xb + 2u32]).cast::<f32>();
        let x3_raw = load(input[xb + 3u32]).cast::<f32>();
        let x4 = load(input[xb + 4u32]).cast::<f32>();
        let x5_raw = load(input[xb + 5u32]).cast::<f32>();
        let x6_raw = load(input[xb + 6u32]).cast::<f32>();
        let x7_raw = load(input[xb + 7u32]).cast::<f32>();
        let x8 = load(input[xb + 8u32]).cast::<f32>();
        let x9_raw = load(input[xb + 9u32]).cast::<f32>();
        let x10_raw = load(input[xb + 10u32]).cast::<f32>();
        let x11_raw = load(input[xb + 11u32]).cast::<f32>();
        let x12 = load(input[xb + 12u32]).cast::<f32>();
        let x13_raw = load(input[xb + 13u32]).cast::<f32>();
        let x14_raw = load(input[xb + 14u32]).cast::<f32>();
        let x15_raw = load(input[xb + 15u32]).cast::<f32>();
        let xs = x0
            + x1_raw
            + x2_raw
            + x3_raw
            + x4
            + x5_raw
            + x6_raw
            + x7_raw
            + x8
            + x9_raw
            + x10_raw
            + x11_raw
            + x12
            + x13_raw
            + x14_raw
            + x15_raw;
        let x1 = x1_raw * s_16;
        let x2 = x2_raw * s_256;
        let x3 = x3_raw * s_4096;
        let x5 = x5_raw * s_16;
        let x6 = x6_raw * s_256;
        let x7 = x7_raw * s_4096;
        let x9 = x9_raw * s_16;
        let x10 = x10_raw * s_256;
        let x11 = x11_raw * s_4096;
        let x13 = x13_raw * s_16;
        let x14 = x14_raw * s_256;
        let x15 = x15_raw * s_4096;
        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;
        for _r in range(0u32, 4u32, 1u32) {
            let out_row = base_out_row + _r;
            let source_row = load(row_indices[out_row]);
            let w_base = source_row * packs_per_row;
            let sb_base = source_row * gs_per_row;
            let p_lo = load(weight[w_base + pack_off]);
            let p_hi_word = load(weight[w_base + pack_off + 1u32]);
            let p_lo_hi = p_lo >> 16u32;
            let p_hi_hi = p_hi_word >> 16u32;
            let s = load(scales[sb_base + g]).cast::<f32>();
            let bi = load(biases[sb_base + g]).cast::<f32>();
            let qd = (p_lo & 15u32).cast::<f32>() * x0
                + (p_lo & 240u32).cast::<f32>() * x1
                + (p_lo & 3840u32).cast::<f32>() * x2
                + (p_lo & 61440u32).cast::<f32>() * x3
                + (p_lo_hi & 15u32).cast::<f32>() * x4
                + (p_lo_hi & 240u32).cast::<f32>() * x5
                + (p_lo_hi & 3840u32).cast::<f32>() * x6
                + (p_lo_hi & 61440u32).cast::<f32>() * x7
                + (p_hi_word & 15u32).cast::<f32>() * x8
                + (p_hi_word & 240u32).cast::<f32>() * x9
                + (p_hi_word & 3840u32).cast::<f32>() * x10
                + (p_hi_word & 61440u32).cast::<f32>() * x11
                + (p_hi_hi & 15u32).cast::<f32>() * x12
                + (p_hi_hi & 240u32).cast::<f32>() * x13
                + (p_hi_hi & 3840u32).cast::<f32>() * x14
                + (p_hi_hi & 61440u32).cast::<f32>() * x15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * xs);
        }
    }
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(output[base_out_row + _r], r.cast::<T>());
        }
    }
}

/// Per-kernel opt-in for the indirect Swift-wrapper variant. Iron's
/// GPU-router dispatches the int4 dequant-GEMV indirectly so the GPU
/// can drive the per-MoE-layer grid shape from a buffer; the other
/// dequant-GEMV bit-widths have no indirect consumer today.
///
/// Lives here (next to the kernel definitions) rather than in
/// `wh-iron-codegen` so that adding a new kernel that wants the
/// indirect variant is a one-line edit in the same file as the
/// kernel, not a special-case match buried in the codegen pass.
/// The `tile emit` driver consumes this on the way to setting
/// `Kernel::wants_indirect_variant` before codegen runs.
pub fn iron_dequant_gemv_wants_indirect(kernel_name: &str) -> bool {
    matches!(kernel_name, "iron_dequant_gemv_int4_f16" | "iron_dequant_gemv_int4_bf16")
}

/// New-syntax correctness tests for the `iron_dequant_gemv_int{2,3,4,5,6,8}`
/// family + the perf-tuned `iron_dequant_gemv_int4_fast`. All are Reduction-mode
/// (one threadgroup per output row, `reduce_sum` across the threadgroup).
///
/// Oracle: synthesize bit-stream-packed int-`bits` weights `[out_dim, in_dim]`
/// (the same `lo | hi` two-word layout the kernel decodes — works for both the
/// pack-strided pow2 widths and the odd widths), per-group scale/bias, then
/// replay the dequant-then-dot `output[row] = Σ_d (q·scale_g + bias_g)·input[d]`
/// in f32. Inputs are dtype-rounded so the GPU sees exactly what the oracle does.
///
/// Grid (scalar variants): `grid_3d(out_dim, 1, 1, [tpg, 1, 1])` — one TG per
/// output row, tpg = 64 (≥32, multiple of 32). The `_fast` variant does 8 rows
/// per TG so `grid_3d(out_dim/8, 1, 1, [64, 1, 1])`.
pub mod kernel_tests {
    use wh_iron::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    /// Bytes for a u32 slice (packed weights bind as a `DType::U32` buffer).
    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// One threadgroup-row's worth of lanes. ≥ 32 and a multiple of 32 per the
    /// Reduction dispatch contract; 64 lanes give a healthy `reduce_sum` tree.
    const TPG: u32 = 64;

    /// Synthesize bit-stream-packed int-`bits` weights for an `[out_dim, in_dim]`
    /// matrix. Codes are written into the row's u32 bit-stream at bit offset
    /// `d * bits`, spilling into the next word when they straddle a u32 boundary
    /// — the exact layout the kernel's `lo | hi` decode (and the legacy test's
    /// `quantize_row`) expects. Works for every supported bit width.
    fn synth_bitstream_w(out_dim: usize, in_dim: usize, bits: u32) -> Vec<u32> {
        let mask = (1u32 << bits) - 1;
        let u32_per_row = in_dim * bits as usize / 32;
        let mut packed = vec![0u32; out_dim * u32_per_row];
        for row in 0..out_dim {
            let row_base = row * u32_per_row;
            for d in 0..in_dim {
                // Deterministic, in-range code; varies per (row, d).
                let code =
                    ((row * in_dim + d) as u32).wrapping_mul(2_654_435_761).wrapping_add(d as u32)
                        & mask;
                let bit_off = (d * bits as usize) as u32;
                let word = (bit_off / 32) as usize;
                let in_w = bit_off & 31;
                let bits_in_w0 = 32 - in_w;
                if bits_in_w0 >= bits {
                    packed[row_base + word] |= code << in_w;
                } else {
                    packed[row_base + word] |= code << in_w;
                    packed[row_base + word + 1] |= code >> bits_in_w0;
                }
            }
        }
        packed
    }

    /// Dequant-then-dot reference (mirrors the legacy `naive_dequant_gemv`).
    /// `weight` packs `[out_dim, in_dim]` int-`bits` codes, `scales`/`biases`
    /// are `[out_dim, in_dim/group_size]`, `input` is `[in_dim]`, out `[out_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn iron_dequant_gemv_oracle(
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        input: &[f32],
        in_dim: usize,
        group_size: usize,
        bits: u32,
        out_dim: usize,
    ) -> Vec<f32> {
        let u32_per_row = in_dim * bits as usize / 32;
        let n_groups = in_dim / group_size;
        let mask: u64 = (1u64 << bits) - 1;
        let mut out = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut acc = 0.0f32;
            let row_w = &weight[row * u32_per_row..(row + 1) * u32_per_row];
            for (d, &x_d) in input.iter().enumerate().take(in_dim) {
                let g = d / group_size;
                let bit_off = (d * bits as usize) as u32;
                let word = (bit_off / 32) as usize;
                let in_w = bit_off & 31;
                let bits_in_w0 = 32 - in_w;
                let q = if bits_in_w0 >= bits {
                    ((row_w[word] as u64) >> in_w) & mask
                } else {
                    let lo_bits = bits_in_w0;
                    let spill = bits - lo_bits;
                    let lo = ((row_w[word] as u64) >> in_w) & ((1u64 << lo_bits) - 1);
                    let hi = ((row_w[word + 1] as u64) & ((1u64 << spill) - 1)) << lo_bits;
                    lo | hi
                };
                acc += ((q as f32) * scales[row * n_groups + g] + biases[row * n_groups + g]) * x_d;
            }
            out[row] = acc;
        }
        out
    }

    /// Shared setup for the scalar (one-row-per-TG) variants. `grid_rows` is the
    /// number of x-groups dispatched (out_dim) and `tpg` the lanes per row.
    #[allow(clippy::too_many_arguments)]
    fn gemv_setup(
        kernel: Kernel,
        bits: u32,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        grid_rows: u32,
        tpg: u32,
        dt: DType,
    ) -> TestSetup {
        let n_groups = in_dim / group_size;
        let w = synth_bitstream_w(out_dim, in_dim, bits);
        let scales_f: Vec<f32> =
            (0..out_dim * n_groups).map(|i| 0.004 + (i % 7) as f32 * 0.0008).collect();
        let biases_f: Vec<f32> =
            (0..out_dim * n_groups).map(|i| ((i % 5) as f32 - 2.0) * 0.0009).collect();
        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let expected = iron_dequant_gemv_oracle(&w, &s, &b, &x, in_dim, group_size, bits, out_dim);
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", u32_bytes(&w), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            .grid_3d(grid_rows, 1, 1, [tpg, 1, 1])
    }

    fn gathered_gemv_setup(
        dt: DType,
        matrix_rows: usize,
        in_dim: usize,
        row_indices: Vec<u32>,
    ) -> TestSetup {
        let group_size = 64usize;
        let n_groups = in_dim / group_size;
        let w = synth_bitstream_w(matrix_rows, in_dim, 4);
        let scales_f: Vec<f32> =
            (0..matrix_rows * n_groups).map(|i| 0.004 + (i % 7) as f32 * 0.0008).collect();
        let biases_f: Vec<f32> =
            (0..matrix_rows * n_groups).map(|i| ((i % 5) as f32 - 2.0) * 0.0009).collect();
        let input_f: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&input_f, dt), dt);
        let full = iron_dequant_gemv_oracle(&w, &s, &b, &x, in_dim, group_size, 4, matrix_rows);
        let expected: Vec<f32> = row_indices.iter().map(|&row| full[row as usize]).collect();
        TestSetup::new(iron_dequant_gemv_int4_gathered::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weight", u32_bytes(&w), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("row_indices", u32_bytes(&row_indices), DType::U32))
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("output", row_indices.len(), dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            .grid_3d((row_indices.len() / 8) as u32, 1, 1, [64, 1, 1])
    }

    // Pack-strided (32 % BITS == 0): BITS ∈ {2, 4, 8}; in_dim a multiple of 32/BITS.
    // Element-strided (32 % BITS != 0): BITS ∈ {3, 5, 6}; in_dim*BITS must be 32-aligned.
    //   int3: 64*3=192; int5: 64*5=320; int6: 64*6=384.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1],
                  variants(BITS = [2, 3, 4, 5, 6, 8], suffix = "int{BITS}"))]
    fn test_dequant_gemv(dt: DType) -> TestSetup {
        if 32u32 % BITS == 0 {
            gemv_setup(iron_dequant_gemv_intBITS::kernel_ir_for(dt), BITS, 4, 256, 64, 4, TPG, dt)
        } else {
            gemv_setup(iron_dequant_gemv_intBITS::kernel_ir_for(dt), BITS, 4, 64, 32, 4, TPG, dt)
        }
    }

    // ── Perf-tuned int4_fast: 8 rows per TG ─────────────────────────────────
    // in_dim a multiple of 512, out_dim a multiple of 8, group_size 64.
    // Grid: [out_dim/8, 1, 1], TPG = 64.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int4_fast(dt: DType) -> TestSetup {
        let (out_dim, in_dim, group_size) = (8usize, 512usize, 64usize);
        gemv_setup(
            iron_dequant_gemv_int4_fast::kernel_ir_for(dt),
            4,
            out_dim,
            in_dim,
            group_size,
            (out_dim / 8) as u32,
            64,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int2_fast(dt: DType) -> TestSetup {
        let (out_dim, in_dim, group_size) = (8usize, 512usize, 64usize);
        gemv_setup(
            iron_dequant_gemv_int2_fast::kernel_ir_for(dt),
            2,
            out_dim,
            in_dim,
            group_size,
            (out_dim / 8) as u32,
            64,
            dt,
        )
    }

    #[test_kernel(dtypes = [bf16], tol = [2e-1])]
    fn test_dequant_gemv_int2_fast_qwen38(dt: DType) -> TestSetup {
        let (out_dim, in_dim, group_size) = (64usize, 5120usize, 64usize);
        gemv_setup(
            iron_dequant_gemv_int2_fast::kernel_ir_for(dt),
            2,
            out_dim,
            in_dim,
            group_size,
            (out_dim / 8) as u32,
            64,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int4_gathered(dt: DType) -> TestSetup {
        gathered_gemv_setup(dt, 16, 512, vec![15, 0, 7, 3, 12, 5, 1, 9])
    }

    #[test_kernel(dtypes = [bf16], tol = [2e-1])]
    fn test_dequant_gemv_int4_gathered_qwen38(dt: DType) -> TestSetup {
        let rows = (0..32).map(|i| ((i * 37 + 11) % 64) as u32).collect();
        gathered_gemv_setup(dt, 64, 5120, rows)
    }
}

/// New-syntax benchmarks for the dequant GEMV family. Production-ish shapes
/// (out_dim/in_dim 4096, group_size 64). bytes_moved counts the packed-weight
/// stream (dominant) + scales/biases + input + output.
pub mod kernel_benches {
    use wh_iron::{bench, core::ir::Kernel, test::*};

    use super::*;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[allow(clippy::too_many_arguments)]
    fn gb(
        kernel: Kernel,
        bits: u32,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        grid_rows: u32,
        tpg: u32,
        dt: DType,
    ) -> BenchSetup {
        let n_groups = in_dim / group_size;
        let u32_per_row = in_dim * bits as usize / 32;
        let sz = dt.size_bytes();
        let bytes =
            out_dim * u32_per_row * 4 + 2 * out_dim * n_groups * sz + in_dim * sz + out_dim * sz;
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", out_dim * u32_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d(grid_rows, 1, 1, [tpg, 1, 1])
            .bytes_moved(bytes as u64)
            // qgemv (B=1): 2 * out_dim * in_dim
            .flops(2 * out_dim as u64 * in_dim as u64)
    }

    #[bench(dtypes = [f32, f16, bf16],
            variants(BITS = [2, 3, 4, 5, 6, 8], suffix = "int{BITS}"))]
    fn bench_dequant_gemv(dt: DType) -> BenchSetup {
        gb(iron_dequant_gemv_intBITS::kernel_ir_for(dt), BITS, 4096, 4096, 64, 4096, 64, dt)
    }

    // 8-rows-per-TG fast int4: grid [out_dim/8, 1, 1], TPG 64.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int4_fast(dt: DType) -> BenchSetup {
        gb(iron_dequant_gemv_int4_fast::kernel_ir_for(dt), 4, 4096, 4096, 64, 4096 / 8, 64, dt)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int2_fast(dt: DType) -> BenchSetup {
        gb(iron_dequant_gemv_int2_fast::kernel_ir_for(dt), 2, 4096, 4096, 64, 4096 / 8, 64, dt)
    }

    #[bench(dtypes = [bf16])]
    fn bench_dequant_gemv_int4_gathered(dt: DType) -> BenchSetup {
        let matrix_rows = 4096usize;
        let candidate_rows = 32usize;
        let in_dim = 5120usize;
        let group_size = 64usize;
        let n_groups = in_dim / group_size;
        let packs_per_row = in_dim / 8;
        let rows: Vec<u32> =
            (0..candidate_rows).map(|i| ((i * 127 + 11) % matrix_rows) as u32).collect();
        let bytes = candidate_rows * (packs_per_row * 4 + 2 * n_groups * dt.size_bytes())
            + in_dim * dt.size_bytes()
            + candidate_rows * (4 + dt.size_bytes());
        BenchSetup::new(iron_dequant_gemv_int4_gathered::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("weight", matrix_rows * packs_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", matrix_rows * n_groups, dt))
            .buffer(BenchBuffer::random("biases", matrix_rows * n_groups, dt))
            .buffer(BenchBuffer::from_vec("row_indices", u32_bytes(&rows), DType::U32))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("output", candidate_rows, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d((candidate_rows / 8) as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops((2 * candidate_rows * in_dim) as u64)
    }
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Inline-dequant GEMV — `out[r] = Σ_k dequant(W[r,k]) · x[k]` — for Q8_0 (1
//! byte/weight) and Q4 (½ byte/weight) weights read straight from their split
//! buffers, so the dense attention / shared-expert projections stay quantized-
//! resident instead of being pre-expanded to f16. The decode path is bandwidth-
//! bound on these projections (cold weights ≫ L2), so halving their bytes
//! roughly halves the per-layer cost.
//!
//! Variants: plain + `_coalesced` (contiguous-word warp walk, the decode fast
//! path) + fused `_relu2` (MoE up-proj activation) / `_accum` (router-weighted
//! down-proj into the layer accumulator), the `grouped_*` forms (one dispatch
//! for N row-groups each on their own x-slice), and the Q4 `_vec` / `_2row`
//! occupancy variants. The batched MoE expert-gather forms live in
//! `kernels/moe/gather_q4.rs`. Accumulation is f32 regardless of `T`.
//!
//! ## Q8_0 block (32 values)
//!   d (f16 scale) + 32 int8 quants;  value[i] = d · q_i8[i]
//!
//! ## Split inputs (loader produces these once, resident)
//!   qs   [m_out * (k_in/32) * 8]  u32  — 32 int8/block packed as 8 LE u32
//!   d    [m_out * (k_in/32)]      f32  — per-block scale (fp16→f32)
//!   x    [k_in]                   T
//!   out  [m_out]                  T
//!
//! Dispatch (Reduction): grid (threadgroups) = [m_out, 1, 1], tg=[32,1,1].

use wh_iron::kernel;

#[kernel]
pub fn iron_gemv_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let x_base = b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                // sign-extend the byte to int8 range in the float domain
                // (avoids ambiguous integer `select` in MSL).
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                let val = d * qf;
                acc = acc + val * load(x[x_base + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Grouped Q8_0 gemv — `out[r] = Σ_k dequant(W[r,k]) · x[(r/rows_per_group)*k_in + k]`.
/// Each contiguous block of `rows_per_group` output rows reads its own
/// `k_in`-slice of `x`. Fuses the DSv4 grouped O-LoRA (8 groups × a
/// [1024,4096] Q8 slice, each on a different 4096-slice of the attention
/// output) into a SINGLE dispatch instead of 8.
#[kernel]
pub fn iron_grouped_gemv_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let x_blk = x_base + b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                acc = acc + d * qf * load(x[x_blk + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// COALESCED per-token grouped Q8 gemv — same math as `iron_grouped_gemv_q8`
/// but the warp walks the row's `u32` words contiguously (lane j, j+32, …) so
/// consecutive lanes hit consecutive addresses. The original strided by 8 u32
/// per lane (each lane owned a whole 32-int8 block), which only reached ~45% of
/// DRAM bandwidth on GB10; this coalesced pattern is the decode-GEMV fast path.
#[kernel]
pub fn iron_gemv_q8_coalesced<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let nwords = bpr * 8u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for j in range(lane, nwords, 32u32) {
        let block = j / 8u32;
        let w = j % 8u32;
        let packed = load(qs[qs_base + j]);
        let d = load(d_f32[d_base + block]).cast::<f32>();
        let x_blk = x_base + block * 32u32 + w * 4u32;
        for i in range(0u32, 4u32, 1u32) {
            let by = (packed >> (i * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            acc = acc + d * qf * load(x[x_blk + i]).cast::<f32>();
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Coalesced Q8 gemv with a fused ReLU² on the output: `out[r] = max(0, Wq·x)²`.
/// Fuses a MoE expert's `up` projection and its activation into one dispatch
/// (was gemv + a separate relu² kernel), keeping per-row occupancy.
#[kernel]
pub fn iron_gemv_q8_coalesced_relu2<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let nwords = bpr * 8u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut dot = 0.0f32;
    for j in range(lane, nwords, 32u32) {
        let block = j / 8u32;
        let w = j % 8u32;
        let packed = load(qs[qs_base + j]);
        let d = load(d_f32[d_base + block]).cast::<f32>();
        let x_blk = x_base + block * 32u32 + w * 4u32;
        for i in range(0u32, 4u32, 1u32) {
            let by = (packed >> (i * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            dot = dot + d * qf * load(x[x_blk + i]).cast::<f32>();
        }
    }
    let total = simd_sum(dot);
    if lane == 0u32 {
        let r = select(total > 0.0f32, total, 0.0f32);
        store(out[row], (r * r).cast::<T>());
    }
}

/// Coalesced Q8 gemv that SCALES + ACCUMULATES in place: `acc[r] += scale[0] ·
/// Σ_k dequant(W[r,k])·x[k]`. Lets a MoE expert's `down` projection fold its
/// router weight and sum into the layer accumulator in ONE kernel — no separate
/// scalar-broadcast upload or `fma` dispatch per expert. `scale` is a 1-element
/// device buffer (the router weight); loaded once per output row.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_gemv_q8_coalesced_accum<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut acc: Tensor<T>,
    scale: Tensor<f32>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let nwords = bpr * 8u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut dot = 0.0f32;
    for j in range(lane, nwords, 32u32) {
        let block = j / 8u32;
        let w = j % 8u32;
        let packed = load(qs[qs_base + j]);
        let d = load(d_f32[d_base + block]).cast::<f32>();
        let x_blk = x_base + block * 32u32 + w * 4u32;
        for i in range(0u32, 4u32, 1u32) {
            let by = (packed >> (i * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            dot = dot + d * qf * load(x[x_blk + i]).cast::<f32>();
        }
    }
    let total = simd_sum(dot);
    if lane == 0u32 {
        let s = load(scale[0]);
        let prev = load(acc[row]).cast::<f32>();
        store(acc[row], (prev + s * total).cast::<T>());
    }
}

// ── Q4 (4-bit) coalesced gemv family — half the weight DRAM of Q8, the decode
// bandwidth lever (decode reads cold weights: 35GB resident ≫ L2). Block 32,
// symmetric int4 in [-7,7], one f32 scale/block. qs packs 8 nibbles per u32
// (4 u32/block). Same coalesced walk + warp reduce as the Q8 variants. ──

/// Plain Q4 coalesced matvec: `out[r] = Σ_k dequant4(W[r,k]) · x[...]`.
#[kernel]
pub fn iron_gemv_q4_coalesced<T>(
    qs: Tensor<u32>,
    // f16 scales (half the bytes — the resident-weight decode/prefill path uploads
    // them as f16). Param name is historical; do NOT change to Tensor<f32> — the
    // production callers feed f16 and reading f16 bytes as f32 yields NaN.
    d_f32: Tensor<f16>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    // Multi-warp: `rows_per_tg` warps per threadgroup, each warp owns one row.
    // The single-warp gemv (rows_per_tg=1) is memory-LATENCY-bound (ncu: ~80%
    // scoreboard stalls, <50% occupancy) — packing several warps per TG keeps
    // more global loads in flight to hide that latency. rows_per_tg=1 is
    // bit-identical to the original (warp=0, lane=tid, row=tgid_x).
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32; // 4 u32 per 32-value block
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let x_base = (row / rows_per_group) * k_in;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let x_blk = x_base + block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                blk = blk + q * load(x[x_blk + i]).cast::<f32>();
            }
            dot = dot + d * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            store(out[row], total.cast::<T>());
        }
    }
}

/// Q4 GEMV, VECTORIZED weight load: each lane owns whole Q4 blocks and reads the
/// block's 4 packed words as 4 CONSECUTIVE loads → the codegen Vectorize pass
/// collapses them into one 128-bit `VectorLoad` (vs the strided scalar-u32 load,
/// which never vectorizes). 4× fewer weight-load instructions → fewer scoreboard
/// stalls (ncu: the latency-bound GEMV's actual bottleneck). Coalesced: adjacent
/// lanes read adjacent 16-byte blocks.
#[kernel]
pub fn iron_gemv_q4_vec<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 4u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut dot = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let wbase = qs_base + b * 4u32;
        // Precompute the 4 index VIDs BEFORE the loads (sdpa_decode trick) so the
        // Vectorize pass sees 4 consecutive bare Op::Load → collapses to one uint4.
        let i0 = wbase;
        let i1 = wbase + 1u32;
        let i2 = wbase + 2u32;
        let i3 = wbase + 3u32;
        let p0 = load(qs[i0]);
        let p1 = load(qs[i1]);
        let p2 = load(qs[i2]);
        let p3 = load(qs[i3]);
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let xb = x_base + b * 32u32;
        let mut acc = 0.0f32;
        for i in range(0u32, 8u32, 1u32) {
            let n0 = (p0 >> (i * 4u32)) & 0xfu32;
            acc = acc
                + (n0.cast::<f32>() - select(n0 > 7u32, 16.0f32, 0.0f32))
                    * load(x[xb + i]).cast::<f32>();
        }
        for i in range(0u32, 8u32, 1u32) {
            let n1 = (p1 >> (i * 4u32)) & 0xfu32;
            acc = acc
                + (n1.cast::<f32>() - select(n1 > 7u32, 16.0f32, 0.0f32))
                    * load(x[xb + 8u32 + i]).cast::<f32>();
        }
        for i in range(0u32, 8u32, 1u32) {
            let n2 = (p2 >> (i * 4u32)) & 0xfu32;
            acc = acc
                + (n2.cast::<f32>() - select(n2 > 7u32, 16.0f32, 0.0f32))
                    * load(x[xb + 16u32 + i]).cast::<f32>();
        }
        for i in range(0u32, 8u32, 1u32) {
            let n3 = (p3 >> (i * 4u32)) & 0xfu32;
            acc = acc
                + (n3.cast::<f32>() - select(n3 > 7u32, 16.0f32, 0.0f32))
                    * load(x[xb + 24u32 + i]).cast::<f32>();
        }
        dot = dot + d * acc;
    }
    let total = simd_sum(dot);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Q4 GEMV, 2 output rows per warp: load the shared activation `x` ONCE and run
/// TWO independent weight streams (rows 2r, 2r+1) → 2× memory-level-parallelism on
/// the latency-bound Q4 weight read (ncu: scoreboard-stalled, <50% BW), plus the x
/// read is shared (halved). Stacks with multi-warp (`rows_per_tg` warps/TG).
///
/// CONTRACT: `rows_per_group % 2 == 0` (or `rows_per_group >= m_out`, the plain
/// matvec case). `x_base` is derived from `row_a`'s group and shared by both
/// rows — that's the whole point of the kernel — so a group boundary must never
/// fall between the even/odd row pair. Odd `m_out` is safe: the dangling
/// `row_b` clamps its weight reads to `row_a` and skips its store.
#[kernel]
pub fn iron_gemv_q4_coalesced_2row<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row_a = (tgid_x * rows_per_tg + warp) * 2u32;
    let row_b = row_a + 1u32;
    if row_a < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        // Clamp row_b's bases before any load (`select` pre-evaluates both
        // arms): for odd m_out the last pair's row_b would read past qs/d_f32.
        // Clamped-to-row_a results are discarded (store guarded below).
        let row_b_ok = row_b < m_out;
        let row_b_safe = select(row_b_ok, row_b, row_a);
        let qa = row_a * bpr * 4u32;
        let qb = row_b_safe * bpr * 4u32;
        let da = row_a * bpr;
        let db = row_b_safe * bpr;
        let x_base = (row_a / rows_per_group) * k_in;
        let mut dot_a = 0.0f32;
        let mut dot_b = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let pa = load(qs[qa + j]);
            let pb = load(qs[qb + j]);
            let dda = load(d_f32[da + block]);
            let ddb = load(d_f32[db + block]);
            let xb = x_base + block * 32u32 + sub * 8u32;
            let mut acc_a = 0.0f32;
            let mut bb = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let xv = load(x[xb + i]).cast::<f32>();
                let na = (pa >> (i * 4u32)) & 0xfu32;
                let nb = (pb >> (i * 4u32)) & 0xfu32;
                acc_a = acc_a + (na.cast::<f32>() - select(na > 7u32, 16.0f32, 0.0f32)) * xv;
                bb = bb + (nb.cast::<f32>() - select(nb > 7u32, 16.0f32, 0.0f32)) * xv;
            }
            dot_a = dot_a + dda * acc_a;
            dot_b = dot_b + ddb * bb;
        }
        let ta = simd_sum(dot_a);
        let tb = simd_sum(dot_b);
        if lane == 0u32 {
            store(out[row_a], ta.cast::<T>());
            if row_b < m_out {
                store(out[row_b], tb.cast::<T>());
            }
        }
    }
}

/// M-WAY BATCHED Q4 coalesced matvec — reads each weight row's DRAM ONCE
/// and applies it to `m` independent activation vectors, producing `m`
/// independent dot products per row. Sibling of `iron_gemv_q4_coalesced`
/// (M=1); built for MTP speculative-decode BATCHED VERIFY, where γ+1
/// candidate positions each carry their own hidden state but share every
/// dense projection's weight matrix — this amortizes the weight-DRAM read
/// γ+1-fold instead of paying it once per position (the whole point: decode
/// is bandwidth-bound on these cold Q4 weights, see the module doc).
///
/// Nibble unpacking is done ONCE per word into a small `qv` register stack
/// and reused across all `m` activations — dequant is ALU-bound (~56 ops/
/// word vs 1 load, see `iron_gemv_q4_coalesced_relu2`'s doc), so sharing it
/// m-fold (instead of re-unpacking per activation) is the actual lever here,
/// not just the weight-DRAM saving.
///
/// `x` is `[m, n_groups*k_in]` (each candidate's own activation row,
/// row-major — same per-row grouping convention as the M=1 kernel's
/// `rows_per_group`); `out` is `[m, m_out]`.
///
/// BIT-EXACTNESS vs calling `iron_gemv_q4_coalesced` once per `t in 0..m`:
/// for a fixed `t`, this kernel's `j`-loop order (`range(lane, nwords,
/// 32)`), per-word 8-nibble unpack order, and `d*blk` word-accumulate order
/// are IDENTICAL to the M=1 kernel's — only the accumulator storage
/// (`stack_alloc` slot `t` vs a plain register) differs, which does not
/// change the floating-point operation sequence. `simd_sum`'s reduction
/// tree is deterministic for a fixed lane/warp geometry, so each `t`'s
/// final `total` reproduces the M=1 kernel's result bit-for-bit — required
/// for the spec-decode correctness bar (argmax must not drift from the
/// sequential-verify path).
///
/// CONTRACT / IMPLEMENTATION NOTE — compile-time `M` variants, not a
/// runtime loop bound:
///
/// A first version made `m` an ordinary `#[constexpr]` (a dispatch-time
/// scalar binding, like `k_in`/`m_out`) and looped `range(0u32, m, 1u32)`
/// over the `acc`/`qv` `stack_alloc` arrays. It was CORRECT (passes the
/// oracle tests bit-for-bit within the standard per-dtype tolerance) but
/// LOST THE GATE HARD: wall time scaled ~linearly with `m` instead of
/// amortizing (measured ratio vs single-M 1.9x @ m=2 up to 4.6-5.3x @ m=5,
/// i.e. barely faster than `m` sequential dispatches — see the
/// `qwen35_spec_gemv_multi_microbench` git history for the raw numbers).
/// Root cause: `range(0, m, 1)` with a NON-literal (dispatch-time-bound)
/// trip count cannot be unrolled at codegen, so the compiler can't prove a
/// static iteration count and can't promote `stack_alloc`'s per-t slots to
/// real registers — every `stack_load`/`stack_store` on `acc`/`qv` inside
/// that loop round-trips through actual (slow) local memory, `m`-fold. The
/// exact failure mode `iron_gated_delta_prep_chunk_fast`'s `NPT`
/// compile-time-variant doc warns about for its own `state_reg`/`k_cache`
/// arrays ("this array is fully unrollable and register-promotable" only
/// once the bound is a genuine compile-time constant).
///
/// Fix: `#[kernel(variants(...))]` (same mechanism as the `d128_*` GDN fast
/// kernels) generates one FULLY SPECIALIZED module per `M` value, each with
/// `M` baked in as a literal — `range(0, M, 1)` unrolls, `stack_alloc`'s
/// `acc`/`qv` slots become real per-t registers, and the weight-DRAM read
/// is genuinely amortized `M`-fold with the per-candidate FMA cost hidden
/// behind it (see the microbench for the after numbers). `M∈{2,3,4,5}`
/// covers the γ+1 range this was built for (γ∈{2,3,4}); the generated
/// modules are `iron_gemv_q4_multi_cand2` .. `iron_gemv_q4_multi_cand5`.
///
/// BIT-EXACTNESS vs calling `iron_gemv_q4_coalesced` once per `t in 0..M`:
/// for a fixed `t`, this kernel's `j`-loop order (`range(lane, nwords,
/// 32)`), per-word 8-nibble unpack order, and `d*blk` word-accumulate order
/// are IDENTICAL to the M=1 kernel's — only the accumulator storage
/// (register slot `t` vs a plain register) differs, which does not change
/// the floating-point operation sequence. `simd_sum`'s reduction tree is
/// deterministic for a fixed lane/warp geometry, so each `t`'s final
/// `total` reproduces the M=1 kernel's result bit-for-bit — required for
/// the spec-decode correctness bar (argmax must not drift from the
/// sequential-verify path).
#[kernel(variants(
    M = [2u32, 3u32, 4u32, 5u32],
    suffix = "cand{M}"
))]
#[allow(clippy::too_many_arguments)]
pub fn iron_gemv_q4_multi<T>(
    qs: Tensor<u32>,
    // f16 scales — see `iron_gemv_q4_coalesced`'s doc; same contract.
    d_f32: Tensor<f16>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let n_groups = m_out / rows_per_group;
        let x_stride = n_groups * k_in; // per-candidate activation row length
        let x_base = (row / rows_per_group) * k_in;

        // `M` is now a codegen-time literal (per variant) — this array is
        // fully unrollable/register-promotable, same as `NPT`'s `state_reg`
        // in the GDN fast kernel this pattern was copied from.
        stack_alloc("acc", M, "f32");
        for t in range(0u32, M, 1u32) {
            stack_store("acc", t, 0.0f32);
        }
        stack_alloc("qv", 8, "f32");
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let kk = block * 32u32 + sub * 8u32;
            // Unpack this word's 8 nibbles ONCE, reused by every candidate.
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                stack_store("qv", i, q);
            }
            for t in range(0u32, M, 1u32) {
                let xt_base = t * x_stride + x_base + kk;
                let mut blk = 0.0f32;
                for i in range(0u32, 8u32, 1u32) {
                    let qi = stack_load("qv", i);
                    blk = blk + qi * load(x[xt_base + i]).cast::<f32>();
                }
                let prev = stack_load("acc", t);
                stack_store("acc", t, prev + d * blk);
            }
        }
        for t in range(0u32, M, 1u32) {
            let total = simd_sum(stack_load("acc", t));
            if lane == 0u32 {
                store(out[t * m_out + row], total.cast::<T>());
            }
        }
    }
}

/// Q4 coalesced matvec with fused ReLU² (MoE expert up).
/// Multi-warp: `rows_per_tg` warps per threadgroup, each warp owns one output
/// row. The single-warp form (`rows_per_tg=1`) is memory-LATENCY-bound (small
/// shared-expert matrices: ~50% BW); packing several warps per TG keeps more
/// global Q4 loads in flight to hide that latency. `rows_per_tg=1` is
/// bit-identical to the original (warp=0, lane=tid, row=tgid_x).
#[kernel]
pub fn iron_gemv_q4_coalesced_relu2<T>(
    qs: Tensor<u32>,
    // f16 scales (half the bytes — the resident-weight decode/prefill path uploads
    // them as f16). Param name is historical; do NOT change to Tensor<f32> — the
    // production callers feed f16 and reading f16 bytes as f32 yields NaN.
    d_f32: Tensor<f16>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let x_base = (row / rows_per_group) * k_in;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let x_blk = x_base + block * 32u32 + sub * 8u32;
            // Scale `d` is constant across the block's 8 nibbles — factor it OUT of
            // the inner loop (`d·Σ q·x` not `Σ d·q·x`): the dequant is ALU-bound
            // (~56 ALU ops/word vs 1 load), so dropping 7 mul/word is a real win.
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                blk = blk + q * load(x[x_blk + i]).cast::<f32>();
            }
            dot = dot + d * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            let r = select(total > 0.0f32, total, 0.0f32);
            store(out[row], (r * r).cast::<T>());
        }
    }
}

/// Q4 coalesced matvec, scale + accumulate in place (MoE expert down).
/// Multi-warp (`rows_per_tg` warps/TG, one output row each) — same latency-
/// hiding rationale as the relu2 variant; `rows_per_tg=1` is bit-identical.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_gemv_q4_coalesced_accum<T>(
    qs: Tensor<u32>,
    // f16 scales (half the bytes — the resident-weight decode/prefill path uploads
    // them as f16). Param name is historical; do NOT change to Tensor<f32> — the
    // production callers feed f16 and reading f16 bytes as f32 yields NaN.
    d_f32: Tensor<f16>,
    x: Tensor<T>,
    mut acc: Tensor<T>,
    scale: Tensor<f32>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let x_base = (row / rows_per_group) * k_in;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let x_blk = x_base + block * 32u32 + sub * 8u32;
            // Scale `d` is constant across the block's 8 nibbles — factor it OUT of
            // the inner loop (`d·Σ q·x` not `Σ d·q·x`): the dequant is ALU-bound
            // (~56 ALU ops/word vs 1 load), so dropping 7 mul/word is a real win.
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                blk = blk + q * load(x[x_blk + i]).cast::<f32>();
            }
            dot = dot + d * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            let s = load(scale[0]);
            let prev = load(acc[row]).cast::<f32>();
            store(acc[row], (prev + s * total).cast::<T>());
        }
    }
}

/// BATCHED grouped Q8_0 gemv — iron_grouped_gemv_q8 over `n_tokens` rows in
/// ONE dispatch (grid z/y = token). Prefill O-LoRA looped the per-token
/// grouped gemv N times; this folds it. x is [n_tokens, n_groups*k_in],
/// out is [n_tokens, m_out]; n_groups = m_out/rows_per_group.
/// Grid (Reduction): [m_out, n_tokens, 1], tg=[32,1,1].
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_grouped_gemv_q8_rows<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let token = tgid_y;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let n_groups = m_out / rows_per_group;
    let x_base = token * n_groups * k_in + (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let x_blk = x_base + b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                acc = acc + d * qf * load(x[x_blk + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[token * m_out + row], total.cast::<T>());
    }
}

/// TOKEN-TILED grouped Q8 gemv — the amortized fix for the prefill O-LoRA-A
/// hotspot. `iron_grouped_gemv_q8_rows` re-reads each weight row from DRAM
/// once PER TOKEN (no amortization); at N=512 that's the single biggest
/// op in the attention block (~47 ms/layer). Here each threadgroup owns one
/// output row and a TILE of `tokens_per_tile` tokens: the Q8 weight block
/// (d + 8 packed = 32 int8) is loaded ONCE and applied to all T tokens, so
/// the weight DRAM traffic drops T-fold. T accumulators in a register stack.
/// grid (threadgroups) = [m_out, ceil(n_tokens/T), 1], threadgroup [32,1,1].
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_grouped_gemv_q8_rows_tiled<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] n_tokens: u32,
) {
    // Tile size is a compile-time LITERAL (stack_alloc can't size on a
    // constexpr param — codegen emits the array decl before the constant
    // is bound). T=8 tokens/tile → 8-fold weight-DRAM amortization.
    let tokens_per_tile = 8u32;
    let row = tgid_x;
    let ttile = tgid_y;
    let lane = tid;
    let tok0 = ttile * tokens_per_tile;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let n_groups = m_out / rows_per_group;
    let group_off = (row / rows_per_group) * k_in;

    stack_alloc("acc", 8, "f32");
    for t in range(0u32, tokens_per_tile, 1u32) {
        stack_store("acc", t, 0.0f32);
    }
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let blk = b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]); // 4 int8 weights, read ONCE
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                let wv = d * qf; // weight value, reused across the T tokens
                let kk = blk + w * 4u32 + i;
                for t in range(0u32, tokens_per_tile, 1u32) {
                    let tok = tok0 + t;
                    if tok < n_tokens {
                        let xb = tok * n_groups * k_in + group_off + kk;
                        let prev = stack_load("acc", t);
                        stack_store("acc", t, prev + wv * load(x[xb]).cast::<f32>());
                    }
                }
            }
        }
    }
    for t in range(0u32, tokens_per_tile, 1u32) {
        let tok = tok0 + t;
        let total = simd_sum(stack_load("acc", t));
        if (lane == 0u32) & (tok < n_tokens) {
            store(out[tok * m_out + row], total.cast::<T>());
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{
        iron_gemv_q4_multi_cand2,
        iron_gemv_q4_multi_cand3,
        iron_gemv_q4_multi_cand4,
        iron_gemv_q4_multi_cand5,
    };
    use crate::utils::pack_f32;

    /// Reference Q4 quantizer — mirrors `iron_ops::quantize_q4` /
    /// `dequant_q4::kernel_tests::quantize_q4` exactly: signed 4-bit
    /// (range [-7,7], symmetric), per-32-block scale = amax/7, 4 u32
    /// words/block, 8 nibbles/word.
    fn quantize_q4(w: &[f32], m_out: usize, k_in: usize) -> (Vec<u32>, Vec<f32>) {
        let bpr = k_in / 32;
        let mut qs = vec![0u32; m_out * bpr * 4];
        let mut scales = vec![0f32; m_out * bpr];
        for r in 0..m_out {
            for b in 0..bpr {
                let base = r * k_in + b * 32;
                let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
                let d = amax / 7.0;
                scales[r * bpr + b] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for word in 0..4 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                        packed |= ((q as u32) & 0xf) << (i * 4);
                    }
                    qs[r * bpr * 4 + b * 4 + word] = packed;
                }
            }
        }
        (qs, scales)
    }

    /// CPU oracle: `out[t, r] = Σ_k dequant4(W[r,k]) · x[t, k]` for
    /// `t in 0..cand`, `r in 0..m_out`. No row grouping (matches the
    /// dense-projection call sites `iron_gemv_q4_multi` targets — the
    /// GDN/full-attn/FFN qkv+ffn projections in `qwen35.rs` never use
    /// `rows_per_group < m_out`).
    fn cpu_gemv_multi(
        qs: &[u32],
        scales_f32: &[f32],
        x: &[f32],
        m_out: usize,
        k_in: usize,
        cand: usize,
    ) -> Vec<f32> {
        let bpr = k_in / 32;
        let mut out = vec![0f32; cand * m_out];
        for r in 0..m_out {
            for b in 0..bpr {
                let d = scales_f32[r * bpr + b];
                for word in 0..4 {
                    let packed = qs[r * bpr * 4 + b * 4 + word];
                    for i in 0..8 {
                        let nib = (packed >> (i * 4)) & 0xf;
                        let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                        let qf = q as f32;
                        let k = b * 32 + word * 8 + i;
                        for t in 0..cand {
                            out[t * m_out + r] += d * qf * x[t * k_in + k];
                        }
                    }
                }
            }
        }
        out
    }

    /// Round a value through the storage dtype `dt` (f32 = no-op). The GPU
    /// kernel loads `x` from a `dt`-typed buffer (`load(x[..]).cast::<f32>()`)
    /// — for f16/bf16 that load itself is lossy, so the CPU oracle must see
    /// the SAME rounded value or it's comparing against numbers the kernel
    /// never actually had (this is what made the initial f16/bf16 cells
    /// mismatch by a full ULP-scale margin: the oracle was using un-rounded
    /// f32 `x`). Mirrors the existing `scales_f16` round-trip below.
    fn round_trip(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn setup(
        ir: wh_iron::core::ir::Kernel,
        m_out: usize,
        k_in: usize,
        cand: usize,
        dt: DType,
    ) -> TestSetup {
        let wn = m_out * k_in;
        let weights: Vec<f32> = (0..wn).map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.7).collect();
        let (qs, scales) = quantize_q4(&weights, m_out, k_in);
        let scales_f16: Vec<f32> =
            scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();

        let xn = cand * k_in;
        let x: Vec<f32> = (0..xn).map(|i| (i as f32 * 0.029 + 0.11).cos() * 0.9).collect();
        let x_rounded: Vec<f32> = x.iter().map(|&v| round_trip(v, dt)).collect();

        let expected = cpu_gemv_multi(&qs, &scales_f16, &x_rounded, m_out, k_in, cand);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|v| v.to_le_bytes()).collect();

        TestSetup::new(ir)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("d_f32", pack_f32(&scales_f16, DType::F16), DType::F16))
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::zeros("out", cand * m_out, dt))
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", m_out as u32)
            .constexpr("rows_per_tg", 1u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
    }

    // γ+1 candidate counts the batched-verify path actually dispatches
    // (γ ∈ {2,3,4} ⇒ cand ∈ {3,4,5}), plus cand=2. Each cell hits its own
    // compile-time-specialized `#[kernel(variants(...))]` module (see the
    // kernel's doc for why M must be a codegen-time literal, not a runtime
    // constexpr, to actually win the gate).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_iron_gemv_q4_multi_cand2(dt: DType) -> TestSetup {
        setup(iron_gemv_q4_multi_cand2::kernel_ir_for(dt), 64, 128, 2, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_iron_gemv_q4_multi_cand3(dt: DType) -> TestSetup {
        setup(iron_gemv_q4_multi_cand3::kernel_ir_for(dt), 64, 128, 3, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_iron_gemv_q4_multi_cand4(dt: DType) -> TestSetup {
        setup(iron_gemv_q4_multi_cand4::kernel_ir_for(dt), 64, 128, 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_iron_gemv_q4_multi_cand5(dt: DType) -> TestSetup {
        setup(iron_gemv_q4_multi_cand5::kernel_ir_for(dt), 64, 128, 5, dt)
    }

    // Realistic-K sanity cell (Qwen3.6-27B hidden=5120), smaller m_out to
    // keep host-side oracle/test time reasonable — catches any block/word-
    // stride bug that a k_in=128 (4-block) shape is too small to expose.
    // f32 tol widened 1e-4->3e-4 vs the small-k cells above: 160 blocks/row
    // (20x the k_in=128 cells' 8) means 20x more f32 add-order terms for
    // ULP-level rounding to accumulate across — same "wider reduction needs
    // looser tol" rationale documented on the gated-delta fast-kernel tests.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [3e-4, 5e-3, 5e-2])]
    fn test_iron_gemv_q4_multi_wide_k(dt: DType) -> TestSetup {
        setup(iron_gemv_q4_multi_cand3::kernel_ir_for(dt), 32, 5120, 3, dt)
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::{
        iron_gemv_q4_coalesced,
        iron_gemv_q4_multi_cand2,
        iron_gemv_q4_multi_cand3,
        iron_gemv_q4_multi_cand4,
        iron_gemv_q4_multi_cand5,
        iron_gemv_q8,
        iron_grouped_gemv_q8,
        iron_grouped_gemv_q8_rows,
        iron_grouped_gemv_q8_rows_tiled,
    };

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q8(dt: DType) -> BenchSetup {
        let m_out = 4096usize;
        let k_in = 8192usize;
        let bpr = k_in / 32;
        BenchSetup::new(iron_gemv_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + k_in * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let bpr = k_in / 32;
        BenchSetup::new(iron_grouped_gemv_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", 8 * k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + 8 * k_in * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8_rows(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let n_tokens = 256usize;
        let n_groups = m_out / 1024;
        let bpr = k_in / 32;
        BenchSetup::new(iron_grouped_gemv_q8_rows::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", n_tokens * n_groups * k_in, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .grid_3d(m_out as u32, n_tokens as u32, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + n_tokens * n_groups * k_in * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8_rows_tiled(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let n_tokens = 256usize;
        let tokens_per_tile = 8usize;
        let n_groups = m_out / 1024;
        let bpr = k_in / 32;
        BenchSetup::new(iron_grouped_gemv_q8_rows_tiled::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", n_tokens * n_groups * k_in, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .constexpr("n_tokens", n_tokens as u32)
            .grid_3d(m_out as u32, (n_tokens as u32).div_ceil(tokens_per_tile as u32), 1, [
                32, 1, 1,
            ])
            .bytes_moved((m_out * bpr * 36 + n_tokens * n_groups * k_in * dt.size_bytes()) as u64)
    }

    // ── iron_gemv_q4_multi microbench: F-85 "final lever" gate (batched
    // MTP verify). Compares wall time for `cand` (γ+1) candidate columns
    // in ONE dispatch against `iron_gemv_q4_coalesced` (cand=1, the M=1
    // production decode kernel) run once — the gate is `T_multi(cand) ≤
    // ~1.4 × T_single` (≥70% of single-M GB/s), because the weight-DRAM
    // read is amortized `cand`-fold while the tiny per-candidate x reads
    // are ~free by comparison. Four dominant qwen3.6-27B decode shapes:
    // ffn_gate/up, ffn_down, full-attn wq, gdn wqkv (all Q4-resident,
    // f16-scale, no row grouping — see `qwen35.rs::Q4Dense::gemv`). ──

    fn setup_q4_coalesced(dt: DType, m_out: usize, k_in: usize) -> BenchSetup {
        let bpr = k_in / 32;
        BenchSetup::new(iron_gemv_q4_coalesced::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 4, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F16))
            .buffer(BenchBuffer::random("x", k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", m_out as u32)
            .constexpr("rows_per_tg", 1u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 18 + k_in * dt.size_bytes()) as u64)
    }

    fn setup_q4_multi(dt: DType, m_out: usize, k_in: usize, cand: usize) -> BenchSetup {
        let bpr = k_in / 32;
        // `cand` picks the compile-time-specialized `#[kernel(variants(...))]`
        // module (see `iron_gemv_q4_multi`'s doc) — there is no generic
        // runtime-`m` entry point anymore.
        let ir = match cand {
            2 => iron_gemv_q4_multi_cand2::kernel_ir_for(dt),
            3 => iron_gemv_q4_multi_cand3::kernel_ir_for(dt),
            4 => iron_gemv_q4_multi_cand4::kernel_ir_for(dt),
            5 => iron_gemv_q4_multi_cand5::kernel_ir_for(dt),
            _ => panic!(
                "setup_q4_multi: no iron_gemv_q4_multi_cand{cand} variant (cand must be 2..=5)"
            ),
        };
        BenchSetup::new(ir)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 4, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F16))
            .buffer(BenchBuffer::random("x", cand * k_in, dt))
            .buffer(BenchBuffer::zeros("out", cand * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", m_out as u32)
            .constexpr("rows_per_tg", 1u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 18 + cand * k_in * dt.size_bytes()) as u64)
    }

    // -- ffn_gate/up [17408,5120] --
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_coalesced_ffn_gate_up(dt: DType) -> BenchSetup {
        setup_q4_coalesced(dt, 17408, 5120)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_gate_up_c2(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 17408, 5120, 2)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_gate_up_c3(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 17408, 5120, 3)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_gate_up_c4(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 17408, 5120, 4)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_gate_up_c5(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 17408, 5120, 5)
    }

    // -- ffn_down [5120,17408] --
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_coalesced_ffn_down(dt: DType) -> BenchSetup {
        setup_q4_coalesced(dt, 5120, 17408)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_down_c2(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 5120, 17408, 2)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_down_c3(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 5120, 17408, 3)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_down_c4(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 5120, 17408, 4)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_ffn_down_c5(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 5120, 17408, 5)
    }

    // -- full-attn wq [12288,5120] (2*n_head*head_dim = 2*24*256) --
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_coalesced_full_wq(dt: DType) -> BenchSetup {
        setup_q4_coalesced(dt, 12288, 5120)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_full_wq_c2(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 12288, 5120, 2)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_full_wq_c3(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 12288, 5120, 3)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_full_wq_c4(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 12288, 5120, 4)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_full_wq_c5(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 12288, 5120, 5)
    }

    // -- gdn wqkv [10240,5120] (key_dim*2+value_dim = 16*128*2+48*128) --
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_coalesced_gdn_wqkv(dt: DType) -> BenchSetup {
        setup_q4_coalesced(dt, 10240, 5120)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_gdn_wqkv_c2(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 10240, 5120, 2)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_gdn_wqkv_c3(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 10240, 5120, 3)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_gdn_wqkv_c4(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 10240, 5120, 4)
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q4_multi_gdn_wqkv_c5(dt: DType) -> BenchSetup {
        setup_q4_multi(dt, 10240, 5120, 5)
    }
}

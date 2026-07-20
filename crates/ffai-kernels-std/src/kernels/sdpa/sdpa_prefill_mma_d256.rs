//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MMA (simdgroup-matrix) prefill-append SDPA for `head_dim == 256`.
//!
//! F-85 attention-attribution follow-up: Stage A profiling
//! (FFAI_OP_ATTRIB diagnostic, ffai-rebaseline @ 592b0de) showed the
//! attention mixer's share of prefill GPU time growing from ~25% (T=4096)
//! to ~54% (T=16384) to ~70% (T=32768) while GDN and MoE per-chunk costs
//! stayed flat, and per-call attn_mixer time grew super-linearly with T --
//! consistent with `ffai_sdpa_prefill_qtiled_d256`'s pure scalar
//! `simd_sum` dot products (no MMA) paying an increasing matmul-shaped
//! compute cost as the causal K/V prefix grows, work that a matrix
//! coprocessor is built to absorb instead.
//! This kernel keeps that scalar kernel's exact geometry contract (BQ=32
//! tiling generalised from its BQ=16, BK=8 KV block, causal masking,
//! GQA via `heads_per_group`, ragged `n_query` guard, and the append
//! regime's `base_kv` prefix / `kv_stride` allocation-depth separation)
//! and replaces the Q.K / P.V scalar dot products with Apple 8x8
//! simdgroup-matrix fragments (see `ffai_sdpa_prefill_mma`, the existing
//! head_dim=128 MMA sibling, for the frag-lane mapping this borrows).
//!
//! ## Geometry
//!
//! TPG = 128 (4 simdgroups x 32 lanes). `BQ_SG = 8` query rows per
//! simdgroup (one full 8x8 M-fragment row-count, so each simdgroup owns
//! exactly one frag's worth of query rows along M) x 4 simdgroups = `BQ
//! = 32` rows per threadgroup -- double the scalar kernel's BQ=16,
//! because an MMA fragment's M dimension is fixed at 8 and BQ_SG=4 (the
//! scalar kernel's per-simdgroup row count) does not divide evenly into
//! an 8-row fragment. `BK = 8` (`K_CHUNKS = BK / 8 = 1`): one KV block is
//! exactly one fragment's worth of K/V rows, so there is only one
//! S/P/K^T/V fragment set per K-block (no k_chunk-pair bookkeeping like
//! the head_dim=128 sibling's BK=16/2-chunks). `head_dim = 256` needs 32
//! d_frags (256 / 8) -- double the d128 kernel's 16 -- for both the Q
//! preload/O accumulator and the Q.K^T / P.V matmul unrolling.
//!
//! `kv_ld = head_dim + 8 = 264`: the same bank-skew pad the d128 MMA
//! kernel uses on its column-major K^T reads, kept here for the same
//! reason (avoids a bank-conflict pattern on the transposed threadgroup
//! read). Threadgroup budget at BK=8: `2 * BK * kv_ld * sizeof(T)` =
//! 16896 B for T=f32 (8*264*4*2), 8448 B for f16/bf16 -- both comfortably
//! under Apple's 32 KiB/threadgroup ceiling (the GDN WY-scan kernel hit
//! that ceiling exactly at Dv=Dk=128; this is ~half that at its worst
//! case). BK=16 (matching the d128 sibling, reusing each loaded KV
//! block across twice as many query rows before advancing) was
//! evaluated but not shipped in this pass: at BK=16 the f32
//! specialization needs `2*16*264*4` = 33792 B, over the 32 KiB
//! ceiling by ~1 KiB; f16/bf16 fit (16896 B) but bumping BK is a
//! separate, dtype-conditional follow-up, not bundled with this
//! MMA-vs-scalar isolation change.
//!
//! ## Causal / append / ragged semantics (unchanged from the scalar
//! kernel this replaces)
//!
//! Query row `r` (0-indexed in the chunk) has absolute position
//! `base_kv + r` and, when causal, attends `[0, base_kv + r + 1)`; the
//! non-causal case attends `[0, base_kv + n_query)` for every row. K rows
//! past the logical length are masked to `-inf` after loading (the
//! cooperative load stays uniform); the physical load index is clamped
//! to `kv_stride - 1` to avoid reading past the cache allocation (masked
//! data is always discarded). A ragged final Q-tile (`n_query` not a
//! multiple of `BQ = 32`) is guarded at the OUTPUT store only -- guard
//! rows still run the full compute (uniform barrier count across the
//! threadgroup) but never write.

use ffai_kernels::kernel;

#[kernel]
pub fn ffai_sdpa_prefill_mma_d256<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] base_kv: u32,
    #[constexpr] n_query: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] causal: u32,
    #[constexpr] scale: f32,
) {
    let q_tile = tgid_x;
    let q_head = tgid_y;
    let kv_head = q_head / heads_per_group;
    let lane = simd_lane;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + lane;
    // ── 8x8 frag lane mapping (Apple steel_gemm layout, same as
    // ffai_sdpa_prefill_mma) ──
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    let bq = 32u32;
    let bq_sg = 8u32;
    let bk = 8u32;
    let scale_log2 = scale * 1.4426950408889634f32;
    let n_kv_logical = base_kv + n_query;
    let kv_head_base = kv_head * kv_stride * head_dim;
    let q_tile_first = q_tile * bq + sg * bq_sg;
    // Guard row for out-of-range loads (ragged final tile): clamp to the
    // last valid query so every lane still reads/computes something
    // in-bounds; the store at the end is what actually gets skipped.
    let qrow_c = select(q_tile_first + fm < n_query, q_tile_first + fm, n_query - 1u32);
    let row_base = (qrow_c * n_q_heads + q_head) * head_dim;
    // kv_ld = head_dim + 8 bank-skew pad (see module doc).
    let kv_ld = 264u32;
    threadgroup_alloc("tg_ks", 2112, T);
    threadgroup_alloc("tg_vs", 2112, T);

    let q_f0 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f0, 0, load(q[row_base + 0u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f0, 1, load(q[row_base + 0u32 + fn1]).cast::<T>());
    let q_f1 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1, 0, load(q[row_base + 8u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1, 1, load(q[row_base + 8u32 + fn1]).cast::<T>());
    let q_f2 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f2, 0, load(q[row_base + 16u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f2, 1, load(q[row_base + 16u32 + fn1]).cast::<T>());
    let q_f3 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f3, 0, load(q[row_base + 24u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f3, 1, load(q[row_base + 24u32 + fn1]).cast::<T>());
    let q_f4 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f4, 0, load(q[row_base + 32u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f4, 1, load(q[row_base + 32u32 + fn1]).cast::<T>());
    let q_f5 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f5, 0, load(q[row_base + 40u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f5, 1, load(q[row_base + 40u32 + fn1]).cast::<T>());
    let q_f6 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f6, 0, load(q[row_base + 48u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f6, 1, load(q[row_base + 48u32 + fn1]).cast::<T>());
    let q_f7 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f7, 0, load(q[row_base + 56u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f7, 1, load(q[row_base + 56u32 + fn1]).cast::<T>());
    let q_f8 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f8, 0, load(q[row_base + 64u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f8, 1, load(q[row_base + 64u32 + fn1]).cast::<T>());
    let q_f9 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f9, 0, load(q[row_base + 72u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f9, 1, load(q[row_base + 72u32 + fn1]).cast::<T>());
    let q_fa = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fa, 0, load(q[row_base + 80u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fa, 1, load(q[row_base + 80u32 + fn1]).cast::<T>());
    let q_fb = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fb, 0, load(q[row_base + 88u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fb, 1, load(q[row_base + 88u32 + fn1]).cast::<T>());
    let q_fc = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fc, 0, load(q[row_base + 96u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fc, 1, load(q[row_base + 96u32 + fn1]).cast::<T>());
    let q_fd = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fd, 0, load(q[row_base + 104u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fd, 1, load(q[row_base + 104u32 + fn1]).cast::<T>());
    let q_fe = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_fe, 0, load(q[row_base + 112u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_fe, 1, load(q[row_base + 112u32 + fn1]).cast::<T>());
    let q_ff = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_ff, 0, load(q[row_base + 120u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_ff, 1, load(q[row_base + 120u32 + fn1]).cast::<T>());
    let q_f10 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f10, 0, load(q[row_base + 128u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f10, 1, load(q[row_base + 128u32 + fn1]).cast::<T>());
    let q_f11 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f11, 0, load(q[row_base + 136u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f11, 1, load(q[row_base + 136u32 + fn1]).cast::<T>());
    let q_f12 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f12, 0, load(q[row_base + 144u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f12, 1, load(q[row_base + 144u32 + fn1]).cast::<T>());
    let q_f13 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f13, 0, load(q[row_base + 152u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f13, 1, load(q[row_base + 152u32 + fn1]).cast::<T>());
    let q_f14 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f14, 0, load(q[row_base + 160u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f14, 1, load(q[row_base + 160u32 + fn1]).cast::<T>());
    let q_f15 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f15, 0, load(q[row_base + 168u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f15, 1, load(q[row_base + 168u32 + fn1]).cast::<T>());
    let q_f16 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f16, 0, load(q[row_base + 176u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f16, 1, load(q[row_base + 176u32 + fn1]).cast::<T>());
    let q_f17 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f17, 0, load(q[row_base + 184u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f17, 1, load(q[row_base + 184u32 + fn1]).cast::<T>());
    let q_f18 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f18, 0, load(q[row_base + 192u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f18, 1, load(q[row_base + 192u32 + fn1]).cast::<T>());
    let q_f19 = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f19, 0, load(q[row_base + 200u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f19, 1, load(q[row_base + 200u32 + fn1]).cast::<T>());
    let q_f1a = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1a, 0, load(q[row_base + 208u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1a, 1, load(q[row_base + 208u32 + fn1]).cast::<T>());
    let q_f1b = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1b, 0, load(q[row_base + 216u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1b, 1, load(q[row_base + 216u32 + fn1]).cast::<T>());
    let q_f1c = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1c, 0, load(q[row_base + 224u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1c, 1, load(q[row_base + 224u32 + fn1]).cast::<T>());
    let q_f1d = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1d, 0, load(q[row_base + 232u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1d, 1, load(q[row_base + 232u32 + fn1]).cast::<T>());
    let q_f1e = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1e, 0, load(q[row_base + 240u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1e, 1, load(q[row_base + 240u32 + fn1]).cast::<T>());
    let q_f1f = simdgroup_alloc::<T, 8, 8>();
    simdgroup_elem_store(q_f1f, 0, load(q[row_base + 248u32 + fn0]).cast::<T>());
    simdgroup_elem_store(q_f1f, 1, load(q[row_base + 248u32 + fn1]).cast::<T>());

    let o_f0 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f0, 0, 0.0f32);
    simdgroup_elem_store(o_f0, 1, 0.0f32);
    let o_f1 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1, 0, 0.0f32);
    simdgroup_elem_store(o_f1, 1, 0.0f32);
    let o_f2 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f2, 0, 0.0f32);
    simdgroup_elem_store(o_f2, 1, 0.0f32);
    let o_f3 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f3, 0, 0.0f32);
    simdgroup_elem_store(o_f3, 1, 0.0f32);
    let o_f4 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f4, 0, 0.0f32);
    simdgroup_elem_store(o_f4, 1, 0.0f32);
    let o_f5 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f5, 0, 0.0f32);
    simdgroup_elem_store(o_f5, 1, 0.0f32);
    let o_f6 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f6, 0, 0.0f32);
    simdgroup_elem_store(o_f6, 1, 0.0f32);
    let o_f7 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f7, 0, 0.0f32);
    simdgroup_elem_store(o_f7, 1, 0.0f32);
    let o_f8 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f8, 0, 0.0f32);
    simdgroup_elem_store(o_f8, 1, 0.0f32);
    let o_f9 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f9, 0, 0.0f32);
    simdgroup_elem_store(o_f9, 1, 0.0f32);
    let o_fa = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fa, 0, 0.0f32);
    simdgroup_elem_store(o_fa, 1, 0.0f32);
    let o_fb = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fb, 0, 0.0f32);
    simdgroup_elem_store(o_fb, 1, 0.0f32);
    let o_fc = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fc, 0, 0.0f32);
    simdgroup_elem_store(o_fc, 1, 0.0f32);
    let o_fd = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fd, 0, 0.0f32);
    simdgroup_elem_store(o_fd, 1, 0.0f32);
    let o_fe = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_fe, 0, 0.0f32);
    simdgroup_elem_store(o_fe, 1, 0.0f32);
    let o_ff = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_ff, 0, 0.0f32);
    simdgroup_elem_store(o_ff, 1, 0.0f32);
    let o_f10 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f10, 0, 0.0f32);
    simdgroup_elem_store(o_f10, 1, 0.0f32);
    let o_f11 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f11, 0, 0.0f32);
    simdgroup_elem_store(o_f11, 1, 0.0f32);
    let o_f12 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f12, 0, 0.0f32);
    simdgroup_elem_store(o_f12, 1, 0.0f32);
    let o_f13 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f13, 0, 0.0f32);
    simdgroup_elem_store(o_f13, 1, 0.0f32);
    let o_f14 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f14, 0, 0.0f32);
    simdgroup_elem_store(o_f14, 1, 0.0f32);
    let o_f15 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f15, 0, 0.0f32);
    simdgroup_elem_store(o_f15, 1, 0.0f32);
    let o_f16 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f16, 0, 0.0f32);
    simdgroup_elem_store(o_f16, 1, 0.0f32);
    let o_f17 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f17, 0, 0.0f32);
    simdgroup_elem_store(o_f17, 1, 0.0f32);
    let o_f18 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f18, 0, 0.0f32);
    simdgroup_elem_store(o_f18, 1, 0.0f32);
    let o_f19 = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f19, 0, 0.0f32);
    simdgroup_elem_store(o_f19, 1, 0.0f32);
    let o_f1a = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1a, 0, 0.0f32);
    simdgroup_elem_store(o_f1a, 1, 0.0f32);
    let o_f1b = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1b, 0, 0.0f32);
    simdgroup_elem_store(o_f1b, 1, 0.0f32);
    let o_f1c = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1c, 0, 0.0f32);
    simdgroup_elem_store(o_f1c, 1, 0.0f32);
    let o_f1d = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1d, 0, 0.0f32);
    simdgroup_elem_store(o_f1d, 1, 0.0f32);
    let o_f1e = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1e, 0, 0.0f32);
    simdgroup_elem_store(o_f1e, 1, 0.0f32);
    let o_f1f = simdgroup_alloc::<f32, 8, 8>();
    simdgroup_elem_store(o_f1f, 0, 0.0f32);
    simdgroup_elem_store(o_f1f, 1, 0.0f32);

    // S/P frags (one chunk: BK=8 == one fragment's worth of KV rows).
    let s_f0 = simdgroup_alloc::<f32, 8, 8>();
    let p_f0 = simdgroup_alloc::<T, 8, 8>();
    let kt_a = simdgroup_alloc::<T, 8, 8>();
    let v_a = simdgroup_alloc::<T, 8, 8>();
    let mut m_row = neg_infinity();
    let mut s_row = 0.0f32;
    let q_abs = base_kv + qrow_c;

    // K-block loop bound MUST be uniform across the whole threadgroup
    // (it wraps `threadgroup_barrier`, which every lane in every
    // simdgroup has to hit the same number of times): `kb_lim` is
    // computed from the TILE's last row (all 32, not just this
    // simdgroup's 8), mirroring the scalar kernel's `tg_last_row`. A
    // separate per-simdgroup `sg_kb_lim` (this simdgroup's own last
    // row) only gates which iterations do real matmul/softmax work --
    // the coop load and both barriers stay unconditional every
    // iteration. Safe to diverge simdgroups on `sg_kb_lim` because the
    // gated compute uses only `simdgroup_barrier_mem_none` (simdgroup-
    // scoped) and `simdgroup_matmul`, never `threadgroup_barrier`.
    let full_kb = (n_kv_logical + bk - 1u32) / bk;
    let tg_first_row = q_tile * bq;
    let tg_last_row = tg_first_row + bq - 1u32;
    let tg_last_row_c = select(tg_last_row < n_query, tg_last_row, n_query - 1u32);
    let tg_last_abs = base_kv + tg_last_row_c;
    let kb_lim = select(causal == 1u32, (tg_last_abs / bk) + 1u32, full_kb);
    let sg_last_row = q_tile_first + bq_sg - 1u32;
    let sg_last_row_c = select(sg_last_row < n_query, sg_last_row, n_query - 1u32);
    let sg_last_abs = base_kv + sg_last_row_c;
    let sg_kb_lim = select(causal == 1u32, (sg_last_abs / bk) + 1u32, full_kb);

    for kb in range(0u32, kb_lim, 1u32) {
        let kb_off = kb * bk;
        // ── Coop K/V load: 128 lanes x 2 elems/row cover head_dim=256.
        // Clamp the physical row to kv_stride (append contract); masked
        // below so over-range loaded data is always discarded.
        for kr in range(0u32, bk, 1u32) {
            let kv_logical_row = kb_off + kr;
            let kv_row = select(kv_logical_row < kv_stride, kv_logical_row, kv_stride - 1u32);
            let phys_base = kv_head_base + kv_row * head_dim;
            let e0 = lane_in_tg;
            let e1 = lane_in_tg + 128u32;
            let kr_off = kr * kv_ld;
            threadgroup_store("tg_ks", kr_off + e0, load(k[phys_base + e0]).cast::<T>());
            threadgroup_store("tg_ks", kr_off + e1, load(k[phys_base + e1]).cast::<T>());
            threadgroup_store("tg_vs", kr_off + e0, load(v[phys_base + e0]).cast::<T>());
            threadgroup_store("tg_vs", kr_off + e1, load(v[phys_base + e1]).cast::<T>());
        }
        threadgroup_barrier();
        if kb < sg_kb_lim {
            // ── S = Q . K^T (32 matmuls per SG: 32 d_frags x 1 k_chunk) ──
            simdgroup_elem_store(s_f0, 0, 0.0f32);
            simdgroup_elem_store(s_f0, 1, 0.0f32);
            // d=0
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 0u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 0u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f0, kt_a, s_f0);
            // d=1
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 8u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 8u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1, kt_a, s_f0);
            // d=2
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 16u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 16u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f2, kt_a, s_f0);
            // d=3
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 24u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 24u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f3, kt_a, s_f0);
            // d=4
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 32u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 32u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f4, kt_a, s_f0);
            // d=5
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 40u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 40u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f5, kt_a, s_f0);
            // d=6
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 48u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 48u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f6, kt_a, s_f0);
            // d=7
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 56u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 56u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f7, kt_a, s_f0);
            // d=8
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 64u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 64u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f8, kt_a, s_f0);
            // d=9
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 72u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 72u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f9, kt_a, s_f0);
            // d=a
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 80u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 80u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fa, kt_a, s_f0);
            // d=b
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 88u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 88u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fb, kt_a, s_f0);
            // d=c
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 96u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 96u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fc, kt_a, s_f0);
            // d=d
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 104u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 104u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fd, kt_a, s_f0);
            // d=e
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 112u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 112u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fe, kt_a, s_f0);
            // d=f
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 120u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 120u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_ff, kt_a, s_f0);
            // d=10
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 128u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 128u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f10, kt_a, s_f0);
            // d=11
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 136u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 136u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f11, kt_a, s_f0);
            // d=12
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 144u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 144u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f12, kt_a, s_f0);
            // d=13
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 152u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 152u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f13, kt_a, s_f0);
            // d=14
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 160u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 160u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f14, kt_a, s_f0);
            // d=15
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 168u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 168u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f15, kt_a, s_f0);
            // d=16
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 176u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 176u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f16, kt_a, s_f0);
            // d=17
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 184u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 184u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f17, kt_a, s_f0);
            // d=18
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 192u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 192u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f18, kt_a, s_f0);
            // d=19
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 200u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 200u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f19, kt_a, s_f0);
            // d=1a
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 208u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 208u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1a, kt_a, s_f0);
            // d=1b
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 216u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 216u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1b, kt_a, s_f0);
            // d=1c
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 224u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 224u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1c, kt_a, s_f0);
            // d=1d
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 232u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 232u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1d, kt_a, s_f0);
            // d=1e
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 240u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 240u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1e, kt_a, s_f0);
            // d=1f
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 248u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 248u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1f, kt_a, s_f0);
            // ── Online softmax. One chunk (BK=8): row-reduce s_f0's 8
            // columns across the 4 lanes that jointly own this fm row (2
            // shuffle steps, same lane-group pattern as the d128 sibling).
            let raw_s0 = simdgroup_elem_load(s_f0, 0) * scale_log2;
            let raw_s1 = simdgroup_elem_load(s_f0, 1) * scale_log2;
            let k_abs0 = kb_off + fn0;
            let k_abs1 = kb_off + fn1;
            let over0 = k_abs0 >= n_kv_logical;
            let over1 = k_abs1 >= n_kv_logical;
            let causal_block0 = k_abs0 > q_abs;
            let causal_block1 = k_abs1 > q_abs;
            let masked0 = select(causal == 1u32, causal_block0, over0);
            let masked1 = select(causal == 1u32, causal_block1, over1);
            let s0 = select(masked0, neg_infinity(), raw_s0);
            let s1 = select(masked1, neg_infinity(), raw_s1);
            let lane_max = select(s0 > s1, s0, s1);
            let mxor1 = simd_shuffle_xor(lane_max, 1u32);
            let mx_after1 = select(lane_max > mxor1, lane_max, mxor1);
            let mxor8 = simd_shuffle_xor(mx_after1, 8u32);
            let row_max = select(mx_after1 > mxor8, mx_after1, mxor8);
            let new_m = select(row_max > m_row, row_max, m_row);
            let m_diff = exp2(m_row - new_m);
            let p0 = exp2(s0 - new_m);
            let p1 = exp2(s1 - new_m);
            let lane_sum = p0 + p1;
            let sxor1 = simd_shuffle_xor(lane_sum, 1u32);
            let sum_after1 = lane_sum + sxor1;
            let sxor8 = simd_shuffle_xor(sum_after1, 8u32);
            let row_sum = sum_after1 + sxor8;
            s_row = s_row * m_diff + row_sum;
            m_row = new_m;
            simdgroup_elem_store(p_f0, 0, p0.cast::<T>());
            simdgroup_elem_store(p_f0, 1, p1.cast::<T>());
            simdgroup_elem_store(o_f0, 0, simdgroup_elem_load(o_f0, 0) * m_diff);
            simdgroup_elem_store(o_f0, 1, simdgroup_elem_load(o_f0, 1) * m_diff);
            simdgroup_elem_store(o_f1, 0, simdgroup_elem_load(o_f1, 0) * m_diff);
            simdgroup_elem_store(o_f1, 1, simdgroup_elem_load(o_f1, 1) * m_diff);
            simdgroup_elem_store(o_f2, 0, simdgroup_elem_load(o_f2, 0) * m_diff);
            simdgroup_elem_store(o_f2, 1, simdgroup_elem_load(o_f2, 1) * m_diff);
            simdgroup_elem_store(o_f3, 0, simdgroup_elem_load(o_f3, 0) * m_diff);
            simdgroup_elem_store(o_f3, 1, simdgroup_elem_load(o_f3, 1) * m_diff);
            simdgroup_elem_store(o_f4, 0, simdgroup_elem_load(o_f4, 0) * m_diff);
            simdgroup_elem_store(o_f4, 1, simdgroup_elem_load(o_f4, 1) * m_diff);
            simdgroup_elem_store(o_f5, 0, simdgroup_elem_load(o_f5, 0) * m_diff);
            simdgroup_elem_store(o_f5, 1, simdgroup_elem_load(o_f5, 1) * m_diff);
            simdgroup_elem_store(o_f6, 0, simdgroup_elem_load(o_f6, 0) * m_diff);
            simdgroup_elem_store(o_f6, 1, simdgroup_elem_load(o_f6, 1) * m_diff);
            simdgroup_elem_store(o_f7, 0, simdgroup_elem_load(o_f7, 0) * m_diff);
            simdgroup_elem_store(o_f7, 1, simdgroup_elem_load(o_f7, 1) * m_diff);
            simdgroup_elem_store(o_f8, 0, simdgroup_elem_load(o_f8, 0) * m_diff);
            simdgroup_elem_store(o_f8, 1, simdgroup_elem_load(o_f8, 1) * m_diff);
            simdgroup_elem_store(o_f9, 0, simdgroup_elem_load(o_f9, 0) * m_diff);
            simdgroup_elem_store(o_f9, 1, simdgroup_elem_load(o_f9, 1) * m_diff);
            simdgroup_elem_store(o_fa, 0, simdgroup_elem_load(o_fa, 0) * m_diff);
            simdgroup_elem_store(o_fa, 1, simdgroup_elem_load(o_fa, 1) * m_diff);
            simdgroup_elem_store(o_fb, 0, simdgroup_elem_load(o_fb, 0) * m_diff);
            simdgroup_elem_store(o_fb, 1, simdgroup_elem_load(o_fb, 1) * m_diff);
            simdgroup_elem_store(o_fc, 0, simdgroup_elem_load(o_fc, 0) * m_diff);
            simdgroup_elem_store(o_fc, 1, simdgroup_elem_load(o_fc, 1) * m_diff);
            simdgroup_elem_store(o_fd, 0, simdgroup_elem_load(o_fd, 0) * m_diff);
            simdgroup_elem_store(o_fd, 1, simdgroup_elem_load(o_fd, 1) * m_diff);
            simdgroup_elem_store(o_fe, 0, simdgroup_elem_load(o_fe, 0) * m_diff);
            simdgroup_elem_store(o_fe, 1, simdgroup_elem_load(o_fe, 1) * m_diff);
            simdgroup_elem_store(o_ff, 0, simdgroup_elem_load(o_ff, 0) * m_diff);
            simdgroup_elem_store(o_ff, 1, simdgroup_elem_load(o_ff, 1) * m_diff);
            simdgroup_elem_store(o_f10, 0, simdgroup_elem_load(o_f10, 0) * m_diff);
            simdgroup_elem_store(o_f10, 1, simdgroup_elem_load(o_f10, 1) * m_diff);
            simdgroup_elem_store(o_f11, 0, simdgroup_elem_load(o_f11, 0) * m_diff);
            simdgroup_elem_store(o_f11, 1, simdgroup_elem_load(o_f11, 1) * m_diff);
            simdgroup_elem_store(o_f12, 0, simdgroup_elem_load(o_f12, 0) * m_diff);
            simdgroup_elem_store(o_f12, 1, simdgroup_elem_load(o_f12, 1) * m_diff);
            simdgroup_elem_store(o_f13, 0, simdgroup_elem_load(o_f13, 0) * m_diff);
            simdgroup_elem_store(o_f13, 1, simdgroup_elem_load(o_f13, 1) * m_diff);
            simdgroup_elem_store(o_f14, 0, simdgroup_elem_load(o_f14, 0) * m_diff);
            simdgroup_elem_store(o_f14, 1, simdgroup_elem_load(o_f14, 1) * m_diff);
            simdgroup_elem_store(o_f15, 0, simdgroup_elem_load(o_f15, 0) * m_diff);
            simdgroup_elem_store(o_f15, 1, simdgroup_elem_load(o_f15, 1) * m_diff);
            simdgroup_elem_store(o_f16, 0, simdgroup_elem_load(o_f16, 0) * m_diff);
            simdgroup_elem_store(o_f16, 1, simdgroup_elem_load(o_f16, 1) * m_diff);
            simdgroup_elem_store(o_f17, 0, simdgroup_elem_load(o_f17, 0) * m_diff);
            simdgroup_elem_store(o_f17, 1, simdgroup_elem_load(o_f17, 1) * m_diff);
            simdgroup_elem_store(o_f18, 0, simdgroup_elem_load(o_f18, 0) * m_diff);
            simdgroup_elem_store(o_f18, 1, simdgroup_elem_load(o_f18, 1) * m_diff);
            simdgroup_elem_store(o_f19, 0, simdgroup_elem_load(o_f19, 0) * m_diff);
            simdgroup_elem_store(o_f19, 1, simdgroup_elem_load(o_f19, 1) * m_diff);
            simdgroup_elem_store(o_f1a, 0, simdgroup_elem_load(o_f1a, 0) * m_diff);
            simdgroup_elem_store(o_f1a, 1, simdgroup_elem_load(o_f1a, 1) * m_diff);
            simdgroup_elem_store(o_f1b, 0, simdgroup_elem_load(o_f1b, 0) * m_diff);
            simdgroup_elem_store(o_f1b, 1, simdgroup_elem_load(o_f1b, 1) * m_diff);
            simdgroup_elem_store(o_f1c, 0, simdgroup_elem_load(o_f1c, 0) * m_diff);
            simdgroup_elem_store(o_f1c, 1, simdgroup_elem_load(o_f1c, 1) * m_diff);
            simdgroup_elem_store(o_f1d, 0, simdgroup_elem_load(o_f1d, 0) * m_diff);
            simdgroup_elem_store(o_f1d, 1, simdgroup_elem_load(o_f1d, 1) * m_diff);
            simdgroup_elem_store(o_f1e, 0, simdgroup_elem_load(o_f1e, 0) * m_diff);
            simdgroup_elem_store(o_f1e, 1, simdgroup_elem_load(o_f1e, 1) * m_diff);
            simdgroup_elem_store(o_f1f, 0, simdgroup_elem_load(o_f1f, 0) * m_diff);
            simdgroup_elem_store(o_f1f, 1, simdgroup_elem_load(o_f1f, 1) * m_diff);
            // ── O += P . V (32 matmuls per SG: 32 d_frags x 1 k_chunk) ──
            // d=0
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f0);
            // d=1
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1);
            // d=2
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f2);
            // d=3
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f3);
            // d=4
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f4);
            // d=5
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f5);
            // d=6
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f6);
            // d=7
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f7);
            // d=8
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f8);
            // d=9
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f9);
            // d=a
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fa);
            // d=b
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fb);
            // d=c
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fc);
            // d=d
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fd);
            // d=e
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fe);
            // d=f
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_ff);
            // d=10
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 128u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 128u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f10);
            // d=11
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 136u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 136u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f11);
            // d=12
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 144u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 144u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f12);
            // d=13
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 152u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 152u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f13);
            // d=14
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 160u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 160u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f14);
            // d=15
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 168u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 168u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f15);
            // d=16
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 176u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 176u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f16);
            // d=17
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 184u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 184u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f17);
            // d=18
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 192u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 192u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f18);
            // d=19
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 200u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 200u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f19);
            // d=1a
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 208u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 208u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1a);
            // d=1b
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 216u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 216u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1b);
            // d=1c
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 224u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 224u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1c);
            // d=1d
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 232u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 232u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1d);
            // d=1e
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 240u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 240u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1e);
            // d=1f
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 248u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 248u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1f);
        }
        threadgroup_barrier();
    }
    // ── Final normalize + guarded write (ragged tiles skip the store) ──
    let is_row = select(s_row > 0.0f32, 1.0f32 / s_row, 0.0f32);
    if q_tile_first + fm < n_query {
        store(out[row_base + 0u32 + fn0], (simdgroup_elem_load(o_f0, 0) * is_row).cast::<T>());
        store(out[row_base + 0u32 + fn1], (simdgroup_elem_load(o_f0, 1) * is_row).cast::<T>());
        store(out[row_base + 8u32 + fn0], (simdgroup_elem_load(o_f1, 0) * is_row).cast::<T>());
        store(out[row_base + 8u32 + fn1], (simdgroup_elem_load(o_f1, 1) * is_row).cast::<T>());
        store(out[row_base + 16u32 + fn0], (simdgroup_elem_load(o_f2, 0) * is_row).cast::<T>());
        store(out[row_base + 16u32 + fn1], (simdgroup_elem_load(o_f2, 1) * is_row).cast::<T>());
        store(out[row_base + 24u32 + fn0], (simdgroup_elem_load(o_f3, 0) * is_row).cast::<T>());
        store(out[row_base + 24u32 + fn1], (simdgroup_elem_load(o_f3, 1) * is_row).cast::<T>());
        store(out[row_base + 32u32 + fn0], (simdgroup_elem_load(o_f4, 0) * is_row).cast::<T>());
        store(out[row_base + 32u32 + fn1], (simdgroup_elem_load(o_f4, 1) * is_row).cast::<T>());
        store(out[row_base + 40u32 + fn0], (simdgroup_elem_load(o_f5, 0) * is_row).cast::<T>());
        store(out[row_base + 40u32 + fn1], (simdgroup_elem_load(o_f5, 1) * is_row).cast::<T>());
        store(out[row_base + 48u32 + fn0], (simdgroup_elem_load(o_f6, 0) * is_row).cast::<T>());
        store(out[row_base + 48u32 + fn1], (simdgroup_elem_load(o_f6, 1) * is_row).cast::<T>());
        store(out[row_base + 56u32 + fn0], (simdgroup_elem_load(o_f7, 0) * is_row).cast::<T>());
        store(out[row_base + 56u32 + fn1], (simdgroup_elem_load(o_f7, 1) * is_row).cast::<T>());
        store(out[row_base + 64u32 + fn0], (simdgroup_elem_load(o_f8, 0) * is_row).cast::<T>());
        store(out[row_base + 64u32 + fn1], (simdgroup_elem_load(o_f8, 1) * is_row).cast::<T>());
        store(out[row_base + 72u32 + fn0], (simdgroup_elem_load(o_f9, 0) * is_row).cast::<T>());
        store(out[row_base + 72u32 + fn1], (simdgroup_elem_load(o_f9, 1) * is_row).cast::<T>());
        store(out[row_base + 80u32 + fn0], (simdgroup_elem_load(o_fa, 0) * is_row).cast::<T>());
        store(out[row_base + 80u32 + fn1], (simdgroup_elem_load(o_fa, 1) * is_row).cast::<T>());
        store(out[row_base + 88u32 + fn0], (simdgroup_elem_load(o_fb, 0) * is_row).cast::<T>());
        store(out[row_base + 88u32 + fn1], (simdgroup_elem_load(o_fb, 1) * is_row).cast::<T>());
        store(out[row_base + 96u32 + fn0], (simdgroup_elem_load(o_fc, 0) * is_row).cast::<T>());
        store(out[row_base + 96u32 + fn1], (simdgroup_elem_load(o_fc, 1) * is_row).cast::<T>());
        store(out[row_base + 104u32 + fn0], (simdgroup_elem_load(o_fd, 0) * is_row).cast::<T>());
        store(out[row_base + 104u32 + fn1], (simdgroup_elem_load(o_fd, 1) * is_row).cast::<T>());
        store(out[row_base + 112u32 + fn0], (simdgroup_elem_load(o_fe, 0) * is_row).cast::<T>());
        store(out[row_base + 112u32 + fn1], (simdgroup_elem_load(o_fe, 1) * is_row).cast::<T>());
        store(out[row_base + 120u32 + fn0], (simdgroup_elem_load(o_ff, 0) * is_row).cast::<T>());
        store(out[row_base + 120u32 + fn1], (simdgroup_elem_load(o_ff, 1) * is_row).cast::<T>());
        store(out[row_base + 128u32 + fn0], (simdgroup_elem_load(o_f10, 0) * is_row).cast::<T>());
        store(out[row_base + 128u32 + fn1], (simdgroup_elem_load(o_f10, 1) * is_row).cast::<T>());
        store(out[row_base + 136u32 + fn0], (simdgroup_elem_load(o_f11, 0) * is_row).cast::<T>());
        store(out[row_base + 136u32 + fn1], (simdgroup_elem_load(o_f11, 1) * is_row).cast::<T>());
        store(out[row_base + 144u32 + fn0], (simdgroup_elem_load(o_f12, 0) * is_row).cast::<T>());
        store(out[row_base + 144u32 + fn1], (simdgroup_elem_load(o_f12, 1) * is_row).cast::<T>());
        store(out[row_base + 152u32 + fn0], (simdgroup_elem_load(o_f13, 0) * is_row).cast::<T>());
        store(out[row_base + 152u32 + fn1], (simdgroup_elem_load(o_f13, 1) * is_row).cast::<T>());
        store(out[row_base + 160u32 + fn0], (simdgroup_elem_load(o_f14, 0) * is_row).cast::<T>());
        store(out[row_base + 160u32 + fn1], (simdgroup_elem_load(o_f14, 1) * is_row).cast::<T>());
        store(out[row_base + 168u32 + fn0], (simdgroup_elem_load(o_f15, 0) * is_row).cast::<T>());
        store(out[row_base + 168u32 + fn1], (simdgroup_elem_load(o_f15, 1) * is_row).cast::<T>());
        store(out[row_base + 176u32 + fn0], (simdgroup_elem_load(o_f16, 0) * is_row).cast::<T>());
        store(out[row_base + 176u32 + fn1], (simdgroup_elem_load(o_f16, 1) * is_row).cast::<T>());
        store(out[row_base + 184u32 + fn0], (simdgroup_elem_load(o_f17, 0) * is_row).cast::<T>());
        store(out[row_base + 184u32 + fn1], (simdgroup_elem_load(o_f17, 1) * is_row).cast::<T>());
        store(out[row_base + 192u32 + fn0], (simdgroup_elem_load(o_f18, 0) * is_row).cast::<T>());
        store(out[row_base + 192u32 + fn1], (simdgroup_elem_load(o_f18, 1) * is_row).cast::<T>());
        store(out[row_base + 200u32 + fn0], (simdgroup_elem_load(o_f19, 0) * is_row).cast::<T>());
        store(out[row_base + 200u32 + fn1], (simdgroup_elem_load(o_f19, 1) * is_row).cast::<T>());
        store(out[row_base + 208u32 + fn0], (simdgroup_elem_load(o_f1a, 0) * is_row).cast::<T>());
        store(out[row_base + 208u32 + fn1], (simdgroup_elem_load(o_f1a, 1) * is_row).cast::<T>());
        store(out[row_base + 216u32 + fn0], (simdgroup_elem_load(o_f1b, 0) * is_row).cast::<T>());
        store(out[row_base + 216u32 + fn1], (simdgroup_elem_load(o_f1b, 1) * is_row).cast::<T>());
        store(out[row_base + 224u32 + fn0], (simdgroup_elem_load(o_f1c, 0) * is_row).cast::<T>());
        store(out[row_base + 224u32 + fn1], (simdgroup_elem_load(o_f1c, 1) * is_row).cast::<T>());
        store(out[row_base + 232u32 + fn0], (simdgroup_elem_load(o_f1d, 0) * is_row).cast::<T>());
        store(out[row_base + 232u32 + fn1], (simdgroup_elem_load(o_f1d, 1) * is_row).cast::<T>());
        store(out[row_base + 240u32 + fn0], (simdgroup_elem_load(o_f1e, 0) * is_row).cast::<T>());
        store(out[row_base + 240u32 + fn1], (simdgroup_elem_load(o_f1e, 1) * is_row).cast::<T>());
        store(out[row_base + 248u32 + fn0], (simdgroup_elem_load(o_f1f, 0) * is_row).cast::<T>());
        store(out[row_base + 248u32 + fn1], (simdgroup_elem_load(o_f1f, 1) * is_row).cast::<T>());
    }
}

/// Bench at the append shape this kernel targets: a 1024-token query
/// chunk (32 Q-tiles of BQ=32) landing on a base_kv=8192 prefix,
/// causal, GQA 32/8 (same shape as the scalar sibling's bench, BQ
/// doubled to this kernel's MMA-friendly tile size). Emits every dtype
/// FFAI dispatches so `ffaik build --emit` picks the kernel up for all
/// three.
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_sdpa_prefill_mma_d256;
    use crate::utils::{InputDomain, input_buffer};

    const HEAD_DIM: usize = 256;
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 8;
    const GQA_FACTOR: usize = N_Q_HEADS / N_KV_HEADS;
    const N_QUERY: usize = 1024;
    const BASE_KV: usize = 8192;
    const KV_STRIDE: usize = BASE_KV + N_QUERY;
    const BQ: usize = 32;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_sdpa_prefill_mma_d256(dt: DType) -> BenchSetup {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let q_elems = N_QUERY * N_Q_HEADS * HEAD_DIM;
        let kv_elems = N_KV_HEADS * KV_STRIDE * HEAD_DIM;
        let bytes = (2 * q_elems + 2 * kv_elems) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_prefill_mma_d256::kernel_ir_for(dt))
            .mode(KernelMode::SimdGroup2D)
            .buffer(input_buffer("q", q_elems, dt, InputDomain::Signed))
            .buffer(input_buffer("k", kv_elems, dt, InputDomain::Signed))
            .buffer(input_buffer("v", kv_elems, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", q_elems, dt).output())
            .constexpr("head_dim", HEAD_DIM as u32)
            .constexpr("n_q_heads", N_Q_HEADS as u32)
            .constexpr("base_kv", BASE_KV as u32)
            .constexpr("n_query", N_QUERY as u32)
            .constexpr("kv_stride", KV_STRIDE as u32)
            .constexpr("heads_per_group", GQA_FACTOR as u32)
            .constexpr("causal", 1u32)
            .constexpr("scale", scale)
            .grid_3d((N_QUERY / BQ) as u32, N_Q_HEADS as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(4 * (N_Q_HEADS as u64) * (N_QUERY as u64) * (BASE_KV as u64) * (HEAD_DIM as u64))
    }
}

/// `ffaik test`-harness CPU-oracle coverage, following the same
/// convention as `sdpa_prefill_qtiled_d256`'s `kernel_tests` (this
/// kernel's scalar sibling): realistic GQA (32 q-heads / 8 kv-heads,
/// matching Qwen3.6-A3B's full-attention layers), a nontrivial append
/// prefix (`base_kv = 300`), and a ragged `n_query` (37 is not a
/// multiple of `BQ = 32`: one full tile + a 5-row guarded remainder)
/// exercising the pad/guard path. Same oracle math as the scalar
/// kernel's `naive_sdpa_append` (duplicated locally, same reasoning: no
/// cross-module oracle sharing).
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_sdpa_prefill_mma_d256;
    use crate::utils::{pack_f32, unpack_f32};

    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa_append(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        base_kv: usize,
        n_query: usize,
        kv_stride: usize,
        causal: bool,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
        for r in 0..n_query {
            let n_kv = if causal { base_kv + r + 1 } else { base_kv + n_query };
            for qh in 0..n_q_heads {
                let kvh = qh / gqa;
                let q_off = (r * n_q_heads + qh) * head_dim;
                let kv_slab = kvh * kv_stride * head_dim;
                let mut scores = vec![0.0f32; n_kv];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_off = kv_slab + t * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    *score = dot * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (t, s) in scores.iter().enumerate() {
                        acc += *s * inv * v[kv_slab + t * head_dim + d];
                    }
                    out[q_off + d] = acc;
                }
            }
        }
        out
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    fn mma_d256_setup(dt: DType, causal: bool) -> TestSetup {
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 256usize);
        let (base_kv, n_query) = (300usize, 37usize);
        let bq = 32usize;
        let kv_stride = base_kv + n_query;
        let heads_per_group = n_q_heads / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q = unpack_f32(&pack_f32(&ramp(n_query * n_q_heads * head_dim, 0.013, -0.4), dt), dt);
        let k =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.011, -0.5), dt), dt);
        let v =
            unpack_f32(&pack_f32(&ramp(n_kv_heads * kv_stride * head_dim, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa_append(
            &q, &k, &v, n_q_heads, n_kv_heads, head_dim, base_kv, n_query, kv_stride, causal, scale,
        );

        TestSetup::new(ffai_sdpa_prefill_mma_d256::kernel_ir_for(dt))
            .mode(KernelMode::SimdGroup2D)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", n_query * n_q_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_q_heads", n_q_heads as u32)
            .constexpr("base_kv", base_kv as u32)
            .constexpr("n_query", n_query as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .constexpr("causal", u32::from(causal))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_query.div_ceil(bq) as u32, n_q_heads as u32, 1, [128, 1, 1])
    }

    // Causal append: query row `r` attends `[0, base_kv + r + 1)`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 1e-2, 5e-2])]
    fn test_ffai_sdpa_prefill_mma_d256_causal(dt: DType) -> TestSetup { mma_d256_setup(dt, true) }

    // Full (bidirectional): every row attends `[0, base_kv + n_query)`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 1e-2, 5e-2])]
    fn test_ffai_sdpa_prefill_mma_d256_full(dt: DType) -> TestSetup { mma_d256_setup(dt, false) }
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! BK=16 follow-up to the MMA (simdgroup-matrix) prefill-append SDPA for
//! `head_dim == 256`, f16/bf16 only.
//!
//! `iron_sdpa_prefill_mma_d256`'s own doc comment recorded this as a
//! deferred follow-up: BK=16 (matching the d128 MMA sibling,
//! `iron_sdpa_prefill_mma`) halves the number of K-block loop iterations
//! by reusing each cooperative K/V threadgroup load across twice as many
//! query rows before advancing, which halves the per-block loop overhead
//! (barriers, online-softmax bookkeeping) paid per KV row. It was not
//! shipped in the BK=8 kernel because the threadgroup budget
//! `2 * BK * kv_ld * sizeof(T)` (two buffers, tg_ks + tg_vs) exceeds
//! Apple's 32 KiB/threadgroup ceiling at f32: `2*16*264*4` = 33792 B,
//! over by roughly 1 KiB. f16/bf16 fit: `2*16*264*2` = 16896 B, half the
//! ceiling. This kernel is the dtype-conditional BK=16 variant for those
//! two dtypes only; f32 keeps the BK=8 kernel. There is no compile-time
//! guard against instantiating this generic kernel at T=f32 (the DSL's
//! `#[kernel]` has no dtype-restriction attribute), so the f32 exclusion
//! is enforced entirely by convention: `kernel_benches` / `kernel_tests`
//! below only register f16/bf16, and the Swift wrapper only ever
//! dispatches this kernel's f16/bf16 entry points, routing f32 to the
//! original BK=8 kernel unconditionally. Do not add an f32 bench/test
//! here or wire an f32 Swift dispatch path to it -- the threadgroup
//! allocation would fail pipeline creation.
//!
//! ## Geometry
//!
//! Same outer contract as `iron_sdpa_prefill_mma_d256`: TPG = 128 (4
//! simdgroups x 32 lanes), `BQ_SG = 8` query rows per simdgroup x 4
//! simdgroups = `BQ = 32` rows per threadgroup, causal masking, GQA via
//! `heads_per_group`, ragged `n_query` guard, append regime `base_kv` /
//! `kv_stride`. `head_dim = 256` needs 32 d_frags (unchanged).
//!
//! What changes is `BK = 16` (`K_CHUNKS = BK / 8 = 2`): one K-block now
//! covers two fragments' worth of K/V rows, so the Q.K^T and P.V matmul
//! sequences each unroll to 32 d_frags x 2 k_chunks (64 matmuls per SG
//! per phase, following the k_chunk-pair bookkeeping pattern
//! `iron_sdpa_prefill_mma` (the d128 MMA sibling) established: separate
//! `s_f0`/`s_f1` (Q.K^T accumulators) and `p_f0`/`p_f1` (softmax output
//! -> P.V input) frags per chunk, `kt_a`/`kt_b` and `v_a`/`v_b` reused
//! per d_frag within a chunk. Online softmax reduces both chunks' 4
//! lane-owned score elements together before scaling the 32 O frags by
//! `m_diff`, same log2-space math as the BK=8 kernel. Unlike the d128
//! sibling's selectively-placed `simdgroup_barrier_mem_none()` calls,
//! this kernel keeps a barrier before every matmul (both chunks) --
//! matching the BK=8 d256 kernel's own conservative pattern rather than
//! the d128 sibling's irregular one, since the two-chunk restructuring
//! is exactly where fragment-aliasing bugs previously entered this MMA
//! bring-up (missing per-fragment barriers, a double-counted row
//! offset) and the extra simdgroup-scoped barriers are cheap next to a
//! wrong answer.
//!
//! `kv_ld = 264` (the same bank-skew pad) and the causal / append /
//! ragged-tile semantics are byte-for-byte identical to the BK=8
//! kernel's doc comment -- see `sdpa_prefill_mma_d256.rs` for the full
//! account, not repeated here.

use wh_iron::kernel;

#[kernel]
pub fn iron_sdpa_prefill_mma_d256_bk16<T>(
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
    // iron_sdpa_prefill_mma / iron_sdpa_prefill_mma_d256) ──
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;
    let bq = 32u32;
    let bq_sg = 8u32;
    let bk = 16u32;
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
    // BK=16 doubles the row count per buffer vs the BK=8 kernel's 2112:
    // 16 * 264 = 4224 elements. At f16/bf16 (2 B/elem) that's 8448 B per
    // buffer, 16896 B for both (tg_ks + tg_vs) -- see module doc for the
    // f32-exclusion budget math.
    threadgroup_alloc("tg_ks", 4224, T);
    threadgroup_alloc("tg_vs", 4224, T);

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

    // S/P frags (2 chunks: BK=16 == two fragments' worth of KV rows).
    let s_f0 = simdgroup_alloc::<f32, 8, 8>();
    let s_f1 = simdgroup_alloc::<f32, 8, 8>();
    let p_f0 = simdgroup_alloc::<T, 8, 8>();
    let p_f1 = simdgroup_alloc::<T, 8, 8>();
    let kt_a = simdgroup_alloc::<T, 8, 8>();
    let kt_b = simdgroup_alloc::<T, 8, 8>();
    let v_a = simdgroup_alloc::<T, 8, 8>();
    let v_b = simdgroup_alloc::<T, 8, 8>();
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
        // ── Coop K/V load: 128 lanes x 2 elems/row cover head_dim=256,
        // now BK=16 rows (double the BK=8 kernel's loop trip count).
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
            // ── S = Q . K^T (64 matmuls per SG: 32 d_frags x 2 k_chunks) ──
            simdgroup_elem_store(s_f0, 0, 0.0f32);
            simdgroup_elem_store(s_f0, 1, 0.0f32);
            simdgroup_elem_store(s_f1, 0, 0.0f32);
            simdgroup_elem_store(s_f1, 1, 0.0f32);
            // K^T frag elem layout, chunk 0 (rows kb_off..kb_off+7):
            //   elem[i] = K[kb_off + fn_i, d_base + fm] = tg_ks[fn_i * kv_ld + d_base + fm]
            // chunk 1 (rows kb_off+8..kb_off+15): fn_i -> fn_i + 8.
            // d=0
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 0u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 0u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f0, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 0u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 0u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f0, kt_b, s_f1);
            // d=1
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 8u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 8u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 8u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 8u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1, kt_b, s_f1);
            // d=2
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 16u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 16u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f2, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 16u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 16u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f2, kt_b, s_f1);
            // d=3
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 24u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 24u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f3, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 24u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 24u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f3, kt_b, s_f1);
            // d=4
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 32u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 32u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f4, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 32u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 32u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f4, kt_b, s_f1);
            // d=5
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 40u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 40u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f5, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 40u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 40u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f5, kt_b, s_f1);
            // d=6
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 48u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 48u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f6, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 48u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 48u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f6, kt_b, s_f1);
            // d=7
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 56u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 56u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f7, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 56u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 56u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f7, kt_b, s_f1);
            // d=8
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 64u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 64u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f8, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 64u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 64u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f8, kt_b, s_f1);
            // d=9
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 72u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 72u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f9, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 72u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 72u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f9, kt_b, s_f1);
            // d=a
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 80u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 80u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fa, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 80u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 80u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fa, kt_b, s_f1);
            // d=b
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 88u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 88u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fb, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 88u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 88u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fb, kt_b, s_f1);
            // d=c
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 96u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 96u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fc, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 96u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 96u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fc, kt_b, s_f1);
            // d=d
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 104u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 104u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fd, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 104u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 104u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fd, kt_b, s_f1);
            // d=e
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 112u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 112u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fe, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 112u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 112u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_fe, kt_b, s_f1);
            // d=f
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 120u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 120u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_ff, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 120u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 120u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_ff, kt_b, s_f1);
            // d=10
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 128u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 128u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f10, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 128u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 128u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f10, kt_b, s_f1);
            // d=11
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 136u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 136u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f11, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 136u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 136u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f11, kt_b, s_f1);
            // d=12
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 144u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 144u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f12, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 144u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 144u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f12, kt_b, s_f1);
            // d=13
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 152u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 152u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f13, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 152u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 152u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f13, kt_b, s_f1);
            // d=14
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 160u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 160u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f14, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 160u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 160u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f14, kt_b, s_f1);
            // d=15
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 168u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 168u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f15, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 168u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 168u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f15, kt_b, s_f1);
            // d=16
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 176u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 176u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f16, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 176u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 176u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f16, kt_b, s_f1);
            // d=17
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 184u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 184u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f17, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 184u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 184u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f17, kt_b, s_f1);
            // d=18
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 192u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 192u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f18, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 192u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 192u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f18, kt_b, s_f1);
            // d=19
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 200u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 200u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f19, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 200u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 200u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f19, kt_b, s_f1);
            // d=1a
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 208u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 208u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1a, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 208u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 208u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1a, kt_b, s_f1);
            // d=1b
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 216u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 216u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1b, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 216u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 216u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1b, kt_b, s_f1);
            // d=1c
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 224u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 224u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1c, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 224u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 224u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1c, kt_b, s_f1);
            // d=1d
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 232u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 232u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1d, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 232u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 232u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1d, kt_b, s_f1);
            // d=1e
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 240u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 240u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1e, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 240u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 240u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1e, kt_b, s_f1);
            // d=1f
            simdgroup_elem_store(kt_a, 0, threadgroup_load("tg_ks", fn0 * kv_ld + 248u32 + fm));
            simdgroup_elem_store(kt_a, 1, threadgroup_load("tg_ks", fn1 * kv_ld + 248u32 + fm));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1f, kt_a, s_f0);
            simdgroup_elem_store(
                kt_b,
                0,
                threadgroup_load("tg_ks", (fn0 + 8u32) * kv_ld + 248u32 + fm),
            );
            simdgroup_elem_store(
                kt_b,
                1,
                threadgroup_load("tg_ks", (fn1 + 8u32) * kv_ld + 248u32 + fm),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(q_f1f, kt_b, s_f1);
            // ── Online softmax. Two chunks (BK=16): row-reduce s_f0's
            // and s_f1's columns together across the 4 lanes that jointly
            // own this fm row (2 shuffle steps over all 4 lane-owned
            // values), same pattern as the d128 sibling but with this
            // kernel's general append/non-causal/ragged masking (the d128
            // sibling only has a hardcoded causal-prefix mask; this one
            // reuses the BK=8 kernel's `causal`-select + `n_kv_logical`
            // out-of-range guard for both chunks).
            let raw_s00 = simdgroup_elem_load(s_f0, 0) * scale_log2;
            let raw_s01 = simdgroup_elem_load(s_f0, 1) * scale_log2;
            let raw_s10 = simdgroup_elem_load(s_f1, 0) * scale_log2;
            let raw_s11 = simdgroup_elem_load(s_f1, 1) * scale_log2;
            let k_abs00 = kb_off + fn0;
            let k_abs01 = kb_off + fn1;
            let k_abs10 = kb_off + 8u32 + fn0;
            let k_abs11 = kb_off + 8u32 + fn1;
            let over00 = k_abs00 >= n_kv_logical;
            let over01 = k_abs01 >= n_kv_logical;
            let over10 = k_abs10 >= n_kv_logical;
            let over11 = k_abs11 >= n_kv_logical;
            let causal_block00 = k_abs00 > q_abs;
            let causal_block01 = k_abs01 > q_abs;
            let causal_block10 = k_abs10 > q_abs;
            let causal_block11 = k_abs11 > q_abs;
            let masked00 = select(causal == 1u32, causal_block00, over00);
            let masked01 = select(causal == 1u32, causal_block01, over01);
            let masked10 = select(causal == 1u32, causal_block10, over10);
            let masked11 = select(causal == 1u32, causal_block11, over11);
            let s00 = select(masked00, neg_infinity(), raw_s00);
            let s01 = select(masked01, neg_infinity(), raw_s01);
            let s10 = select(masked10, neg_infinity(), raw_s10);
            let s11 = select(masked11, neg_infinity(), raw_s11);
            let mxa = select(s00 > s01, s00, s01);
            let mxb = select(s10 > s11, s10, s11);
            let lane_max = select(mxa > mxb, mxa, mxb);
            let mxor1 = simd_shuffle_xor(lane_max, 1u32);
            let mx_after1 = select(lane_max > mxor1, lane_max, mxor1);
            let mxor8 = simd_shuffle_xor(mx_after1, 8u32);
            let row_max = select(mx_after1 > mxor8, mx_after1, mxor8);
            let new_m = select(row_max > m_row, row_max, m_row);
            let m_diff = exp2(m_row - new_m);
            let p00 = exp2(s00 - new_m);
            let p01 = exp2(s01 - new_m);
            let p10 = exp2(s10 - new_m);
            let p11 = exp2(s11 - new_m);
            let lane_sum = p00 + p01 + p10 + p11;
            let sxor1 = simd_shuffle_xor(lane_sum, 1u32);
            let sum_after1 = lane_sum + sxor1;
            let sxor8 = simd_shuffle_xor(sum_after1, 8u32);
            let row_sum = sum_after1 + sxor8;
            s_row = s_row * m_diff + row_sum;
            m_row = new_m;
            simdgroup_elem_store(p_f0, 0, p00.cast::<T>());
            simdgroup_elem_store(p_f0, 1, p01.cast::<T>());
            simdgroup_elem_store(p_f1, 0, p10.cast::<T>());
            simdgroup_elem_store(p_f1, 1, p11.cast::<T>());
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
            // ── O += P . V (64 matmuls per SG: 32 d_frags x 2 k_chunks) ──
            // V frag elem, chunk 0: tg_vs[fm * kv_ld + d_base + fn_i].
            // chunk 1: tg_vs[(fm + 8) * kv_ld + d_base + fn_i]. Both
            // chunks accumulate into the same o_f<d> (P0.V0 + P1.V1).
            // d=0
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 0u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f0);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 0u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 0u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f0);
            // d=1
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 8u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 8u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 8u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1);
            // d=2
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 16u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f2);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 16u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 16u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f2);
            // d=3
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 24u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f3);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 24u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 24u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f3);
            // d=4
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 32u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f4);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 32u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 32u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f4);
            // d=5
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 40u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f5);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 40u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 40u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f5);
            // d=6
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 48u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f6);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 48u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 48u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f6);
            // d=7
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 56u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f7);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 56u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 56u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f7);
            // d=8
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 64u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f8);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 64u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 64u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f8);
            // d=9
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 72u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f9);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 72u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 72u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f9);
            // d=a
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 80u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fa);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 80u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 80u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_fa);
            // d=b
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 88u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fb);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 88u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 88u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_fb);
            // d=c
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 96u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fc);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 96u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 96u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_fc);
            // d=d
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 104u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fd);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 104u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 104u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_fd);
            // d=e
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 112u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_fe);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 112u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 112u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_fe);
            // d=f
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 120u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_ff);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 120u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 120u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_ff);
            // d=10
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 128u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 128u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f10);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 128u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 128u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f10);
            // d=11
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 136u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 136u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f11);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 136u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 136u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f11);
            // d=12
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 144u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 144u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f12);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 144u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 144u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f12);
            // d=13
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 152u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 152u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f13);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 152u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 152u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f13);
            // d=14
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 160u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 160u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f14);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 160u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 160u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f14);
            // d=15
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 168u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 168u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f15);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 168u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 168u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f15);
            // d=16
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 176u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 176u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f16);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 176u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 176u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f16);
            // d=17
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 184u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 184u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f17);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 184u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 184u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f17);
            // d=18
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 192u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 192u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f18);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 192u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 192u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f18);
            // d=19
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 200u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 200u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f19);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 200u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 200u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f19);
            // d=1a
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 208u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 208u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1a);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 208u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 208u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1a);
            // d=1b
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 216u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 216u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1b);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 216u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 216u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1b);
            // d=1c
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 224u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 224u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1c);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 224u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 224u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1c);
            // d=1d
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 232u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 232u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1d);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 232u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 232u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1d);
            // d=1e
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 240u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 240u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1e);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 240u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 240u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1e);
            // d=1f
            simdgroup_elem_store(v_a, 0, threadgroup_load("tg_vs", fm * kv_ld + 248u32 + fn0));
            simdgroup_elem_store(v_a, 1, threadgroup_load("tg_vs", fm * kv_ld + 248u32 + fn1));
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f0, v_a, o_f1f);
            simdgroup_elem_store(
                v_b,
                0,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 248u32 + fn0),
            );
            simdgroup_elem_store(
                v_b,
                1,
                threadgroup_load("tg_vs", (fm + 8u32) * kv_ld + 248u32 + fn1),
            );
            simdgroup_barrier_mem_none();
            simdgroup_matmul(p_f1, v_b, o_f1f);
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

/// Bench at the same append shape as `iron_sdpa_prefill_mma_d256`'s own
/// bench (1024-token query chunk on an 8192 base_kv prefix, causal, GQA
/// 32/8) so the two are directly comparable. f16/bf16 only -- see the
/// module doc for why f32 is excluded (threadgroup budget).
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_sdpa_prefill_mma_d256_bk16;
    use crate::utils::{InputDomain, input_buffer};

    const HEAD_DIM: usize = 256;
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 8;
    const GQA_FACTOR: usize = N_Q_HEADS / N_KV_HEADS;
    const N_QUERY: usize = 1024;
    const BASE_KV: usize = 8192;
    const KV_STRIDE: usize = BASE_KV + N_QUERY;
    const BQ: usize = 32;

    #[bench(dtypes = [f16, bf16])]
    fn bench_sdpa_prefill_mma_d256_bk16(dt: DType) -> BenchSetup {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let q_elems = N_QUERY * N_Q_HEADS * HEAD_DIM;
        let kv_elems = N_KV_HEADS * KV_STRIDE * HEAD_DIM;
        let bytes = (2 * q_elems + 2 * kv_elems) * dt.size_bytes();
        BenchSetup::new(iron_sdpa_prefill_mma_d256_bk16::kernel_ir_for(dt))
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

/// `iron test`-harness CPU-oracle coverage, following
/// `iron_sdpa_prefill_mma_d256`'s own `kernel_tests` convention: realistic
/// GQA (32 q-heads / 8 kv-heads, matching Qwen3.6-A3B's full-attention
/// layers), a nontrivial append prefix (`base_kv = 300`), and a ragged
/// `n_query` (37 is not a multiple of `BQ = 32`). f16/bf16 only -- no f32
/// bench/test here, see the module doc. Same oracle math as the BK=8
/// kernel's `naive_sdpa_append` (duplicated locally, same reasoning: no
/// cross-module oracle sharing).
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_sdpa_prefill_mma_d256_bk16;
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

    #[allow(clippy::too_many_arguments)]
    fn mma_d256_bk16_setup(
        dt: DType,
        causal: bool,
        n_q_heads: usize,
        n_kv_heads: usize,
        base_kv: usize,
        n_query: usize,
    ) -> TestSetup {
        let head_dim = 256usize;
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

        TestSetup::new(iron_sdpa_prefill_mma_d256_bk16::kernel_ir_for(dt))
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

    // Causal append: query row `r` attends `[0, base_kv + r + 1)`. Realistic
    // GQA 32/8, ragged n_query=37 (one full BQ=32 tile + a 5-row guarded
    // remainder) exercises both the ragged-tile store guard and the
    // BK=16 K-block loop's own remainder (37 rows span 3 BK=16 blocks:
    // 16 + 16 + 5 logical KV rows in the last block before base_kv is
    // folded in).
    #[test_kernel(dtypes = [f16, bf16], tol = [1e-2, 5e-2])]
    fn test_iron_sdpa_prefill_mma_d256_bk16_causal(dt: DType) -> TestSetup {
        mma_d256_bk16_setup(dt, true, 32, 8, 300, 37)
    }

    // Full (bidirectional): every row attends `[0, base_kv + n_query)`.
    #[test_kernel(dtypes = [f16, bf16], tol = [1e-2, 5e-2])]
    fn test_iron_sdpa_prefill_mma_d256_bk16_full(dt: DType) -> TestSetup {
        mma_d256_bk16_setup(dt, false, 32, 8, 300, 37)
    }

    // BK=16 spans two 8-row fragment chunks per K-block; a base_kv that
    // lands mid-block (not a multiple of 16) exercises the boundary
    // between the two chunks' causal masks landing on different rows
    // within the same coop-loaded block.
    #[test_kernel(dtypes = [f16, bf16], tol = [1e-2, 5e-2])]
    fn test_iron_sdpa_prefill_mma_d256_bk16_odd_base_kv(dt: DType) -> TestSetup {
        mma_d256_bk16_setup(dt, true, 32, 8, 41, 65)
    }
}

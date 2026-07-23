//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Q-tiled prefill-append SDPA for `head_dim == 256`.
//!
//! The current d=256 prefill path (`iron_sdpa_multi_d256`, wrapped by
//! `Ops.sdpaMulti`) gives one threadgroup per (query, q_head) and each
//! threadgroup streams the whole causal K/V prefix independently. So the
//! prefix is read once PER QUERY ROW: at a 4096-token chunk landing on a
//! long prefix, the same K/V bytes are re-read thousands of times. That
//! is the bandwidth waste that dominates the append regime (every prefill
//! chunk after the first).
//!
//! This kernel tiles the QUERY axis instead. One threadgroup owns a tile
//! of `BQ = 16` query rows for a single q_head, loads a `BK = 8` row
//! block of K (and V) into threadgroup memory ONCE, and reuses it across
//! all 16 queries in the tile before advancing to the next K block. The
//! K/V prefix is therefore read once per 16 queries instead of once per
//! query, a 16x cut in K/V bandwidth. This is the Flash-Attention-2 tile
//! structure (the d=128 `iron_sdpa_prefill` scalar-flash kernel does the
//! same at head_dim 128); this file is the head_dim-256 sibling adapted
//! to the append case (a strided K/V cache with `base_kv` prefix and a
//! separate `kv_stride` allocation depth, exactly `Ops.sdpaMulti`'s
//! contract).
//!
//! ## Geometry
//!
//! TPG = 128 (4 simdgroups x 32 lanes). Each simdgroup owns `BQ_SG = 4`
//! query rows; 4 simdgroups give `BQ = 16` rows per threadgroup. Within a
//! simdgroup, each lane owns 8 consecutive head-dim elements
//! (`256 / 32`), and a Q.K dot is a `simd_sum` over the 32 lanes' 8-wide
//! stripe partials (scalar flash, no MMA). Per-query online softmax
//! (running max + running sum) and the O = P.V accumulator live in
//! per-lane stack arrays. Q/out layout `[n_query, n_q_heads, head_dim]`,
//! K/V `[n_kv_heads, kv_stride, head_dim]`, GQA via
//! `kv_head = q_head / heads_per_group`.
//!
//! Grid: threadgroup counts `(ceil(n_query / BQ), n_q_heads, 1)`, so a
//! ragged final chunk (n_query not a multiple of BQ) is padded up and the
//! extra query rows are guarded off (loads clamped, stores skipped).
//!
//! ## Causal / append masking
//!
//! Query row `r` (0-indexed in the chunk) has absolute position
//! `base_kv + r` and, when causal, attends `[0, base_kv + r + 1)`; the
//! non-causal case attends `[0, base_kv + n_query)` for every row. The K
//! block loop is bounded per threadgroup (all simdgroups run the same
//! barrier count) to the last-query causal length, with a per-simdgroup
//! inner guard so a simdgroup does no dot work past its own tile's causal
//! reach. K rows past the logical length are causally masked (their score
//! forced to -inf) after being loaded, so the cooperative load stays
//! uniform; the physical load index is clamped to `kv_stride - 1` to
//! avoid reading past the cache allocation (the clamped data is always
//! masked out).
//!
//! Every query sees `k_abs = 0` as valid on the first block (position 0
//! is `<= base_kv + r` always), so the running max is set on the first
//! inner step and the online softmax never evaluates `exp2(-inf - -inf)`,
//! the NaN trap a fully-masked-so-far block would otherwise hit.

use wh_iron::kernel;

#[kernel]
pub fn iron_sdpa_prefill_qtiled_d256<T>(
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
    let sg = simd_id;
    let lane = simd_lane;
    let bq = 16u32;
    let bq_sg = 4u32;
    let bk = 8u32;
    let lane_in_tg = sg * 32u32 + lane;
    let d0 = lane * 8u32;
    let scale_log2 = scale * 1.4426950408889634f32;
    let n_kv_logical = base_kv + n_query;
    let kv_head_base = kv_head * kv_stride * head_dim;

    threadgroup_alloc("tg_ks", 2048, T);
    threadgroup_alloc("tg_vs", 2048, T);

    // First query row this simdgroup owns, and the tile's first row.
    let sg_first_row = q_tile * bq + sg * bq_sg;

    // Load + pre-scale this simdgroup's BQ_SG query stripes. Clamp the row
    // to the last valid query so guard rows (a ragged final tile) read in
    // bounds; their output is never stored.
    stack_alloc("qs", 32u32, "f32");
    for _i in range(0u32, bq_sg, 1u32) {
        let qrow = sg_first_row + _i;
        let qrow_c = select(qrow < n_query, qrow, n_query - 1u32);
        let q_off = (qrow_c * n_q_heads + q_head) * head_dim;
        for _d in range(0u32, 8u32, 1u32) {
            stack_store("qs", _i * 8u32 + _d, load(q[q_off + d0 + _d]).cast::<f32>() * scale_log2);
        }
    }

    // Per-query online-softmax state + output accumulator (per lane).
    stack_alloc("m", 4u32, "f32");
    stack_alloc("s", 4u32, "f32");
    stack_alloc("os", 32u32, "f32");
    for _i in range(0u32, bq_sg, 1u32) {
        stack_store("m", _i, neg_infinity());
        stack_store("s", _i, 0.0f32);
        for _d in range(0u32, 8u32, 1u32) {
            stack_store("os", _i * 8u32 + _d, 0.0f32);
        }
    }

    // K-block loop bound. Causal: bound at the last query's causal reach
    // (clamped to the last valid row). Non-causal: cover the whole logical
    // K range. The TG-wide `kb_lim` sets the barrier count for every
    // simdgroup; the per-simdgroup `sg_kb_lim` trims useless dot work.
    let full_kb = (n_kv_logical + bk - 1u32) / bk;
    let tg_last_row = q_tile * bq + bq - 1u32;
    let tg_last_row_c = select(tg_last_row < n_query, tg_last_row, n_query - 1u32);
    let tg_last_abs = base_kv + tg_last_row_c;
    let kb_lim = select(causal == 1u32, (tg_last_abs / bk) + 1u32, full_kb);
    let sg_last_row = sg_first_row + bq_sg - 1u32;
    let sg_last_row_c = select(sg_last_row < n_query, sg_last_row, n_query - 1u32);
    let sg_last_abs = base_kv + sg_last_row_c;
    let sg_kb_lim = select(causal == 1u32, (sg_last_abs / bk) + 1u32, full_kb);

    stack_alloc("ks", 8u32, "f32");
    stack_alloc("vs", 8u32, "f32");
    for kb in range(0u32, kb_lim, 1u32) {
        let kb_off = kb * bk;
        // Cooperative K/V load: 128 lanes each move 2 of the 256 head-dim
        // elements per K row (indices lane_in_tg and lane_in_tg+128).
        // Clamp the physical row to the cache depth; over-range rows are
        // causally masked below so their loaded data is discarded.
        for _kr in range(0u32, bk, 1u32) {
            let kv_logical_row = kb_off + _kr;
            let kv_row = select(kv_logical_row < kv_stride, kv_logical_row, kv_stride - 1u32);
            let phys_base = kv_head_base + kv_row * head_dim;
            let e0 = lane_in_tg;
            let e1 = lane_in_tg + 128u32;
            let kr_off = _kr * head_dim;
            threadgroup_store("tg_ks", kr_off + e0, load(k[phys_base + e0]).cast::<T>());
            threadgroup_store("tg_ks", kr_off + e1, load(k[phys_base + e1]).cast::<T>());
            threadgroup_store("tg_vs", kr_off + e0, load(v[phys_base + e0]).cast::<T>());
            threadgroup_store("tg_vs", kr_off + e1, load(v[phys_base + e1]).cast::<T>());
        }
        threadgroup_barrier();
        if kb < sg_kb_lim {
            for _ko in range(0u32, bk, 1u32) {
                let k_abs = kb_off + _ko;
                let kr_off = _ko * head_dim;
                for _d in range(0u32, 8u32, 1u32) {
                    stack_store(
                        "ks",
                        _d,
                        threadgroup_load("tg_ks", kr_off + d0 + _d).cast::<f32>(),
                    );
                    stack_store(
                        "vs",
                        _d,
                        threadgroup_load("tg_vs", kr_off + d0 + _d).cast::<f32>(),
                    );
                }
                let over = k_abs >= n_kv_logical;
                for _i in range(0u32, bq_sg, 1u32) {
                    let mut partial = 0.0f32;
                    for _d in range(0u32, 8u32, 1u32) {
                        partial = partial + stack_load("qs", _i * 8u32 + _d) * stack_load("ks", _d);
                    }
                    let r = simd_sum(partial);
                    let q_abs = base_kv + sg_first_row + _i;
                    let causal_block = k_abs > q_abs;
                    let masked = select(causal == 1u32, causal_block, over);
                    let score = select(masked, neg_infinity(), r);
                    let m_old = stack_load("m", _i);
                    let nm = select(score > m_old, score, m_old);
                    let f = exp2(m_old - nm);
                    let w = exp2(score - nm);
                    stack_store("s", _i, stack_load("s", _i) * f + w);
                    stack_store("m", _i, nm);
                    for _d in range(0u32, 8u32, 1u32) {
                        stack_store(
                            "os",
                            _i * 8u32 + _d,
                            stack_load("os", _i * 8u32 + _d) * f + w * stack_load("vs", _d),
                        );
                    }
                }
            }
        }
        threadgroup_barrier();
    }

    // Normalise + store each in-range query row's 8-element lane stripe.
    for _i in range(0u32, bq_sg, 1u32) {
        let qrow = sg_first_row + _i;
        if qrow < n_query {
            let s_i = stack_load("s", _i);
            let inv = select(s_i > 0.0f32, 1.0f32 / s_i, 0.0f32);
            let out_off = (qrow * n_q_heads + q_head) * head_dim + d0;
            for _d in range(0u32, 8u32, 1u32) {
                store(out[out_off + _d], (stack_load("os", _i * 8u32 + _d) * inv).cast::<T>());
            }
        }
    }
}

/// Bench at the append shape this kernel targets: a 1024-token query chunk
/// (64 Q-tiles of BQ=16) landing on a base_kv=8192 prefix, causal, GQA
/// 32/8. Emits every dtype Iron dispatches so `iron build --emit` picks
/// the kernel up for all three.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_sdpa_prefill_qtiled_d256;
    use crate::utils::{InputDomain, input_buffer};

    const HEAD_DIM: usize = 256;
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 8;
    const GQA_FACTOR: usize = N_Q_HEADS / N_KV_HEADS;
    const N_QUERY: usize = 1024;
    const BASE_KV: usize = 8192;
    const KV_STRIDE: usize = BASE_KV + N_QUERY;
    const BQ: usize = 16;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_sdpa_prefill_qtiled_d256(dt: DType) -> BenchSetup {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let q_elems = N_QUERY * N_Q_HEADS * HEAD_DIM;
        let kv_elems = N_KV_HEADS * KV_STRIDE * HEAD_DIM;
        let bytes = (2 * q_elems + 2 * kv_elems) * dt.size_bytes();
        BenchSetup::new(iron_sdpa_prefill_qtiled_d256::kernel_ir_for(dt))
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

/// `iron test`-harness CPU-oracle coverage (the `#[test_kernel]`
/// convention used by the sibling `sdpa_multi_d256` kernel), on top of
/// the standalone GPU-integration tests in
/// `tests/sdpa_prefill_qtiled_gpu.rs`. Realistic GQA (32 q-heads / 8 kv
/// heads matches Qwen3.6-A3B's full-attention layers) with a ragged
/// `n_query` (37 is not a multiple of `BQ = 16`) so the padded-tile guard
/// path is exercised by the same harness `iron test` runs in CI.
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_sdpa_prefill_qtiled_d256;
    use crate::utils::{pack_f32, unpack_f32};

    // Same causal-append math as `sdpa_multi_d256`'s `naive_sdpa_multi`
    // (`n_kv = base_kv + r + 1` per query row); duplicated locally so this
    // kernel's CPU oracle doesn't reach across module boundaries.
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

    fn qtiled_d256_setup(dt: DType, causal: bool) -> TestSetup {
        // GQA 32/8 matches Qwen3.6-A3B; base_kv=300 is a nontrivial append
        // prefix (exercises K/V-reuse across the tile past the first
        // block); n_query=37 is ragged against BQ=16 (2 full tiles + a
        // 5-row guarded remainder), exercising the pad/guard path this
        // kernel adds over `sdpa_multi_d256`.
        let (n_q_heads, n_kv_heads, head_dim) = (32usize, 8usize, 256usize);
        let (base_kv, n_query) = (300usize, 37usize);
        let bq = 16usize;
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

        TestSetup::new(iron_sdpa_prefill_qtiled_d256::kernel_ir_for(dt))
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
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 1.5e-2])]
    fn test_iron_sdpa_prefill_qtiled_d256_causal(dt: DType) -> TestSetup {
        qtiled_d256_setup(dt, true)
    }

    // Full (bidirectional): every row attends `[0, base_kv + n_query)`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 1.5e-2])]
    fn test_iron_sdpa_prefill_qtiled_d256_full(dt: DType) -> TestSetup {
        qtiled_d256_setup(dt, false)
    }
}

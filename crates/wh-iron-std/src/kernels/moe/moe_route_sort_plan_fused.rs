//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Single-dispatch fusion of the `moe_sort_plan_counting.rs` three-pass
//! chain (`iron_moe_sort_plan_hist` -> `_offsets` -> `_scatter`). See that
//! file's module doc for the counting-sort algorithm and stability proof;
//! this file only changes WHERE each pass's inputs come from, not the
//! arithmetic — the output contract (`sorted_experts`/`source_tokens`/
//! `inv_perm`, same stable order) is byte-for-byte identical.
//!
//! ## Why fuse
//!
//! The three-pass chain sits on the critical path of every MoE prefill
//! step: the expert GEMMs cannot start dispatching work until the whole
//! plan is written, so the chain's two inter-pass round trips through
//! global memory (`block_counts`, `block_offsets`) plus two extra
//! kernel-launch/encoder overheads are fully exposed, not pipeline-hidden
//! behind other GPU work. This kernel removes both round trips and two of
//! the three dispatches while keeping the SAME multi-threadgroup
//! occupancy as `iron_moe_sort_plan_hist` (grid = one threadgroup per
//! block, `n_experts` lanes) — unlike a naive single-threadgroup fusion,
//! which would collapse the whole plan onto one GPU core and trade the
//! saved dispatches for lost parallelism.
//!
//! ## Algorithm
//!
//! Each threadgroup (one per block, same partitioning as `_hist`)
//! independently reconstructs the two numbers `_offsets` previously
//! needed a shared global table for:
//!   - `total[e]`  - expert `e`'s count over the WHOLE array (every
//!     threadgroup needs this to build the same ascending-expert `base[e]`
//!     table `_offsets` computed once and shared via `block_offsets`;
//!     here every threadgroup recomputes it directly from `topk_ids`).
//!   - `before[e]` - expert `e`'s count in blocks strictly before this
//!     threadgroup's own block (the per-block running offset `_offsets`
//!     built incrementally by walking blocks in order; here computed
//!     directly as "count of matches at indices < block_start").
//!
//! Recomputing these per-threadgroup instead of sharing one global table
//! means `n_blocks` threadgroups each cooperatively re-read the full
//! `m_total`-length `topk_ids` array once — `n_blocks`x the input reads
//! of the 3-pass chain's `_hist` pass alone (which only reads its own
//! block). That is real added bandwidth, traded for the two removed
//! dispatches and their intermediate-buffer round trips; the wired
//! benchmark decides whether the trade wins at Laguna's shapes.
//!
//! Both counters are built with `atomic_add_tg` (the standard MSL
//! threadgroup-atomic idiom already proven in this codebase by
//! `aura_encode`'s `atomic_or_tg` pack stage): a lane's read of
//! `topk_ids[idx]` doesn't determine which lane accumulates it — lane
//! `e`'s strided scan (`idx = e, e + n_experts, e + 2*n_experts, ...`)
//! visits rows belonging to EVERY expert, not just its own — so
//! concurrent lanes CAN target the same counter slot. This differs from
//! `_hist`/`_offsets`, where every lane owns exactly one expert and never
//! races another lane. Sums are commutative, so any interleaving of the
//! `n_experts` lanes' strided reads yields the identical final counts —
//! determinism (and thus bit-exact output) comes from the counts being
//! sums, not from execution order.
//!
//! The actual scatter (`off` -> row assignment) stays fully
//! deterministic and atomics-free: one lane per expert walks its own
//! block in ascending index order, exactly like
//! `iron_moe_sort_plan_scatter`'s per-row stable rank, so the write
//! order — and therefore every output byte — matches the 3-pass chain
//! exactly.
//!
//! ## Router fusion (not done here)
//!
//! The upstream router (`iron_moe_router_topk_biased` /
//! `iron_moe_sigmoid_bias_rows`) is NOT folded into this kernel. Its
//! natural parallelism axis is per-TOKEN (`T` rows, `n_experts`-wide
//! top-k search per row); this kernel's is per-EXPERT (`n_experts` lanes,
//! `m_total`-wide scan per lane). Merging both axes into one threadgroup
//! shape would need either a two-phase threadgroup (T-wide then
//! E-wide, with a barrier between — no cheaper than a second dispatch on
//! the same command buffer) or picking one axis and looping the other
//! serially per lane (which would blow up the cheaper axis's work by the
//! other axis's size). Neither is "clean" at Laguna's shapes (T up to
//! ~2048, E=256): the complexity and correctness risk outweigh removing
//! two dispatches that are already cheap relative to `T*E`-scaling work.
//! Left as a separate follow-up if the wired result here justifies going
//! further.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_blocks, 1, 1]` (one threadgroup per
//!   block, SAME as `iron_moe_sort_plan_hist`); threadgroup
//!   `[n_experts, 1, 1]` (one lane per expert, <= 256 — same threadgroup
//!   scratch cap as `_offsets`/`iron_moe_build_tile_plan`).
//! - Inputs/outputs, `m_total`/`k`/`n_experts`/`block_size` constexprs,
//!   and the `sorted_experts`/`source_tokens`/`inv_perm` output contract
//!   are byte-for-byte identical to `Ops.moeSortPlanCounting`'s three-pass
//!   call — this is a drop-in ABI match, not a new one (see
//!   `Ops.moeSortPlanFused` in Butter).

use wh_iron::kernel;

#[kernel]
pub fn iron_moe_route_sort_plan_fused(
    topk_ids: Tensor<u32>,
    mut sorted_experts: Tensor<u32>,
    mut source_tokens: Tensor<u32>,
    mut inv_perm: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] k: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] block_size: u32,
) {
    let b = tgid_x;
    let e = tid;
    let block_start = b * block_size;
    let remaining = m_total - block_start;
    let block_len = select(remaining < block_size, remaining, block_size);

    // ---- Phase 1: cooperative full-array histogram (`total[e]`) and this
    // block's "earlier blocks" count (`before[e]`). `n_experts` lanes
    // stride across the whole `m_total`-length array together, so every
    // row is read once per threadgroup (not once per lane) — cheap enough
    // that no lane needs to "own" a row the way `_hist`'s per-block lanes
    // own their block. ----
    threadgroup_alloc("tg_total", 256u32, u32);
    threadgroup_alloc("tg_before", 256u32, u32);
    threadgroup_store("tg_total", e, 0u32);
    threadgroup_store("tg_before", e, 0u32);
    threadgroup_barrier();

    for idx in range(e, m_total, n_experts) {
        let ej = load(topk_ids[idx]);
        atomic_add_tg("tg_total", ej, 1u32);
        if idx < block_start {
            atomic_add_tg("tg_before", ej, 1u32);
        }
    }
    threadgroup_barrier();

    // ---- Phase 2: exclusive prefix sum over experts -> base[e]. Same
    // ascending-expert-order arithmetic as `iron_moe_sort_plan_offsets`,
    // just recomputed locally per threadgroup instead of once globally. ----
    let mut base = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        if e2 < e {
            base = base + threadgroup_load("tg_total", e2);
        }
    }
    let block_offset = base + threadgroup_load("tg_before", e);

    // ---- Phase 3: scatter this block's own rows for expert `e`, walked
    // in ascending index order — identical write order/positions to
    // `iron_moe_sort_plan_scatter`'s per-row stable rank. No atomics: each
    // lane owns exactly one expert within this one block. ----
    let mut off = block_offset;
    for j in range(0u32, block_len, 1u32) {
        let idx = block_start + j;
        let ej = load(topk_ids[idx]);
        if ej == e {
            store(sorted_experts[off], e);
            store(source_tokens[off], idx / k);
            store(inv_perm[idx], off);
            off = off + 1u32;
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::*;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // Same fixture as `moe_sort_plan_counting.rs`'s `test_moe_sort_plan_scatter`
    // — two blocks of 6, byte-exact against the hand-worked stable-sort oracle.
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_moe_route_sort_plan_fused(_dt: DType) -> TestSetup {
        let (n_tokens, k) = (6usize, 2usize);
        let m_total = n_tokens * k;
        let n_experts = 5usize;
        let block_size = 6usize; // 2 blocks of 6
        let ids: Vec<u32> = vec![2, 0, 1, 2, 0, 0, 4, 1, 2, 4, 0, 1];

        let mut order: Vec<usize> = (0..m_total).collect();
        order.sort_by_key(|&i| (ids[i], i));
        let mut se = vec![0u32; m_total];
        let mut st = vec![0u32; m_total];
        let mut ip = vec![0u32; m_total];
        for (dst, &i) in order.iter().enumerate() {
            se[dst] = ids[i];
            st[dst] = (i / k) as u32;
            ip[i] = dst as u32;
        }

        TestSetup::new(iron_moe_route_sort_plan_fused::kernel_ir())
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("topk_ids", u32_bytes(&ids), DType::U32))
            .input(TestBuffer::zeros("sorted_experts", m_total, DType::U32))
            .input(TestBuffer::zeros("source_tokens", m_total, DType::U32))
            .input(TestBuffer::zeros("inv_perm", m_total, DType::U32))
            .constexpr("m_total", m_total as u32)
            .constexpr("k", k as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("block_size", block_size as u32)
            .expect(TestBuffer::from_vec("sorted_experts", u32_bytes(&se), DType::U32))
            .expect(TestBuffer::from_vec("source_tokens", u32_bytes(&st), DType::U32))
            .expect(TestBuffer::from_vec("inv_perm", u32_bytes(&ip), DType::U32))
            .grid_3d(2, 1, 1, [n_experts as u32, 1, 1])
    }
}

/// Bench registration — required for `iron build` to discover this kernel;
/// see the sibling counting-sort passes' identical note.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::*;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    const BENCH_N_EXPERTS: usize = 256;
    const BENCH_BLOCK_SIZE: usize = 256;

    fn bench_ids(m_total: usize, n_experts: usize) -> Vec<u32> {
        (0..m_total).map(|i| ((i * 2654435761usize) % n_experts) as u32).collect()
    }

    #[bench(dtypes = [f32])]
    fn bench_moe_route_sort_plan_fused(_dt: DType) -> BenchSetup {
        let (t, k) = (2048usize, 8usize);
        let m_total = t * k;
        let n_experts = BENCH_N_EXPERTS;
        let block_size = BENCH_BLOCK_SIZE;
        let n_blocks = m_total.div_ceil(block_size);
        let ids = bench_ids(m_total, n_experts);
        BenchSetup::new(iron_moe_route_sort_plan_fused::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("topk_ids", u32_bytes(&ids), DType::U32))
            .buffer(BenchBuffer::zeros("sorted_experts", m_total, DType::U32).output())
            .buffer(BenchBuffer::zeros("source_tokens", m_total, DType::U32).output())
            .buffer(BenchBuffer::zeros("inv_perm", m_total, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("k", k as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("block_size", block_size as u32)
            .grid_3d(n_blocks as u32, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((m_total * 4 * 4) as u64)
    }
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Bucketed counting-sort replacement for `ffai_moe_sort_plan`
//! (`moe_permute.rs`). The original kernel gives every one of `m_total`
//! threads a full linear scan over all `m_total` entries to compute its
//! stable rank - O(m_total^2) total compares. This file splits that same
//! stable counting sort into three passes whose combined work is
//! O(m_total * n_experts), each pass fully data-parallel:
//!
//!   1. `ffai_moe_sort_plan_hist`    - per-(block, expert) row counts.
//!   2. `ffai_moe_sort_plan_offsets` - counts -> each block's per-expert
//!      write offset (base-by-expert prefix sum, then a running
//!      per-block offset within that expert's segment).
//!   3. `ffai_moe_sort_plan_scatter` - each row looks up its block's
//!      offset for its own expert, adds its in-block stable rank (a scan
//!      bounded by `block_size`, not `m_total`), and writes.
//!
//! Rows are processed in original-index order throughout (blocks in
//! ascending order, rows within a block in ascending order), so the
//! result is byte-identical to `ffai_moe_sort_plan`'s stable order:
//! `dst(i) = #{j : ids[j] < ids[i]} + #{j < i : ids[j] == ids[i]}`. Pass 2
//! computes exactly `#{j : ids[j] < ids[i]}` (the ascending-expert-order
//! base) plus `#{j < i, j in an earlier block : ids[j] == ids[i]}` (the
//! per-block running offset); pass 3's bounded scan supplies the
//! remaining `#{j < i, j in the same block : ids[j] == ids[i]}` term. Sum
//! of the two is the same-block/earlier-block split of the original's
//! whole-array `j < i` count - identical set, just partitioned by block.
//!
//! No atomics, no host readback between passes - all three dispatch on
//! the caller's command buffer (see `Ops.moeSortPlanCounting` in FFAI).
//! `n_experts` is capped at 256 (this model's expert count; matches the
//! existing scope limit on `ffai_moe_build_tile_plan`'s threadgroup
//! scratch).

use ffai_kernels::kernel;

// ── ffai_moe_sort_plan_hist ─────────────────────────────────────────────────
//
// Per-block, per-expert row counts. One threadgroup per block, one lane
// per expert; each lane scans its block's rows once.
//
// Inputs:
//   topk_ids    , [m_total]              flat top-k expert ids
//   block_counts, [n_blocks * n_experts] output: count of expert `e` in
//                                          block `b`, laid out `b*n_experts+e`
//
// Constexpr:
//   m_total   , total routed rows (T * k)
//   n_experts , expert count, <= 256
//   block_size, rows per block (last block may be shorter)
//
// Geometry:
//   grid = n_blocks threadgroups, tpg = n_experts (one lane per expert).
#[kernel]
pub fn ffai_moe_sort_plan_hist(
    topk_ids: Tensor<u32>,
    mut block_counts: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] block_size: u32,
) {
    let b = tgid_x;
    let e = tid;
    let block_start = b * block_size;
    let remaining = m_total - block_start;
    let block_len = select(remaining < block_size, remaining, block_size);
    let mut count = 0u32;
    for j in range(0u32, block_len, 1u32) {
        let idx = block_start + j;
        let ej = load(topk_ids[idx]);
        let matched = select(ej == e, 1u32, 0u32);
        count = count + matched;
    }
    store(block_counts[b * n_experts + e], count);
}

// ── ffai_moe_sort_plan_offsets ──────────────────────────────────────────────
//
// Turns `block_counts` into each block's per-expert write offset. One
// threadgroup, one lane per expert:
//   1. sum this expert's count across all blocks (threadgroup memory holds
//      the per-expert totals for the next step's prefix sum), barrier.
//   2. exclusive prefix sum over experts -> `base[e]` (ascending-expert
//      concatenation, matching the original kernel's `#{ids[j] < e}` term).
//   3. walk blocks in order, emitting a running offset starting at
//      `base[e]` -> `block_offsets[b][e]`, the position where block `b`'s
//      FIRST expert-`e` row lands.
//
// Inputs:
//   block_counts , [n_blocks * n_experts]
//   block_offsets, [n_blocks * n_experts] output
//
// Constexpr:
//   n_blocks, n_experts (<= 256)
//
// Geometry:
//   grid = 1 threadgroup, tpg = n_experts.
#[kernel]
pub fn ffai_moe_sort_plan_offsets(
    block_counts: Tensor<u32>,
    mut block_offsets: Tensor<u32>,
    #[constexpr] n_blocks: u32,
    #[constexpr] n_experts: u32,
) {
    let e = tid;
    // Capacity fixed at 256: this model's expert count, same scope limit
    // as `ffai_moe_build_tile_plan`'s threadgroup scratch.
    threadgroup_alloc("counts_local", 256u32, u32);

    let mut total = 0u32;
    for b in range(0u32, n_blocks, 1u32) {
        total = total + load(block_counts[b * n_experts + e]);
    }
    threadgroup_store("counts_local", e, total);
    threadgroup_barrier();

    // Exclusive prefix sum over experts < e -> base[e].
    let mut base = 0u32;
    for e2 in range(0u32, n_experts, 1u32) {
        if e2 < e {
            base = base + threadgroup_load("counts_local", e2);
        }
    }

    // Running per-block offset within this expert's segment, blocks in
    // ascending original-index order.
    let mut running = base;
    for b in range(0u32, n_blocks, 1u32) {
        store(block_offsets[b * n_experts + e], running);
        running = running + load(block_counts[b * n_experts + e]);
    }
}

// ── ffai_moe_sort_plan_scatter ──────────────────────────────────────────────
//
// Final pass: each row looks up its block's base offset for its own
// expert, adds its in-block stable rank (rows before it in the SAME
// block with the same expert id - a scan bounded by `block_size`, not
// `m_total`), and writes the three output rows exactly as
// `ffai_moe_sort_plan` does.
//
// Inputs:
//   topk_ids     , [m_total]
//   block_offsets, [n_blocks * n_experts]
//   sorted_experts, source_tokens, inv_perm, same outputs/semantics as
//     `ffai_moe_sort_plan`.
//
// Constexpr:
//   m_total, k, n_experts, block_size (must match the hist/offsets pass).
//
// Geometry:
//   grid is data-sized over m_total (dispatchThreads style); tpg is a
//   free occupancy knob independent of `block_size`.
#[kernel]
pub fn ffai_moe_sort_plan_scatter(
    topk_ids: Tensor<u32>,
    block_offsets: Tensor<u32>,
    mut sorted_experts: Tensor<u32>,
    mut source_tokens: Tensor<u32>,
    mut inv_perm: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] k: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id(0);
    if i < m_total {
        let e = load(topk_ids[i]);
        let b = i / block_size;
        let block_start = b * block_size;
        let base = load(block_offsets[b * n_experts + e]);
        let mut local_rank = 0u32;
        for j in range(block_start, i, 1u32) {
            let ej = load(topk_ids[j]);
            let matched = select(ej == e, 1u32, 0u32);
            local_rank = local_rank + matched;
        }
        let dst = base + local_rank;
        store(sorted_experts[dst], e);
        store(source_tokens[dst], i / k);
        store(inv_perm[i], dst);
    }
}

pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::*;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // ffai_moe_sort_plan_hist: single-block sanity (block_size >= m_total,
    // so this degenerates to "count every expert over the whole array").
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_moe_sort_plan_hist(_dt: DType) -> TestSetup {
        let n_experts = 5usize;
        let ids: Vec<u32> = vec![2, 0, 1, 2, 0, 0, 4, 1, 2, 4, 0, 1];
        let m_total = ids.len();
        let block_size = 16usize; // one block covers all rows
        let mut expected = vec![0u32; n_experts];
        for &id in &ids {
            expected[id as usize] += 1;
        }
        TestSetup::new(ffai_moe_sort_plan_hist::kernel_ir())
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("topk_ids", u32_bytes(&ids), DType::U32))
            .input(TestBuffer::zeros("block_counts", n_experts, DType::U32))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("block_size", block_size as u32)
            .expect(TestBuffer::from_vec("block_counts", u32_bytes(&expected), DType::U32))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
    }

    // ffai_moe_sort_plan_offsets: two blocks, hand-worked expected offsets.
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_moe_sort_plan_offsets(_dt: DType) -> TestSetup {
        let n_experts = 3usize;
        let n_blocks = 2usize;
        // block 0: e0=1,e1=2,e2=0 ; block 1: e0=1,e1=0,e2=2
        let block_counts: Vec<u32> = vec![1, 2, 0, 1, 0, 2];
        // totals: e0=2,e1=2,e2=2 -> base = [0,2,4]
        // block_offsets[0][e] = base[e]
        // block_offsets[1][e] = base[e] + block_counts[0][e]
        let expected: Vec<u32> = vec![0, 2, 4, 1, 4, 4];
        TestSetup::new(ffai_moe_sort_plan_offsets::kernel_ir())
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("block_counts", u32_bytes(&block_counts), DType::U32))
            .input(TestBuffer::zeros("block_offsets", n_blocks * n_experts, DType::U32))
            .constexpr("n_blocks", n_blocks as u32)
            .constexpr("n_experts", n_experts as u32)
            .expect(TestBuffer::from_vec("block_offsets", u32_bytes(&expected), DType::U32))
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
    }

    // ffai_moe_sort_plan_scatter: two blocks of 6, block_offsets precomputed
    // by hand from the same fixture as `test_moe_sort_plan` (moe_permute.rs).
    #[test_kernel(dtypes = [f32], tol = 0.0)]
    fn test_moe_sort_plan_scatter(_dt: DType) -> TestSetup {
        let (n_tokens, k) = (6usize, 2usize);
        let m_total = n_tokens * k;
        let n_experts = 5usize;
        let block_size = 6usize; // 2 blocks of 6
        let ids: Vec<u32> = vec![2, 0, 1, 2, 0, 0, 4, 1, 2, 4, 0, 1];

        // CPU reference for block_counts/block_offsets (same algorithm as
        // the two prior kernels) so this test is self-contained.
        let n_blocks = m_total.div_ceil(block_size);
        let mut block_counts = vec![0u32; n_blocks * n_experts];
        for b in 0..n_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(m_total);
            for &id in &ids[start..end] {
                block_counts[b * n_experts + id as usize] += 1;
            }
        }
        // Exclusive prefix sum over per-expert totals -> base[e].
        let mut totals = vec![0u32; n_experts];
        for b in 0..n_blocks {
            for e in 0..n_experts {
                totals[e] += block_counts[b * n_experts + e];
            }
        }
        let mut base = vec![0u32; n_experts];
        for e in 1..n_experts {
            base[e] = base[e - 1] + totals[e - 1];
        }
        let mut block_offsets = vec![0u32; n_blocks * n_experts];
        let mut running = base;
        for b in 0..n_blocks {
            for e in 0..n_experts {
                block_offsets[b * n_experts + e] = running[e];
                running[e] += block_counts[b * n_experts + e];
            }
        }

        // Overall oracle (same stable sort the whole pipeline targets).
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

        TestSetup::new(ffai_moe_sort_plan_scatter::kernel_ir())
            .input(TestBuffer::from_vec("topk_ids", u32_bytes(&ids), DType::U32))
            .input(TestBuffer::from_vec("block_offsets", u32_bytes(&block_offsets), DType::U32))
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
            .grid_1d(m_total, 256)
    }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::*;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    const BENCH_N_EXPERTS: usize = 256;
    const BENCH_BLOCK_SIZE: usize = 256;

    fn bench_ids(m_total: usize, n_experts: usize) -> Vec<u32> {
        (0..m_total).map(|i| ((i * 2654435761usize) % n_experts) as u32).collect()
    }

    #[bench(dtypes = [f32])]
    fn bench_moe_sort_plan_hist(_dt: DType) -> BenchSetup {
        let (t, k) = (2048usize, 8usize);
        let m_total = t * k;
        let n_experts = BENCH_N_EXPERTS;
        let block_size = BENCH_BLOCK_SIZE;
        let n_blocks = m_total.div_ceil(block_size);
        let ids = bench_ids(m_total, n_experts);
        BenchSetup::new(ffai_moe_sort_plan_hist::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("topk_ids", u32_bytes(&ids), DType::U32))
            .buffer(BenchBuffer::zeros("block_counts", n_blocks * n_experts, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("block_size", block_size as u32)
            .grid_3d(n_blocks as u32, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((m_total * 4 + n_blocks * n_experts * 4) as u64)
    }

    #[bench(dtypes = [f32])]
    fn bench_moe_sort_plan_offsets(_dt: DType) -> BenchSetup {
        let (t, k) = (2048usize, 8usize);
        let m_total = t * k;
        let n_experts = BENCH_N_EXPERTS;
        let block_size = BENCH_BLOCK_SIZE;
        let n_blocks = m_total.div_ceil(block_size);
        let ids = bench_ids(m_total, n_experts);
        let mut block_counts = vec![0u32; n_blocks * n_experts];
        for b in 0..n_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(m_total);
            for &id in &ids[start..end] {
                block_counts[b * n_experts + id as usize] += 1;
            }
        }
        BenchSetup::new(ffai_moe_sort_plan_offsets::kernel_ir())
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::from_vec("block_counts", u32_bytes(&block_counts), DType::U32))
            .buffer(BenchBuffer::zeros("block_offsets", n_blocks * n_experts, DType::U32).output())
            .constexpr("n_blocks", n_blocks as u32)
            .constexpr("n_experts", n_experts as u32)
            .grid_3d(1, 1, 1, [n_experts as u32, 1, 1])
            .bytes_moved((n_blocks * n_experts * 4 * 2) as u64)
    }

    #[bench(dtypes = [f32])]
    fn bench_moe_sort_plan_scatter(_dt: DType) -> BenchSetup {
        let (t, k) = (2048usize, 8usize);
        let m_total = t * k;
        let n_experts = BENCH_N_EXPERTS;
        let block_size = BENCH_BLOCK_SIZE;
        let n_blocks = m_total.div_ceil(block_size);
        let ids = bench_ids(m_total, n_experts);
        let mut block_counts = vec![0u32; n_blocks * n_experts];
        for b in 0..n_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(m_total);
            for &id in &ids[start..end] {
                block_counts[b * n_experts + id as usize] += 1;
            }
        }
        let mut totals = vec![0u32; n_experts];
        for b in 0..n_blocks {
            for e in 0..n_experts {
                totals[e] += block_counts[b * n_experts + e];
            }
        }
        let mut base = vec![0u32; n_experts];
        for e in 1..n_experts {
            base[e] = base[e - 1] + totals[e - 1];
        }
        let mut block_offsets = vec![0u32; n_blocks * n_experts];
        let mut running = base;
        for b in 0..n_blocks {
            for e in 0..n_experts {
                block_offsets[b * n_experts + e] = running[e];
                running[e] += block_counts[b * n_experts + e];
            }
        }
        BenchSetup::new(ffai_moe_sort_plan_scatter::kernel_ir())
            .buffer(BenchBuffer::from_vec("topk_ids", u32_bytes(&ids), DType::U32))
            .buffer(BenchBuffer::from_vec("block_offsets", u32_bytes(&block_offsets), DType::U32))
            .buffer(BenchBuffer::zeros("sorted_experts", m_total, DType::U32).output())
            .buffer(BenchBuffer::zeros("source_tokens", m_total, DType::U32).output())
            .buffer(BenchBuffer::zeros("inv_perm", m_total, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("k", k as u32)
            .constexpr("n_experts", n_experts as u32)
            .constexpr("block_size", block_size as u32)
            .grid_1d(m_total, 256)
            .bytes_moved((m_total * 4 * 4) as u64)
    }
}

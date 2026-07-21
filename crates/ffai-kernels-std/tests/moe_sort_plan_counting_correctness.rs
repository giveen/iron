//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Byte-exact stable-order equality for the bucketed counting-sort
//! replacement of `ffai_moe_sort_plan` (`moe_sort_plan_counting.rs`):
//! `ffai_moe_sort_plan_hist` -> `ffai_moe_sort_plan_offsets` ->
//! `ffai_moe_sort_plan_scatter`.
//!
//! `ffai_moe_sort_plan`'s output feeds every downstream gather-GEMM, so
//! this is not "a valid sort" - it must be the SAME stable order,
//! byte-for-byte, ties broken by original index. Every case here checks
//! the three-pass pipeline against BOTH a CPU stable-sort oracle AND a
//! live GPU dispatch of the original `ffai_moe_sort_plan` kernel on the
//! identical input, so a divergence can never hide behind an oracle bug.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::moe::{
    moe_permute::ffai_moe_sort_plan,
    moe_sort_plan_counting::{
        ffai_moe_sort_plan_hist,
        ffai_moe_sort_plan_offsets,
        ffai_moe_sort_plan_scatter,
    },
};

fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Poison fill - any output byte the kernel fails to write shows up as
/// `0xDEAD_BEEF`, not a lucky zero match against the oracle.
const POISON: u32 = 0xDEAD_BEEF;
fn poisoned(n: usize) -> Vec<u8> { u32_bytes(&vec![POISON; n]) }

/// CPU stable-sort oracle: the exact semantics `ffai_moe_sort_plan` (and
/// this replacement) must reproduce.
fn cpu_oracle(ids: &[u32], k: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let m_total = ids.len();
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
    (se, st, ip)
}

/// Runs the ORIGINAL O(m_total^2) kernel on `ids`, one dispatch.
fn run_original(ctx: &Context, ids: &[u32], k: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let m_total = ids.len();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("topk_ids".into(), u32_bytes(ids));
    buffers.insert("sorted_experts".into(), poisoned(m_total));
    buffers.insert("source_tokens".into(), poisoned(m_total));
    buffers.insert("inv_perm".into(), poisoned(m_total));
    buffers.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let k_ir = ffai_moe_sort_plan::kernel_ir();
    let tg = m_total.min(256);
    let groups = m_total.div_ceil(tg.max(1));
    let r = ctx
        .dispatch_with_grid(&k_ir, &buffers, &BTreeMap::new(), [groups.max(1), 1, 1], [
            tg.max(1),
            1,
            1,
        ])
        .expect("original sort_plan dispatch");
    (
        unpack_u32(r.outputs.get("sorted_experts").unwrap()),
        unpack_u32(r.outputs.get("source_tokens").unwrap()),
        unpack_u32(r.outputs.get("inv_perm").unwrap()),
    )
}

/// Runs the new three-pass counting-sort pipeline on `ids`.
fn run_counting_sort(
    ctx: &Context,
    ids: &[u32],
    k: usize,
    n_experts: usize,
    block_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let m_total = ids.len();
    let n_blocks = m_total.div_ceil(block_size);

    // Pass 1: hist.
    let mut b1: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b1.insert("topk_ids".into(), u32_bytes(ids));
    b1.insert("block_counts".into(), poisoned(n_blocks * n_experts));
    b1.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    b1.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    b1.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    let mut k1 = ffai_moe_sort_plan_hist::kernel_ir();
    k1.mode = KernelMode::Reduction;
    let r1 = ctx
        .dispatch_with_grid(&k1, &b1, &BTreeMap::new(), [n_blocks, 1, 1], [n_experts, 1, 1])
        .expect("hist dispatch");
    let block_counts = r1.outputs.get("block_counts").unwrap().clone();

    // Pass 2: offsets.
    let mut b2: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b2.insert("block_counts".into(), block_counts);
    b2.insert("block_offsets".into(), poisoned(n_blocks * n_experts));
    b2.insert("n_blocks".into(), (n_blocks as u32).to_le_bytes().to_vec());
    b2.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    let mut k2 = ffai_moe_sort_plan_offsets::kernel_ir();
    k2.mode = KernelMode::Reduction;
    let r2 = ctx
        .dispatch_with_grid(&k2, &b2, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("offsets dispatch");
    let block_offsets = r2.outputs.get("block_offsets").unwrap().clone();

    // Pass 3: scatter.
    let mut b3: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b3.insert("topk_ids".into(), u32_bytes(ids));
    b3.insert("block_offsets".into(), block_offsets);
    b3.insert("sorted_experts".into(), poisoned(m_total));
    b3.insert("source_tokens".into(), poisoned(m_total));
    b3.insert("inv_perm".into(), poisoned(m_total));
    b3.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    b3.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    b3.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    b3.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    let k3 = ffai_moe_sort_plan_scatter::kernel_ir();
    let tg3 = m_total.min(256);
    let groups3 = m_total.div_ceil(tg3.max(1));
    let r3 = ctx
        .dispatch_with_grid(&k3, &b3, &BTreeMap::new(), [groups3.max(1), 1, 1], [tg3.max(1), 1, 1])
        .expect("scatter dispatch");
    (
        unpack_u32(r3.outputs.get("sorted_experts").unwrap()),
        unpack_u32(r3.outputs.get("source_tokens").unwrap()),
        unpack_u32(r3.outputs.get("inv_perm").unwrap()),
    )
}

/// Deterministic xorshift - no external RNG dependency needed for a
/// reproducible shuffle/fixture generator.
struct Xorshift(u64);
impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_below(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
}

/// Builds an ids array from per-expert counts, then shuffles rows into a
/// scattered (non-pre-sorted) original-token order - the shape the real
/// router hands the kernel, and the shape that actually stresses
/// stability (unlike a fixture already in expert order).
fn shuffled_ids(counts: &[usize], seed: u64) -> Vec<u32> {
    let mut ids = Vec::new();
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            ids.push(e as u32);
        }
    }
    let mut rng = Xorshift(seed | 1);
    let n = ids.len();
    for i in (1..n).rev() {
        let j = rng.next_below(i + 1);
        ids.swap(i, j);
    }
    ids
}

fn zipf_counts(m_total: usize, n_experts: usize, seed: u64) -> Vec<usize> {
    let mut weights: Vec<f64> = (0..n_experts)
        .map(|e| {
            let jitter = 1.0
                + 0.4
                    * (((e as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f64
                        / 1000.0
                        - 0.5);
            jitter / ((e + 1) as f64)
        })
        .collect();
    let total_w: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total_w;
    }
    let mut counts: Vec<usize> = weights.iter().map(|w| (w * m_total as f64) as usize).collect();
    let assigned: usize = counts.iter().sum();
    if assigned < m_total {
        counts[0] += m_total - assigned;
    } else if assigned > m_total {
        let over = assigned - m_total;
        counts[0] = counts[0].saturating_sub(over);
    }
    counts
}

/// Checks the counting-sort pipeline against both oracles for one
/// `(ids, k, n_experts, block_size)` fixture.
fn check_case(ids: &[u32], k: usize, n_experts: usize, block_size: usize, label: &str) {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");

    let (exp_se, exp_st, exp_ip) = cpu_oracle(ids, k);
    let (orig_se, orig_st, orig_ip) = run_original(&ctx, ids, k);
    assert_eq!(orig_se, exp_se, "{label}: original kernel diverges from CPU oracle (test bug?)");
    assert_eq!(orig_st, exp_st, "{label}: original kernel diverges from CPU oracle (test bug?)");
    assert_eq!(orig_ip, exp_ip, "{label}: original kernel diverges from CPU oracle (test bug?)");

    let (new_se, new_st, new_ip) = run_counting_sort(&ctx, ids, k, n_experts, block_size);
    assert_eq!(new_se, exp_se, "{label}: counting-sort sorted_experts diverges from CPU oracle");
    assert_eq!(new_st, exp_st, "{label}: counting-sort source_tokens diverges from CPU oracle");
    assert_eq!(new_ip, exp_ip, "{label}: counting-sort inv_perm diverges from CPU oracle");

    // Byte-exact against the CURRENT kernel's own GPU output, not just
    // "a valid sort" - the load-bearing check this campaign exists for.
    assert_eq!(
        new_se, orig_se,
        "{label}: sorted_experts diverges from ffai_moe_sort_plan's own output"
    );
    assert_eq!(
        new_st, orig_st,
        "{label}: source_tokens diverges from ffai_moe_sort_plan's own output"
    );
    assert_eq!(new_ip, orig_ip, "{label}: inv_perm diverges from ffai_moe_sort_plan's own output");
}

const K: usize = 8;
const N_EXPERTS: usize = 256;
const BLOCK_SIZE: usize = 256;

#[test]
fn uniform_routing_production_scale() {
    for &t in &[512usize, 1024] {
        let m_total = t * K;
        let ids: Vec<u32> = (0..m_total).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
        check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, &format!("uniform T={t}"));
    }
}

#[test]
fn zipf_skewed_routing() {
    for &t in &[512usize, 4096] {
        let m_total = t * K;
        let counts = zipf_counts(m_total, N_EXPERTS, 0x5EED_0001);
        let ids = shuffled_ids(&counts, 0x1234_5678);
        check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, &format!("zipf T={t}"));
    }
}

#[test]
fn zero_count_experts() {
    // Only every 5th expert gets any rows - the rest are hard zero.
    let n_experts = 40;
    let m_total = 2048;
    let mut counts = vec![0usize; n_experts];
    let mut remaining = m_total;
    let mut e = 0;
    while remaining > 0 {
        if e % 5 == 0 {
            let take = remaining.min(37);
            counts[e % n_experts] += take;
            remaining -= take;
        }
        e += 1;
    }
    let ids = shuffled_ids(&counts, 0xABCD_EF01);
    check_case(&ids, K, n_experts, 128, "zero-count experts");
}

#[test]
fn single_expert_takes_all() {
    for &m_total in &[64usize, 4096] {
        let ids = vec![7u32; m_total];
        check_case(&ids, K, 16, 128, &format!("single-expert-takes-all m={m_total}"));
    }
}

#[test]
fn m_total_not_multiple_of_block_size() {
    // block_size=256, m_total deliberately NOT a multiple of it.
    let m_total = 4100usize;
    let ids: Vec<u32> = (0..m_total).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
    check_case(&ids, K, N_EXPERTS, 256, "m_total not multiple of block_size (4100/256)");
    // Also check a block_size that doesn't divide a "round" m_total.
    let m_total2 = 4096usize;
    let ids2: Vec<u32> = (0..m_total2).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
    check_case(&ids2, K, N_EXPERTS, 300, "block_size not a divisor of m_total (4096/300)");
}

#[test]
fn tiny_m_total() {
    for &m_total in &[8usize, 32] {
        let ids: Vec<u32> = (0..m_total).map(|i| ((i * 7 + 3) % 5) as u32).collect();
        check_case(&ids, 2, 8, 256, &format!("tiny m_total={m_total}"));
    }
}

#[test]
fn production_scale_uniform_and_zipf() {
    for &t in &[512usize, 1024, 2048, 4096] {
        let m_total = t * K;
        let ids: Vec<u32> = (0..m_total).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
        check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, &format!("production uniform T={t}"));

        let counts = zipf_counts(m_total, N_EXPERTS, 0x9E37_0001u64.wrapping_add(t as u64));
        let zids = shuffled_ids(&counts, 0x2468_ACE0u64.wrapping_add(t as u64));
        check_case(&zids, K, N_EXPERTS, BLOCK_SIZE, &format!("production zipf T={t}"));
    }
}

#[test]
fn deterministic_repeat_dispatch() {
    // Same input dispatched twice must produce bit-identical output -
    // no reliance on undefined-order atomics or nondeterministic
    // scheduling (this design has no atomics at all, but this is the
    // cheapest possible guard against a regression that introduces one).
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let counts = zipf_counts(4096, N_EXPERTS, 0x42);
    let ids = shuffled_ids(&counts, 0x99);
    let (se1, st1, ip1) = run_counting_sort(&ctx, &ids, K, N_EXPERTS, BLOCK_SIZE);
    let (se2, st2, ip2) = run_counting_sort(&ctx, &ids, K, N_EXPERTS, BLOCK_SIZE);
    assert_eq!(se1, se2, "sorted_experts nondeterministic across identical dispatches");
    assert_eq!(st1, st2, "source_tokens nondeterministic across identical dispatches");
    assert_eq!(ip1, ip2, "inv_perm nondeterministic across identical dispatches");
}

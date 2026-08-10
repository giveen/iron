//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Byte-exact stable-order equality for the single-dispatch fusion of the
//! `moe_sort_plan_counting.rs` three-pass chain:
//! `iron_moe_route_sort_plan_fused` (`moe_route_sort_plan_fused.rs`) vs.
//! `iron_moe_sort_plan_hist` -> `_offsets` -> `_scatter`.
//!
//! Every case checks the fused kernel's single dispatch against BOTH a CPU
//! stable-sort oracle AND a live GPU dispatch of the three-pass chain on
//! the identical input, so a divergence can never hide behind an oracle
//! bug. The fused kernel is the first kernel in this MoE route/sort family
//! to use threadgroup atomics (`atomic_add_tg`, for the cooperative
//! full-array histogram — see the kernel's module doc), so this file also
//! carries an explicit determinism regression guard (repeat dispatch, same
//! input, must be bit-identical).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::gpu_lock;
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::moe::{
    moe_mpp_shared::zipfish_counts,
    moe_route_sort_plan_fused::iron_moe_route_sort_plan_fused,
    moe_sort_plan_counting::{
        iron_moe_sort_plan_hist,
        iron_moe_sort_plan_offsets,
        iron_moe_sort_plan_scatter,
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

/// CPU stable-sort oracle: the exact semantics every implementation in
/// this family must reproduce.
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

/// Runs the existing three-pass counting-sort chain on `ids` (the
/// production default path this kernel replaces).
fn run_three_pass(
    ctx: &Context,
    ids: &[u32],
    k: usize,
    n_experts: usize,
    block_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let m_total = ids.len();
    let n_blocks = m_total.div_ceil(block_size);

    let mut b1: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b1.insert("topk_ids".into(), u32_bytes(ids));
    b1.insert("block_counts".into(), poisoned(n_blocks * n_experts));
    b1.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    b1.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    b1.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    let mut k1 = iron_moe_sort_plan_hist::kernel_ir();
    k1.mode = KernelMode::Reduction;
    let r1 = ctx
        .dispatch_with_grid(&k1, &b1, &BTreeMap::new(), [n_blocks, 1, 1], [n_experts, 1, 1])
        .expect("hist dispatch");
    let block_counts = r1.outputs.get("block_counts").unwrap().clone();

    let mut b2: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b2.insert("block_counts".into(), block_counts);
    b2.insert("block_offsets".into(), poisoned(n_blocks * n_experts));
    b2.insert("n_blocks".into(), (n_blocks as u32).to_le_bytes().to_vec());
    b2.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    let mut k2 = iron_moe_sort_plan_offsets::kernel_ir();
    k2.mode = KernelMode::Reduction;
    let r2 = ctx
        .dispatch_with_grid(&k2, &b2, &BTreeMap::new(), [1, 1, 1], [n_experts, 1, 1])
        .expect("offsets dispatch");
    let block_offsets = r2.outputs.get("block_offsets").unwrap().clone();

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
    let k3 = iron_moe_sort_plan_scatter::kernel_ir();
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

/// Runs the fused single-dispatch kernel on `ids`.
fn run_fused(
    ctx: &Context,
    ids: &[u32],
    k: usize,
    n_experts: usize,
    block_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let m_total = ids.len();
    let n_blocks = m_total.div_ceil(block_size);

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("topk_ids".into(), u32_bytes(ids));
    b.insert("sorted_experts".into(), poisoned(m_total));
    b.insert("source_tokens".into(), poisoned(m_total));
    b.insert("inv_perm".into(), poisoned(m_total));
    b.insert("m_total".into(), (m_total as u32).to_le_bytes().to_vec());
    b.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    b.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    b.insert("block_size".into(), (block_size as u32).to_le_bytes().to_vec());
    let mut kf = iron_moe_route_sort_plan_fused::kernel_ir();
    kf.mode = KernelMode::Reduction;
    let r = ctx
        .dispatch_with_grid(&kf, &b, &BTreeMap::new(), [n_blocks.max(1), 1, 1], [n_experts, 1, 1])
        .expect("fused dispatch");
    (
        unpack_u32(r.outputs.get("sorted_experts").unwrap()),
        unpack_u32(r.outputs.get("source_tokens").unwrap()),
        unpack_u32(r.outputs.get("inv_perm").unwrap()),
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

/// Checks the fused kernel against both oracles for one
/// `(ids, k, n_experts, block_size)` fixture.
fn check_case(ids: &[u32], k: usize, n_experts: usize, block_size: usize, label: &str) {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");

    let (exp_se, exp_st, exp_ip) = cpu_oracle(ids, k);
    let (chain_se, chain_st, chain_ip) = run_three_pass(&ctx, ids, k, n_experts, block_size);
    assert_eq!(chain_se, exp_se, "{label}: three-pass chain diverges from CPU oracle (test bug?)");
    assert_eq!(chain_st, exp_st, "{label}: three-pass chain diverges from CPU oracle (test bug?)");
    assert_eq!(chain_ip, exp_ip, "{label}: three-pass chain diverges from CPU oracle (test bug?)");

    let (fused_se, fused_st, fused_ip) = run_fused(&ctx, ids, k, n_experts, block_size);
    assert_eq!(fused_se, exp_se, "{label}: fused sorted_experts diverges from CPU oracle");
    assert_eq!(fused_st, exp_st, "{label}: fused source_tokens diverges from CPU oracle");
    assert_eq!(fused_ip, exp_ip, "{label}: fused inv_perm diverges from CPU oracle");

    // Byte-exact against the three-pass chain's own GPU output, not just
    // "a valid sort" - the load-bearing check this fusion exists for.
    assert_eq!(
        fused_se, chain_se,
        "{label}: sorted_experts diverges from the three-pass chain's own output"
    );
    assert_eq!(
        fused_st, chain_st,
        "{label}: source_tokens diverges from the three-pass chain's own output"
    );
    assert_eq!(
        fused_ip, chain_ip,
        "{label}: inv_perm diverges from the three-pass chain's own output"
    );
}

const K: usize = 8;
const N_EXPERTS: usize = 256;
const BLOCK_SIZE: usize = 256;

/// Laguna's production shapes, per the Stage-1 fusion gate: T=1 (decode),
/// 512/1042/2048 (prefill chunk sizes), uniform routing.
#[test]
fn uniform_routing_laguna_shapes() {
    for &t in &[1usize, 512, 1042, 2048] {
        let m_total = t * K;
        let ids: Vec<u32> = (0..m_total).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
        check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, &format!("uniform T={t}"));
    }
}

#[test]
fn zipf_skewed_routing_laguna_shapes() {
    for &t in &[1usize, 512, 1042, 2048] {
        let m_total = t * K;
        let counts = zipfish_counts(m_total, N_EXPERTS, 0x5EED_0002u64.wrapping_add(t as u64));
        let ids = shuffled_ids(&counts, 0x1234_5679u64.wrapping_add(t as u64));
        check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, &format!("zipf T={t}"));
    }
}

#[test]
fn zero_count_experts() {
    // Only every 5th expert gets any rows - the rest are hard zero, and
    // sit at Laguna's real n_experts/block_size (256/256) so `total[e]`
    // for a zero-count expert must come out exactly 0 without any lane
    // ever touching its atomic counter.
    let n_experts = 256;
    for &t in &[512usize, 2048] {
        let m_total = t * K;
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
        let ids = shuffled_ids(&counts, 0xABCD_EF02u64.wrapping_add(t as u64));
        check_case(&ids, K, n_experts, BLOCK_SIZE, &format!("zero-count experts T={t}"));
    }
}

#[test]
fn single_expert_takes_all() {
    for &m_total in &[8usize, 4096, 16384] {
        let ids = vec![7u32; m_total];
        check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, &format!("single-expert-takes-all m={m_total}"));
    }
}

#[test]
fn t_equals_one_decode() {
    // T=1, k=8: m_total=8, n_blocks=1 - the decode-shaped edge case where
    // the whole plan is one threadgroup's own block (no "earlier blocks"
    // to count, block_offset degenerates to just `base`).
    let ids: Vec<u32> = vec![3, 200, 3, 0, 255, 3, 128, 0];
    check_case(&ids, K, N_EXPERTS, BLOCK_SIZE, "T=1 decode");
}

#[test]
fn m_total_not_multiple_of_block_size() {
    let m_total = 4100usize;
    let ids: Vec<u32> = (0..m_total).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
    check_case(&ids, K, N_EXPERTS, 256, "m_total not multiple of block_size (4100/256)");
    let m_total2 = 8336usize; // 1042 * 8, task's explicit T=1042 shape
    let ids2: Vec<u32> = (0..m_total2).map(|i| ((i * 2654435761) % N_EXPERTS) as u32).collect();
    check_case(&ids2, K, N_EXPERTS, 256, "T=1042 (8336/256)");
}

/// The fused kernel is the first in this family to use threadgroup
/// atomics (`atomic_add_tg`, for the cooperative histogram) - unlike the
/// three-pass chain, which has none. Same input dispatched twice must
/// still produce bit-identical output: sums are commutative, so this
/// should hold regardless of scheduling, but it is the cheapest possible
/// guard against a regression that makes the histogram order-dependent.
#[test]
fn deterministic_repeat_dispatch() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context");
    let counts = zipfish_counts(16384, N_EXPERTS, 0x43);
    let ids = shuffled_ids(&counts, 0x9A);
    let (se1, st1, ip1) = run_fused(&ctx, &ids, K, N_EXPERTS, BLOCK_SIZE);
    let (se2, st2, ip2) = run_fused(&ctx, &ids, K, N_EXPERTS, BLOCK_SIZE);
    assert_eq!(se1, se2, "sorted_experts nondeterministic across identical dispatches");
    assert_eq!(st1, st2, "source_tokens nondeterministic across identical dispatches");
    assert_eq!(ip1, ip2, "inv_perm nondeterministic across identical dispatches");
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MoE router top-k expert selection — `iron_moe_router_topk` picks the top-k
//! experts per token by logit and emits the normalised routing weights
//! (softmax over chosen-k, or global softmax), on-device so routing never
//! round-trips to the CPU. The biased/unbiased dual-score variant lives in
//! `router_topk_biased.rs`.

use wh_iron::kernel;

// ── iron_moe_router_topk ───────────────────────────────────────────────────
//
// Per-token select top-k experts from `router_logits`, plus softmax
// weights over the chosen k.
//
// Inputs:
//   router_logits — [B*T, n_experts]  (any float dtype, computed in f32)
//   indices_out   — [B*T, k]          (u32)
//   weights_out   — [B*T, k]          (same dtype as router_logits, softmax weights)
//
// Constexpr:
//   n_experts   — typical Qwen3.6-A3B: 128.  Must fit one simdgroup
//                 (≤ 32×32 = 1024) — every reasonable MoE topology.
//   k           — typical 6-8 for production MoE.  Hard cap k ≤ 32.
//
// Geometry:
//   tpg=32  (one simdgroup per token row)
//   grid = [B*T, 1, 1]  (Reduction mode)
//
// Algorithm — k iterations of simd-parallel argmax with mask of
// previously-chosen indices stored in TG memory.  After k passes,
// softmax over the chosen k values in-place on lane 0..k-1.
//
// Bench spec uses BenchDispatch::Generic + shapes: &[] so `iron bench`
// skips it; correctness lives in unit tests + downstream MoE
// integration. Same convention as other iron/ kernels (gather, sampling).
#[kernel]
pub fn iron_moe_router_topk<T>(
    router_logits: Tensor<T>,
    mut indices_out: Tensor<u32>,
    mut weights_out: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] k: u32,
    // 1 = Qwen3-MoE style (softmax over chosen-k, sum-to-1 — `norm_topk_prob=True`)
    // 0 = Qwen3-Next style (softmax over ALL n_experts, return chosen probs
    //     un-renormalized — `norm_topk_prob=False`)
    // Mathematically equivalent at mode 1: softmax-over-chosen-k is the
    // same as (softmax-over-all → renormalize-over-chosen). Mode 0
    // returns probs that sum to < 1 across the chosen k, matching MLX's
    // qwen3_next.py:334-341.
    //
    // INVARIANT: this kernel pins tpg=32 (one simdgroup per token row).
    // The `simdgroup_barrier_mem_none()` below is correct only at tpg=32.
    // Caller must dispatch with `[n_rows, 1, 1] × [32, 1, 1]`.
    #[constexpr] norm_topk_prob: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;
    // TG scratch: chosen indices + values from each of the k argmax passes.
    // 32 slots covers any reasonable k (typical 6-8). Kernel assumes
    // k ≤ 32 — caller MUST enforce this in the host-side dispatcher
    // (no GPU-side check, would silently scribble into adjacent TG mem).
    threadgroup_alloc("tg_chosen_idx", 32u32);
    threadgroup_alloc("tg_chosen_val", 32u32);
    // Cache the all-experts-softmax sum for Qwen3-Next mode (mode 0).
    // 1 slot, written by lane 0 in the prepass.
    threadgroup_alloc("tg_full_sum", 1u32);
    threadgroup_alloc("tg_full_max", 1u32);
    // ── Pre-pass: compute softmax denominator over ALL n_experts ─────
    // Needed only for norm_topk_prob=0 (Qwen3-Next), but the cost is
    // trivial (one simd_max + simd_sum) and emitting it unconditionally
    // keeps the codegen tight (the codegen DCE will drop the dead path
    // when the constexpr branch is unreachable).
    let mut local_max_all = neg_infinity();
    let n_per_lane_pre = (n_experts + 31u32) / 32u32;
    for r in range(0u32, n_per_lane_pre, 1u32) {
        let j = r * 32u32 + lane;
        if j < n_experts {
            let v = load(router_logits[row_base + j]).cast::<f32>();
            let better = v > local_max_all;
            local_max_all = select(better, v, local_max_all);
        }
    }
    let row_max_all = simd_max(local_max_all);
    let mut local_sum_all = 0.0f32;
    for r in range(0u32, n_per_lane_pre, 1u32) {
        let j = r * 32u32 + lane;
        if j < n_experts {
            let v = load(router_logits[row_base + j]).cast::<f32>();
            local_sum_all = local_sum_all + exp(v - row_max_all);
        }
    }
    let row_sum_all = simd_sum(local_sum_all);
    if lane == 0u32 {
        threadgroup_store("tg_full_max", 0u32, row_max_all);
        threadgroup_store("tg_full_sum", 0u32, row_sum_all);
    }
    simdgroup_barrier_mem_none();
    // ── k argmax passes with chosen-mask ─────────────────────────────
    for it in range(0u32, k, 1u32) {
        // Per-lane local argmax over its slice of n_experts.
        // Each lane covers ceil(n_experts/32) experts.
        let mut best_val = neg_infinity();
        let mut best_idx = 0u32;
        let n_per_lane = (n_experts + 31u32) / 32u32;
        for r in range(0u32, n_per_lane, 1u32) {
            let j = r * 32u32 + lane;
            if j < n_experts {
                let v = load(router_logits[row_base + j]).cast::<f32>();
                // Mask: was j picked in a previous iter?
                // Scan tg_chosen_idx[0..it] — k ≤ 8 typically so this
                // is fast even without early exit.
                let mut chosen_mask = 0u32;
                for p in range(0u32, it, 1u32) {
                    let cp = threadgroup_load("tg_chosen_idx", p);
                    chosen_mask = chosen_mask | select(j == cp, 1u32, 0u32);
                }
                let candidate = select(chosen_mask > 0u32, neg_infinity(), v);
                let better = candidate > best_val;
                best_val = select(better, candidate, best_val);
                best_idx = select(better, j, best_idx);
            }
        }
        // Cross-lane reduce.  simd_max gives the global best value;
        // ties broken to smaller idx via simd_min on (idx | sentinel).
        let global_best_val = simd_max(best_val);
        let i_have = best_val == global_best_val;
        let my_idx_or_max = select(i_have, best_idx, 4294967295u32); // u32::MAX
        let global_best_idx = simd_min(my_idx_or_max);
        // Lane 0 writes the iter's chosen slot.
        if lane == 0u32 {
            threadgroup_store("tg_chosen_idx", it, global_best_idx);
            threadgroup_store("tg_chosen_val", it, global_best_val);
        }
        simdgroup_barrier_mem_none();
    }
    // ── Softmax / weight emit per `norm_topk_prob` ──────────────────
    // Mode 1 (Qwen3-MoE, default): softmax over chosen-k (sum-to-1).
    //   numerator   = exp(z_i - max_chosen);  divisor = Σ_j∈chosen
    //   == exp(z_i - max_all) · const / Σ_j∈chosen exp(z_j - max_all) · const
    //   so we can use the SAME numerator as mode 0 (exp(z - max_all)) and
    //   just swap the divisor.  Avoids needing a Rust `if`-expression
    //   which the DSL doesn't unify across arms.
    // Mode 0 (Qwen3-Next): un-normalized chosen probs (sum < 1).
    //   weight_i = exp(z_i - max_all) / Σ_j∈all exp(z_j - max_all)
    let my_val = select(lane < k, threadgroup_load("tg_chosen_val", lane), neg_infinity());
    let row_max_full = threadgroup_load("tg_full_max", 0u32);
    let row_sum_full = threadgroup_load("tg_full_sum", 0u32);
    let exp_val = exp(my_val - row_max_full);
    let masked_exp = select(lane < k, exp_val, 0.0f32);
    let sum_chosen = simd_sum(masked_exp);
    // Pick divisor: chosen-k sum for renormalized (mode 1) or all-experts
    // sum for raw probs (mode 0). select() forces both to be live; codegen
    // const-folds when `norm_topk_prob` bakes in.
    let divisor = select(norm_topk_prob == 1u32, sum_chosen, row_sum_full);
    let weight = masked_exp / divisor;
    // ── Write outputs ───────────────────────────────────────────────
    if lane < k {
        let out_base = row * k + lane;
        store(indices_out[out_base], threadgroup_load("tg_chosen_idx", lane));
        store(weights_out[out_base], weight.cast::<T>());
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // ── Router top-k selection ─────────────────────────────────────────────────────────────

    /// Router oracle: softmax over all experts for the denominator, pick the
    /// top-k logits (well-separated test inputs → no ties), then weight either
    /// by renormalised softmax over the chosen k (`norm_topk_prob`) or by the
    /// global softmax (raw probs that sum to < 1).
    fn router_oracle(
        logits: &[f32],
        n_rows: usize,
        n_experts: usize,
        k: usize,
        norm_topk_prob: bool,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut idx_out = vec![0u32; n_rows * k];
        let mut w_out = vec![0.0f32; n_rows * k];
        for row in 0..n_rows {
            let row_l = &logits[row * n_experts..(row + 1) * n_experts];
            let max_all = row_l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_all: f32 = row_l.iter().map(|&l| (l - max_all).exp()).sum();
            // Top-k by descending logit (stable: smaller index wins ties).
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| row_l[b].partial_cmp(&row_l[a]).unwrap().then(a.cmp(&b)));
            let chosen = &order[..k];
            let sum_chosen: f32 = chosen.iter().map(|&e| (row_l[e] - max_all).exp()).sum();
            for (i, &e) in chosen.iter().enumerate() {
                idx_out[row * k + i] = e as u32;
                let num = (row_l[e] - max_all).exp();
                w_out[row * k + i] = if norm_topk_prob { num / sum_chosen } else { num / sum_all };
            }
        }
        (idx_out, w_out)
    }

    fn router_setup(dt: DType, norm_topk_prob: bool) -> TestSetup {
        let (n_rows, n_experts, k) = (4usize, 8usize, 4usize);
        // Well-separated logits (distinct multiples of 0.5 per row → no ties,
        // gap ≫ dtype epsilon so the selection is dtype-stable).
        let logits_f: Vec<f32> = (0..n_rows * n_experts)
            .map(|i| {
                let row = i / n_experts;
                let e = i % n_experts;
                ((e * 5 + row * 3) % n_experts) as f32 * 0.5
            })
            .collect();
        let logits = unpack_f32(&pack_f32(&logits_f, dt), dt);
        let (idx, w) = router_oracle(&logits, n_rows, n_experts, k, norm_topk_prob);
        TestSetup::new(iron_moe_router_topk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("router_logits", pack_f32(&logits_f, dt), dt))
            .input(TestBuffer::zeros("indices_out", n_rows * k, DType::U32))
            .input(TestBuffer::zeros("weights_out", n_rows * k, dt))
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .constexpr("norm_topk_prob", u32::from(norm_topk_prob))
            .expect(TestBuffer::from_vec("indices_out", u32_bytes(&idx), DType::U32))
            .expect(TestBuffer::from_vec("weights_out", pack_f32(&w, dt), dt))
            .grid_3d(n_rows as u32, 1, 1, [32, 1, 1])
    }

    // norm_topk_prob = 1: weights renormalised over the chosen k (Qwen3-MoE).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_moe_router_topk_norm(dt: DType) -> TestSetup { router_setup(dt, true) }
    // norm_topk_prob = 0: raw global-softmax probs (Qwen3-Next).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_moe_router_topk_global(dt: DType) -> TestSetup { router_setup(dt, false) }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::*;

    // ── router_topk — data-dependent argmax, bench-only ───────────────────
    // ABI: router_logits, indices_out, weights_out + {n_experts, k,
    // norm_topk_prob}. Grid [B*T, 1, 1], tpg [32,1,1] (pinned in the doc).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_router_topk(dt: DType) -> BenchSetup {
        let n_rows = 4096usize; // B*T
        let n_experts = 128usize;
        let k = 8usize;
        let sz = dt.size_bytes();
        let bytes = n_rows * n_experts * sz + n_rows * k * 4 + n_rows * k * sz;
        BenchSetup::new(iron_moe_router_topk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("router_logits", n_rows * n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", n_rows * k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", n_rows * k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .constexpr("norm_topk_prob", 1u32)
            .with_shape_label(format!(
                "BT{n_rows} E{n_experts} k{k} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_rows as u32, 1, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
    }

    /// Softmax-router bench at Hy3 width (E=192, k=8). Hy3 itself uses a
    /// sigmoid+bias scorer (`iron_moe_sigmoid_bias` / `iron_moe_router_sigmoid_bias`);
    /// this keeps the shared top-k geometry warm for prefill-scale BT.
    #[bench(dtypes = [f32])]
    fn bench_moe_router_topk_hy3_width(dt: DType) -> BenchSetup {
        let n_rows = 512usize;
        let n_experts = 192usize;
        let k = 8usize;
        let sz = dt.size_bytes();
        let bytes = n_rows * n_experts * sz + n_rows * k * 4 + n_rows * k * sz;
        BenchSetup::new(iron_moe_router_topk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("router_logits", n_rows * n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", n_rows * k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", n_rows * k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .constexpr("norm_topk_prob", 1u32)
            .with_shape_label(format!(
                "hy3width BT{n_rows} E{n_experts} k{k} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_rows as u32, 1, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
    }
}

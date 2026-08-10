//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Single-dispatch fusion of the Laguna Path B prefill router's two-kernel
//! chain: `iron_moe_sigmoid_bias_rows` (T·E-wide sigmoid+bias pre-score,
//! materializes `unbiased`/`biased` score matrices) followed by
//! `iron_moe_router_topk_biased` (T-row masked-argmax top-K selection over
//! the materialized `biased` scores, weights from `unbiased`). This kernel
//! removes both the intermediate `[T, n_experts]` score matrices AND one of
//! the two dispatches, computing sigmoid+bias inline per masked-argmax pass
//! exactly the way `LagunaRouterTop8Fuse`'s decode-only fusion (Butter,
//! `Sources/ButterSwift/Models/Text/LagunaRouterTop8Fuse.swift`) already
//! does for the single-row case — this is that same fusion, T-batched.
//!
//! ## Exactness — 2-stage rounding replication
//!
//! The two-kernel chain it replaces round-trips through `T` at each stage:
//!   1. `unbiased[t,e] = T(sigmoid(logits[t,e]))`             (bf16 store)
//!   2. `biased[t,e]   = T(float(unbiased[t,e]) + float(bias[e]))`  (bf16 store,
//!      reads the ALREADY-ROUNDED `unbiased`, not raw sigmoid)
//!   3. selection: masked-argmax over `float(biased[t,e])` (`iron_moe_router_topk_biased`)
//!   4. weight: `float(unbiased[t,chosen]) / Σ float(unbiased[t,chosen])`, `T`-rounded
//!
//! This kernel reproduces every stage's rounding exactly rather than
//! computing everything in one wide float and rounding once at the end
//! (the same "replicate every intermediate round" discipline
//! `LagunaRouterTop8Fuse` documents and that `LagunaRouteFuse`'s earlier
//! float-only attempt got wrong): each masked-argmax pass recomputes
//! `sig -> T(sig) -> T(float(T(sig)) + float(bias[e]))` inline per
//! candidate instead of reading a precomputed table, and the final weight
//! step recomputes `T(sig)` at the chosen index again — a pure function of
//! the (unchanged) input logit, so recomputing is bit-identical to caching
//! it. `n_per_lane` redundant `exp()` calls per pass is the trade for one
//! fewer dispatch and no `[T, n_experts]` intermediate materialization; see
//! the sibling `iron_moe_router_topk_biased`'s module doc for why the
//! per-lane masked-argmax structure itself is exact against a stable
//! descending sort with smallest-index tie-break.
//!
//! Sigmoid formula matches `iron_moe_sigmoid_bias_rows` exactly — the plain
//! `1 / (1 + exp(-x))` form, NOT the `abs`-stabilized form
//! `LagunaRouterTop8Fuse`'s DECODE kernel uses (those are only bit-identical
//! near x=0; this file fuses the PREFILL chain, so it must match the
//! PREFILL sigmoid kernel's own formula, not decode's).
//!
//! ## Router fusion re-evaluated (see W61's `moe_route_sort_plan_fused.rs`)
//!
//! W61 evaluated folding this fusion INTO the sort-plan kernel and declined
//! (different parallelism axis — per-token vs per-expert). This kernel does
//! NOT attempt that; it only merges the two router-chain kernels, which
//! already share the same per-token/per-row parallelism axis and the same
//! `[n_rows,1,1] x [32,1,1]` dispatch shape as `iron_moe_router_topk_biased`
//! — a clean merge, unlike the sort-plan case.
//!
//! ## Dispatch invariants
//!
//! Mode `Reduction`; grid `[n_rows, 1, 1]` (one threadgroup per token row,
//! SAME as `iron_moe_router_topk_biased`); threadgroup `[32, 1, 1]` (one
//! simdgroup). `logits`/`bias` are the RAW router GEMV outputs (no
//! precomputed score tensors) — `logits: [n_rows * n_experts]`,
//! `bias: [n_experts]` (tiled across rows), `indices_out`/`weights_out:
//! [n_rows * k]`. `routed_scaling_factor` is NOT applied here (same
//! contract as `iron_moe_router_topk_biased`) — the caller multiplies
//! after, unchanged.

use wh_iron::kernel;

#[kernel]
pub fn iron_moe_router_topk_biased_rows_fused<T>(
    logits: Tensor<T>,
    bias: Tensor<T>,
    mut indices_out: Tensor<u32>,
    mut weights_out: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] k: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;
    threadgroup_alloc("tg_chosen_idx", 32u32);

    // ── k masked-argmax passes over the inline-computed biased score ──────
    for it in range(0u32, k, 1u32) {
        let mut best_val = neg_infinity();
        let mut best_idx = 0u32;
        let n_per_lane = (n_experts + 31u32) / 32u32;
        for r in range(0u32, n_per_lane, 1u32) {
            let j = r * 32u32 + lane;
            if j < n_experts {
                // Stage 1 (`iron_moe_sigmoid_bias_rows`'s `unbiased` store)
                // + stage 2 (its `biased` store): round to `T` at EACH
                // stage, matching the two-kernel chain's two separate
                // stores exactly — same formula as that kernel
                // (`1 / (1 + exp(-x))`, not the abs-stabilized decode form).
                let x = load(logits[row_base + j]).cast::<f32>();
                let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - x));
                let unbiased_t = sig.cast::<T>();
                let biased_t = (unbiased_t.cast::<f32>() + load(bias[j]).cast::<f32>()).cast::<T>();
                let v = biased_t.cast::<f32>();

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
        let global_best_val = simd_max(best_val);
        let i_have = best_val == global_best_val;
        let my_idx_or_max = select(i_have, best_idx, 4294967295u32);
        let global_best_idx = simd_min(my_idx_or_max);
        if lane == 0u32 {
            threadgroup_store("tg_chosen_idx", it, global_best_idx);
        }
        simdgroup_barrier_mem_none();
    }

    // ── weights = unbiased[chosen] / Σ unbiased[chosen] ────────────────
    // Recompute `T(sigmoid(logits[chosen]))` — bit-identical to the
    // two-kernel chain's `unbiased[row_base + my_idx]` memory read, since
    // it is the same deterministic formula over the same (unchanged)
    // input logit.
    let my_idx_f = select(lane < k, threadgroup_load("tg_chosen_idx", lane), 0.0f32);
    let my_idx = my_idx_f.cast::<u32>();
    let my_x = load(logits[row_base + my_idx]).cast::<f32>();
    let my_sig = 1.0f32 / (1.0f32 + exp(0.0f32 - my_x));
    let my_unbiased_t = my_sig.cast::<T>();
    let my_unbiased = select(lane < k, my_unbiased_t.cast::<f32>(), 0.0f32);
    let sum_chosen = simd_sum(my_unbiased);
    let weight = my_unbiased / sum_chosen;
    if lane < k {
        let out_base = row * k + lane;
        store(indices_out[out_base], my_idx);
        store(weights_out[out_base], weight.cast::<T>());
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_moe_router_topk_biased_rows_fused;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// CPU oracle replicating the exact two-stage-rounding chain this
    /// kernel fuses: sigmoid -> round(T) -> +bias -> round(T) -> select ->
    /// renorm(unbiased) -> round(T). `dt` controls the rounding width via
    /// pack/unpack through that dtype (same trick `moe_sigmoid_bias_rows.rs`
    /// and `moe_router_topk_biased.rs` use).
    fn chain_oracle(
        logits: &[f32],
        bias: &[f32],
        t_rows: usize,
        n_experts: usize,
        k: usize,
        dt: DType,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut idx_out = vec![0u32; t_rows * k];
        let mut w_out = vec![0.0f32; t_rows * k];
        for t in 0..t_rows {
            let row = &logits[t * n_experts..(t + 1) * n_experts];
            let unbiased: Vec<f32> = row
                .iter()
                .map(|&x| {
                    let s = 1.0f32 / (1.0f32 + (-x).exp());
                    unpack_f32(&pack_f32(&[s], dt), dt)[0]
                })
                .collect();
            let biased: Vec<f32> = unbiased
                .iter()
                .zip(bias.iter())
                .map(|(&u, &b)| unpack_f32(&pack_f32(&[u + b], dt), dt)[0])
                .collect();
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap().then(a.cmp(&b)));
            let chosen = &order[..k];
            let sum: f32 = chosen.iter().map(|&e| unbiased[e]).sum();
            for (slot, &e) in chosen.iter().enumerate() {
                idx_out[t * k + slot] = e as u32;
                let w = unbiased[e] / sum;
                w_out[t * k + slot] = unpack_f32(&pack_f32(&[w], dt), dt)[0];
            }
        }
        (idx_out, w_out)
    }

    fn setup(dt: DType, t_rows: usize, n_experts: usize, k: usize, logits: Vec<f32>) -> TestSetup {
        let bias: Vec<f32> = (0..n_experts).map(|i| (i % 7) as f32 * 0.05 - 0.15).collect();
        let l_dt = unpack_f32(&pack_f32(&logits, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&bias, dt), dt);
        let (idx, w) = chain_oracle(&l_dt, &b_dt, t_rows, n_experts, k, dt);
        TestSetup::new(iron_moe_router_topk_biased_rows_fused::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("logits", pack_f32(&logits, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("indices_out", t_rows * k, DType::U32))
            .input(TestBuffer::zeros("weights_out", t_rows * k, dt))
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("indices_out", u32_bytes(&idx), DType::U32))
            .expect(TestBuffer::from_vec("weights_out", pack_f32(&w, dt), dt))
            .grid_3d(t_rows as u32, 1, 1, [32, 1, 1])
    }

    /// Uniform-ish spread logits, well-separated post-sigmoid scores —
    /// Laguna's production shape (256 experts, top-8), multi-row.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 5e-2])]
    fn test_moe_router_topk_biased_rows_fused_uniform(dt: DType) -> TestSetup {
        let (t_rows, n_experts, k) = (4usize, 256usize, 8usize);
        let logits: Vec<f32> =
            (0..t_rows * n_experts).map(|i| ((i * 37 + 11) % 251) as f32 * 0.05 - 6.0).collect();
        setup(dt, t_rows, n_experts, k, logits)
    }

    /// Zipf-shaped logit magnitudes (a few large positive, long negative
    /// tail) — exercises the sigmoid saturation ends, unlike the uniform
    /// fixture above.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 5e-2])]
    fn test_moe_router_topk_biased_rows_fused_zipf(dt: DType) -> TestSetup {
        let (t_rows, n_experts, k) = (4usize, 256usize, 8usize);
        let logits: Vec<f32> = (0..t_rows * n_experts)
            .map(|i| {
                let e = (i % n_experts) as f32;
                10.0 / (e + 1.0) - 4.0
            })
            .collect();
        setup(dt, t_rows, n_experts, k, logits)
    }

    /// Tied logits at the top-K boundary (several experts share the exact
    /// same post-round biased score, cut lands inside the tie) — the sharp
    /// test of the masked-argmax tie-break (smallest index wins) surviving
    /// the inline sigmoid+bias recompute.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 5e-2])]
    fn test_moe_router_topk_biased_rows_fused_tied(dt: DType) -> TestSetup {
        let (t_rows, n_experts, k) = (2usize, 32usize, 6usize);
        let mut logits = vec![0.0f32; t_rows * n_experts];
        for t in 0..t_rows {
            for e in 0..n_experts {
                logits[t * n_experts + e] = if (4..12).contains(&e) {
                    2.0 // ties: several experts share this exact logit
                } else if e < 4 {
                    5.0 - (e as f32) * 0.1 // strictly above the tie
                } else {
                    -2.0 - (e as f32) * 0.01 // strictly below the tie
                };
            }
        }
        setup(dt, t_rows, n_experts, k, logits)
    }

    /// Single-row (T=1) sanity check against the same shape
    /// `iron_moe_router_topk_biased`'s own tests use.
    #[test_kernel(dtypes = [f32], tol = [1e-4])]
    fn test_moe_router_topk_biased_rows_fused_single_row(dt: DType) -> TestSetup {
        let (t_rows, n_experts, k) = (1usize, 256usize, 6usize);
        let logits: Vec<f32> =
            (0..t_rows * n_experts).map(|i| ((i * 37 + 11) % 251) as f32 * 0.1 - 12.0).collect();
        setup(dt, t_rows, n_experts, k, logits)
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_router_topk_biased_rows_fused;

    /// Laguna production prefill shape: 256 experts, top-8, T=512 rows.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_router_topk_biased_rows_fused(dt: DType) -> BenchSetup {
        let t_rows = 512usize;
        let n_experts = 256usize;
        let k = 8usize;
        BenchSetup::new(iron_moe_router_topk_biased_rows_fused::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("logits", t_rows * n_experts, dt))
            .buffer(BenchBuffer::random("bias", n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", t_rows * k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", t_rows * k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .with_shape_label(format!(
                "laguna T{t_rows} E{n_experts} k{k} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(t_rows as u32, 1, 1, [32, 1, 1])
            .bytes_moved(
                ((t_rows * n_experts + n_experts + t_rows * k * 2) * dt.size_bytes()) as u64,
            )
    }
}

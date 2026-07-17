//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! DeepSeek V4 / NemotronH-style router top-K selection — GPU-side, so the
//! per-layer MoE routing never round-trips to the CPU.
//!
//! Also the selection half of the candidate **Hy3** path when combined with
//! `ffai_moe_sigmoid_bias` (192 experts, top-8). See `docs/specs/HY3_MOE_PREP.md`.
//!
//! DSv4 selects the top-K experts by the **biased** sqrtsoftplus score
//! and weights them by the **unbiased** score, renormalised to sum to 1.
//! This kernel emits the sum-to-1 weights only. `routed_scaling_factor`
//! is applied by the CALLER (`gpuRoutedFfnTail`) AFTER this renorm —
//! `final_weight_k = scaling * unbiased_k / Σ unbiased` — so it does NOT
//! cancel. Keeping it out of the kernel lets the kernel take just the two
//! score vectors:
//!
//! ```text
//!   chosen      = argmax-k(score_biased)            (indices)
//!   weight_i    = score_unbiased[chosen_i] / Σ_j∈chosen score_unbiased[chosen_j]
//! ```
//!
//! Mirrors the CPU selection in `forwardFfnSubblock` (top-K by biased,
//! weights from unbiased, sum-to-1). Structure adapted from
//! `ffai_moe_router_topk` (masked argmax passes, one simdgroup/row).
//!
//! Dispatch: `[n_rows, 1, 1] × [32, 1, 1]` (pins tpg=32; one simdgroup
//! per token row — decode is n_rows=1).

use ffai_kernels::kernel;

#[kernel]
pub fn ffai_moe_router_topk_biased<T>(
    score_biased: Tensor<T>,
    score_unbiased: Tensor<T>,
    mut indices_out: Tensor<u32>,
    mut weights_out: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] k: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let row_base = row * n_experts;
    threadgroup_alloc("tg_chosen_idx", 32u32);

    // ── k masked-argmax passes over score_biased ─────────────────────
    for it in range(0u32, k, 1u32) {
        let mut best_val = neg_infinity();
        let mut best_idx = 0u32;
        let n_per_lane = (n_experts + 31u32) / 32u32;
        for r in range(0u32, n_per_lane, 1u32) {
            let j = r * 32u32 + lane;
            if j < n_experts {
                let v = load(score_biased[row_base + j]).cast::<f32>();
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

    // ── weights = unbiased[chosen] / Σ unbiased[chosen] ──────────────
    // tg_chosen_idx is f32-typed scratch; cast to u32 for the subscript.
    let my_idx_f = select(lane < k, threadgroup_load("tg_chosen_idx", lane), 0.0f32);
    let my_idx = my_idx_f.cast::<u32>();
    let my_unbiased =
        select(lane < k, load(score_unbiased[row_base + my_idx]).cast::<f32>(), 0.0f32);
    let sum_chosen = simd_sum(my_unbiased);
    let weight = my_unbiased / sum_chosen;
    if lane < k {
        let out_base = row * k + lane;
        store(indices_out[out_base], my_idx);
        store(weights_out[out_base], weight.cast::<T>());
    }
}

/// out[i] = table[idx[i]] — gather a u32 table by u32 indices. Used to
/// remap raw routed expert ids into resident-pool packed slot ids on the
/// GPU (so the gather dispatch's `expert_ids` never touches the CPU).
#[kernel]
pub fn ffai_remap_u32(
    table: Tensor<u32>,
    idx: Tensor<u32>,
    mut out: Tensor<u32>,
    #[constexpr] n: u32,
) {
    let i = tid;
    if i < n {
        let e = load(idx[i]);
        store(out[i], load(table[e]));
    }
}

pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_moe_router_topk_biased;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Top-k by `score_biased`; weights = renorm of `score_unbiased` on chosen.
    fn topk_biased_oracle(
        biased: &[f32],
        unbiased: &[f32],
        n_experts: usize,
        k: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut order: Vec<usize> = (0..n_experts).collect();
        // Descending biased score; smaller index wins ties (matches kernel simd_min on ties).
        order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap().then(a.cmp(&b)));
        let chosen = &order[..k];
        let sum: f32 = chosen.iter().map(|&e| unbiased[e]).sum();
        let idx: Vec<u32> = chosen.iter().map(|&e| e as u32).collect();
        let w: Vec<f32> = chosen.iter().map(|&e| unbiased[e] / sum).collect();
        (idx, w)
    }

    fn setup(dt: DType, n_experts: usize, k: usize) -> TestSetup {
        // Well-separated biased scores so selection is dtype-stable.
        let biased_f: Vec<f32> =
            (0..n_experts).map(|e| ((e * 7 + 3) % n_experts) as f32 * 0.5 + 0.1).collect();
        // Unbiased must be positive for a clean renorm (sigmoid range).
        let unbiased_f: Vec<f32> =
            (0..n_experts).map(|e| 0.05 + ((e * 5 + 1) % n_experts) as f32 * 0.01).collect();
        let biased = unpack_f32(&pack_f32(&biased_f, dt), dt);
        let unbiased = unpack_f32(&pack_f32(&unbiased_f, dt), dt);
        let (idx, w) = topk_biased_oracle(&biased, &unbiased, n_experts, k);
        TestSetup::new(ffai_moe_router_topk_biased::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("score_biased", pack_f32(&biased_f, dt), dt))
            .input(TestBuffer::from_vec("score_unbiased", pack_f32(&unbiased_f, dt), dt))
            .input(TestBuffer::zeros("indices_out", k, DType::U32))
            .input(TestBuffer::zeros("weights_out", k, dt))
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("indices_out", u32_bytes(&idx), DType::U32))
            .expect(TestBuffer::from_vec("weights_out", pack_f32(&w, dt), dt))
            .grid_3d(1, 1, 1, [32, 1, 1])
    }

    /// DeepSeek-V4 / Nemotron-ish default bench shape.
    #[test_kernel(dtypes = [f32], tol = [1e-4])]
    fn test_moe_router_topk_biased_dsv4(dt: DType) -> TestSetup { setup(dt, 256, 6) }

    /// Hy3 / hy_v3 — 192 experts, top-8.
    #[test_kernel(dtypes = [f32], tol = [1e-4])]
    fn test_moe_router_topk_biased_hy3(dt: DType) -> TestSetup { setup(dt, 192, 8) }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::{ffai_moe_router_topk_biased, ffai_remap_u32};

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_router_topk_biased(dt: DType) -> BenchSetup {
        let n_experts = 256usize;
        let k = 6usize;
        BenchSetup::new(ffai_moe_router_topk_biased::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("score_biased", n_experts, dt))
            .buffer(BenchBuffer::random("score_unbiased", n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .grid_3d(1, 1, 1, [32, 1, 1])
            .bytes_moved((2 * n_experts * dt.size_bytes()) as u64)
    }

    /// Hy3 decode-shaped router selection (192 experts, top-8, 1 token row).
    #[bench(dtypes = [f32])]
    fn bench_moe_router_topk_biased_hy3(dt: DType) -> BenchSetup {
        let n_experts = 192usize;
        let k = 8usize;
        BenchSetup::new(ffai_moe_router_topk_biased::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("score_biased", n_experts, dt))
            .buffer(BenchBuffer::random("score_unbiased", n_experts, dt))
            .buffer(BenchBuffer::zeros("indices_out", k, DType::U32).output())
            .buffer(BenchBuffer::zeros("weights_out", k, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("k", k as u32)
            .with_shape_label(format!("hy3 E{n_experts} k{k} {}", crate::utils::dtype_label(dt)))
            .grid_3d(1, 1, 1, [32, 1, 1])
            .bytes_moved((2 * n_experts * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32])]
    fn bench_remap_u32(_dt: DType) -> BenchSetup {
        let n = 6usize;
        BenchSetup::new(ffai_remap_u32::kernel_ir())
            .buffer(BenchBuffer::zeros("table", 256, DType::U32))
            .buffer(BenchBuffer::zeros("idx", n, DType::U32))
            .buffer(BenchBuffer::zeros("out", n, DType::U32).output())
            .constexpr("n", n as u32)
            .grid_1d(n, 32)
            .bytes_moved((n * 4 * 2) as u64)
    }
}

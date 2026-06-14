//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MoE permute / unpermute — `mt_moe_permute` gathers token rows into
//! expert-sorted order ahead of the grouped expert matmul; `mt_moe_unpermute`
//! scatters the per-expert outputs back to token order with the router-weight
//! combine. Both keep the MoE dispatch on-device.

use metaltile::kernel;

// ── mt_moe_unpermute ─────────────────────────────────────────────────────
//
// Combine k expert outputs back into the original token order with
// top-k softmax weights.
//
// Inputs:
//   expert_outputs  — [k*B*T, hidden]   per-expert dense outputs at the
//                                       expert-sorted positions
//   inv_perm        — [B*T, k]          where (token i, slot j) was placed
//                                       in expert_outputs (computed by
//                                       caller's sort step)
//   top_k_weights   — [B*T, k]          softmax weights from
//                                       mt_moe_router_topk
//   out             — [B*T, hidden]     weighted sum across k experts
//
// Constexpr:
//   hidden — model hidden dim (e.g. 2048 for Qwen3-MoE)
//   k      — top-k expert count (e.g. 8)
//
// Geometry:
//   tpg=128  (split hidden across 128 lanes via 4-wide vectorize)
//   grid=[B*T, 1, 1]
//
// Per-token cost: read k * hidden / 128 = (k * hidden) / 128 expert
// values + k weights, do k FMAs per output column, one store per
// column. At hidden=2048, k=8 → ~1k FMAs per token. Bandwidth-bound,
// not ALU-bound.
#[kernel]
pub fn mt_moe_unpermute<T>(
    expert_outputs: Tensor<T>,
    inv_perm: Tensor<u32>,
    top_k_weights: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] k: u32,
) {
    let token = tgid_x;
    let lane = tid;
    let row_base_inv = token * k;
    let row_base_w = token * k;
    let row_base_out = token * hidden;
    let n_per_lane = (hidden + 127u32) / 128u32;
    for r in range(0u32, n_per_lane, 1u32) {
        let h = r * 128u32 + lane;
        if h < hidden {
            let mut acc = 0.0f32;
            for j in range(0u32, k, 1u32) {
                let pos = load(inv_perm[row_base_inv + j]);
                let v = load(expert_outputs[pos * hidden + h]).cast::<f32>();
                let w = load(top_k_weights[row_base_w + j]).cast::<f32>();
                acc = acc + w * v;
            }
            store(out[row_base_out + h], acc.cast::<T>());
        }
    }
}

// ── mt_moe_permute ───────────────────────────────────────────────────────
//
// Gather tokens into per-expert contiguous buffers given a pre-computed
// sort permutation. The expensive sort step (argsort over top-k expert
// indices) is done by the caller — typically CPU-side via Rust sort,
// or via a future sort kernel. This kernel is just the data-movement
// half: each output position copies the row indicated by sort_token_idx.
//
// Inputs:
//   tokens          — [B*T, hidden]      activations to gather
//   sort_token_idx  — [k * B*T]          for each permuted position p,
//                                        which original token row sourced it.
//                                        Caller computes via argsort over
//                                        top-k indices flattened to
//                                        (token * k + slot) → token (this is
//                                        the "permute" direction; the inverse
//                                        is `inv_perm` consumed by unpermute).
//   permuted        — [k * B*T, hidden]  expert-sorted output. Each k*B*T
//                                        row corresponds to one (expert, token)
//                                        pair; consecutive rows with the same
//                                        expert form that expert's input slab.
//
// Constexpr:
//   hidden — model hidden dim
//
// Geometry:
//   tpg=128  (split hidden across 128 lanes, ceil(hidden/128) iters)
//   grid=[k*B*T, 1, 1]
//
// Per-permuted-row cost: hidden / 128 = 16 loads + 16 stores (at
// hidden=2048). Bandwidth-bound — no FMAs, just a vector copy.
#[kernel]
pub fn mt_moe_permute<T>(
    tokens: Tensor<T>,
    sort_token_idx: Tensor<u32>,
    mut permuted: Tensor<T>,
    #[constexpr] hidden: u32,
) {
    let permuted_pos = tgid_x;
    let lane = tid;
    let token = load(sort_token_idx[permuted_pos]);
    let src_base = token * hidden;
    let dst_base = permuted_pos * hidden;
    let n_per_lane = (hidden + 127u32) / 128u32;
    for r in range(0u32, n_per_lane, 1u32) {
        let h = r * 128u32 + lane;
        if h < hidden {
            let v = load(tokens[src_base + h]);
            store(permuted[dst_base + h], v);
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // mt_moe_permute: gather token rows into expert-sorted order.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_moe_permute(dt: DType) -> TestSetup {
        let (n_tokens, k, hidden) = (4usize, 2usize, 128usize);
        let n_permuted = k * n_tokens;
        let tokens_f: Vec<f32> =
            (0..n_tokens * hidden).map(|i| ((i as f32) * 0.013).sin()).collect();
        let sort_token_idx: Vec<u32> = vec![0, 2, 1, 3, 1, 0, 3, 2];
        let tokens = unpack_f32(&pack_f32(&tokens_f, dt), dt);
        let mut expected = vec![0.0f32; n_permuted * hidden];
        for p in 0..n_permuted {
            let src = sort_token_idx[p] as usize;
            for h in 0..hidden {
                expected[p * hidden + h] = tokens[src * hidden + h];
            }
        }
        TestSetup::new(mt_moe_permute::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("tokens", pack_f32(&tokens_f, dt), dt))
            .input(TestBuffer::from_vec("sort_token_idx", u32_bytes(&sort_token_idx), DType::U32))
            .input(TestBuffer::zeros("permuted", n_permuted * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .expect(TestBuffer::from_vec("permuted", pack_f32(&expected, dt), dt))
            .grid_3d(n_permuted as u32, 1, 1, [128, 1, 1])
    }

    // mt_moe_unpermute: weighted sum of k expert outputs back to token order.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_moe_unpermute(dt: DType) -> TestSetup {
        let (n_tokens, k, hidden) = (4usize, 2usize, 128usize);
        let n_permuted = k * n_tokens;
        let expert_outputs_f: Vec<f32> =
            (0..n_permuted * hidden).map(|i| ((i as f32) * 0.011).sin()).collect();
        let inv_perm: Vec<u32> = vec![0, 5, 2, 7, 4, 1, 6, 3];
        let weights_f: Vec<f32> = (0..n_tokens * k).map(|i| 0.3 + 0.1 * (i as f32)).collect();
        let eo = unpack_f32(&pack_f32(&expert_outputs_f, dt), dt);
        let w = unpack_f32(&pack_f32(&weights_f, dt), dt);
        let mut expected = vec![0.0f32; n_tokens * hidden];
        for token in 0..n_tokens {
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for j in 0..k {
                    let pos = inv_perm[token * k + j] as usize;
                    acc += w[token * k + j] * eo[pos * hidden + h];
                }
                expected[token * hidden + h] = acc;
            }
        }
        TestSetup::new(mt_moe_unpermute::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("expert_outputs", pack_f32(&expert_outputs_f, dt), dt))
            .input(TestBuffer::from_vec("inv_perm", u32_bytes(&inv_perm), DType::U32))
            .input(TestBuffer::from_vec("top_k_weights", pack_f32(&weights_f, dt), dt))
            .input(TestBuffer::zeros("out", n_tokens * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_tokens as u32, 1, 1, [128, 1, 1])
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::*;

    // ── permute — pure gather, bench-only ─────────────────────────────────
    // ABI: tokens, sort_token_idx, permuted + {hidden}. Grid [k*B*T, 1, 1],
    // tpg [128,1,1].
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_permute(dt: DType) -> BenchSetup {
        let bt = 512usize;
        let k = 8usize;
        let hidden = 2048usize;
        let rows = k * bt;
        let sz = dt.size_bytes();
        // Reads `rows` source rows (worst case all distinct) + writes `rows`.
        let bytes = 2 * rows * hidden * sz;
        BenchSetup::new(mt_moe_permute::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("tokens", bt * hidden, dt))
            .buffer(BenchBuffer::zeros("sort_token_idx", rows, DType::U32))
            .buffer(BenchBuffer::zeros("permuted", rows * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .with_shape_label(format!("rows{rows} h{hidden} {}", crate::utils::dtype_label(dt)))
            .grid_3d(rows as u32, 1, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
    }

    // ── unpermute — weighted scatter-combine, bench-only ──────────────────
    // ABI: expert_outputs, inv_perm, top_k_weights, out + {hidden, k}.
    // Grid [B*T, 1, 1], tpg [128,1,1].
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_unpermute(dt: DType) -> BenchSetup {
        let bt = 512usize;
        let k = 8usize;
        let hidden = 2048usize;
        let sz = dt.size_bytes();
        let bytes = k * bt * hidden * sz + bt * k * 4 + bt * k * sz + bt * hidden * sz;
        BenchSetup::new(mt_moe_unpermute::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("expert_outputs", k * bt * hidden, dt))
            .buffer(BenchBuffer::zeros("inv_perm", bt * k, DType::U32))
            .buffer(BenchBuffer::random("top_k_weights", bt * k, dt))
            .buffer(BenchBuffer::zeros("out", bt * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .constexpr("k", k as u32)
            .with_shape_label(format!("BT{bt} h{hidden} k{k} {}", crate::utils::dtype_label(dt)))
            .grid_3d(bt as u32, 1, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
    }
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MoE combine tail (Laguna W57): fuses `iron_moe_unpermute`'s weighted
//! expert-output combine with the two elementwise adds that always follow
//! it on the Laguna prefill sorted-MoE tail — shared-expert add (present on
//! layers/paths with a shared expert, elided via `has_shared` otherwise) and
//! the residual add (always present) — into ONE dispatch, one read of each
//! input, one write of `out`. Eliminates the two extra full `[T, hidden]`
//! read-modify-write round-trips the unfused 3-dispatch path
//! (`Ops.moeUnpermute` -> `Ops.add` shared -> `Ops.add` residual) pays.
//!
//! Byte-identical to unfused, INCLUDING intermediate rounding: the unfused
//! sequence is not one continuous f32 accumulation — `iron_moe_unpermute`
//! rounds its weighted-sum to `T` and stores it, THEN `Ops.add`'s generated
//! kernel (`vector_add`, `crates/wh-iron-std/src/kernels/ops/binary.rs`)
//! promotes both `T` operands to f32, adds, and rounds back to `T` — once
//! per `Ops.add` call, i.e. once for the shared-expert add and once more
//! for the (separate-dispatch) residual add. That's 2-3 independent
//! round-to-`T` steps, not one. This kernel replicates that exact
//! round-per-step schedule (`acc.cast::<T>()` after the weighted sum, then
//! `cast::<f32>() + shared -> cast::<T>()`, then `cast::<f32>() + residual
//! -> cast::<T>()` for the final store) rather than accumulating
//! everything in one f32 register and rounding once at the end — the
//! latter is MORE precise in isolation but is a genuinely different
//! computation once compounded across ~40 MoE layers with discrete top-k
//! routing (small perturbations can flip which expert a token routes to),
//! which is exactly the class of numerics risk `LagunaSortedCombine`'s
//! file header already flags for this call site. Empirically: the
//! single-accumulator version measured cos=0.995 vs the unfused sequence
//! on a real prefill (argmax preserved, but well under the ≥0.99999 bar);
//! this per-step-rounded version is the fix.
use wh_iron::kernel;

// ── iron_moe_unpermute_tail ─────────────────────────────────────────────────
//
// out = unpermute(expert_outputs, inv_perm, top_k_weights)
//       [+ shared_expert_out]      (elided when has_shared == 0)
//       + residual
//
// Inputs:
//   expert_outputs    — [k*B*T, hidden]  per-expert dense outputs at the
//                                        expert-sorted positions
//   inv_perm          — [B*T, k]         where (token i, slot j) was placed
//                                        in expert_outputs
//   top_k_weights     — [B*T, k]         softmax weights
//   shared_expert_out — [B*T, hidden]    shared-expert output. When
//                                        has_shared == 0 the caller may bind
//                                        ANY [B*T,hidden]-or-larger buffer
//                                        here (e.g. `residual` itself) — the
//                                        load is inside a dispatch-uniform
//                                        `if has_shared` and never executes.
//   residual          — [B*T, hidden]    layer input to add back (always
//                                        present)
//   out               — [B*T, hidden]    final layer output
//
// Constexpr:
//   hidden     — model hidden dim
//   k          — top-k expert count
//   has_shared — u32 flag (NOT bool — the Swift codegen path for a
//                constexpr `bool` mismatches MSL's 4-byte `uint` slot with
//                a 1-byte `setBytes`; `u32` keeps MSL/Swift ABI consistent,
//                matching the `has_mask` precedent in
//                `ssm/gated_delta_replay.rs`). Identical across every
//                thread in a given launch (dispatch-uniform), so the `if`
//                below costs no divergence — same uniform-branch pattern as
//                `gemm/steel/steel_gemm_masked.rs`'s epilogue.
//
// Geometry: tpg=128, grid=[B*T, 1, 1] — identical to `iron_moe_unpermute`.
#[kernel]
pub fn iron_moe_unpermute_tail<T>(
    expert_outputs: Tensor<T>,
    inv_perm: Tensor<u32>,
    top_k_weights: Tensor<T>,
    shared_expert_out: Tensor<T>,
    residual: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] k: u32,
    #[constexpr] has_shared: u32,
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
            // Stage 1: weighted top-k sum, f32 accumulator, round to T once
            // — byte-identical to `iron_moe_unpermute`'s own store.
            let mut acc = 0.0f32;
            for j in range(0u32, k, 1u32) {
                let pos = load(inv_perm[row_base_inv + j]);
                let v = load(expert_outputs[pos * hidden + h]).cast::<f32>();
                let w = load(top_k_weights[row_base_w + j]).cast::<f32>();
                acc = acc + w * v;
            }
            let mut result = acc.cast::<T>();
            // Stage 2 (optional): shared-expert add, promote both `T`
            // operands to f32, add, round back to `T` — matches
            // `vector_add`'s generated MSL (`float(a) + float(b)` ->
            // `T(...)`) exactly, not a continued f32 accumulation.
            if has_shared != 0u32 {
                let shared_v = load(shared_expert_out[row_base_out + h]).cast::<f32>();
                let sum2 = result.cast::<f32>() + shared_v;
                result = sum2.cast::<T>();
            }
            // Stage 3: residual add, same promote-add-round-once pattern
            // as the (separate-dispatch, in the unfused sequence) residual
            // `Ops.add` call.
            let residual_v = load(residual[row_base_out + h]).cast::<f32>();
            let sum3 = result.cast::<f32>() + residual_v;
            result = sum3.cast::<T>();
            store(out[row_base_out + h], result);
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // Simulate one `T`-rounding round-trip on a single f32 value (i.e. what
    // storing to a `Tensor<T>` and reading it back does) — used to build a
    // CPU oracle that mirrors the kernel's exact per-stage rounding
    // schedule (see file header: 3 independent round-to-`T` steps, not one
    // continuous f32 accumulation).
    fn round_trip(v: f32, dt: DType) -> f32 { unpack_f32(&pack_f32(&[v], dt), dt)[0] }

    // has_shared = 1: combine + shared-expert add + residual add, all fused.
    // Tight tolerance (not the generous quantization-noise tol other
    // combine tests use) because the oracle below replicates the exact
    // per-stage T-rounding schedule the kernel performs, matching
    // `iron_moe_unpermute` -> `Ops.add`(shared) -> `Ops.add`(residual)
    // bit-for-bit, not just "close in f32".
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-6, 1e-6])]
    fn test_moe_unpermute_tail_with_shared(dt: DType) -> TestSetup {
        let (n_tokens, k, hidden) = (4usize, 2usize, 128usize);
        let n_permuted = k * n_tokens;
        let expert_outputs_f: Vec<f32> =
            (0..n_permuted * hidden).map(|i| ((i as f32) * 0.011).sin()).collect();
        let inv_perm: Vec<u32> = vec![0, 5, 2, 7, 4, 1, 6, 3];
        let weights_f: Vec<f32> = (0..n_tokens * k).map(|i| 0.3 + 0.1 * (i as f32)).collect();
        let shared_f: Vec<f32> =
            (0..n_tokens * hidden).map(|i| ((i as f32) * 0.017).cos() * 0.5).collect();
        let residual_f: Vec<f32> =
            (0..n_tokens * hidden).map(|i| ((i as f32) * 0.023).sin() * 0.25).collect();
        let eo = unpack_f32(&pack_f32(&expert_outputs_f, dt), dt);
        let w = unpack_f32(&pack_f32(&weights_f, dt), dt);
        let sh = unpack_f32(&pack_f32(&shared_f, dt), dt);
        let res = unpack_f32(&pack_f32(&residual_f, dt), dt);
        let mut expected = vec![0.0f32; n_tokens * hidden];
        for token in 0..n_tokens {
            for h in 0..hidden {
                // Stage 1: f32-accumulated weighted sum, round to T once —
                // matches `iron_moe_unpermute`'s own store.
                let mut acc = 0.0f32;
                for j in 0..k {
                    let pos = inv_perm[token * k + j] as usize;
                    acc += w[token * k + j] * eo[pos * hidden + h];
                }
                let mut result = round_trip(acc, dt);
                // Stage 2: shared add, promote-add-round — matches
                // `vector_add`'s generated MSL.
                result = round_trip(result + sh[token * hidden + h], dt);
                // Stage 3: residual add, same pattern.
                result = round_trip(result + res[token * hidden + h], dt);
                expected[token * hidden + h] = result;
            }
        }
        TestSetup::new(iron_moe_unpermute_tail::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("expert_outputs", pack_f32(&expert_outputs_f, dt), dt))
            .input(TestBuffer::from_vec("inv_perm", u32_bytes(&inv_perm), DType::U32))
            .input(TestBuffer::from_vec("top_k_weights", pack_f32(&weights_f, dt), dt))
            .input(TestBuffer::from_vec("shared_expert_out", pack_f32(&shared_f, dt), dt))
            .input(TestBuffer::from_vec("residual", pack_f32(&residual_f, dt), dt))
            .input(TestBuffer::zeros("out", n_tokens * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .constexpr("k", k as u32)
            .constexpr("has_shared", 1u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_tokens as u32, 1, 1, [128, 1, 1])
    }

    // has_shared = 0: combine + residual add only (shared_expert_out bound
    // to the residual buffer itself as the dummy — never read). Tight
    // tolerance — see `test_moe_unpermute_tail_with_shared`'s comment.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-6, 1e-6])]
    fn test_moe_unpermute_tail_no_shared(dt: DType) -> TestSetup {
        let (n_tokens, k, hidden) = (4usize, 2usize, 128usize);
        let n_permuted = k * n_tokens;
        let expert_outputs_f: Vec<f32> =
            (0..n_permuted * hidden).map(|i| ((i as f32) * 0.011).sin()).collect();
        let inv_perm: Vec<u32> = vec![0, 5, 2, 7, 4, 1, 6, 3];
        let weights_f: Vec<f32> = (0..n_tokens * k).map(|i| 0.3 + 0.1 * (i as f32)).collect();
        let residual_f: Vec<f32> =
            (0..n_tokens * hidden).map(|i| ((i as f32) * 0.023).sin() * 0.25).collect();
        let eo = unpack_f32(&pack_f32(&expert_outputs_f, dt), dt);
        let w = unpack_f32(&pack_f32(&weights_f, dt), dt);
        let res = unpack_f32(&pack_f32(&residual_f, dt), dt);
        let mut expected = vec![0.0f32; n_tokens * hidden];
        for token in 0..n_tokens {
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for j in 0..k {
                    let pos = inv_perm[token * k + j] as usize;
                    acc += w[token * k + j] * eo[pos * hidden + h];
                }
                // Stage 1 rounding, then Stage 3 (no shared -> Stage 2 is
                // skipped, matching `has_shared=0`).
                let result = round_trip(round_trip(acc, dt) + res[token * hidden + h], dt);
                expected[token * hidden + h] = result;
            }
        }
        TestSetup::new(iron_moe_unpermute_tail::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("expert_outputs", pack_f32(&expert_outputs_f, dt), dt))
            .input(TestBuffer::from_vec("inv_perm", u32_bytes(&inv_perm), DType::U32))
            .input(TestBuffer::from_vec("top_k_weights", pack_f32(&weights_f, dt), dt))
            .input(TestBuffer::from_vec("shared_expert_out", pack_f32(&residual_f, dt), dt))
            .input(TestBuffer::from_vec("residual", pack_f32(&residual_f, dt), dt))
            .input(TestBuffer::zeros("out", n_tokens * hidden, dt))
            .constexpr("hidden", hidden as u32)
            .constexpr("k", k as u32)
            .constexpr("has_shared", 0u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_tokens as u32, 1, 1, [128, 1, 1])
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::*;

    // ── unpermute_tail — fused weighted scatter-combine + shared + residual,
    // bench-only ────────────────────────────────────────────────────────────
    // ABI: expert_outputs, inv_perm, top_k_weights, shared_expert_out,
    // residual, out + {hidden, k, has_shared}. Grid [B*T, 1, 1], tpg
    // [128,1,1]. Mirrors `bench_moe_unpermute` in `moe_permute.rs` plus the
    // two extra [B*T,hidden] read streams the fused kernel absorbs.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_unpermute_tail(dt: DType) -> BenchSetup {
        let bt = 512usize;
        let k = 8usize;
        let hidden = 2048usize;
        let sz = dt.size_bytes();
        let bytes = k * bt * hidden * sz  // expert_outputs
            + bt * k * 4                  // inv_perm
            + bt * k * sz                 // top_k_weights
            + bt * hidden * sz            // shared_expert_out
            + bt * hidden * sz            // residual
            + bt * hidden * sz; // out
        BenchSetup::new(iron_moe_unpermute_tail::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("expert_outputs", k * bt * hidden, dt))
            .buffer(BenchBuffer::zeros("inv_perm", bt * k, DType::U32))
            .buffer(BenchBuffer::random("top_k_weights", bt * k, dt))
            .buffer(BenchBuffer::random("shared_expert_out", bt * hidden, dt))
            .buffer(BenchBuffer::random("residual", bt * hidden, dt))
            .buffer(BenchBuffer::zeros("out", bt * hidden, dt).output())
            .constexpr("hidden", hidden as u32)
            .constexpr("k", k as u32)
            .constexpr("has_shared", 1u32)
            .with_shape_label(format!("BT{bt} h{hidden} k{k} {}", crate::utils::dtype_label(dt)))
            .grid_3d(bt as u32, 1, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
    }
}

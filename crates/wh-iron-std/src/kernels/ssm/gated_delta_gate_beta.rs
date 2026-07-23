//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Fused GDN gate/beta — `iron_gdn_gate_beta`.
//!
//! Collapses the ~6-dispatch elementwise chain the Swift-side
//! `Ops.gdnGateBeta` used to feed [`super::gated_delta_wy_plan`] /
//! [`super::gated_delta_wy_scan`]'s precomputed `g`/`beta` inputs
//! (`sigmoid` + broadcast-add + `softplus` + broadcast-mul + `exp`, all
//! over tiny `[T, Hv]` tensors, one elementwise dispatch each) into ONE
//! dispatch per GDN layer per prefill chunk.
//!
//! ## Math
//!
//! Identical to [`super::gated_delta_prep_chunk`]'s in-kernel per-token
//! g/beta computation — same un-clamped `log(exp(x) + 1)` softplus baked
//! directly into that kernel (see its `pre_softplus` / `dt_val` lines).
//! The old Swift-side dispatch chain instead routed through the
//! numerically-stable `iron_softplus` kernel (`max(x,0) + log1p(exp(-|x|))`)
//! — the same real value away from overflow (which the learned
//! `a_raw`/`dt_bias` activations never approach), but this fused kernel
//! now matches the production `prep_chunk` oracle bit-for-bit-*formula*
//! instead, which is the stronger invariant to hold.
//!
//!   - `dt    = log(exp(a_raw + dt_bias) + 1)`             (per `(row, h)`)
//!   - `g     = exp(neg_exp_a_log · dt)`  (`neg_exp_a_log = -exp(a_log)`,
//!     precomputed once on the host at model load — see
//!     `Qwen35GDNMixer.negExpALogTF32` on the Swift side, same convention
//!     the old `Ops.gdnGateBeta` used)
//!   - `beta  = sigmoid(b_raw)`
//!
//! ## Layout
//!
//! `a_raw`/`b_raw`/`g`/`beta` are `[rows, Hv]` flat (`rows = B·T`);
//! `dt_bias`/`neg_exp_a_log` are `[Hv]`, broadcast across `rows` via
//! `idx % hv` (matches the `sinks[q_idx % num_q_heads]` broadcast idiom
//! used elsewhere in this crate, e.g. `sdpa/flash_quantized_sdpa.rs`).
//!
//! Pure elementwise — one thread per `(row, h)` output element, no
//! reduction, no cross-thread state, no threadgroup memory.
use wh_iron::kernel;

#[kernel]
pub fn iron_gdn_gate_beta<T>(
    a_raw: Tensor<T>,
    b_raw: Tensor<T>,
    dt_bias: Tensor<T>,
    neg_exp_a_log: Tensor<T>,
    mut g: Tensor<T>,
    mut beta: Tensor<T>,
    #[constexpr] hv: u32,
) {
    let idx = program_id(0);
    let h = idx % hv;
    let a = load(a_raw[idx]).cast::<f32>();
    let b = load(b_raw[idx]).cast::<f32>();
    let bias = load(dt_bias[h]).cast::<f32>();
    let neg_exp_a = load(neg_exp_a_log[h]).cast::<f32>();
    let pre_softplus = a + bias;
    let dt = log(exp(pre_softplus) + 1.0f32);
    let g_val = exp(neg_exp_a * dt);
    let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b));
    store(g[idx], g_val.cast::<T>());
    store(beta[idx], beta_val.cast::<T>());
}

/// New-syntax correctness for `iron_gdn_gate_beta`. CPU oracle mirrors
/// `gated_delta_prep_chunk`'s own `softplus_unclamped` / `sigmoid` g-beta
/// computation exactly (un-clamped softplus, same formula). Inputs are
/// dtype-rounded before both the GPU dispatch and the CPU oracle so
/// neither side is artificially more precise than the other.
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_gdn_gate_beta;
    use crate::utils::{pack_f32, unpack_f32};

    fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    fn setup(rows: usize, hv: usize, dt: DType) -> TestSetup {
        let n = rows * hv;
        let a_raw: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.03 - 0.35).collect();
        let b_raw: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.025 - 0.2).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.4 + (i as f32) * 0.04).collect();
        // neg_exp_a_log = -exp(a_log); a_log spans a realistic learned-decay
        // range so exp(a_log) (and its negation) stays in a well-conditioned
        // band — mirrors gated_delta_prep_chunk's own test fixture range.
        let neg_exp_a_log: Vec<f32> =
            (0..hv).map(|i| -((-4.0 + (i as f32) * 0.3f32).exp())).collect();

        // Dtype-round every input the same way the kernel will read it
        // back, so the CPU oracle isn't more precise than the GPU inputs
        // it's compared against.
        let a_raw_dt = unpack_f32(&pack_f32(&a_raw, dt), dt);
        let b_raw_dt = unpack_f32(&pack_f32(&b_raw, dt), dt);
        let dt_bias_dt = unpack_f32(&pack_f32(&dt_bias, dt), dt);
        let neg_exp_a_log_dt = unpack_f32(&pack_f32(&neg_exp_a_log, dt), dt);

        let mut g = vec![0.0f32; n];
        let mut beta = vec![0.0f32; n];
        for row in 0..rows {
            for h in 0..hv {
                let idx = row * hv + h;
                let dt_val = softplus_unclamped(a_raw_dt[idx] + dt_bias_dt[h]);
                g[idx] = (neg_exp_a_log_dt[h] * dt_val).exp();
                beta[idx] = sigmoid(b_raw_dt[idx]);
            }
        }

        TestSetup::new(iron_gdn_gate_beta::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("neg_exp_a_log", pack_f32(&neg_exp_a_log, dt), dt))
            .input(TestBuffer::zeros("g", n, dt))
            .input(TestBuffer::zeros("beta", n, dt))
            .expect(TestBuffer::from_vec("g", pack_f32(&g, dt), dt))
            .expect(TestBuffer::from_vec("beta", pack_f32(&beta, dt), dt))
            .constexpr("hv", hv as u32)
            .grid_1d(n, 256)
    }

    // Non-power-of-two rows (96) × Qwen3.6-A3B-class Hv=32 — exercises the
    // `idx % hv` broadcast at a chunk-sized shape without a suspiciously
    // round row count.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_gdn_gate_beta(dt: DType) -> TestSetup { setup(96, 32, dt) }
}

/// New-syntax benchmark for `iron_gdn_gate_beta` at a Qwen3.6-A3B-class
/// prefill-chunk shape (`Hv = 32`, `T = 1024`).
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_gdn_gate_beta;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gdn_gate_beta(dt: DType) -> BenchSetup {
        let (t, hv) = (1024usize, 32usize);
        let n = t * hv;
        BenchSetup::new(iron_gdn_gate_beta::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a_raw", n, dt))
            .buffer(BenchBuffer::random("b_raw", n, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("neg_exp_a_log", hv, dt))
            .buffer(BenchBuffer::zeros("g", n, dt).output())
            .buffer(BenchBuffer::zeros("beta", n, dt).output())
            .constexpr("hv", hv as u32)
            .grid_1d(n, 256)
            .bytes_moved((4 * n * dt.size_bytes()) as u64)
    }
}

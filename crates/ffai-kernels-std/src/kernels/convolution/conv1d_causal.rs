//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Depthwise causal 1-D convolution — the streaming-decode step and the
//! batched-prefill form. Used by the Mamba / SSM short-conv (the conv that
//! precedes the selective scan), but it is a plain causal conv1d, so it lives
//! with the convolution family; its fused sibling is
//! `conv1d_causal_step_silu_cast_many`.
//!
//! - `ffai_conv1d_causal_step` — one thread per channel, streaming decode:
//!   `y[d] = bias[d] + w[K-1][d]·x[d] + Σ_{k<K-1} w[k][d]·state[k][d]`, then
//!   shifts `state` in place (drop `state[0]`, append `x`). Each channel is
//!   owned by one thread, so the read-then-write shift is barrier-free.
//! - `ffai_conv1d_causal_prefill` — all S prompt tokens in one dispatch from zero
//!   initial state, one thread per `(token, channel)`, with SiLU applied inline
//!   (saves a second dispatch). Out-of-bounds taps read 0.
//!
//! Accumulation is in f32 regardless of `T`.

use ffai_kernels::kernel;

#[kernel]
pub fn ffai_conv1d_causal_step<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    mut state: Tensor<T>,
    mut y: Tensor<T>,
    #[constexpr] n_channels: u32,
    #[constexpr] kernel_size: u32,
) {
    let d = program_id::<0>();
    let x_d = load(x[d]).cast::<f32>();
    let b_d = load(b[d]).cast::<f32>();
    // Convolution: w[K-1] pairs with current input x[d]; w[0]..w[K-2]
    // pair with state[0]..state[K-2].
    let w_last = load(w[(kernel_size - 1u32) * n_channels + d]).cast::<f32>();
    let mut acc = b_d + w_last * x_d;
    // `kernel_size` is contractually >= 2 (a causal conv with state).
    // Guard the unsigned subtraction anyway: a stray `kernel_size == 0`
    // would make `kernel_size - 1` underflow to ~4e9 — a GPU-pinning
    // loop. `select` clamps the trip count to 0 instead.
    let conv_taps = select(kernel_size > 1u32, kernel_size - 1u32, 0u32);
    for k in range(0u32, conv_taps, 1u32) {
        let s_kd = load(state[k * n_channels + d]).cast::<f32>();
        let w_kd = load(w[k * n_channels + d]).cast::<f32>();
        acc = acc + w_kd * s_kd;
    }
    store(y[d], acc.cast::<T>());
    // Shift state up by one (drop state[0], append x[d] at the tail).
    // Sequential within the thread → safe even though state[k] is read
    // after being written: we read state[k+1] each iteration, never
    // state[k].
    // Same underflow guard: `kernel_size - 2` would wrap to ~4e9 for
    // any `kernel_size < 2`.
    let shift_taps = select(kernel_size > 2u32, kernel_size - 2u32, 0u32);
    for k in range(0u32, shift_taps, 1u32) {
        let next = load(state[(k + 1u32) * n_channels + d]);
        store(state[k * n_channels + d], next);
    }
    // Same `kernel_size < 2` hazard as above, but for the tail STORE: the
    // slot index would wrap to ~4e9 AND `state` has K-1 = 0 slots, so there
    // is nothing valid to clamp to — skip the store entirely.
    if kernel_size > 1u32 {
        store(state[(kernel_size - 2u32) * n_channels + d], load(x[d]));
    }
}

#[kernel]
pub fn ffai_conv1d_causal_prefill(
    xbc_in: Tensor<f32>, // [s * conv_dim] flat row-major
    w: Tensor<f32>,      // [kc * conv_dim] reorganized same as decode step
    bias: Tensor<f32>,   // [conv_dim]
    mut y: Tensor<f32>,  // [s * conv_dim] output with silu applied
    #[constexpr] conv_dim: u32,
    #[constexpr] kc: u32,
) {
    let idx = program_id::<0>();
    let ti = idx / conv_dim;
    let ch = idx - ti * conv_dim;
    let b_ch = load(bias[ch]);
    // Accumulate: w[k, ch] pairs with xbc_in[ti - (kc-1-k), ch].
    // k=0 → lag (kc-1); k=kc-1 → current token (lag 0).
    let mut acc = b_ch;
    for k in range(0u32, kc, 1u32) {
        let lag = kc - 1u32 - k;
        // Only include this tap if it's within the valid prefix.
        if ti >= lag {
            let src_ti = ti - lag;
            let v = load(xbc_in[src_ti * conv_dim + ch]);
            let wk = load(w[k * conv_dim + ch]);
            acc = acc + wk * v;
        }
    }
    // Silu activation: y = acc / (1 + exp(-acc)).
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - acc));
    store(y[idx], acc * sig);
}

// Causal-conv state roll (prefill->decode handoff for the short conv).
/// Roll a causal-conv state ON-DEVICE: `new = [old[conv_dim..], xbc]` (drop the
/// oldest conv_dim, append the current input) — keeps the Mamba conv history on
/// the GPU. `keep = (kc-2)*conv_dim`; indices clamped so both select branches
/// are in-bounds.
#[kernel]
pub fn ffai_conv_roll<T>(
    old: Tensor<T>,
    xbc: Tensor<T>,
    mut newst: Tensor<T>,
    #[constexpr] conv_dim: u32,
    #[constexpr] keep: u32,
    #[constexpr] n: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let oi = select(i < keep, i + conv_dim, 0u32);
        let xi = select(i < keep, 0u32, i - keep);
        let v = select(i < keep, load(old[oi]), load(xbc[xi]));
        store(newst[i], v);
    }
}

pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::{ffai_conv1d_causal_prefill, ffai_conv1d_causal_step};
    use crate::utils::pack_f32;

    /// CPU oracle: `y[d] = b[d] + w[K-1][d]·x[d] + Σ_{k<K-1} w[k][d]·state[k][d]`,
    /// then shift state up and append `x`. Returns `(y, shifted_state)`.
    fn conv1d_oracle(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        state_in: &[f32],
        n_channels: usize,
        kernel_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n_channels];
        let mut state = state_in.to_vec();
        let k_last = kernel_size - 1;
        for d in 0..n_channels {
            let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
            for k in 0..k_last {
                acc += w[k * n_channels + d] * state_in[k * n_channels + d];
            }
            y[d] = acc;
        }
        for d in 0..n_channels {
            for k in 0..kernel_size.saturating_sub(2) {
                state[k * n_channels + d] = state_in[(k + 1) * n_channels + d];
            }
            if kernel_size >= 2 {
                state[(kernel_size - 2) * n_channels + d] = x[d];
            }
        }
        (y, state)
    }

    fn conv1d_setup(n_channels: usize, kernel_size: usize, dt: DType) -> TestSetup {
        let x: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let state_in: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();

        let (y_exp, state_exp) = conv1d_oracle(&x, &w, &b, &state_in, n_channels, kernel_size);

        TestSetup::new(ffai_conv1d_causal_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::from_vec("state", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("y", n_channels, dt))
            .constexpr("n_channels", n_channels as u32)
            .constexpr("kernel_size", kernel_size as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state", pack_f32(&state_exp, dt), dt))
            .grid_3d(n_channels as u32, 1, 1, [1, 1, 1])
    }

    // Mamba 2 short-conv: kernel_size=4. One thread per channel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_conv1d_causal_step(dt: DType) -> TestSetup { conv1d_setup(128, 4, dt) }

    fn conv1d_causal_prefill_oracle(
        xbc: &[f32],
        w: &[f32],
        bias: &[f32],
        s: usize,
        conv_dim: usize,
        kc: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; s * conv_dim];
        for ti in 0..s {
            for ch in 0..conv_dim {
                let mut acc = bias[ch];
                for k in 0..kc {
                    let lag = kc - 1 - k;
                    if ti >= lag {
                        acc += w[k * conv_dim + ch] * xbc[(ti - lag) * conv_dim + ch];
                    }
                }
                let sig = 1.0 / (1.0f32 + (-acc).exp());
                y[ti * conv_dim + ch] = acc * sig;
            }
        }
        y
    }

    fn conv1d_causal_prefill_setup(s: usize, conv_dim: usize, kc: usize) -> TestSetup {
        let dt = DType::F32;
        let xbc: Vec<f32> = (0..s * conv_dim).map(|i| ((i as f32) * 0.011).sin() * 0.5).collect();
        let w: Vec<f32> =
            (0..kc * conv_dim).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let bias: Vec<f32> = (0..conv_dim).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let y_exp = conv1d_causal_prefill_oracle(&xbc, &w, &bias, s, conv_dim, kc);
        TestSetup::new(ffai_conv1d_causal_prefill::kernel_ir_for())
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("xbc_in", pack_f32(&xbc, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("y", s * conv_dim, dt))
            .constexpr("conv_dim", conv_dim as u32)
            .constexpr("kc", kc as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .grid_3d((s * conv_dim) as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32], tol = [1e-5])]
    fn test_conv1d_causal_prefill(_dt: DType) -> TestSetup { conv1d_causal_prefill_setup(8, 32, 4) }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_conv1d_causal_step;

    // Mamba 2 short-conv at a realistic channel count, K=4. One thread/channel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv1d_causal_step(dt: DType) -> BenchSetup {
        let (n_channels, kernel_size) = (1536usize, 4usize);
        BenchSetup::new(ffai_conv1d_causal_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", n_channels, dt))
            .buffer(BenchBuffer::random("w", kernel_size * n_channels, dt))
            .buffer(BenchBuffer::random("b", n_channels, dt))
            .buffer(BenchBuffer::random("state", (kernel_size - 1) * n_channels, dt).output())
            .buffer(BenchBuffer::zeros("y", n_channels, dt).output())
            .constexpr("n_channels", n_channels as u32)
            .constexpr("kernel_size", kernel_size as u32)
            .grid_3d(n_channels as u32, 1, 1, [1, 1, 1])
            .bytes_moved((kernel_size * n_channels * dt.size_bytes()) as u64)
    }
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Forward depthwise (grouped, `groups == channels`) 1D convolution.
//!
//! The per-channel conv1d a Conformer convolution module runs over its full
//! frame sequence — `nn.Conv1d(C, C, kernel_size=k, padding=pad, groups=C)`
//! with a large odd kernel (typically 15 / 31) — and the per-channel conv1d
//! depthwise variants of audio codecs use. Each output channel convolves
//! ONLY its own input channel (no cross-channel sum), distinct from a dense
//! conv1d (which sums over all input channels) and from an upsampling
//! transpose conv1d.
//!
//! `out[c, op] = bias[c] + Σ_k input[c, op·stride − pad + k·dilation] ·
//! weight[c, k]`, taps outside `[0, in_len)` contributing 0 (same/zero pad).
//! Forward (non-causal, non-transpose) — for a streaming single-step
//! depthwise conv use [`super::conv1d_causal_step_silu_cast_many`].
//!
//! Layouts:
//!   input  `[channels, in_len]`    T   (NCL, batch 1)
//!   weight `[channels, k]`         T   (depthwise `[C, 1, k]`)
//!   bias   `[channels]`            T
//!   out    `[channels, out_len]`   T
//!   out_len = (in_len + 2·pad − dilation·(k − 1) − 1) / stride + 1
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with
//! `grid_1d(channels · out_len, 256)`.
//!   - `out_len` matches the formula above (caller-computed).
//!   - `weight` is `channels · k`; `bias` is `channels`.

use wh_iron::kernel;

#[kernel]
pub fn iron_depthwise_conv1d<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] channels: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr] dilation: u32,
) {
    let raw = program_id::<0>();
    let in_range = raw < channels * out_len;
    let idx = select(in_range, raw, 0u32);
    let op = idx % out_len;
    let c = idx / out_len;
    let in_base = c * in_len;
    let w_base = c * k;
    let mut acc = load(bias[c]).cast::<f32>();
    // Input position of tap kx: op*stride + kx*dilation - pad. Compute the
    // pre-pad offset so the `>= pad` test keeps everything in u32.
    let start = op * stride;
    for kx in range(0u32, k, 1u32) {
        let pos_plus = start + kx * dilation; // = in_pos + pad
        let ge_pad = pos_plus >= pad;
        let ip = select(ge_pad, pos_plus - pad, 0u32);
        let valid = ge_pad & (ip < in_len);
        let ix = select(valid, ip, 0u32);
        let x = load(input[in_base + ix]).cast::<f32>();
        let x_m = select(valid, x, 0.0f32);
        let w = load(weight[w_base + kx]).cast::<f32>();
        acc = acc + x_m * w;
    }
    if in_range {
        store(out[idx], acc.cast::<T>());
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_depthwise_conv1d;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn naive(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        channels: usize,
        in_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let mut out = vec![0.0f32; channels * out_len];
        for c in 0..channels {
            for op in 0..out_len {
                let mut acc = bias[c];
                for kx in 0..k {
                    // in_pos = op*stride + kx*dilation - pad (may be negative).
                    let pos_plus = op * stride + kx * dilation;
                    if pos_plus >= pad {
                        let ip = pos_plus - pad;
                        if ip < in_len {
                            acc += input[c * in_len + ip] * weight[c * k + kx];
                        }
                    }
                }
                out[c * out_len + op] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        dt: DType,
        channels: usize,
        in_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let input_f = ramp(channels * in_len, 17, 4.0);
        let weight_f = ramp(channels * k, 7, 2.0);
        let bias_f = ramp(channels, 5, 0.5);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let weight = unpack_f32(&pack_f32(&weight_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let exp = naive(&input, &weight, &bias, channels, in_len, k, stride, pad, dilation);
        TestSetup::new(iron_depthwise_conv1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", channels * out_len, dt))
            .constexpr("channels", channels as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_1d(channels * out_len, 256)
    }

    // Conformer depthwise conv: 256 channels, 200 frames, kernel 15, same pad 7.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_depthwise_conv1d_conformer(dt: DType) -> TestSetup { setup(dt, 256, 200, 15, 1, 7, 1) }

    // Strided + dilated variant (codec / downsample). channels·out_len =
    // 8·15 = 120, deliberately NOT a multiple of the 256 threadgroup — so
    // `grid_1d` over-dispatches 136 tail threads, exercising the bounds
    // guard. Regression cover for the pre-guard OOB flake (max|Δ|=inf).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_depthwise_conv1d_strided(dt: DType) -> TestSetup { setup(dt, 8, 32, 3, 2, 1, 2) }

    // Small prime-ish tail: channels·out_len = 3·17 = 51, a second, very
    // differently-sized over-dispatch case so the guard is covered
    // independent of the strided shape's arithmetic.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_depthwise_conv1d_odd_tail(dt: DType) -> TestSetup { setup(dt, 3, 17, 3, 1, 1, 1) }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_depthwise_conv1d;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_depthwise_conv1d(dt: DType) -> BenchSetup {
        let (channels, in_len, k, stride, pad, dilation) =
            (512usize, 1500usize, 15usize, 1usize, 7usize, 1usize);
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = channels * out_len;
        BenchSetup::new(iron_depthwise_conv1d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", channels * in_len, dt))
            .buffer(BenchBuffer::random("weight", channels * k, dt))
            .buffer(BenchBuffer::random("bias", channels, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("channels", channels as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .grid_1d(n_out, 256)
            .bytes_moved((2 * n_out * dt.size_bytes()) as u64)
            .flops((n_out as u64) * (k as u64) * 2)
    }
}

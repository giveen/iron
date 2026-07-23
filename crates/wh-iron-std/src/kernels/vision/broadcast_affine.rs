//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Per-channel broadcast affine — `out[r, c] = x[r, c] · scale[c] + bias[c]`.
//!
//! The fused scale-then-shift a row-major `[rows, channels]` activation
//! needs after a per-channel statistic (Gemma 4's post-pool
//! `√hidden · pooled` then `(· − std_bias) · std_scale`, any "multiply by a
//! per-channel vector then add a per-channel vector" tail). `scale` / `bias`
//! are `[channels]`, broadcast down every row. One dispatch, no host
//! round-trip — keeps the tower GPU-resident where it used to read back,
//! affine on the CPU, and re-upload.
//!
//! Layouts:
//!   input  `[rows, channels]`   T
//!   scale  `[channels]`         T
//!   bias   `[channels]`         T
//!   out    `[rows, channels]`   T
//!
//! ## DISPATCH INVARIANTS
//!
//! Grid3D, one thread per output element — dispatch with
//! `grid_1d(rows · channels, 256)`.

use wh_iron::kernel;

#[kernel]
pub fn iron_broadcast_affine<T>(
    input: Tensor<T>,
    scale: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] channels: u32,
) {
    let idx = program_id::<0>();
    let c = idx % channels;
    let x = load(input[idx]).cast::<f32>();
    let s = load(scale[c]).cast::<f32>();
    let b = load(bias[c]).cast::<f32>();
    let y = x * s + b;
    store(out[idx], y.cast::<T>());
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_broadcast_affine;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    fn naive(input: &[f32], scale: &[f32], bias: &[f32], rows: usize, channels: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * channels];
        for r in 0..rows {
            for c in 0..channels {
                out[r * channels + c] = input[r * channels + c] * scale[c] + bias[c];
            }
        }
        out
    }

    fn setup(dt: DType, rows: usize, channels: usize) -> TestSetup {
        let n = rows * channels;
        let input_f = ramp(n, 17, 4.0);
        let scale_f = ramp(channels, 5, 2.0);
        let bias_f = ramp(channels, 7, 1.0);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let scale = unpack_f32(&pack_f32(&scale_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        let expected = naive(&input, &scale, &bias, rows, channels);
        TestSetup::new(iron_broadcast_affine::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("scale", pack_f32(&scale_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("channels", channels as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // Gemma 4 post-pool: 256 soft tokens × hidden 1152.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_broadcast_affine_gemma4(dt: DType) -> TestSetup { setup(dt, 256, 1152) }

    // Small ragged case.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_broadcast_affine_small(dt: DType) -> TestSetup { setup(dt, 5, 13) }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_broadcast_affine;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_broadcast_affine(dt: DType) -> BenchSetup {
        let (rows, channels) = (256usize, 1152usize);
        let n = rows * channels;
        BenchSetup::new(iron_broadcast_affine::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::random("scale", channels, dt))
            .buffer(BenchBuffer::random("bias", channels, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("channels", channels as u32)
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
            .flops((n as u64) * 2)
    }
}

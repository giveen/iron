//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Multi-row MoE router pre-score: per (token, expert)
//! `unbiased[t·E+e] = sigmoid(logits[...])`,
//! `biased[t·E+e] = unbiased + bias[e]`.
//!
//! Sibling of `ffai_moe_sigmoid_bias` (single-token); used by Hy3 Path B
//! multi-token prefill so the whole T·E score matrix stays on device.

use ffai_kernels::kernel;

/// T-row sigmoid + expert bias. `bias` is length `n_experts` (tiled across rows).
/// Grid: one thread per `t_rows * n_experts` element.
#[kernel]
pub fn ffai_moe_sigmoid_bias_rows<T>(
    logits: Tensor<T>,
    bias: Tensor<T>,
    mut unbiased: Tensor<T>,
    mut biased: Tensor<T>,
    #[constexpr] n_experts: u32,
    #[constexpr] t_rows: u32,
) {
    let tid = program_id::<0>();
    let total = t_rows * n_experts;
    if tid < total {
        let e = tid % n_experts;
        let s = 1.0f32 / (1.0f32 + exp(0.0f32 - load(logits[tid]).cast::<f32>()));
        store(unbiased[tid], s.cast::<T>());
        store(biased[tid], (s + load(bias[e]).cast::<f32>()).cast::<T>());
    }
}

pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_moe_sigmoid_bias_rows;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(dt: DType, t_rows: usize, n_experts: usize) -> TestSetup {
        let total = t_rows * n_experts;
        let logits: Vec<f32> = (0..total).map(|i| (i % 31) as f32 * 0.1 - 1.5).collect();
        let bias: Vec<f32> = (0..n_experts).map(|i| (i % 7) as f32 * 0.05 - 0.15).collect();
        let l_dt = unpack_f32(&pack_f32(&logits, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&bias, dt), dt);
        let mut unbiased = vec![0.0f32; total];
        let mut biased = vec![0.0f32; total];
        for t in 0..t_rows {
            for (e, &bv) in b_dt.iter().enumerate() {
                let i = t * n_experts + e;
                let s = 1.0_f32 / (1.0 + (-l_dt[i]).exp());
                unbiased[i] = s;
                biased[i] = s + bv;
            }
        }
        TestSetup::new(ffai_moe_sigmoid_bias_rows::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("logits", pack_f32(&logits, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("unbiased", total, dt))
            .input(TestBuffer::zeros("biased", total, dt))
            .constexpr("n_experts", n_experts as u32)
            .constexpr("t_rows", t_rows as u32)
            .expect(TestBuffer::from_vec("unbiased", pack_f32(&unbiased, dt), dt))
            .expect(TestBuffer::from_vec("biased", pack_f32(&biased, dt), dt))
            .grid_1d(total, 64)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-4, 2e-2, 5e-2])]
    fn test_moe_sigmoid_bias_rows_hy3(dt: DType) -> TestSetup {
        // Hy3: 192 experts, multi-token prefill slice.
        setup(dt, 4, 192)
    }

    #[test_kernel(dtypes = [f32], tol = [2e-4])]
    fn test_moe_sigmoid_bias_rows_single(dt: DType) -> TestSetup { setup(dt, 1, 256) }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_moe_sigmoid_bias_rows;

    /// Hy3 shape: 192 experts, multi-token prefill slice.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_sigmoid_bias_rows(dt: DType) -> BenchSetup {
        let t_rows = 4usize;
        let n_experts = 192usize;
        let total = t_rows * n_experts;
        BenchSetup::new(ffai_moe_sigmoid_bias_rows::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("logits", total, dt))
            .buffer(BenchBuffer::random("bias", n_experts, dt))
            .buffer(BenchBuffer::zeros("unbiased", total, dt).output())
            .buffer(BenchBuffer::zeros("biased", total, dt).output())
            .constexpr("n_experts", n_experts as u32)
            .constexpr("t_rows", t_rows as u32)
            .with_shape_label(format!(
                "hy3 T{t_rows} E{n_experts} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_1d(total, 64)
            .bytes_moved(((2 * total + n_experts) * dt.size_bytes()) as u64)
    }
}

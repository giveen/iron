//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Per-expert indexed dequantizing GEMV (MLX affine, scales + biases).
//!
//! Stacked weights `[n_experts, out_dim, packs]` with GPU-resident
//! `expert_index[0]` selecting the slab. int4 + int2 pack-strided forms.

use ffai_kernels::kernel;

/// int4 expert-indexed dequant GEMV.
#[kernel]
pub fn ffai_dequant_gemv_int4_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<T>,
    biases_stacked: Tensor<T>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] group_size: u32,
) {
    let vals_per_pack = 8u32;
    let mask = 0xFu32;
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_groups;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_group_off = scale_expert_off + row * n_groups;
    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[row_group_off + g]).cast::<f32>();
            let bias = load(biases_stacked[row_group_off + g]).cast::<f32>();
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 4u32)) & mask;
                acc = acc + (q.cast::<f32>() * scale + bias) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

/// int2 expert-indexed dequant GEMV — Hy3 oQ2 switch_mlp path.
#[kernel]
pub fn ffai_dequant_gemv_int2_expert_indexed<T>(
    weights_stacked: Tensor<u32>,
    scales_stacked: Tensor<T>,
    biases_stacked: Tensor<T>,
    input: Tensor<T>,
    expert_index: Tensor<u32>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] group_size: u32,
) {
    let vals_per_pack = 16u32;
    let mask = 3u32;
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / vals_per_pack;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / vals_per_pack;
    let expert = load(expert_index[0u32]);
    let weight_expert_off = expert * out_dim * n_packs_per_row;
    let scale_expert_off = expert * out_dim * n_groups;
    let row_pack_off = weight_expert_off + row * n_packs_per_row;
    let row_group_off = scale_expert_off + row * n_groups;
    let mut acc = 0.0f32;
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales_stacked[row_group_off + g]).cast::<f32>();
            let bias = load(biases_stacked[row_group_off + g]).cast::<f32>();
            let packed = load(weights_stacked[row_pack_off + pack_idx]);
            let p_off = pack_idx * vals_per_pack;
            for i in range(0u32, vals_per_pack, 1u32) {
                let q = (packed >> (i * 2u32)) & mask;
                acc = acc + (q.cast::<f32>() * scale + bias) * load(input[p_off + i]).cast::<f32>();
            }
        }
    }
    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

pub mod kernel_tests {
    use ffai_kernels::{prelude::Kernel, test::*, test_kernel};

    use super::{ffai_dequant_gemv_int2_expert_indexed, ffai_dequant_gemv_int4_expert_indexed};
    use crate::utils::{pack_f32, unpack_f32};

    fn pack_codes(codes: &[u32], bits: u32) -> Vec<u32> {
        let vals_per_pack = 32 / bits;
        let mut out = Vec::new();
        let mut i = 0;
        while i < codes.len() {
            let mut word = 0u32;
            for v in 0..vals_per_pack {
                if i + (v as usize) < codes.len() {
                    word |= (codes[i + v as usize] & ((1u32 << bits) - 1)) << (v * bits);
                }
            }
            out.push(word);
            i += vals_per_pack as usize;
        }
        out
    }

    // Packed affine oracle: many shape fields by design (weights/scales/x/bits).
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        w_packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        expert: usize,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        bits: u32,
    ) -> Vec<f32> {
        let n_groups = in_dim / group_size;
        let vals_per_pack = (32 / bits) as usize;
        let n_packs = in_dim / vals_per_pack;
        let mut out = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut acc = 0.0f32;
            for (i, &xi) in x.iter().enumerate().take(in_dim) {
                let pack_idx = i / vals_per_pack;
                let lane = i % vals_per_pack;
                let word = w_packed[expert * out_dim * n_packs + row * n_packs + pack_idx];
                let q = (word >> (lane as u32 * bits)) & ((1u32 << bits) - 1);
                let g = i / group_size;
                let s = scales[expert * out_dim * n_groups + row * n_groups + g];
                let b = biases[expert * out_dim * n_groups + row * n_groups + g];
                acc += (q as f32 * s + b) * xi;
            }
            out[row] = acc;
        }
        out
    }

    fn setup_bits(bits: u32, dt: DType, kernel: Kernel) -> TestSetup {
        let (n_experts, out_dim, in_dim, group_size) = (4usize, 4usize, 256usize, 64usize);
        let n_groups = in_dim / group_size;
        let vals_per_pack = (32 / bits) as usize;
        let n_packs = in_dim / vals_per_pack;
        let expert = 2usize;

        let mut codes = vec![0u32; n_experts * out_dim * in_dim];
        for e in 0..n_experts {
            for r in 0..out_dim {
                for i in 0..in_dim {
                    codes[e * out_dim * in_dim + r * in_dim + i] =
                        ((e * 3 + r * 5 + i * 7) as u32) % (1u32 << bits);
                }
            }
        }
        let mut packed = Vec::with_capacity(n_experts * out_dim * n_packs);
        for e in 0..n_experts {
            for r in 0..out_dim {
                let base = e * out_dim * in_dim + r * in_dim;
                packed.extend(pack_codes(&codes[base..base + in_dim], bits));
            }
        }
        let scales_f: Vec<f32> =
            (0..n_experts * out_dim * n_groups).map(|i| 0.01 + (i % 17) as f32 * 0.001).collect();
        let biases_f: Vec<f32> =
            (0..n_experts * out_dim * n_groups).map(|i| -0.05 + (i % 13) as f32 * 0.002).collect();
        let x_f: Vec<f32> = (0..in_dim).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let expected = oracle(&packed, &s, &b, &x, expert, out_dim, in_dim, group_size, bits);
        let mut wbytes = Vec::new();
        for w in &packed {
            wbytes.extend_from_slice(&w.to_le_bytes());
        }
        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("weights_stacked", wbytes, DType::U32))
            .input(TestBuffer::from_vec("scales_stacked", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases_stacked", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("input", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec(
                "expert_index",
                (expert as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .input(TestBuffer::zeros("output", out_dim, dt))
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("output", pack_f32(&expected, dt), dt))
            .grid_3d(out_dim as u32, 1, 1, [32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int4_expert_indexed(dt: DType) -> TestSetup {
        setup_bits(4, dt, ffai_dequant_gemv_int4_expert_indexed::kernel_ir_for(dt))
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_dequant_gemv_int2_expert_indexed(dt: DType) -> TestSetup {
        setup_bits(2, dt, ffai_dequant_gemv_int2_expert_indexed::kernel_ir_for(dt))
    }
}

/// New-syntax benchmarks for expert-indexed dequant GEMV.
/// int4: classic 4096×4096 gs=64. int2: Hy3 oQ2 switch shape 1536×4096 gs=64.
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::{ffai_dequant_gemv_int2_expert_indexed, ffai_dequant_gemv_int4_expert_indexed};

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int4_expert_indexed(dt: DType) -> BenchSetup {
        let (n_experts, out_dim, in_dim, group_size) = (8usize, 4096usize, 4096usize, 64usize);
        let n_groups = in_dim / group_size;
        let packs_per_row = in_dim / 8;
        let sz = dt.size_bytes();
        let bytes =
            out_dim * packs_per_row * 4 + 2 * out_dim * n_groups * sz + in_dim * sz + out_dim * sz;
        BenchSetup::new(ffai_dequant_gemv_int4_expert_indexed::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random(
                "weights_stacked",
                n_experts * out_dim * packs_per_row,
                DType::U32,
            ))
            .buffer(BenchBuffer::random("scales_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("expert_index", 1, DType::U32))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64)
    }

    /// Hy3 oQ2 intermediate: out=1536, in=4096, gs=64, int2.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_dequant_gemv_int2_expert_indexed(dt: DType) -> BenchSetup {
        let (n_experts, out_dim, in_dim, group_size) = (8usize, 1536usize, 4096usize, 64usize);
        let n_groups = in_dim / group_size;
        let packs_per_row = in_dim / 16; // int2: 16 codes/u32
        let sz = dt.size_bytes();
        let bytes =
            out_dim * packs_per_row * 4 + 2 * out_dim * n_groups * sz + in_dim * sz + out_dim * sz;
        BenchSetup::new(ffai_dequant_gemv_int2_expert_indexed::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random(
                "weights_stacked",
                n_experts * out_dim * packs_per_row,
                DType::U32,
            ))
            .buffer(BenchBuffer::random("scales_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("biases_stacked", n_experts * out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("input", in_dim, dt))
            .buffer(BenchBuffer::zeros("expert_index", 1, DType::U32))
            .buffer(BenchBuffer::zeros("output", out_dim, dt).output())
            .constexpr("in_dim", in_dim as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d(out_dim as u32, 1, 1, [64, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * out_dim as u64 * in_dim as u64)
    }
}

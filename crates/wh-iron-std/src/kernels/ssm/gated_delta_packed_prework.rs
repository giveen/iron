//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Packed Gated DeltaNet verifier prework.
//!
//! This kernel combines the causal depthwise convolution, SiLU, paired
//! input casts, and Q/K normalization used before a chunked recurrence.
//! Each threadgroup owns one token and one logical Q, K, or V head. The
//! causal window is reconstructed directly from the incoming state and
//! preceding source rows, so tokens remain parallel without changing the
//! sequential convolution result.
//!
//! Layouts:
//!   src             [T, C] T
//!   weight          [4, C] T
//!   bias            [C] T
//!   a_raw, b_raw    [T, Hv] T
//!   state_in/out    [3, C] T
//!   conv_out        [T, C] f32
//!   q_normed        [T, Hk, Dk] f32
//!   k_normed        [T, Hk, Dk] f32
//!   a_out, b_out    [T, Hv] f32
//!
//! Grid: `[T, 2*Hk+Hv, 1]`, threadgroup: `[32, 1, 1]`.

use wh_iron::kernel;

#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn iron_gated_delta_packed_prework<T>(
    src: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    a_raw: Tensor<T>,
    b_raw: Tensor<T>,
    state_in: Tensor<T>,
    q_norm_weight: Tensor<f32>,
    k_norm_weight: Tensor<f32>,
    mut conv_out: Tensor<f32>,
    mut state_out: Tensor<T>,
    mut state_planes: Tensor<T>,
    planes_enabled: Tensor<u32>,
    mut q_normed: Tensor<f32>,
    mut k_normed: Tensor<f32>,
    mut a_out: Tensor<f32>,
    mut b_out: Tensor<f32>,
    #[constexpr] t_len: u32,
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let row = tgid_x;
    let logical_head = tgid_y;
    let lane = tid;
    let is_q = logical_head < hk;
    let is_k = logical_head >= hk && logical_head < 2u32 * hk;
    // Dk == Dv is a dispatch invariant. It makes the Q, K, and V slabs a
    // single run of equal-width logical heads.
    let head =
        select(is_q, logical_head, select(is_k, logical_head - hk, logical_head - 2u32 * hk));
    let head_dim = dk;
    let channel_base = logical_head * dk;
    let conv_dim = 2u32 * hk * dk + hv * dv;
    let n_per_lane = head_dim / 32u32;
    let planes_on = load(planes_enabled[0]);

    stack_alloc("activated", 8u32, "f32");
    let mut sumsq = 0.0f32;
    for i in range(0u32, n_per_lane, 1u32) {
        let local_channel = lane * n_per_lane + i;
        let channel = channel_base + local_channel;

        let mut x0 = 0.0f32;
        let mut x1 = 0.0f32;
        let mut x2 = 0.0f32;
        if row == 0u32 {
            x0 = load(state_in[channel]).cast::<f32>();
            x1 = load(state_in[conv_dim + channel]).cast::<f32>();
            x2 = load(state_in[2u32 * conv_dim + channel]).cast::<f32>();
        } else if row == 1u32 {
            x0 = load(state_in[conv_dim + channel]).cast::<f32>();
            x1 = load(state_in[2u32 * conv_dim + channel]).cast::<f32>();
            x2 = load(src[channel]).cast::<f32>();
        } else if row == 2u32 {
            x0 = load(state_in[2u32 * conv_dim + channel]).cast::<f32>();
            x1 = load(src[channel]).cast::<f32>();
            x2 = load(src[conv_dim + channel]).cast::<f32>();
        } else {
            x0 = load(src[(row - 3u32) * conv_dim + channel]).cast::<f32>();
            x1 = load(src[(row - 2u32) * conv_dim + channel]).cast::<f32>();
            x2 = load(src[(row - 1u32) * conv_dim + channel]).cast::<f32>();
        }
        let x3 = load(src[row * conv_dim + channel]).cast::<f32>();

        let b = load(bias[channel]).cast::<f32>();
        let w0 = load(weight[channel]).cast::<f32>();
        let w1 = load(weight[conv_dim + channel]).cast::<f32>();
        let w2 = load(weight[2u32 * conv_dim + channel]).cast::<f32>();
        let w3 = load(weight[3u32 * conv_dim + channel]).cast::<f32>();
        let acc = b + w3 * x3 + w0 * x0 + w1 * x1 + w2 * x2;
        let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - acc));
        let activated = acc * sig;
        store(conv_out[row * conv_dim + channel], activated);
        stack_store("activated", i, activated);
        sumsq = sumsq + activated * activated;

        let is_prefix = row + 1u32 < t_len;
        let is_tail_prefix = is_prefix && row + 5u32 >= t_len;
        if planes_on == 1u32
            || (planes_on == 2u32 && is_prefix)
            || (planes_on == 3u32 && is_tail_prefix)
        {
            let plane_base = row * 3u32 * conv_dim + channel;
            store(state_planes[plane_base], x1.cast::<T>());
            store(state_planes[plane_base + conv_dim], x2.cast::<T>());
            store(state_planes[plane_base + 2u32 * conv_dim], x3.cast::<T>());
        }
        if row + 1u32 == t_len {
            store(state_out[channel], x1.cast::<T>());
            store(state_out[conv_dim + channel], x2.cast::<T>());
            store(state_out[2u32 * conv_dim + channel], x3.cast::<T>());
        }
    }

    if is_q || is_k {
        let total = simd_sum(sumsq);
        let inv = rsqrt(total / head_dim.cast::<f32>() + 0.000001f32);
        let out_base = (row * hk + head) * dk;
        for i in range(0u32, n_per_lane, 1u32) {
            let local_channel = lane * n_per_lane + i;
            let value = stack_load("activated", i) * inv;
            if is_q {
                let scale = load(q_norm_weight[head * dk + local_channel]);
                store(q_normed[out_base + local_channel], value * scale);
            } else {
                let scale = load(k_norm_weight[head * dk + local_channel]);
                store(k_normed[out_base + local_channel], value * scale);
            }
        }
    } else if lane == 0u32 {
        let scalar = row * hv + head;
        store(a_out[scalar], load(a_raw[scalar]).cast::<f32>());
        store(b_out[scalar], load(b_raw[scalar]).cast::<f32>());
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_gated_delta_packed_prework;
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn oracle(
        src: &[f32],
        weight: &[f32],
        bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        state_in: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        t: usize,
        dk: usize,
        hv: usize,
        hk: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let conv_dim = (2 * hk + hv) * dk;
        let mut conv = vec![0.0f32; t * conv_dim];
        let mut state_out = vec![0.0f32; 3 * conv_dim];
        let mut planes = vec![0.0f32; t * 3 * conv_dim];
        for d in 0..conv_dim {
            let mut s0 = state_in[d];
            let mut s1 = state_in[conv_dim + d];
            let mut s2 = state_in[2 * conv_dim + d];
            for row in 0..t {
                let x = src[row * conv_dim + d];
                let acc = bias[d]
                    + weight[3 * conv_dim + d] * x
                    + weight[d] * s0
                    + weight[conv_dim + d] * s1
                    + weight[2 * conv_dim + d] * s2;
                conv[row * conv_dim + d] = acc / (1.0 + (-acc).exp());
                s0 = s1;
                s1 = s2;
                s2 = x;
                let plane = row * 3 * conv_dim + d;
                planes[plane] = s0;
                planes[plane + conv_dim] = s1;
                planes[plane + 2 * conv_dim] = s2;
            }
            state_out[d] = s0;
            state_out[conv_dim + d] = s1;
            state_out[2 * conv_dim + d] = s2;
        }

        let mut q_normed = vec![0.0f32; t * hk * dk];
        let mut k_normed = vec![0.0f32; t * hk * dk];
        for row in 0..t {
            for head in 0..hk {
                let q_base = row * conv_dim + head * dk;
                let k_base = row * conv_dim + (hk + head) * dk;
                let q_ssq: f32 = conv[q_base..q_base + dk].iter().map(|x| x * x).sum();
                let k_ssq: f32 = conv[k_base..k_base + dk].iter().map(|x| x * x).sum();
                let q_inv = 1.0 / (q_ssq / dk as f32 + 1e-6).sqrt();
                let k_inv = 1.0 / (k_ssq / dk as f32 + 1e-6).sqrt();
                let out_base = (row * hk + head) * dk;
                for d in 0..dk {
                    q_normed[out_base + d] = conv[q_base + d] * q_inv * q_weight[head * dk + d];
                    k_normed[out_base + d] = conv[k_base + d] * k_inv * k_weight[head * dk + d];
                }
            }
        }
        (conv, state_out, planes, q_normed, k_normed, a_raw.to_vec(), b_raw.to_vec())
    }

    fn setup(dt: DType) -> TestSetup {
        let (t, dk, hv, hk) = (6usize, 32usize, 4usize, 2usize);
        let dv = dk;
        let conv_dim = (2 * hk + hv) * dk;
        let src_f = ramp(t * conv_dim, 17, 1.0);
        let weight_f = ramp(4 * conv_dim, 11, 0.2);
        let bias_f = ramp(conv_dim, 7, 0.1);
        let a_f = ramp(t * hv, 9, 0.8);
        let b_f = ramp(t * hv, 13, 0.8);
        let state_f = ramp(3 * conv_dim, 19, 1.0);
        let q_weight: Vec<f32> = (0..hk * dk).map(|i| 0.8 + (i % 7) as f32 * 0.03).collect();
        let k_weight: Vec<f32> = (0..hk * dk).map(|i| 0.7 + (i % 5) as f32 * 0.04).collect();
        let round = |x: &[f32]| unpack_f32(&pack_f32(x, dt), dt);
        let (conv, state, mut planes, q, k, a, b) = oracle(
            &round(&src_f),
            &round(&weight_f),
            &round(&bias_f),
            &round(&a_f),
            &round(&b_f),
            &round(&state_f),
            &q_weight,
            &k_weight,
            t,
            dk,
            hv,
            hk,
        );
        let state = unpack_f32(&pack_f32(&state, dt), dt);
        planes = unpack_f32(&pack_f32(&planes, dt), dt);
        let plane_size = 3 * conv_dim;
        planes[..plane_size].fill(0.0);
        planes[(t - 1) * plane_size..].fill(0.0);

        TestSetup::new(iron_gated_delta_packed_prework::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("src", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec("weight", pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_f, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_f, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_f, dt), dt))
            .input(TestBuffer::from_vec(
                "q_norm_weight",
                pack_f32(&q_weight, DType::F32),
                DType::F32,
            ))
            .input(TestBuffer::from_vec(
                "k_norm_weight",
                pack_f32(&k_weight, DType::F32),
                DType::F32,
            ))
            .input(TestBuffer::zeros("conv_out", t * conv_dim, DType::F32))
            .input(TestBuffer::zeros("state_out", 3 * conv_dim, dt))
            .input(TestBuffer::zeros("state_planes", t * 3 * conv_dim, dt))
            .input(TestBuffer::from_vec("planes_enabled", 3u32.to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::zeros("q_normed", t * hk * dk, DType::F32))
            .input(TestBuffer::zeros("k_normed", t * hk * dk, DType::F32))
            .input(TestBuffer::zeros("a_out", t * hv, DType::F32))
            .input(TestBuffer::zeros("b_out", t * hv, DType::F32))
            .constexpr("t_len", t as u32)
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .expect(TestBuffer::from_vec("conv_out", pack_f32(&conv, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state, dt), dt))
            .expect(TestBuffer::from_vec("state_planes", pack_f32(&planes, dt), dt))
            .expect(TestBuffer::from_vec("q_normed", pack_f32(&q, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("k_normed", pack_f32(&k, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("a_out", pack_f32(&a, DType::F32), DType::F32))
            .expect(TestBuffer::from_vec("b_out", pack_f32(&b, DType::F32), DType::F32))
            .grid_3d(t as u32, (2 * hk + hv) as u32, 1, [32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 1e-2, 5e-2])]
    fn test_gated_delta_packed_prework(dt: DType) -> TestSetup { setup(dt) }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_gated_delta_packed_prework;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_packed_prework(dt: DType) -> BenchSetup {
        let t = 9usize;
        let dk = 128usize;
        let dv = 128usize;
        let hv = 48usize;
        let hk = 16usize;
        let conv_dim = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_packed_prework::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("src", t * conv_dim, dt))
            .buffer(BenchBuffer::random("weight", 4 * conv_dim, dt))
            .buffer(BenchBuffer::random("bias", conv_dim, dt))
            .buffer(BenchBuffer::random("a_raw", t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", t * hv, dt))
            .buffer(BenchBuffer::random("state_in", 3 * conv_dim, dt))
            .buffer(BenchBuffer::random("q_norm_weight", hk * dk, DType::F32))
            .buffer(BenchBuffer::random("k_norm_weight", hk * dk, DType::F32))
            .buffer(BenchBuffer::zeros("conv_out", t * conv_dim, DType::F32).output())
            .buffer(BenchBuffer::zeros("state_out", 3 * conv_dim, dt).output())
            .buffer(BenchBuffer::zeros("state_planes", t * 3 * conv_dim, dt).output())
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("q_normed", t * hk * dk, DType::F32).output())
            .buffer(BenchBuffer::zeros("k_normed", t * hk * dk, DType::F32).output())
            .buffer(BenchBuffer::zeros("a_out", t * hv, DType::F32).output())
            .buffer(BenchBuffer::zeros("b_out", t * hv, DType::F32).output())
            .constexpr("t_len", t as u32)
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .grid_3d(t as u32, (2 * hk + hv) as u32, 1, [32, 1, 1])
    }
}

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet — q/k RMSNorm pre-pass for the chunked prep kernel.
//!
//! `mt_gated_delta_prep_chunk`'s "Phase 0a" (per-head RMSNorm of q/k) does
//! not depend on recurrent state — only on `conv_out` for that token. But
//! the chunk kernel's dispatch grid is `[Dv, B·Hv, 1]`: every one of the
//! `Dv` threadgroups sharing a given `(b, hv_idx)` redundantly recomputes
//! the *identical* q/k RMSNorm (q/k have no `dv_idx` dependence), and GQA
//! groups (`hk_per_hv = Hv/Hk`) redundantly recompute it again across
//! their shared `hk_idx`. Net redundancy factor: `Dv · (Hv/Hk)`.
//!
//! This kernel computes q/k RMSNorm **once per `(b, t, hk_idx)`** — grid
//! `[T, B·Hk, 1]`, fully parallel across tokens — and writes dense
//! `q_normed` / `k_normed` tensors that the (now-lighter) chunk kernel
//! loads directly, dropping its Phase 0a to two tensor reads.
//!
//! Inputs:
//!   - `conv_out`     : Tensor<T> [B, T, 2·Hk·Dk + Hv·Dv]   q | k | v slabs
//!   - `q_norm_weight`: Tensor<T> [Hk·Dk]
//!   - `k_norm_weight`: Tensor<T> [Hk·Dk]
//!   - `t_len`        : Tensor<u32> [1]                     runtime chunk length
//!
//! Outputs:
//!   - `q_normed`     : Tensor<T> [B, T, Hk, Dk]
//!   - `k_normed`     : Tensor<T> [B, T, Hk, Dk]
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** Each TG is one simdgroup (32 threads).
//! - **Grid: `[T, B·Hk, 1]`, TG: `[32, 1, 1]`.**
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` slots.
//! - **`t_len` is runtime u32** so a single PSO compiles for every chunk size,
//!   matching `mt_gated_delta_prep_chunk`.
//! - `hv` / `dv` are constexpr purely to reconstruct `conv_out`'s per-token
//!   stride (`2·Hk·Dk + Hv·Dv`) — this kernel never indexes by `hv_idx`/`dv_idx`.

use ffai_kernels::kernel;

#[kernel]
pub fn mt_gated_delta_qknorm_prepass<T>(
    conv_out: Tensor<T>,      // [B, T, 2·Hk·Dk + Hv·Dv]
    q_norm_weight: Tensor<T>, // [Hk·Dk]
    k_norm_weight: Tensor<T>, // [Hk·Dk]
    mut q_normed: Tensor<T>,  // [B, T, Hk, Dk]
    mut k_normed: Tensor<T>,  // [B, T, Hk, Dk]
    t_len: Tensor<u32>,       // [1] scalar
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let t = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;
    let hk_idx = n - (n / hk) * hk;
    let b = n / hk;
    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);
    let stride_b = 2u32 * hk * dk + hv * dv;
    let eps = 0.000001f32;
    let dk_f = dk.cast::<f32>();
    let bt = b * t_total + t;
    let conv_base = bt * stride_b;
    let q_off = conv_base + hk_idx * dk;
    let k_off = conv_base + hk * dk + hk_idx * dk;
    // ─── Phase 0a: per-head RMSNorm of q / k (state-independent) ─────────
    stack_alloc("q_raw", 8u32, "f32");
    stack_alloc("k_raw", 8u32, "f32");
    let mut q_ssq = 0.0f32;
    let mut k_ssq = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qv = load(conv_out[q_off + s_idx]).cast::<f32>();
        let kv = load(conv_out[k_off + s_idx]).cast::<f32>();
        stack_store("q_raw", i, qv);
        stack_store("k_raw", i, kv);
        q_ssq = q_ssq + qv * qv;
        k_ssq = k_ssq + kv * kv;
    }
    let q_ssq_sum = simd_sum(q_ssq);
    let k_ssq_sum = simd_sum(k_ssq);
    let q_inv = rsqrt(q_ssq_sum / dk_f + eps);
    let k_inv = rsqrt(k_ssq_sum / dk_f + eps);
    // ─── Write dense q_normed / k_normed [B, T, Hk, Dk] ──────────────────
    let out_base = (bt * hk + hk_idx) * dk;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qw = load(q_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        let kw = load(k_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        let q_normed_val = stack_load("q_raw", i) * q_inv * qw;
        let k_normed_val = stack_load("k_raw", i) * k_inv * kw;
        store(q_normed[out_base + s_idx], q_normed_val.cast::<T>());
        store(k_normed[out_base + s_idx], k_normed_val.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use ffai_kernels::core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    #[test]
    fn dump() {
        use ffai_kernels::codegen::msl::MslGenerator;
        let mut k = mt_gated_delta_qknorm_prepass::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

/// Correctness for the q/k RMSNorm pre-pass. Oracle is the same per-head
/// RMSNorm math as `mt_gated_delta_prep_chunk`'s Phase 0a, evaluated once
/// per `(b, t, hk_idx)` instead of redundantly per `(b, t, hv_idx, dv_idx)`.
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(t_total, b*hk, 1, [32,1,1])`;
/// `t_len` is a runtime u32 scalar buffer.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::mt_gated_delta_qknorm_prepass;
    use crate::utils::{pack_f32, unpack_f32};

    /// CPU oracle for the q/k RMSNorm pre-pass. Layout matches the kernel:
    /// `conv_out`/`a_raw`/`b_raw` carry a T dim; outputs are `[B,T,Hk,Dk]`.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        conv_out: &[f32], // [B, T, 2·Hk·Dk + Hv·Dv]
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-6_f32;
        let stride_b = 2 * hk * dk + hv * dv;
        let mut q_normed = vec![0.0_f32; b * t_total * hk * dk];
        let mut k_normed = vec![0.0_f32; b * t_total * hk * dk];
        for batch in 0..b {
            for t in 0..t_total {
                let bt = batch * t_total + t;
                let conv_base = bt * stride_b;
                for hk_idx in 0..hk {
                    let q_row = conv_base + hk_idx * dk;
                    let k_row = conv_base + hk * dk + hk_idx * dk;
                    let mut q_ssq = 0.0_f32;
                    let mut k_ssq = 0.0_f32;
                    for d in 0..dk {
                        let qv = conv_out[q_row + d];
                        let kv = conv_out[k_row + d];
                        q_ssq += qv * qv;
                        k_ssq += kv * kv;
                    }
                    let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                    let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();
                    let out_base = (bt * hk + hk_idx) * dk;
                    for d in 0..dk {
                        q_normed[out_base + d] =
                            conv_out[q_row + d] * q_inv * q_norm_weight[hk_idx * dk + d];
                        k_normed[out_base + d] =
                            conv_out[k_row + d] * k_inv * k_norm_weight[hk_idx * dk + d];
                    }
                }
            }
        }
        (q_normed, k_normed)
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        weight_scale: f32,
        conv_scale: f32,
        dt: DType,
    ) -> TestSetup {
        let stride_b = 2 * hk * dk + hv * dv;
        let conv_out: Vec<f32> =
            (0..b * t_total * stride_b).map(|i| ((i as f32) * 0.0131).sin() * conv_scale).collect();
        let q_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect();
        let k_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect();

        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (q_exp, k_exp) = oracle(
            &r(&conv_out),
            &r(&q_norm_weight),
            &r(&k_norm_weight),
            b,
            t_total,
            hv,
            hk,
            dv,
            dk,
        );

        TestSetup::new(mt_gated_delta_qknorm_prepass::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("conv_out", pack_f32(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("q_norm_weight", pack_f32(&q_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("k_norm_weight", pack_f32(&k_norm_weight, dt), dt))
            .input(TestBuffer::zeros("q_normed", b * t_total * hk * dk, dt))
            .input(TestBuffer::zeros("k_normed", b * t_total * hk * dk, dt))
            .input(TestBuffer::from_vec(
                "t_len",
                (t_total as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .expect(TestBuffer::from_vec("q_normed", pack_f32(&q_exp, dt), dt))
            .expect(TestBuffer::from_vec("k_normed", pack_f32(&k_exp, dt), dt))
            .grid_3d(t_total as u32, (b * hk) as u32, 1, [32, 1, 1])
    }

    // GQA (Hv = 2·Hk), T=4 tokens.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_mt_gated_delta_qknorm_prepass_gqa(dt: DType) -> TestSetup {
        setup(1, 4, 4, 2, 8, 64, 0.3, 0.02, dt)
    }

    // Hv == Hk (no key-sharing) at minimum dk=32, T=3 tokens.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_mt_gated_delta_qknorm_prepass_no_gqa(dt: DType) -> TestSetup {
        setup(1, 3, 4, 4, 4, 32, 1.0, 0.4, dt)
    }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::mt_gated_delta_qknorm_prepass;

    // Grid `[t, b*hk, 1]`, TG `[32,1,1]`, Reduction. `t_len` is a runtime
    // u32 scalar. Shape mirrors `bench_gated_delta_prep_chunk`.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_qknorm_prepass(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 64usize, 4usize, 2usize, 8usize, 64usize);
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(mt_gated_delta_qknorm_prepass::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("q_norm_weight", hk * dk, dt))
            .buffer(BenchBuffer::random("k_norm_weight", hk * dk, dt))
            .buffer(BenchBuffer::zeros("q_normed", b * t * hk * dk, dt).output())
            .buffer(BenchBuffer::zeros("k_normed", b * t * hk * dk, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .grid_3d(t as u32, (b * hk) as u32, 1, [32, 1, 1])
            .bytes_moved((b * t * hk * dk * 2 * dt.size_bytes()) as u64)
    }
}

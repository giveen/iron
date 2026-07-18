//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet — **fused** prep + chunked-prefill kernel.
//!
//! `ffai_gated_delta_prep_chunk` extends
//! [`ffai_gated_delta_prep_step`](super::gated_delta_prep::ffai_gated_delta_prep_step)
//! over a chunk of `T` tokens, mirroring the relationship between
//! [`ffai_gated_delta_step`](super::gated_delta::ffai_gated_delta_step) and
//! [`ffai_gated_delta_chunk`](super::gated_delta::ffai_gated_delta_chunk).
//!
//! State stays register-resident across the entire `T`-loop — one
//! load_state at entry and one store_state at exit, regardless of `T`.
//! This collapses the dominant `ffai_gated_delta_prep_step`-per-token T-loop
//! in `Qwen35GDNMixer.forwardMany` to a single dispatch per layer.
//!
//! The per-head RMSNorm of q/k (state-independent) is **not** computed
//! here — it's hoisted into
//! [`ffai_gated_delta_qknorm_prepass`](super::gated_delta_qknorm_prepass),
//! run once per `(b, t, hk_idx)` ahead of this kernel, because this
//! kernel's own grid (`[Dv, B·Hv, 1]`) would otherwise redundantly
//! recompute the identical q/k norm `Dv · (Hv/Hk)` times per token. This
//! kernel just loads the dense `q_normed` / `k_normed` the pre-pass wrote.
//!
//! Inputs (note the added `T` dimension on conv_out / a_raw / b_raw):
//!   - `conv_out`  : Tensor<T> [B, T, 2·Hk·Dk + Hv·Dv]   q | k | v slabs (v used here; q/k slabs are read by the pre-pass, not this kernel)
//!   - `a_log`     : Tensor<T> [Hv]                      per-Hv learnable
//!   - `dt_bias`   : Tensor<T> [Hv]
//!   - `a_raw`     : Tensor<T> [B, T, Hv]
//!   - `b_raw`     : Tensor<T> [B, T, Hv]
//!   - `q_normed`  : Tensor<T> [B, T, Hk, Dk]   from `ffai_gated_delta_qknorm_prepass`
//!   - `k_normed`  : Tensor<T> [B, T, Hk, Dk]   from `ffai_gated_delta_qknorm_prepass`
//!   - `state_in`  : Tensor<T> [B, Hv, Dv, Dk]           (one state per (b, hv))
//!   - `t_len`     : Tensor<u32> [1]                     runtime chunk length
//!
//! Outputs:
//!   - `state_out`    : Tensor<T> [B, Hv, Dv, Dk]
//!   - `y`            : Tensor<T> [B, T, Hv, Dv]
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** TG packs **4 simdgroups** (128 threads) — one SG
//!   per `dv` slot inside the TG (matches reference Metal GDN `(32,4,1)`).
//! - **Grid: `[ceil(Dv/4), B·Hv, 1]`, TG: `[128, 1, 1]`.**
//!   `dv_idx = tgid_x·4 + simd_group_id`, `dk_idx = simd_lane`.
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` slots.
//! - **Hv divisible by Hk.** GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//! - **`t_len` is runtime u32** so a single PSO compiles for every chunk size.
//! - **Must run after `ffai_gated_delta_qknorm_prepass`** on the same
//!   `conv_out` / `t_len` — this kernel does not validate that `q_normed`
//!   / `k_normed` were populated for the same chunk.
//!
//! ## Per-iter cost vs prep_step
//!
//! Prep-step pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane)
//!   - prep math + recurrence math
//!
//! Prep-chunk pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane), TOTAL — not per-t
//!   - T × recurrence math (q/k norm hoisted to the pre-pass, see above)
//!
//! State traffic per layer drops by `T`× at the dispatch boundary. For
//! Qwen3.6-A3B (Dk=256, Dv=128, Hv=16, B=1): state size = 16·128·256·4 B =
//! 2 MiB per direction. At T=512 the per-token loop did `T × (state R+W) = 2
//! GiB device traffic per layer per direction × 30 GDN layers = 120 GiB
//! per prefill step in state traffic alone. The chunked variant does
//! 2 MiB × 30 = 60 MiB.

use ffai_kernels::kernel;

#[kernel]
pub fn ffai_gated_delta_prep_chunk<T>(
    conv_out: Tensor<T>, // [B, T, 2·Hk·Dk + Hv·Dv]  (v slab; q/k slabs unused here)
    a_log: Tensor<T>,    // [Hv]
    dt_bias: Tensor<T>,  // [Hv]
    a_raw: Tensor<T>,    // [B, T, Hv]
    b_raw: Tensor<T>,    // [B, T, Hv]
    q_normed: Tensor<T>, // [B, T, Hk, Dk]  from ffai_gated_delta_qknorm_prepass
    k_normed: Tensor<T>, // [B, T, Hk, Dk]  from ffai_gated_delta_qknorm_prepass
    state_in: Tensor<T>, // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,    // [B, T, Hv, Dv]
    t_len: Tensor<u32>,  // [1] scalar
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    // 4 SGs per TG pack 4 Dv slots (reference Metal GDN geometry).
    let sg = simd_group_id();
    let dk_idx = simd_lane;
    let dv_idx = tgid_x * 4u32 + sg;
    let n = tgid_y;
    // GQA decomposition.
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;
    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);
    let stride_b = 2u32 * hk * dk + hv * dv;
    // Partial last TG when Dv % 4 != 0: idle SGs must not touch state/y.
    if dv_idx < dv {
        // Per-layer constants (loaded once per active SG).
        let a_log_val = load(a_log[hv_idx]).cast::<f32>();
        let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
        let exp_a_log = exp(a_log_val);
        let state_base = n * dv * dk + dv_idx * dk;
        // ─── Load state into per-lane registers ONCE — persists across T.
        stack_alloc("state_reg", 8u32, "f32");
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let val = load(state_in[state_base + s_idx]).cast::<f32>();
            stack_store("state_reg", i, val);
        }
        // k_cache holds this token's k_normed (read once in Phase 1, reused in
        // Phase 2's rank-1 update without a second load).
        stack_alloc("k_cache", 8u32, "f32");
        // ─── Inner T-loop: recurrence per token ──────────────────────────
        for t in range(0u32, t_total, 1u32) {
            let bt = b * t_total + t;
            let conv_base = bt * stride_b;
            let v_off = conv_base + 2u32 * hk * dk + hv_idx * dv;
            let qk_off = (bt * hk + hk_idx) * dk;
            let gbeta_idx = bt * hv + hv_idx;
            // ─── g / beta ──────────────────────────────────────────────
            let a_raw_val = load(a_raw[gbeta_idx]).cast::<f32>();
            let b_raw_val = load(b_raw[gbeta_idx]).cast::<f32>();
            let pre_softplus = a_raw_val + dt_bias_val;
            let dt_val = log(exp(pre_softplus) + 1.0f32);
            let g_val = exp(0.0f32 - exp_a_log * dt_val);
            let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));
            // v: one read per Dv slot per token.
            let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();
            // ─── Phase 1: decay state + accumulate kv_mem; cache k ─────
            let mut kv_mem = 0.0f32;
            for i in range(0u32, n_per_t, 1u32) {
                let s_idx = n_per_t * dk_idx + i;
                let s_old = stack_load("state_reg", i);
                let s_decayed = s_old * g_val;
                stack_store("state_reg", i, s_decayed);
                let k_normed_val = load(k_normed[qk_off + s_idx]).cast::<f32>();
                stack_store("k_cache", i, k_normed_val);
                kv_mem = kv_mem + s_decayed * k_normed_val;
            }
            let kv_mem_sum = simd_sum(kv_mem);
            let delta = (v_val - kv_mem_sum) * beta_val;
            // ─── Phase 2: rank-1 update + output projection ───────────
            let mut out_acc = 0.0f32;
            for i in range(0u32, n_per_t, 1u32) {
                let s_idx = n_per_t * dk_idx + i;
                let s_decayed = stack_load("state_reg", i);
                let k_normed_val = stack_load("k_cache", i);
                let s_new = s_decayed + k_normed_val * delta;
                stack_store("state_reg", i, s_new);
                let q_normed_val = load(q_normed[qk_off + s_idx]).cast::<f32>();
                out_acc = out_acc + s_new * q_normed_val;
            }
            let out_sum = simd_sum(out_acc);
            // ─── Phase 3: lane 0 writes y[t, n, dv_idx] ───────────────
            if dk_idx == 0u32 {
                store(y[(bt * hv + hv_idx) * dv + dv_idx], out_sum.cast::<T>());
            }
        }
        // ─── Write final state ONCE at the end ──────────────────────────
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use ffai_kernels::core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    /// `cargo test -p ffai-kernels-std --lib --release -- kernels::ssm::gated_delta_prep_chunk::tests::dump --nocapture`
    #[test]
    fn dump() {
        use ffai_kernels::codegen::msl::MslGenerator;
        let mut k = ffai_gated_delta_prep_chunk::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

/// New-syntax correctness for the fused chunked GDN prep+recurrence kernel
/// (`ffai_gated_delta_prep_chunk`). Oracle is the per-token prep + sequential GDN
/// recurrence with state carried across the T-loop (state_out of token t is
/// state_in of token t+1) — the legacy `gated_delta_prep_step` oracle composed
/// over T tokens, which is exactly the recurrence the kernel runs register-
/// resident across its inner T-loop. Same un-clamped softplus as the kernel;
/// inputs are dtype-rounded.
///
/// Grid (Reduction, 4 SGs/TG): `grid_3d(ceil(dv/4), b*hv, 1, [128,1,1])`;
/// `t_len` is a runtime u32 scalar buffer.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_gated_delta_prep_chunk;
    use crate::utils::{pack_f32, unpack_f32};

    fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// Per-head RMSNorm of q/k — the same math
    /// `ffai_gated_delta_qknorm_prepass` computes, evaluated once per
    /// `(b, t, hk_idx)`. Returns dense `(q_normed, k_normed)` [B,T,Hk,Dk],
    /// mirroring what that kernel writes.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm(
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

    /// Per-token CPU GDN recurrence over a chunk of `t_total` tokens, with
    /// the state threaded across tokens. Returns `(y [B,T,Hv,Dv], state_out
    /// [B,Hv,Dv,Dk])`. `q_normed`/`k_normed` are dense `[B,T,Hk,Dk]` — the
    /// pre-pass's output, matching what the real kernel now consumes.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        conv_out: &[f32], // [B, T, 2·Hk·Dk + Hv·Dv]  (v slab only)
        a_log: &[f32],    // [Hv]
        dt_bias: &[f32],  // [Hv]
        a_raw: &[f32],    // [B, T, Hv]
        b_raw: &[f32],    // [B, T, Hv]
        q_normed: &[f32], // [B, T, Hk, Dk]
        k_normed: &[f32], // [B, T, Hk, Dk]
        state_in: &[f32], // [B, Hv, Dv, Dk]
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let stride_b = 2 * hk * dk + hv * dv;
        let hk_per_hv = hv / hk;
        let mut y = vec![0.0_f32; b * t_total * hv * dv];
        let mut state = state_in.to_vec(); // carried across the T-loop
        for batch in 0..b {
            for t in 0..t_total {
                let bt = batch * t_total + t;
                let conv_base = bt * stride_b;
                let v_base = conv_base + 2 * hk * dk;
                for hv_idx in 0..hv {
                    let n = batch * hv + hv_idx;
                    let hk_idx = hv_idx / hk_per_hv;
                    let qk_base = (bt * hk + hk_idx) * dk;
                    // g / beta.
                    let gbeta_idx = bt * hv + hv_idx;
                    let dt = softplus_unclamped(a_raw[gbeta_idx] + dt_bias[hv_idx]);
                    let g_val = (-a_log[hv_idx].exp() * dt).exp();
                    let beta_val = sigmoid(b_raw[gbeta_idx]);
                    for dv_idx in 0..dv {
                        let v_val = conv_out[v_base + hv_idx * dv + dv_idx];
                        let s_base = n * dv * dk + dv_idx * dk;
                        // Phase 1: decay + kv_mem (k normed per-dim).
                        let mut kv_mem = 0.0_f32;
                        let mut decayed = vec![0.0_f32; dk];
                        for d in 0..dk {
                            let s = state[s_base + d] * g_val;
                            decayed[d] = s;
                            kv_mem += s * k_normed[qk_base + d];
                        }
                        let delta = (v_val - kv_mem) * beta_val;
                        // Phase 2: rank-1 update + output projection.
                        let mut out = 0.0_f32;
                        for d in 0..dk {
                            let s_new = decayed[d] + k_normed[qk_base + d] * delta;
                            state[s_base + d] = s_new;
                            out += s_new * q_normed[qk_base + d];
                        }
                        y[(bt * hv + hv_idx) * dv + dv_idx] = out;
                    }
                }
            }
        }
        (y, state)
    }

    /// Small fused chunked GDN-prep shape: dk a multiple of 32, Hv divisible by
    /// Hk; `t_total` tokens with state carryover.
    ///
    /// `conv_scale` / `state_scale` / `a_log0` control the recurrence dynamics.
    /// The GDN recurrence amplifies state by ~Σ_t δ_t; with the larger Dk=64
    /// GQA reduction and 4 tokens, hot inputs drive y into the 10⁵ range where
    /// the per-step bf16 state store diverges from the f32 oracle by O(100) in
    /// absolute terms — far above any sane tol. Keeping conv/state small and
    /// `a_log ≤ -3` (so single-step gain is well under 1) bounds y to O(1) and
    /// the dtype-store error to well under tol. Production keeps this stable via
    /// learned `a_log`/`dt_bias`; the fixture mimics that conditioning.
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
        state_scale: f32,
        a_log0: f32,
        dt: DType,
    ) -> TestSetup {
        let n_total = b * hv;
        let stride_b = 2 * hk * dk + hv * dv;
        let conv_out: Vec<f32> =
            (0..b * t_total * stride_b).map(|i| ((i as f32) * 0.0131).sin() * conv_scale).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| a_log0 - (i as f32) * 0.05).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.5 + (i as f32) * 0.05).collect();
        let a_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.3 + (i as f32) * 0.01).collect();
        let b_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.2 + (i as f32) * 0.008).collect();
        let q_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect();
        let k_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.0073).cos() * state_scale).collect();

        // Dtype-round inputs so the oracle sees the GPU's load precision.
        // q_normed/k_normed are additionally dtype-rounded a *second* time
        // here (post-qk_norm) because in production they round-trip through
        // a real Tensor<T> buffer written by ffai_gated_delta_qknorm_prepass —
        // this kernel never sees f32 q/k norm results.
        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (q_normed, k_normed) = qk_norm(
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
        let (q_normed, k_normed) = (r(&q_normed), r(&k_normed));
        let (y_exp, state_exp) = oracle(
            &r(&conv_out),
            &r(&a_log),
            &r(&dt_bias),
            &r(&a_raw),
            &r(&b_raw),
            &q_normed,
            &k_normed,
            &r(&state_in),
            b,
            t_total,
            hv,
            hk,
            dv,
            dk,
        );

        TestSetup::new(ffai_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("conv_out", pack_f32(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_normed", pack_f32(&q_normed, dt), dt))
            .input(TestBuffer::from_vec("k_normed", pack_f32(&k_normed, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", b * t_total * hv * dv, dt))
            .input(TestBuffer::from_vec(
                "t_len",
                (t_total as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
    }

    // GQA (Hv = 2·Hk), T=4 tokens with state carryover, weighted RMSNorm.
    // Dk=64 (longer reduction) + 4-token recurrence is highly gain-sensitive,
    // so the inputs are kept small (conv 0.02 / state 0.01 / a_log -3.0) to
    // bound y to O(1) — see `setup` doc. This keeps the dtype-store error well
    // inside tol across f32/f16/bf16 while still exercising GQA head-sharing.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_ffai_gated_delta_prep_chunk_gqa(dt: DType) -> TestSetup {
        setup(1, 4, 4, 2, 8, 64, 0.3, 0.02, 0.01, -3.0, dt)
    }

    // Hv == Hk (no key-sharing) at minimum dk=32, T=3 tokens.
    //
    // The f32 tol is 1e-2 (vs 5e-3 on the GQA sibling) because this
    // variant uses `a_log=-1.5` (a 4× weaker decay than the GQA setup's
    // -3.0), which amplifies y to magnitude ~24K. At that magnitude the
    // GQA tol of 5e-3 corresponds to <1 ULP, leaving no headroom for the
    // ~2-ULP rounding difference between HIP's OCML `expf`/`logf` and the
    // Rust f32 oracle's libm. Vulkan's GLSL exp/log happens to round
    // close enough to libm to clear 5e-3 — HIP doesn't. 1e-2 = ~3 ULPs at
    // peak magnitude, which both backends clear and which is still tight
    // for a 3-token gain-sensitive recurrence.
    //
    // bf16 tol bumped 2e-1 → 3.0 for the same reason, one layer up: since
    // the q/k RMSNorm pre-pass split, q_normed/k_normed round-trip through
    // a real Tensor<T> buffer between the two kernels instead of staying
    // in f32 registers for the whole prep+recurrence — an extra bf16
    // quantization step this specific fixture's amplification (~24K peak)
    // makes visible (observed max abs err ~2.0, still <1e-4 relative).
    // Every other bf16 case here (including both pre-pass tests and the
    // GQA sibling) is unaffected — FFAI itself never dispatches this pair
    // at bf16 (Qwen3.6 GDN prep always runs the f32 shadow path), so this
    // is a synthetic-fixture-only concern, not a production one.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 3.0])]
    fn test_ffai_gated_delta_prep_chunk_no_gqa(dt: DType) -> TestSetup {
        setup(1, 3, 4, 4, 4, 32, 1.0, 0.4, 0.1, -1.5, dt)
    }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_gated_delta_prep_chunk;

    // Grid `[ceil(dv/4), b*hv, 1]`, TG `[128,1,1]` (4 SGs). Production
    // Qwen3.6 shape: Dv=128, Hv=16, Dk=128/256, T chunk 64+.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 64usize, 16usize, 8usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(ffai_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Qwen3.5/3.6 GDN decode-step fusion (F-85/qwen35-port round-11, the
//! "fusion round" — see the round-11 brief).
//!
//! `wh-butter-models::qwen35::gdn_layer_full`'s per-layer GDN chain was 15
//! device dispatches (conv1d+silu, 3x slice, 2x rms_norm(q/k), 2x gemv
//! (alpha/beta), gdn_gate_beta, 2x gather (GQA cyclic expand),
//! gated_delta_step, rms_norm(ssm_norm), silu(z), mul) — each dispatch on
//! GB10 costs both launch latency AND an intermediate-tensor HBM/L2
//! round-trip for what are mostly tiny (`Hv`- or `Dk`-wide) buffers.
//!
//! This file adds TWO fused kernels that collapse that chain to 2
//! dispatches for everything downstream of the alpha/beta GEMVs and the
//! conv1d+slice:
//!
//!   - [`iron_gdn_decode_fused`]: extends `iron_gated_delta_step` (see
//!     `gated_delta.rs`) to also do q/k RMSNorm (weightless-scaled) and
//!     gate/beta computation IN-KERNEL, reading q/k directly from their
//!     PRE-norm `[Hk,Dk]` conv1d-split slices with the CYCLIC GQA mapping
//!     (`hk_idx = hv_idx % Hk`) baked in directly — replacing 2x
//!     `rms_norm` + `gdn_gate_beta` + 2x `gather` (5 dispatches) with 0
//!     extra dispatches (the delta-step kernel was already going to run).
//!   - [`iron_gdn_tail_gate`]: fuses `rms_norm(y, ssm_norm_w)` +
//!     `silu(z)` + `mul` (gate-AFTER-norm, `Qwen3NextRMSNormGated`
//!     convention — see `qwen35.rs`'s module doc for why this is NOT the
//!     already-wired `gated_group_rmsnorm` gate-BEFORE-norm kernel) into
//!     ONE dispatch.
//!
//! ## Numerical-match rationale (why this isn't a new source of drift)
//!
//! Both fusions reuse the EXACT per-lane element-ownership + reduction
//! shape the kernels they replace already use:
//!
//!   - `iron_gated_delta_step`'s own state loop already owns `n_per_t =
//!     dk/32` CONTIGUOUS elements per lane (`s_idx = n_per_t*dk_idx + i`)
//!     and reduces via `simd_sum` at TPG=32. `iron_rms_norm`'s fast path
//!     (the kernel `ops::rms_norm` dispatches for `Dk=128`, satisfying its
//!     `n%128==0` fast-path gate) owns the SAME contiguous 4-element slice
//!     per lane (`col = tid*4`) and reduces via `reduce_sum` at TPG=32 (=
//!     `dk/4` for `Dk=128`). Both `simd_sum` and `reduce_sum` lower to
//!     `ReduceKind::Sum` at the IR level and the CUDA backend
//!     (`wh-iron-codegen/src/cuda/mod.rs::emit_reduce_tree`) emits the SAME
//!     `__shfl_down_sync` warp-reduce tree for both when TPG==32 (the
//!     two-level cross-warp combine only triggers for TPG>32) — so
//!     `iron_gdn_decode_fused`'s in-kernel q/k RMSNorm is computed by
//!     the IDENTICAL instruction sequence `ops::rms_norm` used, just
//!     inlined instead of dispatched separately.
//!   - `iron_gdn_tail_gate` is a direct copy of `iron_rms_norm`'s body
//!     (same `col = tid*4`, same `reduce_sum`, same `rsqrt(ssq/n+eps)`)
//!     with the final store extended to multiply by `silu(z)` — the
//!     RMSNorm math is untouched, only the epilogue changes.
//!
//! The one deliberate, DOCUMENTED numerical change: q/k RMSNorm and
//! gate/beta are now recomputed redundantly across all `Dv` (128)
//! threadgroups sharing an `(b, hv_idx)` in `iron_gdn_decode_fused` (same
//! redundancy `iron_gated_delta_qknorm_prepass`'s doc identifies for the
//! chunk kernel) instead of being computed once and read back — this is a
//! REDUNDANT-RECOMPUTE, not an approximation: every replica computes the
//! exact same formula over the exact same inputs, so it's launch-latency
//! and HBM-traffic trade against ALU (which is not the bottleneck for
//! these tiny `Dk`/`Hv`-wide ops), not a numerical change. Note this is
//! the OPPOSITE-direction lever from `iron_gated_delta_qknorm_prepass`
//! (which HOISTS q/k-norm OUT of the chunk kernel to avoid this same
//! redundancy during PREFILL, where large `T` makes the redundant-compute
//! cost dominate); for decode (`T=1`), dispatch-launch overhead dominates
//! instead, so fusing IN is the correct direction — see that kernel's doc
//! for the prefill-side rationale this deliberately does not re-litigate.

use wh_iron::kernel;

/// Fused Qwen3.5/3.6 GDN decode step: q/k RMSNorm (weightless-scaled) +
/// gate/beta + the gated-delta-net recurrence, in one dispatch. Same
/// dispatch geometry as `iron_gated_delta_step` (`gated_delta.rs`) —
/// Grid `[Dv, B*Hv, 1]`, TG `[32,1,1]`, Reduction mode — so this is a
/// drop-in dispatch-site replacement, not a new grid shape.
///
/// Layouts:
///   q_raw, k_raw    : [B, Hk, Dk]   (post-conv1d+silu, PRE-norm — the
///                                    `conv1d_causal_silu_many` q/k slices)
///   v               : [B, Hv, Dv]   (post-conv1d+silu)
///   q_scale, k_scale: [Dk]          (weightless RMSNorm scale, constant
///                                    per-element — `invScale^2`/`invScale`)
///   a_raw, b_raw    : [B, Hv]       (raw alpha/beta gemv output)
///   dt_bias, neg_exp_a_log: [Hv]
///   state_in/out    : [B, Hv, Dv, Dk]  (state_out aliased == state_in,
///                                       same discipline as the kernel
///                                       this replaces)
///   y               : [B, Hv, Dv]
///
/// `hk_idx` uses the CYCLIC GQA mapping (`hv_idx % Hk`) directly — this
/// checkpoint's validated mapping (see `qwen35.rs`'s
/// `gdn_gqa_cyclic_idx` doc for why NOT the kernel-default contiguous-block
/// mapping) — so callers pass q_raw/k_raw at their NATIVE `[Hk,Dk]` shape;
/// no pre-expand gather needed.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn iron_gdn_decode_fused<T>(
    q_raw: Tensor<T>,
    k_raw: Tensor<T>,
    v: Tensor<T>,
    q_scale: Tensor<T>,
    k_scale: Tensor<T>,
    a_raw: Tensor<T>,
    b_raw: Tensor<T>,
    dt_bias: Tensor<T>,
    neg_exp_a_log: Tensor<T>,
    state_in: Tensor<T>,
    mut state_out: Tensor<T>,
    mut y: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y; // n = b*Hv + hv_idx
    let dk_idx = tid;
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    // CYCLIC GQA mapping baked in directly (validated mapping for this
    // checkpoint — see module doc). NOT `hv_idx / (hv/hk)`.
    let hk_idx = hv_idx - (hv_idx / hk) * hk;
    let n_per_t = dk / 32u32;
    let qk_base = (b * hk + hk_idx) * dk;

    // ─── q/k RMSNorm (weightless-scaled), computed in-kernel ───────────
    //
    // Same contiguous-4-per-lane ownership + simd_sum reduction
    // `ops::rms_norm`'s fast path uses at Dk=128 (see module doc) — this
    // is the SAME reduction, just inlined instead of a separate dispatch.
    stack_alloc("qn", 8u32, "f32");
    stack_alloc("kn", 8u32, "f32");
    let mut q_ssq = 0.0f32;
    let mut k_ssq = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qv = load(q_raw[qk_base + s_idx]).cast::<f32>();
        let kv = load(k_raw[qk_base + s_idx]).cast::<f32>();
        stack_store("qn", i, qv);
        stack_store("kn", i, kv);
        q_ssq = q_ssq + qv * qv;
        k_ssq = k_ssq + kv * kv;
    }
    let q_tot = simd_sum(q_ssq);
    let k_tot = simd_sum(k_ssq);
    let eps = load(eps_buf[0]);
    let q_rms = rsqrt(q_tot / dk + eps);
    let k_rms = rsqrt(k_tot / dk + eps);
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qs = load(q_scale[s_idx]).cast::<f32>();
        let ks = load(k_scale[s_idx]).cast::<f32>();
        let qn_v = stack_load("qn", i) * q_rms * qs;
        let kn_v = stack_load("kn", i) * k_rms * ks;
        stack_store("qn", i, qn_v);
        stack_store("kn", i, kn_v);
    }

    // ─── gate/beta, computed in-kernel (matches `iron_gdn_gate_beta`) ───
    let a = load(a_raw[n]).cast::<f32>();
    let braw = load(b_raw[n]).cast::<f32>();
    let bias = load(dt_bias[hv_idx]).cast::<f32>();
    let neg_exp_a = load(neg_exp_a_log[hv_idx]).cast::<f32>();
    let pre_softplus = a + bias;
    let dt_val = log(exp(pre_softplus) + 1.0f32);
    let g_val = exp(neg_exp_a * dt_val);
    let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - braw));

    let v_val = load(v[n * dv + dv_idx]).cast::<f32>();
    let state_base = n * dv * dk + dv_idx * dk;

    // ─── Phase 1: decay + kv_mem reduction (identical to
    // `iron_gated_delta_step`, reading normalized q/k from registers) ───
    stack_alloc("decayed", 8u32, "f32");
    let mut kv_mem = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = load(state_in[state_base + s_idx]).cast::<f32>() * g_val;
        stack_store("decayed", i, s_decayed);
        let k_val = stack_load("kn", i);
        kv_mem = kv_mem + s_decayed * k_val;
    }
    let kv_mem_sum = simd_sum(kv_mem);
    let delta = (v_val - kv_mem_sum) * beta_val;

    // ─── Phase 2: rank-1 update + output projection ─────────────────────
    let mut out = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = stack_load("decayed", i);
        let k_val = stack_load("kn", i);
        let s_new = s_decayed + k_val * delta;
        store(state_out[state_base + s_idx], s_new.cast::<T>());
        let q_val = stack_load("qn", i);
        out = out + s_new * q_val;
    }
    let out_sum = simd_sum(out);
    if dk_idx == 0u32 {
        store(y[n * dv + dv_idx], out_sum.cast::<T>());
    }
}

/// Fused GDN output tail: `out = rmsNorm(y, ssm_norm_w) * silu(z)`
/// (gate-AFTER-norm, `Qwen3NextRMSNormGated` convention — see
/// `qwen35.rs`'s module doc). Direct copy of `iron_rms_norm`'s body (same
/// `col = tid*4`, same `reduce_sum`, same `rsqrt(ssq/n+eps)` — see module
/// doc) with the epilogue extended to multiply by `silu(z)` instead of
/// just storing the normed value. Same dispatch geometry as
/// `ops::rms_norm`'s fast path: Grid `[rows,1,1]`, TG `[n/4,1,1]`.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn iron_gdn_tail_gate<T>(
    y: Tensor<T>,
    z: Tensor<T>,
    w: Tensor<T>,
    mut out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let col = tid * 4u32;
    let y0 = load(y[rs + col]).cast::<f32>();
    let y1 = load(y[rs + col + 1u32]).cast::<f32>();
    let y2 = load(y[rs + col + 2u32]).cast::<f32>();
    let y3 = load(y[rs + col + 3u32]).cast::<f32>();
    let ssq = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    let tot = reduce_sum(ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tot / n + eps);

    let w0 = load(w[col]).cast::<f32>();
    let w1 = load(w[col + 1u32]).cast::<f32>();
    let w2 = load(w[col + 2u32]).cast::<f32>();
    let w3 = load(w[col + 3u32]).cast::<f32>();

    let z0 = load(z[rs + col]).cast::<f32>();
    let z1 = load(z[rs + col + 1u32]).cast::<f32>();
    let z2 = load(z[rs + col + 2u32]).cast::<f32>();
    let z3 = load(z[rs + col + 3u32]).cast::<f32>();
    let silu0 = z0 * (1.0f32 / (1.0f32 + exp(0.0f32 - z0)));
    let silu1 = z1 * (1.0f32 / (1.0f32 + exp(0.0f32 - z1)));
    let silu2 = z2 * (1.0f32 / (1.0f32 + exp(0.0f32 - z2)));
    let silu3 = z3 * (1.0f32 / (1.0f32 + exp(0.0f32 - z3)));

    store(out[rs + col], (y0 * rms * w0 * silu0).cast::<T>());
    store(out[rs + col + 1u32], (y1 * rms * w1 * silu1).cast::<T>());
    store(out[rs + col + 2u32], (y2 * rms * w2 * silu2).cast::<T>());
    store(out[rs + col + 3u32], (y3 * rms * w3 * silu3).cast::<T>());
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{iron_gdn_decode_fused, iron_gdn_tail_gate};
    use crate::utils::{pack_f32, unpack_f32};

    fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    fn silu(x: f32) -> f32 { x * sigmoid(x) }
    fn rms_norm_row(row: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = row.len() as f32;
        let ssq: f32 = row.iter().map(|v| v * v).sum();
        let rms = 1.0 / (ssq / n + eps).sqrt();
        row.iter().zip(w.iter()).map(|(v, wv)| v * rms * wv).collect()
    }

    /// CPU oracle for `iron_gdn_decode_fused`: q/k RMSNorm (weightless,
    /// constant scale) + gate/beta + the same delta-rule recurrence
    /// `iron_gated_delta_step`'s oracle uses (see `gated_delta.rs`), with
    /// the CYCLIC GQA mapping baked in.
    #[allow(clippy::too_many_arguments)]
    fn decode_fused_oracle(
        q_raw: &[f32],
        k_raw: &[f32],
        v: &[f32],
        q_scale: &[f32],
        k_scale: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        dt_bias: &[f32],
        neg_exp_a_log: &[f32],
        state_in: &[f32],
        eps: f32,
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; b * hv * dv];
        let mut state_out = vec![0.0_f32; b * hv * dv * dk];
        for batch in 0..b {
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx % hk; // CYCLIC
                let qk_base = (batch * hk + hk_idx) * dk;
                let q_row = rms_norm_row(&q_raw[qk_base..qk_base + dk], q_scale, eps);
                let k_row = rms_norm_row(&k_raw[qk_base..qk_base + dk], k_scale, eps);
                let a = a_raw[n];
                let braw = b_raw[n];
                let dt = softplus_unclamped(a + dt_bias[hv_idx]);
                let g_val = (neg_exp_a_log[hv_idx] * dt).exp();
                let beta_val = sigmoid(braw);
                for dv_idx in 0..dv {
                    let v_val = v[n * dv + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;
                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    for s in 0..dk {
                        let sv = state_in[s_base + s] * g_val;
                        decayed[s] = sv;
                        kv_mem += sv * k_row[s];
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s in 0..dk {
                        let sn = decayed[s] + k_row[s] * delta;
                        state_out[s_base + s] = sn;
                        out += sn * q_row[s];
                    }
                    y[n * dv + dv_idx] = out;
                }
            }
        }
        (y, state_out)
    }

    fn setup(b: usize, hv: usize, hk: usize, dv: usize, dk: usize, dt: DType) -> TestSetup {
        let eps = 1e-6f32;
        let q_raw: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
        let k_raw: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect();
        let v: Vec<f32> = (0..b * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let inv_scale = 1.0 / (dk as f32).sqrt();
        let q_scale = vec![inv_scale * inv_scale; dk];
        let k_scale = vec![inv_scale; dk];
        let a_raw: Vec<f32> = (0..b * hv).map(|i| (i % 23) as f32 * 0.03 - 0.35).collect();
        let b_raw: Vec<f32> = (0..b * hv).map(|i| (i % 19) as f32 * 0.025 - 0.2).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.4 + (i as f32) * 0.04).collect();
        let neg_exp_a_log: Vec<f32> =
            (0..hv).map(|i| -((-4.0 + (i as f32) * 0.3f32).exp())).collect();
        let state_in: Vec<f32> =
            (0..b * hv * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

        let round = |raw: &[f32]| unpack_f32(&pack_f32(raw, dt), dt);
        let q_raw = round(&q_raw);
        let k_raw = round(&k_raw);
        let v = round(&v);
        let q_scale = round(&q_scale);
        let k_scale = round(&k_scale);
        let a_raw = round(&a_raw);
        let b_raw = round(&b_raw);
        let dt_bias = round(&dt_bias);
        let neg_exp_a_log = round(&neg_exp_a_log);
        let state_in = round(&state_in);

        let (y_exp, state_exp) = decode_fused_oracle(
            &q_raw,
            &k_raw,
            &v,
            &q_scale,
            &k_scale,
            &a_raw,
            &b_raw,
            &dt_bias,
            &neg_exp_a_log,
            &state_in,
            eps,
            b,
            hv,
            hk,
            dv,
            dk,
        );

        TestSetup::new(iron_gdn_decode_fused::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q_raw", pack_f32(&q_raw, dt), dt))
            .input(TestBuffer::from_vec("k_raw", pack_f32(&k_raw, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec("q_scale", pack_f32(&q_scale, dt), dt))
            .input(TestBuffer::from_vec("k_scale", pack_f32(&k_scale, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("neg_exp_a_log", pack_f32(&neg_exp_a_log, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", b * hv * dv, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d(dv as u32, (b * hv) as u32, 1, [32, 1, 1])
    }

    // GQA (Hv = 2*Hk) single step, f32 only (this port keeps GDN F32-
    // resident end to end — see `qwen35.rs`'s module doc).
    #[test_kernel(dtypes = [f32], tol = [5e-5])]
    fn test_iron_gdn_decode_fused_gqa(dt: DType) -> TestSetup { setup(2, 4, 2, 8, 64, dt) }

    // Qwen3.6-27B-class shape: Hk=16, Hv=48 (cyclic GQA, 3 Hv-heads/Hk
    // group), Dk=128, Dv=128, B=1.
    #[test_kernel(dtypes = [f32], tol = [5e-5])]
    fn test_iron_gdn_decode_fused_qwen36_shape(dt: DType) -> TestSetup {
        setup(1, 48, 16, 128, 128, dt)
    }

    // NOTE: the 3-sequential-decode-step state-carry check (the correctness
    // bar the round-11 brief specifically calls out for any new GDN kernel)
    // lives at the `wh-butter-cuda` integration-test layer instead of here
    // — `qwen35_units.rs::qwen35_gdn_decode_fused_matches_old_composition_3steps`
    // — because that layer already has the real `Device`/`Tensor`/dispatch
    // stack wired (see `qwen35_gdn_decode_composition_matches_host_oracle`,
    // the pre-existing sibling test this one is modeled on) rather than
    // inventing a second manual-dispatch harness here for the same purpose.

    fn tail_gate_oracle(
        y: &[f32],
        z: &[f32],
        w: &[f32],
        eps: f32,
        rows: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for r in 0..rows {
            let row = &y[r * n..(r + 1) * n];
            let normed = rms_norm_row(row, w, eps);
            for c in 0..n {
                out[r * n + c] = normed[c] * silu(z[r * n + c]);
            }
        }
        out
    }

    fn tail_setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        let eps = 1e-6f32;
        let y: Vec<f32> = (0..rows * n).map(|i| ((i as f32) * 0.019).sin() * 0.4).collect();
        let z: Vec<f32> = (0..rows * n).map(|i| ((i as f32) * 0.023).cos() * 0.6).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32) * 0.001).collect();
        let round = |raw: &[f32]| unpack_f32(&pack_f32(raw, dt), dt);
        let y = round(&y);
        let z = round(&z);
        let w = round(&w);
        let out_exp = tail_gate_oracle(&y, &z, &w, eps, rows, n);
        TestSetup::new(iron_gdn_tail_gate::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("y", pack_f32(&y, dt), dt))
            .input(TestBuffer::from_vec("z", pack_f32(&z, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&out_exp, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
    }

    // Qwen3.6-27B GDN shape: Hv=48 rows, Dv=128.
    #[test_kernel(dtypes = [f32], tol = [5e-5])]
    fn test_iron_gdn_tail_gate_qwen36_shape(dt: DType) -> TestSetup { tail_setup(48, 128, dt) }

    // Small GQA-scale shape for a second, cheaper geometry.
    #[test_kernel(dtypes = [f32], tol = [5e-5])]
    fn test_iron_gdn_tail_gate_small(dt: DType) -> TestSetup { tail_setup(4, 128, dt) }
}

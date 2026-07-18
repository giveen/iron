//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end GPU integration test for the two-kernel chunked-WY GDN
//! pipeline: `ffai_gdn_wy_plan` (chunk-parallel) -> `ffai_gdn_wy_scan`
//! (sequential state scan).
//!
//! Two independent checks:
//!
//! 1. **Pipeline vs. CPU oracle** at a production-ish shape (`Dv=Dk=128`,
//!    the shape the monolithic `ffai_gated_delta_wy_chunk` kernel cannot
//!    dispatch at all — see that kernel's module doc). The oracle is
//!    `sequential_gdn`, the same per-token recurrence
//!    `tests/gated_delta_wy_cpu_oracle.rs` and `gated_delta_wy.rs`'s own
//!    `kernel_tests` validate against (duplicated here in miniature — each
//!    `tests/*.rs` file is its own crate, so there is no cross-file `pub`
//!    import between integration tests; this keeps the file self-contained
//!    the way the existing test files already do).
//! 2. **Pipeline vs. the existing monolithic kernel**, at a small shape
//!    both can dispatch — the strongest form of "matches the EXISTING
//!    gated_delta_wy" since it compares GPU output to GPU output, not just
//!    to a CPU reference.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::ssm::{
    gated_delta_wy::ffai_gated_delta_wy_chunk,
    gated_delta_wy_plan::ffai_gdn_wy_plan,
    gated_delta_wy_scan::ffai_gdn_wy_scan,
};

/// Per-token GDN recurrence (mirrors `sequential_gdn` in
/// `tests/gated_delta_wy_cpu_oracle.rs` / `gated_delta_wy.rs`'s
/// `kernel_tests`). B=1. State layout `[hv, dv, dk]`, modified in place.
#[allow(clippy::too_many_arguments)]
fn sequential_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t_total: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let hv_per_hk = hv / hk;
    let mut y = vec![0.0_f32; t_total * hv * dv];
    for t in 0..t_total {
        for h_v in 0..hv {
            let h_k = h_v / hv_per_hk;
            let gt = g[t * hv + h_v];
            let bt = beta[t * hv + h_v];
            for d_v in 0..dv {
                let v_val = v[(t * hv + h_v) * dv + d_v];
                let s_base = (h_v * dv + d_v) * dk;
                let mut kv_mem = 0.0_f32;
                let mut decayed = vec![0.0_f32; dk];
                for s_idx in 0..dk {
                    let s = state[s_base + s_idx] * gt;
                    decayed[s_idx] = s;
                    kv_mem += s * k[(t * hk + h_k) * dk + s_idx];
                }
                let delta = (v_val - kv_mem) * bt;
                let mut out = 0.0_f32;
                for s_idx in 0..dk {
                    let s_new = decayed[s_idx] + k[(t * hk + h_k) * dk + s_idx] * delta;
                    state[s_base + s_idx] = s_new;
                    out += s_new * q[(t * hk + h_k) * dk + s_idx];
                }
                y[(t * hv + h_v) * dv + d_v] = out;
            }
        }
    }
    y
}

/// Deterministic synthetic inputs, B=1. `kscale` keeps `‖k‖² ≈ 1` so the
/// recurrence stays well-conditioned (same rationale as
/// `tests/gated_delta_wy_cpu_oracle.rs::synthetic_inputs`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn synthetic_inputs(
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let kscale = (2.0_f32 / dk as f32).sqrt();
    let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
    let v: Vec<f32> = (0..t * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..t * hv).map(|i| 0.8 + 0.15 * ((i as f32) * 0.013).sin()).collect();
    let beta: Vec<f32> = (0..t * hv).map(|i| 0.4 + 0.3 * ((i as f32) * 0.017).cos()).collect();
    let state: Vec<f32> = (0..hv * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
    (q, k, v, g, beta, state)
}

/// Runs `ffai_gdn_wy_plan` then `ffai_gdn_wy_scan` on the given inputs,
/// returning `(y [t*hv*dv], state_out [hv*dv*dk])`. B=1.
#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    ctx: &Context,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    c: usize,
    dt: Dt,
) -> (Vec<f32>, Vec<f32>) {
    let n_total = hv; // B=1
    let nc = t / c;
    let dtype = dt.to_dtype();

    // ── Pass 1: ffai_gdn_wy_plan ────────────────────────────────────────
    let mut plan_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    plan_buffers.insert("q".into(), pack_bytes(q, dt));
    plan_buffers.insert("k".into(), pack_bytes(k, dt));
    plan_buffers.insert("v".into(), pack_bytes(v, dt));
    plan_buffers.insert("g".into(), pack_bytes(g, dt));
    plan_buffers.insert("beta".into(), pack_bytes(beta, dt));
    plan_buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());
    plan_buffers.insert("u".into(), pack_bytes(&vec![0.0_f32; n_total * nc * c * dv], dt));
    plan_buffers.insert("y_local".into(), pack_bytes(&vec![0.0_f32; n_total * nc * c * dv], dt));
    plan_buffers.insert("q_eff".into(), pack_bytes(&vec![0.0_f32; n_total * nc * c * dk], dt));
    plan_buffers.insert("p_s".into(), pack_bytes(&vec![0.0_f32; n_total * nc * dk * dk], dt));
    plan_buffers.insert("u_s".into(), pack_bytes(&vec![0.0_f32; n_total * nc * dv * dk], dt));
    plan_buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    plan_buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    plan_buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    plan_buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
    plan_buffers.insert("c".into(), (c as u32).to_le_bytes().to_vec());

    let mut plan_k = ffai_gdn_wy_plan::kernel_ir_for(dtype);
    plan_k.mode = KernelMode::Reduction;
    let plan_r = ctx
        .dispatch_with_grid(&plan_k, &plan_buffers, &BTreeMap::new(), [nc, n_total, 1], [512, 1, 1])
        .expect("ffai_gdn_wy_plan dispatch");

    // ── Pass 2: ffai_gdn_wy_scan ─────────────────────────────────────────
    let mut scan_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    scan_buffers.insert("q_eff".into(), plan_r.outputs.get("q_eff").unwrap().clone());
    scan_buffers.insert("y_local".into(), plan_r.outputs.get("y_local").unwrap().clone());
    scan_buffers.insert("p_s".into(), plan_r.outputs.get("p_s").unwrap().clone());
    scan_buffers.insert("u_s".into(), plan_r.outputs.get("u_s").unwrap().clone());
    scan_buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    scan_buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; hv * dv * dk], dt));
    scan_buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; t * hv * dv], dt));
    scan_buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());
    scan_buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    scan_buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    scan_buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    scan_buffers.insert("c".into(), (c as u32).to_le_bytes().to_vec());

    let mut scan_k = ffai_gdn_wy_scan::kernel_ir_for(dtype);
    scan_k.mode = KernelMode::Reduction;
    let scan_r = ctx
        .dispatch_with_grid(
            &scan_k,
            &scan_buffers,
            &BTreeMap::new(),
            [(dv as u32 / 32) as usize, n_total, 1],
            [128, 1, 1],
        )
        .expect("ffai_gdn_wy_scan dispatch");

    let y = unpack_bytes(scan_r.outputs.get("y").unwrap(), dt);
    let state_out = unpack_bytes(scan_r.outputs.get("state_out").unwrap(), dt);
    (y, state_out)
}

/// Production-ish shape (`Dv=Dk=128`, the exact shape the monolithic
/// `ffai_gated_delta_wy_chunk` kernel cannot dispatch — its `[Dv,Dk]`
/// TG-resident state would be 64 KiB there). Validates the two-kernel
/// pipeline against the CPU `sequential_gdn` oracle directly.
fn pipeline_matches_oracle(dt: Dt, tol: f32) {
    let _g = gpu_lock();
    let (t, hk, hv, dk, dv, c) = (256usize, 2usize, 4usize, 128usize, 128usize, 64usize);
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    // Dtype-round inputs so the oracle sees what the GPU loads.
    let r = |xs: &[f32]| xs.iter().map(|&x| dt.round(x)).collect::<Vec<f32>>();
    let (qr, kr, vr, gr, br, sr) = (r(&q), r(&k), r(&v), r(&g), r(&beta), r(&state));

    let mut state_seq = sr.clone();
    let y_exp = sequential_gdn(&qr, &kr, &vr, &gr, &br, &mut state_seq, t, hk, hv, dk, dv);

    let ctx = Context::new().expect("Context::new");
    let (y_got, state_got) =
        run_pipeline(&ctx, &q, &k, &v, &g, &beta, &state, t, hk, hv, dk, dv, c, dt);

    let dy = max_abs_diff(&y_exp, &y_got);
    let ds = max_abs_diff(&state_seq, &state_got);
    eprintln!("[{dt:?}] pipeline-vs-oracle: y diff={dy:.3e} state diff={ds:.3e}");
    assert!(dy < tol, "y diff {dy:.3e} exceeds tol {tol:.3e}");
    assert!(ds < tol, "state diff {ds:.3e} exceeds tol {tol:.3e}");
}

#[test]
fn gdn_wy_pipeline_matches_oracle_f32() { pipeline_matches_oracle(Dt::F32, 5e-3); }

#[test]
fn gdn_wy_pipeline_matches_oracle_bf16() { pipeline_matches_oracle(Dt::Bf16, 2e-2); }

/// Cross-checks the two-kernel pipeline against the EXISTING monolithic
/// `ffai_gated_delta_wy_chunk` kernel, GPU output vs. GPU output, at a
/// small shape both can dispatch (Dk=32, Dv=16 — well under the
/// monolithic kernel's ~32 KiB TG state-buffer limit).
#[test]
fn gdn_wy_pipeline_matches_monolithic_kernel() {
    let _g = gpu_lock();
    let dt = Dt::F32;
    let dtype = dt.to_dtype();
    // Dv must be a multiple of 32 for `ffai_gdn_wy_scan`'s tile dispatch
    // (see its DISPATCH INVARIANTS); Dk=Dv=32 comfortably fits the
    // monolithic kernel's TG budget too.
    let (t, hk, hv, dk, dv, c) = (32usize, 2usize, 4usize, 32usize, 32usize, 8usize);
    let n_total = hv; // B=1
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    let ctx = Context::new().expect("Context::new");

    // ── Monolithic kernel ────────────────────────────────────────────
    let mut mono_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    mono_buffers.insert("q".into(), pack_bytes(&q, dt));
    mono_buffers.insert("k".into(), pack_bytes(&k, dt));
    mono_buffers.insert("v".into(), pack_bytes(&v, dt));
    mono_buffers.insert("g".into(), pack_bytes(&g, dt));
    mono_buffers.insert("beta".into(), pack_bytes(&beta, dt));
    mono_buffers.insert("state_in".into(), pack_bytes(&state, dt));
    mono_buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; hv * dv * dk], dt));
    mono_buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; t * hv * dv], dt));
    mono_buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    mono_buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    mono_buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    mono_buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
    mono_buffers.insert("c".into(), (c as u32).to_le_bytes().to_vec());
    mono_buffers.insert("t_len".into(), (t as u32).to_le_bytes().to_vec());

    let mut mono_k = ffai_gated_delta_wy_chunk::kernel_ir_for(dtype);
    mono_k.mode = KernelMode::Reduction;
    let mono_r = ctx
        .dispatch_with_grid(&mono_k, &mono_buffers, &BTreeMap::new(), [1, n_total, 1], [32, 1, 1])
        .expect("ffai_gated_delta_wy_chunk dispatch");
    let y_mono = unpack_bytes(mono_r.outputs.get("y").unwrap(), dt);
    let state_mono = unpack_bytes(mono_r.outputs.get("state_out").unwrap(), dt);

    // ── Two-kernel pipeline, same inputs ─────────────────────────────
    let (y_pipe, state_pipe) =
        run_pipeline(&ctx, &q, &k, &v, &g, &beta, &state, t, hk, hv, dk, dv, c, dt);

    let dy = max_abs_diff(&y_mono, &y_pipe);
    let ds = max_abs_diff(&state_mono, &state_pipe);
    eprintln!("pipeline-vs-monolithic: y diff={dy:.3e} state diff={ds:.3e}");
    assert!(dy < 4e-2, "y diff {dy:.3e} vs monolithic kernel");
    assert!(ds < 4e-2, "state diff {ds:.3e} vs monolithic kernel");
}

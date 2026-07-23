//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Tight per-branch CPU oracle for `iron_batched_4_qmm_fast` (the M>1
//! sibling of `iron_batched_4_qgemv_fast`) at production-realistic
//! UNEQUAL output dims.
//!
//! Context: the in-source `#[test_kernel]` in `batched_4_qmm.rs` only
//! covers `out_a/out_b/out_c/out_d = 16/8/8/8` (nearly equal, tiny) at
//! `tol=2e-1` (loose), and the harness reports a single error aggregated
//! across all four `expect()` buffers (`max_abs_diff` folded over every
//! buffer, see `run_kernel_test` in `wh-iron/src/runner/harness.rs`).
//! A bug isolated to one branch (say a scale/bias group off-by-one on the
//! tiny C/D tail) could pass unnoticed behind a correct A/B branch and a
//! loose combined tolerance.
//!
//! This test closes that gap:
//!   1. Dims mirror the Qwen3.6-A3B GDN qkv/z/b/a input-projection shape
//!      (big/big/tiny/tiny: out_a=512, out_b=256, out_c=32, out_d=32,
//!      scaled down from the real 8192/4096/32/32 so the CPU oracle
//!      stays fast) at M=32 (M>1, a real prefill batch row count). This
//!      is the asymmetry (two big branches, two 32-wide tails) that was
//!      never exercised by the existing near-equal test.
//!   2. Each of the four outputs is computed by an INDEPENDENT
//!      single-projection dequant-affine-GEMM oracle in natural column
//!      order (0..in_dim sequential). The kernel accumulates in
//!      16-column-per-lane blocks reduced across a 32-lane SIMD group
//!      per 512-column K-tile: a genuinely different summation order.
//!      Agreement at a tight tolerance is therefore real evidence of
//!      correctness, not an artifact of matching the kernel's own
//!      reduction order.
//!   3. Each branch is compared SEPARATELY against a tight per-dtype
//!      tolerance (f32 1e-4, f16 5e-2, bf16 2e-1) so a per-branch bug
//!      cannot hide behind an aggregate max.
//!
//! Run:
//!   cargo test --release -p wh-iron-std --test batched_4_qmm_fast_oracle -- --nocapture

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::gemm::batched_4_qmm::iron_batched_4_qmm_fast;

// -- Deterministic xorshift source, same recipe as the in-source ---------
//    `#[test_kernel]` in batched_4_qmm.rs. Replicated for test-file
//    isolation per this crate's integration-test convention (see the
//    header comment in qmm_mma_dynamic_m_correctness.rs).
fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
        })
        .collect()
}

fn quantize_int4_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let in_dim = row.len();
    let n_groups = in_dim / group_size;
    let mut packed = vec![0u32; in_dim / 8];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let gs = &row[g * group_size..(g + 1) * group_size];
        let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / 15.0 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in gs.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
            let d = g * group_size + i;
            packed[d / 8] |= q << ((d % 8) * 4);
        }
    }
    (packed, scales, biases)
}

fn quantize_matrix(
    rows: &[f32],
    out_dim: usize,
    in_dim: usize,
    gs: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let (mut w, mut s, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for row in 0..out_dim {
        let (pw, ps, pb) = quantize_int4_row(&rows[row * in_dim..(row + 1) * in_dim], gs);
        w.extend(pw);
        s.extend(ps);
        b.extend(pb);
    }
    (w, s, b)
}

// ── Independent single-projection oracle. Natural column order, NOT the
//    kernel's per-16-column-lane + SIMD-group tree-reduction order, so a
//    pass here is real evidence of correctness rather than an artifact of
//    replaying the kernel's own accumulation order.
#[allow(clippy::too_many_arguments)]
fn naive_qmm(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    in_dim: usize,
    gs: usize,
    out_dim: usize,
) -> Vec<f32> {
    let u32_per_row = in_dim / 8;
    let n_groups = in_dim / gs;
    let mut out = vec![0.0_f32; m * out_dim];
    for mi in 0..m {
        let xr = &x[mi * in_dim..(mi + 1) * in_dim];
        for row in 0..out_dim {
            let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
            let rs = &scales[row * n_groups..(row + 1) * n_groups];
            let rb = &biases[row * n_groups..(row + 1) * n_groups];
            let mut acc = 0.0_f32;
            for d in 0..in_dim {
                let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                let g = d / gs;
                acc += (q as f32 * rs[g] + rb[g]) * xr[d];
            }
            out[mi * out_dim + row] = acc;
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

fn round_all(xs: &[f32], dt: Dt) -> Vec<f32> { xs.iter().map(|&v| dt.round(v)).collect() }

/// Dispatches the fused kernel ONCE (all four branches share one grid via
/// `program_id::<2>()` as the matrix-select dim), then compares each of
/// the four output blocks against its own independent oracle at a tight
/// per-dtype tolerance. Prints a per-branch error table so a failing
/// branch is immediately identifiable.
fn run_unequal_dims_case(dt: Dt, tol: f32) {
    let (m, in_dim, gs) = (32usize, 512usize, 64usize);
    // Big/big/tiny/tiny, mirrors the Qwen3.6-A3B GDN qkv/z/b/a shape
    // (8192/4096/32/32 scaled down to keep the CPU oracle fast). This is
    // deliberately UNEQUAL, unlike the in-source #[test_kernel]'s
    // near-equal 16/8/8/8. The asymmetry (two big branches feeding the
    // same `out_limit`-gated grid as two 32-wide tails) is exactly what
    // was never exercised before.
    let (out_a, out_b, out_c, out_d) = (512usize, 256usize, 32usize, 32usize);

    // Small-magnitude generation (mirrors `build_quant_inputs_small_mag`
    // in qmm_mma_dynamic_m_correctness.rs), keeps the Σ over in_dim=512
    // terms well inside bf16 range so a tight bf16 tolerance is
    // meaningful rather than swamped by legitimate quantization noise.
    let x_f32: Vec<f32> = round_all(&source(m * in_dim, 0x1001, 0.2, 0.02), dt);
    let wa = source(out_a * in_dim, 0x2002, 0.2, 0.0);
    let wb = source(out_b * in_dim, 0x3003, 0.2, 0.0);
    let wc = source(out_c * in_dim, 0x4004, 0.2, 0.0);
    let wd = source(out_d * in_dim, 0x5005, 0.2, 0.0);
    let (wa_p, sa, bias_a) = quantize_matrix(&wa, out_a, in_dim, gs);
    let (wb_p, sb, bias_b) = quantize_matrix(&wb, out_b, in_dim, gs);
    let (wc_p, sc, bias_c) = quantize_matrix(&wc, out_c, in_dim, gs);
    let (wd_p, sd, bias_d) = quantize_matrix(&wd, out_d, in_dim, gs);
    let sa_r = round_all(&sa, dt);
    let bias_a_r = round_all(&bias_a, dt);
    let sb_r = round_all(&sb, dt);
    let bias_b_r = round_all(&bias_b, dt);
    let sc_r = round_all(&sc, dt);
    let bias_c_r = round_all(&bias_c, dt);
    let sd_r = round_all(&sd, dt);
    let bias_d_r = round_all(&bias_d, dt);

    let ea = naive_qmm(&wa_p, &sa_r, &bias_a_r, &x_f32, m, in_dim, gs, out_a);
    let eb = naive_qmm(&wb_p, &sb_r, &bias_b_r, &x_f32, m, in_dim, gs, out_b);
    let ec = naive_qmm(&wc_p, &sc_r, &bias_c_r, &x_f32, m, in_dim, gs, out_c);
    let ed = naive_qmm(&wd_p, &sd_r, &bias_d_r, &x_f32, m, in_dim, gs, out_d);

    let dtype = dt.to_dtype();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x_f32, dt));
    buffers.insert("w_a".into(), pack_u32_bytes(&wa_p));
    buffers.insert("scales_a".into(), pack_bytes(&sa_r, dt));
    buffers.insert("biases_a".into(), pack_bytes(&bias_a_r, dt));
    buffers.insert("w_b".into(), pack_u32_bytes(&wb_p));
    buffers.insert("scales_b".into(), pack_bytes(&sb_r, dt));
    buffers.insert("biases_b".into(), pack_bytes(&bias_b_r, dt));
    buffers.insert("w_c".into(), pack_u32_bytes(&wc_p));
    buffers.insert("scales_c".into(), pack_bytes(&sc_r, dt));
    buffers.insert("biases_c".into(), pack_bytes(&bias_c_r, dt));
    buffers.insert("w_d".into(), pack_u32_bytes(&wd_p));
    buffers.insert("scales_d".into(), pack_bytes(&sd_r, dt));
    buffers.insert("biases_d".into(), pack_bytes(&bias_d_r, dt));
    buffers.insert("a_buf".into(), pack_bytes(&vec![0.0f32; m * out_a], dt));
    buffers.insert("b_buf".into(), pack_bytes(&vec![0.0f32; m * out_b], dt));
    buffers.insert("c_buf".into(), pack_bytes(&vec![0.0f32; m * out_c], dt));
    buffers.insert("d_buf".into(), pack_bytes(&vec![0.0f32; m * out_d], dt));
    buffers.insert("out_a".into(), (out_a as u32).to_le_bytes().to_vec());
    buffers.insert("out_b".into(), (out_b as u32).to_le_bytes().to_vec());
    buffers.insert("out_c".into(), (out_c as u32).to_le_bytes().to_vec());
    buffers.insert("out_d".into(), (out_d as u32).to_le_bytes().to_vec());
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (gs as u32).to_le_bytes().to_vec());

    let n_tgs = out_a.max(out_b).max(out_c).max(out_d).div_ceil(8);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let mut kernel = iron_batched_4_qmm_fast::kernel_ir_for(dtype);
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tgs, m, 4], [64, 1, 1])
        .expect("dispatch iron_batched_4_qmm_fast");

    let ga = unpack_bytes(result.outputs.get("a_buf").expect("a_buf"), dt);
    let gb = unpack_bytes(result.outputs.get("b_buf").expect("b_buf"), dt);
    let gc = unpack_bytes(result.outputs.get("c_buf").expect("c_buf"), dt);
    let gd = unpack_bytes(result.outputs.get("d_buf").expect("d_buf"), dt);

    let err_a = max_abs_diff(&ga, &ea);
    let err_b = max_abs_diff(&gb, &eb);
    let err_c = max_abs_diff(&gc, &ec);
    let err_d = max_abs_diff(&gd, &ed);

    println!(
        "[batched_4_qmm_fast {dt:?} M={m} in_dim={in_dim} out_a/b/c/d={out_a}/{out_b}/{out_c}/{out_d}] \
         A={err_a:.3e}  B={err_b:.3e}  C={err_c:.3e}  D={err_d:.3e}  tol={tol:.1e}"
    );

    assert!(err_a <= tol, "branch A (out={out_a}) max|Δ|={err_a:.3e} > tol {tol:.3e}");
    assert!(err_b <= tol, "branch B (out={out_b}) max|Δ|={err_b:.3e} > tol {tol:.3e}");
    assert!(err_c <= tol, "branch C tail (out={out_c}) max|Δ|={err_c:.3e} > tol {tol:.3e}");
    assert!(err_d <= tol, "branch D tail (out={out_d}) max|Δ|={err_d:.3e} > tol {tol:.3e}");
}

#[test]
fn batched_4_qmm_fast_unequal_dims_f32() { run_unequal_dims_case(Dt::F32, 1e-4); }

#[test]
fn batched_4_qmm_fast_unequal_dims_f16() { run_unequal_dims_case(Dt::F16, 5e-2); }

#[test]
fn batched_4_qmm_fast_unequal_dims_bf16() { run_unequal_dims_case(Dt::Bf16, 2e-1); }

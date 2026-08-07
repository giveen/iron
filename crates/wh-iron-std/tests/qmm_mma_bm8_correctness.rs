//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness oracle for `wh_iron_std::kernels::gemm::quantized::
//! iron_qmm_mma_bm8` — the quarter-height (BM=8) simdgroup-matrix MMA
//! kernel. Sized for T≈1-8 speculative-decode verify batches, where
//! `iron_qmm_bm4`'s compute-bound marginal cost loses to an MMA-class
//! kernel's bandwidth-bound marginal cost, but nothing today is sized to
//! M=8 (the existing `iron_qmm_mma`/`iron_qmm_mma_m16` require
//! `M % 32 == 0` / `M == 16`).
//!
//! Dispatches `iron_qmm_mma_bm8` directly (not through the
//! `quantized_mma_dynamic_m` module, which is hardwired to BM=32) via a
//! local pad-to-8 + dispatch-grid helper mirroring
//! `quantized_mma_dynamic_m::pad_t_to_bm` / `dispatch_grid` but tiled at
//! BM=8.
//!
//! Coverage matrix:
//!   - f16  T=8  (single BM=8 tile — the exact-fit verify-batch shape)
//!   - f16  T=16 (multi-tile, grid y=2 — exercises tiling across TGs)
//!   - bf16 T=8  (production dtype, single tile)
//!   - bf16 T=16 (production dtype, multi-tile)
//!   - f32  T=8  (reference-precision single tile)
//!
//! Run:
//!   cargo test --release -p wh-iron-std --test qmm_mma_bm8_correctness -- --nocapture

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use wh_iron::{Context, core::dtype::DType};
use wh_iron_std::kernels::gemm::quantized::iron_qmm_mma_bm8;

// ── Tile geometry — mirrors `quantized_mma_dynamic_m` but at BM=8 ────────
const BM_TILE: usize = 8;
const BN_TILE: usize = 32;
const TPG: [usize; 3] = [64, 1, 1];

/// Round `t` up to the next multiple of [`BM_TILE`] (8).
const fn pad_t_to_bm8(t: usize) -> usize { t.div_ceil(BM_TILE) * BM_TILE }

/// Pad an X buffer `[t, k]` to `[m_padded, k]` by appending zero rows.
fn pad_x_rows_bytes(x_bytes: &[u8], t: usize, k: usize, bytes_per_elem: usize) -> Vec<u8> {
    let m_padded = pad_t_to_bm8(t);
    let row_bytes = k * bytes_per_elem;
    assert_eq!(x_bytes.len(), t * row_bytes, "x_bytes must be t * k * bytes_per_elem");
    let mut out = Vec::with_capacity(m_padded * row_bytes);
    out.extend_from_slice(x_bytes);
    out.resize(m_padded * row_bytes, 0);
    out
}

/// Dispatch grid `[N/32, m_padded/8, 1]` for a logical token count `t`.
fn dispatch_grid_bm8(t: usize, n: usize) -> [usize; 3] {
    assert!(n.is_multiple_of(BN_TILE), "n must be multiple of {BN_TILE} (BN tile)");
    let m_padded = pad_t_to_bm8(t);
    [n / BN_TILE, m_padded / BM_TILE, 1]
}

// ── Triple-loop CPU oracle — same algorithm as ────────────────────────────
//    `qmm_mma_dynamic_m_correctness.rs::cpu_qmm_reference`, replicated for
//    test-file isolation per this crate's integration-test convention.

#[allow(clippy::too_many_arguments)]
fn cpu_qmm_reference(
    w: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for g in 0..gs_per_row {
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let mut q_dot = 0.0f32;
                let mut x_sum = 0.0f32;
                for p in 0..8usize {
                    let packed = w[n_col * k / 8 + g * 8 + p];
                    for bit in 0..8u32 {
                        let q = ((packed >> (bit * 4)) & 0xF) as f32;
                        let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                        q_dot += q * xv;
                        x_sum += xv;
                    }
                }
                acc += s * q_dot + bias * x_sum;
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

// ── Host-side dispatcher exercising `iron_qmm_mma_bm8` directly. ─────────
//
// Pads X (zero-fill) to m_padded = ceil(T/8)*8, dispatches
// `iron_qmm_mma_bm8` with grid `[N/32, m_padded/8, 1]`, then slices the
// first `T * N` element-bytes of the output.

#[allow(clippy::too_many_arguments)]
fn run_bm8(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    t: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    assert!(n.is_multiple_of(BN_TILE), "n must be multiple of 32 (BN tile)");
    assert!(k.is_multiple_of(32), "k must be multiple of 32 (BK step)");

    let m_padded = pad_t_to_bm8(t);
    let padded_x = pad_x_rows_bytes(x_bytes, t, k, out_bytes_per_elem);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), padded_x);
    buffers.insert("out".into(), vec![0u8; m_padded * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    // `Reduction` mode is required for the `tgid_x`/`tgid_y` aliases the
    // kernel body references — mirrors `iron_qmm_for` / `dyn_m::kernel_ir_for`.
    let mut kernel = iron_qmm_mma_bm8::kernel_ir_for(dtype);
    kernel.mode = wh_iron::core::ir::KernelMode::Reduction;
    let grid = dispatch_grid_bm8(t, n);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), grid, TPG)
        .expect("dispatch iron_qmm_mma_bm8");
    let out_padded = result.outputs.get("out").expect("`out` buffer").clone();
    out_padded[..(t * n * out_bytes_per_elem)].to_vec()
}

// ── Dtype byte helpers. ─────────────────────────────────────────────────

fn f32_to_f32_bytes(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect()
}
fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect()
}
fn round_f16(v: f32) -> f32 { half::f16::from_f32(v).to_f32() }
fn round_bf16(v: f32) -> f32 { half::bf16::from_f32(v).to_f32() }

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}

// Cosine similarity is scale-invariant: `cosine(v, k*v) == 1.0` for any
// positive scalar `k`, so a uniform-scale bug (wrong dequant scale, a
// dropped `group_size` factor, an off-by-one on the bias term applied
// uniformly, etc.) sails through a cosine-only check. `max_abs_diff`
// closes that gap — it's sensitive to exactly the class of bug cosine
// is blind to. Every case below asserts both.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// Calibrated 2026-08-06 on Metal (M5 Max). Output magnitudes here are large
// (up to ~1.1e4 for T=8, ~2.1e4 for T=16 — `build_quant_inputs` sums 512
// quantized columns per output element), so `max_abs_diff` needs to scale
// with that, not with a generic "small tensor" constant. Each bound is
// ~3x the max|Δ| observed across repeated runs (Metal dispatch here is
// deterministic — repeat runs reproduced the same value bit-for-bit, so
// the multiplier is headroom against a legitimate different-but-still-
// correct summation order on other hardware, not run-to-run noise).
// All bounds land far below what a uniform-scale bug would produce (a
// 2% scale error alone would add ~230-420 of error at these magnitudes),
// which is the actual point of adding them: cosine similarity is
// invariant to uniform scaling and would not have caught that class of
// bug (see the `max_abs_diff` doc comment above).
const CAL_F16_T8: f32 = 12.0; // observed 3.966e0
const CAL_F16_T16: f32 = 26.0; // observed 8.656e0
const CAL_BF16_T8: f32 = 105.0; // observed 3.475e1
const CAL_BF16_T16: f32 = 217.0; // observed 7.233e1
const CAL_F32_T8: f32 = 0.03; // observed 8.301e-3

// ── Deterministic q4 weights — same per-pack pattern as the sibling ──────
//    dynamic-M correctness test (`qmm_mma_dynamic_m_correctness.rs`).

fn build_quant_inputs(
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    (w, scales, biases, x)
}

// ═════════════════════════════════════════════════════════════════════════
// Case 1: f16 T=8 — single BM=8 tile, the exact-fit speculative-decode
// verify-batch shape this kernel targets.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn bm8_f16_t8_single_tile() {
    let t = 8usize;
    let n = 64usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_bm8(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=8 element count");
    let cos = cosine(&expected, &actual);
    let max_diff = max_abs_diff(&expected, &actual);
    println!("[f16 T=8 single-tile] cos={cos:.6} max|Δ|={max_diff:.3e}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=8)");
    assert!(max_diff <= CAL_F16_T8, "max|Δ| {max_diff:.3e} > {CAL_F16_T8:.3e} (f16 T=8)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 2: f16 T=16 — two BM=8 tiles (grid y=2). Exercises tiling across
// multiple threadgroups along the M axis.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn bm8_f16_t16_multi_tile() {
    let t = 16usize;
    let n = 64usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    assert_eq!(dispatch_grid_bm8(t, n), [2, 2, 1], "T=16 should dispatch grid y=2");
    let out_bytes = run_bm8(
        &ctx,
        DType::F16,
        &w,
        &f32_to_f16_bytes(&scales),
        &f32_to_f16_bytes(&biases),
        &f32_to_f16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=16 element count");
    let cos = cosine(&expected, &actual);
    let max_diff = max_abs_diff(&expected, &actual);
    println!("[f16 T=16 multi-tile] cos={cos:.6} max|Δ|={max_diff:.3e}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f16 T=16)");
    assert!(max_diff <= CAL_F16_T16, "max|Δ| {max_diff:.3e} > {CAL_F16_T16:.3e} (f16 T=16)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 3: bf16 T=8 — production dtype, single tile.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn bm8_bf16_t8_single_tile() {
    let t = 8usize;
    let n = 64usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_bm8(
        &ctx,
        DType::BF16,
        &w,
        &f32_to_bf16_bytes(&scales),
        &f32_to_bf16_bytes(&biases),
        &f32_to_bf16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=8 element count");
    let cos = cosine(&expected, &actual);
    let max_diff = max_abs_diff(&expected, &actual);
    println!("[bf16 T=8 single-tile] cos={cos:.6} max|Δ|={max_diff:.3e}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (bf16 T=8)");
    assert!(max_diff <= CAL_BF16_T8, "max|Δ| {max_diff:.3e} > {CAL_BF16_T8:.3e} (bf16 T=8)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 4: bf16 T=16 — production dtype, multi-tile (grid y=2).
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn bm8_bf16_t16_multi_tile() {
    let t = 16usize;
    let n = 64usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales_f32, biases_f32, x_f32) = build_quant_inputs(t, n, k, gs_per_row);
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_bm8(
        &ctx,
        DType::BF16,
        &w,
        &f32_to_bf16_bytes(&scales),
        &f32_to_bf16_bytes(&biases),
        &f32_to_bf16_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(actual.len(), expected.len(), "T=16 element count");
    let cos = cosine(&expected, &actual);
    let max_diff = max_abs_diff(&expected, &actual);
    println!("[bf16 T=16 multi-tile] cos={cos:.6} max|Δ|={max_diff:.3e}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (bf16 T=16)");
    assert!(max_diff <= CAL_BF16_T16, "max|Δ| {max_diff:.3e} > {CAL_BF16_T16:.3e} (bf16 T=16)");
}

// ═════════════════════════════════════════════════════════════════════════
// Case 5: f32 T=8 — reference precision, single tile. No dtype rounding
// error; sanity check that the kernel itself (not dtype narrowing) is
// numerically correct.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn bm8_f32_t8_reference() {
    let t = 8usize;
    let n = 64usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let (w, scales, biases, x) = build_quant_inputs(t, n, k, gs_per_row);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new");
    let out_bytes = run_bm8(
        &ctx,
        DType::F32,
        &w,
        &f32_to_f32_bytes(&scales),
        &f32_to_f32_bytes(&biases),
        &f32_to_f32_bytes(&x),
        t,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), expected.len(), "T=8 element count");
    let cos = cosine(&expected, &actual);
    let mut max_diff = 0.0f32;
    for (e, a) in expected.iter().zip(actual.iter()) {
        max_diff = max_diff.max((e - a).abs());
    }
    println!("[f32 T=8 ref] cos={cos:.6} max|Δ|={max_diff:.3e}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999 (f32 T=8)");
    assert!(max_diff <= CAL_F32_T8, "max|Δ| {max_diff:.3e} > {CAL_F32_T8:.3e} (f32 T=8)");
}

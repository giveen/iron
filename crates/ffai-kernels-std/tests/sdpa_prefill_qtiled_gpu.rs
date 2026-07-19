//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! End-to-end correctness for the Q-tiled prefill-append SDPA kernel
//! `ffai_sdpa_prefill_qtiled_d256` (single dispatch, no merge pass).
//!
//! Covers the append case (base_kv > 0, a query chunk on an existing KV
//! prefix), the first-chunk case (base_kv == 0), GQA, all three dtypes,
//! ragged tiles (n_query not a multiple of BQ=16), and non-causal mode.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use ffai_kernels::{Context, core::ir::KernelMode};
use ffai_kernels_std::kernels::sdpa::sdpa_prefill_qtiled_d256::ffai_sdpa_prefill_qtiled_d256;

/// Naive causal-append SDPA oracle. Q/out `[n_query, n_q_heads,
/// head_dim]`, K/V `[n_kv_heads, kv_stride, head_dim]`. Query row `r`
/// (absolute position `base_kv + r`) attends `[0, base_kv + r + 1)` when
/// causal, else `[0, base_kv + n_query)` for every row.
#[allow(clippy::too_many_arguments)]
fn naive_sdpa_append_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let gqa = n_q_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_query * n_q_heads * head_dim];
    for r in 0..n_query {
        let n_kv = if causal { base_kv + r + 1 } else { base_kv + n_query };
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = (r * n_q_heads + qh) * head_dim;
            let kv_slab = kvh * kv_stride * head_dim;
            let mut scores = vec![0.0f32; n_kv];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_off = kv_slab + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - m).exp();
                sum += *score;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, score) in scores.iter().enumerate() {
                    acc += *score * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Deterministic ramp, bounded to keep dot products inside f16/bf16 range
/// at head_dim=256.
fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.02).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_qtiled(
    ctx: &Context,
    dt: Dt,
    n_q_heads: usize,
    n_kv_heads: usize,
    base_kv: usize,
    n_query: usize,
    kv_stride: usize,
    causal: bool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    const HEAD_DIM: usize = 256;
    const BQ: usize = 16;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let dtype = dt.to_dtype();

    let mut kern = ffai_sdpa_prefill_qtiled_d256::kernel_ir_for(dtype);
    kern.mode = KernelMode::SimdGroup2D;
    let mut bufs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    bufs.insert("q".into(), pack_bytes(q, dt));
    bufs.insert("k".into(), pack_bytes(k, dt));
    bufs.insert("v".into(), pack_bytes(v, dt));
    bufs.insert("out".into(), vec![0u8; n_query * n_q_heads * HEAD_DIM * dt.bytes()]);
    bufs.insert("head_dim".into(), (HEAD_DIM as u32).to_le_bytes().to_vec());
    bufs.insert("n_q_heads".into(), (n_q_heads as u32).to_le_bytes().to_vec());
    bufs.insert("base_kv".into(), (base_kv as u32).to_le_bytes().to_vec());
    bufs.insert("n_query".into(), (n_query as u32).to_le_bytes().to_vec());
    bufs.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    bufs.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    bufs.insert("causal".into(), (causal as u32).to_le_bytes().to_vec());
    bufs.insert("scale".into(), scale.to_le_bytes().to_vec());

    let n_tiles = n_query.div_ceil(BQ);
    let empty: BTreeMap<String, u32> = BTreeMap::new();
    let result = ctx
        .dispatch_with_grid(&kern, &bufs, &empty, [n_tiles, n_q_heads, 1], [128, 1, 1])
        .expect("qtiled dispatch");
    unpack_bytes(result.outputs.get("out").expect("out"), dt)
}

#[allow(clippy::too_many_arguments)]
fn check(
    n_q_heads: usize,
    n_kv_heads: usize,
    base_kv: usize,
    n_query: usize,
    causal: bool,
    dt: Dt,
    tol: f32,
    msg: &str,
) {
    const HEAD_DIM: usize = 256;
    let _lock = gpu_lock();
    let kv_stride = base_kv + n_query;
    let q = ramp(n_query * n_q_heads * HEAD_DIM, 31, 15.0);
    let k = ramp(n_kv_heads * kv_stride * HEAD_DIM, 37, 18.0);
    let v = ramp(n_kv_heads * kv_stride * HEAD_DIM, 41, 20.0);
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let expected = naive_sdpa_append_f32(
        &q, &k, &v, n_q_heads, n_kv_heads, HEAD_DIM, base_kv, n_query, kv_stride, causal, scale,
    );
    let ctx = Context::new().expect("Context");
    let actual = run_qtiled(
        &ctx, dt, n_q_heads, n_kv_heads, base_kv, n_query, kv_stride, causal, &q, &k, &v,
    );
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < tol, "{msg}: max |diff| = {diff:.2e}");
}

// First-chunk case: base_kv == 0, short causal lengths on the diagonal.

#[test]
fn qtiled_first_chunk_f32() { check(4, 1, 0, 48, true, Dt::F32, 1e-3, "first chunk, no prefix"); }

#[test]
fn qtiled_first_chunk_gqa_f32() { check(32, 8, 0, 32, true, Dt::F32, 1e-3, "first chunk gqa"); }

// Append case: base_kv > 0, the regime this kernel targets.

#[test]
fn qtiled_append_f32() { check(4, 1, 300, 48, true, Dt::F32, 1e-3, "append, small prefix"); }

#[test]
fn qtiled_append_gqa_f32() {
    check(32, 8, 2000, 64, true, Dt::F32, 1e-3, "append gqa, large prefix");
}

#[test]
fn qtiled_append_gqa_f16() { check(32, 8, 2000, 64, true, Dt::F16, 5e-2, "append gqa f16"); }

#[test]
fn qtiled_append_gqa_bf16() { check(32, 8, 2000, 64, true, Dt::Bf16, 2e-1, "append gqa bf16"); }

// Ragged tile: n_query not a multiple of BQ=16 (37, 3 tiles, last tile
// only 5 rows) exercises the guard on both loads and stores.

#[test]
fn qtiled_ragged_f32() { check(8, 2, 500, 37, true, Dt::F32, 1e-3, "ragged n_query=37"); }

// Non-causal (full) mode: every row attends the whole logical range.

#[test]
fn qtiled_full_noncausal_f32() { check(4, 1, 128, 32, false, Dt::F32, 1e-3, "full non-causal"); }

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`
//! (F-85 low-context occupancy fix) against the per-row dequant-matmul CPU
//! oracle, on a deliberately SKEWED/ragged routing fixture: several
//! zero-count experts, one expert spanning many BM=16 tiles, several
//! short runs under 16 rows, and no run boundary aligned to 16 - the case
//! that stresses the host `build_tile_plan` boundary math and the
//! kernel's `row_count < 16` tail mask, mirroring
//! `moe_bm64_ragged_correctness.rs`'s bm64 ragged coverage for this
//! kernel's own tile geometry.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, is_unsupported_coop_tensor, pack_bytes, pack_u32_bytes, unpack_bytes};
use wh_iron::{Context, core::ir::KernelMode};
use wh_iron_std::kernels::moe::moe_mpp_tileplan::{
    build_tile_plan,
    iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan,
};

fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    weights
        .chunks_exact(8)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xf) << (i * 4);
            }
            packed
        })
        .collect()
}

/// Per-row-expert int4 dequant-then-matmul CPU oracle (same math as the
/// production kernel: affine `s*q+b`, row `t`'s expert is `indices[t]`).
#[allow(clippy::too_many_arguments)]
fn cpu_oracle(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    indices: &[u32],
    m_total: usize,
    k_in: usize,
    n_out: usize,
    group_size: usize,
) -> Vec<f32> {
    let weight_stride_m = k_in / 8;
    let groups_per_row = k_in / group_size;
    let mut out = vec![0.0f32; m_total * n_out];
    for row in 0..m_total {
        let expert = indices[row] as usize;
        for n in 0..n_out {
            let weight_row_base = expert * n_out * weight_stride_m + n * weight_stride_m;
            let scale_row_base = expert * n_out * groups_per_row + n * groups_per_row;
            let x_row_base = row * k_in;
            let mut acc = 0.0f32;
            for pack_idx in 0..(k_in / 8) {
                let packed = weight_packed[weight_row_base + pack_idx];
                let k_first = pack_idx * 8;
                let g = k_first / group_size;
                let scale = scales[scale_row_base + g];
                let bias = biases[scale_row_base + g];
                for bit in 0..8 {
                    let q = ((packed >> (bit * 4)) & 0xf) as f32;
                    acc += (q * scale + bias) * x[x_row_base + k_first + bit];
                }
            }
            out[row * n_out + n] = acc;
        }
    }
    out
}

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

#[allow(clippy::too_many_arguments)]
fn run_tileplan(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    counts: &[usize],
    m_total: usize,
    n_out: usize,
    k_in: usize,
    group_size: usize,
    dt: Dt,
) -> Option<Vec<f32>> {
    let (tile_expert, tile_row_start, tile_row_count) = build_tile_plan(counts);
    let num_tiles = tile_expert.len();
    let x_rows: Vec<u32> = (0..m_total as u32).collect();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("w".into(), pack_u32_bytes(weight_packed));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("tile_expert".into(), pack_u32_bytes(&tile_expert));
    buffers.insert("tile_row_start".into(), pack_u32_bytes(&tile_row_start));
    buffers.insert("tile_row_count".into(), pack_u32_bytes(&tile_row_count));
    buffers.insert("x_rows".into(), pack_u32_bytes(&x_rows));
    buffers.insert("out".into(), vec![0u8; m_total * n_out * dt.bytes()]);
    buffers.insert("n_out".into(), (n_out as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut k = iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan::kernel_ir_for(dt.to_dtype());
    k.mode = KernelMode::Reduction;
    let grid = [n_out / 32, num_tiles.max(1), 1];
    let r = match ctx.dispatch_with_grid(&k, &buffers, &BTreeMap::new(), grid, [32, 1, 1]) {
        Ok(r) => r,
        // MPP cooperative-tensor kernel the macos-26 toolchain can't build;
        // valid + runs on dev machines. See `common::is_unsupported_coop_tensor`.
        Err(e) if is_unsupported_coop_tensor(&e.to_string()) => {
            eprintln!("skip tileplan: toolchain cannot build the MPP kernel ({e})");
            return None;
        },
        Err(e) => panic!("tileplan dispatch: {e:?}"),
    };
    Some(unpack_bytes(r.outputs.get("out").unwrap(), dt))
}

/// Skewed/ragged fixture: three empty experts, one expert spanning three
/// tiles, several short runs, no run boundary aligned to 16. Same spirit
/// as the bm64 ragged test's `[1, 0, 130, 0, 5, 40, 0, 74]` counts,
/// K_in/N_out kept small so the test is fast.
fn run_case(dtype: Dt) {
    let _g = gpu_lock();
    let counts = [1usize, 0, 47, 0, 5, 22, 0, 3, 16, 9];
    let n_experts = counts.len();
    let m_total: usize = counts.iter().sum();
    let k_in = 64usize;
    let n_out = 64usize;
    let group_size = 32usize;

    let mut rng = Lcg(0xF85_711);
    let total_weights = n_experts * n_out * k_in;
    let weight_unpacked: Vec<u32> = (0..total_weights).map(|_| rng.next_u32() & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();
    let groups_total = n_experts * n_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.001 * (i as f32 * 0.031).sin()).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.004 * (i as f32 * 0.071).cos()).collect();
    let x: Vec<f32> = (0..m_total * k_in).map(|i| 0.05 * (i as f32 * 0.017).sin()).collect();

    let mut indices = Vec::with_capacity(m_total);
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            indices.push(e as u32);
        }
    }

    let s = common::unpack_bytes(&pack_bytes(&scales, dtype), dtype);
    let b = common::unpack_bytes(&pack_bytes(&biases, dtype), dtype);
    let xr = common::unpack_bytes(&pack_bytes(&x, dtype), dtype);
    let expected =
        cpu_oracle(&xr, &weight_packed, &s, &b, &indices, m_total, k_in, n_out, group_size);

    let Some(got) = run_tileplan(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &counts,
        m_total,
        n_out,
        k_in,
        group_size,
        dtype,
    ) else {
        return;
    };

    let mag: f32 = expected.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
    let tol = mag
        * match dtype {
            Dt::F32 => 1e-4,
            _ => 4e-3,
        };
    assert!(
        expected.iter().map(|v| v.abs()).sum::<f32>() > 1e-3,
        "oracle output is all-zero - test is vacuous"
    );

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for row in 0..m_total {
        for c in 0..n_out {
            let d = (got[row * n_out + c] - expected[row * n_out + c]).abs();
            if d > worst {
                worst = d;
                worst_row = row;
            }
        }
    }
    eprintln!(
        "[tileplan] dtype={dtype:?} m_total={m_total} counts={counts:?} worst={worst:.4e} @row={worst_row} (expert={})",
        indices[worst_row]
    );
    assert!(
        worst < tol,
        "tileplan kernel diverges from per-row CPU oracle ({dtype:?}): worst |diff|={worst:.4e} @row={worst_row} tol={tol:.3e}"
    );
}

#[test]
fn tileplan_matches_oracle_on_skewed_ragged_routing_f32() { run_case(Dt::F32); }

#[test]
fn tileplan_matches_oracle_on_skewed_ragged_routing_f16() { run_case(Dt::F16); }

#[test]
fn tileplan_matches_oracle_on_skewed_ragged_routing_bf16() { run_case(Dt::Bf16); }

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Round-5 F-85 batched-verify follow-up: BN-narrow retile of
//! `iron_gemm_q4_bm8` -- ARM B of the round-5 small-M W4 GEMM/GEMV task.
//!
//! `iron_gemm_q4_bm8`'s BM=8/BN=32 `coop_tile` fails the batched-verify gate
//! by 4.5-8.6x, root-caused (see that kernel's doc + the butter
//! `qwen35_spec_gemm_bm8_microbench` verdict) to threadgroup-COUNT collapse:
//! `grid_x = ceil(out_dim/32)` gives only 160-544 threadgroups vs plain
//! `iron_gemv_q4_coalesced`'s `out_dim`-many (5120-17408) -- 32x fewer,
//! independent, cheap threadgroups for GB10's SMs to interleave and hide
//! DRAM latency behind.
//!
//! IMPORTANT CORRECTION discovered while building this retile: `coop_tile`
//! on this CUDA backend is NOT hardware tensor-core MMA despite
//! `TargetProfile::cuda()`'s nominal `mma: MmaStrategy::Wmma16x16x16` --
//! `wh-iron-codegen/src/cuda/mod.rs`'s `CoopTileRun`/`CoopTileZero`/
//! `CoopTileLoadA`/`CoopTileLoadB` emitters branch ONLY on
//! `MmaStrategy::SoftwareLocalC` (`local_c`); every other strategy,
//! including the CUDA default `Wmma16x16x16`, falls through to the same
//! shared-memory nested-loop software GEMM (`for _l in 0..k { _acc +=
//! _CTA[...] * _CTB[...] }` under a per-K-tile `__syncthreads()`). There is
//! no `wmma::fragment`/`mma.sync` emission anywhere in this backend file --
//! `iron_gemm_q4_bm8`'s doc claim of putting the dequant+FMA "on the
//! tensor-core MMA pipe" does not hold on GB10/this codegen; it is a
//! differently-tiled SOFTWARE GEMM with extra `__syncthreads()` overhead
//! per 16-wide K-tile, which independently explains (on top of the TG-count
//! argument) why it measured only 12-22% of the M=1 gemv's bandwidth.
//!
//! Because `coop_tile` has no real hardware fragment-shape constraint here
//! (it is just shared-memory + a nested loop), BN is a free retiling
//! parameter -- unlike a real `wmma`/`mma.sync` backend where 8x8x16 isn't
//! a legal fp16 fragment shape. This file retiles BM=8/BN=32 down to
//! BM=8/BN=16 and BM=8/BN=8 to characterize the TG-count-vs-per-TG-overhead
//! curve per the round-5 task's explicit ask ("cheap while thinking about
//! C") -- NOT expected to clear the 1.4x gate outright (still paying the
//! `__syncthreads()`-per-K-tile software-GEMM tax the M=1 gemv doesn't),
//! but informative about how much of the 4.5-8.6x gap is TG-count vs.
//! kernel-class overhead.
//!
//! Layout / dispatch: identical contract to `iron_gemm_q4_bm8` (see that
//! kernel's doc) except `n_tile_base = tgid_x * BN` and the weight-staging
//! loop uses a grid-stride `range(lane, BN*2, 32)` instead of the BN=32
//! kernel's fixed 2-packs/lane unrolled form, since `BN*2` is not always a
//! multiple of 32 (BN=8 -> 16 packs over 32 lanes: half the lanes get 0
//! iterations, which the grid-stride form handles for free; BN=16 -> 32
//! packs, exactly 1/lane; BN=32 -> 64 packs, exactly 2/lane, matching the
//! original unrolled form's trip count).

use wh_iron::kernel;

#[kernel]
pub fn iron_gemm_q4_bm8n16<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    scales: Tensor<f16>,
    mut out: Tensor<T>,
    #[constexpr] n_rows: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] k_in: u32,
) {
    let n_tile_base = tgid_x * 16u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let bpr = k_in / 32u32;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 x 16
    threadgroup_alloc("ws", 256, coop_stage(T)); // 16 x 16
    threadgroup_alloc("out_scratch", 128, f32); // 8 x 16
    coop_tile_setup(
        "gemm",
        8,
        16,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true,
        true,
        16,
        8,
        true,
        16,
        16,
    );
    coop_tile_zero("gemm");
    for kb in range(0u32, k_in, 16u32) {
        for _e in range(0u32, 4u32, 1u32) {
            let flat = lane * 4u32 + _e;
            let mr = flat / 16u32;
            let kc = flat & 15u32;
            let gr = m_tile_base + mr;
            let in_run = gr < n_rows;
            let safe_g = select(in_run, gr, 0u32);
            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
        }
        // Weight stage: 16 rows x 2 packs/row = 32 packs, exactly 1/lane.
        for pack_id in range(lane, 32u32, 32u32) {
            let w_row = pack_id / 2u32; // 0..15
            let pack_col = pack_id & 1u32;
            let global_col = n_tile_base + w_row;
            let in_run_w = global_col < out_dim;
            let safe_col = select(in_run_w, global_col, 0u32);
            let k_local = pack_col * 8u32;
            let k = kb + k_local;
            let blk = safe_col * bpr + k / 32u32;
            let lane_in_blk = k & 31u32;
            let word = load(qs[blk * 4u32 + lane_in_blk / 8u32]);
            let sc = load(scales[blk]).cast::<f32>();
            let dst = w_row * 16u32 + k_local;
            for _j in range(0u32, 8u32, 1u32) {
                let nib = (word >> (_j * 4u32)) & 0xfu32;
                let q_signed = select(nib >= 8u32, nib - 16u32, nib);
                let qf = q_signed.cast::<i32>().cast::<f32>();
                let w = sc * qf;
                threadgroup_store("ws", dst + _j, select(in_run_w, w, 0.0f32));
            }
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 16, true);
        coop_tile_run("gemm", true);
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "out_scratch", true, f32, 16, 8);
    threadgroup_barrier();
    for _e in range(0u32, 4u32, 1u32) {
        let flat = lane * 4u32 + _e;
        let mr = flat / 16u32;
        let nc = flat & 15u32;
        let gr = m_tile_base + mr;
        let gc = n_tile_base + nc;
        let in_run = (gr < n_rows) & (gc < out_dim);
        if in_run {
            let v = threadgroup_load("out_scratch", mr * 16u32 + nc);
            store(out[gr * out_dim + gc], v.cast::<T>());
        }
    }
}

#[kernel]
pub fn iron_gemm_q4_bm8n8<T>(
    x: Tensor<T>,
    qs: Tensor<u32>,
    scales: Tensor<f16>,
    mut out: Tensor<T>,
    #[constexpr] n_rows: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] k_in: u32,
) {
    let n_tile_base = tgid_x * 8u32;
    let m_tile_base = tgid_y * 8u32;
    let lane = simd_lane;
    let bpr = k_in / 32u32;
    threadgroup_alloc("xs", 128, coop_stage(T)); // 8 x 16
    threadgroup_alloc("ws", 128, coop_stage(T)); // 8 x 16
    threadgroup_alloc("out_scratch", 64, f32); // 8 x 8
    coop_tile_setup(
        "gemm",
        8,
        8,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
        true,
        true,
        16,
        8,
        true,
        16,
        8,
    );
    coop_tile_zero("gemm");
    for kb in range(0u32, k_in, 16u32) {
        for _e in range(0u32, 4u32, 1u32) {
            let flat = lane * 4u32 + _e;
            let mr = flat / 16u32;
            let kc = flat & 15u32;
            let gr = m_tile_base + mr;
            let in_run = gr < n_rows;
            let safe_g = select(in_run, gr, 0u32);
            let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
        }
        // Weight stage: 8 rows x 2 packs/row = 16 packs over 32 lanes --
        // grid-stride loop, half the lanes do zero iterations.
        for pack_id in range(lane, 16u32, 32u32) {
            let w_row = pack_id / 2u32; // 0..7
            let pack_col = pack_id & 1u32;
            let global_col = n_tile_base + w_row;
            let in_run_w = global_col < out_dim;
            let safe_col = select(in_run_w, global_col, 0u32);
            let k_local = pack_col * 8u32;
            let k = kb + k_local;
            let blk = safe_col * bpr + k / 32u32;
            let lane_in_blk = k & 31u32;
            let word = load(qs[blk * 4u32 + lane_in_blk / 8u32]);
            let sc = load(scales[blk]).cast::<f32>();
            let dst = w_row * 16u32 + k_local;
            for _j in range(0u32, 8u32, 1u32) {
                let nib = (word >> (_j * 4u32)) & 0xfu32;
                let q_signed = select(nib >= 8u32, nib - 16u32, nib);
                let qf = q_signed.cast::<i32>().cast::<f32>();
                let w = sc * qf;
                threadgroup_store("ws", dst + _j, select(in_run_w, w, 0.0f32));
            }
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 8, true);
        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 8, true);
        coop_tile_run("gemm", true);
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "out_scratch", true, f32, 8, 8);
    threadgroup_barrier();
    for _e in range(0u32, 2u32, 1u32) {
        let flat = lane * 2u32 + _e;
        let mr = flat / 8u32;
        let nc = flat & 7u32;
        let gr = m_tile_base + mr;
        let gc = n_tile_base + nc;
        let in_run = (gr < n_rows) & (gc < out_dim);
        if in_run {
            let v = threadgroup_load("out_scratch", mr * 8u32 + nc);
            store(out[gr * out_dim + gc], v.cast::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for k in [iron_gemm_q4_bm8n16::kernel_ir_for(dt), iron_gemm_q4_bm8n8::kernel_ir_for(dt)]
            {
                assert_eq!(k.params.len(), 4);
                assert_eq!(k.constexprs.len(), 3);
                let all_ops =
                    || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
                assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
                assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
                assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
            }
        }
    }
}

/// Correctness tests, same oracle shape + API as `gemm_q4_bm8::kernel_tests`
/// (`TestSetup`/`TestBuffer` new-syntax harness, `#[test_kernel]` attribute).
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    #[cfg(not(target_os = "macos"))]
    use super::iron_gemm_q4_bm8n8;
    use super::iron_gemm_q4_bm8n16;
    use crate::utils::pack_f32;

    fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
        let bpr = k / 32;
        let mut qs = vec![0u32; m * bpr * 4];
        let mut scales = vec![0f32; m * bpr];
        for r in 0..m {
            for b in 0..bpr {
                let base = r * k + b * 32;
                let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
                let d = amax / 7.0;
                scales[r * bpr + b] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for word in 0..4 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                        packed |= ((q as u32) & 0xf) << (i * 4);
                    }
                    qs[r * bpr * 4 + b * 4 + word] = packed;
                }
            }
        }
        (qs, scales)
    }

    fn naive(
        x: &[f32],
        qs: &[u32],
        scales_f16: &[f32],
        n_rows: usize,
        out_dim: usize,
        k_in: usize,
    ) -> Vec<f32> {
        let bpr = k_in / 32;
        let mut out = vec![0f32; n_rows * out_dim];
        for r in 0..n_rows {
            for o in 0..out_dim {
                let mut acc = 0f32;
                for kk in 0..k_in {
                    let blk = o * bpr + kk / 32;
                    let lane = kk % 32;
                    let word = qs[blk * 4 + lane / 8];
                    let nib = (word >> ((lane % 8) * 4)) & 0xf;
                    let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                    acc += (q as f32 * scales_f16[blk]) * x[r * k_in + kk];
                }
                out[r * out_dim + o] = acc;
            }
        }
        out
    }

    fn setup_n16(n_rows: usize, out_dim: usize, k_in: usize, dt: DType) -> TestSetup {
        let xv: Vec<f32> =
            (0..n_rows * k_in).map(|i| (i as f32 * 0.011 - 0.5).sin() * 1.3).collect();
        let wv: Vec<f32> =
            (0..out_dim * k_in).map(|i| (i as f32 * 0.017 - 0.3).cos() * 0.9).collect();
        let (qs, scales) = quantize_q4(&wv, out_dim, k_in);
        let scales_f16: Vec<f32> =
            scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
        let expected = naive(&xv, &qs, &scales_f16, n_rows, out_dim, k_in);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
        TestSetup::new(iron_gemm_q4_bm8n16::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&xv, dt), dt))
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F16), DType::F16))
            .input(TestBuffer::zeros("out", n_rows * out_dim, dt))
            .constexpr("n_rows", n_rows as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("k_in", k_in as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_dim as u32).div_ceil(16), (n_rows as u32).div_ceil(8), 1, [32, 1, 1])
    }

    #[cfg(not(target_os = "macos"))]
    fn setup_n8(n_rows: usize, out_dim: usize, k_in: usize, dt: DType) -> TestSetup {
        let xv: Vec<f32> =
            (0..n_rows * k_in).map(|i| (i as f32 * 0.011 - 0.5).sin() * 1.3).collect();
        let wv: Vec<f32> =
            (0..out_dim * k_in).map(|i| (i as f32 * 0.017 - 0.3).cos() * 0.9).collect();
        let (qs, scales) = quantize_q4(&wv, out_dim, k_in);
        let scales_f16: Vec<f32> =
            scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
        let expected = naive(&xv, &qs, &scales_f16, n_rows, out_dim, k_in);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
        TestSetup::new(iron_gemm_q4_bm8n8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&xv, dt), dt))
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F16), DType::F16))
            .input(TestBuffer::zeros("out", n_rows * out_dim, dt))
            .constexpr("n_rows", n_rows as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("k_in", k_in as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_dim as u32).div_ceil(8), (n_rows as u32).div_ceil(8), 1, [32, 1, 1])
    }

    #[test_kernel(dtypes = [f16, bf16], tol = [8e-2, 7e-1])]
    fn test_gemm_q4_bm8n16_tile(dt: DType) -> TestSetup { setup_n16(8, 16, 64, dt) }

    #[test_kernel(dtypes = [f16, bf16], tol = [8e-2, 7e-1])]
    fn test_gemm_q4_bm8n16_verify_shape(dt: DType) -> TestSetup { setup_n16(5, 96, 128, dt) }

    #[test_kernel(dtypes = [f16, bf16], tol = [8e-2, 7e-1])]
    fn test_gemm_q4_bm8n16_edges(dt: DType) -> TestSetup { setup_n16(3, 40, 96, dt) }

    // `iron_gemm_q4_bm8n8`'s BM=8/BN=8 tile cannot run on the Metal backend:
    // `coop_tile_setup`'s `m, n` (8, 8) lower to Metal's
    // `MetalPerformancePrimitives` `matmul2d` cooperative-tensor descriptor,
    // which hard-requires `m % 16 == 0 || n % 16 == 0` (confirmed via
    // `compute-sanitizer`-adjacent MSL compile error: "At least one of M or
    // N must be a multiple of 16") -- neither axis qualifies here, so the
    // shader fails to compile, not a numerics bug. This module's own doc
    // (top of file) already notes `coop_tile` on CUDA is software-emulated
    // (no such fragment-shape constraint), which is exactly why this BN=8
    // retile was buildable/measurable there but is Metal-incompatible.
    // `iron_gemm_q4_bm8n16` (n=16) is unaffected and stays covered above.
    // No CUDA correctness CI exists yet (see correctness.yml) to cover this
    // shape instead -- tracked for when one lands.
    #[cfg(not(target_os = "macos"))]
    #[test_kernel(dtypes = [f16, bf16], tol = [8e-2, 7e-1])]
    fn test_gemm_q4_bm8n8_tile(dt: DType) -> TestSetup { setup_n8(8, 8, 64, dt) }

    #[cfg(not(target_os = "macos"))]
    #[test_kernel(dtypes = [f16, bf16], tol = [8e-2, 7e-1])]
    fn test_gemm_q4_bm8n8_verify_shape(dt: DType) -> TestSetup { setup_n8(5, 96, 128, dt) }

    #[cfg(not(target_os = "macos"))]
    #[test_kernel(dtypes = [f16, bf16], tol = [8e-2, 7e-1])]
    fn test_gemm_q4_bm8n8_edges(dt: DType) -> TestSetup { setup_n8(3, 40, 96, dt) }
}

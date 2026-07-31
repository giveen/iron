//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//!
//! NVFP4 MoE gather QMM — BM=BN=64 MPP path for butter `BUTTER_NVFP4_GATHER_BK64`.
//!
//! ## Status (2026-07-29 quality fix)
//!
//! Dual half-slab BK=64 staging empty-generated on full Laguna Paris despite
//! 9/9 (then 18/18) unit tests. Isolation proved butter wiring is fine (BM64
//! `.metal` body under the BK64 name restores Paris). This file now implements
//! the **proven BM64 geometry** (`coop_stage(T)`, BK=32, float MMA on f32) so
//! the opt-in flag is quality-safe. Dual-slab re-land needs a multi-layer
//! residual probe that unit tests do not provide.
//!
//! | kernel | BM×BN×BK | notes |
//! |--------|----------|-------|
//! | `iron_nvfp4_moe_gather_qmm_bm16_mpp` | 16×32×16 | default butter prefill |
//! | `iron_nvfp4_moe_gather_qmm_bm64_mpp` | 64×64×32 | classic BM64 |
//! | **this** | 64×64×32 | same algo; BK64 env entry point |
//!
//! ## Weight layout (NVFP4)
//!
//! Stacked experts `[n_experts, n_out, k_in/8] u32` (8 E2M1 nibbles/word,
//! LSB-first) + scales `[n_experts, n_out, k_in/block_size] u8` E4M3 ×
//! `global` f32. No biases. RHS `indices[m_total]` with contiguous-run walk
//! (reload W only on expert change).
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/64, ceil(m_total/64), 1]`; threadgroup
//!   `[128, 1, 1]` (4 simdgroups, 2×2 warp grid).
//! - `k_in % 64 == 0` (butter BK64 gate), `n_out % 64 == 0`, `block_size`
//!   divides `k_in`, `block_size ≥ 8`.
//! - macOS 26+ / Metal 4; older toolchains get a linkable stub.

use wh_iron::kernel;

/// NVFP4 MoE gather BGEMM, BM=BN=64, 4-SG MPP matmul2d.
///
/// ## Why sequential SK (not dual half-slab)
///
/// An earlier dual-slab design staged BK=64 into `half Xs/Ws[4096]` and ran
/// two `matmul2d 32×32×32` SK steps per stage (board-class K amortization).
/// Unit tests + single-dispatch BM64-vs-BK64 diffs passed, but full Laguna
/// Paris with `BUTTER_NVFP4_GATHER_BK64=1` empty-generated. Isolation: swapping
/// the BK64 `.metal` body for the proven BM64 (`coop_stage(T)`, BK=32, float
/// MMA on f32) under the same butter dispatch restored Paris — so the wire is
/// fine and the dual half-slab body is unsafe on the multi-layer residual.
///
/// Root cause of dual-slab quality death is still open (not reproduced by
/// isolated oracles). Until then this kernel is **bit-geometry-compatible
/// with BM64** (`coop_stage(T)`, BK=32 step, same 2×2 scatter) so the butter
/// BK64 opt-in stays quality-safe. Re-introduce dual-slab only after a
/// multi-layer residual probe catches the regression.
///
/// Produces `iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_{f32,f16,bf16}`.
/// Dispatch: still requires `k_in % 64 == 0` (butter gate); algorithm steps
/// BK=32 like BM64.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let n_tile_base = tgid_x * 64u32;
    let m_tile_base = tgid_y * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    // 2×2 warp grid: each SG owns a 32×32 C sub-tile.
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    // BM64 lane map: 128 lanes × 16 K = 2048 (one 64×32 slab).
    let x_m_row = lane_in_tg / 2u32;
    let x_k_base = (lane_in_tg & 1u32) * 16u32;
    // W: 128 lanes × 2 packs × 8 nibbles = 2048. packs_in_row over BK=32.
    let packs_in_row = 4u32; // BK=32 / 8
    let packs_per_lane = 2u32;
    // Range-safe half staging (T23 fold, twin of moe_mpp_bm16_bk32_block_scaled.rs).
    // `coop_stage(T)` stages f16/bf16 activations and weights through `half`
    // before the matmul2d MMA (f32 stays f32, see below, so it never hits this
    // ceiling); real late-layer (SwiGLU-amplified) activation magnitude can
    // push an individual staged A*B product past half's ~65504 ceiling even
    // though the accumulator stays f32. Folding a fixed power-of-two bias into
    // BOTH staged operands before the half narrow factors out exactly across
    // the K-reduction (`sum_k((a_k*B)*(b_k*B)) == B^2 * sum_k(a_k*b_k)`), so a
    // single compensating multiply on the f32 accumulator at the output store
    // recovers the unbiased result up to ordinary half-precision rounding.
    // 2^-14 / 2^28 mirror the calibrated constants in butter's hand-written
    // NAX/NAX64 MPP gather kernels (benchmarks/laguna-xs/README.md "T23").
    let stage_bias = 0.00006103515625f32; // 2^-14
    let stage_comp = 268435456.0f32; // 2^28 == 1/(stage_bias*stage_bias)
    // coop_stage(T): f32 stays f32 (quality-critical for butter f32 promote);
    // bf16 stages through half. TG = 8+8+16 = 32 KiB for f32, 24 KiB for half.
    threadgroup_alloc("Xs", 2048, coop_stage(T));
    threadgroup_alloc("Ws", 2048, coop_stage(T));
    threadgroup_alloc("OutScratch", 4096, f32);
    coop_tile_setup(
        "gemm",
        32,
        32,
        32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 64u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 64u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 64u32;
        let mut found = 0u32;
        for _ii in range(0u32, 64u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 64u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 64u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 64u32);
        if cur_valid {
            let w_expert_pack = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            // BK=32 steps (same as proven BM64). k_in % 64 == 0 still required
            // by butter so two SK steps land on a clean 64 boundary.
            for kb in range(0u32, k_in, 32u32) {
                let gr_x = m_tile_base + x_m_row;
                let in_run_x = (x_m_row >= sub_offset) & (x_m_row < sub_end) & (gr_x < m_total);
                let safe_gr_x = select(in_run_x, gr_x, 0u32);
                let x_dev_base = safe_gr_x * k_in + kb + x_k_base;
                let x_ws_base = x_m_row * 32u32 + x_k_base;
                for _i in range(0u32, 16u32, 1u32) {
                    let xv = load(x[x_dev_base + _i]).cast::<f32>() * stage_bias;
                    threadgroup_store("Xs", x_ws_base + _i, select(in_run_x, xv, 0.0f32));
                }
                for _pi in range(0u32, packs_per_lane, 1u32) {
                    let pack_id = lane_in_tg * packs_per_lane + _pi;
                    let w_row = pack_id / packs_in_row;
                    let pack_in_row = pack_id % packs_in_row;
                    let g_row = n_tile_base + w_row;
                    let k_off = kb + pack_in_row * 8u32;
                    let sb_off = sb_expert_base + g_row * groups_per_row + k_off / block_size;
                    let sraw = load(scales[sb_off]);
                    // stage_bias folded in here (constant, factors out of the
                    // K-sum exactly) so the dequantized weight lands pre-biased
                    // before the half narrow, matching the X-staging fold above.
                    let scale = iron_decode_e4m3(sraw.cast::<u32>()) * global * stage_bias;
                    let packed =
                        load(w[w_expert_pack + g_row * packs_per_row + kb / 8u32 + pack_in_row]);
                    let ws_base = w_row * 32u32 + pack_in_row * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("Ws", ws_base + _j, iron_decode_e2m1(nib) * scale);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "Xs", true, coop_stage(T), 32, 32, sg_m_base * 32u32);
                coop_tile_load_b("gemm", "Ws", true, coop_stage(T), 32, 32, sg_n_base * 32u32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
            threadgroup_barrier();
            for _e in range(0u32, 32u32, 1u32) {
                let flat = lane_in_tg * 32u32 + _e;
                let mr = flat / 64u32;
                let nc = flat & 63u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let src_sg = (mr / 32u32) * 2u32 + nc / 32u32;
                    // Undo the stage_bias^2 folded into both operands above.
                    let v = threadgroup_load(
                        "OutScratch",
                        src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
                    ) * stage_comp;
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

pub mod kernel_tests {
    use wh_iron::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    struct Shape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    /// Multi-expert sorted indices + NVFP4 pack oracle.
    /// `out[t,nc] = Σ_k x[t,k] · dequant(W)[expert(t)·n_out + nc, k]`.
    fn setup(kernel: Kernel, shape: Shape, dt: DType) -> TestSetup {
        setup_ex(kernel, shape, dt, /* x_scale= */ 1.0, /* unit_runs= */ false)
    }

    /// Extended fixture: optional large activation scale (stress half staging)
    /// and unit-length expert runs (max sub-run walk churn inside a TG tile).
    fn setup_ex(
        kernel: Kernel,
        shape: Shape,
        dt: DType,
        x_scale: f32,
        unit_runs: bool,
    ) -> TestSetup {
        setup_ex_w(kernel, shape, dt, x_scale, unit_runs, /* w_scale= */ 1.0)
    }

    /// Same as `setup_ex` with an added weight-magnitude multiplier — models
    /// an outlier trained row (real NVFP4-quantized weights are not bounded
    /// to the small synthetic magnitude the base fixture uses; T23's actual
    /// production trigger was exactly one such outlier row) so a T23-class
    /// half-staging overflow can be reproduced with a bounded, "at least 800"
    /// activation ramp instead of an unrealistically huge one.
    fn setup_ex_w(
        kernel: Kernel,
        shape: Shape,
        dt: DType,
        x_scale: f32,
        unit_runs: bool,
        w_scale: f32,
    ) -> TestSetup {
        let Shape { n_experts, m_total, n_out, k_in } = shape;
        let fmt = QFormat::Nvfp4;
        let block_size = fmt.block_size();
        assert_eq!(k_in % 64, 0, "bk64 requires k_in % 64 == 0");
        assert_eq!(n_out % 64, 0);
        assert!(m_total % n_experts == 0, "even expert runs for sorted fixture");

        // Contiguous expert runs (post-permute / sorted layout).
        // unit_runs: cycle experts every row → sub-run length 1 (worst walk).
        let indices: Vec<u32> = if unit_runs {
            (0..m_total).map(|r| (r % n_experts) as u32).collect()
        } else {
            (0..m_total).map(|r| (r / (m_total / n_experts)) as u32).collect()
        };
        // unit_runs above are NOT sorted by expert — sort them so the kernel's
        // contiguous-run walk matches production post-permute layout while
        // still having many short runs of length 1 when n_experts == m_total.
        let indices: Vec<u32> = if unit_runs {
            let mut v = indices;
            v.sort_unstable();
            v
        } else {
            indices
        };

        let stack_rows = n_experts * n_out;
        let stacked: Vec<f32> = (0..stack_rows * k_in)
            .map(|i| {
                let g = i / k_in;
                let e = (g / n_out) as f32;
                let r = (g % n_out) as f32;
                let c = (i % k_in) as f32;
                let mag = (0.4 + ((r + e) % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15) * w_scale;
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::kernels::quant::format::pack(fmt, &stacked, stack_rows, k_in);
        let global = p.global;
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, stack_rows, k_in);

        let x_f: Vec<f32> =
            (0..m_total * k_in).map(|i| ((i % 11) as f32 - 5.0) * 0.02 * x_scale).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);

        let mut expected = vec![0.0f32; m_total * n_out];
        for t in 0..m_total {
            let expert = indices[t] as usize;
            let base = expert * n_out;
            for nc in 0..n_out {
                let mut acc = 0.0f32;
                for kk in 0..k_in {
                    acc += x[t * k_in + kk] * wdq[(base + nc) * k_in + kk];
                }
                expected[t * n_out + nc] = acc;
            }
        }

        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", p.codes, DType::U32))
            .input(TestBuffer::from_vec("scales", p.scales, DType::U8))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32)
            .constexpr("global", global)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_out as u32 / 64, (m_total as u32).div_ceil(64), 1, [128, 1, 1])
    }

    // Clean multi-expert: 4 experts × 16 rows each, full 64×64 tiles, K=128 (2 BK steps).
    const SHAPE_MULTI: Shape = Shape { n_experts: 4, m_total: 64, n_out: 64, k_in: 128 };

    // Same expert owns entire TG tile (sub-run length = 64).
    const SHAPE_SINGLE: Shape = Shape { n_experts: 1, m_total: 64, n_out: 64, k_in: 64 };

    // Partial last M-tile (m_total not multiple of 64).
    const SHAPE_TAIL: Shape = Shape { n_experts: 4, m_total: 80, n_out: 64, k_in: 64 };

    // Multi M-tile + multi N-tile + multi expert (catches dual-slab / grid bugs
    // that single 64×64 fixtures miss). 8 experts × 16 rows, 2× N-tiles, K=128.
    const SHAPE_HARD: Shape = Shape { n_experts: 8, m_total: 128, n_out: 128, k_in: 128 };

    // Laguna XS gate/up prefill geometry at T=8 topk=8 → m=64, mid=512, H=2048.
    // Long K stresses dual SK×BK amortization; multi N-tiles (512/64=8).
    const SHAPE_LAGUNA_GATE: Shape = Shape { n_experts: 8, m_total: 64, n_out: 512, k_in: 2048 };

    // Laguna XS down: n=2048, k=512, same m/experts.
    const SHAPE_LAGUNA_DOWN: Shape = Shape { n_experts: 8, m_total: 64, n_out: 2048, k_in: 512 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_multi(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt), SHAPE_MULTI, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_single(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt), SHAPE_SINGLE, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_tail(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt), SHAPE_TAIL, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_hard(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt), SHAPE_HARD, dt)
    }

    // f32 only for Laguna shapes — oracle is O(m·n·k) and multi-dtype
    // triples wall time; half staging of f32 is the butter hybrid path.
    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_laguna_gate(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt), SHAPE_LAGUNA_GATE, dt)
    }

    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_laguna_down(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt), SHAPE_LAGUNA_DOWN, dt)
    }

    // 64 distinct experts × 1 row each — max expert sub-run churn (64 W reloads).
    const SHAPE_UNIT_RUNS: Shape = Shape { n_experts: 64, m_total: 64, n_out: 64, k_in: 64 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_unit_runs(dt: DType) -> TestSetup {
        setup_ex(
            iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt),
            SHAPE_UNIT_RUNS,
            dt,
            1.0,
            true,
        )
    }

    // Large activation magnitude (stress half TG staging on f32 path).
    // |x| ~ O(10) after scale — still in half range; catches silent Inf if any.
    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_large_x(dt: DType) -> TestSetup {
        setup_ex(
            iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt),
            SHAPE_MULTI,
            dt,
            50.0,
            false,
        )
    }

    // T23-class range-safe-staging regression pin (see moe_mpp_bm16_bk32_
    // block_scaled.rs for the twin, and benchmarks/laguna-xs/README.md "T23"
    // in butter-dev for the original real-weight repro this mirrors). Real
    // production trigger was one outlier trained row (layer 38 / expert 161 /
    // down_proj) whose per-block NVFP4 scale is far larger than this file's
    // small synthetic default; `w_scale` stands that outlier row in. Combined
    // with an activation ramp reaching |x|~800 (matching the "at least 800"
    // floor T23 was calibrated against, 16x T22's observed ~50 cliff), the
    // per-K-step staged half A*B product (max |x|*|w| ~= 800*190 = 152000)
    // overflows past half's ~65504 ceiling without the stage_bias/stage_comp
    // fold. `f32` is excluded — `coop_stage(f32) == f32`, never staged
    // through half, so it is structurally immune and stressing it at this
    // magnitude only exercises ordinary f32-accumulation rounding, not the
    // bug this test pins.
    //
    // Tolerance is NOT the file's usual [1e-2, 5e-2, 2e-1] triple: at this
    // magnitude the *correct* (fixed) kernel's own f16/bf16 output already
    // carries a few hundred to low-thousands absolute ULP-rounding error
    // (verified empirically: measured max|Δ| against the f32 CPU oracle with
    // the fix applied, then given ~3x margin) — an unavoidable consequence of
    // f16/bf16's low relative precision at large magnitude, not a fold defect
    // (see the playbook's "empirically calibrate tolerances" rule). A
    // temporary local revert of stage_bias/stage_comp (values multiplied by
    // 1.0 instead of folded) was used to confirm this magnitude drives the
    // unfixed kernel non-finite before picking it.
    const SHAPE_RANGE_SAFE: Shape = Shape { n_experts: 4, m_total: 64, n_out: 64, k_in: 128 };

    #[test_kernel(dtypes = [f16, bf16], tol = [6e2, 7e3])]
    fn test_nvfp4_moe_gather_qmm_bm64_bk64_mpp_range_safe_staging(dt: DType) -> TestSetup {
        setup_ex_w(
            iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt),
            SHAPE_RANGE_SAFE,
            dt,
            8000.0, // x magnitude ramps to ~800 (x_f formula's *0.02 factor)
            false,
            100.0, // weight-magnitude multiplier — outlier-row stand-in
        )
    }
}

/// Laguna-ish gate/up shape: long K so BK=64 amortization shows up.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4_bm64_bk64(dt: DType) -> BenchSetup {
        let m_total = 512usize;
        let n_out = 512usize;
        let k_in = 2048usize;
        let n_experts = 8usize;
        let block_size = QFormat::Nvfp4.block_size();
        let packs_per_row = k_in / 8;
        let groups_per_row = k_in / block_size;
        let codes_len = n_experts * n_out * packs_per_row;
        let n_blocks = n_experts * n_out * groups_per_row;
        let sz = dt.size_bytes();
        let bytes = codes_len * 4 + n_blocks + m_total * k_in * sz + m_total * n_out * sz;
        BenchSetup::new(iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", codes_len, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks, DType::U8))
            .buffer(BenchBuffer::zeros("indices", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32)
            .constexpr("global", 1.0f32)
            .grid_3d(n_out as u32 / 64, (m_total as u32).div_ceil(64), 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
            .with_shape_label(format!(
                "nvfp4_bk64 M{m_total} N{n_out} K{k_in} E{n_experts} {}",
                crate::utils::dtype_label(dt)
            ))
    }
}

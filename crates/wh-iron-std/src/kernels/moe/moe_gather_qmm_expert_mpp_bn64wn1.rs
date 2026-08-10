//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Alternate tiling for the NVFP4 expert gather-QMM kernel: a 4×1
//! SIMD-group grid (each SG owns a 16-row × full-64-col C sub-tile)
//! instead of the shipped kernel's 2×2 grid (each SG owns a 32×32
//! sub-tile). No epilogue fusion here — this kernel isolates the
//! tiling-geometry question from any epilogue-fusion question, so a
//! later fused-epilogue kernel built on this geometry can be judged
//! against a clean, epilogue-free baseline.
//!
//! ## Expressing N=64 with a single-accumulator cap of N∈{16,32}
//!
//! A single simdgroup-scope `matmul2d` accumulator cannot cover a
//! 64-wide N band directly: Apple's MPP framework hard-caps `N` at 16 or
//! 32 for any `matmul2d` call where both operands are cooperative
//! tensors. A first attempt at `coop_tile_setup("gemm", 16, 64, 32, ...)`
//! fails MSL compilation with Apple's own static_assert from
//! `MPPTensorOpsMatMul2dImpl.h`:
//!
//! ```text
//! static_assert failed due to requirement 'descriptor{16, 64, 32, ...}.n
//! == 16 || descriptor{...}.n == 32' "N must be 16 or 32 if both inputs
//! are cooperative tensors"
//! ```
//!
//! i.e. `mpp::tensor_ops::matmul2d` at `execution_simdgroup` scope caps
//! `N` at 16 or 32 when both operands are cooperative tensors — a single
//! MPP call cannot cover a 64-wide N band for one simdgroup. iron's
//! `coop_tile_*` DSL is built entirely on MPP (`emit_block.rs` emits
//! `mpp::tensor_ops::matmul2d_descriptor` directly), so every kernel that
//! uses it inherits this constraint.
//!
//! **The legal analog built here**: each SG still owns a 16-row band
//! (`WM=4`), but the 64-wide N band is expressed as **two independent
//! N=32 MPP accumulators** (`gemm_lo` for N∈[0,32), `gemm_hi` for
//! N∈[32,64)) run back-to-back inside the same K-step, both fed from the
//! same staged `Ws` tile (no extra W bandwidth — `Ws` was already staged
//! full-width) and the same 16-row `x` window (re-loaded once per
//! accumulator — see "A-bandwidth accounting" below for why this is NOT
//! extra bytes vs the shipped kernel).
//!
//! Two MPP cooperative-tensor accumulators are live per SG, concurrently
//! open across the whole K-loop (they can't be sequenced — that would
//! re-stage `Ws` for the same K range twice, which the whole point of
//! this design is to avoid). That concurrent-accumulator register
//! pressure at BM=16 is the property this kernel exists to measure
//! against the shipped kernel's single-accumulator-per-SG baseline.
//!
//! ## A-bandwidth accounting (why 2 A-loads/SG isn't extra bytes)
//!
//! The shipped kernel's 2×2 grid ALSO reads every physical A row twice
//! per k-step: SGs 0/1 share `sg_m_base=0` (rows 0-31, one per N-half),
//! SGs 2/3 share `sg_m_base=32` — each of the 64 chunk rows is read by
//! exactly 2 SGs (one per N-half) at 32 rows × 32 K × 2B = 2048B/SG ×
//! 4 SGs = 8192B/k-step. This file's 4×1 grid with 2 sub-accumulators
//! reads: 4 SGs × 2 accumulators × 16 rows × 32 K × 2B = 8192B/k-step —
//! byte-identical total. Both designs pay for "N-halving forces 2 reads
//! of A per physical row"; this file just pays it within one SG's own
//! 2 sequential loads instead of across 2 different SGs.
//!
//! ## Result
//!
//! Measured strictly ≥ the shipped 2×2 kernel in isolated GB/s (gateup
//! and down, T512 shape) across repeated runs, and byte-identical wired
//! against the shipped kernel end to end: the concurrent-accumulator
//! register pressure at BM=16 is not, by itself, expensive on this
//! hardware. This kernel is a strict retile of the shipped kernel with
//! no epilogue change, so it is safe to use as the default expert-GEMM
//! path.

use wh_iron::kernel;

/// Retiled variant of `iron_nvfp4_moe_gather_qmm_expert_mpp`: 4×1 SG
/// grid (SM=16/SG), N=64 expressed as two concurrent N=32 MPP
/// accumulators per SG (Apple's MPP caps a single simdgroup-scope
/// `matmul2d` at N∈{16,32} — see module docs for the compile error this
/// works around). No epilogue fusion. See module docs for full detail.
#[kernel(variants(K = [2048, 512], suffix = "k{K}"))]
#[allow(clippy::too_many_arguments)]
pub fn iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1<T>(
    x: Tensor<f16>,
    w: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let expert = tgid_y;
    let n_tile_base = tgid_x * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    // 4×1 warp grid (WM=4, WN=1): each SG owns a 16-row × FULL-64-col C
    // sub-tile — the de-risk retile (module docs).
    let sg_m_base = sg * 16u32;
    let packs_per_row = K / 8u32;
    let groups_per_row = K / block_size;
    // W staging map unchanged from the shipped kernel: 64 N-rows × 4 packs
    // = 256 packs / 128 lanes = 2 packs per lane; both packs share one row
    // and one scale group. The full 64-wide `Ws` is staged once by all 128
    // lanes regardless of how the 4 SGs later split reads of it.
    let pack_id0 = lane_in_tg * 2u32;
    let w_row = pack_id0 / 4u32;
    let pack_in_row0 = pack_id0 & 3u32; // 0 or 2
    let g_row = n_tile_base + w_row;
    let stage_comp = 16384.0f32; // 2^14, same W-only T23 fold as the shipped kernel
    threadgroup_alloc("Ws", 2048, f16); // 64 (N) × 32 (K), 4 KiB — unchanged
    // Two N=32 accumulators per SG (module docs: MPP caps a single
    // simdgroup-scope matmul2d at N∈{16,32}) — both live concurrently
    // across the whole K-loop, the actual thing under test.
    coop_tile_setup("gemm_lo", 16, 32, 32, f16, "accumulate", "simdgroup", f32, false, true, false);
    coop_tile_setup("gemm_hi", 16, 32, 32, f16, "accumulate", "simdgroup", f32, false, true, false);

    // Expert-run binary search — byte-identical to the shipped kernel.
    let mut lo0 = 0u32;
    let mut hi0 = m_total;
    for _s0 in range(0u32, 16u32, 1u32) {
        let has = lo0 < hi0;
        let mid = (lo0 + hi0) >> 1u32;
        let v = load(indices[select(has, mid, 0u32)]);
        let go_right = has & (v < expert);
        let go_left = has & (v >= expert);
        lo0 = select(go_right, mid + 1u32, lo0);
        hi0 = select(go_left, mid, hi0);
    }
    let run_start = lo0;
    let expert_hi = expert + 1u32;
    let mut lo1 = 0u32;
    let mut hi1 = m_total;
    for _s1 in range(0u32, 16u32, 1u32) {
        let has = lo1 < hi1;
        let mid = (lo1 + hi1) >> 1u32;
        let v = load(indices[select(has, mid, 0u32)]);
        let go_right = has & (v < expert_hi);
        let go_left = has & (v >= expert_hi);
        lo1 = select(go_right, mid + 1u32, lo1);
        hi1 = select(go_left, mid, hi1);
    }
    let run_end = lo1;

    let w_expert_pack = expert * n_out * packs_per_row;
    let sb_expert_base = expert * n_out * groups_per_row;

    for chunk_start in range(run_start, run_end, 64u32) {
        let row_count = min(64u32, run_end - chunk_start);
        // Per-SG band skip: SG-uniform, so the coop collective ops below
        // are never lane-diverged. Now 4-way (16-row bands) instead of
        // 2-way (32-row bands).
        let sg_active = sg_m_base < row_count;
        coop_tile_zero("gemm_lo");
        coop_tile_zero("gemm_hi");
        for kb in range(0u32, K, 32u32) {
            threadgroup_barrier();
            let k_off = kb + pack_in_row0 * 8u32;
            let sb_off = sb_expert_base + g_row * groups_per_row + k_off / block_size;
            let sraw = load(scales[sb_off]);
            let scale = iron_decode_e4m3(sraw.cast::<u32>()) * global;
            for _pi in range(0u32, 2u32, 1u32) {
                let pack_in_row = pack_in_row0 + _pi;
                let packed =
                    load(w[w_expert_pack + g_row * packs_per_row + kb / 8u32 + pack_in_row]);
                let ws_base = w_row * 32u32 + pack_in_row * 8u32;
                nvfp4_decode8_store("Ws", ws_base, packed, scale);
            }
            threadgroup_barrier();
            if sg_active {
                // A device-direct: 16 consecutive sorted rows starting at
                // this SG's band (was 32 in the shipped kernel). Loaded
                // once per accumulator — see module docs' A-bandwidth
                // accounting for why this isn't extra total bytes vs the
                // shipped kernel's 2×2 grid.
                let a_off = (chunk_start + sg_m_base) * K + kb;
                coop_tile_load_a("gemm_lo", "x", false, f16, K, 16u32, a_off);
                coop_tile_load_b("gemm_lo", "Ws", true, f16, 32, 32, 0u32);
                coop_tile_run("gemm_lo");
                coop_tile_load_a("gemm_hi", "x", false, f16, K, 16u32, a_off);
                coop_tile_load_b("gemm_hi", "Ws", true, f16, 32, 32, 1024u32);
                coop_tile_run("gemm_hi");
            }
        }
        if sg_active {
            // Device-direct masked store — same mechanism as the shipped
            // kernel (module docs #7 there), run twice (once per N=32
            // accumulator). `sg_row_bound` correctness argument is
            // identical: rows >= run_end belong to a different
            // threadgroup's expert.
            let sg_row_bound = min(16u32, row_count - sg_m_base);
            let cap_lo = coop_tile_capacity("gemm_lo");
            for _e in range(0u32, cap_lo, 1u32) {
                // Axis order (N, M) — see moe_gather_qmm_expert_mpp.rs's
                // store loop for the cross-checked derivation.
                let c = coop_tile_coord("gemm_lo", _e, 0u32);
                let r = coop_tile_coord("gemm_lo", _e, 1u32);
                if r < sg_row_bound {
                    let v = coop_tile_get("gemm_lo", _e) * stage_comp;
                    let mr = chunk_start + sg_m_base + r;
                    let nc = n_tile_base + c;
                    store(out[mr * n_out + nc], v.cast::<T>());
                }
            }
            let cap_hi = coop_tile_capacity("gemm_hi");
            for _e in range(0u32, cap_hi, 1u32) {
                let c = coop_tile_coord("gemm_hi", _e, 0u32);
                let r = coop_tile_coord("gemm_hi", _e, 1u32);
                if r < sg_row_bound {
                    let v = coop_tile_get("gemm_hi", _e) * stage_comp;
                    let mr = chunk_start + sg_m_base + r;
                    let nc = n_tile_base + 32u32 + c;
                    store(out[mr * n_out + nc], v.cast::<T>());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::{
        codegen::msl::MslGenerator,
        core::{dtype::DType, ir::Op},
    };

    use super::*;

    #[test]
    fn kernel_ir_bn64wn1_dual_n32_accumulators() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt);
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            let tg_allocs =
                all_ops().filter(|op| matches!(op, Op::ThreadgroupAlloc { .. })).count();
            assert_eq!(tg_allocs, 1, "expected Ws only, same as the shipped kernel");
            // MPP caps a single simdgroup-scope matmul2d at N∈{16,32}
            // (module docs) — the 64-wide N band is two N=32 setups, not
            // one N=64 setup.
            let setups: Vec<(u32, u32)> = all_ops()
                .filter_map(|op| match op {
                    Op::CoopTileSetup { m, n, .. } => Some((*m, *n)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                setups,
                vec![(16, 32), (16, 32)],
                "expected two SM=16/SN=32 accumulators (gemm_lo, gemm_hi), \
                 MPP disallows a single N=64 simdgroup-scope call"
            );
        }
    }

    #[test]
    fn codegen_bakes_bn64wn1_shape() {
        let mut k = iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k2048::kernel_ir_for(DType::BF16);
        k.name = format!("{}_bf16", k.name);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(
            msl.contains("extents<int, 2048, 16>"),
            "expected K=2048 baked into the device-direct A stride at SM=16:\n{msl}"
        );
        assert!(msl.contains("threadgroup half Ws"), "Ws staging slab missing:\n{msl}");
    }
}

pub mod kernel_tests {
    use wh_iron::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::kernels::moe::moe_gather_qmm_expert_mpp::kernel_tests::{Shape, setup_ex_w};

    fn base(kernel: Kernel, shape: Shape, dt: DType) -> TestSetup {
        setup_ex_w(kernel, shape, dt, 1.0, 0, 1.0)
    }

    const SHAPE_MULTI: Shape = Shape { n_experts: 4, m_total: 64, n_out: 128, k_in: 512 };
    const SHAPE_SINGLE: Shape = Shape { n_experts: 1, m_total: 64, n_out: 64, k_in: 512 };
    const SHAPE_LONG_RUN: Shape = Shape { n_experts: 2, m_total: 256, n_out: 64, k_in: 512 };
    const SHAPE_RAGGED: Shape = Shape { n_experts: 8, m_total: 200, n_out: 64, k_in: 512 };
    const SHAPE_UNIT: Shape = Shape { n_experts: 64, m_total: 64, n_out: 64, k_in: 512 };
    const SHAPE_LAGUNA_GATE: Shape = Shape { n_experts: 8, m_total: 64, n_out: 512, k_in: 2048 };
    const SHAPE_LAGUNA_DOWN: Shape = Shape { n_experts: 8, m_total: 64, n_out: 2048, k_in: 512 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_bn64wn1_multi(dt: DType) -> TestSetup {
        base(iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt), SHAPE_MULTI, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_bn64wn1_single(dt: DType) -> TestSetup {
        base(iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt), SHAPE_SINGLE, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_bn64wn1_long_run(dt: DType) -> TestSetup {
        base(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt),
            SHAPE_LONG_RUN,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_bn64wn1_ragged(dt: DType) -> TestSetup {
        setup_ex_w(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt),
            SHAPE_RAGGED,
            dt,
            1.0,
            5,
            1.0,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_bn64wn1_unit_runs(dt: DType) -> TestSetup {
        base(iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt), SHAPE_UNIT, dt)
    }

    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_bn64wn1_laguna_gate(dt: DType) -> TestSetup {
        base(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k2048::kernel_ir_for(dt),
            SHAPE_LAGUNA_GATE,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_bn64wn1_laguna_down(dt: DType) -> TestSetup {
        base(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt),
            SHAPE_LAGUNA_DOWN,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_bn64wn1_scale_injection(dt: DType) -> TestSetup {
        setup_ex_w(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt),
            Shape { n_experts: 4, m_total: 64, n_out: 128, k_in: 512 },
            dt,
            7.5,
            0,
            1.0,
        )
    }
}

/// Isolated GB/s bench, same shapes as the shipped kernel's
/// `bench_nvfp4_expert_gateup_t512` / `bench_nvfp4_expert_down_t512` — the
/// de-risk gate compares this directly against those two numbers (~298 /
/// ~347 GB/s on the shipped kernel).
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    // Duplicated verbatim from moe_gather_qmm_expert_mpp.rs's private
    // `sorted_indices` helper (not pub there) — deterministic uneven runs
    // in [base/2, base/2 + base], mean = base, the production routing
    // distribution class.
    fn sorted_indices(m_total: usize, n_experts: usize, ragged: bool) -> Vec<u8> {
        let base = m_total / n_experts;
        let mut v: Vec<u32> = Vec::with_capacity(m_total);
        for e in 0..n_experts {
            let run = if ragged { base / 2 + (e * 7919) % (base + 1) } else { base };
            for _ in 0..run {
                if v.len() < m_total {
                    v.push(e as u32);
                }
            }
        }
        while v.len() < m_total {
            v.push((n_experts - 1) as u32);
        }
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    struct B {
        m_total: usize,
        n_out: usize,
        k_in: usize,
        n_experts: usize,
        ragged: bool,
    }

    fn expert_setup(kernel: wh_iron::core::ir::Kernel, b: B, dt: DType) -> BenchSetup {
        let B { m_total, n_out, k_in, n_experts, ragged } = b;
        let block_size = QFormat::Nvfp4.block_size();
        let packs_per_row = k_in / 8;
        let codes_len = n_experts * n_out * packs_per_row;
        let n_blocks = n_experts * n_out * (k_in / block_size);
        let bytes = codes_len * 4
            + n_blocks
            + m_total * k_in * 2 // x f16
            + m_total * n_out * dt.size_bytes();
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", (m_total + 64) * k_in, DType::F16))
            .buffer(BenchBuffer::random("w", codes_len, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks, DType::U8))
            .buffer(BenchBuffer::from_vec(
                "indices",
                sorted_indices(m_total, n_experts, ragged),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("block_size", block_size as u32)
            .constexpr("global", 1.0f32)
            .grid_3d(n_out as u32 / 64, n_experts as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    // Laguna T=512 gate/up: m=4096, n=512, k=2048, 256 experts.
    #[bench(dtypes = [bf16])]
    fn bench_bn64wn1_gateup_t512(dt: DType) -> BenchSetup {
        expert_setup(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k2048::kernel_ir_for(dt),
            B { m_total: 4096, n_out: 512, k_in: 2048, n_experts: 256, ragged: false },
            dt,
        )
        .with_shape_label(format!(
            "bn64wn1_derisk gateup M4096 N512 K2048 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    // Laguna T=512 down: m=4096, n=2048, k=512.
    #[bench(dtypes = [bf16])]
    fn bench_bn64wn1_down_t512(dt: DType) -> BenchSetup {
        expert_setup(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k512::kernel_ir_for(dt),
            B { m_total: 4096, n_out: 2048, k_in: 512, n_experts: 256, ragged: false },
            dt,
        )
        .with_shape_label(format!(
            "bn64wn1_derisk down M4096 N2048 K512 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    // Ragged-tail arm.
    #[bench(dtypes = [bf16])]
    fn bench_bn64wn1_gateup_ragged(dt: DType) -> BenchSetup {
        expert_setup(
            iron_nvfp4_moe_gather_qmm_expert_mpp_bn64wn1_k2048::kernel_ir_for(dt),
            B { m_total: 4096, n_out: 512, k_in: 2048, n_experts: 256, ragged: true },
            dt,
        )
        .with_shape_label(format!(
            "bn64wn1_derisk gateup-ragged M4096 N512 K2048 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }
}

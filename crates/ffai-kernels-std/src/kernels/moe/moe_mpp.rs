//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped BGEMM, BM=16 — `ffai_moe_gather_qmm_mma_int{4,8}_bm16_mpp`.
//!
//! Routes the per-tile matmul through Apple's MetalPerformancePrimitives
//! `mpp::tensor_ops::matmul2d` (cooperative-tensor path, 1 simdgroup). The int4
//! and int8 forms share this entire kernel — the matmul descriptor, the X
//! staging, and the C write-back are identical; only the weight unpack differs,
//! which is folded onto the compile-time `BITS` axis: `vals_per_pack = 32/BITS`
//! values per `u32` (8 nibbles for int4, 4 bytes for int8), decoded by the
//! generic `(packed >> (j*BITS)) & ((1<<BITS)-1)`.
//!
//! ## bf16 staging
//!
//! Apple's `matmul2d` mishandles `bfloat` cooperative tensors, so bf16
//! activations are staged through `half` (10-bit mantissa losslessly covers
//! bf16's 7; accumulation is fp32 regardless). `coop_stage(T)` yields `half`
//! for `T = bf16` and `T` otherwise.
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 32, 16, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — `N=32` satisfies Apple's "at least one of M/N/K = 32"
//! rule; `tb=true` reads W in its native `[N, K]` layout.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[N/32, ceil(M/16), 1]`; threadgroup `[32, 1, 1]`.
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`.
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a linkable stub.
//!
//! Correctness validated by `tests/moe_gather_qmm_mpp_correctness.rs` and
//! `tests/moe_gather_qmm_mpp_int8_correctness.rs` (cosine ≥ 0.999 vs the m1
//! scalar oracle).

use ffai_kernels::kernel;

/// MPP MoE grouped BGEMM, BM=16 / BN=32 / BK=16, one simdgroup. `BITS` ∈ {4, 8}
/// selects the weight precision; produces `ffai_moe_gather_qmm_mma_int4_bm16_mpp`
/// and `_int8_bm16_mpp`.
///
/// Params: `x [m_total, k_in]`, `w [n_experts, n_out, k_in*BITS/32]`
/// (`32/BITS` codes packed per uint32, LSB-first), `scales`/`biases
/// [n_experts, n_out, k_in/group]`, `indices [m_total]` (per-row expert id),
/// `out [m_total, n_out]`.
#[kernel(variants(BITS = [4, 8], suffix = "int{BITS}_bm16_mpp"))]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gather_qmm_mma<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    // Weight packing: `32/BITS` codes per u32 (int4 → 8 nibbles, int8 → 4 bytes).
    let vals_per_pack = 32u32 / BITS;
    let packs_per_row = k_in / vals_per_pack;
    let groups_per_row = k_in / group_size;
    // Threadgroup staging tiles. `coop_stage(T)` = half for bf16, else T —
    // the matmul reads these as cooperative tensors. `out_scratch` is
    // fp32: `coop_tile_store_c` requires the destination elem-type to
    // match the accumulator.
    threadgroup_alloc("xs", 256, coop_stage(T)); // 16 × 16
    threadgroup_alloc("ws", 512, coop_stage(T)); // 32 × 16
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    // MPP descriptor 16×32×16, ta=false tb=true tc=false, accumulate.
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    // Walk the BM=16 rows in contiguous-expert sub-runs.
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        // Find the run end — first row whose expert differs (or OOB).
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_base = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 16u32) {
                // Stage X[m_tile_base..+16, kb..kb+16] → xs. 32 lanes × 8.
                for _e in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + _e;
                    let mr = flat / 16u32;
                    let kc = flat % 16u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
                }
                // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws.
                // 32 lanes × `packs_per_lane` packs/lane; `vals_per_pack` codes/pack.
                let packs_per_lane = 16u32 / vals_per_pack;
                let mask = (1u32 << BITS) - 1u32;
                for _pi in range(0u32, packs_per_lane, 1u32) {
                    let pack_id = lane * packs_per_lane + _pi;
                    let w_row = pack_id / packs_per_lane; // 0..31 (BN rows)
                    let pack_col = pack_id % packs_per_lane; // which u32 in the BK=16 slice
                    let pack_dev = w_expert_base
                        + (n_tile_base + w_row) * packs_per_row
                        + kb / vals_per_pack
                        + pack_col;
                    let packed = load(w[pack_dev]);
                    let k_off = kb + pack_col * vals_per_pack;
                    let g = k_off / group_size;
                    let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
                    let s = load(scales[sb_off]).cast::<f32>();
                    let b = load(biases[sb_off]).cast::<f32>();
                    let dst = w_row * 16u32 + pack_col * vals_per_pack;
                    for _j in range(0u32, vals_per_pack, 1u32) {
                        let q = ((packed >> (_j * BITS)) & mask).cast::<f32>();
                        threadgroup_store("ws", dst + _j, s * q + b);
                    }
                }
                threadgroup_barrier();
                // A = xs [M=16, K=16] (ta=false → extents K,M = 16,16).
                // B = ws [N=32, K=16] (tb=true  → extents K,N = 16,32).
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            // C [M=16, N=32] row-major → extents N,M = 32,16.
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            // Coop-write out_scratch → out with the per-row expert mask.
            // 32 lanes × 16 elems = 512 = BM*BN.
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
    }
}

#[cfg(test)]
mod tests {
    use ffai_kernels::{
        codegen::msl::MslGenerator,
        core::{DType, ir::Op},
    };

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = ffai_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(dt);
            assert_eq!(k.name, "ffai_moe_gather_qmm_mma_int4_bm16_mpp");
            assert_eq!(k.params.len(), 6);
            assert!(k.params[5].is_output);
            assert_eq!(k.constexprs.len(), 4);
            // No raw inline MSL — the matmul is CoopTile* ops.
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }

    /// bf16 must stage through `half`: the `coop_stage(T)` tiles and
    /// cooperative tensors resolve to `half`, never `bfloat`.
    #[test]
    fn bf16_stages_through_half() {
        let k = ffai_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::BF16);
        let setup = std::iter::once(&k.body)
            .chain(k.blocks.values())
            .flat_map(|b| b.ops.iter())
            .find_map(|op| match op {
                Op::CoopTileSetup { act_dtype, .. } => Some(*act_dtype),
                _ => None,
            })
            .expect("CoopTileSetup present");
        assert_eq!(setup, DType::F16, "bf16 activation must stage as half for matmul2d");
    }

    /// Codegen sanity — the MPP header + descriptor land in the MSL, for both
    /// the int4 and int8 variants.
    #[test]
    fn codegen_emits_mpp_include() {
        for (mut k, name) in [
            (
                ffai_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(DType::F32),
                "ffai_moe_gather_qmm_mma_int4_bm16_mpp_f32",
            ),
            (
                ffai_moe_gather_qmm_mma_int8_bm16_mpp::kernel_ir_for(DType::F32),
                "ffai_moe_gather_qmm_mma_int8_bm16_mpp_f32",
            ),
        ] {
            k.name = name.into();
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void {name}")));
        }
    }
}

/// New-syntax correctness tests for the MPP MoE BGEMM (BM=16), int4 + int8.
/// Oracle is the clean per-row-`indices` dequant-then-grouped-matmul: each row
/// `t` resolves its expert from `indices[t]`, dequantizes that expert's weight
/// (`32/BITS` codes/u32, per-group scale/bias), and dots against the row's
/// input. Inputs are dtype-rounded so the GPU sees exactly what the oracle
/// computes; tolerance is wide because the MPP cooperative-tensor accumulator
/// reorders the K reduction.
///
/// Grid (Reduction, 1 simdgroup per TG): `grid_3d(n_out/32, ceil(m_total/16), 1, [32,1,1])`.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::{ffai_moe_gather_qmm_mma_int4_bm16_mpp, ffai_moe_gather_qmm_mma_int8_bm16_mpp};
    use crate::kernels::moe::moe_mpp_shared::{
        MmaTestShape,
        int4_indexed_setup,
        int8_indexed_setup,
    };

    // Clean tile: BM=16 → ceil(64/16)=4 m-tiles, BN=32 → 64/32=2 n-tiles.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int4_bm16_mpp(dt: DType) -> TestSetup {
        int4_indexed_setup(
            ffai_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(dt),
            MmaTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64, group_size: 32 },
            32, // bn
            16, // bm
            32, // tpg
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int8_bm16_mpp(dt: DType) -> TestSetup {
        int8_indexed_setup(
            ffai_moe_gather_qmm_mma_int8_bm16_mpp::kernel_ir_for(dt),
            MmaTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64, group_size: 32 },
            32, // bn
            16, // bm
            32, // tpg
            dt,
        )
    }
}

/// New-syntax benchmarks for the MPP MoE BGEMM (BM=16), int4 + int8.
/// Qwen3.6-A3B-ish.
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::{ffai_moe_gather_qmm_mma_int4_bm16_mpp, ffai_moe_gather_qmm_mma_int8_bm16_mpp};
    use crate::kernels::moe::moe_mpp_shared::{MmaBenchShape, int4_mma_bench};

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int4_bm16_mpp(dt: DType) -> BenchSetup {
        int4_mma_bench(
            ffai_moe_gather_qmm_mma_int4_bm16_mpp::kernel_ir_for(dt),
            MmaBenchShape {
                bits: 4,
                bn: 32,
                bm: 16,
                tpg: 32,
                m_total: 1024,
                n_out: 256,
                k_in: 2048,
                n_experts: 128,
                group_size: 64,
            },
            dt,
        )
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int8_bm16_mpp(dt: DType) -> BenchSetup {
        int4_mma_bench(
            ffai_moe_gather_qmm_mma_int8_bm16_mpp::kernel_ir_for(dt),
            MmaBenchShape {
                bits: 8,
                bn: 32,
                bm: 16,
                tpg: 32,
                m_total: 1024,
                n_out: 256,
                k_in: 2048,
                n_experts: 128,
                group_size: 64,
            },
            dt,
        )
    }
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Expert-outer MoE gather QMM (MPP) — `iron_moe_gather_qmm_mma_int{2,4,8}_expert_grid_mpp`.
//!
//! Grid is `[N/32, n_experts, 1]` (one M-strip of N-tiles per expert) instead of
//! `[N/32, ceil(M/16)]`. Each TG owns a single expert id (`tgid_y`) and binary-
//! searches the **sorted** `indices` for that expert's contiguous run, then
//! walks the run in BM=16 chunks. All N-tiles of the same expert load the same
//! weight rows — much better L2 reuse when only a subset of experts fire
//! (Hy3 T=128 Path B ≈ 84/192 unique).
//!
//! Affine dequant: `scale * q + bias`. Dispatch: Reduction, tpg=32 (1 SG).
//! `indices` must be expert-sorted (same contract as `moe_mpp` BM16).
//!
//! ## DISPATCH INVARIANTS
//!
//! - Mode: `Reduction`; threadgroup `[32, 1, 1]` (1 simdgroup).
//! - Grid: `[n_out / 32, n_experts, 1]` — one TG per (N-tile, expert).
//! - `n_out % 32 == 0`, `k_in % 16 == 0`, `group_size` divides `k_in`.
//! - `indices` length `m_total`, expert-sorted ascending (empty experts allowed).
//! - Empty experts: TG exits after the run scan (no write).
//! - Chunk walk bound is `ceil(m_total / 16)` BM=16 steps, so a single expert
//!   may own every row of the sequence.

use wh_iron::kernel;

/// Expert-outer MPP MoE gather. `BITS` ∈ {2, 4, 8}.
#[kernel(variants(BITS = [2, 4, 8], suffix = "int{BITS}_expert_grid_mpp"))]
#[allow(clippy::too_many_arguments)]
pub fn iron_moe_gather_qmm_mma_eg<T>(
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
    #[constexpr] n_experts: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let expert = tgid_y;
    let lane = simd_lane;
    let vals_per_pack = 32u32 / BITS;
    let packs_per_row = k_in / vals_per_pack;
    let groups_per_row = k_in / group_size;

    // Linear scan for [lo, hi) run of this expert in sorted indices.
    // (Binary search is fine at m_total=1k but linear is clearer in the DSL
    // and only ~1k scalar compares per TG — noise vs the K-loop.)
    let mut lo = m_total;
    let mut hi = m_total;
    let mut found_lo = 0u32;
    for i in range(0u32, m_total, 1u32) {
        let e = load(indices[i]);
        if (e == expert) & (found_lo == 0u32) {
            lo = i;
            found_lo = 1u32;
        }
        if (e == expert) & (found_lo == 1u32) {
            hi = i + 1u32;
        }
    }

    threadgroup_alloc("xs", 512, coop_stage(T)); // 16 × 32
    threadgroup_alloc("ws", 1024, coop_stage(T)); // 32 × 32
    threadgroup_alloc("out_scratch", 512, f32); // 16 × 32
    coop_tile_setup(
        "gemm",
        16,
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

    let w_expert_base = expert * n_out * packs_per_row;
    let sb_expert_base = expert * n_out * groups_per_row;
    let packs_in_bk = 32u32 / vals_per_pack;
    let packs_per_lane = packs_in_bk;
    let mask = (1u32 << BITS) - 1u32;

    // Walk this expert's rows in BM=16 chunks (no-op when lo>=hi).
    // Bound is ceil(m_total/16) so one expert can own the full sequence
    // (the previous hard 64 capped silent-drop at 1024 rows/expert).
    let mut row0 = lo;
    let n_chunks = (m_total + 15u32) / 16u32;
    for _chunk in range(0u32, n_chunks, 1u32) {
        if (expert < n_experts) & (row0 < hi) {
            let row1 = select(row0 + 16u32 < hi, row0 + 16u32, hi);
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                // Stage X for rows [row0, row0+16), mask past row1.
                for _e in range(0u32, 16u32, 1u32) {
                    let flat = lane * 16u32 + _e;
                    let mr = flat / 32u32;
                    let kc = flat % 32u32;
                    let gr = row0 + mr;
                    let in_run = (gr < row1) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>();
                    threadgroup_store("xs", mr * 32u32 + kc, select(in_run, xv, 0.0f32));
                }
                // Dequant W for this expert, N-tile, K-block.
                for _pi in range(0u32, packs_per_lane, 1u32) {
                    let pack_id = lane * packs_per_lane + _pi;
                    let w_row = pack_id / packs_in_bk;
                    let pack_col = pack_id % packs_in_bk;
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
                    let dst = w_row * 32u32 + pack_col * vals_per_pack;
                    for _j in range(0u32, vals_per_pack, 1u32) {
                        let q = ((packed >> (_j * BITS)) & mask).cast::<f32>();
                        threadgroup_store("ws", dst + _j, s * q + b);
                    }
                }
                threadgroup_barrier();
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 32, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 32, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = row0 + mr;
                let gc = n_tile_base + nc;
                let in_run = (gr < row1) & (gr < m_total) & (gc < n_out);
                if in_run {
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
            row0 = row0 + 16u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::{
        codegen::msl::MslGenerator,
        core::{DType, ir::Op},
    };

    use super::*;

    #[test]
    fn kernel_ir_constructs() {
        let k = iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp::kernel_ir_for(DType::BF16);
        assert!(k.name.contains("expert_grid"));
        assert!(
            std::iter::once(&k.body)
                .chain(k.blocks.values())
                .flat_map(|b| b.ops.iter())
                .any(|op| matches!(op, Op::CoopTileSetup { .. }))
        );
    }

    #[test]
    fn codegen_emits_mpp() {
        let k = iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp::kernel_ir_for(DType::F32);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("MetalPerformancePrimitives"));
        assert!(k.name.contains("eg_int2_expert_grid_mpp"));
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{
        iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp,
        iron_moe_gather_qmm_mma_eg_int4_expert_grid_mpp,
        iron_moe_gather_qmm_mma_eg_int8_expert_grid_mpp,
    };
    use crate::kernels::moe::moe_mpp_shared::{
        MmaTestShape,
        int2_indexed_setup,
        int2_indexed_setup_with_indices,
        int4_indexed_setup,
        int8_indexed_setup,
    };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int2_expert_grid_mpp(dt: DType) -> TestSetup {
        // n_experts=4, m_total=64 → 16 rows/expert when sorted evenly.
        let shape = MmaTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64, group_size: 32 };
        let mut setup = int2_indexed_setup(
            iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp::kernel_ir_for(dt),
            shape,
            32, // bn
            16, // bm ignored — we override grid
            32,
            dt,
        );
        // Expert-outer grid: [n_out/32, n_experts]
        setup = setup.grid_3d(64 / 32, 4, 1, [32, 1, 1]);
        // n_experts constexpr — int2_indexed_setup doesn't set it; append.
        setup = setup.constexpr("n_experts", 4u32);
        setup
    }

    /// One expert owns >16 rows (multi-chunk BM=16 walk); experts 1..3 empty.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int2_expert_grid_mpp_multichunk_empty(dt: DType) -> TestSetup {
        // 48 rows → expert 0 only (3 BM=16 chunks); experts 1..3 empty TGs.
        let shape = MmaTestShape { n_experts: 4, m_total: 48, n_out: 64, k_in: 64, group_size: 32 };
        int2_indexed_setup_with_indices(
            iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp::kernel_ir_for(dt),
            shape,
            32,
            16,
            32,
            dt,
            &[0u32; 48],
        )
        .grid_3d(2, 4, 1, [32, 1, 1])
        .constexpr("n_experts", 4u32)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int4_expert_grid_mpp(dt: DType) -> TestSetup {
        let shape = MmaTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64, group_size: 32 };
        int4_indexed_setup(
            iron_moe_gather_qmm_mma_eg_int4_expert_grid_mpp::kernel_ir_for(dt),
            shape,
            32,
            16,
            32,
            dt,
        )
        .grid_3d(2, 4, 1, [32, 1, 1])
        .constexpr("n_experts", 4u32)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_int8_expert_grid_mpp(dt: DType) -> TestSetup {
        let shape = MmaTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64, group_size: 32 };
        int8_indexed_setup(
            iron_moe_gather_qmm_mma_eg_int8_expert_grid_mpp::kernel_ir_for(dt),
            shape,
            32,
            16,
            32,
            dt,
        )
        .grid_3d(2, 4, 1, [32, 1, 1])
        .constexpr("n_experts", 4u32)
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int2_expert_grid_mpp(dt: DType) -> BenchSetup {
        // Realistic Hy3-ish: 84 active experts × ~12 rows (sorted).
        let n_experts = 192u32;
        let m_total = 1024usize;
        let n_out = 1536usize;
        let k_in = 4096usize;
        let group_size = 64usize;
        let active = 84usize;
        let mut indices = vec![0u32; m_total];
        let per = m_total / active;
        for e in 0..active {
            let s = e * per;
            let end = if e + 1 == active { m_total } else { s + per };
            for slot in indices.iter_mut().take(end).skip(s) {
                *slot = e as u32;
            }
        }
        let words_per_row = k_in * 2 / 32;
        let groups_per_row = k_in / group_size;
        let sz = dt.size_bytes();
        let bytes = (active * n_out * words_per_row * 4)
            + 2 * active * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;
        BenchSetup::new(iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random(
                "w",
                n_experts as usize * n_out * words_per_row,
                DType::U32,
            ))
            .buffer(BenchBuffer::random("scales", n_experts as usize * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts as usize * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::from_vec(
                "indices",
                indices.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .constexpr("n_experts", n_experts)
            .grid_3d((n_out as u32) / 32, n_experts, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }
}

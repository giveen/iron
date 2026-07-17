//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Active-only expert-outer MPP gather — `ffai_moe_gather_qmm_mma_eg_int2_active_mpp`.
//!
//! Like `moe_mpp_expert_grid` but `tgid_y` indexes a compact active list
//! (`active_experts` / `row_lo` / `row_hi` from `ffai_moe_compact_active_experts`)
//! so empty experts never launch. Prefer `dispatchThreadgroupsIndirect`.

use ffai_kernels::kernel;

/// Active-list expert-outer MPP int2 gather (f16 path primary for Hy3).
///
/// DISPATCH: Reduction; tpg `[32,1,1]`; grid `[n_out/32, n_active, 1]`
/// (or indirect). `k_in % 16 == 0`, `n_out % 32 == 0`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gather_qmm_mma_eg_int2_active_mpp<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    active_experts: Tensor<u32>,
    row_lo: Tensor<u32>,
    row_hi: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let a = tgid_y;
    let expert = load(active_experts[a]);
    let lo = load(row_lo[a]);
    let hi = load(row_hi[a]);
    let lane = simd_lane;
    let vals_per_pack = 16u32;
    let packs_per_row = k_in / vals_per_pack;
    let groups_per_row = k_in / group_size;

    threadgroup_alloc("xs", 512, coop_stage(T));
    threadgroup_alloc("ws", 1024, coop_stage(T));
    threadgroup_alloc("out_scratch", 512, f32);
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
    let mask = 3u32;

    let mut row0 = lo;
    let n_chunks = (m_total + 15u32) / 16u32;
    for _chunk in range(0u32, n_chunks, 1u32) {
        if row0 < hi {
            let row1 = select(row0 + 16u32 < hi, row0 + 16u32, hi);
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
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
                        let q = ((packed >> (_j * 2u32)) & mask).cast::<f32>();
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

pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_moe_gather_qmm_mma_eg_int2_active_mpp;
    use crate::{
        kernels::moe::moe_mpp_shared::{MmaTestShape, int2_indexed_setup_with_indices},
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_moe_gather_qmm_mma_eg_int2_active_mpp(dt: DType) -> TestSetup {
        // 2 active experts (0 and 2), 32 rows each; expert 1 empty (not in list).
        let m_total = 64usize;
        let n_out = 64usize;
        let k_in = 64usize;
        let group_size = 32usize;
        let n_experts = 4usize;
        let mut indices = vec![0u32; m_total];
        indices[32..64].fill(2);
        let active = vec![0u32, 2];
        let lo = vec![0u32, 32];
        let hi = vec![32u32, 64];
        // Build weight/x via shared helper then rebind buffers for active ABI.
        let base = int2_indexed_setup_with_indices(
            ffai_moe_gather_qmm_mma_eg_int2_active_mpp::kernel_ir_for(dt),
            MmaTestShape { n_experts, m_total, n_out, k_in, group_size },
            32,
            16,
            32,
            dt,
            &indices,
        );
        // Re-construct setup with active buffers (helper uses indices form).
        // Pull expect from base by re-running oracle path: reuse base's expect
        // by building a custom setup from the same tensors.
        let _ = base;
        // Manual setup matching active ABI.
        let mut weight_unpacked = vec![0u32; n_experts * n_out * k_in];
        for (i, w) in weight_unpacked.iter_mut().enumerate() {
            *w = ((i as u32) * 7 + 3) & 0x3;
        }
        let weight_packed: Vec<u32> = weight_unpacked
            .chunks_exact(k_in)
            .flat_map(|row| {
                row.chunks_exact(16).map(|chunk| {
                    let mut packed = 0u32;
                    for (i, &q) in chunk.iter().enumerate() {
                        packed |= (q & 0x3) << (i * 2);
                    }
                    packed
                })
            })
            .collect();
        let n_groups = k_in / group_size;
        let scales_f: Vec<f32> = (0..n_experts * n_out * n_groups)
            .map(|i| 0.005 + 0.001 * (i as f32 * 0.03).sin())
            .collect();
        let biases_f: Vec<f32> = (0..n_experts * n_out * n_groups)
            .map(|i| -0.02 + 0.005 * (i as f32 * 0.07).cos())
            .collect();
        let x_f: Vec<f32> = (0..m_total * k_in).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();
        let s = unpack_f32(&pack_f32(&scales_f, dt), dt);
        let b = unpack_f32(&pack_f32(&biases_f, dt), dt);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        // Oracle via per-row indices.
        let mut expected = vec![0.0f32; m_total * n_out];
        let packs = k_in / 16;
        for row in 0..m_total {
            let expert = indices[row] as usize;
            for n in 0..n_out {
                let wrb = expert * n_out * packs + n * packs;
                let srb = expert * n_out * n_groups + n * n_groups;
                let mut acc = 0.0f32;
                for p in 0..packs {
                    let packed = weight_packed[wrb + p];
                    let k0 = p * 16;
                    let g = k0 / group_size;
                    let scale = s[srb + g];
                    let bias = b[srb + g];
                    for i in 0..16 {
                        let q = ((packed >> (i * 2)) & 0x3) as f32;
                        acc += (q * scale + bias) * x[row * k_in + k0 + i];
                    }
                }
                expected[row * n_out + n] = acc;
            }
        }
        TestSetup::new(ffai_moe_gather_qmm_mma_eg_int2_active_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", u32_bytes(&weight_packed), DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales_f, dt), dt))
            .input(TestBuffer::from_vec("biases", pack_f32(&biases_f, dt), dt))
            .input(TestBuffer::from_vec("active_experts", u32_bytes(&active), DType::U32))
            .input(TestBuffer::from_vec("row_lo", u32_bytes(&lo), DType::U32))
            .input(TestBuffer::from_vec("row_hi", u32_bytes(&hi), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((n_out as u32) / 32, active.len() as u32, 1, [32, 1, 1])
    }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_moe_gather_qmm_mma_eg_int2_active_mpp;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_eg_int2_active_mpp(dt: DType) -> BenchSetup {
        let m_total = 1024usize;
        let n_out = 256usize;
        let k_in = 2048usize;
        let group_size = 64usize;
        let n_experts = 128usize;
        let n_active = 64usize;
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in / 16;
        let sz = dt.size_bytes();
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;
        BenchSetup::new(ffai_moe_gather_qmm_mma_eg_int2_active_mpp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::zeros("active_experts", n_active, DType::U32))
            .buffer(BenchBuffer::zeros("row_lo", n_active, DType::U32))
            .buffer(BenchBuffer::zeros("row_hi", n_active, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "M{m_total} N{n_out} K{k_in} E{n_experts} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / 32, n_active as u32, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }
}

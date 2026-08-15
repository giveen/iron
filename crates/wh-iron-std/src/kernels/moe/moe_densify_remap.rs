//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Densify sparse expert int2 weights into contiguous active slabs, and remap
//! sorted expert ids → dense `[0, n_active)`.

use wh_iron::kernel;

/// Copy active experts' int2 weight + f16 scales/biases into dense slabs.
/// 256 threads per active expert walk packs / scale groups.
///
/// `weight_*`: u32 packs (16 int2/code). `scales_*`/`biases_*`: half.
/// Global thread grid `n_active * 256` (`a = gid/256`, `lane = gid%256`) —
/// the non-generic kernel path lowers `program_id`, not `tgid_x`/`tid`.
#[kernel]
pub fn iron_moe_densify_int2_experts(
    weight_src: Tensor<u32>,
    scales_src: Tensor<f16>,
    biases_src: Tensor<f16>,
    active_experts: Tensor<u32>,
    mut weight_dst: Tensor<u32>,
    mut scales_dst: Tensor<f16>,
    mut biases_dst: Tensor<f16>,
    #[constexpr] n_out: u32,
    #[constexpr] packs_per_row: u32,
    #[constexpr] groups_per_row: u32,
    #[constexpr] n_active: u32,
) {
    let gid = program_id::<0>();
    let a = gid / 256u32;
    let lane = gid % 256u32;
    if a < n_active {
        let expert = load(active_experts[a]);
        let w_src_base = expert * n_out * packs_per_row;
        let w_dst_base = a * n_out * packs_per_row;
        let total_packs = n_out * packs_per_row;
        let n_iters = (total_packs + 255u32) / 256u32;
        for it in range(0u32, n_iters, 1u32) {
            let p = it * 256u32 + lane;
            if p < total_packs {
                store(weight_dst[w_dst_base + p], load(weight_src[w_src_base + p]));
            }
        }
        let sb_src_base = expert * n_out * groups_per_row;
        let sb_dst_base = a * n_out * groups_per_row;
        let total_sb = n_out * groups_per_row;
        let n_sb = (total_sb + 255u32) / 256u32;
        for it in range(0u32, n_sb, 1u32) {
            let p = it * 256u32 + lane;
            if p < total_sb {
                store(scales_dst[sb_dst_base + p], load(scales_src[sb_src_base + p]));
                store(biases_dst[sb_dst_base + p], load(biases_src[sb_src_base + p]));
            }
        }
    }
}

/// Remap sorted expert ids → dense index in the active list.
/// Grid: threads over `m_total`.
#[kernel]
pub fn iron_moe_remap_experts_dense(
    sorted_experts: Tensor<u32>,
    active_experts: Tensor<u32>,
    mut dense_indices: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] n_active: u32,
) {
    let i = program_id::<0>();
    if i < m_total {
        let e = load(sorted_experts[i]);
        let mut dense = 0u32;
        for a in range(0u32, n_active, 1u32) {
            let ae = load(active_experts[a]);
            dense = select(ae == e, a, dense);
        }
        store(dense_indices[i], dense);
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{iron_moe_densify_int2_experts, iron_moe_remap_experts_dense};
    use crate::utils::pack_f32;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32], tol = [0.0])]
    fn test_moe_remap_experts_dense(_dt: DType) -> TestSetup {
        // sorted experts 0,0,2,2,5 → active [0,2,5] → dense 0,0,1,1,2
        let sorted = vec![0u32, 0, 2, 2, 5];
        let active = vec![0u32, 2, 5];
        let expect = vec![0u32, 0, 1, 1, 2];
        TestSetup::new(iron_moe_remap_experts_dense::kernel_ir())
            .input(TestBuffer::from_vec("sorted_experts", u32_bytes(&sorted), DType::U32))
            .input(TestBuffer::from_vec("active_experts", u32_bytes(&active), DType::U32))
            .input(TestBuffer::zeros("dense_indices", sorted.len(), DType::U32))
            .constexpr("m_total", sorted.len() as u32)
            .constexpr("n_active", active.len() as u32)
            .expect(TestBuffer::from_vec("dense_indices", u32_bytes(&expect), DType::U32))
            .grid_1d(sorted.len(), 32)
    }

    #[test_kernel(dtypes = [f16], tol = [0.0])]
    fn test_moe_densify_int2_experts(dt: DType) -> TestSetup {
        // n_experts=4, n_active=2 (experts 1,3), n_out=2, packs=2, groups=2
        let n_experts = 4usize;
        let n_out = 2usize;
        let packs = 2usize;
        let groups = 2usize;
        let active = vec![1u32, 3];
        let n_active = active.len();
        let w_src: Vec<u32> = (0..n_experts * n_out * packs).map(|i| i as u32 + 7).collect();
        let s_src: Vec<f32> =
            (0..n_experts * n_out * groups).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        let b_src: Vec<f32> =
            (0..n_experts * n_out * groups).map(|i| -0.01 * (i as f32 + 1.0)).collect();
        let mut w_dst = vec![0u32; n_active * n_out * packs];
        let mut s_dst = vec![0.0f32; n_active * n_out * groups];
        let mut b_dst = vec![0.0f32; n_active * n_out * groups];
        for (a, &e) in active.iter().enumerate() {
            let e = e as usize;
            for i in 0..(n_out * packs) {
                w_dst[a * n_out * packs + i] = w_src[e * n_out * packs + i];
            }
            for i in 0..(n_out * groups) {
                s_dst[a * n_out * groups + i] = s_src[e * n_out * groups + i];
                b_dst[a * n_out * groups + i] = b_src[e * n_out * groups + i];
            }
        }
        TestSetup::new(iron_moe_densify_int2_experts::kernel_ir())
            // Grid3D, not the default Elementwise: this kernel launches a
            // FIXED 256 threads per active expert (`grid_1d(n_active*256,
            // 256)`) with internal per-thread loops copying multiple
            // elements each — the launched thread count has no relation to
            // any output buffer's element count. CUDA's Elementwise-mode
            // dispatch adds an implicit `if (_gtid >= _n_elems) return`
            // bounds guard sized from "the first output param's element
            // count" (a correct assumption for ordinary 1-thread-per-
            // element kernels, wrong here): with `weight_dst` = 8 elements
            // but 512 threads launched (n_active=2), every thread with
            // `_gtid >= 8` returned immediately — silently dropping every
            // active expert past the first (see GDN_PREFILL_CONTRACT.md
            // §7.3). Grid3D has no such guard (parity with Metal's exact
            // dispatchThreads count); this kernel already carries its own
            // correct explicit bounds (`if a < n_active`, `if p <
            // total_packs/total_sb`), so Grid3D is the right mode.
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("weight_src", u32_bytes(&w_src), DType::U32))
            .input(TestBuffer::from_vec("scales_src", pack_f32(&s_src, dt), dt))
            .input(TestBuffer::from_vec("biases_src", pack_f32(&b_src, dt), dt))
            .input(TestBuffer::from_vec("active_experts", u32_bytes(&active), DType::U32))
            .input(TestBuffer::zeros("weight_dst", n_active * n_out * packs, DType::U32))
            .input(TestBuffer::zeros("scales_dst", n_active * n_out * groups, dt))
            .input(TestBuffer::zeros("biases_dst", n_active * n_out * groups, dt))
            .constexpr("n_out", n_out as u32)
            .constexpr("packs_per_row", packs as u32)
            .constexpr("groups_per_row", groups as u32)
            .constexpr("n_active", n_active as u32)
            .expect(TestBuffer::from_vec("weight_dst", u32_bytes(&w_dst), DType::U32))
            .expect(TestBuffer::from_vec("scales_dst", pack_f32(&s_dst, dt), dt))
            .expect(TestBuffer::from_vec("biases_dst", pack_f32(&b_dst, dt), dt))
            .grid_1d(n_active * 256, 256)
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::{iron_moe_densify_int2_experts, iron_moe_remap_experts_dense};

    #[bench(dtypes = [f32])]
    fn bench_moe_remap_experts_dense(_dt: DType) -> BenchSetup {
        let m_total = 2048usize;
        let n_active = 128usize;
        BenchSetup::new(iron_moe_remap_experts_dense::kernel_ir())
            .buffer(BenchBuffer::zeros("sorted_experts", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("active_experts", n_active, DType::U32))
            .buffer(BenchBuffer::zeros("dense_indices", m_total, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_active", n_active as u32)
            .grid_1d(m_total, 32)
            .bytes_moved((m_total * 8) as u64)
    }

    #[bench(dtypes = [f16])]
    fn bench_moe_densify_int2_experts(dt: DType) -> BenchSetup {
        let n_out = 256usize;
        let packs_per_row = 128usize;
        let groups_per_row = 32usize;
        let n_active = 64usize;
        let n_experts = 128usize;
        BenchSetup::new(iron_moe_densify_int2_experts::kernel_ir())
            // Grid3D, not the default Elementwise — see the matching
            // comment on `test_moe_densify_int2_experts`'s TestSetup for
            // why (CUDA's Elementwise bounds guard is sized from the wrong
            // quantity for this kernel's threading pattern). Not exposed
            // at this bench's shape (`n_out*packs_per_row` >> 256 here),
            // but correct regardless of shape.
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::zeros("weight_src", n_experts * n_out * packs_per_row, DType::U32))
            .buffer(BenchBuffer::zeros("scales_src", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::zeros("biases_src", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::zeros("active_experts", n_active, DType::U32))
            .buffer(
                BenchBuffer::zeros("weight_dst", n_active * n_out * packs_per_row, DType::U32)
                    .output(),
            )
            .buffer(
                BenchBuffer::zeros("scales_dst", n_active * n_out * groups_per_row, dt).output(),
            )
            .buffer(
                BenchBuffer::zeros("biases_dst", n_active * n_out * groups_per_row, dt).output(),
            )
            .constexpr("n_out", n_out as u32)
            .constexpr("packs_per_row", packs_per_row as u32)
            .constexpr("groups_per_row", groups_per_row as u32)
            .constexpr("n_active", n_active as u32)
            .grid_1d(n_active * 256, 256)
            .bytes_moved((n_active * n_out * packs_per_row * 4) as u64)
    }
}

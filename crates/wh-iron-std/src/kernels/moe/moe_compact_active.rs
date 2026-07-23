//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Compact sorted expert ids into an active-only list + row spans, and write
//! `MTLDispatchThreadgroupsIndirectArguments` for expert-outer GU / down QMM.
//!
//! Single TG, thread 0 does the scan (`m_total` ≤ ~8K typical).

use wh_iron::kernel;

/// Compact sorted experts → active list + [lo, hi) spans + indirect grids.
///
/// DISPATCH: one TG, one thread (`tpg = [1,1,1]`).
#[kernel]
pub fn iron_moe_compact_active_experts(
    sorted_experts: Tensor<u32>,
    mut active_experts: Tensor<u32>,
    mut row_lo: Tensor<u32>,
    mut row_hi: Tensor<u32>,
    mut n_active_out: Tensor<u32>,
    mut indirect_gu: Tensor<u32>,
    mut indirect_down: Tensor<u32>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out_gu: u32,
    #[constexpr] n_out_down: u32,
    #[constexpr] n_experts_cap: u32,
) {
    let t = tid;
    if t == 0u32 {
        if m_total == 0u32 {
            store(n_active_out[0u32], 0u32);
            store(indirect_gu[0u32], n_out_gu / 32u32);
            store(indirect_gu[1u32], 0u32);
            store(indirect_gu[2u32], 1u32);
            store(indirect_down[0u32], n_out_down / 32u32);
            store(indirect_down[1u32], 0u32);
            store(indirect_down[2u32], 1u32);
        } else {
            let mut n_active = 0u32;
            let mut i = 0u32;
            // Cap outer iterations by n_experts_cap (each step advances a run).
            for _step in range(0u32, n_experts_cap, 1u32) {
                if i < m_total {
                    let e = load(sorted_experts[i]);
                    let lo = i;
                    let mut j = i + 1u32;
                    // Scan run end — bound by m_total. Clamp the probe index so
                    // the load never reads past the buffer (j == m_total after
                    // the run ends; `&` does not short-circuit the load).
                    for _inner in range(0u32, m_total, 1u32) {
                        let jj = select(j < m_total, j, m_total - 1u32);
                        let cont = (j < m_total) & (load(sorted_experts[jj]) == e);
                        j = select(cont, j + 1u32, j);
                    }
                    store(active_experts[n_active], e);
                    store(row_lo[n_active], lo);
                    store(row_hi[n_active], j);
                    n_active = n_active + 1u32;
                    i = j;
                }
            }
            store(n_active_out[0u32], n_active);
            store(indirect_gu[0u32], n_out_gu / 32u32);
            store(indirect_gu[1u32], n_active);
            store(indirect_gu[2u32], 1u32);
            store(indirect_down[0u32], n_out_down / 32u32);
            store(indirect_down[1u32], n_active);
            store(indirect_down[2u32], 1u32);
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_moe_compact_active_experts;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32], tol = [0.0])]
    fn test_moe_compact_active_experts(_dt: DType) -> TestSetup {
        // sorted: 0,0,0, 2,2, 5,5,5,5 → 3 active, spans [0,3),[3,5),[5,9)
        let sorted = vec![0u32, 0, 0, 2, 2, 5, 5, 5, 5];
        let m_total = sorted.len();
        let n_out_gu = 64u32;
        let n_out_down = 128u32;
        let cap = 16u32;
        let n_active = 3u32;
        let mut active_pad = vec![0u32; cap as usize];
        let mut lo_pad = vec![0u32; cap as usize];
        let mut hi_pad = vec![0u32; cap as usize];
        active_pad[0] = 0;
        active_pad[1] = 2;
        active_pad[2] = 5;
        lo_pad[0] = 0;
        lo_pad[1] = 3;
        lo_pad[2] = 5;
        hi_pad[0] = 3;
        hi_pad[1] = 5;
        hi_pad[2] = 9;
        let ind_gu = vec![n_out_gu / 32, n_active, 1];
        let ind_down = vec![n_out_down / 32, n_active, 1];
        TestSetup::new(iron_moe_compact_active_experts::kernel_ir())
            .input(TestBuffer::from_vec("sorted_experts", u32_bytes(&sorted), DType::U32))
            .input(TestBuffer::zeros("active_experts", cap as usize, DType::U32))
            .input(TestBuffer::zeros("row_lo", cap as usize, DType::U32))
            .input(TestBuffer::zeros("row_hi", cap as usize, DType::U32))
            .input(TestBuffer::zeros("n_active_out", 1, DType::U32))
            .input(TestBuffer::zeros("indirect_gu", 3, DType::U32))
            .input(TestBuffer::zeros("indirect_down", 3, DType::U32))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out_gu", n_out_gu)
            .constexpr("n_out_down", n_out_down)
            .constexpr("n_experts_cap", cap)
            .expect(TestBuffer::from_vec("active_experts", u32_bytes(&active_pad), DType::U32))
            .expect(TestBuffer::from_vec("row_lo", u32_bytes(&lo_pad), DType::U32))
            .expect(TestBuffer::from_vec("row_hi", u32_bytes(&hi_pad), DType::U32))
            .expect(TestBuffer::from_vec("n_active_out", u32_bytes(&[n_active]), DType::U32))
            .expect(TestBuffer::from_vec("indirect_gu", u32_bytes(&ind_gu), DType::U32))
            .expect(TestBuffer::from_vec("indirect_down", u32_bytes(&ind_down), DType::U32))
            .grid_3d(1, 1, 1, [1, 1, 1])
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_moe_compact_active_experts;

    #[bench(dtypes = [f32])]
    fn bench_moe_compact_active_experts(_dt: DType) -> BenchSetup {
        let m_total = 1024usize;
        let n_out_gu = 256u32;
        let n_out_down = 4096u32;
        let n_experts_cap = 128u32;
        let cap = 128usize;
        BenchSetup::new(iron_moe_compact_active_experts::kernel_ir())
            .buffer(BenchBuffer::zeros("sorted_experts", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("active_experts", cap, DType::U32).output())
            .buffer(BenchBuffer::zeros("row_lo", cap, DType::U32).output())
            .buffer(BenchBuffer::zeros("row_hi", cap, DType::U32).output())
            .buffer(BenchBuffer::zeros("n_active_out", 1, DType::U32).output())
            .buffer(BenchBuffer::zeros("indirect_gu", 3, DType::U32).output())
            .buffer(BenchBuffer::zeros("indirect_down", 3, DType::U32).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out_gu", n_out_gu)
            .constexpr("n_out_down", n_out_down)
            .constexpr("n_experts_cap", n_experts_cap)
            .grid_3d(1, 1, 1, [1, 1, 1])
            .bytes_moved((m_total * 4) as u64)
    }
}

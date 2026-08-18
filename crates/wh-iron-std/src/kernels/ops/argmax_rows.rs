//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Exact row-wise argmax with one threadgroup per contiguous row.

use wh_iron::kernel;

/// Select one index from every row of a contiguous matrix.
///
/// Dispatch one 256-thread threadgroup per row. Values are ordered descending,
/// equal values prefer the lower column, and NaNs follow numeric values.
#[kernel]
pub fn iron_argmax_rows<T>(inp: Tensor<T>, mut out: Tensor<u32>, #[constexpr] n_cols: u32) {
    let row = tgid_x;
    let lane = simd_lane;
    let simdgroup = simd_id;
    let invalid_id = 4294967295u32;
    let mut best_value = neg_infinity();
    let mut best_id = invalid_id;
    let mut best_valid = 0u32;
    let n_iters = (n_cols + 255u32) / 256u32;

    for slot in range(0u32, n_iters, 1u32) {
        let column = tid + slot * 256u32;
        if column < n_cols {
            let value = load(inp[row * n_cols + column]).cast::<f32>();
            let numeric = select(value == value, 1u32, 0u32);
            let better = (numeric > best_valid)
                | ((numeric == best_valid)
                    & ((value > best_value) | ((value == best_value) & (column < best_id))));
            best_value = select(better, value, best_value);
            best_id = select(better, column, best_id);
            best_valid = select(better, numeric, best_valid);
        }
    }

    let simd_valid = simd_max(best_valid);
    let simd_value = simd_max(select(best_valid == simd_valid, best_value, neg_infinity()));
    let simd_winner_id = simd_min(select(
        (best_valid == simd_valid) & (best_value == simd_value),
        best_id,
        invalid_id,
    ));

    threadgroup_alloc("simd_values", 8u32, "f32");
    threadgroup_alloc("simd_ids", 8u32, "u32");
    threadgroup_alloc("simd_valid", 8u32, "u32");
    if lane == 0u32 {
        threadgroup_store("simd_values", simdgroup, simd_value);
        threadgroup_store("simd_ids", simdgroup, simd_winner_id);
        threadgroup_store("simd_valid", simdgroup, simd_valid);
    }
    threadgroup_barrier();

    if simdgroup == 0u32 {
        let in_range = lane < 8u32;
        best_value = threadgroup_load("simd_values", select(in_range, lane, 0u32));
        best_id = threadgroup_load("simd_ids", select(in_range, lane, 0u32));
        best_valid =
            select(in_range, threadgroup_load("simd_valid", select(in_range, lane, 0u32)), 0u32);
        best_value = select(in_range, best_value, neg_infinity());
        best_id = select(in_range, best_id, invalid_id);

        let final_valid = simd_max(best_valid);
        let final_value = simd_max(select(best_valid == final_valid, best_value, neg_infinity()));
        let final_id = simd_min(select(
            (best_valid == final_valid) & (best_value == final_value),
            best_id,
            invalid_id,
        ));
        if lane == 0u32 {
            store(out[row], final_id);
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_argmax_rows;
    use crate::utils::pack_f32;

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    #[test_kernel(dtypes = [bf16], tol = 0.0)]
    fn test_argmax_rows_production_width(dt: DType) -> TestSetup {
        let (n_rows, n_cols) = (9usize, 248_320usize);
        let mut source: Vec<f32> = (0..n_rows * n_cols)
            .map(|index| ((index * 73 + 29) % 1_009) as f32 / 1_024.0)
            .collect();
        let mut expected = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let lower = 97 + row * 1_003;
            let higher = lower + 65_536;
            source[row * n_cols + lower] = 10.0;
            source[row * n_cols + higher] = 10.0;
            source[row * n_cols + lower + 1] = f32::NAN;
            expected.push(lower as u32);
        }
        TestSetup::new(iron_argmax_rows::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&source, dt), dt))
            .input(TestBuffer::zeros("out", n_rows, DType::U32))
            .constexpr("n_cols", n_cols as u32)
            .expect(TestBuffer::from_vec("out", u32_bytes(&expected), DType::U32))
            .grid_3d(n_rows as u32, 1, 1, [256, 1, 1])
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_argmax_rows;

    #[bench(dtypes = [bf16])]
    fn bench_argmax_rows(dt: DType) -> BenchSetup {
        let (n_rows, n_cols) = (8usize, 248_320usize);
        BenchSetup::new(iron_argmax_rows::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", n_rows * n_cols, dt))
            .buffer(BenchBuffer::zeros("out", n_rows, DType::U32).output())
            .constexpr("n_cols", n_cols as u32)
            .grid_3d(n_rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((n_rows * n_cols * dt.size_bytes() + n_rows * 4) as u64)
    }
}

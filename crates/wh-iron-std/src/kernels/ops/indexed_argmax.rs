//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Exact argmax over a compact value list with caller-provided output IDs.

use wh_iron::kernel;

/// Select the ID paired with the strongest value.
///
/// Dispatch one 32-thread SIMDgroup. `n` must be at most 32. Numeric values
/// precede NaNs, and equal values prefer the lower caller-provided ID.
#[kernel]
pub fn iron_indexed_argmax<T>(
    values: Tensor<T>,
    indices: Tensor<u32>,
    mut out: Tensor<u32>,
    #[constexpr] n: u32,
) {
    let lane = simd_lane;
    let in_range = lane < n;
    let source = select(in_range, lane, 0u32);
    let value = load(values[source]).cast::<f32>();
    let id = load(indices[source]);
    let numeric = select(in_range & (value == value), 1u32, 0u32);
    let eligible_value = select(numeric == 1u32, value, neg_infinity());
    let best_value = simd_max(eligible_value);
    let invalid_id = 4294967295u32;
    let eligible_id = select((numeric == 1u32) & (value == best_value), id, invalid_id);
    let best_id = simd_min(eligible_id);
    if lane == 0u32 {
        store(out[0u32], best_id);
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_indexed_argmax;
    use crate::utils::pack_f32;

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_indexed_argmax_prefers_lower_id(dt: DType) -> TestSetup {
        let values = [1.0, 7.0, f32::NAN, 3.0, 7.0, -2.0, 4.0, 0.0];
        let indices = [90u32, 81, 2, 17, 12, 44, 9, 31];
        TestSetup::new(iron_indexed_argmax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("values", pack_f32(&values, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", 1, DType::U32))
            .constexpr("n", values.len() as u32)
            .expect(TestBuffer::from_vec("out", u32_bytes(&[12]), DType::U32))
            .grid_3d(1, 1, 1, [32, 1, 1])
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_indexed_argmax;

    #[bench(dtypes = [bf16])]
    fn bench_indexed_argmax(dt: DType) -> BenchSetup {
        BenchSetup::new(iron_indexed_argmax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("values", 16, dt))
            .buffer(BenchBuffer::random("indices", 16, DType::U32))
            .buffer(BenchBuffer::zeros("out", 1, DType::U32).output())
            .constexpr("n", 16u32)
            .grid_3d(1, 1, 1, [32, 1, 1])
            .bytes_moved((16 * (dt.size_bytes() + 4) + 4) as u64)
    }
}

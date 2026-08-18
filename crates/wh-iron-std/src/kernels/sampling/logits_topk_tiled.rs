//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Two-stage exact top-k selection for one wide logits row.
//!
//! The partial kernel partitions the row across independent threadgroups and
//! emits each partition's strongest `k` indices. The finalize kernel reduces
//! that compact candidate set. Any global top-k member must appear in its
//! partition's local top-k, so the decomposition is exact.

use wh_iron::kernel;

/// Emit the strongest `k` indices from each strided partition.
///
/// Dispatch `n_tiles` threadgroups of 256 threads. `values_per_thread` must
/// cover `ceil(n / (n_tiles * 256))` and be at most 8. `k` must be at most 16.
#[kernel]
pub fn iron_logits_topk_tiled_partial<T>(
    logits: Tensor<T>,
    mut candidate_indices: Tensor<u32>,
    #[constexpr] n: u32,
    #[constexpr] n_tiles: u32,
    #[constexpr] values_per_thread: u32,
    #[constexpr] k: u32,
) {
    let tile = tgid_x;
    let lane = simd_lane;
    let simdgroup = simd_id;
    let invalid_id = 4294967295u32;
    let stride = n_tiles * 256u32;

    stack_alloc("values", 8u32, "f32");
    stack_alloc("ids", 8u32, "u32");
    stack_alloc("valid", 8u32, "u32");
    for slot in range(0u32, 8u32, 1u32) {
        stack_store("values", slot, neg_infinity());
        stack_store("ids", slot, invalid_id);
        stack_store("valid", slot, 0u32);
    }
    for slot in range(0u32, values_per_thread, 1u32) {
        let index = tile * 256u32 + tid + slot * stride;
        if index < n {
            let value = load(logits[index]).cast::<f32>();
            let numeric = select(value == value, 1u32, 0u32);
            stack_store("values", slot, select(numeric == 1u32, value, neg_infinity()));
            stack_store("ids", slot, index);
            stack_store("valid", slot, numeric);
        }
    }

    threadgroup_alloc("simd_candidates", 128u32, "u32");
    let mut chosen = 0u32;
    for rank in range(0u32, k, 1u32) {
        let mut best_value = neg_infinity();
        let mut best_id = invalid_id;
        let mut best_valid = 0u32;
        let mut best_slot = invalid_id;
        for slot in range(0u32, values_per_thread, 1u32) {
            let available = (chosen & (1u32 << slot)) == 0u32;
            let value = stack_load("values", slot);
            let id = stack_load("ids", slot);
            let valid = stack_load("valid", slot);
            let better = available
                & ((valid > best_valid)
                    | ((valid == best_valid)
                        & ((value > best_value) | ((value == best_value) & (id < best_id)))));
            best_value = select(better, value, best_value);
            best_id = select(better, id, best_id);
            best_valid = select(better, valid, best_valid);
            best_slot = select(better, slot, best_slot);
        }

        let global_valid = simd_max(best_valid);
        let eligible_value = select(best_valid == global_valid, best_value, neg_infinity());
        let global_value = simd_max(eligible_value);
        let eligible_id = select(
            (best_valid == global_valid) & (best_value == global_value),
            best_id,
            invalid_id,
        );
        let global_id = simd_min(eligible_id);
        if best_id == global_id && best_slot != invalid_id {
            chosen = chosen | (1u32 << best_slot);
        }
        if lane == 0u32 {
            threadgroup_store("simd_candidates", simdgroup * k + rank, global_id);
        }
    }
    threadgroup_barrier();

    if simdgroup == 0u32 {
        stack_alloc("merge_values", 4u32, "f32");
        stack_alloc("merge_ids", 4u32, "u32");
        stack_alloc("merge_valid", 4u32, "u32");
        for slot in range(0u32, 4u32, 1u32) {
            let candidate = slot * 32u32 + lane;
            let in_range = candidate < 8u32 * k;
            let id = threadgroup_load("simd_candidates", select(in_range, candidate, 0u32));
            let value = load(logits[select(in_range, id, 0u32)]).cast::<f32>();
            let numeric = select(value == value, 1u32, 0u32);
            stack_store(
                "merge_values",
                slot,
                select(in_range & (numeric == 1u32), value, neg_infinity()),
            );
            stack_store("merge_ids", slot, select(in_range, id, invalid_id));
            stack_store("merge_valid", slot, select(in_range, numeric, 0u32));
        }

        let mut merge_chosen = 0u32;
        for rank in range(0u32, k, 1u32) {
            let mut best_value = neg_infinity();
            let mut best_id = invalid_id;
            let mut best_valid = 0u32;
            let mut best_slot = invalid_id;
            for slot in range(0u32, 4u32, 1u32) {
                let available = (merge_chosen & (1u32 << slot)) == 0u32;
                let value = stack_load("merge_values", slot);
                let id = stack_load("merge_ids", slot);
                let valid = stack_load("merge_valid", slot);
                let better = available
                    & ((valid > best_valid)
                        | ((valid == best_valid)
                            & ((value > best_value) | ((value == best_value) & (id < best_id)))));
                best_value = select(better, value, best_value);
                best_id = select(better, id, best_id);
                best_valid = select(better, valid, best_valid);
                best_slot = select(better, slot, best_slot);
            }
            let global_valid = simd_max(best_valid);
            let eligible_value = select(best_valid == global_valid, best_value, neg_infinity());
            let global_value = simd_max(eligible_value);
            let eligible_id = select(
                (best_valid == global_valid) & (best_value == global_value),
                best_id,
                invalid_id,
            );
            let global_id = simd_min(eligible_id);
            if best_id == global_id && best_slot != invalid_id {
                merge_chosen = merge_chosen | (1u32 << best_slot);
            }
            if lane == 0u32 {
                store(candidate_indices[tile * k + rank], global_id);
            }
        }
    }
}

/// Reduce the tiled candidate set to the final strongest `k` indices.
/// Dispatch one 32-thread threadgroup. `values_per_thread` must cover
/// `ceil(n_candidates / 32)` and be at most 32. `k` must be at most 16.
#[kernel]
pub fn iron_logits_topk_tiled_finalize<T>(
    logits: Tensor<T>,
    candidate_indices: Tensor<u32>,
    mut indices_out: Tensor<u32>,
    #[constexpr] n_candidates: u32,
    #[constexpr] values_per_thread: u32,
    #[constexpr] k: u32,
) {
    let lane = simd_lane;
    let invalid_id = 4294967295u32;
    stack_alloc("values", 32u32, "f32");
    stack_alloc("ids", 32u32, "u32");
    stack_alloc("valid", 32u32, "u32");
    for slot in range(0u32, values_per_thread, 1u32) {
        let candidate = slot * 32u32 + lane;
        let in_range = candidate < n_candidates;
        let id = load(candidate_indices[select(in_range, candidate, 0u32)]);
        let value = load(logits[select(in_range, id, 0u32)]).cast::<f32>();
        let numeric = select(value == value, 1u32, 0u32);
        stack_store("values", slot, select(in_range & (numeric == 1u32), value, neg_infinity()));
        stack_store("ids", slot, select(in_range, id, invalid_id));
        stack_store("valid", slot, select(in_range, numeric, 0u32));
    }

    let mut chosen = 0u32;
    for rank in range(0u32, k, 1u32) {
        let mut best_value = neg_infinity();
        let mut best_id = invalid_id;
        let mut best_valid = 0u32;
        let mut best_slot = invalid_id;
        for slot in range(0u32, values_per_thread, 1u32) {
            let available = (chosen & (1u32 << slot)) == 0u32;
            let value = stack_load("values", slot);
            let id = stack_load("ids", slot);
            let valid = stack_load("valid", slot);
            let better = available
                & ((valid > best_valid)
                    | ((valid == best_valid)
                        & ((value > best_value) | ((value == best_value) & (id < best_id)))));
            best_value = select(better, value, best_value);
            best_id = select(better, id, best_id);
            best_valid = select(better, valid, best_valid);
            best_slot = select(better, slot, best_slot);
        }
        let global_valid = simd_max(best_valid);
        let eligible_value = select(best_valid == global_valid, best_value, neg_infinity());
        let global_value = simd_max(eligible_value);
        let eligible_id = select(
            (best_valid == global_valid) & (best_value == global_value),
            best_id,
            invalid_id,
        );
        let global_id = simd_min(eligible_id);
        if best_id == global_id && best_slot != invalid_id {
            chosen = chosen | (1u32 << best_slot);
        }
        if lane == 0u32 {
            store(indices_out[rank], global_id);
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{iron_logits_topk_tiled_finalize, iron_logits_topk_tiled_partial};
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn fixture(dt: DType) -> (Vec<f32>, Vec<u32>) {
        let n = 98_330usize;
        let mut source: Vec<f32> = (0..n)
            .map(|i| (((i * 73 + 29) % 65_521) as f32 - 32_760.0) * 0.000_976_562_5)
            .collect();
        source[137] = f32::NAN;
        source[41_223] = source[41_224];
        let rounded = unpack_f32(&pack_f32(&source, dt), dt);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let av = rounded[a];
            let bv = rounded[b];
            match (av.is_nan(), bv.is_nan()) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (true, true) => a.cmp(&b),
                (false, false) => bv.total_cmp(&av).then(a.cmp(&b)),
            }
        });
        (source, order[..16].iter().map(|&index| index as u32).collect())
    }

    #[test_kernel(dtypes = [bf16], tol = 0.0)]
    fn test_logits_topk_tiled_partial(dt: DType) -> TestSetup {
        let (source, _) = fixture(dt);
        let n_tiles = 64usize;
        let k = 16usize;
        let values_per_thread = source.len().div_ceil(n_tiles * 256);
        let rounded = unpack_f32(&pack_f32(&source, dt), dt);
        let stride = n_tiles * 256;
        let mut expected = Vec::with_capacity(n_tiles * k);
        for tile in 0..n_tiles {
            let mut partition = Vec::new();
            for thread in 0..256 {
                for slot in 0..values_per_thread {
                    let index = tile * 256 + thread + slot * stride;
                    if index < source.len() {
                        partition.push(index);
                    }
                }
            }
            partition.sort_by(|&a, &b| {
                let av = rounded[a];
                let bv = rounded[b];
                match (av.is_nan(), bv.is_nan()) {
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (true, true) => a.cmp(&b),
                    (false, false) => bv.total_cmp(&av).then(a.cmp(&b)),
                }
            });
            expected.extend(partition[..k].iter().map(|&index| index as u32));
        }
        TestSetup::new(iron_logits_topk_tiled_partial::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("logits", pack_f32(&source, dt), dt))
            .input(TestBuffer::zeros("candidate_indices", n_tiles * k, DType::U32))
            .constexpr("n", source.len() as u32)
            .constexpr("n_tiles", n_tiles as u32)
            .constexpr("values_per_thread", values_per_thread as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("candidate_indices", u32_bytes(&expected), DType::U32))
            .grid_3d(n_tiles as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [bf16], tol = 0.0)]
    fn test_logits_topk_tiled_finalize(dt: DType) -> TestSetup {
        let (source, expected) = fixture(dt);
        let k = 16usize;
        let mut candidates = expected.clone();
        let expected_set: std::collections::HashSet<u32> = expected.iter().copied().collect();
        for index in 0..source.len() as u32 {
            if !expected_set.contains(&index) {
                candidates.push(index);
            }
            if candidates.len() == 1_024 {
                break;
            }
        }
        let n_candidates = candidates.len();
        let values_per_thread = n_candidates.div_ceil(32);
        TestSetup::new(iron_logits_topk_tiled_finalize::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("logits", pack_f32(&source, dt), dt))
            .input(TestBuffer::from_vec("candidate_indices", u32_bytes(&candidates), DType::U32))
            .input(TestBuffer::zeros("indices_out", k, DType::U32))
            .constexpr("n_candidates", n_candidates as u32)
            .constexpr("values_per_thread", values_per_thread as u32)
            .constexpr("k", k as u32)
            .expect(TestBuffer::from_vec("indices_out", u32_bytes(&expected), DType::U32))
            .grid_3d(1, 1, 1, [32, 1, 1])
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::{iron_logits_topk_tiled_finalize, iron_logits_topk_tiled_partial};

    #[bench(dtypes = [bf16])]
    fn bench_logits_topk_tiled_partial(dt: DType) -> BenchSetup {
        let (n, n_tiles, k) = (98_330usize, 64usize, 16usize);
        let values_per_thread = n.div_ceil(n_tiles * 256);
        BenchSetup::new(iron_logits_topk_tiled_partial::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("logits", n, dt))
            .buffer(BenchBuffer::zeros("candidate_indices", n_tiles * k, DType::U32).output())
            .constexpr("n", n as u32)
            .constexpr("n_tiles", n_tiles as u32)
            .constexpr("values_per_thread", values_per_thread as u32)
            .constexpr("k", k as u32)
            .grid_3d(n_tiles as u32, 1, 1, [256, 1, 1])
            .bytes_moved((n * dt.size_bytes() + n_tiles * k * 4) as u64)
    }

    #[bench(dtypes = [bf16])]
    fn bench_logits_topk_tiled_finalize(dt: DType) -> BenchSetup {
        let (n, n_candidates, k) = (98_330usize, 1_024usize, 16usize);
        let values_per_thread = n_candidates.div_ceil(32);
        BenchSetup::new(iron_logits_topk_tiled_finalize::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("logits", n, dt))
            .buffer(BenchBuffer::zeros("candidate_indices", n_candidates, DType::U32))
            .buffer(BenchBuffer::zeros("indices_out", k, DType::U32).output())
            .constexpr("n_candidates", n_candidates as u32)
            .constexpr("values_per_thread", values_per_thread as u32)
            .constexpr("k", k as u32)
            .grid_3d(1, 1, 1, [32, 1, 1])
            .bytes_moved((n_candidates * (4 + dt.size_bytes()) + k * 4) as u64)
    }
}

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Compact K/V from padded `[n_kv, max_seq, head_dim]` → packed
//! `[n_kv, k_len, head_dim]` (first `k_len` positions).
//!
//! Used by Hy3 prefill SDPA path so MMA SDPA sees a dense K/V slab.
//! Also provides `[T,H,D] ↔ [H,T,D]` transpose helpers.

use wh_iron::kernel;

/// Compact first `k_len` positions of each KV head.
/// DISPATCH: grid threads `(n_kv, k_len * head_dim)` (2-D via program_id).
#[kernel]
pub fn iron_compact_kv<T>(
    src: Tensor<T>,
    mut dst: Tensor<T>,
    #[constexpr] n_kv: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] k_len: u32,
    #[constexpr] head_dim: u32,
) {
    let h = program_id::<0>();
    let td = program_id::<1>();
    if (h < n_kv) & (td < k_len * head_dim) {
        let t = td / head_dim;
        let d = td % head_dim;
        store(dst[(h * k_len + t) * head_dim + d], load(src[(h * max_seq + t) * head_dim + d]));
    }
}

/// Transpose `[T, H, D]` → `[H, T, D]`.
#[kernel]
pub fn iron_transpose_thd<T>(
    input: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] t_len: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] head_dim: u32,
) {
    let t = program_id::<0>();
    let hd = program_id::<1>();
    if (t < t_len) & (hd < n_heads * head_dim) {
        let h = hd / head_dim;
        let d = hd % head_dim;
        store(out[(h * t_len + t) * head_dim + d], load(input[(t * n_heads + h) * head_dim + d]));
    }
}

/// Transpose `[H, T, D]` → `[T, H, D]`.
#[kernel]
pub fn iron_transpose_htd<T>(
    input: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] t_len: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] head_dim: u32,
) {
    let t = program_id::<0>();
    let hd = program_id::<1>();
    if (t < t_len) & (hd < n_heads * head_dim) {
        let h = hd / head_dim;
        let d = hd % head_dim;
        store(out[(t * n_heads + h) * head_dim + d], load(input[(h * t_len + t) * head_dim + d]));
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{iron_compact_kv, iron_transpose_htd, iron_transpose_thd};
    use crate::utils::{pack_f32, unpack_f32};

    #[test_kernel(dtypes = [f16, f32], tol = [0.0, 0.0])]
    fn test_transpose_thd(dt: DType) -> TestSetup {
        let (t_len, n_heads, head_dim) = (4usize, 2usize, 8usize);
        let n = t_len * n_heads * head_dim;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut expect = vec![0.0f32; n];
        for t in 0..t_len {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    let in_idx = (t * n_heads + h) * head_dim + d;
                    let out_idx = (h * t_len + t) * head_dim + d;
                    expect[out_idx] = input[in_idx];
                }
            }
        }
        let _ = unpack_f32(&pack_f32(&input, dt), dt);
        TestSetup::new(iron_transpose_thd::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("t_len", t_len as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expect, dt), dt))
            .grid_3d(t_len as u32, (n_heads * head_dim) as u32, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f16, f32], tol = [0.0, 0.0])]
    fn test_transpose_htd(dt: DType) -> TestSetup {
        let (t_len, n_heads, head_dim) = (4usize, 2usize, 8usize);
        let n = t_len * n_heads * head_dim;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut expect = vec![0.0f32; n];
        for h in 0..n_heads {
            for t in 0..t_len {
                for d in 0..head_dim {
                    let in_idx = (h * t_len + t) * head_dim + d;
                    let out_idx = (t * n_heads + h) * head_dim + d;
                    expect[out_idx] = input[in_idx];
                }
            }
        }
        TestSetup::new(iron_transpose_htd::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("t_len", t_len as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expect, dt), dt))
            .grid_3d(t_len as u32, (n_heads * head_dim) as u32, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f16, f32], tol = [0.0, 0.0])]
    fn test_compact_kv(dt: DType) -> TestSetup {
        let (n_kv, max_seq, k_len, head_dim) = (2usize, 8usize, 3usize, 4usize);
        let src_n = n_kv * max_seq * head_dim;
        let dst_n = n_kv * k_len * head_dim;
        let src: Vec<f32> = (0..src_n).map(|i| i as f32 * 0.25).collect();
        let mut expect = vec![0.0f32; dst_n];
        for h in 0..n_kv {
            for t in 0..k_len {
                for d in 0..head_dim {
                    expect[(h * k_len + t) * head_dim + d] = src[(h * max_seq + t) * head_dim + d];
                }
            }
        }
        TestSetup::new(iron_compact_kv::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::zeros("dst", dst_n, dt))
            .constexpr("n_kv", n_kv as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("k_len", k_len as u32)
            .constexpr("head_dim", head_dim as u32)
            .expect(TestBuffer::from_vec("dst", pack_f32(&expect, dt), dt))
            .grid_3d(n_kv as u32, (k_len * head_dim) as u32, 1, [1, 1, 1])
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::{iron_compact_kv, iron_transpose_htd, iron_transpose_thd};

    /// Hy3 prefill shape: 8 KV heads, head_dim 128, 4K-token compact.
    #[bench(dtypes = [f16, f32])]
    fn bench_compact_kv(dt: DType) -> BenchSetup {
        let (n_kv, max_seq, k_len, head_dim) = (8usize, 4096usize, 2048usize, 128usize);
        let src_n = n_kv * max_seq * head_dim;
        let dst_n = n_kv * k_len * head_dim;
        BenchSetup::new(iron_compact_kv::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", src_n, dt))
            .buffer(BenchBuffer::zeros("dst", dst_n, dt).output())
            .constexpr("n_kv", n_kv as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("k_len", k_len as u32)
            .constexpr("head_dim", head_dim as u32)
            .grid_3d(n_kv as u32, (k_len * head_dim) as u32, 1, [1, 1, 1])
            .bytes_moved((dst_n * dt.size_bytes() * 2) as u64)
    }

    /// Hy3 prefill shape: 64 heads, head_dim 128, 2K-token prefill slice.
    #[bench(dtypes = [f16, f32])]
    fn bench_transpose_thd(dt: DType) -> BenchSetup {
        let (t_len, n_heads, head_dim) = (2048usize, 64usize, 128usize);
        let n = t_len * n_heads * head_dim;
        BenchSetup::new(iron_transpose_thd::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("t_len", t_len as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .grid_3d(t_len as u32, (n_heads * head_dim) as u32, 1, [1, 1, 1])
            .bytes_moved((n * dt.size_bytes() * 2) as u64)
    }

    #[bench(dtypes = [f16, f32])]
    fn bench_transpose_htd(dt: DType) -> BenchSetup {
        let (t_len, n_heads, head_dim) = (2048usize, 64usize, 128usize);
        let n = t_len * n_heads * head_dim;
        BenchSetup::new(iron_transpose_htd::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("t_len", t_len as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .grid_3d(t_len as u32, (n_heads * head_dim) as u32, 1, [1, 1, 1])
            .bytes_moved((n * dt.size_bytes() * 2) as u64)
    }
}

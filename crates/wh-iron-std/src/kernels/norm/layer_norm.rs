//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Layer normalization benchmark — #[kernel] DSL vs MLX metal/layer_norm.metal

use wh_iron::kernel;

#[kernel]
pub fn iron_layer_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(x[base]).cast::<f32>();
        let v1 = load(x[base + 1u32]).cast::<f32>();
        let v2 = load(x[base + 2u32]).cast::<f32>();
        let v3 = load(x[base + 3u32]).cast::<f32>();
        s = s + v0 + v1 + v2 + v3;
        sq = sq + v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        s = s + xi;
        sq = sq + xi * xi;
    }
    let st = reduce_sum(s);
    let sqt = reduce_sum(sq);
    let mean = st / n;
    // Naive two-pass E[x^2] - E[x]^2 form — catastrophic-cancellation-prone
    // for large-mean/tiny-variance rows (both terms ~mean^2, the true
    // variance is their tiny difference). Confirmed via the RST
    // oracle-coverage sweep (2026-08-07): a row of offset=1e4 + ~1e-4 noise
    // drives `var_raw` negative under real f32/f16/bf16 rounding, and
    // `rsqrt` of a negative argument produces NaN — a real production
    // hazard, not just a theoretical one. Clamp to 0 before `rsqrt`; a
    // clamped variance of exactly 0 is the correct degenerate answer for a
    // (numerically) constant row anyway.
    let var_raw = sqt / n - mean * mean;
    let var = select(var_raw > 0.0f32, var_raw, 0.0f32);
    let eps = load(eps_buf[0]);
    let is = rsqrt(var + eps);
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = (load(x[base]).cast::<f32>() - mean) * is * load(w[col]).cast::<f32>()
            + load(b[col]).cast::<f32>();
        let n1 =
            (load(x[base + 1u32]).cast::<f32>() - mean) * is * load(w[col + 1u32]).cast::<f32>()
                + load(b[col + 1u32]).cast::<f32>();
        let n2 =
            (load(x[base + 2u32]).cast::<f32>() - mean) * is * load(w[col + 2u32]).cast::<f32>()
                + load(b[col + 2u32]).cast::<f32>();
        let n3 =
            (load(x[base + 3u32]).cast::<f32>() - mean) * is * load(w[col + 3u32]).cast::<f32>()
                + load(b[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        let ci = _i - rs;
        let norm = (xi - mean) * is * load(w[ci]).cast::<f32>() + load(b[ci]).cast::<f32>();
        store(out[_i], norm.cast::<T>());
    }
}

/// New-syntax correctness for `iron_layer_norm` (Reduction mode, one threadgroup
/// per row, `tpg = n/4`). Per-row oracle on dtype-rounded inputs:
/// `out_i = (x_i - mean) / sqrt(var + eps) * w_i + b_i`.
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_layer_norm;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        setup_with_row_gen(rows, n, dt, |r, i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.03)
    }

    /// `setup`, parameterized over the per-`(row, col)` value generator so
    /// boundary-case tests (all-zero row, near-zero-variance large-offset
    /// row) can reuse the same weight/bias fixtures and oracle path.
    ///
    /// The oracle here uses the numerically STABLE one-pass-after-mean form
    /// (`var = mean((x - mean)^2)`), which does NOT reproduce the kernel's
    /// `var = sqt/n - mean*mean` catastrophic-cancellation risk — so a real
    /// divergence under a boundary case is a genuine signal, not a
    /// shared-bug false negative.
    fn setup_with_row_gen(
        rows: usize,
        n: usize,
        dt: DType,
        row_gen: impl Fn(usize, usize) -> f32,
    ) -> TestSetup {
        let eps = 1e-5f32;
        let w: Vec<f32> = (0..n).map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.02).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let w_dt = unpack_f32(&pack_f32(&w, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&b, dt), dt);
        let mut x = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows * n);
        for r in 0..rows {
            let row: Vec<f32> = (0..n).map(|i| row_gen(r, i)).collect();
            let xr = unpack_f32(&pack_f32(&row, dt), dt);
            let mean: f32 = xr.iter().sum::<f32>() / n as f32;
            let var: f32 = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..n {
                expected.push((xr[i] - mean) * inv * w_dt[i] + b_dt[i]);
            }
            x.extend_from_slice(&row);
        }
        TestSetup::new(iron_layer_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_iron_layer_norm(dt: DType) -> TestSetup { setup(4, 512, dt) }

    // ── Boundary cases (RST oracle-coverage sweep) ────────────────────────
    //
    // All-zero row: mean=0, var=0, `(0-0)*rsqrt(0+eps)*w + b = b` — finite,
    // exercises the `rsqrt(eps)` edge without any cancellation risk (both
    // terms of `sqt/n - mean*mean` are exactly 0 here).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_iron_layer_norm_all_zero_row(dt: DType) -> TestSetup {
        setup_with_row_gen(4, 512, dt, |_r, _i| 0.0)
    }

    // Near-zero-variance, large-offset row: every value is `offset + tiny
    // noise` — the regime that stresses the kernel's naive two-pass
    // `var = sqt/n - mean*mean` (both terms ~offset^2, difference ~noise^2,
    // right at the edge of dtype precision).
    //
    // Investigation (2026-08-07): with the clamp REMOVED, `var` went
    // negative — and `rsqrt(negative)` produced NaN output — at EVERY
    // offset tried (1e3, 1e4, 1e5) across f32 and bf16 (f16 additionally
    // overflows to infinity above ~6.5e4, a separate failure mode). This
    // is not a narrow edge case; it reproduces readily. `iron_layer_norm`
    // now clamps `var` to `>= 0` before `rsqrt` (see the kernel body
    // above) — this test locks in offset=1e4 as the regression guard.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_iron_layer_norm_near_zero_variance_large_offset(dt: DType) -> TestSetup {
        let offset = 1.0e4_f32;
        setup_with_row_gen(4, 512, dt, move |_r, i| offset + ((i % 5) as f32 - 2.0) * 1.0e-4)
    }
}

/// New-syntax benchmark for `iron_layer_norm` (vs MLX `metal/layer_norm.metal`).
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_layer_norm;
    use crate::utils::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    // MLX `layer_norm_looped*` buffer order: `x`[[buffer(0)]], `w`[[buffer(1)]],
    // `b`[[buffer(2)]], `out`[[buffer(3)]], `eps`(float)[[buffer(4)]],
    // `axis_size`(uint)[[buffer(5)]], `w_stride`(uint)[[buffer(6)]],
    // `b_stride`(uint)[[buffer(7)]]. x/w/b/eps_buf are shared by name with the
    // Iron inputs; w_stride=b_stride=1 (contiguous per-channel weight/bias, the
    // legacy `U32V(1)` args). `eps` reuses the Iron `eps_buf` (1e-5, F32) so both
    // sides normalise identically.
    //
    // MLX dispatch geometry: one threadgroup per row (grid=[rows,1,1]) at
    // tpg=1024 (= n/4 = Iron tpg for n=4096; legacy RowNorm `mlx_tpg: 1024`).
    // `layer_norm_looped` zero-inits its threadgroup array explicitly, so the
    // larger tpg is safe here (unlike softmax/logsumexp).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_layer_norm(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        let tn = mlx_tname(dt);
        BenchSetup::new(iron_layer_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(input_buffer("x", rows * n, dt, InputDomain::Signed))
            .buffer(input_buffer("w", n, dt, InputDomain::Positive))
            .buffer(input_buffer("b", n, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
            .with_reference(
                RefKernel::new(
                    format!("layer_norm_looped{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/layer_norm.metal")),
                )
                // x/w/b/eps_buf shared by name with the Iron inputs (placeholders).
                .buffer(BenchBuffer::zeros("x", rows * n, dt))
                .buffer(BenchBuffer::zeros("w", n, dt))
                .buffer(BenchBuffer::zeros("b", n, dt))
                .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
                .buffer(BenchBuffer::zeros("eps_buf", 1, DType::F32))
                .buffer(BenchBuffer::from_vec(
                    "axis_size",
                    (n as u32).to_le_bytes().to_vec(),
                    DType::U32,
                ))
                .buffer(BenchBuffer::from_vec("w_stride", 1u32.to_le_bytes().to_vec(), DType::U32))
                .buffer(BenchBuffer::from_vec("b_stride", 1u32.to_le_bytes().to_vec(), DType::U32))
                .grid(Grid::new_3d(rows as u32, 1, 1, [1024, 1, 1]))
                .tol(dtype_tol(dt).max(1e-4)),
            )
    }
}

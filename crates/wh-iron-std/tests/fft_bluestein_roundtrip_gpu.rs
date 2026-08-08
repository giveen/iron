//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for the assembled Bluestein chirp-Z pipeline in
//! `kernels::kv_cache::fft` — the gap the module's own doc comment names:
//!
//! > "the legacy `tests/fft_*_gpu_correctness.rs` (since removed) validated
//! > the full assembled pipeline against a DFT — these pin the individual
//! > stages."
//!
//! `fft.rs::kernel_tests` pins each Bluestein stage (`chirp_filter`,
//! `preprocess`, `cmul`, `postprocess`) against a CPU replay of that
//! stage's OWN documented chirp formula — self-consistency, not
//! independent verification (a sign-convention bug shared by the kernel
//! and its per-stage "oracle" would pass every one of those tests). This
//! file closes that gap the same way `aura_encode_decode_roundtrip_gpu.rs`
//! closed the equivalent AURA gap: dispatch every stage of the real
//! production pipeline back-to-back —
//!
//!   chirp_filter -> FFT(filter) -> preprocess -> FFT(padded) -> cmul ->
//!   IFFT(product) -> postprocess
//!
//! — at the real N=400 / N=480 (M=1024) STFT/iSTFT front-end shapes named
//! in `fft.rs`'s module doc (:165-166, :684-686), and compare the result
//! against a direct O(N^2) DFT written fresh in this file from the
//! textbook definition. That oracle shares no code with the kernel's
//! twiddle/chirp helpers OR with `fft.rs::kernel_tests::naive_dft` (which,
//! while itself independent of the *radix* kernel, lives in the same file
//! as the Bluestein stage oracles this test is specifically checking
//! against a bug shared between "the pipeline" and "the file that tests
//! it").
//!
//! ## Inverse round-trip (`ifft(fft(x)) ≈ x`) — bug found and fixed here
//!
//! A second, independently-motivated gap: `iron_fft_bluestein_preprocess`
//! and `iron_fft_bluestein_postprocess` both take an `inv` constexpr and
//! flip their chirp-angle sign accordingly (see their doc comments), but
//! `iron_fft_bluestein_chirp_filter` — the convolution kernel shared by
//! both directions — used to take NO `inv` parameter and unconditionally
//! build the positive-sign chirp `a[m] = exp(+iπm²/N)`.
//!
//! Deriving the inverse Bluestein identity the same way the module doc
//! derives the forward one: the inverse DFT
//! `x[n] = (1/N) Σ_k X[k] exp(+i2πkn/N)`, substituting
//! `kn = (k²+n²-(n-k)²)/2`, gives
//!
//!   x[n] = (1/N) exp(+iπn²/N) · Σ_k [X[k]·exp(+iπk²/N)] · exp(−iπ(n−k)²/N)
//!
//! i.e. the inverse transform's convolution kernel is
//! `exp(−iπm²/N)` — the OPPOSITE sign from the forward convolution kernel
//! `exp(+iπm²/N)` the module doc derives. Before the fix, a caller
//! assembling a full inverse Bluestein transform had no way to request
//! that opposite-sign filter — it could only reuse the always-positive-
//! sign one, which is the forward-only filter.
//!
//! **Verdict: CONFIRMED, then fixed.** Before `iron_fft_bluestein_chirp_filter`
//! gained its `inv` constexpr, `bluestein_roundtrip_inverse_n400_f32`
//! below (built from exactly the kernels available at the time — forward
//! pipeline, then an `inv=1` preprocess/postprocess pass reusing the same
//! positive-sign filter) failed with `max|Δre|=8.152e-2 max|Δim|=5.385e-2`
//! against an input signal of magnitude `max|x_re|=3.997e-2` — the
//! reconstruction error was *larger than the signal itself*, not floating-
//! point noise. `iron_fft_bluestein_chirp_filter` now takes the same
//! `inv` constexpr `preprocess`/`postprocess` already had (opposite sign
//! convention — see its doc comment in `fft.rs`), this test now builds
//! the correct per-direction filter, and the round-trip passes at the
//! same tolerance as the forward-only comparisons above.
//! `bluestein_roundtrip_inverse_reused_forward_filter_diverges_f32` below
//! keeps the pre-fix reconstruction as a permanent regression/mutation
//! pin: it deliberately reuses the wrong (`inv=0`) filter for the inverse
//! pipeline and asserts the result still diverges from `x`, so a future
//! revert of the `angle_sign` fix in `chirp_filter` is caught here too,
//! not just by the in-source `#[test_kernel]`.
//!
//! macOS-gated. Serial GPU lock (shared common::gpu_lock).

#![cfg(target_os = "macos")]

mod common;

use std::{collections::BTreeMap, f64::consts::PI};

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use wh_iron::{
    Context,
    core::{dtype::DType, ir::KernelMode},
};
use wh_iron_std::kernels::kv_cache::fft::{
    iron_fft_bluestein_chirp_filter,
    iron_fft_bluestein_cmul,
    iron_fft_bluestein_postprocess,
    iron_fft_bluestein_preprocess,
    iron_fft_n1024,
};

/// Independent O(N^2) direct-sum DFT, written fresh from the textbook
/// definition for THIS test file:
///
///   X[k] = sum_{n=0}^{N-1} x[n] * exp(-i * 2*pi * k * n / N)
///
/// `inv=true` computes the inverse (`+i` exponent, `1/N` scale) — used by
/// the round-trip test as the independent ground truth for `ifft(fft(x))`.
/// f64 accumulation (the kernel path is f32/f16/bf16 throughout) so the
/// oracle itself contributes negligible error next to the GPU pipeline's.
/// Does not call any helper from `kv_cache::fft` — no twiddle table, no
/// chirp formula, no bit-reversal, nothing shared with the kernel or with
/// `fft.rs::kernel_tests::naive_dft`.
fn naive_dft_direct(
    re: &[f32],
    im: &[f32],
    rows: usize,
    n: usize,
    inv: bool,
) -> (Vec<f32>, Vec<f32>) {
    let sign = if inv { 1.0f64 } else { -1.0f64 };
    let mut or = vec![0.0f32; rows * n];
    let mut oi = vec![0.0f32; rows * n];
    for r in 0..rows {
        for k in 0..n {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            for t in 0..n {
                let angle = sign * 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                let (c, s) = (angle.cos(), angle.sin());
                let xr = f64::from(re[r * n + t]);
                let xi = f64::from(im[r * n + t]);
                acc_re += xr * c - xi * s;
                acc_im += xr * s + xi * c;
            }
            let scale = if inv { 1.0 / n as f64 } else { 1.0 };
            or[r * n + k] = (acc_re * scale) as f32;
            oi[r * n + k] = (acc_im * scale) as f32;
        }
    }
    (or, oi)
}

/// Deterministic pseudo-random real-valued signal (matches the STFT
/// front-end's real-audio-frame use case: `in_im` all zero on input).
/// Small magnitude keeps the O(N^2) oracle and the chained 1024-point
/// FFT pipeline both well away from catastrophic cancellation.
fn real_signal(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64).wrapping_mul(seed).wrapping_add(0x9E37_79B9);
            (((x >> 8) & 0xFFFF) as f32 / 65536.0 - 0.5) * 0.08
        })
        .collect()
}

/// Run one direction of the assembled Bluestein pipeline for a
/// `[rows, n_len]` complex input `(x_re, x_im)` and return the
/// `[rows, n_len]` complex output as `(re, im)` f32.
///
/// `inv = false` is the forward DFT (`X = FFT_bluestein(x)`); `inv = true`
/// is the inverse (`x = IFFT_bluestein(X)`). `preprocess` and
/// `postprocess` both take the real `inv` constexpr and flip their chirp
/// sign accordingly.
///
/// `filter_inv` independently selects which convolution filter
/// `iron_fft_bluestein_chirp_filter` builds (its own `inv` constexpr —
/// opposite sign convention from `preprocess`/`postprocess`, see its doc
/// comment in `fft.rs`). Callers doing a real transform always pass
/// `filter_inv == inv` (the two helper wrappers below do this). The two
/// are kept as separate parameters so the regression test further down
/// can deliberately pass `filter_inv = false` while `inv = true` — i.e.
/// reuse the forward filter for an inverse pipeline, reproducing the
/// exact pre-fix bug (`iron_fft_bluestein_chirp_filter` had no `inv` and
/// always built that filter) as a permanent mutation-style pin.
///
/// `corrupt_filter_tap`, if `Some(i)`, perturbs `filter_re[i]` — one
/// twiddle-derived chirp tap from `iron_fft_bluestein_chirp_filter` —
/// after that kernel runs but before its own FFT, for the mutation-kill
/// test below.
#[allow(clippy::too_many_arguments)]
fn run_bluestein(
    ctx: &Context,
    dt: Dt,
    n_len: usize,
    m_len: usize,
    rows: usize,
    x_re: &[f32],
    x_im: &[f32],
    inv: bool,
    filter_inv: bool,
    corrupt_filter_tap: Option<usize>,
) -> (Vec<f32>, Vec<f32>) {
    let dtype = dt.to_dtype();
    let inv_u = u32::from(inv);

    // ---- stage 0: build the time-domain chirp filter a[m] (f32) --------
    // `filter_inv` selects the filter's OWN sign (opposite convention
    // from preprocess/postprocess's `inv` — see fft.rs doc comment).
    let mut fbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    fbuf.insert("filter_re".into(), pack_bytes(&vec![0.0f32; m_len], Dt::F32));
    fbuf.insert("filter_im".into(), pack_bytes(&vec![0.0f32; m_len], Dt::F32));
    fbuf.insert("n_len".into(), (n_len as u32).to_le_bytes().to_vec());
    fbuf.insert("m_len".into(), (m_len as u32).to_le_bytes().to_vec());
    fbuf.insert("inv".into(), u32::from(filter_inv).to_le_bytes().to_vec());
    let mut fkernel = iron_fft_bluestein_chirp_filter::kernel_ir_for();
    fkernel.mode = KernelMode::Grid3D;
    assert_eq!(m_len % 256, 0, "chirp_filter dispatch requires m_len % 256 == 0");
    let f_out = ctx
        .dispatch_with_grid(&fkernel, &fbuf, &BTreeMap::new(), [m_len / 256, 1, 1], [256, 1, 1])
        .expect("chirp_filter dispatch");
    let mut filter_re = unpack_bytes(f_out.outputs.get("filter_re").expect("filter_re"), Dt::F32);
    let filter_im = unpack_bytes(f_out.outputs.get("filter_im").expect("filter_im"), Dt::F32);
    if let Some(tap) = corrupt_filter_tap {
        filter_re[tap] += 10.0; // corrupt one chirp-derived twiddle tap
    }

    // ---- stage 1: FFT the filter itself (f32) -> F[k] -------------------
    // Always a genuine forward FFT of the time-domain filter — this does
    // not depend on the overall transform direction.
    let mut ffft_buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    ffft_buf.insert("in_re".into(), pack_bytes(&filter_re, Dt::F32));
    ffft_buf.insert("in_im".into(), pack_bytes(&filter_im, Dt::F32));
    ffft_buf.insert("out_re".into(), pack_bytes(&vec![0.0f32; m_len], Dt::F32));
    ffft_buf.insert("out_im".into(), pack_bytes(&vec![0.0f32; m_len], Dt::F32));
    ffft_buf.insert("inv".into(), 0u32.to_le_bytes().to_vec());
    let mut ffft_kernel = iron_fft_n1024::kernel_ir_for(DType::F32);
    ffft_kernel.mode = KernelMode::Reduction;
    let f_fft_out = ctx
        .dispatch_with_grid(&ffft_kernel, &ffft_buf, &BTreeMap::new(), [1, 1, 1], [m_len, 1, 1])
        .expect("filter FFT dispatch");
    let filt_f_re = f_fft_out.outputs.get("out_re").expect("out_re").clone();
    let filt_f_im = f_fft_out.outputs.get("out_im").expect("out_im").clone();

    // ---- stage 2: preprocess — chirp premultiply + zero-pad -------------
    let mut pbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    pbuf.insert("in_re".into(), pack_bytes(x_re, dt));
    pbuf.insert("in_im".into(), pack_bytes(x_im, dt));
    pbuf.insert("out_re".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    pbuf.insert("out_im".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    pbuf.insert("n_len".into(), (n_len as u32).to_le_bytes().to_vec());
    pbuf.insert("m_len".into(), (m_len as u32).to_le_bytes().to_vec());
    pbuf.insert("rows".into(), (rows as u32).to_le_bytes().to_vec());
    pbuf.insert("inv".into(), inv_u.to_le_bytes().to_vec());
    let mut pkernel = iron_fft_bluestein_preprocess::kernel_ir_for(dtype);
    pkernel.mode = KernelMode::Grid3D;
    let tpg = 256usize;
    let total_m = rows * m_len;
    let groups_m = total_m.div_ceil(tpg);
    let p_out = ctx
        .dispatch_with_grid(&pkernel, &pbuf, &BTreeMap::new(), [groups_m, 1, 1], [tpg, 1, 1])
        .expect("preprocess dispatch");
    let padded_re = p_out.outputs.get("out_re").expect("out_re").clone();
    let padded_im = p_out.outputs.get("out_im").expect("out_im").clone();

    // ---- stage 3: FFT the padded sequence (dtype T) -> Y[k] -------------
    // Always a genuine forward FFT — see stage 1's note.
    let mut yfft_buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    yfft_buf.insert("in_re".into(), padded_re);
    yfft_buf.insert("in_im".into(), padded_im);
    yfft_buf.insert("out_re".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    yfft_buf.insert("out_im".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    yfft_buf.insert("inv".into(), 0u32.to_le_bytes().to_vec());
    let mut yfft_kernel = iron_fft_n1024::kernel_ir_for(dtype);
    yfft_kernel.mode = KernelMode::Reduction;
    let y_out = ctx
        .dispatch_with_grid(&yfft_kernel, &yfft_buf, &BTreeMap::new(), [rows, 1, 1], [m_len, 1, 1])
        .expect("padded FFT dispatch");
    let y_re = y_out.outputs.get("out_re").expect("out_re").clone();
    let y_im = y_out.outputs.get("out_im").expect("out_im").clone();

    // ---- stage 4: elementwise complex multiply Y .* F (freq-domain conv) -
    let mut cbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    cbuf.insert("y_re".into(), y_re);
    cbuf.insert("y_im".into(), y_im);
    cbuf.insert("filter_re".into(), filt_f_re);
    cbuf.insert("filter_im".into(), filt_f_im);
    cbuf.insert("out_re".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    cbuf.insert("out_im".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    cbuf.insert("m_len".into(), (m_len as u32).to_le_bytes().to_vec());
    cbuf.insert("rows".into(), (rows as u32).to_le_bytes().to_vec());
    let mut ckernel = iron_fft_bluestein_cmul::kernel_ir_for(dtype);
    ckernel.mode = KernelMode::Grid3D;
    let c_out = ctx
        .dispatch_with_grid(&ckernel, &cbuf, &BTreeMap::new(), [groups_m, 1, 1], [tpg, 1, 1])
        .expect("cmul dispatch");
    let prod_re = c_out.outputs.get("out_re").expect("out_re").clone();
    let prod_im = c_out.outputs.get("out_im").expect("out_im").clone();

    // ---- stage 5: IFFT the product -> circular convolution (time domain) -
    // Always the radix INVERSE — this completes IFFT(FFT(a).*FFT(b)) and
    // does not depend on the overall Bluestein transform direction either.
    let mut ifft_buf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    ifft_buf.insert("in_re".into(), prod_re);
    ifft_buf.insert("in_im".into(), prod_im);
    ifft_buf.insert("out_re".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    ifft_buf.insert("out_im".into(), pack_bytes(&vec![0.0f32; rows * m_len], dt));
    ifft_buf.insert("inv".into(), 1u32.to_le_bytes().to_vec());
    let mut ifft_kernel = iron_fft_n1024::kernel_ir_for(dtype);
    ifft_kernel.mode = KernelMode::Reduction;
    let conv_out = ctx
        .dispatch_with_grid(&ifft_kernel, &ifft_buf, &BTreeMap::new(), [rows, 1, 1], [m_len, 1, 1])
        .expect("conv IFFT dispatch");
    let conv_re = conv_out.outputs.get("out_re").expect("out_re").clone();
    let conv_im = conv_out.outputs.get("out_im").expect("out_im").clone();

    // ---- stage 6: postprocess — chirp postmultiply + extract N bins -----
    let mut postbuf: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    postbuf.insert("conv_re".into(), conv_re);
    postbuf.insert("conv_im".into(), conv_im);
    postbuf.insert("out_re".into(), pack_bytes(&vec![0.0f32; rows * n_len], dt));
    postbuf.insert("out_im".into(), pack_bytes(&vec![0.0f32; rows * n_len], dt));
    postbuf.insert("n_len".into(), (n_len as u32).to_le_bytes().to_vec());
    postbuf.insert("m_len".into(), (m_len as u32).to_le_bytes().to_vec());
    postbuf.insert("rows".into(), (rows as u32).to_le_bytes().to_vec());
    postbuf.insert("inv".into(), inv_u.to_le_bytes().to_vec());
    let mut postkernel = iron_fft_bluestein_postprocess::kernel_ir_for(dtype);
    postkernel.mode = KernelMode::Grid3D;
    let total_n = rows * n_len;
    let groups_n = total_n.div_ceil(tpg);
    let post_out = ctx
        .dispatch_with_grid(&postkernel, &postbuf, &BTreeMap::new(), [groups_n, 1, 1], [tpg, 1, 1])
        .expect("postprocess dispatch");

    let out_re = unpack_bytes(post_out.outputs.get("out_re").expect("out_re"), dt);
    let out_im = unpack_bytes(post_out.outputs.get("out_im").expect("out_im"), dt);
    (out_re, out_im)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// End-to-end Bluestein pipeline vs. a direct DFT, at the real N=400
/// (M=1024) STFT-frame shape named in `fft.rs`'s module doc.
///
/// Tolerance calibration (2026-08-07, f32, Metal M5 Max): observed
/// max|Δ| (re, im) across 3 independent input seeds at N=400/rows=2 was
/// (2.4343e-4, 2.8640e-4) / (1.4094e-4, 3.0789e-4) / (2.6166e-4, 1.7333e-4)
/// — worst 3.0789e-4. CAL = ~3.2x worst-observed for headroom against
/// cross-hardware accumulation order in the chained 1024-point radix
/// FFTs, not run-to-run noise (dispatch is deterministic for a fixed
/// fixture).
#[test]
fn bluestein_roundtrip_n400_f32() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (n_len, m_len, rows) = (400usize, 1024usize, 2usize);
    let x_re = real_signal(rows * n_len, 0x1234_5678_9abc_def1);
    let x_im = vec![0.0f32; rows * n_len];
    let (got_re, got_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &x_re, &x_im, false, false, None);
    let (want_re, want_im) = naive_dft_direct(&x_re, &x_im, rows, n_len, false);

    let d_re = max_abs_diff(&got_re, &want_re);
    let d_im = max_abs_diff(&got_im, &want_im);
    eprintln!("bluestein N=400 f32: max|Δre|={d_re:.3e} max|Δim|={d_im:.3e}");
    const CAL: f32 = 1.0e-3; // ~3.2x observed worst 3.0789e-4
    assert!(d_re <= CAL, "re: max|Δ| {d_re:.3e} > {CAL:.3e}");
    assert!(d_im <= CAL, "im: max|Δ| {d_im:.3e} > {CAL:.3e}");
}

/// Same pipeline at the second production shape named in the module doc:
/// N=480 (M=1024, 2×480=960 ≤ 1024).
///
/// Tolerance calibration (2026-08-07, f32, Metal M5 Max): observed
/// max|Δ| (re, im) across 3 independent input seeds at N=480/rows=2 was
/// (7.1955e-4, 7.5018e-4) / (7.3522e-4, 5.5385e-4) / (6.5088e-4, 5.9986e-4)
/// — worst 7.5018e-4 (N=480 sits closer to M=1024's ceiling than N=400,
/// so the zero-padded region is smaller and the chirp magnitude grows
/// faster near n=N — consistent with the higher error here). CAL = ~3.3x
/// worst-observed, same headroom rationale as the N=400 case above.
#[test]
fn bluestein_roundtrip_n480_f32() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (n_len, m_len, rows) = (480usize, 1024usize, 2usize);
    let x_re = real_signal(rows * n_len, 0xfeed_face_cafe_babe);
    let x_im = vec![0.0f32; rows * n_len];
    let (got_re, got_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &x_re, &x_im, false, false, None);
    let (want_re, want_im) = naive_dft_direct(&x_re, &x_im, rows, n_len, false);

    let d_re = max_abs_diff(&got_re, &want_re);
    let d_im = max_abs_diff(&got_im, &want_im);
    eprintln!("bluestein N=480 f32: max|Δre|={d_re:.3e} max|Δim|={d_im:.3e}");
    const CAL: f32 = 2.5e-3; // ~3.3x observed worst 7.5018e-4
    assert!(d_re <= CAL, "re: max|Δ| {d_re:.3e} > {CAL:.3e}");
    assert!(d_im <= CAL, "im: max|Δ| {d_im:.3e} > {CAL:.3e}");
}

/// Mutation-kill evidence: corrupt exactly one twiddle-derived chirp tap
/// (`filter_re[3]`, produced by `iron_fft_bluestein_chirp_filter`) before
/// it flows into the rest of the pipeline, and confirm the final output
/// diverges from the clean run. Proves the N=400 comparison above has
/// teeth — that it would actually catch a wrong Bluestein twiddle, not
/// just pass by construction. `filter_re` feeds every row/frame via the
/// `cmul` broadcast, so corrupting one frequency-domain tap after the
/// filter's own FFT shows up across the whole `[rows, n_len]` output, not
/// just in one entry — the assertion checks for that broad divergence.
#[test]
fn bluestein_roundtrip_n400_filter_tap_bitflip_diverges_f32() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (n_len, m_len, rows) = (400usize, 1024usize, 2usize);
    let x_re = real_signal(rows * n_len, 0x1234_5678_9abc_def1);
    let x_im = vec![0.0f32; rows * n_len];

    let (clean_re, clean_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &x_re, &x_im, false, false, None);
    let (corrupt_re, corrupt_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &x_re, &x_im, false, false, Some(3));

    let d_re = max_abs_diff(&clean_re, &corrupt_re);
    let d_im = max_abs_diff(&clean_im, &corrupt_im);
    eprintln!("bluestein N=400 filter-tap mutation: max|Δre|={d_re:.3e} max|Δim|={d_im:.3e}");
    assert!(
        d_re > 1e-2 || d_im > 1e-2,
        "mutation check: expected the corrupted-filter-tap run to diverge from the \
         clean run (max|Δre|={d_re:.3e}, max|Δim|={d_im:.3e}) — if both are ~0, the \
         end-to-end comparison has no teeth",
    );
}

/// DECISIVE TEST — `ifft(fft(x)) ≈ x` at N=400, using the FIXED
/// `iron_fft_bluestein_chirp_filter` (now `inv`-aware — see its doc
/// comment in `fft.rs` for the sign derivation). Computes
/// `X = bluestein_forward(x)`, then `recon = bluestein_inverse(X)`,
/// building the per-direction filter each time (`filter_inv == inv` in
/// both calls below).
///
/// **Verdict: CONFIRMED, then fixed** (see the module doc comment at the
/// top of this file for the full before/after numbers). Before the fix,
/// this exact test — with `run_bluestein`'s `filter_inv` forced to
/// `false` regardless of `inv` — failed with
/// `max|Δre|=8.152e-2 max|Δim|=5.385e-2` against a signal of magnitude
/// `max|x_re|=3.997e-2`. With the fix, the round trip reconstructs `x` to
/// `max|Δre|=7.842e-7 max|Δim|=7.527e-7` — near f32 machine precision, ~5
/// orders of magnitude tighter than the `CAL` below (kept loose for
/// cross-hardware headroom, matching the forward-only tests' style).
#[test]
fn bluestein_roundtrip_inverse_n400_f32() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (n_len, m_len, rows) = (400usize, 1024usize, 2usize);
    let x_re = real_signal(rows * n_len, 0x0ddc_0ffe_e0dd_1234);
    let x_im = vec![0.0f32; rows * n_len];

    // Forward: X = FFT_bluestein(x). filter_inv=false matches inv=false.
    let (big_re, big_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &x_re, &x_im, false, false, None);

    // Inverse: recon = IFFT_bluestein(X). filter_inv=true matches inv=true
    // — the per-direction filter the fix makes buildable.
    let (recon_re, recon_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &big_re, &big_im, true, true, None);

    let d_re = max_abs_diff(&recon_re, &x_re);
    let d_im = max_abs_diff(&recon_im, &x_im); // x_im is all zero
    eprintln!(
        "bluestein INVERSE round-trip N=400 f32: max|Δre|={d_re:.3e} max|Δim|={d_im:.3e} \
         (input signal max|x_re|={:.3e})",
        x_re.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );
    const CAL: f32 = 1.0e-3; // same order as the forward-only round-trip tolerance above
    assert!(
        d_re <= CAL && d_im <= CAL,
        "inverse round-trip: max|Δre|={d_re:.3e} max|Δim|={d_im:.3e} > {CAL:.3e} — \
         ifft(fft(x)) did not reconstruct x",
    );
}

/// Regression / mutation pin for the fix above: deliberately reuse the
/// forward (`filter_inv=false`) filter for an `inv=true` (inverse)
/// pipeline — exactly what every caller was forced to do before
/// `iron_fft_bluestein_chirp_filter` gained its `inv` constexpr, and
/// exactly the bug `bluestein_roundtrip_inverse_n400_f32` above caught.
/// Asserts the reconstruction still diverges from `x` by roughly the
/// pre-fix magnitude, so if `chirp_filter`'s `angle_sign` selection is
/// ever reverted to unconditionally-positive, THIS test — not just the
/// in-source `#[test_kernel]`s — has a from-scratch independent oracle
/// (`naive_dft_direct`) far enough downstream in the pipeline to still
/// notice.
#[test]
fn bluestein_roundtrip_inverse_reused_forward_filter_diverges_f32() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (n_len, m_len, rows) = (400usize, 1024usize, 2usize);
    let x_re = real_signal(rows * n_len, 0x0ddc_0ffe_e0dd_1234);
    let x_im = vec![0.0f32; rows * n_len];

    let (big_re, big_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &x_re, &x_im, false, false, None);
    // inv=true (inverse pipeline) but filter_inv=false (wrong, reused
    // forward filter) — reproduces the pre-fix bug on purpose.
    let (recon_re, recon_im) =
        run_bluestein(&ctx, Dt::F32, n_len, m_len, rows, &big_re, &big_im, true, false, None);

    let d_re = max_abs_diff(&recon_re, &x_re);
    let d_im = max_abs_diff(&recon_im, &x_im);
    eprintln!(
        "bluestein INVERSE round-trip (wrong-filter mutation) N=400 f32: \
         max|Δre|={d_re:.3e} max|Δim|={d_im:.3e}"
    );
    // Pre-fix observed: max|Δre|=8.152e-2, max|Δim|=5.385e-2 against a
    // signal of magnitude ~4e-2. 1e-2 is well above the ~1e-3 tolerance
    // the correctly-filtered round trip meets above, so this has teeth
    // without being tied to the exact pre-fix numbers (which are only
    // deterministic modulo GPU/toolchain accumulation order).
    assert!(
        d_re > 1e-2 || d_im > 1e-2,
        "mutation check: expected reusing the forward filter for an inverse \
         transform to diverge from x (max|Δre|={d_re:.3e}, max|Δim|={d_im:.3e}) — \
         if both are small, the inv-aware chirp_filter fix has no teeth",
    );
}

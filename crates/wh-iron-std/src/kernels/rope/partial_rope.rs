//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 partial RoPE — rotates only the tail `n_rot` dims of each
//! head, leaving the leading `n_nope = head_dim - n_rot` dims
//! untouched.
//!
//! DSv4 head_dim=512 splits as nope=448 + rope=64. Q and K are
//! computed full-rank, then this kernel rotates only the last 64
//! dims of each head using the split-pair convention
//! `(dim_i, dim_i + n_rot/2)` for `dim_i ∈ [n_nope, n_nope + n_rot/2)`.
//!
//! ## Forward / inverse
//!
//! Forward rotation:  `(x, y) → (x·cos θ − y·sin θ, x·sin θ + y·cos θ)`
//! Inverse rotation:  `(x, y) → (x·cos θ + y·sin θ, −x·sin θ + y·cos θ)`
//!
//! The inverse is needed on the attention output before the grouped
//! O-LoRA — since K and V share the same tensor in MQA mode and only
//! K is logically rotated, the V-side contribution to the output
//! carries an extra rotation that has to be undone. Toggle via the
//! `inverse_flag` constexpr (`0` = forward, `1` = inverse).
//!
//! ## Dispatch
//!
//! One kernel serves both decode and prefill. `qk`/`out` are
//! `[n_tokens, n_heads, head_dim]`; Grid3D `[n_heads, half_rot, n_tokens]`,
//! one thread per `(head, rot-pair, token)`. Token `t` is roped at absolute
//! position `base_position + t`, so decode is the `n_tokens = 1` case and
//! prefill ropes all tokens in ONE dispatch (the old per-token loop cost 3N
//! tiny dispatches/layer for q + kv + inverse). Operates in-place when
//! `out == qk` — only the rope tail dims are written; the caller passes the
//! nope dims through.
//!
//! (The old single-token `iron_partial_rope` kernel was redundant — `#[constexpr]`
//! is already a runtime buffer read, so the scalar-position variant bought
//! nothing over `n_tokens = 1`.)

use wh_iron::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_partial_rope<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_nope: u32,
    #[constexpr] half_rot: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] base_position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] inverse_flag: u32,
    // YaRN (DSv4 compressed layers). Full layers pass freq_scale=1,
    // ext_factor=0 → theta = pos·inv_freq (unchanged). The DSv4 reference
    // cancels the YaRN magnitude scale (mscale=1), so it is omitted.
    #[constexpr] freq_scale: f32,
    #[constexpr] ext_factor: f32,
    #[constexpr] corr_low: f32,
    #[constexpr] corr_high: f32,
) {
    let head = program_id::<0>();
    let pair_idx = program_id::<1>();
    let token = program_id::<2>();
    let pair_f = pair_idx.cast::<f32>();
    let n_rot_f = (2u32 * half_rot).cast::<f32>();
    let inv_freq = exp2(0.0f32 - 2.0f32 * pair_f * log2(theta_base) / n_rot_f);
    // Token t is roped at absolute position base_position + t.
    let pos_f = (base_position + token).cast::<f32>();
    let theta_extrap = pos_f * inv_freq;
    let theta_interp = freq_scale * theta_extrap;
    // YaRN ramp: ramp = 1 - clamp((pair - corr_low)/(corr_high-corr_low), 0, 1).
    let denom = select((corr_high - corr_low) > 0.001f32, corr_high - corr_low, 0.001f32);
    let y_raw = (pair_f - corr_low) / denom;
    let y_lo = select(y_raw > 0.0f32, y_raw, 0.0f32);
    let y_cl = select(y_lo < 1.0f32, y_lo, 1.0f32);
    let ramp_mix = (1.0f32 - y_cl) * ext_factor;
    let theta_raw = theta_interp * (1.0f32 - ramp_mix) + theta_extrap * ramp_mix;
    // Inverse rotation: flip the sign of theta (cos stays, sin flips).
    let theta_signed = select(inverse_flag == 0u32, theta_raw, 0.0f32 - theta_raw);
    let cos_t = cos(theta_signed);
    let sin_t = sin(theta_signed);
    let tok_base = token * n_heads * head_dim;
    let head_base = tok_base + head * head_dim;
    // DSv4 partial RoPE rotates ADJACENT
    // pairs (tail[2p], tail[2p+1]) — GPT-J/interleaved convention, NOT the
    // NEOX split-half (tail[p], tail[p+half_rot]). Pairing mismatch scrambles
    // the q/k rope dims → wrong attention scores. inv_freq per pair index p
    // is identical either way; only the dim layout differs.
    let dim_lo = head_base + n_nope + 2u32 * pair_idx;
    let dim_hi = head_base + n_nope + 2u32 * pair_idx + 1u32;
    let x_lo = load(qk[dim_lo]).cast::<f32>();
    let x_hi = load(qk[dim_hi]).cast::<f32>();
    let o_lo = x_lo * cos_t - x_hi * sin_t;
    let o_hi = x_lo * sin_t + x_hi * cos_t;
    store(out[dim_lo], o_lo.cast::<T>());
    store(out[dim_hi], o_hi.cast::<T>());
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_partial_rope;
    use crate::utils::{pack_f32, unpack_f32};

    /// f32 oracle: token `t` roped at position `base_position + t`, adjacent
    /// (GPT-J) pairs over the tail `n_rot` dims. YaRN disabled (full-layer path).
    #[allow(clippy::too_many_arguments)]
    fn cpu_reference(
        qk: &[f32],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        base_position: u32,
        theta_base: f32,
        inverse: bool,
    ) -> Vec<f32> {
        let mut out = qk.to_vec();
        for token in 0..n_tokens {
            let position = (base_position as usize + token) as f32;
            for head in 0..n_heads {
                for p in 0..half_rot {
                    let inv_freq =
                        (-(p as f32) * 2.0 * theta_base.ln() / (2.0 * half_rot as f32)).exp();
                    let theta_raw = position * inv_freq;
                    let theta = if inverse { -theta_raw } else { theta_raw };
                    let c = theta.cos();
                    let s = theta.sin();
                    let tok_base = token * n_heads * head_dim;
                    let lo = tok_base + head * head_dim + n_nope + 2 * p;
                    let hi = lo + 1;
                    let x_lo = qk[lo];
                    let x_hi = qk[hi];
                    out[lo] = x_lo * c - x_hi * s;
                    out[hi] = x_lo * s + x_hi * c;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        base_position: u32,
        theta_base: f32,
        inverse: bool,
        dt: DType,
    ) -> TestSetup {
        setup_with_yarn(
            n_tokens,
            n_heads,
            head_dim,
            n_nope,
            half_rot,
            base_position,
            theta_base,
            inverse,
            // Disabled YaRN (full-layer path): freq_scale=1, ext_factor=0
            // collapses the ramp to a no-op, so `cpu_reference` (which
            // hardcodes theta = pos*inv_freq with no YaRN blend at all) is
            // still the correct oracle here.
            1.0,
            0.0,
            0.0,
            0.0,
            dt,
        )
    }

    /// `setup`, parameterized over the 4 YaRN constexprs (`freq_scale`,
    /// `ext_factor`, `corr_low`, `corr_high`) instead of hardcoding the
    /// disabled values. Still uses `cpu_reference` as its oracle — valid
    /// only when the YaRN ramp is a no-op (`ext_factor=0`, as with every
    /// existing call site). Real-value YaRN cases use `setup_yarn` below,
    /// which swaps in the independent `yarn_cpu_reference` oracle instead.
    #[allow(clippy::too_many_arguments)]
    fn setup_with_yarn(
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        base_position: u32,
        theta_base: f32,
        inverse: bool,
        freq_scale: f32,
        ext_factor: f32,
        corr_low: f32,
        corr_high: f32,
        dt: DType,
    ) -> TestSetup {
        let n = n_tokens * n_heads * head_dim;
        let qk: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 0.4).sin() * 1.2).collect();
        let qk_dt = unpack_f32(&pack_f32(&qk, dt), dt);
        let expected = cpu_reference(
            &qk_dt,
            n_tokens,
            n_heads,
            head_dim,
            n_nope,
            half_rot,
            base_position,
            theta_base,
            inverse,
        );
        let inv_flag: u32 = if inverse { 1 } else { 0 };
        // `out` buffer is initialized to a copy of `qk` so the
        // untouched nope dims pass through — kernel only writes the
        // rope pairs (the in-place pattern callers rely on).
        TestSetup::new(iron_partial_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk, dt), dt))
            .input(TestBuffer::from_vec("out", pack_f32(&qk, dt), dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_nope", n_nope as u32)
            .constexpr("half_rot", half_rot as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("base_position", base_position)
            .constexpr("theta_base", theta_base)
            .constexpr("inverse_flag", inv_flag)
            .constexpr("freq_scale", freq_scale)
            .constexpr("ext_factor", ext_factor)
            .constexpr("corr_low", corr_low)
            .constexpr("corr_high", corr_high)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_heads as u32, half_rot as u32, n_tokens as u32, [1, 1, 1])
    }

    /// Independent CPU oracle for the YaRN NTK-by-parts ramp, applied
    /// directly to `inv_freq` — per Peng et al., "YaRN: Efficient Context
    /// Window Extension of Large Language Models" (arXiv:2309.00071), the
    /// "NTK-by-parts interpolation" construction in §3.3-3.4: blend the
    /// per-dimension-pair EXTRAPOLATED and INTERPOLATED frequencies by a
    /// ramp derived from the correction range, then multiply by position.
    ///
    /// This is NOT a transliteration of the kernel's `theta`-space algebra
    /// (which blends `theta_interp`/`theta_extrap` directly, post
    /// position-multiply) — this oracle blends the FREQUENCIES first, which
    /// is the formulation the paper specifies. The two are mathematically
    /// equivalent only when `ext_factor == 1.0`: expanding the kernel's
    /// `theta_raw = theta_interp*(1-ramp_mix) + theta_extrap*ramp_mix` with
    /// `ramp_mix = (1-y_cl)*ext_factor` at `ext_factor=1` gives
    /// `theta_raw = pos * (inv_freq_interp*y_cl + inv_freq_extrap*(1-y_cl))`,
    /// which is exactly `pos * inv_freq` below with `ramp = y_cl`. Production
    /// compressed-layer YaRN always runs at `ext_factor=1.0` (DSv4 reference
    /// `yarnParams`), so this is a valid independent oracle for the
    /// real-value test cases below — it is deliberately NOT used for the
    /// disabled-value (`ext_factor=0`) tests, which stay on `cpu_reference`
    /// (at `ext_factor=0` the two formulations diverge, and `cpu_reference`'s
    /// no-YaRN-at-all form is what the kernel actually collapses to there).
    #[allow(clippy::too_many_arguments)]
    fn yarn_cpu_reference(
        qk: &[f32],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        base_position: u32,
        theta_base: f32,
        inverse: bool,
        freq_scale: f32,
        corr_low: f32,
        corr_high: f32,
    ) -> Vec<f32> {
        let mut out = qk.to_vec();
        for token in 0..n_tokens {
            let position = (base_position as usize + token) as f32;
            for head in 0..n_heads {
                for p in 0..half_rot {
                    let p_f = p as f32;
                    let inv_freq_extrap = theta_base.powf(-2.0 * p_f / (2.0 * half_rot as f32));
                    let inv_freq_interp = freq_scale * inv_freq_extrap;
                    let ramp = ((p_f - corr_low) / (corr_high - corr_low)).clamp(0.0, 1.0);
                    let inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0 - ramp);
                    let theta_raw = position * inv_freq;
                    let theta = if inverse { -theta_raw } else { theta_raw };
                    let c = theta.cos();
                    let s = theta.sin();
                    let tok_base = token * n_heads * head_dim;
                    let lo = tok_base + head * head_dim + n_nope + 2 * p;
                    let hi = lo + 1;
                    let x_lo = qk[lo];
                    let x_hi = qk[hi];
                    out[lo] = x_lo * c - x_hi * s;
                    out[hi] = x_lo * s + x_hi * c;
                }
            }
        }
        out
    }

    /// Real-value YaRN setup — uses `yarn_cpu_reference` (valid at
    /// `ext_factor=1.0`) instead of the disabled-path `cpu_reference`.
    #[allow(clippy::too_many_arguments)]
    fn setup_yarn(
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_nope: usize,
        half_rot: usize,
        base_position: u32,
        theta_base: f32,
        inverse: bool,
        freq_scale: f32,
        ext_factor: f32,
        corr_low: f32,
        corr_high: f32,
        dt: DType,
    ) -> TestSetup {
        assert_eq!(ext_factor, 1.0, "yarn_cpu_reference is only a valid oracle at ext_factor=1.0");
        let n = n_tokens * n_heads * head_dim;
        let qk: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 0.4).sin() * 1.2).collect();
        let qk_dt = unpack_f32(&pack_f32(&qk, dt), dt);
        let expected = yarn_cpu_reference(
            &qk_dt,
            n_tokens,
            n_heads,
            head_dim,
            n_nope,
            half_rot,
            base_position,
            theta_base,
            inverse,
            freq_scale,
            corr_low,
            corr_high,
        );
        let inv_flag: u32 = if inverse { 1 } else { 0 };
        TestSetup::new(iron_partial_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk, dt), dt))
            .input(TestBuffer::from_vec("out", pack_f32(&qk, dt), dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_nope", n_nope as u32)
            .constexpr("half_rot", half_rot as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("base_position", base_position)
            .constexpr("theta_base", theta_base)
            .constexpr("inverse_flag", inv_flag)
            .constexpr("freq_scale", freq_scale)
            .constexpr("ext_factor", ext_factor)
            .constexpr("corr_low", corr_low)
            .constexpr("corr_high", corr_high)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_heads as u32, half_rot as u32, n_tokens as u32, [1, 1, 1])
    }

    /// Decode (n_tokens=1): DSv4 production shape, forward, pos=64.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_decode(dt: DType) -> TestSetup {
        setup(1, 64, 512, 448, 32, 64, 10_000.0, false, dt)
    }

    /// Prefill (n_tokens=2): DSv4 production shape, forward, positions 64/65.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_dsv4_forward(dt: DType) -> TestSetup {
        setup(2, 64, 512, 448, 32, 64, 10_000.0, false, dt)
    }

    /// Prefill (n_tokens=2): DSv4 production shape, inverse, positions 64/65.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_dsv4_inverse(dt: DType) -> TestSetup {
        setup(2, 64, 512, 448, 32, 64, 10_000.0, true, dt)
    }

    /// Prefill (n_tokens=2): HCA compressed-stream theta (160K base), pos 16/17.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_partial_rope_hca(dt: DType) -> TestSetup {
        setup(2, 64, 512, 448, 32, 16, 160_000.0, false, dt)
    }

    // ── YaRN ramp — real DSv4 compressed-layer values ────────────────────
    //
    // Every test above disables the YaRN ramp (freq_scale=1, ext_factor=0),
    // so `ramp_mix` is always forced to 0 and the blend arithmetic
    // (theta_interp/theta_extrap/y_cl/ramp_mix) has never executed with real
    // values. Real production values for a DSv4 compressed layer
    // (qkRopeHeadDim=64, yarnOriginalContext=65536, compressRopeTheta=160000,
    // yarnFactor=16, betaFast=32, betaSlow=1 — see `yarnParams(ratio:)` in
    // the DSv4 reference):
    //   freq_scale = 1/yarnFactor = 0.0625
    //   ext_factor = 1.0
    //   corr_low   = floor(corrDim(betaFast)) = floor(15.4533) = 15.0
    //   corr_high  = ceil(corrDim(betaSlow))  = ceil(24.7084)  = 25.0
    // where corrDim(beta) = n_rot * ln(n_ctx / (beta*2*pi)) / (2*ln(base)).
    // Verified independently via python3 `math.log` before writing this.
    //
    // Tolerance calibration (2026-08-07): ran at f32/f16/bf16 for both
    // forward and inverse, observed max abs err: f32=3.35e-6, f16=4.88e-4,
    // bf16=2.44e-4 (bf16 forward happened to land exactly on 0.00e0 — the
    // inverse run is the binding bf16 number). tol = observed * ~3x margin:
    // [1e-5, 1.5e-3, 1e-3]. bf16 widened to 6e-3 after CI's GPU measured
    // 1.95e-3 on the same case (8x the M5 observation — same
    // cross-hardware accumulation-order spread the aura tolerances hit).
    // The freq_scale-corruption mutation check still fails at this bound
    // (corrupted error: 7.81e-3), so the oracle keeps its teeth.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1.5e-3, 6e-3])]
    fn test_partial_rope_yarn_compressed_forward(dt: DType) -> TestSetup {
        setup_yarn(2, 64, 512, 448, 32, 64, 160_000.0, false, 0.0625, 1.0, 15.0, 25.0, dt)
    }

    // Same real YaRN params, inverse rotation (the attention-output O-LoRA
    // undo path) — exercises `theta_signed` flip combined with a live ramp.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 1.5e-3, 6e-3])]
    fn test_partial_rope_yarn_compressed_inverse(dt: DType) -> TestSetup {
        setup_yarn(2, 64, 512, 448, 32, 64, 160_000.0, true, 0.0625, 1.0, 15.0, 25.0, dt)
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_partial_rope;

    /// Builds an `n_tokens`-token partial-RoPE bench at the DSv4 production shape.
    fn partial_rope_bench(dt: DType, n_tokens: usize) -> BenchSetup {
        let (n_heads, head_dim, n_nope, half_rot) = (64usize, 512usize, 448usize, 32usize);
        let n = n_tokens * n_heads * head_dim;
        BenchSetup::new(iron_partial_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_nope", n_nope as u32)
            .constexpr("half_rot", half_rot as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("base_position", 64u32)
            .constexpr("theta_base", 10_000.0_f32)
            .constexpr("inverse_flag", 0u32)
            .constexpr("freq_scale", 1.0f32)
            .constexpr("ext_factor", 0.0f32)
            .constexpr("corr_low", 0.0f32)
            .constexpr("corr_high", 0.0f32)
            .with_shape_label(format!(
                "N{n_tokens} h{n_heads} d{head_dim} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_heads as u32, half_rot as u32, n_tokens as u32, [1, 1, 1])
            .bytes_moved((4 * n_tokens * n_heads * half_rot * dt.size_bytes()) as u64)
    }

    /// Decode shape — one token.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_partial_rope_decode(dt: DType) -> BenchSetup { partial_rope_bench(dt, 1) }

    /// Prefill shape — 256 tokens in one dispatch.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_partial_rope(dt: DType) -> BenchSetup { partial_rope_bench(dt, 256) }
}

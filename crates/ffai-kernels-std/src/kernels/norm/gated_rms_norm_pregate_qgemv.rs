//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Fused **pre-gated** RMSNorm + 4-bit quantized GEMV for the Mamba-2
//! `RMSNormGated` decode tail (Mamba-2 / NemotronH / Granite-4 / FalconH1
//! SSM mixers).
//!
//! This is the SIBLING of `gated_rms_norm_qgemv.rs`, differing in ONE
//! mathematically-load-bearing way: the **gate is applied BEFORE the
//! variance is taken** (pre-gate), not after. The two conventions are
//! genuinely different functions:
//!
//!   * `gated_rms_norm_qgemv` (Qwen3.5/3.6 Gated-DeltaNet):
//!     `inner = rms_norm(y) · w · silu(z)`      — variance over `y`
//!   * THIS kernel (Mamba-2 `RMSNormGated`):
//!     `inner = rms_norm(y · silu(z)) · w`      — variance over `y·silu(z)`
//!
//! Mirrors the HF `MambaRMSNormGated.forward`:
//! ```text
//!   h = y · silu(z)
//!   h = h · rsqrt(mean(h²) + eps)
//!   inner = w · h
//! ```
//!
//! Two further geometry differences from the GDN kernel, both required by
//! the Mamba-2 norm:
//!   * **`norm_weight` is `[in_dim] = [Hv·Dv]`** (the full `d_inner`
//!     weight), indexed `[r·Dv + d]` — NOT the per-head-dim `[Dv]` weight
//!     the GDN kernel shares across heads.
//!   * **`Hv` may be odd (including `Hv = 1`).** The common Mamba-2 case
//!     is `n_groups = 1` → one normalization group over the full
//!     `d_inner`, i.e. `Hv = 1`, `Dv = d_inner`. Phase 1 assigns rows to
//!     the two simdgroups with a `< hv` guard so a single (or odd) row
//!     set is handled correctly (the GDN kernel hard-requires even `hv`).
//!
//! Phase 2 (the int4 GEMV against the staged `tg_inner`) is byte-for-byte
//! the GDN kernel's Phase 2 — the out projection is identical once the
//! gated-and-normed activation is staged.
//!
//! ## Geometry
//!
//! - **Grid: `[out_dim / 8, 1, 1]`** — one TG per 8-row output tile.
//! - **TPG = 64** (2 simdgroups × 32 lanes).
//!
//! ## DISPATCH INVARIANTS
//!
//! - `in_dim = Hv · Dv` must be a multiple of 512 (Phase-2 reads 16 X per
//!   lane × 32 lanes = 512 per block).
//! - `out_dim` must be a multiple of 8 (8-row-per-TG tiling).
//! - `group_size` must be 64 (one quant group per 4 lanes in Phase 2).
//! - `Dv` must be a multiple of 32 (one Phase-1 simdgroup per row; each of
//!   the 32 lanes owns `Dv/32` consecutive elements).
//! - `Hv` may be any value ≥ 1 (odd allowed) — the `< hv` guard covers it.
//! - **TG memory budget: `Hv · Dv · 4` bytes** of fp32 in `tg_inner`.
//!   Apple9 cap is 32 KiB, so `Hv · Dv ≤ 8192`. mamba2-130m
//!   (`Hv=1`, `Dv=1536`) is 6 KiB.
//!
//! For mamba2-130m: `Hv=1`, `Dv=1536`, `in_dim=1536`, `out_dim=768`. All
//! invariants hold.
//!
//! ## Correctness invariant
//!
//! At identical inputs (within the f32 reorder envelope of `simd_sum`),
//! this kernel equals the unfused chain:
//!
//! ```text
//!   inner = mamba2_rms_norm_gated(y, z, w, eps)      // [Hv, Dv]
//!   out   = ffai_dequant_gemv_int4(inner, Wq, S, B)  // [out_dim]
//! ```
//!
//! Pinned by the in-source `#[test_kernel]`s.

use ffai_kernels::kernel;

/// Fused pre-gated RMSNorm + int4 GEMV — 8 output rows per TG.
///
/// Phase 1 stages `inner[r, d] = w[r·Dv+d] · g · rsqrt(mean_d(g²) + eps)`
/// where `g = y[r, d] · silu(z[r, d])` into `tg_inner` (fp32). Phase 2
/// runs the int4 GEMV reading the staged activation. Grid: `[out_dim/8,
/// 1, 1]`, TPG = 64. See module doc for invariants.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_gated_rms_norm_pregate_qgemv_int4_fast<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    norm_weight: Tensor<T>,
    eps_buf: Tensor<f32>,
    q_weight: Tensor<u32>,
    q_scales: Tensor<T>,
    q_biases: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] hv: u32,
    #[constexpr] dv: u32,
    #[constexpr] out_dim: u32,
    #[constexpr] group_size: u32,
) {
    // ── Threadgroup scratch ────────────────────────────────────────────
    // 8192 = 8 KiB at fp32 — Apple9 cap is 32 KiB. Covers mamba2-130m
    // (Hv*Dv = 1536) with headroom.
    threadgroup_alloc("tg_inner", 8192, "f32");

    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;

    // ── Phase 1: pre-gated RMSNorm into `tg_inner` ─────────────────────
    //
    // `sg=0` does even rows, `sg=1` does odd rows; both stride by 2. The
    // `< hv` guard makes odd `hv` (incl. `hv == 1`, the n_groups=1 Mamba-2
    // case) correct — the GDN sibling hard-requires even `hv`. Each row is
    // owned by one simdgroup: the 32 lanes cover `Dv` with a per-lane
    // stride of `Dv/32`, so a single `simd_sum` gives the full row SSQ.
    //
    // CRITICAL: the gate is applied to `y` BEFORE the variance is taken
    // (`g = y · silu(z)`, SSQ over `g`) — the Mamba-2 `RMSNormGated`
    // convention, distinct from the GDN sibling's variance-over-`y`.
    let dv_per_lane = dv / 32u32;
    let eps = load(eps_buf[0u32]);
    let row_iters = (hv + 1u32) / 2u32;
    for r_it in range(0u32, row_iters, 1u32) {
        let r = r_it * 2u32 + sg;
        if r < hv {
            let row_base = r * dv;
            let lane_base = lane * dv_per_lane;
            // SSQ across this lane's stripe of the GATED row, in fp32.
            let mut partial_ssq = 0.0f32;
            for k in range(0u32, dv_per_lane, 1u32) {
                let idx = row_base + lane_base + k;
                let yv = load(y[idx]);
                let zv = load(z[idx]).cast::<f32>();
                let gate = zv / (1.0f32 + exp(0.0f32 - zv));
                let gv = yv * gate;
                partial_ssq = partial_ssq + gv * gv;
            }
            let row_ssq = simd_sum(partial_ssq);
            let inv_rms = rsqrt(row_ssq / dv + eps);
            // Write the gated-and-normed stripe to `tg_inner`. The full
            // `[in_dim]` norm weight is indexed at the absolute channel
            // `row_base + d` (Mamba-2 weight spans the whole d_inner).
            for k in range(0u32, dv_per_lane, 1u32) {
                let d = lane_base + k;
                let idx = row_base + d;
                let yv = load(y[idx]);
                let zv = load(z[idx]).cast::<f32>();
                let wv = load(norm_weight[idx]).cast::<f32>();
                let gate = zv / (1.0f32 + exp(0.0f32 - zv));
                let gv = yv * gate;
                let inner = gv * inv_rms * wv;
                threadgroup_store("tg_inner", idx, inner);
            }
        }
    }
    // RAW barrier: Phase 2 reads `tg_inner` filled by all lanes above.
    threadgroup_barrier();

    // ── Phase 2: 8-row int4 GEMV against `tg_inner` ────────────────────
    //
    // Byte-for-byte identical to `ffai_gated_rms_norm_qgemv_int4_fast`
    // Phase 2 — the staged `tg_inner` is the only coupling point.
    let in_dim = hv * dv;
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32; // 8 int4 values per u32
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Mask-without-shift constants — identical to `rms_norm_qgemv_fast`.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        // Pull this lane's 16-element X stripe from staged `tg_inner`.
        let n0_raw = threadgroup_load("tg_inner", xb);
        let n1_raw = threadgroup_load("tg_inner", xb + 1u32);
        let n2_raw = threadgroup_load("tg_inner", xb + 2u32);
        let n3_raw = threadgroup_load("tg_inner", xb + 3u32);
        let n4_raw = threadgroup_load("tg_inner", xb + 4u32);
        let n5_raw = threadgroup_load("tg_inner", xb + 5u32);
        let n6_raw = threadgroup_load("tg_inner", xb + 6u32);
        let n7_raw = threadgroup_load("tg_inner", xb + 7u32);
        let n8_raw = threadgroup_load("tg_inner", xb + 8u32);
        let n9_raw = threadgroup_load("tg_inner", xb + 9u32);
        let n10_raw = threadgroup_load("tg_inner", xb + 10u32);
        let n11_raw = threadgroup_load("tg_inner", xb + 11u32);
        let n12_raw = threadgroup_load("tg_inner", xb + 12u32);
        let n13_raw = threadgroup_load("tg_inner", xb + 13u32);
        let n14_raw = threadgroup_load("tg_inner", xb + 14u32);
        let n15_raw = threadgroup_load("tg_inner", xb + 15u32);
        let ns = n0_raw
            + n1_raw
            + n2_raw
            + n3_raw
            + n4_raw
            + n5_raw
            + n6_raw
            + n7_raw
            + n8_raw
            + n9_raw
            + n10_raw
            + n11_raw
            + n12_raw
            + n13_raw
            + n14_raw
            + n15_raw;
        let n1 = n1_raw * s_16;
        let n2 = n2_raw * s_256;
        let n3 = n3_raw * s_4096;
        let n5 = n5_raw * s_16;
        let n6 = n6_raw * s_256;
        let n7 = n7_raw * s_4096;
        let n9 = n9_raw * s_16;
        let n10 = n10_raw * s_256;
        let n11 = n11_raw * s_4096;
        let n13 = n13_raw * s_16;
        let n14 = n14_raw * s_256;
        let n15 = n15_raw * s_4096;
        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;
        for _r in range(0u32, 4u32, 1u32) {
            let row = base_row + _r;
            let w_base = row * packs_per_row;
            let sb_base = row * gs_per_row;
            let p_lo = load(q_weight[w_base + pack_off]);
            let p_hi_word = load(q_weight[w_base + pack_off + 1u32]);
            let p_lo_hi = p_lo >> 16u32;
            let p_hi_hi = p_hi_word >> 16u32;
            let s = load(q_scales[sb_base + g]).cast::<f32>();
            let bi = load(q_biases[sb_base + g]).cast::<f32>();
            let qd = (p_lo & 15u32).cast::<f32>() * n0_raw
                + (p_lo & 240u32).cast::<f32>() * n1
                + (p_lo & 3840u32).cast::<f32>() * n2
                + (p_lo & 61440u32).cast::<f32>() * n3
                + (p_lo_hi & 15u32).cast::<f32>() * n4_raw
                + (p_lo_hi & 240u32).cast::<f32>() * n5
                + (p_lo_hi & 3840u32).cast::<f32>() * n6
                + (p_lo_hi & 61440u32).cast::<f32>() * n7
                + (p_hi_word & 15u32).cast::<f32>() * n8_raw
                + (p_hi_word & 240u32).cast::<f32>() * n9
                + (p_hi_word & 3840u32).cast::<f32>() * n10
                + (p_hi_word & 61440u32).cast::<f32>() * n11
                + (p_hi_hi & 15u32).cast::<f32>() * n12_raw
                + (p_hi_hi & 240u32).cast::<f32>() * n13
                + (p_hi_hi & 3840u32).cast::<f32>() * n14
                + (p_hi_hi & 61440u32).cast::<f32>() * n15;
            let prev = stack_load("accs", _r);
            stack_store("accs", _r, prev + s * qd + bi * ns);
        }
    }
    // Cross-lane reduce: one simd_sum per row → one value per simdgroup.
    for _r in range(0u32, 4u32, 1u32) {
        let v = stack_load("accs", _r);
        let r = simd_sum(v);
        if lane == 0u32 {
            store(out[base_row + _r], r.cast::<T>());
        }
    }
}

mod oracle {
    use ffai_kernels::core::DType;

    use crate::utils::pack_f32;

    /// Per-row affine int4 quantize, 8 nibbles per u32 — same packing the
    /// kernel decodes. Returns (packed_weight, scales, biases) for one row.
    pub fn quantize_int4_row(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let in_dim = row.len();
        let n_groups = in_dim / group_size;
        let mut packed = vec![0u32; in_dim / 8];
        let mut scales = vec![0.0_f32; n_groups];
        let mut biases = vec![0.0_f32; n_groups];
        for g in 0..n_groups {
            let gs = &row[g * group_size..(g + 1) * group_size];
            let mn = gs.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = mx - mn;
            let scale = if range.abs() < 1e-10 { 1.0 } else { range / 15.0 };
            scales[g] = scale;
            biases[g] = mn;
            for (i, &v) in gs.iter().enumerate() {
                let q = ((v - mn) / scale).round().clamp(0.0, 15.0) as u32;
                let d = g * group_size + i;
                packed[d / 8] |= q << ((d % 8) * 4);
            }
        }
        (packed, scales, biases)
    }

    /// Deterministic xorshift source, matching the GDN sibling's generator.
    pub fn source(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20_000) as f32 / 20_000.0 - 0.5) * scale + off
            })
            .collect()
    }

    pub fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// CPU oracle: per-row PRE-gated RMSNorm (gate BEFORE variance, full
    /// `[in_dim]` weight) → int4 GEMV. Mirrors HF `MambaRMSNormGated`.
    #[allow(clippy::too_many_arguments)]
    pub fn naive(
        y: &[f32],
        z: &[f32],
        norm_weight: &[f32],
        weight: &[u32],
        scales: &[f32],
        biases: &[f32],
        hv: usize,
        dv: usize,
        out_dim: usize,
        group_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let in_dim = hv * dv;
        let mut inner = vec![0.0_f32; in_dim];
        for r in 0..hv {
            let base = r * dv;
            // Gate FIRST, then the variance is over the gated row.
            let mut gated = vec![0.0_f32; dv];
            for d in 0..dv {
                let g = z[base + d] / (1.0 + (-z[base + d]).exp());
                gated[d] = y[base + d] * g;
            }
            let ssq: f32 = gated.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (ssq / dv as f32 + eps).sqrt();
            for d in 0..dv {
                inner[base + d] = gated[d] * inv_rms * norm_weight[base + d];
            }
        }
        let u32_per_row = in_dim / 8;
        let n_groups = in_dim / group_size;
        (0..out_dim)
            .map(|row| {
                let rw = &weight[row * u32_per_row..(row + 1) * u32_per_row];
                let rs = &scales[row * n_groups..(row + 1) * n_groups];
                let rb = &biases[row * n_groups..(row + 1) * n_groups];
                let mut acc = 0.0_f32;
                for d in 0..in_dim {
                    let q = (rw[d / 8] >> ((d % 8) * 4)) & 0xf;
                    let g = d / group_size;
                    let w_real = q as f32 * rs[g] + rb[g];
                    acc += w_real * inner[d];
                }
                acc
            })
            .collect()
    }

    /// Round f32 vals through `dt` (so the CPU oracle sees the GPU's load
    /// precision) and re-pack to f32 for the oracle.
    pub fn round(v: &[f32], dt: DType) -> Vec<f32> {
        crate::utils::unpack_f32(&pack_f32(v, dt), dt)
    }
}

pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::{
        ffai_gated_rms_norm_pregate_qgemv_int4_fast,
        oracle::{naive, quantize_int4_row, round, source, u32_bytes},
    };
    use crate::utils::pack_f32;

    /// Build the test for one dtype. Constraints: `in_dim = hv*dv` a
    /// multiple of 512, `out_dim` a multiple of 8, `group_size = 64`,
    /// `dv` a multiple of 32. `hv` may be odd (incl. 1). `norm_weight` is
    /// `[in_dim]` (full d_inner), unlike the GDN sibling's `[dv]`.
    fn setup(hv: usize, dv: usize, out_dim: usize, group_size: usize, dt: DType) -> TestSetup {
        let in_dim = hv * dv;
        let eps = 1e-5_f32;
        let y: Vec<f32> = source(in_dim, 0xA1, 2.0, 0.1);
        let z: Vec<f32> = round(&source(in_dim, 0xD4, 1.5, 0.0), dt);
        // Full [in_dim] norm weight (Mamba-2 d_inner weight).
        let norm_weight: Vec<f32> = round(&source(in_dim, 0xB2, 0.4, 1.0), dt);
        let w_rows = source(out_dim * in_dim, 0xC3, 3.0, 0.0);

        let mut weight = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for row in 0..out_dim {
            let (w, s, b) =
                quantize_int4_row(&w_rows[row * in_dim..(row + 1) * in_dim], group_size);
            weight.extend(w);
            scales.extend(s);
            biases.extend(b);
        }
        let scales_r = round(&scales, dt);
        let biases_r = round(&biases, dt);

        let expected = naive(
            &y,
            &z,
            &norm_weight,
            &weight,
            &scales_r,
            &biases_r,
            hv,
            dv,
            out_dim,
            group_size,
            eps,
        );

        TestSetup::new(ffai_gated_rms_norm_pregate_qgemv_int4_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("y", pack_f32(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack_f32(&z, dt), dt))
            .input(TestBuffer::from_vec("norm_weight", pack_f32(&norm_weight, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .input(TestBuffer::from_vec("q_weight", u32_bytes(&weight), DType::U32))
            .input(TestBuffer::from_vec("q_scales", pack_f32(&scales, dt), dt))
            .input(TestBuffer::from_vec("q_biases", pack_f32(&biases, dt), dt))
            .input(TestBuffer::zeros("out", out_dim, dt))
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
    }

    // Even-hv general shape: hv=4, dv=128, in_dim=512, out_dim=512.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 3e-2, 6e-2])]
    fn test_gated_rms_norm_pregate_qgemv_int4_fast(dt: DType) -> TestSetup {
        setup(4, 128, 512, 64, dt)
    }

    // mamba2-130m shape: hv=1 (n_groups=1, odd), dv=1536 → one big
    // normalization group over the full d_inner. out_dim=768.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 3e-2, 6e-2])]
    fn test_gated_rms_norm_pregate_qgemv_mamba2_130m(dt: DType) -> TestSetup {
        setup(1, 1536, 768, 64, dt)
    }
}

/// New-syntax benchmark at the mamba2-130m production shape
/// (hv=1, dv=1536, in_dim=1536, out_dim=768).
pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_gated_rms_norm_pregate_qgemv_int4_fast;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_rms_norm_pregate_qgemv_int4_fast(dt: DType) -> BenchSetup {
        let (hv, dv, out_dim, group_size) = (1usize, 1536usize, 768usize, 64usize);
        let in_dim = hv * dv;
        let u32_per_row = in_dim / 8;
        let n_groups = in_dim / group_size;
        BenchSetup::new(ffai_gated_rms_norm_pregate_qgemv_int4_fast::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("y", in_dim, DType::F32))
            .buffer(BenchBuffer::random("z", in_dim, dt))
            .buffer(BenchBuffer::random("norm_weight", in_dim, dt))
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .buffer(BenchBuffer::random("q_weight", out_dim * u32_per_row, DType::U32))
            .buffer(BenchBuffer::random("q_scales", out_dim * n_groups, dt))
            .buffer(BenchBuffer::random("q_biases", out_dim * n_groups, dt))
            .buffer(BenchBuffer::zeros("out", out_dim, dt).output())
            .constexpr("hv", hv as u32)
            .constexpr("dv", dv as u32)
            .constexpr("out_dim", out_dim as u32)
            .constexpr("group_size", group_size as u32)
            .grid_3d((out_dim / 8) as u32, 1, 1, [64, 1, 1])
            .bytes_moved((out_dim * in_dim / 2) as u64)
            .flops(2 * out_dim as u64 * in_dim as u64)
    }
}

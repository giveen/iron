//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Batched Q4 mixture-of-experts gather projections — fuse the per-expert
//! up / down GEMVs into single big dispatches (gather the top-k selected
//! experts from one contiguous Q4 weight, run them as one [top_k*rows, k]
//! GEMV at ~90% DRAM bandwidth instead of many small ~52% ones), plus the
//! router-weighted accumulate/sum that folds the expert outputs back in.

use wh_iron::kernel;

/// Batched MoE expert UP-projection + ReLU²: gathers the `top_k` selected
/// experts (indices in `idx`) from one contiguous `[n_exp*inter, hid]` Q4 weight
/// and computes all of them in ONE big GEMV — small per-expert matrices run at
/// ~52% DRAM bandwidth, but a [top_k*inter, hid] batch runs at ~90%. `out` is
/// `[top_k*inter]`. grid = top_k*inter threadgroups.
#[kernel]
pub fn iron_moe_gather_q4_relu2<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f16>,
    x: Tensor<T>,
    idx: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] inter: u32,
    #[constexpr] rows_per_tg: u32,
) {
    // 2D grid [inter/rows_per_tg, top_k]: slot = tgid_y; `rows_per_tg` warps per
    // TG each own one inter-row (multi-warp hides global-load latency, same as
    // the dense gemv). rows_per_tg=1 is bit-identical (warp=0, lane=tid).
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let local = tgid_x * rows_per_tg + warp;
    let slot = tgid_y;
    if local < inter {
        let e = load(idx[slot]);
        let row = e * inter + local;
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f32[d_base + block]).cast::<f32>();
            let xb = block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                blk = blk
                    + (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32))
                        * load(x[xb + i]).cast::<f32>();
            }
            dot = dot + dd * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            let rr = select(total > 0.0f32, total, 0.0f32);
            store(out[slot * inter + local], (rr * rr).cast::<T>());
        }
    }
}

/// Batched MoE expert gather GEMV, PLAIN (no activation): `out[slot*inter +
/// local] = W[e_slot, local]·x` for the `top_k` experts named by `idx`. The
/// gated-FFN (SwiGLU) MoE path runs this twice (gate stack, up stack), applies
/// the activation elementwise on the `[top_k*inter]` pair, then feeds the
/// down gather. Same layout/indexing as the ReLU² variant above.
/// Scale dtype contract: per-block scales are f16 (`d_f16`), NOT f32.
#[kernel]
pub fn iron_moe_gather_q4<T>(
    qs: Tensor<u32>,
    d_f16: Tensor<f16>,
    x: Tensor<T>,
    idx: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] inter: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let local = tgid_x * rows_per_tg + warp;
    let slot = tgid_y;
    if local < inter {
        let e = load(idx[slot]);
        let row = e * inter + local;
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f16[d_base + block]).cast::<f32>();
            let xb = block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                blk = blk
                    + (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32))
                        * load(x[xb + i]).cast::<f32>();
            }
            dot = dot + dd * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            store(out[slot * inter + local], total.cast::<T>());
        }
    }
}

/// Batched MoE expert GATE+UP gather with FUSED inline SwiGLU: computes BOTH
/// the gate and up dot products against the SAME activation `x` for each
/// (slot, inter-row), then writes `silu(gate)·up` straight to `out` — one
/// dispatch instead of `iron_moe_gather_q4` run twice (gate stack, up stack)
/// plus a separate elementwise SwiGLU pass, and `x` is read once per row
/// instead of twice. `out` is `[top_k*inter]` (a single activation buffer,
/// not the old gate/up pair). Same grid/indexing scheme as `iron_moe_gather_q4`:
/// 2D grid [inter/rows_per_tg, top_k], warp per inter-row, row = e*inter+local.
/// Scale dtype contract: both `gate_d` and `up_d` are f16, NOT f32.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_moe_gather_q4_swiglu<T>(
    gate_qs: Tensor<u32>,
    gate_d: Tensor<f16>,
    up_qs: Tensor<u32>,
    up_d: Tensor<f16>,
    x: Tensor<T>,
    idx: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] inter: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let local = tgid_x * rows_per_tg + warp;
    let slot = tgid_y;
    if local < inter {
        let e = load(idx[slot]);
        let row = e * inter + local;
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let mut dot_g = 0.0f32;
        let mut dot_u = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let xb = block * 32u32 + sub * 8u32;
            let packed_g = load(gate_qs[qs_base + j]);
            let dg = load(gate_d[d_base + block]).cast::<f32>();
            let packed_u = load(up_qs[qs_base + j]);
            let du = load(up_d[d_base + block]).cast::<f32>();
            let mut blk_g = 0.0f32;
            let mut blk_u = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                // one x load, shared by both the gate and up dot products
                let xv = load(x[xb + i]).cast::<f32>();
                let nib_g = (packed_g >> (i * 4u32)) & 0xfu32;
                let nib_u = (packed_u >> (i * 4u32)) & 0xfu32;
                blk_g = blk_g + (nib_g.cast::<f32>() - select(nib_g > 7u32, 16.0f32, 0.0f32)) * xv;
                blk_u = blk_u + (nib_u.cast::<f32>() - select(nib_u > 7u32, 16.0f32, 0.0f32)) * xv;
            }
            dot_g = dot_g + dg * blk_g;
            dot_u = dot_u + du * blk_u;
        }
        let total_g = simd_sum(dot_g);
        let total_u = simd_sum(dot_u);
        if lane == 0u32 {
            // silu(gate)·up = (gate / (1 + exp(-gate))) · up, same sigmoid-via-exp
            // idiom as `iron_gated_group_rmsnorm`'s silu(z) (ssm/scan.rs).
            let silu_g = total_g / (1.0f32 + exp(0.0f32 - total_g));
            store(out[slot * inter + local], (silu_g * total_u).cast::<T>());
        }
    }
}

/// Batched MoE expert DOWN-projection + router-weighted accumulate: for each
/// output row `h`, sums the `top_k` experts' `down[e,h]·x_slot` weighted by
/// `wts[slot]`, into `acc[h]`. One dispatch for all experts. `x` is the
/// `[top_k*inter]` up-relu² output; `qs` is the contiguous `[n_exp*hid, inter]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_moe_gather_q4_down_accum<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    idx: Tensor<u32>,
    wts: Tensor<f32>,
    mut acc: Tensor<T>,
    #[constexpr] inter: u32,
    #[constexpr] hid: u32,
    #[constexpr] top_k: u32,
) {
    let h = tgid_x;
    let lane = tid;
    let bpr = inter / 32u32;
    let nwords = bpr * 4u32;
    let mut total = 0.0f32;
    for slot in range(0u32, top_k, 1u32) {
        let e = load(idx[slot]);
        let row = e * hid + h;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let xoff = slot * inter;
        let w = load(wts[slot]);
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f32[d_base + block]).cast::<f32>();
            let xb = xoff + block * 32u32 + sub * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                dot = dot
                    + dd * (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32))
                        * load(x[xb + i]).cast::<f32>();
            }
        }
        total = total + w * simd_sum(dot);
    }
    if lane == 0u32 {
        store(acc[h], (load(acc[h]).cast::<f32>() + total).cast::<T>());
    }
}

/// Batched MoE DOWN gather (no accumulate): `out[slot*hid + h] = down[e_slot, h]·
/// x_slot`, one big `[top_k*hid]` GEMV (grid top_k*hid ⇒ high occupancy, vs the
/// fused-accum variant's grid[hid] which serialized top_k experts at ~50% bw).
#[kernel]
pub fn iron_moe_gather_q4_down<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f16>,
    x: Tensor<T>,
    idx: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] inter: u32,
    #[constexpr] hid: u32,
    #[constexpr] rows_per_tg: u32,
) {
    // 2D grid [hid/rows_per_tg, top_k]: rows_per_tg warps/TG, one hid-row each
    // (multi-warp latency hiding). rows_per_tg=1 is bit-identical.
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let local = tgid_x * rows_per_tg + warp;
    let slot = tgid_y;
    if local < hid {
        let e = load(idx[slot]);
        let row = e * hid + local;
        let bpr = inter / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let xoff = slot * inter;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f32[d_base + block]).cast::<f32>();
            let xb = xoff + block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                blk = blk
                    + (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32))
                        * load(x[xb + i]).cast::<f32>();
            }
            dot = dot + dd * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            store(out[slot * hid + local], total.cast::<T>());
        }
    }
}

/// Router-weighted sum of the per-expert down outputs into `acc`:
/// `acc[h] += Σ_slot wts[slot]·downs[slot*hid + h]`. Cheap (grid hid).
#[kernel]
pub fn iron_moe_weighted_sum<T>(
    downs: Tensor<T>,
    wts: Tensor<f32>,
    mut acc: Tensor<T>,
    #[constexpr] hid: u32,
    #[constexpr] top_k: u32,
) {
    let h = program_id::<0>();
    if h < hid {
        let mut t = load(acc[h]).cast::<f32>();
        for s in range(0u32, top_k, 1u32) {
            t = t + load(wts[s]) * load(downs[s * hid + h]).cast::<f32>();
        }
        store(acc[h], t.cast::<T>());
    }
}

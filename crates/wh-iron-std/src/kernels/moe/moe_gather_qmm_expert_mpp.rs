//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//!
//! NVFP4 MoE gather QMM — **expert-aligned, A-device-direct** MPP rebuild
//! (the F-endgame kernel wave; see `Laguna Parity - Full Campaign
//! Compendium.md` §4 for the structural diff this implements).
//!
//! ## What changed vs `moe_mpp_bm64_bk64_nvfp4.rs` (and the butter EG-MPP twin)
//!
//! 1. **Expert-aligned scheduling** — grid.y is the EXPERT id, not a flat
//!    64-row M-tile. Each threadgroup binary-searches its expert's
//!    contiguous row run in the sorted `indices` (2 × 16-step lower
//!    bounds, ~32 device loads once per TG) and walks it in BM=64
//!    chunks. This deletes the per-tile sub-run scan (the old flat-tile
//!    walk probes up to 64×64 index slots per TG) and never re-stages a
//!    weight tile because a run straddles a 64-row seam.
//! 2. **A operand device-direct** — `x` is `f16` and streams straight
//!    from device memory into the cooperative tensor per SK=32 step
//!    (`coop_tile_load_a(is_tg=false)`), the same proven path as
//!    `gemm_nax_dense_bm64`. The entire X threadgroup staging slab
//!    (16 device loads + 16 TG stores + f32 mul + half cast per lane per
//!    K-tile) is gone, along with its 4 KiB of threadgroup memory.
//! 3. **W-only range fold** — with A no longer staged, the T23 bias
//!    folds into the WEIGHT side alone: staged `w' = w · 2^-14` (exactly
//!    the same staged weight magnitude as before, so no new underflow
//!    class), compensated by a single `· 2^14` at the output store.
//!    Products seen by the half-precision MMA are `|x·w·2^-14| ≤ ~131`
//!    even at the calibrated |x|~800 / outlier-row worst case — far
//!    under half's 65504 ceiling (the range-safe test below pins this).
//!    `x` itself is raw f16 (|x| ≤ 65504 — 80× above the observed ~800
//!    late-layer maximum; the caller's bf16→f16 cast should saturate).
//! 4. **Integer E2M1 decode** — the nibble select-chain is replaced by a
//!    shift decode: `v2 = ((2+(nib&1)) << ((nib&7)>>1)) >> 1` gives
//!    `2·magnitude` for every nonzero code, with the ×0.5 folded into
//!    the per-group scale. ~30% fewer ALU ops per staged weight.
//! 5. **One scale decode per lane per K-tile** — a lane's two packs are
//!    always the same row AND the same `block_size=16` scale group
//!    (`pack_id = lane·2` is even, so `pack_in_row ∈ {0,1}` or `{2,3}`,
//!    both inside one 16-element group), so the E4M3 decode + fold
//!    happens once, not twice.
//! 6. **Per-SG band skip** — each SG owns a 32-row band of the 64-row
//!    chunk. When the chunk has ≤32 valid rows (average Laguna run at
//!    T=512 is 16), the upper band's two SGs skip their A-loads and
//!    MMAs entirely (SG-uniform predicate, so `matmul2d`'s collective
//!    execution is never lane-diverged).
//!
//! Threadgroup memory: `Ws` 64×32 half (4 KiB) + `OutScratch` 4×32×32
//! f32 (16 KiB) = 20 KiB (was 24 KiB with the X slab).
//!
//! ## Contract
//!
//! - `x`: **f16**, `[m_total + 64, K]` — 64 PADDING ROWS REQUIRED past
//!   `m_total` (device-direct A tiles read whole 32-row windows; rows at
//!   or past `m_total` are masked out of the store but must be mapped).
//!   Pad contents are never accumulated into a valid output row (each
//!   C row is the dot of exactly one A row).
//! - `w`: `[n_experts, n_out, K/8]` u32 — 8 E2M1 nibbles/word, LSB-first.
//! - `scales`: `[n_experts, n_out, K/block_size]` u8 E4M3, × `global`.
//! - `indices`: `[m_total]` u32, **sorted ascending** (post sort-plan).
//! - `out`: `[m_total, n_out]` `T` — every row written exactly once.
//! - Grid `[n_out/64, n_experts, 1]`, TPG `[128, 1, 1]`
//!   (`KernelMode::Reduction`), `n_out % 64 == 0`, `block_size == 16`,
//!   `m_total ≤ 65536` (16-step binary search bound).
//! - `K` is a compile-time variant (`k2048` gate/up, `k512` down) —
//!   `coop_tile_load_a`'s `ei` stride must be a literal (Finding #3 in
//!   `gemm_nax_dense_bm64.rs`). The K-loop contains threadgroup
//!   barriers, so `UnrollPass` never unrolls it and Finding #4's
//!   device-direct redefinition trap cannot fire at any K.

use wh_iron::kernel;

/// Expert-aligned NVFP4 gather QMM, A device-direct (f16), BM=BN=64,
/// BK=32, 4 SG (2×2), per-SG band skip. `T` is the OUTPUT dtype only;
/// the A operand is concrete f16 (see module docs).
#[kernel(variants(K = [2048, 512], suffix = "k{K}"))]
#[allow(clippy::too_many_arguments)]
pub fn iron_nvfp4_moe_gather_qmm_expert_mpp<T>(
    x: Tensor<f16>,
    w: Tensor<u32>,
    scales: Tensor<u8>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] block_size: u32,
    #[constexpr] global: f32,
) {
    let expert = tgid_y;
    let n_tile_base = tgid_x * 64u32;
    let sg = simd_group_id();
    let lane_in_tg = sg * 32u32 + simd_lane;
    // 2×2 warp grid: each SG owns a 32×32 C sub-tile.
    let sg_m_base = (sg / 2u32) * 32u32;
    let sg_n_base = (sg & 1u32) * 32u32;
    let packs_per_row = K / 8u32;
    let groups_per_row = K / block_size;
    // W staging map: 64 N-rows × 4 packs = 256 packs / 128 lanes = 2
    // packs per lane; both packs share one row and one scale group.
    let pack_id0 = lane_in_tg * 2u32;
    let w_row = pack_id0 / 4u32;
    let pack_in_row0 = pack_id0 & 3u32; // 0 or 2
    let g_row = n_tile_base + w_row;
    // W-only T23 fold (module docs #3): staged w' = w · 2^-14, single
    // ×2^14 compensation at store. The extra ×0.5 folds the integer
    // E2M1 decode's 2·magnitude convention (module docs #4).
    let scale_fold = global * 0.00006103515625f32 * 0.5f32; // global · 2^-14 · 0.5
    let stage_comp = 16384.0f32; // 2^14
    threadgroup_alloc("Ws", 2048, f16); // 64 (N) × 32 (K), 4 KiB
    threadgroup_alloc("OutScratch", 4096, f32); // 4 SG × 32 × 32, 16 KiB
    coop_tile_setup("gemm", 32, 32, 32, f16, "accumulate", "simdgroup", f32, false, true, false);

    // run_start = lower_bound(indices, expert); run_end = lower_bound(
    // indices, expert + 1). 16 fixed steps cover m_total ≤ 65536; every
    // lane computes redundantly (no barrier, ~32 loads once per TG).
    let mut lo0 = 0u32;
    let mut hi0 = m_total;
    for _s0 in range(0u32, 16u32, 1u32) {
        let has = lo0 < hi0;
        let mid = (lo0 + hi0) >> 1u32;
        let v = load(indices[select(has, mid, 0u32)]);
        let go_right = has & (v < expert);
        let go_left = has & (v >= expert);
        lo0 = select(go_right, mid + 1u32, lo0);
        hi0 = select(go_left, mid, hi0);
    }
    let run_start = lo0;
    let expert_hi = expert + 1u32;
    let mut lo1 = 0u32;
    let mut hi1 = m_total;
    for _s1 in range(0u32, 16u32, 1u32) {
        let has = lo1 < hi1;
        let mid = (lo1 + hi1) >> 1u32;
        let v = load(indices[select(has, mid, 0u32)]);
        let go_right = has & (v < expert_hi);
        let go_left = has & (v >= expert_hi);
        lo1 = select(go_right, mid + 1u32, lo1);
        hi1 = select(go_left, mid, hi1);
    }
    let run_end = lo1;

    let w_expert_pack = expert * n_out * packs_per_row;
    let sb_expert_base = expert * n_out * groups_per_row;

    // Empty run → zero chunk iterations → this TG exits with no work.
    for chunk_start in range(run_start, run_end, 64u32) {
        let row_count = min(64u32, run_end - chunk_start);
        // Per-SG band skip (module docs #6): SG-uniform, so the coop
        // collective ops below are never lane-diverged.
        let sg_active = sg_m_base < row_count;
        coop_tile_zero("gemm");
        for kb in range(0u32, K, 32u32) {
            // WAR: prior iteration's B fragment reads (and, on the first
            // iteration of a later chunk, the scatter's OutScratch reads)
            // must complete before Ws is restaged.
            threadgroup_barrier();
            // --- Stage W: dequant this expert / N-tile / K-slice ---
            // One scale decode per lane (module docs #5): both packs sit
            // in `(kb + pack_in_row0·8) / block_size`'s group at
            // block_size=16.
            let k_off = kb + pack_in_row0 * 8u32;
            let sb_off = sb_expert_base + g_row * groups_per_row + k_off / block_size;
            let sraw = load(scales[sb_off]);
            let scale = iron_decode_e4m3(sraw.cast::<u32>()) * scale_fold;
            for _pi in range(0u32, 2u32, 1u32) {
                let pack_in_row = pack_in_row0 + _pi;
                let packed =
                    load(w[w_expert_pack + g_row * packs_per_row + kb / 8u32 + pack_in_row]);
                let ws_base = w_row * 32u32 + pack_in_row * 8u32;
                for _j in range(0u32, 8u32, 1u32) {
                    let nib = (packed >> (_j * 4u32)) & 15u32;
                    // Integer E2M1 decode (module docs #4): v2 = 2·|value|,
                    // the ×0.5 lives in scale_fold.
                    let m = nib & 7u32;
                    let v2 = ((2u32 + (nib & 1u32)) << (m >> 1u32)) >> 1u32;
                    let v2z = select(m == 0u32, 0u32, v2);
                    let mag = v2z.cast::<f32>();
                    let signed = select((nib & 8u32) != 0u32, -mag, mag);
                    threadgroup_store("Ws", ws_base + _j, signed * scale);
                }
            }
            threadgroup_barrier();
            if sg_active {
                // A device-direct: 32 consecutive sorted rows starting at
                // this SG's band. Rows past run_end (or m_total) are
                // garbage lanes masked at the store; x carries 64 padding
                // rows so the deepest read stays mapped (module docs).
                let a_off = (chunk_start + sg_m_base) * K + kb;
                coop_tile_load_a("gemm", "x", false, f16, K, 32u32, a_off);
                coop_tile_load_b("gemm", "Ws", true, f16, 32, 32, sg_n_base * 32u32);
                coop_tile_run("gemm");
            }
        }
        if sg_active {
            coop_tile_store_c("gemm", "OutScratch", true, f32, 32, 32, sg * 1024u32);
        }
        threadgroup_barrier();
        // Scatter 64×64 from the 2×2 SG tiles, masked to row_count valid
        // rows; ×2^14 undoes the W-side fold once.
        for _e in range(0u32, 32u32, 1u32) {
            let flat = lane_in_tg * 32u32 + _e;
            let mr = flat >> 6u32;
            let nc = flat & 63u32;
            if mr < row_count {
                let src_sg = (mr >> 5u32) * 2u32 + (nc >> 5u32);
                let v = threadgroup_load(
                    "OutScratch",
                    src_sg * 1024u32 + (mr & 31u32) * 32u32 + (nc & 31u32),
                ) * stage_comp;
                store(out[(chunk_start + mr) * n_out + n_tile_base + nc], v.cast::<T>());
            }
        }
        threadgroup_barrier();
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::{
        codegen::msl::MslGenerator,
        core::{dtype::DType, ir::Op},
    };

    use super::*;

    #[test]
    fn kernel_ir_device_direct_a_and_single_stage_slab() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt);
            assert_eq!(k.name, "iron_nvfp4_moe_gather_qmm_expert_mpp_k512");
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            // Exactly two threadgroup_allocs: Ws + OutScratch (no X slab —
            // the lever this rebuild measures).
            let tg_allocs =
                all_ops().filter(|op| matches!(op, Op::ThreadgroupAlloc { .. })).count();
            assert_eq!(tg_allocs, 2, "expected Ws + OutScratch only (no Xs)");
            // A is device-direct; B stays threadgroup-staged.
            let a_is_tg = all_ops().find_map(|op| match op {
                Op::CoopTileLoadA { is_tg, .. } => Some(*is_tg),
                _ => None,
            });
            assert_eq!(a_is_tg, Some(false), "A must be device-direct");
            let b_is_tg = all_ops().find_map(|op| match op {
                Op::CoopTileLoadB { is_tg, .. } => Some(*is_tg),
                _ => None,
            });
            assert_eq!(b_is_tg, Some(true), "B stays threadgroup-staged (dequant)");
            // Finding #3 pin: the device-direct A stride must bake to the
            // real K literal, not literal_u32's silent 0.
            let ei_a = all_ops().find_map(|op| match op {
                Op::CoopTileLoadA { ei, .. } => Some(*ei),
                _ => None,
            });
            assert_eq!(ei_a, Some(512), "coop_tile_load_a ei must bake to K");
        }
    }

    #[test]
    fn codegen_bakes_k_stride_and_half_operands() {
        let mut k = iron_nvfp4_moe_gather_qmm_expert_mpp_k2048::kernel_ir_for(DType::BF16);
        k.name = format!("{}_bf16", k.name);
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(
            msl.contains("extents<int, 2048, 32>"),
            "expected K=2048 baked into the device-direct A stride:\n{msl}"
        );
        assert!(msl.contains("threadgroup half Ws"), "Ws staging slab missing:\n{msl}");
        assert!(msl.contains("threadgroup float OutScratch"));
        assert!(
            !msl.contains("Xs"),
            "no X staging slab may exist in the A-device-direct rebuild:\n{msl}"
        );
    }
}

pub mod kernel_tests {
    use wh_iron::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        kernels::quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    pub struct Shape {
        pub n_experts: usize,
        pub m_total: usize,
        pub n_out: usize,
        pub k_in: usize,
    }

    /// Sorted multi-expert fixture with RAGGED runs: expert `e` owns
    /// `base + (e % spread)` rows (spread=0 → even runs), truncated to
    /// `m_total`. Oracle:
    /// `out[t,nc] = Σ_k x[t,k] · dequant(W)[expert(t)·n_out + nc, k]`
    /// with `x` rounded through f16 exactly as the kernel consumes it.
    pub fn setup_ex_w(
        kernel: Kernel,
        shape: Shape,
        dt: DType,
        x_scale: f32,
        spread: usize,
        w_scale: f32,
    ) -> TestSetup {
        let Shape { n_experts, m_total, n_out, k_in } = shape;
        let fmt = QFormat::Nvfp4;
        let block_size = fmt.block_size();
        assert_eq!(k_in % 32, 0);
        assert_eq!(n_out % 64, 0);

        // Sorted expert ids, ragged when spread > 0.
        let mut indices: Vec<u32> = Vec::with_capacity(m_total);
        if spread == 0 {
            assert!(m_total % n_experts == 0);
            for r in 0..m_total {
                indices.push((r / (m_total / n_experts)) as u32);
            }
        } else {
            let base = m_total / n_experts;
            'fill: for e in 0..n_experts {
                let run = if e + 1 == n_experts {
                    m_total - indices.len() // absorb remainder in last run
                } else {
                    (base + (e % spread)).saturating_sub(spread / 2)
                };
                for _ in 0..run {
                    if indices.len() == m_total {
                        break 'fill;
                    }
                    indices.push(e as u32);
                }
            }
            while indices.len() < m_total {
                indices.push((n_experts - 1) as u32);
            }
        }

        let stack_rows = n_experts * n_out;
        let stacked: Vec<f32> = (0..stack_rows * k_in)
            .map(|i| {
                let g = i / k_in;
                let e = (g / n_out) as f32;
                let r = (g % n_out) as f32;
                let c = (i % k_in) as f32;
                let mag = (0.4 + ((r + e) % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15) * w_scale;
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::kernels::quant::format::pack(fmt, &stacked, stack_rows, k_in);
        let global = p.global;
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, stack_rows, k_in);

        // x is CONSUMED AS f16 by the kernel (device-direct A); round the
        // oracle input identically. 64 padding rows (contract).
        let m_pad = m_total + 64;
        let mut x_f: Vec<f32> = vec![0.0; m_pad * k_in];
        for (i, slot) in x_f.iter_mut().take(m_total * k_in).enumerate() {
            *slot = ((i % 11) as f32 - 5.0) * 0.02 * x_scale;
        }
        let x_bytes = pack_f32(&x_f, DType::F16);
        let x = unpack_f32(&x_bytes, DType::F16);

        let mut expected = vec![0.0f32; m_total * n_out];
        for t in 0..m_total {
            let expert = indices[t] as usize;
            let base = expert * n_out;
            for nc in 0..n_out {
                let mut acc = 0.0f32;
                for kk in 0..k_in {
                    acc += x[t * k_in + kk] * wdq[(base + nc) * k_in + kk];
                }
                expected[t * n_out + nc] = acc;
            }
        }

        TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", x_bytes, DType::F16))
            .input(TestBuffer::from_vec("w", p.codes, DType::U32))
            .input(TestBuffer::from_vec("scales", p.scales, DType::U8))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("block_size", block_size as u32)
            .constexpr("global", global)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_out as u32 / 64, n_experts as u32, 1, [128, 1, 1])
    }

    fn setup(kernel: Kernel, shape: Shape, dt: DType) -> TestSetup {
        setup_ex_w(kernel, shape, dt, 1.0, 0, 1.0)
    }

    // Even runs, multi-expert, K=512 variant, 2 N-tiles.
    const SHAPE_MULTI: Shape = Shape { n_experts: 4, m_total: 64, n_out: 128, k_in: 512 };
    // Whole run in one expert (single chunk of 64).
    const SHAPE_SINGLE: Shape = Shape { n_experts: 1, m_total: 64, n_out: 64, k_in: 512 };
    // Run longer than BM → multi-chunk walk (192 rows / 1 expert).
    const SHAPE_LONG_RUN: Shape = Shape { n_experts: 2, m_total: 256, n_out: 64, k_in: 512 };
    // Ragged runs around the 64-row seam + last-expert remainder.
    const SHAPE_RAGGED: Shape = Shape { n_experts: 8, m_total: 200, n_out: 64, k_in: 512 };
    // Unit runs: 64 experts × 1 row — maximum band-skip churn.
    const SHAPE_UNIT: Shape = Shape { n_experts: 64, m_total: 64, n_out: 64, k_in: 512 };
    // Laguna gate/up geometry (K=2048 variant), small M.
    const SHAPE_LAGUNA_GATE: Shape = Shape { n_experts: 8, m_total: 64, n_out: 512, k_in: 2048 };
    // Laguna down geometry (K=512), n=2048.
    const SHAPE_LAGUNA_DOWN: Shape = Shape { n_experts: 8, m_total: 64, n_out: 2048, k_in: 512 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_gather_expert_mpp_multi(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt), SHAPE_MULTI, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_gather_expert_mpp_single(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt), SHAPE_SINGLE, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_gather_expert_mpp_long_run(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt), SHAPE_LONG_RUN, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_gather_expert_mpp_ragged(dt: DType) -> TestSetup {
        setup_ex_w(
            iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt),
            SHAPE_RAGGED,
            dt,
            1.0,
            5,
            1.0,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_gather_expert_mpp_unit_runs(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt), SHAPE_UNIT, dt)
    }

    // f32-out only for the Laguna shapes — the oracle is O(m·n·k).
    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_nvfp4_gather_expert_mpp_laguna_gate(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_expert_mpp_k2048::kernel_ir_for(dt), SHAPE_LAGUNA_GATE, dt)
    }

    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_nvfp4_gather_expert_mpp_laguna_down(dt: DType) -> TestSetup {
        setup(iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt), SHAPE_LAGUNA_DOWN, dt)
    }

    // T23-class range pin for the W-ONLY fold (module docs #3): |x|→~800
    // through raw f16 A × outlier-magnitude weight rows. Verifies the
    // staged half product chain stays finite with the fold on the weight
    // side alone. Tolerances calibrated as in
    // `moe_mpp_bm64_bk64_nvfp4.rs`'s range-safe twin (large-magnitude
    // f16/bf16 output ULP error is inherent, not a fold defect).
    const SHAPE_RANGE_SAFE: Shape = Shape { n_experts: 4, m_total: 64, n_out: 64, k_in: 512 };

    #[test_kernel(dtypes = [f16, bf16], tol = [6e2, 7e3])]
    fn test_nvfp4_gather_expert_mpp_range_safe(dt: DType) -> TestSetup {
        setup_ex_w(
            iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt),
            SHAPE_RANGE_SAFE,
            dt,
            8000.0, // x ramps to ~800 (fixture's ×0.02)
            0,
            100.0, // outlier-row stand-in
        )
    }

    // Scale-injection mutation: scaling x by a constant must scale the
    // output proportionally (catches a kernel ignoring its inputs or a
    // dead scale path — the oracle itself is recomputed, so agreement at
    // two distinct scales pins real data flow).
    #[test_kernel(dtypes = [f32], tol = [5e-2])]
    fn test_nvfp4_gather_expert_mpp_scale_injection(dt: DType) -> TestSetup {
        setup_ex_w(
            iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt),
            Shape { n_experts: 4, m_total: 64, n_out: 128, k_in: 512 },
            dt,
            7.5,
            0,
            1.0,
        )
    }
}

/// Laguna production shapes, realistic 256-expert sorted routing. Paired
/// baselines (`iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp` at identical
/// shapes/routing) are included so `iron bench nvfp4_expert` prints the
/// A/B in one run.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    fn sorted_indices(m_total: usize, n_experts: usize, ragged: bool) -> Vec<u8> {
        let base = m_total / n_experts;
        let mut v: Vec<u32> = Vec::with_capacity(m_total);
        for e in 0..n_experts {
            // Deterministic uneven runs in [base/2, base/2 + base], mean
            // ≈ base — the production routing distribution class.
            let run = if ragged { base / 2 + (e * 7919) % (base + 1) } else { base };
            for _ in 0..run {
                if v.len() < m_total {
                    v.push(e as u32);
                }
            }
        }
        while v.len() < m_total {
            v.push((n_experts - 1) as u32);
        }
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    struct B {
        m_total: usize,
        n_out: usize,
        k_in: usize,
        n_experts: usize,
        ragged: bool,
    }

    fn expert_setup(kernel: wh_iron::core::ir::Kernel, b: B, dt: DType) -> BenchSetup {
        let B { m_total, n_out, k_in, n_experts, ragged } = b;
        let block_size = QFormat::Nvfp4.block_size();
        let packs_per_row = k_in / 8;
        let codes_len = n_experts * n_out * packs_per_row;
        let n_blocks = n_experts * n_out * (k_in / block_size);
        let bytes = codes_len * 4
            + n_blocks
            + m_total * k_in * 2 // x f16
            + m_total * n_out * dt.size_bytes();
        BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", (m_total + 64) * k_in, DType::F16))
            .buffer(BenchBuffer::random("w", codes_len, DType::U32))
            .buffer(BenchBuffer::random("scales", n_blocks, DType::U8))
            .buffer(BenchBuffer::from_vec(
                "indices",
                sorted_indices(m_total, n_experts, ragged),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("block_size", block_size as u32)
            .constexpr("global", 1.0f32)
            .grid_3d(n_out as u32 / 64, n_experts as u32, 1, [128, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    fn baseline_setup(b: B, dt: DType) -> BenchSetup {
        let B { m_total, n_out, k_in, n_experts, ragged } = b;
        let block_size = QFormat::Nvfp4.block_size();
        let packs_per_row = k_in / 8;
        let codes_len = n_experts * n_out * packs_per_row;
        let n_blocks = n_experts * n_out * (k_in / block_size);
        let sz = dt.size_bytes();
        let bytes = codes_len * 4 + n_blocks + m_total * k_in * sz + m_total * n_out * sz;
        BenchSetup::new(
            crate::kernels::moe::moe_mpp_bm64_bk64_nvfp4::iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp::kernel_ir_for(dt),
        )
        .mode(KernelMode::Reduction)
        .buffer(BenchBuffer::random("x", m_total * k_in, dt))
        .buffer(BenchBuffer::random("w", codes_len, DType::U32))
        .buffer(BenchBuffer::random("scales", n_blocks, DType::U8))
        .buffer(BenchBuffer::from_vec(
            "indices",
            sorted_indices(m_total, n_experts, ragged),
            DType::U32,
        ))
        .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
        .constexpr("m_total", m_total as u32)
        .constexpr("n_out", n_out as u32)
        .constexpr("k_in", k_in as u32)
        .constexpr("block_size", block_size as u32)
        .constexpr("global", 1.0f32)
        .grid_3d(n_out as u32 / 64, (m_total as u32).div_ceil(64), 1, [128, 1, 1])
        .bytes_moved(bytes as u64)
        .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    // Laguna T=512 gate/up: m=4096, n=512, k=2048, 256 experts.
    #[bench(dtypes = [bf16])]
    fn bench_nvfp4_expert_gateup_t512(dt: DType) -> BenchSetup {
        expert_setup(
            iron_nvfp4_moe_gather_qmm_expert_mpp_k2048::kernel_ir_for(dt),
            B { m_total: 4096, n_out: 512, k_in: 2048, n_experts: 256, ragged: false },
            dt,
        )
        .with_shape_label(format!(
            "nvfp4_expert gateup M4096 N512 K2048 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    // Laguna T=512 down: m=4096, n=2048, k=512.
    #[bench(dtypes = [bf16])]
    fn bench_nvfp4_expert_down_t512(dt: DType) -> BenchSetup {
        expert_setup(
            iron_nvfp4_moe_gather_qmm_expert_mpp_k512::kernel_ir_for(dt),
            B { m_total: 4096, n_out: 2048, k_in: 512, n_experts: 256, ragged: false },
            dt,
        )
        .with_shape_label(format!(
            "nvfp4_expert down M4096 N2048 K512 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    // Ragged-tail arm: uneven runs (the production distribution class).
    #[bench(dtypes = [bf16])]
    fn bench_nvfp4_expert_gateup_ragged(dt: DType) -> BenchSetup {
        expert_setup(
            iron_nvfp4_moe_gather_qmm_expert_mpp_k2048::kernel_ir_for(dt),
            B { m_total: 4096, n_out: 512, k_in: 2048, n_experts: 256, ragged: true },
            dt,
        )
        .with_shape_label(format!(
            "nvfp4_expert gateup-ragged M4096 N512 K2048 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    // Paired baselines: the flat-M-tile BM64/BK64 kernel at the same
    // shapes/routing (bf16 x, TG-staged A — the pre-rebuild geometry).
    #[bench(dtypes = [bf16])]
    fn bench_nvfp4_expert_base_gateup_t512(dt: DType) -> BenchSetup {
        baseline_setup(
            B { m_total: 4096, n_out: 512, k_in: 2048, n_experts: 256, ragged: false },
            dt,
        )
        .with_shape_label(format!(
            "nvfp4_expert BASE gateup M4096 N512 K2048 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    #[bench(dtypes = [bf16])]
    fn bench_nvfp4_expert_base_down_t512(dt: DType) -> BenchSetup {
        baseline_setup(
            B { m_total: 4096, n_out: 2048, k_in: 512, n_experts: 256, ragged: false },
            dt,
        )
        .with_shape_label(format!(
            "nvfp4_expert BASE down M4096 N2048 K512 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }

    #[bench(dtypes = [bf16])]
    fn bench_nvfp4_expert_base_gateup_ragged(dt: DType) -> BenchSetup {
        baseline_setup(
            B { m_total: 4096, n_out: 512, k_in: 2048, n_experts: 256, ragged: true },
            dt,
        )
        .with_shape_label(format!(
            "nvfp4_expert BASE gateup-ragged M4096 N512 K2048 E256 {}",
            crate::utils::dtype_label(dt)
        ))
    }
}

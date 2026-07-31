//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MPP-backed MoE grouped block-scaled BGEMM — BM=16 / BN=32 / **BK=32** NVFP4
//! path. Doubles BK vs `moe_mpp_block_scaled` (BK=16) to cut K-loop dequant
//! rounds in half while keeping BM16's high TG count (occupancy) for Laguna
//! mid=512.
//!
//! Algorithmically the same expert sub-run walk / X-staging / coop-tile write
//! as `iron_nvfp4_moe_gather_qmm_bm16_mpp`; only the matmul2d descriptor and
//! the W/X tile footprints change for BK=32.
//!
//! | kernel                                         | element | weight | scale              |
//! |------------------------------------------------|---------|--------|--------------------|
//! | `iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp`       | E2M1    | u32    | E4M3 (u8) × global |
//!
//! ## Geometry (BM=16, BN=32, BK=32)
//!
//! - `xs`: 16×32 = 512 elements (BM × BK)
//! - `ws`: 32×32 = 1024 elements (BN × BK)
//! - `out_scratch`: 16×32 = 512 elements (BM × BN)
//! - X staging: 32 lanes × 16 elems = 512
//! - W 4-bit dequant: 32 lanes × 4 packs × 8 nibbles = 1024
//!   - `pack_id = lane*4 + _pi`; `w_row = pack_id/4`; `stripe = pack_id%4`
//!   - `k_off = kb + stripe*8`; `ws_base = w_row*32 + stripe*8`
//!   - Each 8-elem pack sits inside one NVFP4 block (`block_size = 16`), so
//!     one scale load per pack is exact across the BK=32 window (two blocks).
//!
//! ## Descriptor
//!
//! `matmul2d_descriptor(16, 32, 32, ta=false, tb=true, tc=false,
//! multiply_accumulate)` — same as BM16 but BK doubled.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/32, ceil(m_total/16), 1]`; threadgroup
//!   `[32, 1, 1]` (1 simdgroup — `matmul2d` is `execution_simdgroup`).
//! - `k_in % 32 == 0`, `n_out % 32 == 0`, `block_size` divides `k_in`, and
//!   `block_size ≥ 8` (per-pack K span of 8 sits inside one block).
//! - macOS 26+ / Metal 4; on older toolchains the codegen emits a linkable stub.

use wh_iron::kernel;

/// MPP MoE gather BGEMM (BM=16 / BN=32 / BK=32), NVFP4 only.
/// Produces `iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp`.
#[kernel(variants(
    (FMT,          BITS,  WT,  ST,  WDEC, SKIND) = [
        (nvfp4,        4u32, u32, u8,  0u32, 1u32),
    ],
    suffix = "{FMT}_moe_gather_qmm_bm16_bk32_mpp",
))]
#[allow(clippy::too_many_arguments)]
pub fn iron<T>(
    x: Tensor<T>,
    w: Tensor<WT>,
    scales: Tensor<ST>,
    indices: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] m_total: u32,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let n_tile_base = tgid_x * 32u32;
    let m_tile_base = tgid_y * 16u32;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / block_size;
    // Range-safe half staging (T23 fold, see moe_mpp_bm64_bk64_nvfp4.rs for the
    // twin). `coop_stage(T)` narrows f16/bf16 activations and weights to `half`
    // before the matmul2d MMA; real late-layer (SwiGLU-amplified) activation
    // magnitude can push an individual staged A*B product past half's ~65504
    // ceiling even though the accumulator stays f32. Folding a fixed
    // power-of-two bias into BOTH staged operands before the half narrow
    // factors out exactly across the K-reduction
    // (`sum_k((a_k*B)*(b_k*B)) == B^2 * sum_k(a_k*b_k)`), so a single
    // compensating multiply on the f32 accumulator at the output store
    // recovers the unbiased result up to ordinary half-precision rounding.
    // 2^-14 / 2^28 mirror the calibrated constants in butter's hand-written
    // NAX/NAX64 MPP gather kernels (benchmarks/laguna-xs/README.md "T23").
    let stage_bias = 0.00006103515625f32; // 2^-14
    let stage_comp = 268435456.0f32; // 2^28 == 1/(stage_bias*stage_bias)
    // xs: BM×BK = 16×32; ws: BN×BK = 32×32; out: BM×BN = 16×32
    threadgroup_alloc("xs", 512, coop_stage(T));
    threadgroup_alloc("ws", 1024, coop_stage(T));
    threadgroup_alloc("out_scratch", 512, f32);
    coop_tile_setup(
        "gemm",
        16,
        32,
        32,
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    let mut sub_offset = 0u32;
    for _sub_iter in range(0u32, 16u32, 1u32) {
        let cur_row = m_tile_base + sub_offset;
        let cur_in_range = (sub_offset < 16u32) & (cur_row < m_total);
        let cur_expert = select(cur_in_range, load(indices[cur_row]), 4294967295u32);
        let mut sub_end = 16u32;
        let mut found = 0u32;
        for _ii in range(0u32, 16u32, 1u32) {
            let probe = sub_offset + 1u32 + _ii;
            let probe_row = m_tile_base + probe;
            let probe_in_range = (probe < 16u32) & (probe_row < m_total);
            if probe_in_range & (found == 0u32) {
                let e = load(indices[probe_row]);
                if e != cur_expert {
                    sub_end = probe;
                    found = 1u32;
                }
            }
            if (probe < 16u32) & (probe_row >= m_total) & (found == 0u32) {
                sub_end = probe;
                found = 1u32;
            }
        }
        let cur_valid = (cur_expert != 4294967295u32) & (sub_offset < 16u32);
        if cur_valid {
            let w_expert_pack = cur_expert * n_out * packs_per_row;
            let sb_expert_base = cur_expert * n_out * groups_per_row;
            coop_tile_zero("gemm");
            for kb in range(0u32, k_in, 32u32) {
                // X stage: 32 lanes × 16 elems = 512 = BM×BK.
                for _e in range(0u32, 16u32, 1u32) {
                    let flat = lane * 16u32 + _e;
                    let mr = flat / 32u32;
                    let kc = flat % 32u32;
                    let gr = m_tile_base + mr;
                    let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total);
                    let safe_g = select(in_run, gr, 0u32);
                    let xv = load(x[safe_g * k_in + kb + kc]).cast::<f32>() * stage_bias;
                    threadgroup_store("xs", mr * 32u32 + kc, select(in_run, xv, 0.0f32));
                }
                // W-dequant → ws (NVFP4 E2M1 nibble): 32 lanes × 4 packs × 8.
                for _pi in range(0u32, 4u32, 1u32) {
                    let pack_id = lane * 4u32 + _pi;
                    let w_row = pack_id / 4u32;
                    let stripe = pack_id % 4u32;
                    let g_row = n_tile_base + w_row;
                    let k_off = kb + stripe * 8u32;
                    let sb_off = sb_expert_base + g_row * groups_per_row + k_off / block_size;
                    let sraw = load(scales[sb_off]);
                    // SKIND == 1: E4M3 scale × global (nvfp4). stage_bias folded
                    // in here (constant, factors out of the K-sum exactly) so
                    // the dequantized weight lands pre-biased before the half
                    // narrow, matching the X-staging fold above.
                    let scale = iron_decode_e4m3(sraw.cast::<u32>()) * global * stage_bias;
                    let packed =
                        load(w[w_expert_pack + g_row * packs_per_row + kb / 8u32 + stripe]);
                    let dst = w_row * 32u32 + stripe * 8u32;
                    for _j in range(0u32, 8u32, 1u32) {
                        let nib = (packed >> (_j * 4u32)) & 15u32;
                        threadgroup_store("ws", dst + _j, iron_decode_e2m1(nib) * scale);
                    }
                }
                threadgroup_barrier();
                // load_a: (BK, BM); load_b: (BK, BN) with B stored as [BN][BK].
                coop_tile_load_a("gemm", "xs", true, coop_stage(T), 32, 16);
                coop_tile_load_b("gemm", "ws", true, coop_stage(T), 32, 32);
                coop_tile_run("gemm");
                threadgroup_barrier();
            }
            // store_c: (BN, BM) into out_scratch laid out [BM][BN].
            coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
            threadgroup_barrier();
            for _e in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + _e;
                let mr = flat / 32u32;
                let nc = flat % 32u32;
                let gr = m_tile_base + mr;
                let gc = n_tile_base + nc;
                let in_run = (mr >= sub_offset) & (mr < sub_end) & (gr < m_total) & (gc < n_out);
                if in_run {
                    // Undo the stage_bias^2 folded into both operands above.
                    let v = threadgroup_load("out_scratch", mr * 32u32 + nc) * stage_comp;
                    store(out[gr * n_out + gc], v.cast::<T>());
                }
            }
            threadgroup_barrier();
        }
        sub_offset = sub_end;
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

    struct BlockTestShape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    /// Build a `TestSetup` for the BM16/BK32 NVFP4 MoE-MPP kernel. Same oracle
    /// as BM16/BK16 (`Σ_k x[t,k] · dequant(W_expert)[nc,k]`); only the K step
    /// and tile footprint differ on device.
    fn block_indexed_setup(
        kernel: Kernel,
        fmt: QFormat,
        shape: BlockTestShape,
        dt: DType,
    ) -> TestSetup {
        block_indexed_setup_ex(kernel, fmt, shape, dt, /* x_scale= */ 1.0, /* w_scale= */ 1.0)
    }

    /// Same as `block_indexed_setup` with activation- and weight-magnitude
    /// multipliers — `w_scale` models an outlier trained row (real NVFP4-
    /// quantized weights are not bounded to this fixture's small default
    /// magnitude; T23's actual production trigger was exactly one such
    /// outlier row), so a T23-class half-staging overflow can be reproduced
    /// with a bounded activation ramp.
    #[allow(clippy::too_many_arguments)]
    fn block_indexed_setup_ex(
        kernel: Kernel,
        fmt: QFormat,
        shape: BlockTestShape,
        dt: DType,
        x_scale: f32,
        w_scale: f32,
    ) -> TestSetup {
        let BlockTestShape { n_experts, m_total, n_out, k_in } = shape;
        let block_size = fmt.block_size();
        let stack_rows = n_experts * n_out;

        // Per-row expert indices, sorted (post-permute layout).
        let indices: Vec<u32> = (0..m_total).map(|r| (r / (m_total / n_experts)) as u32).collect();

        // Full `[n_experts·n_out, k_in]` stacked weight matrix packed once.
        let stacked: Vec<f32> = (0..stack_rows * k_in)
            .map(|i| {
                let g_row = i / k_in;
                let e = (g_row / n_out) as f32;
                let r = (g_row % n_out) as f32;
                let c = (i % k_in) as f32;
                let mag = (0.4 + ((r + e) % 7.0) * 0.1) * (0.1 + (c % 13.0) * 0.15) * w_scale;
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect();
        let p = crate::kernels::quant::format::pack(fmt, &stacked, stack_rows, k_in);
        let global = p.global;
        let wdq = crate::kernels::quant::format::dequant(fmt, &p, stack_rows, k_in);

        // Activations: dtype-rounded so the GPU sees exactly the oracle's x.
        let x_f: Vec<f32> =
            (0..m_total * k_in).map(|i| ((i % 11) as f32 - 5.0) * 0.02 * x_scale).collect();
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);

        // Oracle: out[t, nc] = Σ_k x[t, k] · dequant(W)[expert(t)·n_out + nc, k].
        let mut expected = vec![0.0f32; m_total * n_out];
        for t in 0..m_total {
            let base = indices[t] as usize * n_out;
            for nc in 0..n_out {
                let mut acc = 0.0f32;
                for kk in 0..k_in {
                    acc += x[t * k_in + kk] * wdq[(base + nc) * k_in + kk];
                }
                expected[t * n_out + nc] = acc;
            }
        }

        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };

        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::zeros("out", m_total * n_out, dt))
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_3d(
            n_out as u32 / 32,
            (m_total as u32).div_ceil(16),
            1,
            [32, 1, 1],
        )
    }

    // n_experts=4, m_total=64, n_out=64, k_in=64 (divisible by 32).
    const SHAPE: BlockTestShape = BlockTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64 };

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 2e-1])]
    fn test_nvfp4_moe_gather_qmm_bm16_bk32_mpp(dt: DType) -> TestSetup {
        block_indexed_setup(
            iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp::kernel_ir_for(dt),
            QFormat::Nvfp4,
            SHAPE,
            dt,
        )
    }

    // T23-class range-safe-staging regression pin (twin of
    // moe_mpp_bm64_bk64_nvfp4.rs's test of the same name; see
    // benchmarks/laguna-xs/README.md "T23" in butter-dev for the original
    // real-weight repro this mirrors). Real production trigger was one
    // outlier trained row (layer 38 / expert 161 / down_proj) whose per-block
    // NVFP4 scale is far larger than this fixture's small synthetic default;
    // `w_scale` stands that outlier row in. Combined with an activation ramp
    // reaching |x|~800 (matching the "at least 800" floor T23 was calibrated
    // against, 16x T22's observed ~50 cliff), the per-K-step staged half A*B
    // product (max |x|*|w| ~= 800*190 = 152000) overflows past half's ~65504
    // ceiling without the stage_bias/stage_comp fold. `f32` is excluded —
    // `coop_stage(f32) == f32`, never staged through half, so it is
    // structurally immune and stressing it at this magnitude only exercises
    // ordinary f32-accumulation rounding, not the bug this test pins.
    //
    // Tolerance is NOT the file's usual [1e-2, 5e-2, 2e-1] triple: at this
    // magnitude the *correct* (fixed) kernel's own f16/bf16 output already
    // carries a few hundred to low-thousands absolute ULP-rounding error
    // (verified empirically: measured max|Δ| against the f32 CPU oracle with
    // the fix applied, then given ~3x margin) — an unavoidable consequence of
    // f16/bf16's low relative precision at large magnitude, not a fold defect
    // (see the playbook's "empirically calibrate tolerances" rule). A
    // temporary local revert of stage_bias/stage_comp (values multiplied by
    // 1.0 instead of folded) was used to confirm this magnitude drives the
    // unfixed kernel non-finite before picking it.
    const SHAPE_RANGE_SAFE: BlockTestShape =
        BlockTestShape { n_experts: 4, m_total: 64, n_out: 64, k_in: 64 };

    #[test_kernel(dtypes = [f16, bf16], tol = [3e2, 4e3])]
    fn test_nvfp4_moe_gather_qmm_bm16_bk32_mpp_range_safe_staging(dt: DType) -> TestSetup {
        block_indexed_setup_ex(
            iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp::kernel_ir_for(dt),
            QFormat::Nvfp4,
            SHAPE_RANGE_SAFE,
            dt,
            8000.0, // x magnitude ramps to ~800 (x_f formula's *0.02 factor)
            100.0,  // weight-magnitude multiplier — outlier-row stand-in
        )
    }
}

/// Benchmarks for BM16/BK32 NVFP4 MoE MPP. Same FLOPs as BM16/BK16; fewer
/// K-loop iterations (k_in/32 vs k_in/16).
pub mod kernel_benches {
    use wh_iron::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::kernels::quant::format::QFormat;

    struct BlockBenchShape {
        n_experts: usize,
        m_total: usize,
        n_out: usize,
        k_in: usize,
    }

    fn block_bench(kernel: Kernel, fmt: QFormat, shape: BlockBenchShape, dt: DType) -> BenchSetup {
        let BlockBenchShape { n_experts, m_total, n_out, k_in } = shape;
        let block_size = fmt.block_size();
        let groups_per_row = k_in / block_size;
        let stack_n = n_experts * n_out * k_in;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (stack_n, DType::U8)
        } else {
            (
                crate::kernels::quant::format::bitstream_words(stack_n, fmt.element_bits()),
                DType::U32,
            )
        };
        let scales_dt = match fmt.scale_kind() {
            crate::kernels::quant::format::ScaleKind::F32 => DType::F32,
            crate::kernels::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = n_experts * n_out * groups_per_row;
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + m_total * k_in * sz
            + m_total * n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::zeros("indices", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("m_total", m_total as u32)
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("block_size", block_size as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_3d(n_out as u32 / 32, (m_total as u32).div_ceil(16), 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
            .with_shape_label(format!(
                "{} M{m_total} N{n_out} K{k_in} E{n_experts} {}",
                fmt.name(),
                crate::utils::dtype_label(dt)
            ))
    }

    // Laguna-ish mid shape: n_experts=8, m_total=512, n_out=4096, k_in=4096.
    const SHAPE: BlockBenchShape =
        BlockBenchShape { n_experts: 8, m_total: 512, n_out: 4096, k_in: 4096 };

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_nvfp4(dt: DType) -> BenchSetup {
        block_bench(
            iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp::kernel_ir_for(dt),
            QFormat::Nvfp4,
            SHAPE,
            dt,
        )
    }
}

//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MPP MoE grouped BGEMM, BM=16, int4 - host-precomputed EXPERT-ALIGNED
//! tile plan. `iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`.
//!
//! F-85 low-context prefill occupancy fix. The production default
//! (`iron_moe_gather_qmm_mma_int4_bm16_mpp` in `moe_mpp.rs`) ties tile
//! boundaries to a fixed M-stride (every 16 rows) independent of expert
//! boundaries, then walks up to 16 `sub_offset`/`sub_end` sub-runs inside
//! each tile, paying one FULL K-dimension `coop_tile_run` pass per
//! distinct expert the tile touches (see that kernel's dispatch-invariants
//! doc comment). At low T with realistic skewed top-k-of-many routing,
//! average per-expert row counts are close to or below the BM=16 tile
//! height, so a 16-row tile frequently straddles 2+ experts and pays 2+
//! redundant K-loop passes to produce the same 16 output rows a
//! single-expert tile needs only 1 pass for. Isolated microbench evidence
//! (`tests/smallm_occupancy_microbench.rs`, realistic top-8-of-256 skewed
//! routing, Qwen3.6-35B-A3B shapes): at T=512 (avg 16 rows/expert) skewed
//! routing costs 34-40% more than a perfectly even/tile-aligned
//! distribution at the SAME M; by T=16384 (avg 512 rows/expert) the gap is
//! under 2%. Fragmentation, not raw threadgroup count, is the loss -
//! total dispatched threadgroups for this kernel are already in the
//! thousands at T=512, far above the ~40-core device's resident capacity.
//!
//! This variant removes the fragmentation entirely by moving the tile
//! plan to the host, where the per-expert row counts already live (the
//! same counts the permute step computes to build `indices`/`x_rows`).
//! Three new parallel per-tile buffers replace the on-GPU probe:
//!
//! - `tile_expert[i]` - which expert owns tile `i`
//! - `tile_row_start[i]` - that tile's first row in the expert-sorted
//!   row space (i.e. into `x_rows`/`out`)
//! - `tile_row_count[i]` - valid row count for this tile, `1..=16`
//!
//! The host emits `ceil(count[e] / 16)` tiles per non-empty expert `e`,
//! concatenated across experts in ascending expert order (row-count
//! proportional, NOT raw-M-index proportional). Every dispatched tile is
//! therefore guaranteed single-expert by construction - the kernel does
//! exactly ONE K-loop pass per tile, no sub-run probe, no re-walk, no
//! wasted lanes beyond the ordinary `row_count < 16` tail mask any GEMM
//! tile pays. `num_tiles = sum_e ceil(count[e] / 16)` is a runtime value
//! (it depends on that forward's routing), not a `#[constexpr]` - it is
//! simply the dispatch grid's y-extent.
//!
//! Same dequant math, same `coop_tile_*` MPP path, same BM=16/BN=32/BK=16
//! geometry and TG memory budget as the sibling kernel - the only change
//! is where tile boundaries fall. Weight/scale/bias/output layout and the
//! `x_rows` gather-fusion contract are unchanged.
//!
//! ## Dispatch invariants
//!
//! - Mode `Reduction`; grid `[n_out/32, num_tiles, 1]`; threadgroup `[32, 1, 1]`.
//! - `k_in % 16 == 0`, `n_out % 32 == 0`, `group_size` divides `k_in`.
//! - `tile_row_count[i]` in `1..=16` for every `i` (host never emits an
//!   empty tile); `tile_row_start[i] + tile_row_count[i] <= m_total`.
//! - int4 only (`bits = 4` hardcoded) - scope-matched to the production
//!   default this replaces; int2/int8 siblings are a follow-up if this
//!   wins.
//! - macOS 26+ / Metal 4, same as the sibling MPP kernel.

use wh_iron::kernel;

/// MPP int4 MoE grouped BGEMM, BM=16/BN=32/BK=16, expert-aligned tile plan.
/// Params: `x [n_x_rows, k_in]`, `w [n_experts, n_out, k_in/8]` int4 packed
/// (8 nibbles/u32), `scales`/`biases [n_experts, n_out, k_in/group]`,
/// `tile_expert`/`tile_row_start`/`tile_row_count [num_tiles]` (host tile
/// plan, see module docs), `x_rows [m_total]` (gather indirection, identity
/// when `x` is pre-gathered), `out [m_total, n_out]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan<T>(
    x: Tensor<T>,
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    tile_expert: Tensor<u32>,
    tile_row_start: Tensor<u32>,
    tile_row_count: Tensor<u32>,
    x_rows: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] n_out: u32,
    #[constexpr] k_in: u32,
    #[constexpr] group_size: u32,
) {
    let n_tile_base = tgid_x * 32u32;
    let tile_idx = tgid_y;
    let lane = simd_lane;
    let packs_per_row = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    // TG memory: X tile [16 x 16] and dequant W tile [32 x 16] - identical
    // budget to the sibling sub-run kernel.
    threadgroup_alloc("xs", 256u32, coop_stage(T)); // 16 x 16
    threadgroup_alloc("ws", 512u32, coop_stage(T)); // 32 x 16
    threadgroup_alloc("out_scratch", 512u32, f32); // 16 x 32
    coop_tile_setup(
        "gemm",
        16,
        32,
        16, // m, n, k
        coop_stage(T),
        "accumulate",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );

    // Every dispatched tile is a single, real, host-verified expert run -
    // no sentinel / out-of-range probe needed, unlike the fixed-M-stride
    // sibling.
    let m_tile_base = load(tile_row_start[tile_idx]);
    let row_count = load(tile_row_count[tile_idx]);
    let expert = load(tile_expert[tile_idx]);
    let w_expert_base = expert * n_out * packs_per_row;
    let sb_expert_base = expert * n_out * groups_per_row;

    coop_tile_zero("gemm");
    for kb in range(0u32, k_in, 16u32) {
        // Stage X[m_tile_base..+row_count, kb..kb+16] → xs. 32 lanes x 8.
        for _e in range(0u32, 8u32, 1u32) {
            let flat = lane * 8u32 + _e;
            let mr = flat / 16u32;
            let kc = flat % 16u32;
            let in_run = mr < row_count;
            let gr = m_tile_base + mr;
            let safe_g = select(in_run, gr, 0u32);
            let x_row = load(x_rows[safe_g]);
            let xv = load(x[x_row * k_in + kb + kc]).cast::<f32>();
            threadgroup_store("xs", mr * 16u32 + kc, select(in_run, xv, 0.0f32));
        }
        // Dequant W[expert, n_tile_base..+32, kb..kb+16] → ws. int4: 2
        // packs/lane (16 K-cols / 8 vals-per-pack).
        let packs_per_lane = 2u32;
        for _pi in range(0u32, packs_per_lane, 1u32) {
            let pack_id = lane * packs_per_lane + _pi;
            let w_row = pack_id / packs_per_lane;
            let pack_col = pack_id % packs_per_lane;
            let pack_dev =
                w_expert_base + (n_tile_base + w_row) * packs_per_row + kb / 8u32 + pack_col;
            let packed = load(w[pack_dev]);
            let k_off = kb + pack_col * 8u32;
            let g = k_off / group_size;
            let sb_off = sb_expert_base + (n_tile_base + w_row) * groups_per_row + g;
            let s = load(scales[sb_off]).cast::<f32>();
            let b = load(biases[sb_off]).cast::<f32>();
            let dst = w_row * 16u32 + pack_col * 8u32;
            for _j in range(0u32, 8u32, 1u32) {
                let q = ((packed >> (_j * 4u32)) & 15u32).cast::<f32>();
                threadgroup_store("ws", dst + _j, s * q + b);
            }
        }
        threadgroup_barrier();
        coop_tile_load_a("gemm", "xs", true, coop_stage(T), 16, 16);
        coop_tile_load_b("gemm", "ws", true, coop_stage(T), 16, 32);
        coop_tile_run("gemm");
        threadgroup_barrier();
    }
    coop_tile_store_c("gemm", "out_scratch", true, f32, 32, 16);
    threadgroup_barrier();
    // Coop-write out_scratch → out, masked only by the ordinary tail
    // (row_count < 16 on an expert's last tile / n_out edge) - never by a
    // second expert sharing this tile, because none does.
    for _e in range(0u32, 16u32, 1u32) {
        let flat = lane * 16u32 + _e;
        let mr = flat / 32u32;
        let nc = flat % 32u32;
        let gc = n_tile_base + nc;
        let in_run = (mr < row_count) & (gc < n_out);
        if in_run {
            let gr = m_tile_base + mr;
            let v = threadgroup_load("out_scratch", mr * 32u32 + nc);
            store(out[gr * n_out + gc], v.cast::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan::kernel_ir_for(dt);
            assert_eq!(k.name, "iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan");
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileSetup { .. })));
            assert!(all_ops().any(|op| matches!(op, Op::CoopTileRun { .. })));
        }
    }
}

/// Host-side tile plan builder. Real callers derive this from the SAME
/// per-expert row counts the permute step already computes to build
/// `indices`/`x_rows` - this is a small pure function so both the Swift
/// `Ops` wrapper's logic and this crate's tests/benches share one
/// definition of "how tiles map to experts".
///
/// `counts[e]` = number of rows routed to expert `e`, contiguous in the
/// expert-sorted row space starting at `sum(counts[0..e])`. Returns
/// `(tile_expert, tile_row_start, tile_row_count)`, one entry per tile,
/// concatenated in ascending expert order. Experts with `counts[e] == 0`
/// contribute zero tiles (never an under-filled placeholder).
///
/// Thin BM=16 wrapper over `build_tile_plan_with_bm` - see that function's
/// doc comment for the general (BM-parameterized) version this and the
/// coop-core gather sibling (`moe_gather_qmm_coop.rs`, BM=32) both share.
pub fn build_tile_plan(counts: &[usize]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    build_tile_plan_with_bm(counts, 16)
}

/// General tile-plan builder, parameterized by tile height `bm`. Emits
/// `ceil(count[e] / bm)` tiles per non-empty expert `e`, concatenated in
/// ascending expert order - same algorithm as `build_tile_plan`, just with
/// the tile-height constant lifted to a parameter so a sibling GEMM at a
/// different BM (the coop-core gather kernel, BM=32) can share one
/// definition of "how tiles map to experts" instead of hand-duplicating
/// the loop with a different literal.
pub fn build_tile_plan_with_bm(counts: &[usize], bm: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    assert!(bm > 0, "tile height must be positive");
    let mut tile_expert = Vec::new();
    let mut tile_row_start = Vec::new();
    let mut tile_row_count = Vec::new();
    let mut row_base = 0usize;
    for (e, &c) in counts.iter().enumerate() {
        let mut done = 0usize;
        while done < c {
            let n = (c - done).min(bm);
            tile_expert.push(e as u32);
            tile_row_start.push((row_base + done) as u32);
            tile_row_count.push(n as u32);
            done += n;
        }
        row_base += c;
    }
    (tile_expert, tile_row_start, tile_row_count)
}

#[cfg(test)]
mod tile_plan_tests {
    use super::{build_tile_plan, build_tile_plan_with_bm};

    #[test]
    fn empty_experts_contribute_no_tiles() {
        let counts = [1usize, 0, 130, 0, 5, 40, 0, 74];
        let (te, trs, trc) = build_tile_plan(&counts);
        // ceil(1/16)+ceil(130/16)+ceil(5/16)+ceil(40/16)+ceil(74/16)
        //   = 1 + 9 + 1 + 3 + 5 = 19
        assert_eq!(te.len(), 19);
        assert_eq!(trs.len(), 19);
        assert_eq!(trc.len(), 19);
        // Every tile row-count is in 1..=16.
        assert!(trc.iter().all(|&c| (1..=16).contains(&c)));
        // Row starts are monotonically consistent with counts.
        let total_rows: u32 = trc.iter().sum();
        assert_eq!(total_rows as usize, counts.iter().sum::<usize>());
        // First tile is expert 0 at row 0.
        assert_eq!((te[0], trs[0], trc[0]), (0, 0, 1));
    }

    #[test]
    fn build_tile_plan_is_the_bm16_case_of_the_general_builder() {
        let counts = [1usize, 0, 130, 0, 5, 40, 0, 74];
        assert_eq!(build_tile_plan(&counts), build_tile_plan_with_bm(&counts, 16));
    }

    #[test]
    fn bm32_halves_tile_count_on_large_runs_and_handles_short_runs() {
        // Same fixture as `empty_experts_contribute_no_tiles`, at bm=32:
        // ceil(1/32)+ceil(130/32)+ceil(5/32)+ceil(40/32)+ceil(74/32)
        //   = 1 + 5 + 1 + 2 + 3 = 12  (vs 19 at bm=16)
        let counts = [1usize, 0, 130, 0, 5, 40, 0, 74];
        let (te, trs, trc) = build_tile_plan_with_bm(&counts, 32);
        assert_eq!(te.len(), 12);
        assert_eq!(trs.len(), 12);
        assert_eq!(trc.len(), 12);
        assert!(trc.iter().all(|&c| (1..=32).contains(&c)));
        let total_rows: u32 = trc.iter().sum();
        assert_eq!(total_rows as usize, counts.iter().sum::<usize>());
        assert_eq!((te[0], trs[0], trc[0]), (0, 0, 1));
        // Expert 2 (count=130) spans ceil(130/32)=5 tiles, last one short.
        let e2_tiles: Vec<u32> =
            te.iter().zip(&trc).filter(|&(&e, _)| e == 2).map(|(_, &c)| c).collect();
        assert_eq!(e2_tiles, vec![32, 32, 32, 32, 2]);
    }
}

/// Bench registration for the tileplan GEMM. Without this the kernel is
/// invisible to `iron build` (kernel discovery for MSL/metallib/Swift
/// emission walks the bench registry, not the raw `#[kernel]` inventory -
/// see `kernel_registry_consistency.rs`'s doc comment) even though it is
/// fully defined, IR-tested, and GPU-correctness-tested via the direct
/// `Context::dispatch_with_grid` path in
/// `tests/moe_gather_qmm_tileplan_correctness.rs`. This bench is what makes
/// `iron build --emit swift` actually generate the `Ops`-facing wrapper.
pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::{build_tile_plan, iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Qwen3.6-35B-A3B gate/up-ish shape, single-expert degenerate tile plan
    /// (mirrors the sibling `int4_mma_bench` fixture's all-zero `indices` -
    /// realistic skewed-routing throughput numbers live in
    /// `smallm_occupancy_microbench.rs`, which is what this kernel's F-85
    /// win is actually justified by; this bench only needs to exercise the
    /// kernel's dispatch shape, not reproduce that comparison).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_mma_int4_bm16_mpp_tileplan(dt: DType) -> BenchSetup {
        let (m_total, n_out, k_in, n_experts, group_size) =
            (1024usize, 512usize, 2048usize, 128usize, 64usize);
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in / 8; // int4: 8 nibbles/u32
        let sz = dt.size_bytes();
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;
        // Single expert owns every row - `build_tile_plan` still exercises
        // the real ceil(count/16) tiling, just with no fragmentation.
        let (tile_expert, tile_row_start, tile_row_count) = build_tile_plan(&[m_total]);
        let num_tiles = tile_expert.len();

        BenchSetup::new(iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::from_vec("tile_expert", u32_bytes(&tile_expert), DType::U32))
            .buffer(BenchBuffer::from_vec(
                "tile_row_start",
                u32_bytes(&tile_row_start),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "tile_row_count",
                u32_bytes(&tile_row_count),
                DType::U32,
            ))
            // Identity x_rows would be `0..m_total`; the sibling
            // `int4_mma_bench` fixture uses zeros (every row reads x_row 0)
            // since a throughput bench does not check numeric output -
            // matched here for consistency.
            .buffer(BenchBuffer::zeros("x_rows", m_total, DType::U32))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "M{m_total} N{n_out} K{k_in} E{n_experts} tiles{num_tiles} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / 32, num_tiles as u32, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    /// F-85 isolated GO/NO-GO comparison sweep for the coop-core gather
    /// variant (`moe_gather_qmm_coop.rs`): the SAME production leg shapes,
    /// target mTotal values, and `moe_mpp_shared::zipfish_counts` seeds as
    /// that kernel's `bench_moe_gather_qmm_coop_isolated_*` sweep, run
    /// against THIS kernel (the BM=16 production default) so `iron bench
    /// --match-name isolated` reports both sides of the comparison at
    /// identical routing skew. Not the production shape-smoke bench above
    /// (`bench_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`, single-expert,
    /// dispatch-shape-only) - this is the actual A/B fixture.
    #[allow(clippy::too_many_arguments)]
    fn isolated_cmp_bench(
        dt: DType,
        m_total: usize,
        n_out: usize,
        k_in: usize,
        n_experts: usize,
        group_size: usize,
        seed: u64,
        label: &str,
    ) -> BenchSetup {
        let groups_per_row = k_in / group_size;
        let words_per_row = k_in / 8; // int4: 8 nibbles/u32
        let sz = dt.size_bytes();
        let bytes = n_experts * n_out * words_per_row * 4
            + 2 * n_experts * n_out * groups_per_row * sz
            + m_total * k_in * sz
            + m_total * n_out * sz;
        let counts = crate::kernels::moe::moe_mpp_shared::zipfish_counts(m_total, n_experts, seed);
        let (tile_expert, tile_row_start, tile_row_count) = build_tile_plan(&counts);
        let num_tiles = tile_expert.len();

        BenchSetup::new(iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", m_total * k_in, dt))
            .buffer(BenchBuffer::random("w", n_experts * n_out * words_per_row, DType::U32))
            .buffer(BenchBuffer::random("scales", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::random("biases", n_experts * n_out * groups_per_row, dt))
            .buffer(BenchBuffer::from_vec("tile_expert", u32_bytes(&tile_expert), DType::U32))
            .buffer(BenchBuffer::from_vec("tile_row_start", u32_bytes(&tile_row_start), DType::U32))
            .buffer(BenchBuffer::from_vec("tile_row_count", u32_bytes(&tile_row_count), DType::U32))
            .buffer(BenchBuffer::from_vec(
                "x_rows",
                u32_bytes(&(0..m_total as u32).collect::<Vec<_>>()),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("out", m_total * n_out, dt).output())
            .constexpr("n_out", n_out as u32)
            .constexpr("k_in", k_in as u32)
            .constexpr("group_size", group_size as u32)
            .with_shape_label(format!(
                "{label} M{m_total} N{n_out} K{k_in} E{n_experts} tiles{num_tiles} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_out as u32 / 32, num_tiles as u32, 1, [32, 1, 1])
            .bytes_moved(bytes as u64)
            .flops(2 * m_total as u64 * n_out as u64 * k_in as u64)
    }

    // ── gate/up leg (K=2048, N=512) ─────────────────────────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t4096_s0(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t4096_s1(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t4096_s2(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 4096, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t16384_s0(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t16384_s1(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t16384_s2(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 16384, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t32768_s0(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0001, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t32768_s1(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0002, "gate_up")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_gate_up_t32768_s2(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 32768, 512, 2048, 256, 64, 0x9E37_0003, "gate_up")
    }

    // ── down leg (K=512, N=2048) ────────────────────────────────────────
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t4096_s0(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t4096_s1(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t4096_s2(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 4096, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t16384_s0(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t16384_s1(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t16384_s2(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 16384, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t32768_s0(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0001, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t32768_s1(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0002, "down")
    }
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_moe_gather_qmm_bm16_isolated_down_t32768_s2(dt: DType) -> BenchSetup {
        isolated_cmp_bench(dt, 32768, 2048, 512, 256, 64, 0x85EB_0003, "down")
    }
}

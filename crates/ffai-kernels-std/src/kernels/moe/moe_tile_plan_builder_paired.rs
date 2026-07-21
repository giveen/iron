//! Copyright 2026 Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Host-side reference builder for the two-expert PAIR-ONLY tile plan that
//! feeds `ffai_moe_gather_qmm_coop_paired` (`moe_gather_qmm_coop_paired.rs`).
//! Pure Rust, no device dependency - this is the exhaustively-testable
//! ground truth the device builder (`moe_tile_plan_builder_bm32_paired.rs`)
//! mirrors, and the fixture generator the paired GEMM kernel's oracle tests
//! draw on.
//!
//! ## Why a SEPARATE dispatch, not a uniform two-half format everywhere
//!
//! `build_tile_plan_with_bm(counts, 32)` (`moe_mpp_tileplan.rs`) emits one
//! tile per `min(remaining, 32)` rows of a single expert, concatenated in
//! ascending expert order. At realistic low-context routing skew (T=512,
//! top-8 of 256 experts, average 16 rows/expert) most experts finish with a
//! short remainder well under 32 rows, so a large fraction of dispatched
//! tiles carry far fewer than 32 real rows - the rest of the tile's K-loop
//! work is spent on masked-out padding rows that produce no useful output.
//!
//! An earlier version of this module used a UNIFORM two-half
//! representation for every tile (full-32-row tiles split into two
//! same-expert 16-row halves, so a single kernel and a single dispatch
//! could handle both paired and unpaired tiles with one code path). That
//! design was abandoned: staging TWO experts' dequantized weights per
//! tile (see the paired GEMM kernel's module doc) is roughly double the
//! per-tile dequant cost of the existing single-expert kernel, and paying
//! that tax on every tile - including the ~85%+ of tiles at low T whose
//! remainder does not even need pairing (either an exact BM=32 run, or a
//! 17..31-row remainder with no pairing hazard in the first place) - would
//! likely cost more than the padding it removes, and would certainly lose
//! at high T where padding is already near zero (see
//! `paired_coverage`'s doc for the actual numbers: 4.2% tile-count
//! reduction at mTotal=32768 does not justify 2x dequant cost on that
//! mTotal's tiles).
//!
//! This module instead SPLITS a routing fixture's tile plan into TWO
//! independent dispatches, both computed directly against the TRUE
//! sorted-row space (the same `x_rows`/`out` index space the unpaired
//! kernel already uses - neither dispatch renumbers or compacts rows):
//!
//! 1. An "own" tile plan (`build_own_tile_plan_bm32`, the host mirror of
//!    the NEW `ffai_moe_build_tile_plan_bm32_own` device kernel) - every
//!    exact-multiple-of-32 run and every `17..=31`-row remainder, fed to
//!    the COMPLETELY UNCHANGED existing `ffai_moe_gather_qmm_coop` GEMM
//!    kernel. This is NOT simply `build_tile_plan_with_bm(counts, 32)`
//!    fed a shrunken `counts` array - that would shift every subsequent
//!    expert's row_base (an early version of this module had exactly
//!    that bug, caught by `property_full_coverage_across_many_fixtures`
//!    disagreeing with the true row space). `build_own_tile_plan_bm32`
//!    instead computes each tile's true position directly from the
//!    ORIGINAL `counts` (same row-base accumulation
//!    `build_tile_plan_with_bm` uses), simply SKIPPING the one tile a
//!    `1..=16`-row remainder would otherwise get.
//! 2. A `PairedTilePlan` of ONLY the `1..=16`-row remainders, fed to the
//!    NEW `ffai_moe_gather_qmm_coop_paired` kernel via a SECOND, separate
//!    dispatch.
//!
//! `split_for_pairing` is the entry point; the two returned plans are
//! independent dispatches' worth of input, not one combined format, and
//! together their real rows exactly partition `0..m_total` with no
//! overlap and no gap - see `every_row_covered_exactly_once`-style
//! property tests below.
//!
//! ## Fixed-row-16-half pairing, and why
//!
//! Every tile in a `PairedTilePlan` has an "A half" (local rows 0..16) and
//! a "B half" (local rows 16..32), each carrying its own `(expert,
//! row_start, row_count)`, `row_count <= 16`. Splitting at a FIXED row-16
//! boundary (never data-dependent) is deliberate: it aligns with the
//! coop-core GEMM's existing simdgroup partition (`sm = sg / 2` already
//! assigns simdgroups 0-1 to M-rows 0..16 and simdgroups 2-3 to M-rows
//! 16..32, see `moe_gather_qmm_coop.rs`), so each simdgroup's cooperative
//! MMA fragment reads exactly one expert's dequantized weights for its
//! entire 16-row half - never a fragment straddling two experts' weight
//! values. This is the "SAFE DESIGN OPTION" the paired GEMM kernel's
//! module doc calls out: a variable, data-dependent split point would let
//! rows from two different experts land inside the SAME 16x16 MMA
//! fragment, which is exactly the correctness hazard a prior scout
//! flagged and deferred on. Fixing the split at 16 costs some coverage (a
//! 17..31-row remainder cannot pair with anything, since its own A-half
//! is already full) but removes the hazard entirely - see `paired_coverage`
//! for the actual coverage this costs at realistic shapes.
//!
//! ## Pairing algorithm
//!
//! 1. Per expert, in ascending expert order: `full_pairs = count / 32`,
//!    `remainder = count % 32`. If `remainder` is `1..=16`, queue it as a
//!    pairing CANDIDATE and remove it from `own_counts[e]` (leaving an
//!    exact multiple of 32). Otherwise (`remainder == 0` or `17..=31`)
//!    leave `own_counts[e]` untouched.
//! 2. Candidates are greedily paired two at a time into one tile each
//!    (`expert_a`/`expert_b` independent, both `<= 16` rows). Under this
//!    fixed-16/16-half design ANY two candidates fit together (`16 + 16
//!    <= 32` always holds regardless of the two counts), so the number of
//!    tiles this phase produces is `ceil(candidates.len() / 2)`
//!    REGARDLESS of pairing order - there is no "best fit" search to do,
//!    unlike a variable-split design where packing order would change
//!    tile count. Candidates are still sorted (descending count, then
//!    ascending expert id) before pairing for a deterministic, readable
//!    tile layout; this does not change the tile count or the total
//!    padding (`sum(32 - count_a - count_b)` over all pair tiles is also
//!    pairing-order-invariant, since it equals `32 * num_pairs -
//!    sum(candidate counts)`, both of which are order-independent). An
//!    odd leftover candidate (no partner) is emitted as its own tile with
//!    an inert B half (`count_b = 0`).
//!
//! Every real row of every expert appears EXACTLY ONCE across `own_counts`
//! (as fed to the unchanged existing builder) plus the `PairedTilePlan`'s
//! covered rows - see the `every_row_covered_exactly_once` property test.

/// The two-expert PAIR-ONLY tile plan: parallel arrays, one entry per
/// tile, `tile_row_count_a` always `1..=16`. `tile_row_count_b == 0`
/// marks an odd-leftover tile with an inert B half (the GEMM kernel's row
/// mask excludes it from output writeback and from influencing the
/// result; `expert_b`/`row_start_b` are still valid in-bounds values in
/// that case, just unused, matching the existing coop-gather kernel's
/// `select(...clamp-to-0...)` convention for masked reads rather than
/// leaving addresses undefined).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PairedTilePlan {
    pub tile_expert_a: Vec<u32>,
    pub tile_row_start_a: Vec<u32>,
    pub tile_row_count_a: Vec<u32>,
    pub tile_expert_b: Vec<u32>,
    pub tile_row_start_b: Vec<u32>,
    pub tile_row_count_b: Vec<u32>,
}

impl PairedTilePlan {
    pub fn num_tiles(&self) -> usize { self.tile_expert_a.len() }

    fn push(&mut self, ea: u32, sa: u32, ca: u32, eb: u32, sb: u32, cb: u32) {
        self.tile_expert_a.push(ea);
        self.tile_row_start_a.push(sa);
        self.tile_row_count_a.push(ca);
        self.tile_expert_b.push(eb);
        self.tile_row_start_b.push(sb);
        self.tile_row_count_b.push(cb);
    }
}

/// A short remainder (`1..=16` rows) queued for cross-expert pairing.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    expert: u32,
    row_start: u32,
    count: u32,
}

/// Host mirror of the NEW `ffai_moe_build_tile_plan_bm32_own` device
/// kernel (`moe_tile_plan_builder_bm32_own.rs`): every exact-multiple-of-32
/// run and every `17..=31`-row remainder, in the TRUE sorted-row space
/// (same row-base accumulation as `build_tile_plan_with_bm`) - a `1..=16`
/// -row remainder gets NO tile here (it is a pairing candidate instead).
/// Same 3-array format as `build_tile_plan_with_bm`, so it is a drop-in
/// oracle for the existing `ffai_moe_gather_qmm_coop` GEMM kernel.
pub fn build_own_tile_plan_bm32(counts: &[usize]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut tile_expert = Vec::new();
    let mut tile_row_start = Vec::new();
    let mut tile_row_count = Vec::new();
    let mut row_base = 0u32;

    for (e, &c) in counts.iter().enumerate() {
        let e = e as u32;
        let full = c as u32 / 32;
        let remainder = c as u32 - full * 32;
        for i in 0..full {
            tile_expert.push(e);
            tile_row_start.push(row_base + i * 32);
            tile_row_count.push(32);
        }
        if remainder > 16 {
            tile_expert.push(e);
            tile_row_start.push(row_base + full * 32);
            tile_row_count.push(remainder);
        }
        row_base += c as u32;
    }

    (tile_expert, tile_row_start, tile_row_count)
}

/// Host mirror of `ffai_moe_build_tile_plan_bm32_paired`: the `1..=16`-row
/// pairing candidates only, in the TRUE sorted-row space. Greedily pairs
/// candidates two at a time (see module doc for why pairing order does
/// not affect tile count or total padding under the fixed 16/16-half
/// design).
pub fn build_paired_tile_plan_bm32(counts: &[usize]) -> PairedTilePlan {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut row_base = 0u32;

    for (e, &c) in counts.iter().enumerate() {
        let e = e as u32;
        if c > 0 {
            let full = c / 32;
            let remainder = c % 32;
            if (1..=16).contains(&remainder) {
                let after_full = row_base + (full * 32) as u32;
                candidates.push(Candidate {
                    expert: e,
                    row_start: after_full,
                    count: remainder as u32,
                });
            }
        }
        row_base += c as u32;
    }

    // Deterministic pairing order only, not a correctness or coverage
    // requirement - see the module doc for why order doesn't change the
    // tile count or total padding under the fixed 16/16-half design.
    candidates.sort_by(|a, b| b.count.cmp(&a.count).then(a.expert.cmp(&b.expert)));

    let mut plan = PairedTilePlan::default();
    let mut it = candidates.into_iter();
    loop {
        match (it.next(), it.next()) {
            (Some(a), Some(b)) => {
                plan.push(a.expert, a.row_start, a.count, b.expert, b.row_start, b.count);
            },
            (Some(a), None) => {
                // Odd leftover, no partner: own tile, inert B half.
                plan.push(a.expert, a.row_start, a.count, a.expert, a.row_start, 0);
            },
            (None, _) => break,
        }
    }

    plan
}

/// The "own" tile plan's 3 parallel arrays (`tile_expert`, `tile_row_start`,
/// `tile_row_count`) - same shape `build_tile_plan_with_bm` returns, named
/// here only to keep `split_for_pairing`'s signature readable.
pub type OwnTilePlan = (Vec<u32>, Vec<u32>, Vec<u32>);

/// Convenience wrapper bundling `build_own_tile_plan_bm32` and
/// `build_paired_tile_plan_bm32` - the two independent dispatches' worth
/// of tile plan a single routing fixture produces.
pub fn split_for_pairing(counts: &[usize]) -> (OwnTilePlan, PairedTilePlan) {
    (build_own_tile_plan_bm32(counts), build_paired_tile_plan_bm32(counts))
}

/// Worst-case tile capacity for the PAIR-ONLY plan's dispatch, for host
/// buffer pre-sizing (mirrors `moe_tile_plan_builder_bm32.rs`'s
/// `ceil(m_total/32) + n_experts` bound for the unpaired device builder's
/// own dispatch, which is unaffected - see the module doc). At most one
/// candidate per expert (`n_experts` candidates worst case), producing at
/// most `ceil(n_experts/2)` tiles.
pub fn paired_tile_plan_worst_case_capacity(n_experts: usize) -> usize { n_experts.div_ceil(2) }

/// Coverage summary comparing `own_tiles + paired_tiles` (the two-dispatch
/// total this design actually issues) to the single-dispatch
/// `build_tile_plan_with_bm(counts, 32)` baseline at the same routing
/// fixture - the number this task's report is scored on.
#[derive(Debug, Clone, Copy)]
pub struct PairedCoverage {
    pub baseline_tiles: usize,
    pub own_tiles: usize,
    pub paired_tiles: usize,
    pub total_tiles: usize,
    pub tiles_eliminated: usize,
    pub pct_eliminated: f64,
}

pub fn paired_coverage(counts: &[usize]) -> PairedCoverage {
    let baseline = crate::kernels::moe::moe_mpp_tileplan::build_tile_plan_with_bm(counts, 32);
    let baseline_tiles = baseline.0.len();
    let (own, plan) = split_for_pairing(counts);
    let own_tiles = own.0.len();
    let paired_tiles = plan.num_tiles();
    let total_tiles = own_tiles + paired_tiles;
    let tiles_eliminated = baseline_tiles.saturating_sub(total_tiles);
    let pct_eliminated = if baseline_tiles == 0 {
        0.0
    } else {
        100.0 * tiles_eliminated as f64 / baseline_tiles as f64
    };
    PairedCoverage {
        baseline_tiles,
        own_tiles,
        paired_tiles,
        total_tiles,
        tiles_eliminated,
        pct_eliminated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct the (row, expert) pairs the "own" plan plus the
    /// `PairedTilePlan` together cover. Both are already computed in the
    /// TRUE sorted-row space, so no row-base bookkeeping needed here.
    fn covered_rows(counts: &[usize]) -> Vec<(u32, u32)> {
        let (own, plan) = split_for_pairing(counts);
        let mut rows = Vec::new();

        let (te, trs, trc) = own;
        for i in 0..te.len() {
            for r in 0..trc[i] {
                rows.push((trs[i] + r, te[i]));
            }
        }

        for i in 0..plan.num_tiles() {
            for r in 0..plan.tile_row_count_a[i] {
                rows.push((plan.tile_row_start_a[i] + r, plan.tile_expert_a[i]));
            }
            for r in 0..plan.tile_row_count_b[i] {
                rows.push((plan.tile_row_start_b[i] + r, plan.tile_expert_b[i]));
            }
        }
        rows
    }

    fn expected_row_expert(counts: &[usize]) -> Vec<(u32, u32)> {
        let mut expected = Vec::new();
        let mut row = 0u32;
        for (e, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                expected.push((row, e as u32));
                row += 1;
            }
        }
        expected
    }

    fn assert_full_coverage(counts: &[usize]) {
        let mut got = covered_rows(counts);
        let mut want = expected_row_expert(counts);
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "counts = {counts:?}");
    }

    fn assert_halves_bounded(counts: &[usize]) {
        let (_own, plan) = split_for_pairing(counts);
        for i in 0..plan.num_tiles() {
            let ca = plan.tile_row_count_a[i];
            let cb = plan.tile_row_count_b[i];
            assert!((1..=16).contains(&ca), "A half must carry 1..=16 rows, got {ca}");
            assert!(cb <= 16, "B half must carry 0..=16 rows, got {cb}");
        }
    }

    #[test]
    fn empty_input_has_no_pair_tiles() {
        let (own, plan) = split_for_pairing(&[]);
        assert!(own.0.is_empty());
        assert_eq!(plan.num_tiles(), 0);
        let (own, plan) = split_for_pairing(&[0, 0, 0]);
        assert!(own.0.is_empty());
        assert_eq!(plan.num_tiles(), 0);
    }

    #[test]
    fn single_full_tile_stays_in_own_plan_untouched() {
        let (own, plan) = split_for_pairing(&[32]);
        assert_eq!(own.0.len(), 1);
        assert_eq!(own.2[0], 32);
        assert_eq!(plan.num_tiles(), 0);
    }

    #[test]
    fn two_experts_with_complementary_short_remainders_pair() {
        let counts = [16usize, 16usize];
        let (own, plan) = split_for_pairing(&counts);
        assert!(own.0.is_empty(), "both counts are pure pairing candidates, no own tiles");
        assert_eq!(plan.num_tiles(), 1);
        assert_eq!(plan.tile_row_count_a[0], 16);
        assert_eq!(plan.tile_row_count_b[0], 16);
        assert_ne!(
            plan.tile_expert_a[0], plan.tile_expert_b[0],
            "must be a genuine cross-expert pair"
        );
        assert_full_coverage(&counts);
    }

    #[test]
    fn unbalanced_pair_3_and_13() {
        let counts = [3usize, 13usize];
        let (_own, plan) = split_for_pairing(&counts);
        assert_eq!(plan.num_tiles(), 1);
        let mut counts_seen = vec![plan.tile_row_count_a[0], plan.tile_row_count_b[0]];
        counts_seen.sort_unstable();
        assert_eq!(counts_seen, vec![3, 13]);
        assert_full_coverage(&counts);
    }

    #[test]
    fn unbalanced_pair_16_and_1() {
        let counts = [16usize, 1usize];
        let (_own, plan) = split_for_pairing(&counts);
        assert_eq!(plan.num_tiles(), 1);
        assert_full_coverage(&counts);
    }

    #[test]
    fn remainder_17_to_31_never_pairs_and_stays_in_own_plan() {
        // 25-row remainder stays in the own plan as a 16+9 A/B-split tile
        // via `build_own_tile_plan_bm32` (NOT via the paired plan) - no
        // pairing candidate is queued for it, so an unrelated short
        // expert elsewhere stays an odd leftover.
        let counts = [25usize, 5usize];
        let (own, plan) = split_for_pairing(&counts);
        assert_eq!(own.0.len(), 1);
        assert_eq!(own.2[0], 25);
        assert_eq!(plan.num_tiles(), 1); // the 5-row remainder, odd leftover.
        assert_eq!(plan.tile_row_count_b[0], 0);
        assert_full_coverage(&counts);
        assert_halves_bounded(&counts);
    }

    #[test]
    fn odd_number_of_candidates_leaves_one_unpaired() {
        let counts = [1usize, 2usize, 3usize];
        let (own, plan) = split_for_pairing(&counts);
        assert!(own.0.is_empty());
        // 3 candidates -> 1 pair + 1 leftover = 2 tiles.
        assert_eq!(plan.num_tiles(), 2);
        assert_full_coverage(&counts);
        let has_inert_b = (0..plan.num_tiles()).any(|i| plan.tile_row_count_b[i] == 0);
        assert!(has_inert_b, "the odd-one-out must show up as an inert-B tile");
    }

    #[test]
    fn zero_count_experts_are_skipped_entirely() {
        let counts = [0usize, 4usize, 0usize, 0usize, 12usize, 0usize];
        let (own, plan) = split_for_pairing(&counts);
        assert!(own.0.is_empty());
        assert_eq!(plan.num_tiles(), 1); // 4 + 12 = 16 total, one pair.
        assert_full_coverage(&counts);
        for i in 0..plan.num_tiles() {
            assert!(counts[plan.tile_expert_a[i] as usize] > 0);
            assert!(counts[plan.tile_expert_b[i] as usize] > 0);
        }
    }

    #[test]
    fn skewed_realistic_fixture_full_coverage_and_bounds() {
        let counts = [1usize, 0, 200, 0, 5, 22, 0, 3, 32, 9, 0, 47, 16, 16, 8, 8, 8, 1, 1, 1];
        assert_full_coverage(&counts);
        assert_halves_bounded(&counts);
    }

    /// Property test: for a spread of pseudo-random skewed fixtures, every
    /// row of every expert appears exactly once across
    /// `own_counts + PairedTilePlan`, no pair tile exceeds its bounds, and
    /// the pair-only plan never exceeds its documented worst-case
    /// capacity.
    #[test]
    fn property_full_coverage_across_many_fixtures() {
        fn lcg_next(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *state >> 32
        }
        let mut state = 0x1234_5678_9abc_def0u64;
        for trial in 0..200 {
            let n_experts = 1 + (lcg_next(&mut state) % 40) as usize;
            let counts: Vec<usize> = (0..n_experts)
                .map(|_| {
                    let r = lcg_next(&mut state) % 100;
                    if r < 70 {
                        (lcg_next(&mut state) % 17) as usize // 0..16
                    } else if r < 95 {
                        (lcg_next(&mut state) % 40) as usize // 0..39
                    } else {
                        100 + (lcg_next(&mut state) % 300) as usize
                    }
                })
                .collect();
            assert_full_coverage(&counts);
            assert_halves_bounded(&counts);
            let (_own, plan) = split_for_pairing(&counts);
            assert!(
                plan.num_tiles() <= paired_tile_plan_worst_case_capacity(n_experts),
                "trial {trial}: pair-only plan exceeded its documented worst-case capacity bound"
            );
        }
    }

    #[test]
    fn coverage_never_worse_than_baseline() {
        let fixtures: Vec<Vec<usize>> = vec![
            vec![512usize],
            vec![16, 16, 16, 16],
            (0..256).map(|e| if e % 3 == 0 { 0 } else { (e % 20) + 1 }).collect(),
            vec![1; 100],
        ];
        for counts in fixtures {
            let cov = paired_coverage(&counts);
            assert!(cov.total_tiles <= cov.baseline_tiles, "counts summary len={}", counts.len());
        }
    }

    /// Uniform routing (every expert gets exactly the same count) is the
    /// design's best case: if that count is `<=16`, EVERY expert's whole
    /// run is a single pairing candidate, so pairing removes essentially
    /// half the tiles.
    #[test]
    fn uniform_routing_under_16_rows_pairs_almost_everything() {
        let counts = vec![16usize; 256];
        let cov = paired_coverage(&counts);
        assert_eq!(cov.baseline_tiles, 256);
        assert_eq!(cov.own_tiles, 0);
        assert_eq!(cov.paired_tiles, 128);
        assert_eq!(cov.total_tiles, 128);
        assert!((cov.pct_eliminated - 50.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod coverage_report {
    use super::*;
    use crate::kernels::moe::moe_mpp_shared::zipfish_counts;

    /// Informational, not a gate: prints the pairing-coverage table this
    /// task's report is built from (F-85 tile-pairing lever), for the
    /// TWO-DISPATCH design (own_tiles via the unmodified existing
    /// builder, paired_tiles via the new pair-only plan). Run with
    /// `--nocapture` to see the table.
    #[test]
    fn print_coverage_table() {
        let n_experts = 256usize;
        let seeds = [0x9E37_0001u64, 0x9E37_0002, 0x9E37_0003];
        println!(
            "mTotal   seed        baseline_tiles  own_tiles  paired_tiles  total_tiles  eliminated  pct"
        );
        for &m_total in &[512usize, 1024, 2048, 4096, 8192, 16384, 32768] {
            for &seed in &seeds {
                let counts = zipfish_counts(m_total, n_experts, seed);
                let cov = paired_coverage(&counts);
                println!(
                    "{m_total:<8} 0x{seed:08X}  {:<14} {:<10} {:<13} {:<12} {:<11} {:.1}%",
                    cov.baseline_tiles,
                    cov.own_tiles,
                    cov.paired_tiles,
                    cov.total_tiles,
                    cov.tiles_eliminated,
                    cov.pct_eliminated
                );
            }
        }
        println!("--- uniform routing ---");
        for &m_total in &[512usize, 1024, 2048, 4096, 8192, 16384, 32768] {
            let base = m_total / n_experts;
            let rem = m_total % n_experts;
            let counts: Vec<usize> =
                (0..n_experts).map(|e| base + if e < rem { 1 } else { 0 }).collect();
            let cov = paired_coverage(&counts);
            println!(
                "{m_total:<8} uniform     {:<14} {:<10} {:<13} {:<12} {:<11} {:.1}%",
                cov.baseline_tiles,
                cov.own_tiles,
                cov.paired_tiles,
                cov.total_tiles,
                cov.tiles_eliminated,
                cov.pct_eliminated
            );
        }
    }
}

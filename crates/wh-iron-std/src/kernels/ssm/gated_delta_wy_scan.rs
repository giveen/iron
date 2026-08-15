//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Chunked-WY GDN prefill — **scan** kernel (pass 2 of 2) — `iron_gdn_wy_scan`.
//!
//! Consumes the chunk-parallel intermediates
//! [`iron_gdn_wy_plan`](super::gated_delta_wy_plan) produced and threads the
//! recurrent `[Dv, Dk]` state sequentially across chunks. Where `_plan`
//! parallelises over chunks (no state), `_scan` parallelises over `Dv`
//! **tiles** (32 rows each) and walks the chunk axis sequentially — each TG
//! only ever holds a `[32, Dk]` slice of the state, never the full
//! `[Dv, Dk]` (16 KiB at `Dk = 128` vs. 64 KiB for the whole thing), which is
//! what makes production `Dv = Dk = 128` fit in TG memory at all.
//!
//! ## Math — combining `_plan`'s state-free factors with the running state
//!
//! `_plan`'s module doc derives (from `gated_delta_wy.rs` steps 7 and 9):
//!   - `y_pass[t] = S_0 · q_eff[t]`         (`S_0` = state at chunk entry)
//!   - `S_end     = S_0 · P_s + U_s`
//!
//! This kernel computes exactly that, tile-by-tile, then folds in `_plan`'s
//! already-computed `y_local`:
//!
//! ```text
//! for each chunk (sequential):
//!     y_pass_tile[t, v] = sum_d q_eff[t, d] * S_tile[v, d]      // [C,Dk]·[Dk,32]
//!     y[t, v]           = y_pass_tile[t, v] + y_local[t, v]     // write
//!     S_tile[v, d]      <- sum_k S_tile[v, k] * P_s[k, d] + U_s[v, d]   // [32,Dk]·[Dk,Dk]
//! ```
//!
//! `S_tile` is loaded from `state_in` once before the chunk loop and written
//! to `state_out` once after — no per-chunk global state traffic (mirrors
//! `gated_delta_prep_chunk`'s "load once / store once" state discipline).
//!
//! ## Layout (must agree with `iron_gdn_wy_plan`'s doc block)
//!
//!   - `q_eff`   : `[N, NC, C, Dk]`
//!   - `y_local` : `[N, NC, C, Dv]`
//!   - `p_s`     : `[N, NC, Dk, Dk]`
//!   - `u_s`     : `[N, NC, Dv, Dk]`
//!   - `state_in`/`state_out` : `[B, Hv, Dv, Dk]`
//!   - `y`       : `[B, T, Hv, Dv]` (the real production output)
//!   - `t_len`   : `[1]` u32 — runtime; `NC = t_len / c`.
//!
//!   `N = B·Hv`, `n = b·Hv + hv_idx` (matches `_plan`'s grouping and
//!   `state_in`/`state_out`'s `[B,Hv,...]` layout).
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** Grid: `[Dv/32, B·Hv, 1]` — one TG per 32-row `Dv`
//!   tile per `(b, hv)` slot. TG: `[128, 1, 1]` (4 SGs). `tid` = lane
//!   (0..127, spans the full TG).
//! - **`Dv % 32 == 0`** for v1 (one TG owns exactly 32 rows; a ragged last
//!   tile is a follow-up). `dk` ≤ 128 (sizes the TG state-tile literal
//!   below). `t_len` a runtime u32, multiple of `c`.
//! - `tgid_x` = Dv-tile index (`dv_tile_base = tgid_x * 32`), `tgid_y` =
//!   `n = b·Hv + hv_idx`.
//! - Chunk loop is **sequential** (`NC = t_len/c` iterations) — this is the
//!   one place in the two-kernel split where cross-chunk order matters.
//!
//! ## TG memory
//!
//!   - `tg_state`: `32 × Dk` × f32 = `32 × 128 × 4 B` = 16 KiB — the *only*
//!     TG-resident state, one 32-row tile (not the full `[Dv,Dk]`). This is
//!     the one buffer that must stay f32: it's the recurrent state itself,
//!     read and rewritten every chunk for the entire `T`-sequence — the
//!     same "f32 is the floor for stable long-context recurrences" reason
//!     `gated_delta_wy.rs`'s original doc gives for its own state buffer.
//!   - A per-lane `new_state` stack (32 × f32) stages the updated tile
//!     before the barrier'd flush-back into `tg_state` — same double-buffer
//!     discipline the monolithic kernel used for its state update, just
//!     resized to this kernel's `128`-thread / `32`-row-tile geometry.
//!   - `p_s`/`u_s` are read directly from global memory (no TG caching):
//!     each is `[Dk, Dk]`/`[32, Dk]` per chunk — `Dk×Dk` alone (64 KiB at
//!     `Dk=128`) cannot be TG-resident at all, and there is no reuse across
//!     lanes within a single (v,d) update to justify staging the tile
//!     that *would* fit.
//!
//! ## v2 — cooperative-tile MMA (`coop_tile_*`)
//!
//! Both inner products here — `y_pass_tile = q_eff · S_tile^T` ([C,Dk]×
//! [Dk,32]) and the state update `S_tile · P_s` ([32,Dk]×[Dk,Dk]) — are
//! textbook GEMM shapes, now wired through `coop_tile_setup`/
//! `coop_tile_load_a/b`/`coop_tile_run`/`coop_tile_store_c` (the
//! `MetalPerformancePrimitives` `matmul2d` cooperative-tensor path — see
//! `moe_mpp_expert_grid.rs`/`gemm_q8_mpp.rs` for the established staging
//! pattern this follows). Both GEMMs use the same **16×32×32** (M,N,K) tile
//! shape, `ta=false, tb=true` (both `S_tile` and the transposed staging of
//! `P_s` below are stored `[outer=N, inner=K]`, matching the "weight
//! stored `[out,k]`" convention every other MMA kernel in this codebase
//! already uses), `acc_mode="accumulate"` (multiply-accumulate across the
//! `K`-block loop), `acc_dtype=f32`.
//!
//! **Fixed `f16` staging, independent of the kernel's generic `T`.** Like
//! `gated_delta_wy_plan.rs`'s `tg_solve` (always bf16-staged regardless of
//! `T`), the MMA inputs here are *always* staged through `f16` — never
//! `coop_stage(T)` — because `coop_stage(T)` would size the staging buffers
//! at `T`'s width (4 B for the `f32` kernel instantiation), blowing the TG
//! budget below by 2×. `f16`'s 10-bit mantissa is ample for these
//! chunk-local, non-compounding operands (`q_eff`/`p_s`/`u_s` are freshly
//! recomputed per chunk, not accumulated across the recurrence); only
//! `tg_state` itself — read *into* the MMA at reduced precision but never
//! overwritten at reduced precision — carries the actual cross-chunk state,
//! and it stays f32 in `tg_state` throughout (unchanged from v1).
//!
//! ### GEMM1 (`y_pass`) — one M-tile per simdgroup, no N-split
//!
//! `M=C(≤64, 4 tiles of 16 — one per SG), N=32 (single tile, Dv-tile width,
//! shared by all 4 SGs), K=Dk (`K/32` accumulate steps)`. For `C<64` test
//! fixtures, SGs whose 16-row tile falls entirely outside `[0,C)` still run
//! the MMA (`Xs` is zero-padded past `C` by the staging loop, so they just
//! MMA zeros — wasted work, never wrong) — their `ct_c`/`OutScratch` output
//! is simply never read back, since the consumption loop below only ever
//! visits `t < C`. **Intentionally unconditional, not `if sg < c_m_tiles`**:
//! `coop_tile_load_a`/`coop_tile_run`'s CUDA codegen embeds its own
//! `__syncthreads()`, and gating them behind a per-SG runtime `if` made only
//! some warps in the block execute those internal barriers — a
//! divergent-barrier hazard (`compute-sanitizer --tool synccheck` catches
//! it, `racecheck`/`initcheck` don't) that was this kernel's actual
//! `state_out` corruption root cause, see GDN_PREFILL_CONTRACT.md §7.2.
//! Requires **`C` a multiple of 16** (both existing test fixtures — `C=16`,
//! `C=64` — and the production `C=64` bench satisfy this; a future
//! ragged-C tail is a follow-up).
//!
//! ### GEMM2 (`S_tile · P_s + U_s`) — 2×2 SG grid, `nj` outer loop over N
//!
//! `M=32 (2 tiles of 16 → `sg_m = sg/2`), N=Dk (2-tiles-per-iteration via
//! `sg_n_local = sg%2`, outer-looped `nj` in `0..ceil(Dk/64)` to cover all
//! `Dk/32` N-tiles), K=Dk`. Requires **`Dk` a multiple of 32** (both test
//! fixtures — `Dk=32`, `Dk=128` — and the production `Dk=128` bench
//! satisfy this). `n_tile = nj*2 + sg_n_local`; SGs/`nj`-slots whose
//! `n_tile ≥ Dk/32` still run the MMA (same zero-padded-input,
//! unconditional-for-the-same-barrier-safety-reason-as-GEMM1 pattern —
//! see above) and are simply masked out of the consumption step by
//! `d_valid`, so `Dk=32` (only 1 real N-tile) is handled by the same code
//! path that handles `Dk=128` (4 N-tiles) — no separate small-Dk code.
//!
//! **Critical ordering, unchanged from v1's doc:** `S_tile`'s update reads
//! the *entire* old row (`sum_k S_old[v,k]*P_s[k,d]` — every output column
//! `d` needs *all* `K=Dk` of `S_old[v,:]`), so no output column can be
//! written back into `tg_state` until **every** `nj` iteration has finished
//! reading it. The MMA path therefore keeps v1's per-lane `new_state` stack
//! (private, not TG-resident) as the write staging area across the whole
//! `nj` loop, and only flushes `new_state → tg_state` once, after the loop
//! — an in-place per-`nj` write into `tg_state` would corrupt the *next*
//! `nj`'s read of not-yet-updated columns from the *same* row.
//!
//! ## TG memory budget (v2, MMA)
//!
//!   - `tg_state`: 32×Dk×f32 = 16 KiB (unchanged from v1 — the recurrent
//!     state, must stay f32).
//!   - `Xs`, `Ws`: each 2048 elements × f16 = 4 KiB — sized to the larger
//!     of the two GEMMs' per-K-block tiles (GEMM1: `Xs`=C×32=2048,
//!     `Ws`=32×32=1024; GEMM2: `Xs`=32×32=1024, `Ws`=64×32=2048).
//!   - `OutScratch`: 2048 elements × **f32** = 8 KiB — holds 4 SGs' worth
//!     of dense per-SG output tiles (512 elements each, both GEMMs).
//!     Must be f32, *not* the f16 the other staging buffers use: Metal's
//!     `matmul2d` cooperative tensor `store()` requires
//!     `is_same_v<dest_dtype, acc_dtype>` — there is no downcast-on-store
//!     overload, so the C-tile destination has to match `acc_dtype=f32`
//!     exactly (confirmed the hard way: an f16 `OutScratch` compiles the
//!     *load*/*store* calls for `Xs`/`Ws` fine but fails Metal compilation
//!     specifically on `ct_c.store()` with "no matching member function").
//!   - All three reused by name across both GEMM phases (sequential, never
//!     live simultaneously — same reuse discipline `gated_delta_wy_plan.rs`'s
//!     `tg_solve` uses for `p`/`u`).
//!   - Total: 16 + 4 + 4 + 8 = **32 KiB** — exactly Apple's TG budget on
//!     this hardware (`iron device` reports 32 KiB on the M5 Max this was
//!     developed against; confirmed to dispatch and pass at this size, but
//!     there is zero headroom left — a smaller-GPU target with a lower cap
//!     would need to shrink `Xs`/`Ws` further, e.g. a 16-wide K-tile).
//!     (`new_state` is a per-lane *private* stack, not threadgroup memory,
//!     so it doesn't count here — see v1's doc.)

#![allow(clippy::too_many_arguments)]

use wh_iron::kernel;

#[kernel]
pub fn iron_gdn_wy_scan<T>(
    q_eff: Tensor<T>,
    y_local: Tensor<T>,
    p_s: Tensor<T>,
    u_s: Tensor<T>,
    state_in: Tensor<T>,
    mut state_out: Tensor<T>,
    mut y: Tensor<T>,
    t_len: Tensor<u32>,
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] c: u32,
) {
    // ── Geometry ───────────────────────────────────────────────────────
    let dv_tile = tgid_x;
    let n = tgid_y;
    let b_idx = n / hv;
    let hv_idx = n - b_idx * hv;
    let lane = tid; // 0..127, full TG (= "lane_in_tg" in the moe/gemm MMA kernels)
    let sg = simd_group_id(); // 0..3
    let dv_tile_base = dv_tile * 32u32;
    let t_total = load(t_len[0]);
    let num_chunks = t_total / c;

    // ── TG state tile [32, Dk] — the only TG-resident state (see doc). ──
    threadgroup_alloc("tg_state", 4096u32, f32); // 32 * Dk_max(128)
    stack_alloc("new_state", 32u32, "f32"); // (32*Dk_max)/TG(128) per lane

    // ── MMA staging (see module doc for the 28 KiB budget). Reused by
    // name across both GEMM1/GEMM2 phases — sequential, never concurrent.
    threadgroup_alloc("Xs", 2048u32, f16);
    threadgroup_alloc("Ws", 2048u32, f16);
    threadgroup_alloc("OutScratch", 2048u32, f32); // MUST match acc_dtype (Metal's ct_c.store()
    // requires is_same_v<T, acc_dtype> — no downcast-on-store) — see module doc.
    coop_tile_setup("g1", 16, 32, 32, f16, "accumulate", "simdgroup", f32, false, true, false);
    coop_tile_setup("g2", 16, 32, 32, f16, "accumulate", "simdgroup", f32, false, true, false);

    let state_base = n * dv * dk + dv_tile_base * dk;
    let tile_elems = 32u32 * dk;
    for i in range(lane, tile_elems, 128u32) {
        let val = load(state_in[state_base + i]).cast::<f32>();
        threadgroup_store("tg_state", i, val);
    }
    threadgroup_barrier();

    // GEMM2's N (=Dk) tiling: Dk/32 total 32-wide N-tiles, walked 2 at a
    // time (one per sg_n_local) across `nj_outer` outer iterations.
    let n_tiles_total = dk / 32u32;
    let nj_outer = (n_tiles_total + 1u32) / 2u32;
    let sg_m2 = sg / 2u32; // GEMM2 M-tile (0 or 1)
    let sg_n_local = sg % 2u32; // GEMM2 N-slot within the current `nj`

    // ── Sequential chunk loop — the only cross-chunk dependency. ───────
    for chunk_idx in range(0u32, num_chunks, 1u32) {
        let chunk_start = chunk_idx * c;
        let nc_base = n * num_chunks + chunk_idx;
        let qeff_base = nc_base * c * dk;
        let yl_base = nc_base * c * dv;
        let ps_base = nc_base * dk * dk;
        let us_base = nc_base * dv * dk;

        // ── GEMM1: y_pass_tile[t,v] = q_eff[t,:]·S_tile[v,:]^T ────────
        // M=C (4×16, one M-tile/SG), N=32 (single tile, shared), K=Dk.
        coop_tile_zero("g1");
        for kb in range(0u32, dk, 32u32) {
            // Stage Xs[t,k] = q_eff[t, kb+k], t in [0,64) (clamped/zeroed
            // past C — every SG's tile is well-defined even past C, see
            // the unconditional-MMA comment below). 2048 elem / 128 = 16/lane.
            for i0 in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + i0;
                let mr = flat / 32u32;
                let kc = flat % 32u32;
                let valid = mr < c;
                let safe_mr = select(valid, mr, 0u32);
                let xv = load(q_eff[qeff_base + safe_mr * dk + kb + kc]).cast::<f32>();
                threadgroup_store("Xs", flat, select(valid, xv, 0.0f32).cast::<f16>());
            }
            // Stage Ws[v,k] = tg_state[v, kb+k] downcast, v in [0,32).
            // 1024 elem / 128 = 8/lane.
            for i1 in range(0u32, 8u32, 1u32) {
                let flat = lane * 8u32 + i1;
                let vr = flat / 32u32;
                let kc = flat % 32u32;
                let sv = threadgroup_load("tg_state", vr * dk + kb + kc);
                threadgroup_store("Ws", flat, sv.cast::<f16>());
            }
            threadgroup_barrier();
            // Run unconditionally for every SG, even ones whose 16-row tile
            // falls entirely outside [0,C) (Xs is already zero-padded past
            // C by the staging loop above, so those SGs just MMA zeros —
            // wasted work, never wrong). MUST NOT be gated behind an `if`
            // here: `coop_tile_load_a`/`coop_tile_run`'s CUDA codegen
            // embeds its own `__syncthreads()` internally, and `sg` differs
            // *per warp* (this device's warp size equals the simdgroup
            // width), so an `if sg < c_m_tiles { ...load_a/run... }` guard
            // makes only SOME warps in the block execute those internal
            // barriers — a block-level divergent-barrier hazard invisible
            // to `compute-sanitizer --tool racecheck`/`initcheck` but
            // caught by `--tool synccheck` ("Barrier error: Divergent
            // thread(s) in block"). See GDN_PREFILL_CONTRACT.md §7.2.
            coop_tile_load_a("g1", "Xs", true, f16, 32u32, 16u32, sg * 512u32);
            coop_tile_load_b("g1", "Ws", true, f16, 32u32, 32u32);
            coop_tile_run("g1");
            threadgroup_barrier();
        }
        coop_tile_store_c("g1", "OutScratch", true, f32, 32u32, 16u32, sg * 512u32);
        threadgroup_barrier();
        // Consume: y = y_pass (from OutScratch) + y_local. Bounded by
        // `c` — never touches an out-of-range SG's (always-computed but
        // zero-padded-input, hence zero) output.
        for tv in range(lane, c * 32u32, 128u32) {
            let t = tv / 32u32;
            let v = tv % 32u32;
            let src_sg = t / 16u32;
            let local_t = t % 16u32;
            let src_off = src_sg * 512u32 + local_t * 32u32 + v;
            let acc = threadgroup_load("OutScratch", src_off).cast::<f32>();
            let yl = load(y_local[yl_base + t * dv + (dv_tile_base + v)]).cast::<f32>();
            let t_abs = chunk_start + t;
            let y_off = ((b_idx * t_total + t_abs) * hv + hv_idx) * dv + dv_tile_base + v;
            store(y[y_off], (acc + yl).cast::<T>());
        }
        threadgroup_barrier();

        // ── GEMM2: S_new[v,d] = S_old[v,:]·P_s[:,d] + U_s[v,d] ────────
        // M=32 (2×16, sg_m2), N=Dk (2 tiles/iter via sg_n_local, `nj`
        // outer loop), K=Dk. Writes only ever land in the per-lane
        // `new_state` stack — see doc: no tg_state write until every
        // `nj` has read the full old row.
        for nj in range(0u32, nj_outer, 1u32) {
            coop_tile_zero("g2");
            for kb in range(0u32, dk, 32u32) {
                // Stage Xs[v,k] = tg_state[v, kb+k] downcast, v in [0,32).
                for i0 in range(0u32, 8u32, 1u32) {
                    let flat = lane * 8u32 + i0;
                    let vr = flat / 32u32;
                    let kc = flat % 32u32;
                    let sv = threadgroup_load("tg_state", vr * dk + kb + kc);
                    threadgroup_store("Xs", flat, sv.cast::<f16>());
                }
                // Stage Ws[n_local,k] = P_s[kb+k, nj*64+n_local], TRANSPOSED
                // on copy (P_s is [K,N] row-major; Ws must be [N,K] to
                // match tb=true — same "W stored [out,k]" convention every
                // other MMA kernel here uses). n_local in [0,64); OOB
                // columns (small-Dk fixtures) clamp-and-zero.
                for i1 in range(0u32, 16u32, 1u32) {
                    let flat = lane * 16u32 + i1;
                    let n_local = flat / 32u32;
                    let k_local = flat % 32u32;
                    let n_glob = nj * 64u32 + n_local;
                    let col_valid = n_glob < dk;
                    let safe_n = select(col_valid, n_glob, 0u32);
                    let pv = load(p_s[ps_base + (kb + k_local) * dk + safe_n]).cast::<f32>();
                    threadgroup_store(
                        "Ws",
                        n_local * 32u32 + k_local,
                        select(col_valid, pv, 0.0f32).cast::<f16>(),
                    );
                }
                threadgroup_barrier();
                // Run unconditionally for every SG/`nj` slot, even ones
                // whose N-tile falls entirely outside [0,Dk) (Xs/Ws are
                // already zero-padded past `dk` by the staging above, so
                // those slots just MMA zeros — wasted work, never wrong).
                // MUST NOT be gated behind an `if` here — same
                // divergent-`__syncthreads()` hazard as GEMM1 above (see
                // that comment + GDN_PREFILL_CONTRACT.md §7.2): `sg_n_local`
                // is warp-uniform, so `if n_tile_valid { ...load_a/run... }`
                // made only some warps in the block execute the internal
                // barriers `coop_tile_load_a`/`coop_tile_run` emit on CUDA
                // — confirmed via `compute-sanitizer --tool synccheck`
                // ("Barrier error: Divergent thread(s) in block"), and the
                // actual root cause of this kernel's `state_out`
                // zero-row corruption (racecheck/initcheck don't catch this
                // hazard class, which is why they came back clean).
                coop_tile_load_a("g2", "Xs", true, f16, 32u32, 16u32, sg_m2 * 512u32);
                coop_tile_load_b("g2", "Ws", true, f16, 32u32, 32u32, sg_n_local * 1024u32);
                coop_tile_run("g2");
                threadgroup_barrier();
            }
            coop_tile_store_c("g2", "OutScratch", true, f32, 32u32, 16u32, sg * 512u32);
            threadgroup_barrier();
            // Consume this nj's 32×64 output tile: new_val = OutScratch +
            // U_s, staged into new_state (never straight into tg_state —
            // the NEXT nj still needs the OLD row). 2048 elem / 128 = 16/lane.
            for i2 in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + i2;
                let mr = flat / 64u32; // 0..31 (v, this TG's local tile)
                let nc_local = flat % 64u32; // 0..63 (d local within this nj)
                let sg_m_r = mr / 16u32;
                let sg_n_r = nc_local / 32u32;
                let src_sg = sg_m_r * 2u32 + sg_n_r;
                let local_mr = mr % 16u32;
                let local_nc = nc_local % 32u32;
                let src_off = src_sg * 512u32 + local_mr * 32u32 + local_nc;
                let d_glob = nj * 64u32 + nc_local;
                let d_valid = d_glob < dk;
                let safe_d = select(d_valid, d_glob, 0u32);
                let gemm_val = threadgroup_load("OutScratch", src_off).cast::<f32>();
                let us_val = load(u_s[us_base + (dv_tile_base + mr) * dk + safe_d]).cast::<f32>();
                if d_valid {
                    let slot = nj * 16u32 + i2;
                    stack_store("new_state", slot, gemm_val + us_val);
                }
            }
            threadgroup_barrier();
        }
        // Flush new_state -> tg_state once, after every nj has read the
        // full old row (see doc — the whole reason this isn't fused into
        // the loop above).
        for nj in range(0u32, nj_outer, 1u32) {
            for i2 in range(0u32, 16u32, 1u32) {
                let flat = lane * 16u32 + i2;
                let mr = flat / 64u32;
                let nc_local = flat % 64u32;
                let d_glob = nj * 64u32 + nc_local;
                if d_glob < dk {
                    let slot = nj * 16u32 + i2;
                    let val = stack_load("new_state", slot);
                    threadgroup_store("tg_state", mr * dk + d_glob, val);
                }
            }
        }
        threadgroup_barrier();
    }

    // ── Write final state tile out ─────────────────────────────────────
    for i in range(lane, tile_elems, 128u32) {
        let s = threadgroup_load("tg_state", i);
        store(state_out[state_base + i], s.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    #[test]
    fn dump() {
        use wh_iron::codegen::msl::MslGenerator;
        let mut k = iron_gdn_wy_scan::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }

    /// Debug aid — print the CUDA dynamic-shared-memory byte count for the
    /// single-tile fixture's TG size. See `GDN_PREFILL_CONTRACT.md` §7 —
    /// unlike `iron_gdn_wy_plan`, this kernel's total (98304 B pre-fix)
    /// already fit under GB10's ~99 KiB opt-in cap so it dispatches (just
    /// with wrong values, the actual bug under investigation); this probe
    /// exists to confirm the `SoftwareLocalC` fix (see `backend.rs`'s
    /// `TargetProfile::cuda()` doc) doesn't regress that. CPU-only, no
    /// device needed.
    #[test]
    fn cuda_smem_budget() {
        use wh_iron::codegen::cuda::CudaGenerator;
        let mut k = iron_gdn_wy_scan::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let bytes = CudaGenerator::new().shared_bytes(&k, 128);
        println!(
            "iron_gdn_wy_scan (single-tile TG=128) CUDA dynamic smem: {bytes} bytes ({:.1} KiB)",
            bytes as f64 / 1024.0
        );
    }
}

/// New-syntax correctness for the sequential scan kernel (`iron_gdn_wy_scan`).
/// Oracle re-implements exactly the module doc's per-chunk recurrence —
/// `y = y_local + q_eff·S^T`, `S <- S·P_s + U_s` — directly in f64 against
/// synthetic `(q_eff, y_local, p_s, u_s)` inputs (this kernel's own inputs
/// are plan-kernel *outputs*, not raw q/k/v, so the oracle takes them as
/// given rather than re-deriving them — the plan/scan boundary is validated
/// end-to-end by the pipeline integration test instead).
///
/// Grid (Reduction, 4 SGs/TG): `grid_3d(dv/32, hv, 1, [128,1,1])`.
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_gdn_wy_scan;
    use crate::utils::{pack_f32, unpack_f32};

    /// CPU reference: sequential chunk scan for one `(b, hv)` slot.
    /// `state` is `[dv, dk]`, modified in place; returns `y` `[t_total, dv]`
    /// for this slot only (caller places it into the full `[B,T,Hv,Dv]`
    /// buffer at the right stride).
    #[allow(clippy::too_many_arguments)]
    fn scan_reference(
        q_eff: &[f64],     // [nc, c, dk]
        y_local: &[f64],   // [nc, c, dv]
        p_s: &[f64],       // [nc, dk, dk]
        u_s: &[f64],       // [nc, dv, dk]
        state: &mut [f64], // [dv, dk], in place
        nc: usize,
        c: usize,
        dk: usize,
        dv: usize,
    ) -> Vec<f64> {
        let mut y = vec![0.0_f64; nc * c * dv];
        for chunk in 0..nc {
            let qe_base = chunk * c * dk;
            let yl_base = chunk * c * dv;
            let ps_base = chunk * dk * dk;
            let us_base = chunk * dv * dk;
            // y_pass[t,v] = sum_d q_eff[t,d] * state[v,d]; y = y_pass + y_local.
            for t in 0..c {
                for v in 0..dv {
                    let mut acc = 0.0_f64;
                    for d in 0..dk {
                        acc += q_eff[qe_base + t * dk + d] * state[v * dk + d];
                    }
                    y[yl_base + t * dv + v] = acc + y_local[yl_base + t * dv + v];
                }
            }
            // state_new[v,d] = sum_k state[v,k]*p_s[k,d] + u_s[v,d].
            let mut new_state = vec![0.0_f64; dv * dk];
            for v in 0..dv {
                for d in 0..dk {
                    let mut acc = 0.0_f64;
                    for k in 0..dk {
                        acc += state[v * dk + k] * p_s[ps_base + k * dk + d];
                    }
                    new_state[v * dk + d] = acc + u_s[us_base + v * dk + d];
                }
            }
            state.copy_from_slice(&new_state);
        }
        y
    }

    /// Builds a `(nc, c, dk, dv, hv)` fixture (B=1) with synthetic
    /// `q_eff`/`y_local`/`p_s`/`u_s` inputs, runs `scan_reference` per head,
    /// and wires the GPU dispatch.
    #[allow(clippy::too_many_arguments)]
    fn setup(nc: usize, c: usize, hv: usize, dk: usize, dv: usize, dt: DType) -> TestSetup {
        setup_ex(nc, c, hv, dk, dv, dt, false)
    }

    /// `setup` with an extra `zero_state` knob — when true, `state_in` is
    /// all-zero, which makes GEMM1's `y_pass = q_eff·S_tile^T` term exactly
    /// zero (so `y == y_local` bit-for-bit is the CPU-exact expectation,
    /// independent of GEMM1 correctness beyond "reads zero, contributes
    /// zero") and makes GEMM2's `S_new = S_old·P_s + U_s` collapse to
    /// exactly `U_s` (independent of `P_s`/the multiply). Debug aid for
    /// isolating whether a mismatch is in GEMM1 (`y`) or GEMM2
    /// (`state_out`) — see `GDN_PREFILL_CONTRACT.md` §7.2.
    #[allow(clippy::too_many_arguments)]
    fn setup_ex(
        nc: usize,
        c: usize,
        hv: usize,
        dk: usize,
        dv: usize,
        dt: DType,
        zero_state: bool,
    ) -> TestSetup {
        assert!(dv.is_multiple_of(32), "dv must be a multiple of 32 for v1");
        let t_total = nc * c;
        let n_total = hv; // B=1

        let q_eff: Vec<f32> =
            (0..n_total * nc * c * dk).map(|i| ((i as f32) * 0.021).sin() * 0.4).collect();
        let y_local: Vec<f32> =
            (0..n_total * nc * c * dv).map(|i| ((i as f32) * 0.033).cos() * 0.3).collect();
        // P_s ~ I - small perturbation (keeps the recurrence bounded, mirrors
        // production where P_s = G_C*(I - ...) with G_C < 1).
        let p_s: Vec<f32> = (0..n_total * nc * dk * dk)
            .map(|i| {
                let ii = i % (dk * dk);
                let row = ii / dk;
                let col = ii % dk;
                let diag = if row == col { 0.8 } else { 0.0 };
                diag + 0.01 * ((i as f32) * 0.017).sin()
            })
            .collect();
        let u_s: Vec<f32> =
            (0..n_total * nc * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.05).collect();
        let state_in: Vec<f32> = if zero_state {
            vec![0.0f32; n_total * dv * dk]
        } else {
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.007).cos() * 0.2).collect()
        };

        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (qe_r, yl_r, ps_r, us_r, st_r) =
            (r(&q_eff), r(&y_local), r(&p_s), r(&u_s), r(&state_in));

        let mut y_exp = vec![0.0_f32; t_total * n_total * dv];
        let mut state_exp = vec![0.0_f32; n_total * dv * dk];
        for hv_idx in 0..hv {
            let n = hv_idx;
            let qe_base = n * nc * c * dk;
            let yl_base = n * nc * c * dv;
            let ps_base = n * nc * dk * dk;
            let us_base = n * nc * dv * dk;
            let qe_f64: Vec<f64> =
                qe_r[qe_base..qe_base + nc * c * dk].iter().map(|&x| x as f64).collect();
            let yl_f64: Vec<f64> =
                yl_r[yl_base..yl_base + nc * c * dv].iter().map(|&x| x as f64).collect();
            let ps_f64: Vec<f64> =
                ps_r[ps_base..ps_base + nc * dk * dk].iter().map(|&x| x as f64).collect();
            let us_f64: Vec<f64> =
                us_r[us_base..us_base + nc * dv * dk].iter().map(|&x| x as f64).collect();
            let mut state_f64: Vec<f64> =
                st_r[n * dv * dk..(n + 1) * dv * dk].iter().map(|&x| x as f64).collect();
            let y_slot =
                scan_reference(&qe_f64, &yl_f64, &ps_f64, &us_f64, &mut state_f64, nc, c, dk, dv);
            for t in 0..t_total {
                for v in 0..dv {
                    y_exp[(t * hv + hv_idx) * dv + v] = y_slot[t * dv + v] as f32;
                }
            }
            for idx in 0..dv * dk {
                state_exp[n * dv * dk + idx] = state_f64[idx] as f32;
            }
        }

        TestSetup::new(iron_gdn_wy_scan::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q_eff", pack_f32(&q_eff, dt), dt))
            .input(TestBuffer::from_vec("y_local", pack_f32(&y_local, dt), dt))
            .input(TestBuffer::from_vec("p_s", pack_f32(&p_s, dt), dt))
            .input(TestBuffer::from_vec("u_s", pack_f32(&u_s, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", t_total * n_total * dv, dt))
            .input(TestBuffer::from_vec(
                "t_len",
                (t_total as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("c", c as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d((dv as u32) / 32, n_total as u32, 1, [128, 1, 1])
    }

    // Single Dv-tile (Dv=32 exactly one tile), small Dk, 2 chunks, no GQA
    // concerns (scan doesn't touch hk).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_iron_gdn_wy_scan_single_tile(dt: DType) -> TestSetup { setup(2, 16, 2, 32, 32, dt) }

    // Multi-tile (Dv=128 -> 4 tiles), C=64, 3 chunks — closer to production
    // geometry (Dk=Dv=128 handled at reduced NC for test speed).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_iron_gdn_wy_scan_multi_tile(dt: DType) -> TestSetup { setup(3, 64, 2, 128, 128, dt) }

    // ── Regression fixtures for the CUDA single-tile mismatch (see
    // GDN_PREFILL_CONTRACT.md §7.2 — ROOT-CAUSED AND FIXED: the GEMM1/
    // GEMM2 MMA calls were gated behind a per-SG runtime `if` whose
    // predicate (`sg < c_m_tiles` / `n_tile_valid`) differs *by warp*,
    // and `coop_tile_load_a`/`coop_tile_run`'s CUDA codegen embeds its own
    // `__syncthreads()` — so only some warps in the block executed those
    // internal barriers, a divergent-barrier hazard confirmed via
    // `compute-sanitizer --tool synccheck` ("Barrier error: Divergent
    // thread(s) in block"; `racecheck`/`initcheck` don't catch this
    // class). Fix: run the MMA unconditionally for every SG/`nj` slot —
    // `Xs`/`Ws` are already zero-padded past `C`/`Dk` by the staging
    // loops, so out-of-range SGs just MMA zeros (wasted work, never
    // wrong), and their output is masked out of the consumption step as
    // before. All 4 fixtures below now pass; kept permanently as
    // regression coverage for this exact warp-divergence shape (all are
    // still single-Dv-tile, `dv=32, grid.x=1` — only ONE axis moves per
    // fixture relative to `single_tile` above).

    // nc=1: removes the sequential cross-chunk state carry entirely
    // (single chunk, no recurrence). hv=1 removes n-indexing.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_iron_gdn_wy_scan_debug_nc1(dt: DType) -> TestSetup { setup(1, 16, 1, 32, 32, dt) }

    // c=32 (vs single_tile's c=16): GEMM1's `ceil(c/16)` active-tile count
    // goes from 1 -> 2 (2 of 4 SGs previously idle in GEMM1's MMA instead
    // of just SG0). Still nc=2, hv=2, dk=32 (GEMM2 still only 1 of 4 SGs
    // previously active, `n_tiles_total=1`). Regression coverage for
    // GEMM1's now-unconditional MMA path at a different idle/active SG
    // split than `single_tile`.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_iron_gdn_wy_scan_debug_c32(dt: DType) -> TestSetup { setup(2, 32, 2, 32, 32, dt) }

    // dk=64 (vs single_tile's dk=32): GEMM2's `n_tiles_total` goes from 1
    // -> 2 (all 4 SGs previously active in GEMM2, `nj_outer=1`, both
    // `sg_n_local` 0/1 valid, instead of only SG0/SG2). GEMM1 unchanged
    // (c=16). Regression coverage for GEMM2's now-unconditional MMA path
    // at a different idle/active SG split than `single_tile`.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_iron_gdn_wy_scan_debug_dk64(dt: DType) -> TestSetup { setup(2, 16, 2, 64, 32, dt) }

    // Same shape as `debug_nc1` (nc=1, c=16, hv=1, dk=32, dv=32 — smallest
    // failing fixture) but `state_in` zeroed: `y` must equal `y_local`
    // exactly (GEMM1's contribution is provably zero) and `state_out` must
    // equal `u_s` exactly (GEMM2's `S_old·P_s` term is provably zero).
    // Isolates GEMM1 vs GEMM2 vs the state load/store plumbing.
    #[test_kernel(dtypes = [f32], tol = [5e-3])]
    fn test_iron_gdn_wy_scan_debug_zero_state(dt: DType) -> TestSetup {
        setup_ex(1, 16, 1, 32, 32, dt, true)
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_gdn_wy_scan;

    // Production Qwen3.6-class shape: B=1, T=2048, Hv=16, Dv=Dk=128, C=64
    // (NC=32 chunks). Grid `[4, 16, 1]`, TG `[128,1,1]`.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gdn_wy_scan_t2048(dt: DType) -> BenchSetup {
        let (t, hv, dk, dv, c) = (2048usize, 16usize, 128usize, 128usize, 64usize);
        let n_total = hv;
        let nc = t / c;
        BenchSetup::new(iron_gdn_wy_scan::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q_eff", n_total * nc * c * dk, dt))
            .buffer(BenchBuffer::random("y_local", n_total * nc * c * dv, dt))
            .buffer(BenchBuffer::random("p_s", n_total * nc * dk * dk, dt))
            .buffer(BenchBuffer::random("u_s", n_total * nc * dv * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", t * n_total * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("c", c as u32)
            .grid_3d((dv as u32) / 32, n_total as u32, 1, [128, 1, 1])
            .bytes_moved(
                ((n_total * nc * c * dk
                    + n_total * nc * c * dv
                    + n_total * nc * dk * dk
                    + n_total * nc * dv * dk
                    + n_total * dv * dk * 2
                    + t * n_total * dv)
                    * dt.size_bytes()) as u64,
            )
    }
}

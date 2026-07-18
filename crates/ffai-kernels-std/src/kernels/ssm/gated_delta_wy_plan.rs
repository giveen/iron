//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Chunked-WY GDN prefill — **plan** kernel (pass 1 of 2) — `ffai_gdn_wy_plan`.
//!
//! Two-kernel split of the dormant [`ffai_gated_delta_wy_chunk`](super::gated_delta_wy)
//! that makes the chunked-WY (Woodbury–Young) prefill algorithm runnable at
//! production shapes. The monolithic kernel kept the recurrent `[Dv, Dk]`
//! state in threadgroup memory across its whole chunk-loop — at
//! `Dv = Dk = 128` that's 64 KiB, over Apple's ~32 KiB TG budget, so it can
//! never dispatch for Qwen3.6-class dims. This kernel (`_plan`) and its
//! sibling [`ffai_gdn_wy_scan`](super::gated_delta_wy_scan) (`_scan`) fix that
//! by separating the **chunk-parallel, state-independent** algebra (here)
//! from the **chunk-sequential, state-dependent** scan (there). Neither
//! kernel ever holds a `[Dv, Dk]`-shaped buffer in TG memory.
//!
//! ## Math (per chunk of size `C`, single `(b, hv)` slot; mirrors
//! `gated_delta_wy.rs`'s steps 1–9 and the CPU oracle in
//! `tests/gated_delta_wy_cpu_oracle.rs::process_chunk_one_head`)
//!
//! 1. Prefix gates `G_t = Π_{i≤t} g_i` — accumulated in **log space**
//!    (`L_t = Σ log g_i`; `Γ[t,j] = exp(L_t − L_j)`, `G_t = exp(L_t)`).
//!    Production gate magnitudes (`exp(a_log)` ~[1,16] → per-token
//!    `g` ~e⁻⁴) underflow a direct 64-token f32 product to 0, turning
//!    every `Γ = G_t/G_j` into `0/0 = NaN`; log space keeps the ratios
//!    exact and lets absolute `G` multipliers flush gracefully to 0.
//! 2. Solve `(I + L) p = K`, `L[t,j] = β_j · KKT[t,j]` for `j < t`
//!    (unit-lower-triangular forward substitution; `KKT[i,j] = k_i·k_j`).
//! 3. `P_s = G_C · (I − (β⊙p)^T K) ∈ [Dk, Dk]` — this is the *state-free*
//!    factor of step 9's `S_end` passthrough term: since
//!    `S_through = G_C·(S_0 − S_0·(β⊙p)^T·K) = S_0 · P_s`, the chunk can
//!    compute `P_s` without ever touching `S_0`. `S_0 · P_s` is left for
//!    `_scan` (it owns the only `S_0` that's ever materialised).
//! 4. `q_eff[t] = G_t · (q_t − Σ_{i≤t} β_i·QKT[t,i]·p_i) ∈ [Dk]` — the
//!    state-free factor of step 7's `y_pass`: since
//!    `y_pass[t] = G_t·(S_0·q_t − Σ_{i≤t}β_i·QKT[t,i]·S_0·p_i) = S_0·q_eff[t]`,
//!    `_scan` recovers `y_pass[t] = q_eff[t] · S_0^T` without ever expanding
//!    the `Σ β_i QKT p_i` sum against `S_0` directly.
//! 5. Solve `(I + A) u = β⊙V`, `A[t,j] = β_t·Γ[t,j]·KKT[t,j]` for `j < t`,
//!    `Γ[t,j] = G_t/G_j` — this **is** the `u` output (step 5's `u^v`).
//! 6. `y_local[t] = Σ_{j≤t} Γ[t,j]·QKT[t,j]·u_j ∈ [Dv]` (step 6, unchanged).
//! 7. `U_s = Σ_j (G_C/G_j)·u_j⊗k_j ∈ [Dv, Dk]` (step 9's rank-update term,
//!    unchanged — already state-free).
//!
//! `_scan` combines these with the running state: `y = y_local +
//! q_eff·S_0^T` and `S_end = S_0·P_s + U_s`. See
//! [`ffai_gdn_wy_scan`](super::gated_delta_wy_scan) for the derivation check
//! and the sequential chunk-loop.
//!
//! ## Layout
//!
//!   - `q, k`:    `[B, T, Hk, Dk]`
//!   - `v`:       `[B, T, Hv, Dv]`
//!   - `g, beta`: `[B, T, Hv]` — **already-computed** per-token decay/gate
//!     (`g = exp(-exp(a_log)·softplus(a_raw+dt_bias))`, `beta =
//!     sigmoid(b_raw)`), matching `gated_delta_wy.rs`'s convention — *not*
//!     `gated_delta_prep_chunk.rs`'s raw `a_raw`/`b_raw` + `a_log`/`dt_bias`.
//!     FFAI computes `g`/`beta` upstream (the qknorm/prep pass) before this
//!     pipeline runs.
//!   - `t_len`: `[1]` u32 — runtime chunk length. **`t_len % c == 0`
//!     required for v1** (FFAI pads a short prefill tail up to a multiple of
//!     `c` upstream, `g = 1, β = 0` no-op tokens, same contract the
//!     monolithic kernel documented).
//!   - Outputs, all per `(chunk, head)` — `N = B·Hv`, `NC = t_len / c`:
//!     - `u`       : `[N, NC, C, Dv]`
//!     - `y_local` : `[N, NC, C, Dv]`
//!     - `q_eff`   : `[N, NC, C, Dk]`
//!     - `p_s`     : `[N, NC, Dk, Dk]`
//!     - `u_s`     : `[N, NC, Dv, Dk]`
//!
//!   `n = b·Hv + hv_idx` (same grouping as `state_in`/`state_out`'s
//!   `[B, Hv, Dv, Dk]`). GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** Grid: `[T/C, B·Hv, 1]` — **parallel over chunks**
//!   (no cross-chunk state here, unlike the monolithic kernel — every
//!   `(chunk, head)` slot is independent). TG: `[512, 1, 1]` (16 SGs).
//! - `tgid_x` = chunk index, `tgid_y` = `n = b·Hv + hv_idx`, `tid` = lane
//!   (0..511, spans the full TG — NOT `simd_lane`, which would alias to
//!   0..31 across the 16 SGs).
//! - `dk`, `dv` ≤ 128 (the fixed TG-buffer literals below size for
//!   Qwen3.6-class `Dk = Dv = 128`); `c` ≤ 64.
//! - **`dv % 16 == 0` and `dk % 16 == 0`** — the MMA path tiles both in
//!   16-row/16-K units. Production `Dk = Dv = 128` always satisfied this;
//!   there is deliberately no scalar fallback for smaller `dv`/`dk`
//!   (staging zero-pads partial 32-wide N-tiles, so non-multiples of 32
//!   are fine, but 16 is the contract floor). `c` may be any value ≤ 64
//!   (K-blocks over the chunk axis zero-pad past `c`).
//! - `Hv` divisible by `Hk`; `t_len` a runtime u32, must be a multiple of
//!   `c`.
//!
//! ## TG memory budget (the whole reason this kernel exists)
//!
//! No `[Dv, Dk]`-shaped buffer is ever TG-resident. v2 (MMA) layout:
//!   - `tg_big_g`: `C` × f32 = 64 × 4 B = 256 B — log-space prefix gates
//!     (`L_t = Σ log g`, see step 1 in the math section).
//!   - `tg_beta`: `C` × f32 = 256 B — per-token β, loaded once (v1 re-read
//!     it from global inside every inner loop).
//!   - `tg_solve`: `64 × 128` × f16 = 16 KiB — the forward-substitution
//!     solution, at a **fixed LD of 128** regardless of `dk`/`dv` (so MMA
//!     tensor views, whose leading dimension must be a compile-time
//!     literal, can read it directly; unused tail columns are
//!     zero-initialised so padded B-tiles read 0). Reused in place: `p`
//!     (steps 2–4) is fully consumed before step 5 overwrites it with `u`.
//!     f16 staging (10-bit mantissa, up from v1's bf16 7-bit — strictly
//!     better precision at the same size): `p`/`u` are chunk-local scratch
//!     re-derived fresh from full-precision inputs every chunk, so one
//!     reduced-precision round-trip is a mild, non-compounding rounding —
//!     the same argument the v1 doc made for bf16, now with margin.
//!   - `tg_cache`: `[64, 64]` × f16 = 8 KiB — **one combined triangle
//!     cache for both Gram matrices**: `cache[t][j] = QKT[t,j]` for
//!     `j <= t` (lower + diag; steps 4/6 read exactly that triangle) and
//!     `cache[t][j] = KKT[t,j]` for `j > t` (strict upper; steps 2/5 need
//!     `KKT[t,j]` for `j < t`, which by symmetry of `K·Kᵀ` is
//!     `cache[j][t]`). The two consumers' triangles are exactly
//!     complementary, so one `[C,C]` buffer holds both — that packing is
//!     what lets the cache fit next to `tg_solve` at all.
//!   - `Xs`: 2048 × f16 = 4 KiB — MMA staging (q/k `[64,16]` slices).
//!   - `OutScratch`: 512 × f32 = 2 KiB — a single 16×32 MMA C-tile
//!     (f32: Metal's `ct_c.store()` requires dest dtype == acc dtype, see
//!     `gated_delta_wy_scan.rs`). Tiles are flushed through it one SG at
//!     a time (serialised rounds) — the price of keeping it single-tile.
//!   - Total: 256 + 256 + 16384 + 8192 + 4096 + 2048 = **31 KiB**
//!     (30.5 KiB + change), under the 32 KiB cap with ~1 KiB headroom.
//!
//! ## v2 — MMA'd Gram cache + barrier-free solves
//!
//! v1 recomputed `KKT`/`QKT` rows cooperatively at every use site (four
//! `O(C²·Dk)` passes: steps 2, 4, 5, 6) with 2 threadgroup barriers per
//! row — ~512 barriers per chunk at `C = 64`. v2:
//!   - computes each Gram matrix **once** via `coop_tile` MMA (16×32×16
//!     tiles; SGs 0..7 own KKT tiles, 8..15 own QKT tiles, one fused
//!     K-loop over `Dk`), masked into the combined triangle cache;
//!   - restructures both triangular solves so lane `d` owns column `d`
//!     for every row: all cross-row dependencies (`p[j,d]`, `u[j,d]`,
//!     `j < t`) become *same-lane* reads, and the row-cache barriers
//!     vanish entirely — the solves now run **barrier-free** (one barrier
//!     after each solve completes, instead of 2·C during it);
//!   - flattens steps 4 and 6 from row-sequential (barrier-paced) loops
//!     into fully parallel `(t,d)` loops reading the cache.
//!
//! The triangular solves themselves stay scalar by design: they are
//! `O(C²·max(Dk,Dv))` with a strict row-order dependency — the classic
//! blocked-TRSM (scalar 16×16 diagonal blocks + MMA rank-16 updates)
//! is the known next lever if they dominate post-MMA profiles.

#![allow(clippy::too_many_arguments)]

use ffai_kernels::kernel;

#[kernel]
pub fn ffai_gdn_wy_plan<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    g: Tensor<T>,
    beta: Tensor<T>,
    t_len: Tensor<u32>,
    mut u: Tensor<T>,
    mut y_local: Tensor<T>,
    mut q_eff: Tensor<T>,
    mut p_s: Tensor<T>,
    mut u_s: Tensor<T>,
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
    #[constexpr] c: u32,
) {
    // ── Geometry ───────────────────────────────────────────────────────
    let chunk_idx = tgid_x;
    let n = tgid_y;
    let b_idx = n / hv;
    let hv_idx = n - b_idx * hv;
    let hv_per_hk = hv / hk;
    let hk_idx = hv_idx / hv_per_hk;
    let lane = tid;
    let sg = simd_group_id(); // 0..15
    let chunk_start = chunk_idx * c;
    let num_chunks = load(t_len[0]) / c;
    let nc_base = n * num_chunks + chunk_idx;

    // ── TG buffers (see module doc for the budget) ─────────────────────
    threadgroup_alloc("tg_big_g", 64u32, f32);
    threadgroup_alloc("tg_beta", 64u32, f32);
    threadgroup_alloc("tg_solve", 8192u32, f16); // [C, LD=128] — fixed LD (see doc)
    threadgroup_alloc("tg_cache", 4096u32, f16); // [64, 64] QKT-lower / KKT-upper
    threadgroup_alloc("Xs", 2048u32, f16); // MMA staging (q/k slices)
    threadgroup_alloc("OutScratch", 512u32, f32); // one 16×32 f32 C-tile
    // KKT/QKT GEMM: 16×32×32 tiles (the same M/N/K shape every proven MMA
    // kernel in this codebase uses), tb=true (B = k slice in its natural
    // [rows, Dk-slice] layout, which is [N, K] for these Gram products).
    coop_tile_setup("gA", 16, 32, 32, f16, "accumulate", "simdgroup", f32, false, true, false);
    // Output GEMMs (steps 3/7): 16×32×16 tiles, tb=false — the K axis is
    // the chunk (t/j) axis here, and B = a k slice in [K, N] layout. K=16
    // (not 32) so the transpose-staged A block and the B block fit in the
    // single 2048-element `Xs` slab together (1024 + 1024).
    coop_tile_setup("gB", 16, 32, 16, f16, "accumulate", "simdgroup", f32, false, false, false);

    // ── Step 1: prefix gates in LOG space — tg_big_g[i] holds
    // L_i = Σ_{k<=i} log g_k, NOT the product G_i itself. Production GDN
    // gates (`g = exp(-exp(a_log)·dt)`, `exp(a_log)` init'd in ~[1,16])
    // can be ~e⁻⁴ per token, so the direct 64-token prefix product
    // underflows f32 (~e⁻²⁵⁶ vs f32 min ~1e⁻³⁸) and every Γ = G_t/G_j
    // ratio below becomes 0/0 = NaN. In log space the ratios are
    // exp(L_t − L_j) — always ≤ 1 (L is non-increasing) and gracefully
    // flush to 0 for distant (t, j) pairs instead of NaN'ing. Absolute
    // G multipliers (G_t on q_eff, G_C on P_s/U_s) become exp(L_·),
    // whose underflow-to-0 is the mathematically correct limit (fully
    // decayed passthrough). g is clamped at 1e-30 before the log so a
    // flushed-to-zero input gate contributes a finite −69 rather than
    // −inf (−inf − −inf = NaN in the ratios).
    // beta cached to TG once (read in every inner loop below); tg_solve
    // zero-initialised so columns ≥ dk/dv read as 0 later (padded MMA
    // B-tiles and masked copies rely on this).
    if lane == 0u32 {
        let mut l_acc = 0.0f32;
        for i in range(0u32, c, 1u32) {
            let t_abs = chunk_start + i;
            let g_val = load(g[t_abs * hv + hv_idx]).cast::<f32>();
            let g_safe = select(g_val > 1e-30f32, g_val, 1e-30f32);
            l_acc = l_acc + log(g_safe);
            threadgroup_store("tg_big_g", i, l_acc);
        }
    }
    for i in range(lane, c, 512u32) {
        let b_val = load(beta[(chunk_start + i) * hv + hv_idx]).cast::<f32>();
        threadgroup_store("tg_beta", i, b_val);
    }
    // (tg_solve is zero-initialised at the END of phase 1, which also uses
    // its first 2 KiB as the q staging slab — see below.)
    threadgroup_barrier();

    // ── Phase 1: KKT/QKT cache build via MMA ───────────────────────────
    // One combined [64,64] f16 cache (see module doc): lower+diag =
    // QKT[t,j] (j<=t), strict upper = KKT[t,j] (j>t; KKT is symmetric so
    // KKT[t,j] for j<t is read as cache[j][t]). Both Gram matrices are
    // computed in a single fused K-loop: SGs 0..7 own KKT tiles (A = k
    // slice), SGs 8..15 own QKT tiles (A = q slice); B = k slice for both.
    // Tiles whose triangle contribution is empty (or beyond C) skip.
    // Phase 1 staging: k slice -> Xs [64,32] LD=32; q slice -> the first
    // 2048 elements of tg_solve (same [64,32] LD=32 layout) -- tg_solve is
    // dead until step 2, so it doubles as the q staging slab here and is
    // zero-initialised right after (which step 2's padded-column contract
    // needs anyway). Rows >= c and columns >= dk zero-padded.
    coop_tile_zero("gA");
    for kb in range(0u32, dk, 32u32) {
        for f in range(lane, 2048u32, 512u32) {
            let row = f / 32u32;
            let col = f % 32u32;
            let valid = (row < c) & (kb + col < dk);
            let safe_row = select(row < c, row, 0u32);
            let safe_col = select(kb + col < dk, kb + col, 0u32);
            let addr = ((chunk_start + safe_row) * hk + hk_idx) * dk + safe_col;
            let kv = load(k[addr]).cast::<f32>();
            threadgroup_store("Xs", f, select(valid, kv, 0.0f32).cast::<f16>());
            let qv = load(q[addr]).cast::<f32>();
            threadgroup_store("tg_solve", f, select(valid, qv, 0.0f32).cast::<f16>());
        }
        threadgroup_barrier();
        let sl = sg % 8u32;
        let mt = sl / 2u32;
        let nt = sl % 2u32;
        let row0 = mt * 16u32;
        let col0 = nt * 32u32;
        let tri_kkt = select(col0 + 31u32 > row0, 1u32, 0u32);
        let tri_qkt = select(col0 <= row0 + 15u32, 1u32, 0u32);
        if (sg < 8u32) & (row0 < c) & (col0 < c) & (tri_kkt == 1u32) {
            coop_tile_load_a("gA", "Xs", true, f16, 32u32, 16u32, mt * 512u32);
            coop_tile_load_b("gA", "Xs", true, f16, 32u32, 32u32, nt * 1024u32);
            coop_tile_run("gA");
        }
        if (sg >= 8u32) & (row0 < c) & (col0 < c) & (tri_qkt == 1u32) {
            coop_tile_load_a("gA", "tg_solve", true, f16, 32u32, 16u32, mt * 512u32);
            coop_tile_load_b("gA", "Xs", true, f16, 32u32, 32u32, nt * 1024u32);
            coop_tile_run("gA");
        }
        threadgroup_barrier();
    }
    // Serialised C-tile flush through the single-tile OutScratch: round s
    // stores SG s's tile, then all 512 lanes copy one element each into
    // the cache with the triangle + range mask applied.
    for s in range(0u32, 16u32, 1u32) {
        let sl = s % 8u32;
        let mt = sl / 2u32;
        let nt = sl % 2u32;
        let row0 = mt * 16u32;
        let col0 = nt * 32u32;
        let is_kkt = s < 8u32;
        let tri_kkt = select(col0 + 31u32 > row0, 1u32, 0u32);
        let tri_qkt = select(col0 <= row0 + 15u32, 1u32, 0u32);
        let tri_ok = select(is_kkt, tri_kkt, tri_qkt);
        if (row0 < c) & (col0 < c) & (tri_ok == 1u32) {
            if sg == s {
                coop_tile_store_c("gA", "OutScratch", true, f32, 32u32, 16u32);
            }
            threadgroup_barrier();
            let r = lane / 32u32;
            let cc = lane % 32u32;
            let grow = row0 + r;
            let gcol = col0 + cc;
            let tri_k2 = select(gcol > grow, 1u32, 0u32);
            let tri_q2 = select(gcol <= grow, 1u32, 0u32);
            let tri2 = select(is_kkt, tri_k2, tri_q2);
            if (grow < c) & (gcol < c) & (tri2 == 1u32) {
                let cv = threadgroup_load("OutScratch", lane);
                threadgroup_store("tg_cache", grow * 64u32 + gcol, cv.cast::<f16>());
            }
            threadgroup_barrier();
        }
    }
    // Zero tg_solve (clears the q staging residue AND establishes the
    // "unwritten columns read 0" contract steps 4/6 rely on for dk/dv<128).
    for i in range(lane, 8192u32, 512u32) {
        threadgroup_store("tg_solve", i, 0.0f32.cast::<f16>());
    }
    threadgroup_barrier();

    // ── Step 2: solve (I+L) p = K, L[t,j] = beta_j * KKT[t,j], j<t. ────
    // Forward substitution reading KKT from the cache (upper triangle,
    // symmetric). Lane d owns column d for EVERY row t, so p[j,d] reads
    // are same-lane writes — **no barrier inside the loop at all** (the
    // v1 kernel needed 2 per row for its cooperative row cache).
    for t in range(0u32, c, 1u32) {
        let t_abs = chunk_start + t;
        for d in range(lane, dk, 512u32) {
            let mut accum = load(k[(t_abs * hk + hk_idx) * dk + d]).cast::<f32>();
            for j in range(0u32, t, 1u32) {
                let beta_j = threadgroup_load("tg_beta", j);
                let kkt_tj = threadgroup_load("tg_cache", j * 64u32 + t).cast::<f32>();
                let p_jd = threadgroup_load("tg_solve", j * 128u32 + d).cast::<f32>();
                accum = accum - beta_j * kkt_tj * p_jd;
            }
            threadgroup_store("tg_solve", t * 128u32 + d, accum.cast::<f16>());
        }
    }
    threadgroup_barrier();

    // ── Step 3: P_s[i,j] = G_C * (delta_ij - sum_t beta_t*p[t,i]*k[t,j]).
    // MMA over 64×64 (M-half × N-half) output blocks: A = (β⊙p)ᵀ,
    // transpose-staged from tg_solve with β folded in during staging
    // (Xs[0..1024], [64,16] LD=16); B = k slice ([16,64] LD=64 at
    // Xs[1024..2048]). K axis = chunk axis, 16-wide blocks (zero-padded
    // past c). SGs 0..7 own the 4×2 tile grid of each 64×64 block; the
    // G_C·(δ − raw) transform is applied during the serialised C-tile
    // flush straight to global `p_s`.
    let l_end = threadgroup_load("tg_big_g", c - 1u32); // L_{C-1} (log G_C)
    let big_g_c = exp(l_end);
    let ps_base = nc_base * dk * dk;
    let half_count3 = (dk + 63u32) / 64u32;
    for mh in range(0u32, half_count3, 1u32) {
        for nh in range(0u32, half_count3, 1u32) {
            coop_tile_zero("gB");
            for kb in range(0u32, c, 16u32) {
                for f in range(lane, 1024u32, 512u32) {
                    let il = f / 16u32;
                    let tl = f % 16u32;
                    let i_g = mh * 64u32 + il;
                    let t_g = kb + tl;
                    let ok_a = (i_g < dk) & (t_g < c);
                    let safe_i = select(i_g < dk, i_g, 0u32);
                    let safe_t = select(t_g < c, t_g, 0u32);
                    let bt = threadgroup_load("tg_beta", safe_t);
                    let pv = threadgroup_load("tg_solve", safe_t * 128u32 + safe_i).cast::<f32>();
                    threadgroup_store("Xs", f, select(ok_a, bt * pv, 0.0f32).cast::<f16>());
                    let jl = f % 64u32;
                    let tl_b = f / 64u32;
                    let j_g = nh * 64u32 + jl;
                    let tb_g = kb + tl_b;
                    let ok_b = (j_g < dk) & (tb_g < c);
                    let safe_j = select(j_g < dk, j_g, 0u32);
                    let safe_tb = select(tb_g < c, tb_g, 0u32);
                    let kv = load(k[((chunk_start + safe_tb) * hk + hk_idx) * dk + safe_j])
                        .cast::<f32>();
                    threadgroup_store("Xs", 1024u32 + f, select(ok_b, kv, 0.0f32).cast::<f16>());
                }
                threadgroup_barrier();
                let mt3 = sg / 2u32;
                let nt3 = sg % 2u32;
                if (sg < 8u32) & (mh * 64u32 + mt3 * 16u32 < dk) & (nh * 64u32 + nt3 * 32u32 < dk) {
                    coop_tile_load_a("gB", "Xs", true, f16, 16u32, 16u32, mt3 * 256u32);
                    coop_tile_load_b("gB", "Xs", true, f16, 64u32, 16u32, 1024u32 + nt3 * 32u32);
                    coop_tile_run("gB");
                }
                threadgroup_barrier();
            }
            for s3r in range(0u32, 8u32, 1u32) {
                let mt3 = s3r / 2u32;
                let nt3 = s3r % 2u32;
                let row0 = mh * 64u32 + mt3 * 16u32;
                let col0 = nh * 64u32 + nt3 * 32u32;
                if (row0 < dk) & (col0 < dk) {
                    if sg == s3r {
                        coop_tile_store_c("gB", "OutScratch", true, f32, 32u32, 16u32);
                    }
                    threadgroup_barrier();
                    let i_g = row0 + lane / 32u32;
                    let j_g = col0 + lane % 32u32;
                    if (i_g < dk) & (j_g < dk) {
                        let raw = threadgroup_load("OutScratch", lane);
                        let delta_ij = select(i_g == j_g, 1.0f32, 0.0f32);
                        let val = big_g_c * (delta_ij - raw);
                        store(p_s[ps_base + i_g * dk + j_g], val.cast::<T>());
                    }
                    threadgroup_barrier();
                }
            }
        }
    }

    // ── Step 4: q_eff[t,d] = G_t*(q_t[d] - sum_{i<=t} beta_i*QKT[t,i]*p[i,d]).
    // Flat parallel over (t,d) — QKT comes from the cache (lower+diag),
    // so there is no per-row cooperative compute and no barrier: rows are
    // independent given p.
    // MMA: corr = W·p with W[t,i] = β_i·QKT[t,i] masked to i ≤ t, staged
    // from the cache into Xs ([64,16] LD=16); **B = p read directly from
    // `tg_solve`** — no B staging at all, which is exactly what the fixed
    // LD-128 layout buys. M = C, N = Dk (up to 4×4 = 16 tiles → all 16
    // SGs), K = chunk axis. The G_t·(q − corr) transform runs in the
    // serialised flush.
    let qeff_base = nc_base * c * dk;
    let n_tiles4 = (dk + 31u32) / 32u32;
    let m_tiles4 = (c + 15u32) / 16u32;
    let total4 = m_tiles4 * n_tiles4;
    coop_tile_zero("gB");
    for kb in range(0u32, c, 16u32) {
        for f in range(lane, 1024u32, 512u32) {
            let tr = f / 16u32;
            let il = f % 16u32;
            let i_g = kb + il;
            let ok_w = (tr < c) & (i_g <= tr);
            let safe_i = select(i_g < 64u32, i_g, 0u32);
            let beta_i = threadgroup_load("tg_beta", safe_i);
            let qkt_ti = threadgroup_load("tg_cache", tr * 64u32 + safe_i).cast::<f32>();
            threadgroup_store("Xs", f, select(ok_w, beta_i * qkt_ti, 0.0f32).cast::<f16>());
        }
        threadgroup_barrier();
        let mt4 = sg / n_tiles4;
        let nt4 = sg % n_tiles4;
        if (sg < total4) & (mt4 * 16u32 < c) {
            coop_tile_load_a("gB", "Xs", true, f16, 16u32, 16u32, mt4 * 256u32);
            coop_tile_load_b("gB", "tg_solve", true, f16, 128u32, 16u32, kb * 128u32 + nt4 * 32u32);
            coop_tile_run("gB");
        }
        threadgroup_barrier();
    }
    for s4r in range(0u32, 16u32, 1u32) {
        if s4r < total4 {
            let mt4 = s4r / n_tiles4;
            let nt4 = s4r % n_tiles4;
            let row0 = mt4 * 16u32;
            let col0 = nt4 * 32u32;
            if row0 < c {
                if sg == s4r {
                    coop_tile_store_c("gB", "OutScratch", true, f32, 32u32, 16u32);
                }
                threadgroup_barrier();
                let t_g = row0 + lane / 32u32;
                let d_g = col0 + lane % 32u32;
                if (t_g < c) & (d_g < dk) {
                    let corr = threadgroup_load("OutScratch", lane);
                    let big_g_t = exp(threadgroup_load("tg_big_g", t_g));
                    let q_td =
                        load(q[((chunk_start + t_g) * hk + hk_idx) * dk + d_g]).cast::<f32>();
                    store(q_eff[qeff_base + t_g * dk + d_g], (big_g_t * (q_td - corr)).cast::<T>());
                }
                threadgroup_barrier();
            }
        }
    }
    threadgroup_barrier();

    // ── Step 5: solve (I+A) u = beta⊙V, A[t,j]=beta_t*Gamma[t,j]*KKT[t,j].
    // `tg_solve` is fully drained of `p` by steps 3–4 above (the barrier
    // right before this comment is what makes that safe); overwrite with
    // `u`. Same barrier-free column-ownership structure as step 2.
    for t in range(0u32, c, 1u32) {
        let t_abs = chunk_start + t;
        let beta_t = threadgroup_load("tg_beta", t);
        let l_t = threadgroup_load("tg_big_g", t);
        for d in range(lane, dv, 512u32) {
            let v_td = load(v[(t_abs * hv + hv_idx) * dv + d]).cast::<f32>();
            let mut accum = beta_t * v_td;
            for j in range(0u32, t, 1u32) {
                let l_j = threadgroup_load("tg_big_g", j);
                let kkt_tj = threadgroup_load("tg_cache", j * 64u32 + t).cast::<f32>();
                let uv_jd = threadgroup_load("tg_solve", j * 128u32 + d).cast::<f32>();
                // Γ[t,j] = G_t/G_j = exp(L_t − L_j) — see step 1's
                // log-space rationale (the direct ratio NaNs on
                // production gate magnitudes).
                accum = accum - beta_t * exp(l_t - l_j) * kkt_tj * uv_jd;
            }
            threadgroup_store("tg_solve", t * 128u32 + d, accum.cast::<f16>());
        }
    }
    threadgroup_barrier();

    // ── Step 6: y_local[t,d] = sum_{j<=t} Gamma[t,j]*QKT[t,j]*u[j,d]; and
    // the `u` output is just `tg_solve` itself (write-through copy).
    // Same MMA shape as step 4: A = W6[t,j] = Γ[t,j]·QKT[t,j] masked to
    // j ≤ t (staged from cache), B = u direct from tg_solve, N = Dv.
    let u_base = nc_base * c * dv;
    let n_tiles6 = (dv + 31u32) / 32u32;
    let total6 = m_tiles4 * n_tiles6;
    coop_tile_zero("gB");
    for kb in range(0u32, c, 16u32) {
        for f in range(lane, 1024u32, 512u32) {
            let tr = f / 16u32;
            let jl = f % 16u32;
            let j_g = kb + jl;
            let ok_w = (tr < c) & (j_g <= tr);
            let safe_j = select(j_g < 64u32, j_g, 0u32);
            let safe_t = select(tr < c, tr, 0u32);
            let l_t6 = threadgroup_load("tg_big_g", safe_t);
            let l_j6 = threadgroup_load("tg_big_g", safe_j);
            let qkt_tj = threadgroup_load("tg_cache", tr * 64u32 + safe_j).cast::<f32>();
            // Γ[t,j] in log space (see step 1) — exp(L_t − L_j).
            let w6 = exp(l_t6 - l_j6) * qkt_tj;
            threadgroup_store("Xs", f, select(ok_w, w6, 0.0f32).cast::<f16>());
        }
        threadgroup_barrier();
        let mt6 = sg / n_tiles6;
        let nt6 = sg % n_tiles6;
        if (sg < total6) & (mt6 * 16u32 < c) {
            coop_tile_load_a("gB", "Xs", true, f16, 16u32, 16u32, mt6 * 256u32);
            coop_tile_load_b("gB", "tg_solve", true, f16, 128u32, 16u32, kb * 128u32 + nt6 * 32u32);
            coop_tile_run("gB");
        }
        threadgroup_barrier();
    }
    for s6r in range(0u32, 16u32, 1u32) {
        if s6r < total6 {
            let mt6 = s6r / n_tiles6;
            let nt6 = s6r % n_tiles6;
            let row0 = mt6 * 16u32;
            let col0 = nt6 * 32u32;
            if row0 < c {
                if sg == s6r {
                    coop_tile_store_c("gB", "OutScratch", true, f32, 32u32, 16u32);
                }
                threadgroup_barrier();
                let t_g = row0 + lane / 32u32;
                let d_g = col0 + lane % 32u32;
                if (t_g < c) & (d_g < dv) {
                    let yl = threadgroup_load("OutScratch", lane);
                    store(y_local[u_base + t_g * dv + d_g], yl.cast::<T>());
                }
                threadgroup_barrier();
            }
        }
    }
    // u output copy (flat, unchanged): tg_solve rows are the u values.
    for td in range(lane, c * dv, 512u32) {
        let t = td / dv;
        let d = td % dv;
        let uv_td = threadgroup_load("tg_solve", t * 128u32 + d).cast::<f32>();
        store(u[u_base + td], uv_td.cast::<T>());
    }

    // ── Step 7: U_s[v,d] = sum_j (G_C/G_j)*u[j,v]*k[j,d]. Same MMA shape
    // as step 3: A = (rw⊙u)ᵀ transpose-staged with the G_C/G_j row weight
    // folded in; B = k slice; K axis = chunk axis. No output transform.
    threadgroup_barrier();
    let us_base = nc_base * dv * dk;
    let mh_count7 = (dv + 63u32) / 64u32;
    let nh_count7 = (dk + 63u32) / 64u32;
    for mh in range(0u32, mh_count7, 1u32) {
        for nh in range(0u32, nh_count7, 1u32) {
            coop_tile_zero("gB");
            for kb in range(0u32, c, 16u32) {
                for f in range(lane, 1024u32, 512u32) {
                    let vl = f / 16u32;
                    let jl = f % 16u32;
                    let v_g = mh * 64u32 + vl;
                    let j_g = kb + jl;
                    let ok_a = (v_g < dv) & (j_g < c);
                    let safe_v = select(v_g < dv, v_g, 0u32);
                    let safe_j = select(j_g < c, j_g, 0u32);
                    let l_j7 = threadgroup_load("tg_big_g", safe_j);
                    let uv = threadgroup_load("tg_solve", safe_j * 128u32 + safe_v).cast::<f32>();
                    // G_C/G_j = exp(L_end − L_j) (log space, see step 1).
                    let a_val = exp(l_end - l_j7) * uv;
                    threadgroup_store("Xs", f, select(ok_a, a_val, 0.0f32).cast::<f16>());
                    let dl = f % 64u32;
                    let jl_b = f / 64u32;
                    let d_g = nh * 64u32 + dl;
                    let jb_g = kb + jl_b;
                    let ok_b = (d_g < dk) & (jb_g < c);
                    let safe_d = select(d_g < dk, d_g, 0u32);
                    let safe_jb = select(jb_g < c, jb_g, 0u32);
                    let kv = load(k[((chunk_start + safe_jb) * hk + hk_idx) * dk + safe_d])
                        .cast::<f32>();
                    threadgroup_store("Xs", 1024u32 + f, select(ok_b, kv, 0.0f32).cast::<f16>());
                }
                threadgroup_barrier();
                let mt7 = sg / 2u32;
                let nt7 = sg % 2u32;
                if (sg < 8u32) & (mh * 64u32 + mt7 * 16u32 < dv) & (nh * 64u32 + nt7 * 32u32 < dk) {
                    coop_tile_load_a("gB", "Xs", true, f16, 16u32, 16u32, mt7 * 256u32);
                    coop_tile_load_b("gB", "Xs", true, f16, 64u32, 16u32, 1024u32 + nt7 * 32u32);
                    coop_tile_run("gB");
                }
                threadgroup_barrier();
            }
            for s7r in range(0u32, 8u32, 1u32) {
                let mt7 = s7r / 2u32;
                let nt7 = s7r % 2u32;
                let row0 = mh * 64u32 + mt7 * 16u32;
                let col0 = nh * 64u32 + nt7 * 32u32;
                if (row0 < dv) & (col0 < dk) {
                    if sg == s7r {
                        coop_tile_store_c("gB", "OutScratch", true, f32, 32u32, 16u32);
                    }
                    threadgroup_barrier();
                    let v_g = row0 + lane / 32u32;
                    let d_g = col0 + lane % 32u32;
                    if (v_g < dv) & (d_g < dk) {
                        let raw = threadgroup_load("OutScratch", lane);
                        store(u_s[us_base + v_g * dk + d_g], raw.cast::<T>());
                    }
                    threadgroup_barrier();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ffai_kernels::core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    #[test]
    fn dump() {
        use ffai_kernels::codegen::msl::MslGenerator;
        let mut k = ffai_gdn_wy_plan::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

/// New-syntax correctness for the chunk-parallel plan kernel
/// (`ffai_gdn_wy_plan`). Oracle mirrors
/// `tests/gated_delta_wy_cpu_oracle.rs::process_chunk_one_head`'s steps
/// 1–6 and 8, but stops at the *state-free* factors this kernel actually
/// emits (`u`, `y_local`, `q_eff`, `p_s`, `u_s`) rather than combining them
/// with a running state — see the module doc for the `P_s`/`q_eff`
/// derivation that makes this split valid.
///
/// Tolerances are wider than a typical f32-tier kernel across *every*
/// dtype cell: `tg_solve` (the forward-substitution scratch for `p`/`u`)
/// is always bf16-staged internally regardless of the kernel's generic
/// `T`, so even the f32 cell inherits a bf16-scale rounding on `p`/`u` (and
/// everything downstream of them: `q_eff`, `y_local`, `p_s`, `u_s`).
///
/// Grid (Reduction, 16 SGs/TG): `grid_3d(t/c, hv, 1, [512,1,1])`.
pub mod kernel_tests {
    use ffai_kernels::{test::*, test_kernel};

    use super::ffai_gdn_wy_plan;
    use crate::utils::{pack_f32, unpack_f32};

    /// Row-major matmul: `C[m,n] = A[m,k] * B[k,n]`. f64 internal.
    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f64;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    /// Row-major `A^T`: input `[m,n]` -> output `[n,m]`. f64.
    fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = a[i * n + j];
            }
        }
        out
    }

    /// Forward-substitution solve of `L X = rhs`, `L` unit-lower-triangular
    /// `[c,c]`, `rhs` `[c,w]`. f64.
    fn solve_unit_lower_tri(l: &[f64], rhs: &[f64], c: usize, w: usize) -> Vec<f64> {
        let mut x = rhs.to_vec();
        for i in 0..c {
            for j in 0..i {
                let l_ij = l[i * c + j];
                if l_ij == 0.0 {
                    continue;
                }
                for ww in 0..w {
                    x[i * w + ww] -= l_ij * x[j * w + ww];
                }
            }
        }
        x
    }

    /// CPU reference for one `(chunk, head)` slot — returns the plan
    /// kernel's five outputs `(u, y_local, q_eff, p_s, u_s)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn plan_reference_chunk(
        q_c: &[f64],
        k_c: &[f64],
        v_c: &[f64],
        g_c: &[f64],
        beta_c: &[f64],
        c: usize,
        dk: usize,
        dv: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // 1. Prefix gates G_t and masked Gamma[t,j] = G_t/G_j (j<=t).
        let mut big_g = vec![0.0_f64; c];
        big_g[0] = g_c[0];
        for t in 1..c {
            big_g[t] = big_g[t - 1] * g_c[t];
        }
        let mut gamma = vec![0.0_f64; c * c];
        for t in 0..c {
            for j in 0..=t {
                gamma[t * c + j] = big_g[t] / big_g[j];
            }
        }

        let k_t = transpose(k_c, c, dk);
        let kkt = matmul(k_c, &k_t, c, dk, c);
        let q_kt = matmul(q_c, &k_t, c, dk, c);

        // 2. Solve (I+L) p = K, L[t,j] = beta_j*KKT[t,j], j<t.
        let mut one_plus_l = vec![0.0_f64; c * c];
        for t in 0..c {
            one_plus_l[t * c + t] = 1.0;
            for j in 0..t {
                one_plus_l[t * c + j] = beta_c[j] * kkt[t * c + j];
            }
        }
        let p = solve_unit_lower_tri(&one_plus_l, k_c, c, dk);

        // 3. P_s = G_C * (I - (beta⊙p)^T K).
        let big_g_c = big_g[c - 1];
        let mut bp = vec![0.0_f64; c * dk];
        for i in 0..c {
            for d in 0..dk {
                bp[i * dk + d] = beta_c[i] * p[i * dk + d];
            }
        }
        let bp_t = transpose(&bp, c, dk);
        let bp_t_k = matmul(&bp_t, k_c, dk, c, dk);
        let mut p_s = vec![0.0_f64; dk * dk];
        for i in 0..dk {
            for j in 0..dk {
                let delta = if i == j { 1.0 } else { 0.0 };
                p_s[i * dk + j] = big_g_c * (delta - bp_t_k[i * dk + j]);
            }
        }

        // 4. q_eff[t,d] = G_t*(q_t[d] - sum_{i<=t} beta_i*QKT[t,i]*p[i,d]).
        let mut weight = vec![0.0_f64; c * c];
        for t in 0..c {
            for i in 0..=t {
                weight[t * c + i] = beta_c[i] * q_kt[t * c + i];
            }
        }
        let corr = matmul(&weight, &p, c, c, dk);
        let mut q_eff = vec![0.0_f64; c * dk];
        for t in 0..c {
            for d in 0..dk {
                q_eff[t * dk + d] = big_g[t] * (q_c[t * dk + d] - corr[t * dk + d]);
            }
        }

        // 5. Solve (I+A) u = beta⊙V, A[t,j] = beta_t*Gamma[t,j]*KKT[t,j].
        let mut one_plus_a = vec![0.0_f64; c * c];
        for t in 0..c {
            one_plus_a[t * c + t] = 1.0;
            for j in 0..t {
                one_plus_a[t * c + j] = beta_c[t] * gamma[t * c + j] * kkt[t * c + j];
            }
        }
        let mut bv = vec![0.0_f64; c * dv];
        for t in 0..c {
            for d in 0..dv {
                bv[t * dv + d] = beta_c[t] * v_c[t * dv + d];
            }
        }
        let u = solve_unit_lower_tri(&one_plus_a, &bv, c, dv);

        // 6. y_local = (Gamma_masked ⊙ QKT) @ u.
        let mut wt = vec![0.0_f64; c * c];
        for t in 0..c {
            for j in 0..=t {
                wt[t * c + j] = gamma[t * c + j] * q_kt[t * c + j];
            }
        }
        let y_local = matmul(&wt, &u, c, c, dv);

        // 7. U_s = sum_j (G_C/G_j)*u_j⊗k_j.
        let mut rw_u = vec![0.0_f64; c * dv];
        for j in 0..c {
            let scale = big_g_c / big_g[j];
            for d in 0..dv {
                rw_u[j * dv + d] = scale * u[j * dv + d];
            }
        }
        let rw_u_t = transpose(&rw_u, c, dv);
        let u_s = matmul(&rw_u_t, k_c, dv, c, dk);

        (u, y_local, q_eff, p_s, u_s)
    }

    /// Builds a `(t, hk, hv, dk, dv, c)` fixture, computes the expected
    /// `(u, y_local, q_eff, p_s, u_s)` per `(chunk, head)` slot via
    /// `plan_reference_chunk`, and wires the GPU dispatch.
    #[allow(clippy::too_many_arguments)]
    fn setup(
        t: usize,
        hk: usize,
        hv: usize,
        dk: usize,
        dv: usize,
        c: usize,
        dt: DType,
    ) -> TestSetup {
        setup_with_gates(t, hk, hv, dk, dv, c, dt, 0.8, 0.15)
    }

    /// `setup` with the per-token gate range exposed: `g[i] = g_base +
    /// g_amp·sin(...)`. The default (0.8, 0.15) keeps prefix products in
    /// comfortable f32 range; the harsh cell below uses (0.05, 0.045) —
    /// g ∈ [0.005, 0.095] — so a C=64 prefix product underflows f32
    /// (≈1e⁻⁶⁴…1e⁻¹⁴⁷), pinning the kernel's log-space Γ accumulation
    /// against the f64 oracle (which survives down to ~1e⁻³⁰⁸).
    #[allow(clippy::too_many_arguments)]
    fn setup_with_gates(
        t: usize,
        hk: usize,
        hv: usize,
        dk: usize,
        dv: usize,
        c: usize,
        dt: DType,
        g_base: f32,
        g_amp: f32,
    ) -> TestSetup {
        assert!(t.is_multiple_of(c), "t must be a multiple of c");
        let n_total = hv; // B=1
        let nc = t / c;
        let kscale = (2.0_f32 / dk as f32).sqrt();
        let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
        let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
        let v: Vec<f32> = (0..t * n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
        let g: Vec<f32> =
            (0..t * n_total).map(|i| g_base + g_amp * ((i as f32) * 0.013).sin()).collect();
        let beta: Vec<f32> =
            (0..t * n_total).map(|i| 0.4 + 0.3 * ((i as f32) * 0.017).cos()).collect();

        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (qr, kr, vr, gr, br) = (r(&q), r(&k), r(&v), r(&g), r(&beta));

        let hv_per_hk = hv / hk;
        let mut u_exp = vec![0.0_f32; n_total * nc * c * dv];
        let mut yl_exp = vec![0.0_f32; n_total * nc * c * dv];
        let mut qe_exp = vec![0.0_f32; n_total * nc * c * dk];
        let mut ps_exp = vec![0.0_f32; n_total * nc * dk * dk];
        let mut us_exp = vec![0.0_f32; n_total * nc * dv * dk];

        for hv_idx in 0..hv {
            let hk_idx = hv_idx / hv_per_hk;
            for chunk in 0..nc {
                let t0 = chunk * c;
                let mut q_c = vec![0.0_f64; c * dk];
                let mut k_c = vec![0.0_f64; c * dk];
                let mut v_c = vec![0.0_f64; c * dv];
                let mut g_c = vec![0.0_f64; c];
                let mut beta_c = vec![0.0_f64; c];
                for i in 0..c {
                    let tt = t0 + i;
                    for d in 0..dk {
                        q_c[i * dk + d] = qr[(tt * hk + hk_idx) * dk + d] as f64;
                        k_c[i * dk + d] = kr[(tt * hk + hk_idx) * dk + d] as f64;
                    }
                    for d in 0..dv {
                        v_c[i * dv + d] = vr[(tt * n_total + hv_idx) * dv + d] as f64;
                    }
                    g_c[i] = gr[tt * n_total + hv_idx] as f64;
                    beta_c[i] = br[tt * n_total + hv_idx] as f64;
                }
                let (u, yl, qe, ps, us) =
                    plan_reference_chunk(&q_c, &k_c, &v_c, &g_c, &beta_c, c, dk, dv);
                let n = hv_idx; // B=1
                let base_u = (n * nc + chunk) * c * dv;
                let base_qe = (n * nc + chunk) * c * dk;
                let base_ps = (n * nc + chunk) * dk * dk;
                let base_us = (n * nc + chunk) * dv * dk;
                for idx in 0..c * dv {
                    u_exp[base_u + idx] = u[idx] as f32;
                    yl_exp[base_u + idx] = yl[idx] as f32;
                }
                for idx in 0..c * dk {
                    qe_exp[base_qe + idx] = qe[idx] as f32;
                }
                for idx in 0..dk * dk {
                    ps_exp[base_ps + idx] = ps[idx] as f32;
                }
                for idx in 0..dv * dk {
                    us_exp[base_us + idx] = us[idx] as f32;
                }
            }
        }

        TestSetup::new(ffai_gdn_wy_plan::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec("g", pack_f32(&g, dt), dt))
            .input(TestBuffer::from_vec("beta", pack_f32(&beta, dt), dt))
            .input(TestBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .input(TestBuffer::zeros("u", u_exp.len(), dt))
            .input(TestBuffer::zeros("y_local", yl_exp.len(), dt))
            .input(TestBuffer::zeros("q_eff", qe_exp.len(), dt))
            .input(TestBuffer::zeros("p_s", ps_exp.len(), dt))
            .input(TestBuffer::zeros("u_s", us_exp.len(), dt))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("c", c as u32)
            .expect(TestBuffer::from_vec("u", pack_f32(&u_exp, dt), dt))
            .expect(TestBuffer::from_vec("y_local", pack_f32(&yl_exp, dt), dt))
            .expect(TestBuffer::from_vec("q_eff", pack_f32(&qe_exp, dt), dt))
            .expect(TestBuffer::from_vec("p_s", pack_f32(&ps_exp, dt), dt))
            .expect(TestBuffer::from_vec("u_s", pack_f32(&us_exp, dt), dt))
            .grid_3d(nc as u32, n_total as u32, 1, [512, 1, 1])
    }

    // C=64 at the smaller Dk=Dv=32 shape called out by the task spec — one
    // chunk, Hk=Hv=1 (no GQA).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [3e-3, 5e-3, 1e-2])]
    fn test_ffai_gdn_wy_plan_c64_dk32(dt: DType) -> TestSetup { setup(64, 1, 1, 32, 32, 64, dt) }

    // Multi-chunk (T=32, C=16 -> 2 chunks) with GQA (Hv=4, Hk=2): exercises
    // the hk_idx sharing and the P_s/U_s/q_eff/y_local cross-chunk shape
    // addressing (nc>1). Dv=16 (was 8 pre-MMA): the MMA path tiles Dv in
    // 16-row units, so `dv % 16 == 0` is now a documented dispatch
    // invariant — production Dv=128 always satisfied it; the old Dv=8 was
    // purely a fixture choice, not a shape any model uses.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [3e-3, 5e-3, 1e-2])]
    fn test_ffai_gdn_wy_plan_multi_chunk_gqa(dt: DType) -> TestSetup {
        setup(32, 2, 4, 16, 16, 16, dt)
    }

    // Harsh production-magnitude gates (g ∈ [0.005, 0.095] — the
    // Qwen3.6-class `exp(a_log)`∈[1,16] regime): a C=64 prefix product
    // underflows f32 entirely, so this cell fails with NaNs if the kernel
    // ever regresses from log-space Γ accumulation back to direct
    // prefix-product ratios.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [3e-3, 5e-3, 1e-2])]
    fn test_ffai_gdn_wy_plan_harsh_gates(dt: DType) -> TestSetup {
        setup_with_gates(64, 1, 1, 32, 32, 64, dt, 0.05, 0.045)
    }
}

pub mod kernel_benches {
    use ffai_kernels::{bench, test::*};

    use super::ffai_gdn_wy_plan;

    // Production Qwen3.6-class shape: B=1, T=2048, Hv=16, Hk=8, Dv=Dk=128,
    // C=64 (NC=32 chunks). Grid `[32, 16, 1]`, TG `[512,1,1]`.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gdn_wy_plan_t2048(dt: DType) -> BenchSetup {
        let (b, t, hk, hv, dk, dv, c) =
            (1usize, 2048usize, 8usize, 16usize, 128usize, 128usize, 64usize);
        let n_total = b * hv;
        let nc = t / c;
        BenchSetup::new(ffai_gdn_wy_plan::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", t * hk * dk, dt))
            .buffer(BenchBuffer::random("k", t * hk * dk, dt))
            .buffer(BenchBuffer::random("v", t * n_total * dv, dt))
            .buffer(BenchBuffer::random("g", t * n_total, dt))
            .buffer(BenchBuffer::random("beta", t * n_total, dt))
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("u", n_total * nc * c * dv, dt).output())
            .buffer(BenchBuffer::zeros("y_local", n_total * nc * c * dv, dt).output())
            .buffer(BenchBuffer::zeros("q_eff", n_total * nc * c * dk, dt).output())
            .buffer(BenchBuffer::zeros("p_s", n_total * nc * dk * dk, dt).output())
            .buffer(BenchBuffer::zeros("u_s", n_total * nc * dv * dk, dt).output())
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("c", c as u32)
            .grid_3d(nc as u32, n_total as u32, 1, [512, 1, 1])
            .bytes_moved(
                ((t * hk * dk * 2 + t * n_total * dv + t * n_total * 2) * dt.size_bytes()
                    + (n_total * nc * c * dv * 2
                        + n_total * nc * c * dk
                        + n_total * nc * dk * dk
                        + n_total * nc * dv * dk)
                        * dt.size_bytes()) as u64,
            )
    }
}

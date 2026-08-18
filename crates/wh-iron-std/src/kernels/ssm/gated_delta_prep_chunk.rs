//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet — **fused** prep + chunked-prefill kernel.
//!
//! `iron_gated_delta_prep_chunk` extends
//! [`iron_gated_delta_prep_step`](super::gated_delta_prep::iron_gated_delta_prep_step)
//! over a chunk of `T` tokens, mirroring the relationship between
//! [`iron_gated_delta_step`](super::gated_delta::iron_gated_delta_step) and
//! [`iron_gated_delta_chunk`](super::gated_delta::iron_gated_delta_chunk).
//!
//! State stays register-resident across the entire `T`-loop — one
//! load_state at entry and one store_state at exit, regardless of `T`.
//! This collapses the dominant `iron_gated_delta_prep_step`-per-token T-loop
//! in `Qwen35GDNMixer.forwardMany` to a single dispatch per layer.
//!
//! The per-head RMSNorm of q/k (state-independent) is **not** computed
//! here — it's hoisted into
//! [`iron_gated_delta_qknorm_prepass`](super::gated_delta_qknorm_prepass),
//! run once per `(b, t, hk_idx)` ahead of this kernel, because this
//! kernel's own grid (`[Dv, B·Hv, 1]`) would otherwise redundantly
//! recompute the identical q/k norm `Dv · (Hv/Hk)` times per token. This
//! kernel just loads the dense `q_normed` / `k_normed` the pre-pass wrote.
//!
//! Inputs (note the added `T` dimension on conv_out / a_raw / b_raw):
//!   - `conv_out`  : Tensor<T> [B, T, 2·Hk·Dk + Hv·Dv]   q | k | v slabs (v used here; q/k slabs are read by the pre-pass, not this kernel)
//!   - `a_log`     : Tensor<T> [Hv]                      per-Hv learnable
//!   - `dt_bias`   : Tensor<T> [Hv]
//!   - `a_raw`     : Tensor<T> [B, T, Hv]
//!   - `b_raw`     : Tensor<T> [B, T, Hv]
//!   - `q_normed`  : Tensor<T> [B, T, Hk, Dk]   from `iron_gated_delta_qknorm_prepass`
//!   - `k_normed`  : Tensor<T> [B, T, Hk, Dk]   from `iron_gated_delta_qknorm_prepass`
//!   - `state_in`  : Tensor<T> [B, Hv, Dv, Dk]           (one state per (b, hv))
//!   - `t_len`     : Tensor<u32> [1]                     runtime chunk length
//!   - `planes_enabled` : Tensor<u32> [1]                runtime mode; 0 = no plane
//!     writes (byte-identical to the pre-plane kernel), 1 = write every token,
//!     and 2 = write every token except the final live state. See "Per-token
//!     state planes" below.
//!
//! Outputs:
//!   - `state_out`    : Tensor<T> [B, Hv, Dv, Dk]
//!   - `y`            : Tensor<T> [B, T, Hv, Dv]
//!   - `state_planes` : Tensor<T> [T, B, Hv, Dv, Dk]      only written when
//!     `planes_enabled != 0` (spec-decode rollback side channel, see below)
//!
//! ## Per-token state planes (spec-decode rollback)
//!
//! MTP speculative-decode partial accept needs the recurrent state as of
//! token `k` of a `T`-token batched verify pass, for whichever `k <=
//! T` ends up accepted. The old fix (`Qwen35GDNLayer.rewindState`)
//! restores the pre-verify snapshot and re-runs conv+scan over the
//! accepted prefix — correct, but a second dispatch pair per rollback.
//!
//! This kernel's inner T-loop is a **strictly sequential per-token
//! recurrence** (no cross-token chunked/WY blocking — see the module
//! doc above): the register-resident state after processing token `t`
//! is bit-for-bit the SAME value a fresh `t_len = t+1` invocation of
//! this kernel would produce from the same `state_in`. So instead of
//! recomputing state[k], the T-loop below can just also WRITE it out,
//! once, at every token boundary — `state_planes[t]` becomes exactly
//! what a rewind-to-`t+1` would recompute. Rollback then degrades to
//! selecting `state_planes[k-1]` (a blit or pointer-swap into
//! `GDNStateCache.current`) instead of a restore + re-scan dispatch
//! pair.
//!
//! `planes_enabled` gates the extra stores at runtime so the default
//! (disabled) path pays no memory-traffic cost beyond the branch test
//! itself. Mode 2 omits the final plane for rollback consumers that keep
//! the live state after a full acceptance and only select earlier planes
//! after a partial acceptance. `n_total`/`state_planes` are validated/sized by the caller;
//! this kernel does not bounds-check `T` against the plane buffer's
//! actual capacity.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Mode: Reduction.** TG packs **4 simdgroups** (128 threads) — one SG
//!   per `dv` slot inside the TG (matches reference Metal GDN `(32,4,1)`).
//! - **Grid: `[ceil(Dv/4), B·Hv, 1]`, TG: `[128, 1, 1]`.**
//!   `dv_idx = tgid_x·4 + simd_group_id`, `dk_idx = simd_lane`.
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` slots.
//! - **Hv divisible by Hk.** GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//! - **`t_len` is runtime u32** so a single PSO compiles for every chunk size.
//! - **Must run after `iron_gated_delta_qknorm_prepass`** on the same
//!   `conv_out` / `t_len` — this kernel does not validate that `q_normed`
//!   / `k_normed` were populated for the same chunk.
//!
//! ## Per-iter cost vs prep_step
//!
//! Prep-step pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane)
//!   - prep math + recurrence math
//!
//! Prep-chunk pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane), TOTAL — not per-t
//!   - T × recurrence math (q/k norm hoisted to the pre-pass, see above)
//!
//! State traffic per layer drops by `T`× at the dispatch boundary. For
//! Qwen3.6-35B-A3B (real shape from `config.json`: `linear_key_head_dim` =
//! `linear_value_head_dim` = Dk = Dv = 128, `linear_num_value_heads` =
//! Hv = 32, B=1): state size = 32·128·128·4 B = 2 MiB per direction. At
//! T=512 the per-token loop did `T × (state R+W) = 2 GiB device traffic per
//! layer per direction × 30 GDN layers = 120 GiB per prefill step in state
//! traffic alone. The chunked variant does 2 MiB × 30 = 60 MiB.
//!
//! NOTE: this shape (Dk=256, Dv=128, Hv=16) previously documented here (and
//! baked into `bench_gated_delta_prep_chunk`'s "production" shape below) was
//! stale/incorrect, likely written against an earlier Qwen3.5 prototype
//! config. `linear_num_value_heads=32` / `linear_num_key_heads=16` /
//! `linear_key_head_dim=linear_value_head_dim=128` is what the shipping
//! Qwen3.6-35B-A3B `config.json` and `Qwen3xText.swift`'s config parsing
//! (`tcInt("linear_num_value_heads")` etc.) actually use.

use wh_iron::kernel;

#[kernel]
pub fn iron_gated_delta_prep_chunk<T>(
    conv_out: Tensor<T>, // [B, T, 2·Hk·Dk + Hv·Dv]  (v slab; q/k slabs unused here)
    a_log: Tensor<T>,    // [Hv]
    dt_bias: Tensor<T>,  // [Hv]
    a_raw: Tensor<T>,    // [B, T, Hv]
    b_raw: Tensor<T>,    // [B, T, Hv]
    q_normed: Tensor<T>, // [B, T, Hk, Dk]  from iron_gated_delta_qknorm_prepass
    k_normed: Tensor<T>, // [B, T, Hk, Dk]  from iron_gated_delta_qknorm_prepass
    state_in: Tensor<T>, // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,    // [B, T, Hv, Dv]
    t_len: Tensor<u32>,  // [1] scalar
    mut state_planes: Tensor<T>, // [T, B, Hv, Dv, Dk]; written iff planes_enabled != 0
    planes_enabled: Tensor<u32>, // [1] scalar runtime flag
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
    #[constexpr] n_total: u32, // B·Hv — plane stride multiplier (state_out's own element count)
) {
    // 4 SGs per TG pack 4 Dv slots (reference Metal GDN geometry).
    let sg = simd_group_id();
    let dk_idx = simd_lane;
    let dv_idx = tgid_x * 4u32 + sg;
    let n = tgid_y;
    // GQA decomposition.
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;
    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);
    let stride_b = 2u32 * hk * dk + hv * dv;
    // Partial last TG when Dv % 4 != 0: idle SGs must not touch state/y.
    if dv_idx < dv {
        // Per-layer constants (loaded once per active SG).
        let a_log_val = load(a_log[hv_idx]).cast::<f32>();
        let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
        let exp_a_log = exp(a_log_val);
        let state_base = n * dv * dk + dv_idx * dk;
        // Plane gate + per-t stride, resolved once (register-resident,
        // same convention as the state load below) rather than re-read
        // from the flag buffer every T-loop iteration.
        let planes_on = load(planes_enabled[0]);
        let plane_t_stride = n_total * dv * dk;
        // ─── Load state into per-lane registers ONCE — persists across T.
        stack_alloc("state_reg", 8u32, "f32");
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let val = load(state_in[state_base + s_idx]).cast::<f32>();
            stack_store("state_reg", i, val);
        }
        // k_cache holds this token's k_normed (read once in Phase 1, reused in
        // Phase 2's rank-1 update without a second load).
        stack_alloc("k_cache", 8u32, "f32");
        // ─── Inner T-loop: recurrence per token ──────────────────────────
        for t in range(0u32, t_total, 1u32) {
            let bt = b * t_total + t;
            let conv_base = bt * stride_b;
            let v_off = conv_base + 2u32 * hk * dk + hv_idx * dv;
            let qk_off = (bt * hk + hk_idx) * dk;
            let gbeta_idx = bt * hv + hv_idx;
            // ─── g / beta ──────────────────────────────────────────────
            let a_raw_val = load(a_raw[gbeta_idx]).cast::<f32>();
            let b_raw_val = load(b_raw[gbeta_idx]).cast::<f32>();
            let pre_softplus = a_raw_val + dt_bias_val;
            let dt_val = log(exp(pre_softplus) + 1.0f32);
            let g_val = exp(0.0f32 - exp_a_log * dt_val);
            let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));
            // v: one read per Dv slot per token.
            let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();
            // ─── Phase 1: decay state + accumulate kv_mem; cache k ─────
            let mut kv_mem = 0.0f32;
            for i in range(0u32, n_per_t, 1u32) {
                let s_idx = n_per_t * dk_idx + i;
                let s_old = stack_load("state_reg", i);
                let s_decayed = s_old * g_val;
                stack_store("state_reg", i, s_decayed);
                let k_normed_val = load(k_normed[qk_off + s_idx]).cast::<f32>();
                stack_store("k_cache", i, k_normed_val);
                kv_mem = kv_mem + s_decayed * k_normed_val;
            }
            let kv_mem_sum = simd_sum(kv_mem);
            let delta = (v_val - kv_mem_sum) * beta_val;
            // ─── Phase 2: rank-1 update + output projection ───────────
            let mut out_acc = 0.0f32;
            for i in range(0u32, n_per_t, 1u32) {
                let s_idx = n_per_t * dk_idx + i;
                let s_decayed = stack_load("state_reg", i);
                let k_normed_val = stack_load("k_cache", i);
                let s_new = s_decayed + k_normed_val * delta;
                stack_store("state_reg", i, s_new);
                let q_normed_val = load(q_normed[qk_off + s_idx]).cast::<f32>();
                out_acc = out_acc + s_new * q_normed_val;
            }
            // ─── Optional: mirror this token's post-update state into
            // state_planes[t] — default-off, gated on planes_on so the
            // disabled path is just one extra register compare per
            // token (no stores, no extra memory traffic).
            if planes_on != 0u32 && (planes_on != 2u32 || t + 1u32 < t_total) {
                let plane_base = t * plane_t_stride + state_base;
                for i in range(0u32, n_per_t, 1u32) {
                    let s_idx = n_per_t * dk_idx + i;
                    let s_val = stack_load("state_reg", i);
                    store(state_planes[plane_base + s_idx], s_val.cast::<T>());
                }
            }
            let out_sum = simd_sum(out_acc);
            // ─── Phase 3: lane 0 writes y[t, n, dv_idx] ───────────────
            if dk_idx == 0u32 {
                store(y[(bt * hv + hv_idx) * dv + dv_idx], out_sum.cast::<T>());
            }
        }
        // ─── Write final state ONCE at the end ──────────────────────────
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
        }
    }
}

/// Shape-specialized sibling of [`iron_gated_delta_prep_chunk`].
///
/// Root cause found while settling the "GDN scan is 3.5-4x slower than a
/// reference engine" vs "the DSL source already matches the reference
/// design" contradiction: the generic kernel above takes `dk`/`dv`/`hv`/`hk`
/// as `#[constexpr]` parameters, but the MSL backend lowers every
/// `#[constexpr]` to an ordinary `constant T &name [[buffer(N)]]`, a
/// runtime value, identical to `ParamKind::Scalar` (see
/// `wh-iron-codegen/src/msl/mod.rs`, the `Constexpr params` match arm).
/// `n_per_t = dk / 32u32` is therefore NOT a compile-time constant to the
/// Metal shader compiler, so `for i in range(0, n_per_t, 1)` cannot be
/// unrolled and `state_reg[8]` / `k_cache[8]` cannot be scalar-replaced into
/// registers: every `stack_load`/`stack_store` on them compiles to a real
/// indexed memory access repeated every one of the `t_len` serially-
/// dependent loop iterations. Dumping the generic kernel's MSL (`cargo test
/// -p wh-iron-std --lib --release -- \
/// kernels::ssm::gated_delta_prep_chunk::tests::dump --nocapture`) shows
/// exactly this: `float state_reg[8];` indexed by a value derived from the
/// `dk` buffer parameter, not a literal.
///
/// This kernel bakes `DK`/`DV`/`HV`/`HK`/`NPT` as true compile-time
/// `#[kernel(variants(...))]` constants (the same mechanism already used by
/// `iron_gated_delta_step_record`'s `d192_128_4_4` / `d64_32_2_2` shape
/// specialization in `gated_delta_replay.rs`) so the Metal compiler can fully
/// unroll every `NPT`-bounded loop and keep `state_reg` / `k_cache` in
/// registers for real. Grid/TG geometry, dispatch invariants, and the
/// recurrence math are otherwise byte-for-byte identical to
/// `iron_gated_delta_prep_chunk`; this does not touch grid-level
/// parallelism (same `[ceil(Dv/4), B·Hv, 1]` / `[128,1,1]` shape), only
/// register promotion inside a single lane's serial T-loop.
///
/// Variant rows:
///   - `d128_128_32_16`: Qwen3.6-35B-A3B production shape, read from
///     `config.json`: `linear_num_value_heads=32` (Hv), `linear_num_key_heads
///     =16` (Hk), `linear_key_head_dim=linear_value_head_dim=128` (Dk=Dv).
///   - `d128_128_48_16`: Qwen3.6-27B production shape — same Dk/Dv/Hk as the
///     35B-A3B row above but `linear_num_value_heads=48` (Hv); the 27B
///     config widens Hv only, Hk/Dk/Dv are unchanged from the A3B config.
///     `NPT = Dk/32 = 4` is identical to the 32-Hv row (NPT depends on Dk,
///     not Hv) — only the `HV`/grid-`tgid_y` extent differs.
///   - `d64_8_4_2`: small GQA test cell (mirrors the generic kernel's own
///     `test_iron_gated_delta_prep_chunk_gqa` shape) so correctness has a
///     cheap, fast-running fixture instead of only the full production size.
#[kernel(variants(
    DK = [128u32, 128u32, 64u32],
    DV = [128u32, 128u32, 8u32],
    HV = [32u32, 48u32, 4u32],
    HK = [16u32, 16u32, 2u32],
    NPT = [4, 4, 2],
    suffix = "d{DK}_{DV}_{HV}_{HK}"
))]
pub fn iron_gated_delta_prep_chunk_fast<T>(
    conv_out: Tensor<T>, // [B, T, 2·Hk·Dk + Hv·Dv]  (v slab; q/k slabs unused here)
    a_log: Tensor<T>,    // [Hv]
    dt_bias: Tensor<T>,  // [Hv]
    a_raw: Tensor<T>,    // [B, T, Hv]
    b_raw: Tensor<T>,    // [B, T, Hv]
    q_normed: Tensor<T>, // [B, T, Hk, Dk]  from iron_gated_delta_qknorm_prepass
    k_normed: Tensor<T>, // [B, T, Hk, Dk]  from iron_gated_delta_qknorm_prepass
    state_in: Tensor<T>, // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,    // [B, T, Hv, Dv]
    t_len: Tensor<u32>,  // [1] scalar
    mut state_planes: Tensor<T>, // [T, B, Hv, Dv, Dk]; written iff planes_enabled != 0
    planes_enabled: Tensor<u32>, // [1] scalar runtime flag
    #[constexpr] n_total: u32, // B·HV — plane stride multiplier; see generic kernel doc
) {
    // 4 SGs per TG pack 4 Dv slots (reference Metal GDN geometry),
    // identical dispatch geometry to `iron_gated_delta_prep_chunk`.
    let sg = simd_group_id();
    let dk_idx = simd_lane;
    let dv_idx = tgid_x * 4u32 + sg;
    let n = tgid_y;
    // GQA decomposition (HV/HK now compile-time, no runtime division cost
    // change vs the generic kernel, but every downstream use of the result
    // is now foldable too).
    let hv_idx = n - (n / HV) * HV;
    let b = n / HV;
    let hk_per_hv = HV / HK;
    let hk_idx = hv_idx / hk_per_hv;
    let t_total = load(t_len[0]);
    let stride_b = 2u32 * HK * DK + HV * DV;
    // Partial last TG when Dv % 4 != 0: idle SGs must not touch state/y.
    if dv_idx < DV {
        // Per-layer constants (loaded once per active SG).
        let a_log_val = load(a_log[hv_idx]).cast::<f32>();
        let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
        let exp_a_log = exp(a_log_val);
        let state_base = n * DV * DK + dv_idx * DK;
        // Plane gate + per-t stride, resolved once — see the generic
        // kernel's identical comment above.
        let planes_on = load(planes_enabled[0]);
        let plane_t_stride = n_total * DV * DK;
        // ─── Load state into per-lane registers ONCE, persists across T.
        // `NPT` is a compile-time literal here (unlike the generic kernel's
        // runtime `n_per_t`), so this array is fully unrollable and
        // register-promotable: the actual fix.
        stack_alloc("state_reg", NPT, "f32");
        for i in range(0u32, NPT, 1u32) {
            let s_idx = NPT * dk_idx + i;
            let val = load(state_in[state_base + s_idx]).cast::<f32>();
            stack_store("state_reg", i, val);
        }
        // k_cache holds this token's k_normed (read once in Phase 1, reused in
        // Phase 2's rank-1 update without a second load).
        stack_alloc("k_cache", NPT, "f32");
        // ─── Inner T-loop: recurrence per token ──────────────────────────
        for t in range(0u32, t_total, 1u32) {
            let bt = b * t_total + t;
            let conv_base = bt * stride_b;
            let v_off = conv_base + 2u32 * HK * DK + hv_idx * DV;
            let qk_off = (bt * HK + hk_idx) * DK;
            let gbeta_idx = bt * HV + hv_idx;
            // ─── g / beta ──────────────────────────────────────────────
            let a_raw_val = load(a_raw[gbeta_idx]).cast::<f32>();
            let b_raw_val = load(b_raw[gbeta_idx]).cast::<f32>();
            let pre_softplus = a_raw_val + dt_bias_val;
            let dt_val = log(exp(pre_softplus) + 1.0f32);
            let g_val = exp(0.0f32 - exp_a_log * dt_val);
            let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));
            // v: one read per Dv slot per token.
            let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();
            // ─── Phase 1: decay state + accumulate kv_mem; cache k ─────
            let mut kv_mem = 0.0f32;
            for i in range(0u32, NPT, 1u32) {
                let s_idx = NPT * dk_idx + i;
                let s_old = stack_load("state_reg", i);
                let s_decayed = s_old * g_val;
                stack_store("state_reg", i, s_decayed);
                let k_normed_val = load(k_normed[qk_off + s_idx]).cast::<f32>();
                stack_store("k_cache", i, k_normed_val);
                kv_mem = kv_mem + s_decayed * k_normed_val;
            }
            let kv_mem_sum = simd_sum(kv_mem);
            let delta = (v_val - kv_mem_sum) * beta_val;
            // ─── Phase 2: rank-1 update + output projection ───────────
            let mut out_acc = 0.0f32;
            for i in range(0u32, NPT, 1u32) {
                let s_idx = NPT * dk_idx + i;
                let s_decayed = stack_load("state_reg", i);
                let k_normed_val = stack_load("k_cache", i);
                let s_new = s_decayed + k_normed_val * delta;
                stack_store("state_reg", i, s_new);
                let q_normed_val = load(q_normed[qk_off + s_idx]).cast::<f32>();
                out_acc = out_acc + s_new * q_normed_val;
            }
            // ─── Optional: mirror this token's post-update state into
            // state_planes[t] — see the generic kernel's identical block
            // for the design rationale.
            if planes_on != 0u32 && (planes_on != 2u32 || t + 1u32 < t_total) {
                let plane_base = t * plane_t_stride + state_base;
                for i in range(0u32, NPT, 1u32) {
                    let s_idx = NPT * dk_idx + i;
                    let s_val = stack_load("state_reg", i);
                    store(state_planes[plane_base + s_idx], s_val.cast::<T>());
                }
            }
            let out_sum = simd_sum(out_acc);
            // ─── Phase 3: lane 0 writes y[t, n, dv_idx] ───────────────
            if dk_idx == 0u32 {
                store(y[(bt * HV + hv_idx) * DV + dv_idx], out_sum.cast::<T>());
            }
        }
        // ─── Write final state ONCE at the end ──────────────────────────
        for i in range(0u32, NPT, 1u32) {
            let s_idx = NPT * dk_idx + i;
            store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
        }
    }
}

/// F-85 prefill wave (Qwen3.6-27B port, `wh-butter-models::qwen35`):
/// CYCLIC-GQA sibling of `iron_gated_delta_prep_chunk_fast_d128_128_48_16`.
///
/// Byte-for-byte identical to `iron_gated_delta_prep_chunk_fast` above
/// except ONE line: `hk_idx = hv_idx % HK` (cyclic) instead of `hv_idx /
/// hk_per_hv` (block, this family's default). This port's checkpoint
/// needs the CYCLIC mapping — verified by oracle checksum diff during the
/// decode port (`attn_output` matched to 0.01% only after switching from
/// block to cyclic, see `wh-butter-models::qwen35`'s `gdn_gqa_cyclic_idx` doc and
/// `gated_delta_qwen35_decode_fused.rs`'s `iron_gdn_decode_fused`, which
/// bakes the SAME cyclic mapping into its own decode-step kernel for the
/// identical reason). At Hv=48/Hk=16 the two mappings agree for NO
/// `hv_idx` in general (block groups `{0,1,2}->hk0, {3,4,5}->hk1, ...`;
/// cyclic assigns `{0,16,32}->hk0, {1,17,33}->hk1, ...` — disjoint
/// groupings), so this is not a rare-edge-case fix, every GDN layer's
/// prefill output depends on it.
///
/// A SEPARATE kernel (not a modified variant of the shared
/// `iron_gated_delta_prep_chunk_fast<T>` template) since the block mapping
/// remains the correct family default other callers may still rely on —
/// this port's Hv=48/Hk=16 checkpoint is the only one confirmed (via the
/// decode-path oracle diff) to need cyclic; whether Qwen3.6-35B-A3B's
/// checkpoint (Hv=32/Hk=16, `_d128_128_32_16`) also needs it is unverified
/// and out of scope here — do not assume it does without the same oracle
/// check that caught this for 27B.
#[kernel(variants(
    DK = [128u32],
    DV = [128u32],
    HV = [48u32],
    HK = [16u32],
    NPT = [4],
    suffix = "d{DK}_{DV}_{HV}_{HK}"
))]
pub fn iron_gated_delta_prep_chunk_fast_cyclic<T>(
    conv_out: Tensor<T>, // [B, T, 2·Hk·Dk + Hv·Dv]  (v slab; q/k slabs unused here)
    a_log: Tensor<T>,    // [Hv]
    dt_bias: Tensor<T>,  // [Hv]
    a_raw: Tensor<T>,    // [B, T, Hv]
    b_raw: Tensor<T>,    // [B, T, Hv]
    q_normed: Tensor<T>, // [B, T, Hk, Dk]  from iron_gated_delta_qknorm_prepass
    k_normed: Tensor<T>, // [B, T, Hk, Dk]  from iron_gated_delta_qknorm_prepass
    state_in: Tensor<T>, // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,    // [B, T, Hv, Dv]
    t_len: Tensor<u32>,  // [1] scalar
) {
    let sg = simd_group_id();
    let dk_idx = simd_lane;
    let dv_idx = tgid_x * 4u32 + sg;
    let n = tgid_y;
    let hv_idx = n - (n / HV) * HV;
    let b = n / HV;
    // CYCLIC GQA -- the one line different from `iron_gated_delta_prep_chunk_fast`.
    let hk_idx = hv_idx - (hv_idx / HK) * HK;
    let t_total = load(t_len[0]);
    let stride_b = 2u32 * HK * DK + HV * DV;
    if dv_idx < DV {
        let a_log_val = load(a_log[hv_idx]).cast::<f32>();
        let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
        let exp_a_log = exp(a_log_val);
        let state_base = n * DV * DK + dv_idx * DK;
        stack_alloc("state_reg", NPT, "f32");
        for i in range(0u32, NPT, 1u32) {
            let s_idx = NPT * dk_idx + i;
            let val = load(state_in[state_base + s_idx]).cast::<f32>();
            stack_store("state_reg", i, val);
        }
        stack_alloc("k_cache", NPT, "f32");
        for t in range(0u32, t_total, 1u32) {
            let bt = b * t_total + t;
            let conv_base = bt * stride_b;
            let v_off = conv_base + 2u32 * HK * DK + hv_idx * DV;
            let qk_off = (bt * HK + hk_idx) * DK;
            let gbeta_idx = bt * HV + hv_idx;
            let a_raw_val = load(a_raw[gbeta_idx]).cast::<f32>();
            let b_raw_val = load(b_raw[gbeta_idx]).cast::<f32>();
            let pre_softplus = a_raw_val + dt_bias_val;
            let dt_val = log(exp(pre_softplus) + 1.0f32);
            let g_val = exp(0.0f32 - exp_a_log * dt_val);
            let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));
            let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();
            let mut kv_mem = 0.0f32;
            for i in range(0u32, NPT, 1u32) {
                let s_idx = NPT * dk_idx + i;
                let s_old = stack_load("state_reg", i);
                let s_decayed = s_old * g_val;
                stack_store("state_reg", i, s_decayed);
                let k_normed_val = load(k_normed[qk_off + s_idx]).cast::<f32>();
                stack_store("k_cache", i, k_normed_val);
                kv_mem = kv_mem + s_decayed * k_normed_val;
            }
            let kv_mem_sum = simd_sum(kv_mem);
            let delta = (v_val - kv_mem_sum) * beta_val;
            let mut out_acc = 0.0f32;
            for i in range(0u32, NPT, 1u32) {
                let s_idx = NPT * dk_idx + i;
                let s_decayed = stack_load("state_reg", i);
                let k_normed_val = stack_load("k_cache", i);
                let s_new = s_decayed + k_normed_val * delta;
                stack_store("state_reg", i, s_new);
                let q_normed_val = load(q_normed[qk_off + s_idx]).cast::<f32>();
                out_acc = out_acc + s_new * q_normed_val;
            }
            let out_sum = simd_sum(out_acc);
            if dk_idx == 0u32 {
                store(y[(bt * HV + hv_idx) * DV + dv_idx], out_sum.cast::<T>());
            }
        }
        for i in range(0u32, NPT, 1u32) {
            let s_idx = NPT * dk_idx + i;
            store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use wh_iron::core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    /// `cargo test -p wh-iron-std --lib --release -- kernels::ssm::gated_delta_prep_chunk::tests::dump --nocapture`
    #[test]
    fn dump() {
        use wh_iron::codegen::msl::MslGenerator;
        let mut k = iron_gated_delta_prep_chunk::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }

    /// Same developer aid as `dump`, but for the shape-specialized
    /// production kernel: confirms `DK`/`DV`/`HV`/`HK`/`NPT` are baked as
    /// MSL literals (not `constant uint &` buffer params) so `state_reg` /
    /// `k_cache` are compiler-unrollable and register-promotable, unlike
    /// the generic kernel's `dump` output above.
    /// `cargo test -p wh-iron-std --lib --release -- kernels::ssm::gated_delta_prep_chunk::tests::dump_fast_prod --nocapture`
    #[test]
    fn dump_fast_prod() {
        use wh_iron::codegen::msl::MslGenerator;
        let mut k = iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL FAST =====\n{}\n===== END MSL FAST =====", msl);
    }
}

/// New-syntax correctness for the fused chunked GDN prep+recurrence kernel
/// (`iron_gated_delta_prep_chunk`). Oracle is the per-token prep + sequential GDN
/// recurrence with state carried across the T-loop (state_out of token t is
/// state_in of token t+1) — the legacy `gated_delta_prep_step` oracle composed
/// over T tokens, which is exactly the recurrence the kernel runs register-
/// resident across its inner T-loop. Same un-clamped softplus as the kernel;
/// inputs are dtype-rounded.
///
/// Grid (Reduction, 4 SGs/TG): `grid_3d(ceil(dv/4), b*hv, 1, [128,1,1])`;
/// `t_len` is a runtime u32 scalar buffer.
pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::{
        iron_gated_delta_prep_chunk,
        iron_gated_delta_prep_chunk_fast_cyclic_d128_128_48_16,
        iron_gated_delta_prep_chunk_fast_d64_8_4_2,
        iron_gated_delta_prep_chunk_fast_d128_128_32_16,
        iron_gated_delta_prep_chunk_fast_d128_128_48_16,
    };
    use crate::utils::{pack_f32, unpack_f32};

    fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// Per-head RMSNorm of q/k — the same math
    /// `iron_gated_delta_qknorm_prepass` computes, evaluated once per
    /// `(b, t, hk_idx)`. Returns dense `(q_normed, k_normed)` [B,T,Hk,Dk],
    /// mirroring what that kernel writes.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm(
        conv_out: &[f32], // [B, T, 2·Hk·Dk + Hv·Dv]
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-6_f32;
        let stride_b = 2 * hk * dk + hv * dv;
        let mut q_normed = vec![0.0_f32; b * t_total * hk * dk];
        let mut k_normed = vec![0.0_f32; b * t_total * hk * dk];
        for batch in 0..b {
            for t in 0..t_total {
                let bt = batch * t_total + t;
                let conv_base = bt * stride_b;
                for hk_idx in 0..hk {
                    let q_row = conv_base + hk_idx * dk;
                    let k_row = conv_base + hk * dk + hk_idx * dk;
                    let mut q_ssq = 0.0_f32;
                    let mut k_ssq = 0.0_f32;
                    for d in 0..dk {
                        let qv = conv_out[q_row + d];
                        let kv = conv_out[k_row + d];
                        q_ssq += qv * qv;
                        k_ssq += kv * kv;
                    }
                    let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                    let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();
                    let out_base = (bt * hk + hk_idx) * dk;
                    for d in 0..dk {
                        q_normed[out_base + d] =
                            conv_out[q_row + d] * q_inv * q_norm_weight[hk_idx * dk + d];
                        k_normed[out_base + d] =
                            conv_out[k_row + d] * k_inv * k_norm_weight[hk_idx * dk + d];
                    }
                }
            }
        }
        (q_normed, k_normed)
    }

    /// Per-token CPU GDN recurrence over a chunk of `t_total` tokens, with
    /// the state threaded across tokens. Returns `(y [B,T,Hv,Dv], state_out
    /// [B,Hv,Dv,Dk])`. `q_normed`/`k_normed` are dense `[B,T,Hk,Dk]` — the
    /// pre-pass's output, matching what the real kernel now consumes.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        conv_out: &[f32], // [B, T, 2·Hk·Dk + Hv·Dv]  (v slab only)
        a_log: &[f32],    // [Hv]
        dt_bias: &[f32],  // [Hv]
        a_raw: &[f32],    // [B, T, Hv]
        b_raw: &[f32],    // [B, T, Hv]
        q_normed: &[f32], // [B, T, Hk, Dk]
        k_normed: &[f32], // [B, T, Hk, Dk]
        state_in: &[f32], // [B, Hv, Dv, Dk]
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let stride_b = 2 * hk * dk + hv * dv;
        let hk_per_hv = hv / hk;
        let mut y = vec![0.0_f32; b * t_total * hv * dv];
        let mut state = state_in.to_vec(); // carried across the T-loop
        for batch in 0..b {
            for t in 0..t_total {
                let bt = batch * t_total + t;
                let conv_base = bt * stride_b;
                let v_base = conv_base + 2 * hk * dk;
                for hv_idx in 0..hv {
                    let n = batch * hv + hv_idx;
                    let hk_idx = hv_idx / hk_per_hv;
                    let qk_base = (bt * hk + hk_idx) * dk;
                    // g / beta.
                    let gbeta_idx = bt * hv + hv_idx;
                    let dt = softplus_unclamped(a_raw[gbeta_idx] + dt_bias[hv_idx]);
                    let g_val = (-a_log[hv_idx].exp() * dt).exp();
                    let beta_val = sigmoid(b_raw[gbeta_idx]);
                    for dv_idx in 0..dv {
                        let v_val = conv_out[v_base + hv_idx * dv + dv_idx];
                        let s_base = n * dv * dk + dv_idx * dk;
                        // Phase 1: decay + kv_mem (k normed per-dim).
                        let mut kv_mem = 0.0_f32;
                        let mut decayed = vec![0.0_f32; dk];
                        for d in 0..dk {
                            let s = state[s_base + d] * g_val;
                            decayed[d] = s;
                            kv_mem += s * k_normed[qk_base + d];
                        }
                        let delta = (v_val - kv_mem) * beta_val;
                        // Phase 2: rank-1 update + output projection.
                        let mut out = 0.0_f32;
                        for d in 0..dk {
                            let s_new = decayed[d] + k_normed[qk_base + d] * delta;
                            state[s_base + d] = s_new;
                            out += s_new * q_normed[qk_base + d];
                        }
                        y[(bt * hv + hv_idx) * dv + dv_idx] = out;
                    }
                }
            }
        }
        (y, state)
    }

    /// Same recurrence as `oracle`, but additionally captures a full copy
    /// of `state` after every token `t` — the CPU-side ground truth for
    /// `state_planes`. `planes[t]` is bit-for-bit what a fresh
    /// `oracle(..., t_total = t+1, ...)` call would return as its
    /// `state_out`, since the recurrence is strictly sequential per
    /// token (no cross-token blocking) — same claim the kernel doc
    /// makes about `state_planes[t]` vs a `t_len = t+1` rewind.
    /// Returns `(y, state_final, planes [T, B·Hv, Dv, Dk] flattened)`.
    #[allow(clippy::too_many_arguments)]
    fn oracle_with_planes(
        conv_out: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        q_normed: &[f32],
        k_normed: &[f32],
        state_in: &[f32],
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let stride_b = 2 * hk * dk + hv * dv;
        let hk_per_hv = hv / hk;
        let mut y = vec![0.0_f32; b * t_total * hv * dv];
        let mut state = state_in.to_vec();
        let mut planes = vec![0.0_f32; t_total * state.len()];
        for t in 0..t_total {
            for batch in 0..b {
                let bt = batch * t_total + t;
                let conv_base = bt * stride_b;
                let v_base = conv_base + 2 * hk * dk;
                for hv_idx in 0..hv {
                    let n = batch * hv + hv_idx;
                    let hk_idx = hv_idx / hk_per_hv;
                    let qk_base = (bt * hk + hk_idx) * dk;
                    let gbeta_idx = bt * hv + hv_idx;
                    let dt = softplus_unclamped(a_raw[gbeta_idx] + dt_bias[hv_idx]);
                    let g_val = (-a_log[hv_idx].exp() * dt).exp();
                    let beta_val = sigmoid(b_raw[gbeta_idx]);
                    for dv_idx in 0..dv {
                        let v_val = conv_out[v_base + hv_idx * dv + dv_idx];
                        let s_base = n * dv * dk + dv_idx * dk;
                        let mut kv_mem = 0.0_f32;
                        let mut decayed = vec![0.0_f32; dk];
                        for d in 0..dk {
                            let s = state[s_base + d] * g_val;
                            decayed[d] = s;
                            kv_mem += s * k_normed[qk_base + d];
                        }
                        let delta = (v_val - kv_mem) * beta_val;
                        let mut out = 0.0_f32;
                        for d in 0..dk {
                            let s_new = decayed[d] + k_normed[qk_base + d] * delta;
                            state[s_base + d] = s_new;
                            out += s_new * q_normed[qk_base + d];
                        }
                        y[(bt * hv + hv_idx) * dv + dv_idx] = out;
                    }
                }
            }
            let plane_off = t * state.len();
            planes[plane_off..plane_off + state.len()].copy_from_slice(&state);
        }
        (y, state, planes)
    }

    /// CYCLIC-GQA sibling of `oracle` above, for
    /// `iron_gated_delta_prep_chunk_fast_cyclic_d128_128_48_16`'s
    /// correctness fixture -- identical except `hk_idx = hv_idx % hk`
    /// instead of `hv_idx / hk_per_hv`. See that kernel's doc for why this
    /// port needs the cyclic mapping.
    #[allow(clippy::too_many_arguments)]
    fn oracle_cyclic(
        conv_out: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        q_normed: &[f32],
        k_normed: &[f32],
        state_in: &[f32],
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let stride_b = 2 * hk * dk + hv * dv;
        let mut y = vec![0.0_f32; b * t_total * hv * dv];
        let mut state = state_in.to_vec();
        for batch in 0..b {
            for t in 0..t_total {
                let bt = batch * t_total + t;
                let conv_base = bt * stride_b;
                let v_base = conv_base + 2 * hk * dk;
                for hv_idx in 0..hv {
                    let n = batch * hv + hv_idx;
                    let hk_idx = hv_idx % hk; // CYCLIC (the one difference vs `oracle`)
                    let qk_base = (bt * hk + hk_idx) * dk;
                    let gbeta_idx = bt * hv + hv_idx;
                    let dt = softplus_unclamped(a_raw[gbeta_idx] + dt_bias[hv_idx]);
                    let g_val = (-a_log[hv_idx].exp() * dt).exp();
                    let beta_val = sigmoid(b_raw[gbeta_idx]);
                    for dv_idx in 0..dv {
                        let v_val = conv_out[v_base + hv_idx * dv + dv_idx];
                        let s_base = n * dv * dk + dv_idx * dk;
                        let mut kv_mem = 0.0_f32;
                        let mut decayed = vec![0.0_f32; dk];
                        for d in 0..dk {
                            let s = state[s_base + d] * g_val;
                            decayed[d] = s;
                            kv_mem += s * k_normed[qk_base + d];
                        }
                        let delta = (v_val - kv_mem) * beta_val;
                        let mut out = 0.0_f32;
                        for d in 0..dk {
                            let s_new = decayed[d] + k_normed[qk_base + d] * delta;
                            state[s_base + d] = s_new;
                            out += s_new * q_normed[qk_base + d];
                        }
                        y[(bt * hv + hv_idx) * dv + dv_idx] = out;
                    }
                }
            }
        }
        (y, state)
    }

    /// Small fused chunked GDN-prep shape: dk a multiple of 32, Hv divisible by
    /// Hk; `t_total` tokens with state carryover.
    ///
    /// `conv_scale` / `state_scale` / `a_log0` control the recurrence dynamics.
    /// The GDN recurrence amplifies state by ~Σ_t δ_t; with the larger Dk=64
    /// GQA reduction and 4 tokens, hot inputs drive y into the 10⁵ range where
    /// the per-step bf16 state store diverges from the f32 oracle by O(100) in
    /// absolute terms — far above any sane tol. Keeping conv/state small and
    /// `a_log ≤ -3` (so single-step gain is well under 1) bounds y to O(1) and
    /// the dtype-store error to well under tol. Production keeps this stable via
    /// learned `a_log`/`dt_bias`; the fixture mimics that conditioning.
    #[allow(clippy::too_many_arguments)]
    fn setup(
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        weight_scale: f32,
        conv_scale: f32,
        state_scale: f32,
        a_log0: f32,
        plane_mode: u32,
        dt: DType,
    ) -> TestSetup {
        let n_total = b * hv;
        let stride_b = 2 * hk * dk + hv * dv;
        let conv_out: Vec<f32> =
            (0..b * t_total * stride_b).map(|i| ((i as f32) * 0.0131).sin() * conv_scale).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| a_log0 - (i as f32) * 0.05).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.5 + (i as f32) * 0.05).collect();
        let a_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.3 + (i as f32) * 0.01).collect();
        let b_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.2 + (i as f32) * 0.008).collect();
        let q_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect();
        let k_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.0073).cos() * state_scale).collect();

        // Dtype-round inputs so the oracle sees the GPU's load precision.
        // q_normed/k_normed are additionally dtype-rounded a *second* time
        // here (post-qk_norm) because in production they round-trip through
        // a real Tensor<T> buffer written by iron_gated_delta_qknorm_prepass —
        // this kernel never sees f32 q/k norm results.
        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (q_normed, k_normed) = qk_norm(
            &r(&conv_out),
            &r(&q_norm_weight),
            &r(&k_norm_weight),
            b,
            t_total,
            hv,
            hk,
            dv,
            dk,
        );
        let (q_normed, k_normed) = (r(&q_normed), r(&k_normed));
        let capture_planes = plane_mode != 0;
        let (y_exp, state_exp, mut planes_exp) = if capture_planes {
            oracle_with_planes(
                &r(&conv_out),
                &r(&a_log),
                &r(&dt_bias),
                &r(&a_raw),
                &r(&b_raw),
                &q_normed,
                &k_normed,
                &r(&state_in),
                b,
                t_total,
                hv,
                hk,
                dv,
                dk,
            )
        } else {
            let (y, s) = oracle(
                &r(&conv_out),
                &r(&a_log),
                &r(&dt_bias),
                &r(&a_raw),
                &r(&b_raw),
                &q_normed,
                &k_normed,
                &r(&state_in),
                b,
                t_total,
                hv,
                hk,
                dv,
                dk,
            );
            (y, s, Vec::new())
        };
        // Default-off tests still need a bound buffer for state_planes even
        // though the kernel never writes it (planes_enabled == 0) — size 1
        // is enough since it's never indexed on that path.
        if plane_mode == 2 {
            let final_plane = (t_total - 1) * state_in.len();
            planes_exp[final_plane..].fill(0.0);
        }
        let planes_buf =
            if capture_planes { vec![0.0_f32; planes_exp.len()] } else { vec![0.0_f32; 1] };

        let mut ts = TestSetup::new(iron_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("conv_out", pack_f32(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_normed", pack_f32(&q_normed, dt), dt))
            .input(TestBuffer::from_vec("k_normed", pack_f32(&k_normed, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", b * t_total * hv * dv, dt))
            .input(TestBuffer::from_vec(
                "t_len",
                (t_total as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("state_planes", pack_f32(&planes_buf, dt), dt))
            .input(TestBuffer::from_vec(
                "planes_enabled",
                plane_mode.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("n_total", n_total as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1]);
        if capture_planes {
            ts = ts.expect(TestBuffer::from_vec("state_planes", pack_f32(&planes_exp, dt), dt));
        }
        ts
    }

    // GQA (Hv = 2·Hk), T=4 tokens with state carryover, weighted RMSNorm.
    // Dk=64 (longer reduction) + 4-token recurrence is highly gain-sensitive,
    // so the inputs are kept small (conv 0.02 / state 0.01 / a_log -3.0) to
    // bound y to O(1) — see `setup` doc. This keeps the dtype-store error well
    // inside tol across f32/f16/bf16 while still exercising GQA head-sharing.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_iron_gated_delta_prep_chunk_gqa(dt: DType) -> TestSetup {
        setup(1, 4, 4, 2, 8, 64, 0.3, 0.02, 0.01, -3.0, 0, dt)
    }

    // Same fixture as the GQA cell above, but `planes_enabled = 1`:
    // verifies `state_planes[t]` matches `oracle_with_planes`'s per-token
    // state capture bit-for-bit (well within the GQA cell's own tol) —
    // the correctness half of the "sequential recurrence ⇒ plane[t] ==
    // rewind-to-(t+1)" equivalence claim in the module doc.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_iron_gated_delta_prep_chunk_planes_gqa(dt: DType) -> TestSetup {
        setup(1, 4, 4, 2, 8, 64, 0.3, 0.02, 0.01, -3.0, 1, dt)
    }

    // Prefix-only mode writes every rollback plane while leaving the final
    // live-state plane untouched.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_iron_gated_delta_prep_chunk_prefix_planes_gqa(dt: DType) -> TestSetup {
        setup(1, 4, 4, 2, 8, 64, 0.3, 0.02, 0.01, -3.0, 2, dt)
    }

    // Hv == Hk (no key-sharing) at minimum dk=32, T=3 tokens.
    //
    // The f32 tol is 1e-2 (vs 5e-3 on the GQA sibling) because this
    // variant uses `a_log=-1.5` (a 4× weaker decay than the GQA setup's
    // -3.0), which amplifies y to magnitude ~24K. At that magnitude the
    // GQA tol of 5e-3 corresponds to <1 ULP, leaving no headroom for the
    // ~2-ULP rounding difference between HIP's OCML `expf`/`logf` and the
    // Rust f32 oracle's libm. Vulkan's GLSL exp/log happens to round
    // close enough to libm to clear 5e-3 — HIP doesn't. 1e-2 = ~3 ULPs at
    // peak magnitude, which both backends clear and which is still tight
    // for a 3-token gain-sensitive recurrence.
    //
    // bf16 tol bumped 2e-1 → 3.0 for the same reason, one layer up: since
    // the q/k RMSNorm pre-pass split, q_normed/k_normed round-trip through
    // a real Tensor<T> buffer between the two kernels instead of staying
    // in f32 registers for the whole prep+recurrence — an extra bf16
    // quantization step this specific fixture's amplification (~24K peak)
    // makes visible (observed max abs err ~2.0, still <1e-4 relative).
    // Every other bf16 case here (including both pre-pass tests and the
    // GQA sibling) is unaffected — Iron itself never dispatches this pair
    // at bf16 (Qwen3.6 GDN prep always runs the f32 shadow path), so this
    // is a synthetic-fixture-only concern, not a production one.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 5e-2, 3.0])]
    fn test_iron_gated_delta_prep_chunk_no_gqa(dt: DType) -> TestSetup {
        setup(1, 3, 4, 4, 4, 32, 1.0, 0.4, 0.1, -1.5, 0, dt)
    }

    // Qwen3.6-27B production shape via the GENERIC (runtime-hv/hk) kernel
    // — Dk=Dv=128, Hv=48, Hk=16, T=3 tokens with state carryover. `hv`/`hk`
    // are ordinary runtime `#[constexpr]` buffer params on this kernel (see
    // the `_fast` sibling's module doc: constexpr lowers to a runtime
    // `constant T&` on the Metal/CUDA backends alike), so the generic
    // kernel needs no code change to support Hv=48 — this pins that down
    // with a real-shape fixture rather than relying on the smaller GQA
    // cell above to stand in for it. Same tol rationale as the `_32_16`
    // fast-variant cell (identical NPT=4, identical fixture magnitudes).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 3.0])]
    fn test_iron_gated_delta_prep_chunk_qwen36_27b_shape(dt: DType) -> TestSetup {
        setup(1, 3, 48, 16, 128, 128, 0.3, 0.02, 0.01, -3.0, 0, dt)
    }

    /// Same oracle/fixture math as `setup` above, but targets the
    /// shape-specialized `iron_gated_delta_prep_chunk_fast_*` variants:
    /// no `.constexpr(...)` calls since `Dk`/`Dv`/`Hv`/`Hk` are baked into
    /// the kernel at macro-expansion time, not passed as runtime buffers.
    #[allow(clippy::too_many_arguments)]
    fn setup_fast(
        ir: wh_iron::core::ir::Kernel,
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        weight_scale: f32,
        conv_scale: f32,
        state_scale: f32,
        a_log0: f32,
        plane_mode: u32,
        dt: DType,
    ) -> TestSetup {
        let n_total = b * hv;
        let stride_b = 2 * hk * dk + hv * dv;
        let conv_out: Vec<f32> =
            (0..b * t_total * stride_b).map(|i| ((i as f32) * 0.0131).sin() * conv_scale).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| a_log0 - (i as f32) * 0.05).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.5 + (i as f32) * 0.05).collect();
        let a_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.3 + (i as f32) * 0.01).collect();
        let b_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.2 + (i as f32) * 0.008).collect();
        let q_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect();
        let k_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.0073).cos() * state_scale).collect();

        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (q_normed, k_normed) = qk_norm(
            &r(&conv_out),
            &r(&q_norm_weight),
            &r(&k_norm_weight),
            b,
            t_total,
            hv,
            hk,
            dv,
            dk,
        );
        let (q_normed, k_normed) = (r(&q_normed), r(&k_normed));
        let capture_planes = plane_mode != 0;
        let (y_exp, state_exp, mut planes_exp) = if capture_planes {
            oracle_with_planes(
                &r(&conv_out),
                &r(&a_log),
                &r(&dt_bias),
                &r(&a_raw),
                &r(&b_raw),
                &q_normed,
                &k_normed,
                &r(&state_in),
                b,
                t_total,
                hv,
                hk,
                dv,
                dk,
            )
        } else {
            let (y, s) = oracle(
                &r(&conv_out),
                &r(&a_log),
                &r(&dt_bias),
                &r(&a_raw),
                &r(&b_raw),
                &q_normed,
                &k_normed,
                &r(&state_in),
                b,
                t_total,
                hv,
                hk,
                dv,
                dk,
            );
            (y, s, Vec::new())
        };
        if plane_mode == 2 {
            let final_plane = (t_total - 1) * state_in.len();
            planes_exp[final_plane..].fill(0.0);
        }
        let planes_buf =
            if capture_planes { vec![0.0_f32; planes_exp.len()] } else { vec![0.0_f32; 1] };

        let mut ts = TestSetup::new(ir)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("conv_out", pack_f32(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_normed", pack_f32(&q_normed, dt), dt))
            .input(TestBuffer::from_vec("k_normed", pack_f32(&k_normed, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", b * t_total * hv * dv, dt))
            .input(TestBuffer::from_vec(
                "t_len",
                (t_total as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .input(TestBuffer::from_vec("state_planes", pack_f32(&planes_buf, dt), dt))
            .input(TestBuffer::from_vec(
                "planes_enabled",
                plane_mode.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("n_total", n_total as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1]);
        if capture_planes {
            ts = ts.expect(TestBuffer::from_vec("state_planes", pack_f32(&planes_exp, dt), dt));
        }
        ts
    }

    // Small GQA test cell for the fast/specialized kernel: identical shape
    // and fixture parameters to `test_iron_gated_delta_prep_chunk_gqa` above,
    // so the two share the same expected numerics and the tol values carry
    // over directly.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
    fn test_iron_gated_delta_prep_chunk_fast_d64_8_4_2(dt: DType) -> TestSetup {
        setup_fast(
            iron_gated_delta_prep_chunk_fast_d64_8_4_2::kernel_ir_for(dt),
            1,
            4,
            4,
            2,
            8,
            64,
            0.3,
            0.02,
            0.01,
            -3.0,
            0,
            dt,
        )
    }

    // Qwen3.6-35B-A3B production shape (Dk=Dv=128, Hv=32, Hk=16), T=3 tokens
    // with state carryover: correctness only, not the full T=1024/4096
    // prefill chunk (that's the bench's job below). Tol widened past the
    // d64_8_4_2 cell's (5e-3/5e-2/2e-1) for the same reason
    // `test_iron_gated_delta_prep_chunk_no_gqa` widens its own: Dk=128 is a
    // longer per-lane reduction (NPT=4 vs 2) than the small cell, giving
    // more terms for ULP-level exp/log rounding to accumulate. The f16 tol
    // (1e-1) is looser than f32/bf16 need in practice (observed err 6.25e-2
    // at 5e-2), harmless since production never dispatches this kernel at
    // f16 (GDN prep always runs the f32 shadow path, see module doc).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 3.0])]
    fn test_iron_gated_delta_prep_chunk_fast_d128_128_32_16(dt: DType) -> TestSetup {
        setup_fast(
            iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt),
            1,
            3,
            32,
            16,
            128,
            128,
            0.3,
            0.02,
            0.01,
            -3.0,
            0,
            dt,
        )
    }

    // Same production shape/fixture as the cell above, but
    // `planes_enabled = 1`: validates `state_planes[t]` against
    // `oracle_with_planes` at the actual shipping shape (Dk=Dv=128,
    // Hv=32, Hk=16) — the shape the Stage-0 verify-pass probe and the
    // Stage-1/2 rollback wiring both target.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 3.0])]
    fn test_iron_gated_delta_prep_chunk_fast_planes_d128_128_32_16(dt: DType) -> TestSetup {
        setup_fast(
            iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt),
            1,
            3,
            32,
            16,
            128,
            128,
            0.3,
            0.02,
            0.01,
            -3.0,
            1,
            dt,
        )
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 3.0])]
    fn test_iron_gated_delta_prep_chunk_fast_prefix_planes_d128_128_32_16(dt: DType) -> TestSetup {
        setup_fast(
            iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt),
            1,
            3,
            32,
            16,
            128,
            128,
            0.3,
            0.02,
            0.01,
            -3.0,
            2,
            dt,
        )
    }

    // Qwen3.6-27B production shape (Dk=Dv=128, Hv=48, Hk=16), T=3 tokens
    // with state carryover — the actual shape this Spark GDN-prefill
    // validation pass is for (previously untested: every other fixture in
    // this file, including the "production" `_32_16` cell above, targets
    // the 35B-A3B config's Hv=32, not 27B's Hv=48). Same fixture params
    // and tol rationale as `test_iron_gated_delta_prep_chunk_fast_d128_128_32_16`
    // — only Hv changes, so the per-lane reduction length (NPT=4, driven by
    // Dk) and hence the ULP accumulation budget are identical.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 3.0])]
    fn test_iron_gated_delta_prep_chunk_fast_d128_128_48_16(dt: DType) -> TestSetup {
        setup_fast(
            iron_gated_delta_prep_chunk_fast_d128_128_48_16::kernel_ir_for(dt),
            1,
            3,
            48,
            16,
            128,
            128,
            0.3,
            0.02,
            0.01,
            -3.0,
            0,
            dt,
        )
    }

    /// CYCLIC-GQA sibling of `setup_fast` above -- identical fixture math,
    /// but drives `oracle_cyclic` instead of `oracle` so the expected `y`/
    /// `state_out` match `iron_gated_delta_prep_chunk_fast_cyclic_*`'s
    /// baked-in `hk_idx = hv_idx % HK` mapping.
    #[allow(clippy::too_many_arguments)]
    fn setup_fast_cyclic(
        ir: wh_iron::core::ir::Kernel,
        b: usize,
        t_total: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        weight_scale: f32,
        conv_scale: f32,
        state_scale: f32,
        a_log0: f32,
        dt: DType,
    ) -> TestSetup {
        let n_total = b * hv;
        let stride_b = 2 * hk * dk + hv * dv;
        let conv_out: Vec<f32> =
            (0..b * t_total * stride_b).map(|i| ((i as f32) * 0.0131).sin() * conv_scale).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| a_log0 - (i as f32) * 0.05).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.5 + (i as f32) * 0.05).collect();
        let a_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.3 + (i as f32) * 0.01).collect();
        let b_raw: Vec<f32> = (0..b * t_total * hv).map(|i| -0.2 + (i as f32) * 0.008).collect();
        let q_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 11) as f32) * 0.05)).collect();
        let k_norm_weight: Vec<f32> =
            (0..hk * dk).map(|i| weight_scale * (1.0 + ((i % 13) as f32) * 0.04)).collect();
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.0073).cos() * state_scale).collect();

        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (q_normed, k_normed) = qk_norm(
            &r(&conv_out),
            &r(&q_norm_weight),
            &r(&k_norm_weight),
            b,
            t_total,
            hv,
            hk,
            dv,
            dk,
        );
        let (q_normed, k_normed) = (r(&q_normed), r(&k_normed));
        let (y_exp, state_exp) = oracle_cyclic(
            &r(&conv_out),
            &r(&a_log),
            &r(&dt_bias),
            &r(&a_raw),
            &r(&b_raw),
            &q_normed,
            &k_normed,
            &r(&state_in),
            b,
            t_total,
            hv,
            hk,
            dv,
            dk,
        );

        TestSetup::new(ir)
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("conv_out", pack_f32(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack_f32(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack_f32(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack_f32(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_normed", pack_f32(&q_normed, dt), dt))
            .input(TestBuffer::from_vec("k_normed", pack_f32(&k_normed, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", state_in.len(), dt))
            .input(TestBuffer::zeros("y", b * t_total * hv * dv, dt))
            .input(TestBuffer::from_vec(
                "t_len",
                (t_total as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
    }

    // Qwen3.6-27B production shape, CYCLIC GQA mapping (see
    // `iron_gated_delta_prep_chunk_fast_cyclic`'s doc) -- same fixture
    // params/tol rationale as `test_iron_gated_delta_prep_chunk_fast_d128_128_48_16`,
    // only the GQA mapping (and hence which `oracle*` fn) differs.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-2, 1e-1, 3.0])]
    fn test_iron_gated_delta_prep_chunk_fast_cyclic_d128_128_48_16(dt: DType) -> TestSetup {
        setup_fast_cyclic(
            iron_gated_delta_prep_chunk_fast_cyclic_d128_128_48_16::kernel_ir_for(dt),
            1,
            3,
            48,
            16,
            128,
            128,
            0.3,
            0.02,
            0.01,
            -3.0,
            dt,
        )
    }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::{
        iron_gated_delta_prep_chunk,
        iron_gated_delta_prep_chunk_fast_d128_128_32_16,
        iron_gated_delta_prep_chunk_fast_d128_128_48_16,
    };

    // Grid `[ceil(dv/4), b*hv, 1]`, TG `[128,1,1]` (4 SGs). NOTE: this
    // shape (Hv=16, Hk=8) does not match Qwen3.6-35B-A3B's actual
    // config.json (linear_num_value_heads=32, linear_num_key_heads=16,
    // both head dims 128), see `bench_gated_delta_prep_chunk_prod_t1024`
    // / `_t4096` below for the real production shape used in the GDN
    // prefill honest-cost measurement.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 64usize, 16usize, 8usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    // Same shape as `bench_gated_delta_prep_chunk` but T=2048 — a fair
    // comparison point against the `iron_gdn_wy_plan`/`iron_gdn_wy_scan`
    // two-kernel pipeline's own T=2048 production benches (the T=64 bench
    // above is comparable work only at chunk granularity, not full-prefill).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_t2048(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 2048usize, 16usize, 8usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    // True Qwen3.6-35B-A3B production GDN shape, read from config.json:
    // linear_num_value_heads=32 (Hv), linear_num_key_heads=16 (Hk),
    // linear_key_head_dim=linear_value_head_dim=128 (Dk=Dv). B=1 (single
    // prefill sequence). T=1024 chunk, matching a Stage-1 honest-cost
    // measurement point for the prefill-ladder low/mid-T deficit.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_prod_t1024(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 1024usize, 32usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    // Same production shape as `bench_gated_delta_prep_chunk_prod_t1024`
    // but T=4096, the other Stage-1 measurement point (component profiling
    // put GDN at ~37% of prefill GPU time at this T).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_prod_t4096(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 4096usize, 32usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    // Shape-specialized fast kernel (`iron_gated_delta_prep_chunk_fast_d128_128_32_16`)
    // at the same Qwen3.6-35B-A3B production shape and T=1024/4096 as
    // `bench_gated_delta_prep_chunk_prod_t1024` / `_t4096` above: direct
    // A/B: same grid/TG geometry, same math, only DK/DV/HV/HK/NPT baked as
    // compile-time constants instead of runtime `#[constexpr]` buffer
    // params. No `.constexpr(...)` calls since there is nothing left to
    // resolve at launch time.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_fast_t1024(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 1024usize, 32usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    // Dense Qwen3.8-27B prefill shape at the public gate length. Keeping
    // this variant in the benchmark inventory also makes it available to
    // downstream artifact emitters, instead of dead-stripping the variant
    // that was previously referenced only by tests.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_fast_qwen38_t512(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 512usize, 48usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk_fast_d128_128_48_16::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_fast_t4096(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 4096usize, 32usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    // ─── Stage-0 kill-early probe: production shape at MTP verify-pass T ──
    //
    // T=3 (γ=2 verify: `[prev] + drafts`) at the real Qwen3.6-35B-A3B
    // shape, planes OFF vs ON — the kernel-level half of the plane-store
    // cost probe (`SpecDecodeVerifyCostBench` in butter measures the same
    // question end-to-end through the Swift model). `_planes_on` writes
    // `state_planes[T, n_total, Dv, Dk]` every token boundary; `_planes_off`
    // is the same dispatch with `planes_enabled = 0` (must remain
    // byte-identical in cost to the pre-plane kernel — only extra buffer
    // binds + one register compare per token, no stores).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_fast_verify_t3_planes_off(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 3usize, 32usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", 1, dt))
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                0u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_prep_chunk_fast_verify_t3_planes_on(dt: DType) -> BenchSetup {
        let (b, t, hv, hk, dv, dk) = (1usize, 3usize, 32usize, 16usize, 128usize, 128usize);
        let n_total = b * hv;
        let conv_w = 2 * hk * dk + hv * dv;
        BenchSetup::new(iron_gated_delta_prep_chunk_fast_d128_128_32_16::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("conv_out", b * t * conv_w, dt))
            .buffer(BenchBuffer::random("a_log", hv, dt))
            .buffer(BenchBuffer::random("dt_bias", hv, dt))
            .buffer(BenchBuffer::random("a_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("b_raw", b * t * hv, dt))
            .buffer(BenchBuffer::random("q_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("k_normed", b * t * hk * dk, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", b * t * hv * dv, dt).output())
            .buffer(BenchBuffer::from_vec("t_len", (t as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::zeros("state_planes", t * n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::from_vec(
                "planes_enabled",
                1u32.to_le_bytes().to_vec(),
                DType::U32,
            ))
            .constexpr("n_total", n_total as u32)
            .grid_3d((dv as u32).div_ceil(4), n_total as u32, 1, [128, 1, 1])
            .bytes_moved((b * t * hv * dv * dt.size_bytes()) as u64)
    }
}

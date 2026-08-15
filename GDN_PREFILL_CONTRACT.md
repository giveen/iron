# GDN (Gated DeltaNet) prefill kernel contract — Qwen3.6-27B

Pre-validation pass for the Spark (GB10, CUDA/sm_121) Qwen3.6-27B port. Scope:
`crates/wh-iron-std/src/kernels/ssm/gated_delta*.rs` (+ `gated_delta_qknorm_prepass.rs`).
Shape for this model: `linear_num_key_heads=16` (Hk), `linear_num_value_heads=48`
(Hv), `linear_key_head_dim=linear_value_head_dim=128` (Dk=Dv), conv kernel size 4.

Branch: `gdn-prefill-validation` (git worktree at `~/iron-gdnwork` on Spark).

**Read this before touching GDN on this port.** Section 5 ("what to actually
call") is the short answer if you just need the decode-step math or the
prefill call sequence.

---

## 1. Inventory — what's in `kernels/ssm/gated_delta*.rs`

Two independent algorithmic families implement the same math (the gated
delta-rule recurrence, defined in §2) at different granularities:

| File | Kernel(s) | What it computes | Status on this port |
|---|---|---|---|
| `gated_delta.rs` | `iron_gated_delta_step` | single-token decode, **no fused prep** — caller supplies precomputed `g`/`beta` | tested, generic Hv/Hk, works |
| `gated_delta.rs` | `iron_gated_delta_chunk` | **naive** sequential T-token prefill — same math as `_step`, T-loop inside one dispatch, no chunk-parallel algebra | tested, generic Hv/Hk, works |
| `gated_delta_prep.rs` | `iron_gated_delta_prep_step` | decode + **fused prep** (conv-split, q/k RMSNorm, g/beta from raw a/b) | tested at Hv=32; generic, works at Hv=48 (validated this pass) |
| `gated_delta_qknorm_prepass.rs` | `iron_gated_delta_qknorm_prepass` | q/k RMSNorm pre-pass, hoisted out of the chunk kernel's redundant per-`(dv,hv/hk)` recompute | tested at Hv=32; generic, works at Hv=48 (validated this pass) |
| `gated_delta_prep_chunk.rs` | `iron_gated_delta_prep_chunk` | **THE PRODUCTION PREFILL KERNEL.** Fused prep + sequential T-loop recurrence, register-resident state, one dispatch per external chunk | tested at Hv=32; **validated at Hv=48 this pass** |
| `gated_delta_prep_chunk.rs` | `iron_gated_delta_prep_chunk_fast_{d128_128_32_16,d64_8_4_2}` | same math, `Dk/Dv/Hv/Hk` baked as compile-time constants so the Metal/NVRTC compiler can register-promote the per-lane state array | Hv=32 variant tested; **Hv=48 (`d128_128_48_16`) variant added + tested this pass — did not exist before** |
| `gated_delta_gate_beta.rs` | `iron_gdn_gate_beta` | elementwise `g`/`beta` precompute, feeds the WY plan/scan pipeline below | tested, generic |
| `gated_delta_wy.rs` | `iron_gated_delta_wy_chunk` | **monolithic** chunked-WY (Woodbury-Young) prefill — the original chunk-parallel algorithm. **Dormant**: its TG-resident `[Dv,Dk]` state buffer is 64 KiB at production Dk=Dv=128, over Apple's ~32 KiB TG budget, so it can never dispatch at real shapes | small-shape-only, not production-viable at any Dk≥~32 |
| `gated_delta_wy_plan.rs` + `gated_delta_wy_scan.rs` | `iron_gdn_wy_plan` / `iron_gdn_wy_scan` | two-kernel split of the WY algorithm (chunk-parallel "plan" pass + chunk-sequential "scan" pass) that DOES fit production Dk=Dv=128 via MMA tiling, by never materializing a full `[Dv,Dk]` buffer | **not currently wired into any production call site** (no reference to it outside its own tests/benches); **broken on CUDA/GB10, see §4** |
| `gated_delta_replay.rs` | `iron_gated_delta_step_record` / `iron_state_replay` | decode step + delta-tape capture/replay for speculative-decode rollback | out of scope for prefill; uses a stale `Dk=192` shape variant from an earlier Qwen3.5 config, not this port's Dk=128 |
| `ssm_replay.rs` | `iron_ssm_step_record` / `iron_ssm_replay` | **Mamba/Mamba2**, a different SSM family (Nemotron-class models) — despite living in the same directory, this is NOT GDN | out of scope, wrong model family |
| `kernels/convolution/conv1d_causal.rs` | `iron_conv1d_causal_step` / `iron_conv1d_causal_prefill` | the depthwise causal conv (kernel_size=4, SiLU) that produces `conv_out` from raw q/k/v projections — GDN's actual input | not part of this family; generic over `kernel_size` as a runtime param, needs no change for `kernel_size=4` |

**Bottom line: `iron_gated_delta_prep_chunk` (+ its `_fast` shape-specialized
sibling) is the kernel this port needs and the one now validated at the real
27B shape.** The WY plan/scan pipeline is a more algorithmically sophisticated
future-perf path that isn't wired in yet and currently doesn't run on this
CUDA hardware at all (§4) — do not reach for it for this port.

---

## 2. The math — gated delta-rule recurrence

Every kernel in this family computes the same recurrence over a
per-`(batch, Hv-head)` state `S ∈ R^{Dv×Dk}`. Read off `gated_delta.rs`'s
kernel body directly (all kernels in the family implement this identically —
confirmed by diffing `gated_delta.rs`, `gated_delta_prep.rs`, and the inner
T-loop of `gated_delta_prep_chunk.rs`):

```
Per token t, per Hv-head:
  S_decayed = S · g_t                                    # forget-gate decay, elementwise
  kv_mem    = S_decayed · k_t              (∈ R^Dv)       # reduction over Dk
  delta     = (v_t − kv_mem) · beta_t      (∈ R^Dv)
  S_new     = S_decayed + outer(delta, k_t)               # rank-1 (outer-product) update
  y_t       = S_new · q_t                  (∈ R^Dv)       # reduction over Dk
  S ← S_new
```

GQA: `hk_idx = hv_idx / (Hv/Hk)` — each group of `Hv/Hk` value-heads shares
one key/query head.

### Gate / beta derivation (from raw projections)

`gated_delta_prep.rs`, `gated_delta_prep_chunk.rs`, and `gated_delta_gate_beta.rs`
all compute `g`/`beta` from the same raw learned scalars, byte-for-byte
identical formula in all three:

```
dt    = log(exp(a_raw + dt_bias) + 1)          # un-clamped softplus; safe at
                                                # production fp32 magnitudes
g     = exp(−exp(a_log) · dt)                  # forget gate, ∈ (0,1)
beta  = sigmoid(b_raw)                         # write strength, ∈ (0,1)
```

`a_log`/`dt_bias` are per-Hv-head learned parameters (`[Hv]`); `a_raw`/`b_raw`
are per-token per-Hv-head (`[T, Hv]` for the chunk path, `[Hv]` for one decode
step).

### q/k RMSNorm (state-independent, hoisted out of the recurrence)

```
q_normed = q · rsqrt(mean(q²) + 1e-6) · q_norm_weight     # per Hk-head, over Dk
k_normed = k · rsqrt(mean(k²) + 1e-6) · k_norm_weight
```

`iron_gated_delta_qknorm_prepass` computes this ONCE per `(b, t, hk_idx)` —
`iron_gated_delta_prep_chunk`'s own dispatch grid (`[Dv, B·Hv, 1]`) would
otherwise redundantly recompute it `Dv · (Hv/Hk)` times per token if it did
this itself (as `iron_gated_delta_prep_step`, the single-token decode kernel,
does — redundancy doesn't matter there since there's no `T` to amortize
over).

---

## 3. Composition — the production T-token prefill pipeline

```
raw q/k/v projections
        │
        ▼
┌───────────────────────────────────────────┐
│ iron_conv1d_causal_prefill (kernel_size=4) │   NOT part of this kernel family;
│ depthwise causal conv + SiLU               │   see kernels/convolution/conv1d_causal.rs
└───────────────────────────────────────────┘
        │  conv_out : [B, T, 2·Hk·Dk + Hv·Dv]   (q | k | v slabs, interleaved per token)
        ▼
┌────────────────────────────────────────────┐
│ iron_gated_delta_qknorm_prepass             │   grid [T, B·Hk, 1], TG [32,1,1]
│ per-(b,t,hk_idx) q/k RMSNorm                │   reads conv_out's q/k slabs only
└────────────────────────────────────────────┘
        │  q_normed, k_normed : [B, T, Hk, Dk]
        ▼
┌────────────────────────────────────────────┐
│ iron_gated_delta_prep_chunk                 │   grid [ceil(Dv/4), B·Hv, 1], TG [128,1,1]
│  (or _fast_d128_128_48_16 for this shape)   │   (4 simdgroups/warps per TG, one per Dv-in-4 slot)
│                                              │
│  state_reg ← state_in            (once)     │   reads conv_out's v slab + q_normed/k_normed
│  for t in 0..T:                             │   + a_log/dt_bias/a_raw/b_raw
│    g,beta ← softplus/sigmoid(a_raw[t],...)  │
│    state_reg ← recurrence step (§2)         │
│    y[t] ← state_reg · q_normed[t]           │
│  state_out ← state_reg           (once)     │
└────────────────────────────────────────────┘
        │  y : [B, T, Hv, Dv]        state_out : [B, Hv, Dv, Dk]
        ▼
   attention-output projection (outside this family)
```

**External chunking (splitting a long prefill across multiple dispatches)**:
call `iron_gated_delta_qknorm_prepass` + `iron_gated_delta_prep_chunk` once
per external chunk, threading `state_out` of dispatch *i* into `state_in` of
dispatch *i+1*. This is the exact scenario **validated this pass** (see §6,
check 2) — 3 external dispatches of T=3 tokens each, state carried across all
3, agrees with a single 9-token host-reference run to cosine ≥ 0.999999
(relative error ~2.4e-6; the ~0.6 absolute error is against a peak
`|y|`≈2.5e5 — see §6 for why this fixture legitimately reaches that
magnitude and why cosine, not `max|Δ|`, is the right metric there).

### Worked small example (T=1, Dk=Dv=2, Hv=Hk=1, B=1 — by hand)

```
q = k = [1, 0]         v = [2, 3]        S_in = [[0,0],[0,0]]     (2×2, Dv×Dk)
a_log=0, dt_bias=0, a_raw=0 → dt = log(2) ≈ 0.693 → g = exp(-1·0.693) ≈ 0.5
b_raw=0 → beta = sigmoid(0) = 0.5

S_decayed = S_in · g = [[0,0],[0,0]]                       (still zero — empty state)
kv_mem    = S_decayed · k = [0, 0]
delta     = (v − kv_mem) · beta = [2,3]·0.5 = [1, 1.5]
S_new     = S_decayed + outer(delta, k) = [[1·1, 1·0], [1.5·1, 1.5·0]]
          = [[1, 0], [1.5, 0]]
y         = S_new · q = [1·1+0·0, 1.5·1+0·0] = [1, 1.5]
```

This is exactly what `iron_gated_delta_prep_chunk`'s `oracle()` (in its own
`kernel_tests` module) and this pass's `host_decode_step` (in
`tests/gdn_qwen36_27b_shape_cuda.rs`) both compute, at any shape.

---

## 4. The decode single-token form — **for the port agent**

This is the exact single-token recurrent step, spelled out for the
decode-first port:

```
Inputs per token (one batch element, all Hv heads):
  q, k        : [Hk, Dk]      (or raw conv output, if using the _prep variant — see below)
  v           : [Hv, Dv]
  g, beta     : [Hv]          (precomputed) — OR a_raw/b_raw [Hv] + a_log/dt_bias [Hv] (raw, fused-prep variant)
  state       : [Hv, Dv, Dk]

For each hv_idx in 0..Hv:
  hk_idx = hv_idx / (Hv / Hk)                      # GQA group
  g_val, beta_val = as in §2 (either read directly, or derived from a_raw/b_raw)
  for dv_idx in 0..Dv:
    kv_mem = Σ_dk  state[hv_idx,dv_idx,dk] · g_val · k[hk_idx,dk]
    delta  = (v[hv_idx,dv_idx] − kv_mem) · beta_val
    for dk in 0..Dk:
      state[hv_idx,dv_idx,dk] = state[hv_idx,dv_idx,dk]·g_val + k[hk_idx,dk]·delta
    y[hv_idx,dv_idx] = Σ_dk  state[hv_idx,dv_idx,dk] · q[hk_idx,dk]
```

Two on-device kernels implement exactly this:

- **`iron_gated_delta_step`** (`gated_delta.rs`) — takes precomputed `g`/`beta`
  as inputs. Grid `[Dv, B·Hv, 1]`, TG `[32,1,1]` (one warp/simdgroup per
  `(dv_idx, b·Hv+hv_idx)`, `Dk` split across the 32 lanes,
  `n_per_t = Dk/32` per lane, `Dk % 32 == 0` required).
- **`iron_gated_delta_prep_step`** (`gated_delta_prep.rs`) — same recurrence,
  but ALSO does the conv-split + q/k RMSNorm + g/beta derivation from raw
  `a_raw`/`b_raw`/`a_log`/`dt_bias` in the same dispatch (fewer host-sync
  round trips per decode step — the one Iron's own doc comments describe as
  the actual production decode path: *"Drop-in replacement for the
  host-prep + iron_gated_delta_step pair in Qwen35GDNMixer.forward"*).
  **This is almost certainly the kernel the decode-first port should call.**
  Same dispatch geometry as `iron_gated_delta_step`.

**Proven equivalence to the chunk kernel** (this pass, `tests/gdn_qwen36_27b_shape_cuda.rs`
check 3, at the real Hv=48/Hk=16/Dk=Dv=128 shape): 3 sequential
`iron_gated_delta_prep_step` dispatches vs. one `iron_gated_delta_prep_chunk`
dispatch over the same 3 tokens produced **bit-identical** GPU output
(`max|Δy| = max|Δstate| = 0.0`). The decode-step kernel is not just
"the same math" as the chunk kernel's inner loop — on this CUDA backend, at
this shape, it currently reproduces it exactly.

---

## 5. What to actually call for this port

1. `iron_conv1d_causal_prefill` (or `_step` for decode) — kernel_size=4,
   generic, no change needed.
2. `iron_gated_delta_qknorm_prepass` — generic over Hv/Hk, works at
   Hv=48/Hk=16/Dk=Dv=128 (validated this pass).
3. `iron_gated_delta_prep_chunk_fast_d128_128_48_16` (new this pass) for the
   register-promoted fast path at this exact shape, or the generic
   `iron_gated_delta_prep_chunk` (also validated at this shape — slower, but
   correct, and useful as a fallback/oracle) if the fast variant isn't wired
   into the calling code yet.
4. For decode, `iron_gated_delta_prep_step` — proven bit-identical to (3)'s
   per-token behavior at this shape.
5. State threading across external chunks: `state_out` of call *i* → `state_in`
   of call *i+1*, `[Hv, Dv, Dk]` flat layout throughout, no transformation
   needed at the boundary.

**Do not reach for `iron_gdn_wy_plan`/`iron_gdn_wy_scan`** for this port — see
§1 and §7. They're an unfinished, currently-broken-on-CUDA alternative, not a
faster drop-in.

---

## 6. Test coverage added this pass

All at the real Qwen3.6-27B shape (Hk=16, Hv=48, Dk=Dv=128) unless noted.
Every prior fixture in this crate (`gated_delta_prep_chunk.rs`'s own
`kernel_tests`, `tests/gated_delta_prep_chunk_correctness.rs`, every WY-family
fixture) targeted Hv=32 (the 35B-A3B shape) — **Hv=48 was untested anywhere
in the family before this pass.**

### In-source `#[test_kernel]` additions (`gated_delta_prep_chunk.rs`)

- `iron_gated_delta_prep_chunk_fast_d128_128_48_16` — **new kernel variant**
  (register-promoted, compile-time `HV=48`). Did not exist before this pass;
  the existing `_fast` family only had `d128_128_32_16` (35B-A3B) and a small
  `d64_8_4_2` test cell.
- `test_iron_gated_delta_prep_chunk_fast_d128_128_48_16` — correctness for
  the new variant, T=3, tol `[1e-2, 1e-1, 3.0]` (f32/f16/bf16), same fixture
  recipe as the validated `_32_16` sibling.
- `test_iron_gated_delta_prep_chunk_qwen36_27b_shape` — same shape/fixture
  through the GENERIC (runtime-Hv) kernel, confirming it needs no code
  change for Hv=48 (Hv/Hk are ordinary runtime `#[constexpr]` buffer params
  there — ordinary NVRTC/MSL `constant` scalars either way, unlike the
  `_fast` variant's true compile-time constants).

Both pass on CUDA (`cargo test -p wh-iron-std --test gdn_family_cuda_corpus_scoped
--features cuda -- --nocapture`, see below).

### New file: `tests/gdn_qwen36_27b_shape_cuda.rs` (CUDA-backend integration test)

One `#[test]` (mirrors `tests/cuda_kernel_corpus.rs`'s single-test convention
to avoid concurrent `CudaDevice` contexts):

| Check | What | Result |
|---|---|---|
| 1 | Single chunk dispatch (T=3) vs. host reference | `max|Δy|=7.25e-5`, `max|Δstate|=5.48e-6` — **at the task's ~1e-4 target** |
| 2 | Inter-chunk state carry, 3 external dispatches (T=3 each, 9 total) vs. one 9-token host reference | cosine ≥ 0.999999 both y/state; relative error ~2.4e-6 (absolute `max|Δy|=0.625` is against peak `|y|≈2.55e5` — see the file's doc comment on why this fixture legitimately amplifies over 9 sequential steps, matching a hazard `gated_delta_prep_chunk_correctness.rs` already documents) |
| 3 | `iron_gated_delta_prep_step` × 3 (decode) vs. `iron_gated_delta_prep_chunk` (T=3) — GPU vs GPU | **bit-identical** (`max|Δ|=0.0`) |
| 3b | Same chunk-kernel output vs. host reference | `max|Δy|=1.14e-4`, `max|Δstate|=7.87e-6` |
| tiny | Hv=Hk=1, Dk=Dv=32, T=3 — fast debug cell | `max|Δy|=1.34e-7` |

**Why T=3 per dispatch, not more**: the delta-rule recurrence is
gain-sensitive. An earlier draft of this test ran T=16/48/8 directly and saw
`max|Δy|` in the **1e4 range** — not a kernel bug, but the same amplifying-
recurrence hazard `gated_delta_prep_chunk_correctness.rs`'s `make_fixture`
doc comment already warns about ("state overflows f32 around T=20" for a
similar recipe). `gated_delta_prep_chunk.rs`'s own production-shape fixture
(`test_iron_gated_delta_prep_chunk_fast_d128_128_32_16`) independently landed
on T=3 with a 1e-2 f32 tolerance for the identical reason. This file reuses
that validated operating point rather than re-deriving a new one — a real
prefill run stays stable at any T because `a_log`/`dt_bias` are *learned* to
keep gain ≤ 1, which synthetic test fixtures don't reproduce without
hand-tuning.

### New file: `tests/gdn_family_cuda_corpus_scoped.rs`

A fast (~5s), GDN-only-filtered version of `tests/cuda_kernel_corpus.rs`
(which runs the FULL multi-thousand-kernel registered corpus — not something
to run repeatedly under the "seconds-scale, other agents have priority" GPU
discipline this pass operated under). Filters `wh_iron_std::all_tests()` by
name containing `gated_delta` or `gdn_`, applies the same
`DeviceCapability` = not-a-failure classification `cuda_kernel_corpus.rs`
uses. Result before the wy_scan finding below: **PASS=54, UNSUPPORTED=9
(all `iron_gdn_wy_plan`, device-capability), FAIL=3** (`iron_gdn_wy_scan_single_tile`,
all 3 dtypes — see §7).

---

## 7. Gaps found — status after the CUDA-codegen follow-up pass(es)

§7.1/§7.2 are in the WY plan/scan pipeline (§1), which this port does not
need (§5). §7.3 is unrelated (found only as a regression check for §7.1).
§7.1 was found via the scoped CUDA corpus run in §6, and got a dedicated
CUDA-codegen investigation in a follow-up pass on top of this branch
(`gdn-prefill-validation`, still); §7.2 and §7.3 were root-caused and fixed
in a second follow-up pass on the same branch. Summary: **§7.1
(`iron_gdn_wy_plan` smem) got a real, validated, corpus-wide fix that cuts
its request by 27% but does not clear GB10's cap — still blocked, now
understood as fundamental to the current CUDA CoopTile lowering rather than
a simple formula bug (not touched in the second pass). §7.2 (`iron_gdn_wy_scan`
mismatch) is FIXED — root cause was a divergent `__syncthreads()` (CUDA
`compute-sanitizer --tool synccheck` catches it; `racecheck`/`initcheck`
don't). §7.3 (`test_moe_densify_int2_experts [f16]`) is also FIXED —
unrelated root cause, a CUDA-only implicit thread-count bounds guard sized
from the wrong quantity for that kernel's threading pattern.** The full
CUDA corpus (`cargo test -p wh-iron-std --features cuda --test
cuda_kernel_corpus`) is now `PASS=4304 KNOWN_HARD=0 MISMATCH=0
UNSUPPORTED=10 ERROR=0` (`UNSUPPORTED` = the 10 `iron_gdn_wy_plan` entries
from §7.1, still blocked by the smem cap, not a correctness issue). See
below for all three.

### 7.1 `iron_gdn_wy_plan` CUDA dynamic-smem — root-caused, partially fixed, still blocked

**Root cause found:** `TargetProfile::cuda()` (`wh-iron-codegen/src/backend.rs`)
declared `mma: MmaStrategy::Wmma16x16x16` — but no real `wmma`-fragment
codegen exists (`wh-iron-codegen/src/cuda/mod.rs`'s `CoopTileZero`/
`CoopTileRun`/`CoopTileStoreC` emitters never branch on `Wmma16x16x16`, only
on `SoftwareLocalC`, else falling into the same expensive "full per-warp
shared A+B+C, f32-hardcoded" path `MmaStrategy::Software` describes as
CUDA's actual baseline). Declaring `Wmma16x16x16` was a dead/aspirational
value: it silently ran **every** CUDA `CoopTile` kernel (46 call sites
across `moe`/`gemm`/`sdpa`/`ssm`) through the most expensive fallback with
none of the smem savings HIP/Vulkan already get from `SoftwareLocalC`
(per-warp C accumulator moved to lane-local registers instead of shared
memory — a proven optimization, "the difference that makes MPP `bm64`
kernels fit on RDNA4's 64KB cap" per that strategy's own doc comment).

**Exact byte accounting** (confirmed via the new CPU-only
`cuda_smem_budget` tests in both `gated_delta_wy_plan.rs` and
`gated_delta_wy_scan.rs` — no GPU needed, pure codegen math, and the
numbers match the live CUDA dispatch error message exactly):
- `iron_gdn_wy_plan`'s own `threadgroup_alloc` buffers (`tg_big_g`,
  `tg_beta`, `tg_solve`, `tg_cache`, `Xs`, `OutScratch`) sum to **31232 B**
  — matches the module doc's Metal budget (30.5 KiB) exactly. This part of
  the CUDA codegen was already honest.
- The two `coop_tile_setup` groups (`gA`: 16×32×32, `gB`: 16×32×16) at
  TG=`[512,1,1]` (16 warps) each allocate per-warp shared `_CTA_`/`_CTB_`/
  `_CTC_` buffers, **hardcoded to `"float"` (4 B/elem) regardless of the
  tile's declared `act_dtype`** (both are `f16`, i.e. 2 B/elem — a second,
  separate sizing-formula bug, NOT fixed this pass, see below) — this
  contributes **212992 B**.
- Total: 31232 + 212992 = **244224 B**, exactly the error message's number.

**Fix applied:** `TargetProfile::cuda()`'s `mma` now reads
`MmaStrategy::SoftwareLocalC` (matching HIP/Vulkan) instead of the dead
`Wmma16x16x16` value — a one-line, low-risk change since it activates
already-shipped, already-tested codegen (same code path HIP's `local_c &&
simd` branches already exercise), not new logic. This removes the two
`_CTC_` buffers (2 groups × 16 warps × 512 elems × 4 B = 65536 B):
**244224 → 178688 B** (174.5 KiB), confirmed via both the CPU-only smem
budget test and the live CUDA dispatch error (`kernel needs 178688 bytes`).
Validated **zero regressions**: the full `cuda_kernel_corpus` (thousands of
kernel×dtype entries across all 46 `coop_tile_setup` call sites) shows the
same PASS/FAIL set before and after (one unrelated pre-existing failure,
`test_moe_densify_int2_experts [f16]`, confirmed by inspection to not use
`CoopTile` at all — structurally impossible for this change to affect it;
flagged separately, see §7.3).

**Still blocked — this is the "fundamental" case.** 178688 B is still ~1.75×
GB10's real ~99 KiB (101376 B) dynamic-smem opt-in cap (`docs/specs/
CUDA_BACKEND_SCOPE.md` §5). The `act_dtype`-honest-sizing fix (using each
tile's declared 2-byte dtype instead of hardcoded 4-byte `float` for the
`_CTA_`/`_CTB_` operand buffers) would roughly halve the remaining
212992−65536=147456 B of A/B storage to ~73728 B, landing around **104960
B (102.5 KiB)** — *still* over the cap, by a much smaller margin. That fix
was **not applied this pass**: it touches `coop_cfg`'s tuple, `shared_arrays`,
and all of `CoopTileZero`/`LoadA`/`LoadB`/`Run`/`StoreC`'s emit code, which
is shared by all 46 `coop_tile_setup` call sites (many already CUDA-GREEN
in production use) — validating it safely needs a full-corpus CUDA
re-run (~7.5 min) per iteration, and even then does not by itself clear
the cap for this specific kernel. The remaining gap is architectural: each
of the 16 warps stores a **full private copy** of the A and B operand
tiles in shared memory (`coop_base(simd, tile) = "simd_group * tile"`),
even where a warp's B operand happens to be identical to another's
(kernel-data-dependent, not something the codegen can assume in general).
Clearing this needs either a genuine register/warp-shuffle-based
cooperative GEMM lowering (no shared memory for A/B at all — a new
`MmaStrategy`, comparable in scope to real WMMA support) or a smaller TG
(algorithm-level change, out of codegen scope). **Documenting as a
permanent limitation of the current CoopTile software-emulation strategy
on CUDA**, not a simple formula bug, per the follow-up's own framing.
Tracked next steps, in priority order:
1. `act_dtype`-honest A/B sizing (real bug, safe/mechanical, worth doing
   for memory-traffic reasons alone even though it doesn't unblock this
   kernel by itself — needs the full-corpus validation run before landing).
2. A register/shuffle-based warp GEMM `MmaStrategy` (the actual unblock,
   substantial new codegen work).
3. Re-check `sdpa_prefill_mma` (`docs/specs/CUDA_BACKEND_SCOPE.md`'s
   original example of this same class of problem) against the
   `SoftwareLocalC` fix already landed — same call pattern, likely gets the
   same partial (not full) improvement; not re-measured this pass.

### 7.2 `iron_gdn_wy_scan_single_tile` MISMATCH — FIXED (divergent `__syncthreads()`)

**Root cause found and fixed** in a follow-up pass on top of the §7.2
isolation work below (kept for the record — the isolation was accurate,
just didn't reach the final mechanism): `gated_delta_wy_scan.rs`'s GEMM1
and GEMM2 loops wrapped their `coop_tile_load_a`/`coop_tile_load_b`/
`coop_tile_run` calls in a per-simdgroup runtime `if` — GEMM1's `if sg <
c_m_tiles { ... }` (skip SGs whose 16-row M-tile falls entirely outside
`[0,C)`) and GEMM2's `if n_tile_valid { ... }` (skip `nj`/`sg_n_local`
slots whose N-tile falls entirely outside `[0,Dk)`) — as a "don't do
wasted MMA work" optimization. But `coop_tile_load_a` and `coop_tile_run`'s
**CUDA codegen embeds its own `__syncthreads()` internally**
(`wh-iron-codegen/src/cuda/mod.rs`'s `Op::CoopTileLoadA`/`Op::CoopTileRun`
emitters), and `sg`/`sg_n_local` are **warp-uniform but block-divergent**
(this device's warp size equals the simdgroup width, so every lane in a
given warp takes the same branch, but *different warps in the same
threadblock* take different branches for the small-`Dk`/small-`C` test
fixtures — production `C=64`/`Dk=128` never hits this, since `c_m_tiles`/
`n_tiles_total` cover all 4 SGs there, which is why the production-shape
validation in §3–§6 never surfaced it). Some warps in the block therefore
executed the internal `__syncthreads()` calls and some didn't — a
**divergent-barrier hazard**, invisible to `compute-sanitizer --tool
racecheck`/`--tool initcheck` (both reported clean, as §7.2's original
isolation below found) but definitively confirmed via **`compute-sanitizer
--tool synccheck`** (not run in the original isolation pass), which reports
`Barrier error detected. Divergent thread(s) in block.` at
`iron_gdn_wy_scan`'s exact PC, for the exact single-Dv-tile fixtures that
mismatch.

**Fix:** run `coop_tile_load_a`/`coop_tile_load_b`/`coop_tile_run`
**unconditionally** for every SG/`nj` slot in both GEMM1 and GEMM2, deleting
the `if sg < c_m_tiles`/`if n_tile_valid` guards (and the now-dead
`c_m_tiles`/`n_tile`/`n_tile_valid` locals). This is safe with zero
behavior change for in-range SGs, because the staging loops (`Xs`/`Ws`)
were *already* zero-padding out-of-range rows/columns via `select(valid,
…, 0.0)` before this fix — an "invalid" SG now just MMAs zeros (wasted
work, never wrong), and its output was already masked out of the
consumption step (`t < c`, `d_valid`) by both the old and new code. See
`gated_delta_wy_scan.rs`'s module doc (GEMM1/GEMM2 sections) and inline
comments at both call sites for the in-source writeup.

**Verified:** `gdn_family_scoped_cuda_corpus` now `PASS=61 KNOWN_HARD=0
UNSUPPORTED=9 FAIL=0` (was `PASS=51 KNOWN_HARD=7 FAIL=0` — all 7 formerly
`KNOWN_HARD` `wy_scan` entries, `single_tile` × 3 dtypes +
`debug_{nc1,c32,dk64,zero_state}`, now pass outright; `UNSUPPORTED=9` is
unrelated, the §7.1 `wy_plan` smem-cap entries). Re-ran `compute-sanitizer
--tool synccheck` against the fixed corpus: **0 errors** (was multiple
`Barrier error` reports + a cascading `cuModuleLoadData`/`cuStreamSynchronize`
context-corruption failure on every subsequent kernel in the same process —
that cascade is *why* `multi_tile` looked "fine" pre-fix under bare
(non-sanitizer) execution but wasn't exercising the same divergent path:
production-shape fixtures simply never hit `sg < c_m_tiles`/`n_tile_valid`
being `false` for any SG). Debug fixtures kept permanently as regression
coverage for this exact warp-divergence shape (comments updated to match).
The original isolation notes (accurate, kept for provenance):

- The generated CUDA text is byte-identical between `single_tile` and
  `multi_tile` (`dk`/`dv`/`hv`/`c` are `#[constexpr]` runtime launch args,
  not codegen-time literals) — confirming this was always a runtime
  control-flow bug, not a codegen-selection difference.
- `test_iron_gdn_wy_scan_debug_zero_state` (`state_in` forced to all-zero)
  showed `y` bit-exact (GEMM1 correct) but `state_out` off by whole output
  rows silently reading back as exactly `0.0` — correctly isolated to
  GEMM2, though the actual defect turned out to be barrier corruption
  upstream of the "consume/flush" bookkeeping that was suspected, not that
  bookkeeping itself.
- `racecheck`/`initcheck` clean + 100%-deterministic across reruns:
  correctly read as "not a race/uninit read" — the missing step was trying
  `synccheck`, which exists specifically for this hazard class.

### 7.3 `test_moe_densify_int2_experts [f16]` — FIXED (wrong CUDA thread-count bounds guard)

**Unrelated to §7.1/§7.2** — confirmed pre-existing (not caused by the
`SoftwareLocalC` change; `moe_densify_remap.rs` uses no `CoopTile`/
`SimdgroupMatMul` ops at all) when first found, and root-caused + fixed in
the same follow-up pass as §7.2.

**Root cause:** `iron_moe_densify_int2_experts` densifies MoE expert
weights with a **fixed 256 threads per active expert**
(`grid_1d(n_active * 256, 256)`), each thread internally looping over
`total_packs`/`total_sb` elements — a threading pattern where the launched
thread count has no relationship to any output buffer's element count
(deliberately over-provisioned so `256` threads cover an arbitrarily wide
per-expert row via the internal loop). This kernel defaults to
`KernelMode::Elementwise` (no explicit `.mode(...)` was set). CUDA's
Elementwise dispatch (`wh-iron-runtime/src/device/cuda/mod.rs`) adds an
**implicit bounds guard** `if (_gtid >= _n_elems) return;`, where `_n_elems`
is computed host-side as *"the element count of the first output
parameter"* — a correct assumption for ordinary 1-thread-per-element
kernels (needed to guard the padding when `grid_x = ceil(n/tpg)` rounds up
past `n`), but wrong here: the test fixture's first output, `weight_dst`,
has only 8 elements, while the kernel launches `n_active * 256 = 512`
threads. Every thread with `_gtid >= 8` returned immediately — silently
dropping **every active expert past the first** (whose real per-thread
work starts at `_gtid = 256`), which is exactly the `max|Δ|=22.0`
first-active-expert-only-correct, second-active-expert-all-zero pattern a
throwaway diagnostic probe (dispatch + dump `weight_dst`/`scales_dst`/
`biases_dst` directly, bypassing the corpus harness's single scalar
diff) confirmed: `active_experts=[1,3]` (2 active experts) — expert `1`'s
slab came back bit-exact, expert `3`'s slab came back all-`0.0`. `Metal`
(and by extension the rest of the corpus, since this is CUDA-only) has no
equivalent implicit guard — Apple's `dispatchThreads` launches the exact
requested thread count with no host-computed clamp — hence this was never
visible outside the CUDA backend.

**Fix:** set `.mode(KernelMode::Grid3D)` on both `TestSetup` (test) and
`BenchSetup` (bench) for this kernel — `Grid3D` has no implicit bounds
guard on CUDA (parity with Metal's exact-count dispatch), which is correct
here because the kernel already carries its own explicit, correct bounds
checks (`if a < n_active`, `if p < total_packs`/`total_sb`) that don't need
the Elementwise-mode heuristic. Confirmed via the same diagnostic probe:
`weight_dst`/`scales_dst`/`biases_dst` all came back bit-exact for both
active experts after the fix (`max|Δ|=0.0`, matching the fixture's
`tol=[0.0]`). Not exposed at the bench's production-ish shape (`n_out *
packs_per_row = 256*128 = 32768 >> 256`, so `_n_elems` was already larger
than the launched thread count there) — a latent correctness bug rather
than an active one at that scale, but fixed regardless of shape now. This
class of bug (Elementwise mode's `_n_elems` guard assuming "1 thread ≤ 1
output element", violated by any kernel using `program_id` with internal
per-thread multi-element loops and a deliberately over-provisioned thread
count) is worth a broader audit of other `program_id`-based kernels if any
show the same pattern — not done this pass (scope was this one flagged
regression).

---

## 8. Files touched this pass

- `crates/wh-iron-std/src/kernels/ssm/gated_delta_prep_chunk.rs` — added the
  `d128_128_48_16` variant to `iron_gated_delta_prep_chunk_fast`'s
  `#[kernel(variants(...))]` list, updated its module doc, added 2 new
  `#[test_kernel]` fixtures (generic + fast variant) at the real Hv=48 shape.
- `crates/wh-iron-std/tests/gdn_qwen36_27b_shape_cuda.rs` — new. Host
  reference (`host_decode_step`/`host_chunk_oracle`) + the 4 CUDA checks in
  §6.
- `crates/wh-iron-std/tests/gdn_family_cuda_corpus_scoped.rs` — new. Fast
  GDN-only CUDA corpus filter, used to produce §6/§7's numbers.
- `GDN_PREFILL_CONTRACT.md` (this file).

### Files touched in the §7 CUDA-codegen follow-up pass

- `crates/wh-iron-codegen/src/backend.rs` — `TargetProfile::cuda()`'s `mma`
  field: `MmaStrategy::Wmma16x16x16` → `MmaStrategy::SoftwareLocalC` (§7.1
  fix), with a doc comment explaining why and the exact byte accounting.
  Updated `cuda_profile_uses_cuda_idioms`'s assertion to match.
- `crates/wh-iron-std/src/kernels/ssm/gated_delta_wy_plan.rs` — added a
  CPU-only `cuda_smem_budget` test (prints `CudaGenerator::shared_bytes`
  for the 512-thread TG, no GPU needed) as a permanent regression probe for
  §7.1's byte count.
- `crates/wh-iron-std/src/kernels/ssm/gated_delta_wy_scan.rs` — added the
  same `cuda_smem_budget` test (128-thread TG); added `setup_ex` (a
  `setup` variant with a `zero_state` knob) and 4 new debug `#[test_kernel]`
  fixtures (`debug_nc1`, `debug_c32`, `debug_dk64`, `debug_zero_state`) used
  to isolate §7.2 — kept as permanent regression/isolation coverage,
  `KNOWN_HARD`-listed alongside `single_tile` (not yet fixed).
- `crates/wh-iron-std/tests/gdn_family_cuda_corpus_scoped.rs` — `KNOWN_HARD`
  extended with the 4 new debug fixture names + updated commentary pointing
  at §7.2's findings.
- `crates/wh-iron-std/tests/cuda_kernel_corpus.rs` — `KNOWN_HARD` extended
  with `iron_gdn_wy_scan_single_tile` (all 3 dtypes — this full-corpus file
  didn't have it listed before, so this file was already red on this
  fixture pre-pass) + the 4 new debug fixtures. `test_moe_densify_int2_experts
  [f16]` (§7.3) deliberately left un-listed (out of scope, unresearched).
- `GDN_PREFILL_CONTRACT.md` (this file) — this §7 rewrite.

A throwaway `crates/wh-iron-std/tests/debug_wy_scan_perbuf.rs` (per-buffer/
per-row diff harness + a CUDA-text dump test used to investigate §7.2) was
used during this pass and removed before finishing — its findings are
folded into §7.2 above.

### Files touched in the §7.2/§7.3 root-cause-and-fix pass

- `crates/wh-iron-std/src/kernels/ssm/gated_delta_wy_scan.rs` — the actual
  §7.2 fix: removed the `if sg < c_m_tiles`/`if n_tile_valid` guards around
  `coop_tile_load_a`/`coop_tile_load_b`/`coop_tile_run` in both GEMM1 and
  GEMM2 (now unconditional every iteration), deleted the now-dead
  `c_m_tiles`/`n_tile`/`n_tile_valid` locals, and rewrote the module doc's
  GEMM1/GEMM2 sections + added inline comments at both call sites
  explaining the divergent-`__syncthreads()` hazard and why the fix is
  safe. Debug-fixture doc comments updated from "isolation, remove once
  fixed" to "regression coverage, kept permanently."
- `crates/wh-iron-std/src/kernels/moe/moe_densify_remap.rs` — the §7.3 fix:
  added `.mode(KernelMode::Grid3D)` to `test_moe_densify_int2_experts`'s
  `TestSetup` and `bench_moe_densify_int2_experts`'s `BenchSetup`, with
  comments explaining the Elementwise-mode `_n_elems` bounds-guard
  mismatch.
- `crates/wh-iron-std/tests/gdn_family_cuda_corpus_scoped.rs` — `KNOWN_HARD`
  emptied (was the 7 `wy_scan` entries, all now pass); doc comment
  rewritten to record the fix + point at §7.2.
- `crates/wh-iron-std/tests/cuda_kernel_corpus.rs` — `KNOWN_HARD` entries
  for `iron_gdn_wy_scan_*` removed (all now pass); `test_moe_densify_
  int2_experts [f16]`'s absence from the list is now correct (it passes)
  rather than an intentional gap.
- `GDN_PREFILL_CONTRACT.md` (this file) — §7 intro + §7.2/§7.3 rewritten to
  FIXED status with the root-cause writeups above.

Diagnostic tooling used this pass (both removed before finishing, findings
folded into §7.2/§7.3 above): a `compute-sanitizer --tool synccheck` run
against the `gdn_family_cuda_corpus_scoped` test binary (the tool that
actually cracked §7.2 — not run in the prior pass); a throwaway
`crates/wh-iron-std/tests/moe_densify_probe.rs` (direct per-buffer dispatch
+ dump, bypassing the corpus harness's single scalar diff, used to
pinpoint §7.3 to "second active expert's slab reads back all-zero").
Verification commands, for reproducing the PASS numbers above:
```
cargo test -p wh-iron-std --features cuda --test gdn_family_cuda_corpus_scoped -- --nocapture
cargo test -p wh-iron-std --features cuda --test cuda_kernel_corpus -- --nocapture
compute-sanitizer --tool synccheck <gdn_family_cuda_corpus_scoped binary> --exact gdn_family_scoped_cuda_corpus --nocapture
```

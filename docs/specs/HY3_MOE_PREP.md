# Hy3 (`hy_v3`) MoE prep for wh-iron

Target model: [tencent/Hy3](https://huggingface.co/tencent/Hy3) (295B MoE, 21B active).
Practical local checkpoint for Apple Silicon: `mlx-community/Hy3-oQ2` (~99 GB, MLX affine 2-bit, `group_size=64`).

This doc tracks **kernel-layer** readiness. Iron model-family wiring (`HYV3ForCausalLM`) is out of scope here.

## Config snapshot (from `tencent/Hy3` `config.json`)

| Field | Value |
|-------|-------|
| `model_type` / arch | `hy_v3` / `HYV3ForCausalLM` |
| layers | 80 (+ 1 MTP: `num_nextn_predict_layers=1`) |
| hidden / heads | 4096 / GQA 64q 8kv, `head_dim=128` |
| MoE | 192 experts, top-8, `moe_intermediate_size=1536` |
| shared experts | 1 (`num_shared_experts`) |
| first dense layers | `first_k_dense_replace=1` |
| router | `moe_router_use_sigmoid=true`, `moe_router_enable_expert_bias=true` |
| | `route_norm=true`, `router_scaling_factor=2.826` |
| norms / act | `qk_norm=true`, RMSNorm eps 1e-5, SwiGLU (`silu`) |
| rope | θ ≈ 1.115884e7, type default |
| context | 256K |

## Kernel map

| Need | Kernel(s) | Status |
|------|-----------|--------|
| Sigmoid + bias scores (single tensor) | `iron_moe_router_sigmoid_bias` | ready; Hy3 shape test |
| Sigmoid unbiased + biased pair | `iron_moe_sigmoid_bias` | ready; Hy3 shape test |
| Top-k biased / weight unbiased | `iron_moe_router_topk_biased` | ready; Hy3 192/8 test + bench |
| Softmax top-k (not Hy3 scorer) | `iron_moe_router_topk` | width bench only |
| Expert gather / permute / sort | `iron_moe_gather_*`, `iron_moe_permute`, `iron_moe_sort_plan` | ready |
| Expert-outer int2/4/8 gather (MPP) | `iron_moe_gather_qmm_mma_eg_int{2,4,8}_expert_grid_mpp` | ready; Path B prefill |
| int2 BM16 MMA gather | `iron_moe_gather_qmm_mma_int2_bm16` | ready |
| int2 BM8/BM64 MPP gather | `iron_moe_gather_qmm_mma_int2_bm{8,64}_mpp` | ready (opt-in tiles) |
| int2 expert-indexed matvec | `iron_dequant_gemv_int2_expert_indexed` | ready (decode) |
| SwiGLU / down combine | `iron_swiglu`, `iron_moe_down_swiglu_accum_*` | ready |
| Shared-expert helpers | `iron_sigmoid_scalar_fma*` | ready |
| QK-norm | `iron_rms_norm` / `iron_rms_norm_small` | ready (`head_dim=128`) |

## Decisions (locked)

1. **Router equation = Path B** (reference C++/Python ports of `hy_v3`):
   ```
   unbiased = sigmoid(logits)
   biased   = unbiased + expert_bias     # selection only
   top-k by biased
   weights  = renorm(unbiased[chosen]) if route_norm
   weights *= router_scaling_factor
   ```
   Iron: `MoEGatingMode.sigmoidBiasedTopK` + `MoERouter.routedScalingFactor`.
   Kernels: `iron_moe_sigmoid_bias` + `iron_moe_router_topk_biased` (GPU path);
   CPU oracle matches NemotronH / DeepSeek-V3 Path B.

2. **Shared expert** — always-on ungated SwiGLU; `y = routed + shared` (not router-gated).

3. **QK-norm before RoPE** — norm then rotary (same policy as other GQA families with qk_norm).

## Still open

1. **Prefill throughput** — expert-sorted EG gather is the Iron default at large
   `mTotal`; residual multi-× work is BW-bound MoE GU/down, not missing router kernels.

2. **MTP** — `num_nextn_predict_layers=1`;
   `enorm(embed)+hnorm(h)→concat→eh_proj→decoder→final_layernorm→lm_head`.
   Not wired in Iron yet (optional follow-up).

## Local checkpoint

```bash
# Example (adjust path / free disk):
huggingface-cli download mlx-community/Hy3-oQ2 --local-dir ~/models/Hy3-oQ2
```

## Out of scope for kernels

- Chat template / tokenizer / vocab 120832
- MTP scheduling (reuse same decoder blocks)
- RoPE table generation (host)
- Full model registry in Iron

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Mixture-of-experts kernels — the moe family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). The routing layer (top-k expert
//! selection + permute/unpermute + router pre-scores), the per-expert quantized
//! matmul cells in their various forms (MPP grouped BGEMM, GGUF q2k/iq2xxs
//! BGEMM/GEMV, batched Q4 gather), and the down-projection combine. Migrated
//! from the legacy `mlx/` + `ffai/` split.
//!
//! Filenames keep the `moe_` prefix so they match their kernel names
//! (`ffai_moe_*`); the two files whose kernels are format-prefixed instead
//! (`block_scaled_moe`, `dequant_gemv_expert_indexed*`) match those. The
//! per-format `*_block_scaled` matrices keep the format axis as-is; the
//! format-axis fold (plan §7) is deferred.

// Routing — top-k expert selection, permute/unpermute, router pre-scores.
pub mod moe_gather_qmm;
pub mod moe_permute;
pub mod moe_router_sigmoid_bias;
pub mod moe_router_sqrtsoftplus;
pub mod moe_router_topk;
pub mod moe_router_topk_biased;
pub mod moe_sigmoid_bias;

// MPP grouped BGEMM (one ABI, tile-geometry / bit-width variants; shared
// test/bench helpers in `moe_mpp_shared`).
pub mod moe_mpp;
pub mod moe_mpp_block_scaled;
pub mod moe_mpp_bm64;
pub mod moe_mpp_bm64_block_scaled;
pub mod moe_mpp_bm8;
pub mod moe_mpp_bm8_block_scaled;
pub mod moe_mpp_shared;

// GGUF-format per-expert matmul / matvec (q2k, iq2xxs, q4).
pub mod moe_bgemm_iq2xxs_bm64;
pub mod moe_bgemm_iq2xxs_mpp;
pub mod moe_bgemm_iq2xxs_view;
pub mod moe_bgemm_iq2xxs_view_u16_bm64;
pub mod moe_bgemm_q2k_bm64;
pub mod moe_bgemm_q2k_mpp;
pub mod moe_bgemm_q2k_view;
pub mod moe_bgemm_q2k_view_u16_bm64;
pub mod moe_bgemm_q4_bm64;
pub mod moe_gather_down_q2k;
pub mod moe_gather_gemv_iq2xxs;
pub mod moe_gemv_rows_iq2xxs;
pub mod moe_gemv_rows_q2k;
pub mod moe_gemv_rows_view_iq2xxs;
pub mod moe_gemv_ws_iq2xxs;
pub mod moe_gemv_ws_q2k;

// Batched Q4 expert gather (up / down / weighted-sum), seeded from gemv_q8.
pub mod moe_gather_q4;

// Down-projection combine (swiglu-fused accumulate, weighted sum).
pub mod moe_down_swiglu_accum;
pub mod moe_down_weighted_sum_f16;

// Expert-indexed dequant GEMV + block-scaled MoE matmul.
pub mod block_scaled_moe;
pub mod dequant_gemv_expert_indexed;
pub mod dequant_gemv_expert_indexed_block_scaled;

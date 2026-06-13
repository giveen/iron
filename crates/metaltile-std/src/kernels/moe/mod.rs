//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Mixture-of-experts kernels — the moe family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). The routing layer (top-k expert
//! selection + permute/unpermute + router pre-scores), the per-expert quantized
//! matmul cells in their various forms (MPP grouped BGEMM, GGUF q2k/iq2xxs
//! BGEMM/GEMV, batched Q4 gather), and the down-projection combine. Migrated
//! from the legacy `mlx/` + `ffai/` split.
//!
//! Filenames drop the redundant `moe_` prefix (the folder provides it); kernel
//! names keep `mt_moe_*`. The per-format `*_block_scaled` matrices move as-is;
//! the format-axis fold (plan §7) is deferred. `orchestration.rs` is large and
//! is slated for a follow-up split (router_topk / permute / gather_qmm).

// Routing — top-k expert selection, permute/unpermute, router pre-scores.
pub mod orchestration;
pub mod router_topk_biased;
pub mod router_sigmoid_bias;
pub mod router_sqrtsoftplus;
pub mod sigmoid_bias;

// MPP grouped BGEMM (one ABI, tile-geometry / bit-width variants; shared
// test/bench helpers in `mpp_shared`).
pub mod mpp;
pub mod mpp_shared;
pub mod mpp_int8;
pub mod mpp_bm8;
pub mod mpp_bm8_int8;
pub mod mpp_bm64;
pub mod mpp_bm64_int8;
pub mod mpp_block_scaled;
pub mod mpp_bm8_block_scaled;
pub mod mpp_bm64_block_scaled;

// GGUF-format per-expert matmul / matvec (q2k, iq2xxs, q4).
pub mod bgemm_q2k_bm64;
pub mod bgemm_q2k_mpp;
pub mod bgemm_q2k_view;
pub mod bgemm_q2k_view_u16_bm64;
pub mod bgemm_q4_bm64;
pub mod bgemm_iq2xxs_bm64;
pub mod bgemm_iq2xxs_mpp;
pub mod bgemm_iq2xxs_view;
pub mod bgemm_iq2xxs_view_u16_bm64;
pub mod gather_down_q2k;
pub mod gather_gemv_iq2xxs;
pub mod gemv_rows_q2k;
pub mod gemv_rows_iq2xxs;
pub mod gemv_rows_view_iq2xxs;
pub mod gemv_ws_q2k;
pub mod gemv_ws_iq2xxs;

// Batched Q4 expert gather (up / down / weighted-sum), seeded from gemv_q8.
pub mod gather_q4;

// Down-projection combine (swiglu-fused accumulate, weighted sum).
pub mod down_swiglu_accum;
pub mod down_weighted_sum_f16;

// Expert-indexed dequant GEMV + block-scaled MoE matmul.
pub mod dequant_gemv_expert_indexed;
pub mod dequant_gemv_expert_indexed_block_scaled;
pub mod block_scaled;

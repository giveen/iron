//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Matrix-multiply kernels — the gemm family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): dense GEMM / GEMV (+ masked,
//! +axpy), the batched QKV / 4-way projection forms, patch-embed (im2col +
//! matmul), the MLX `steel/` tiled-GEMM templates, and the quantized matmuls
//! (integer `quantized_*`, floating-point `fp_quantized_*`, and the
//! microscaling `block_scaled_*` formats), migrated from the legacy `mlx/` +
//! `ffai/` split.
//!
//! The quantized files keep their format-distinguished names for now; the
//! format-axis fold (plan §7) — collapsing the per-format `mt_<fmt>_qmm`
//! matrices into format-parameterised kernels — is deferred.

// Dense GEMM / GEMV.
pub mod dense;
pub mod gemv;
pub mod gemv_axpy_inplace;
pub mod gemv_masked;
pub mod patch_embed;
pub mod patch_embed_mma;
pub mod steel;

// Integer-quantized matmul / matvec (MLX int2..int8 weights).
pub mod dequant_gemv;
pub mod quantized;
pub mod quantized_mma_dynamic_m;
pub mod quantized_mpp;
pub mod quantized_mpp_int8;
pub mod quantized_nax;
pub mod quantized_nax_int8;

// Floating-point-quantized matmul (fp8 weights).
pub mod fp_quantized;
pub mod fp_quantized_mma;
pub mod fp_quantized_nax;

// Block-scaled / microscaling matmul (mxfp4/nvfp4/mxfp8/… formats).
pub mod block_scaled_matmul;
pub mod block_scaled_mma;
pub mod block_scaled_qmm;
pub mod block_scaled_qmm_mpp;
pub mod block_scaled_qmm_nax;

// Batched projection GEMV / GEMM (QKV 3-way + 4-way fused outputs).
pub mod batched_4_block_scaled_qgemv;
pub mod batched_4_block_scaled_qmm;
pub mod batched_4_qgemv;
pub mod batched_4_qmm;
pub mod batched_qkv_block_scaled_qgemv;
pub mod batched_qkv_block_scaled_qmm;
pub mod batched_qkv_qgemv;
pub mod batched_qkv_qmm;

// Quantized GEMM (Q8_0 / Q4) — scalar + MMA/coop-tile forms.
pub mod gemm_q4_mpp;
pub mod gemm_q8;
pub mod gemm_q8_mpp;

// Quantized inline-dequant GEMV (Q8_0 / Q4, coalesced + grouped/tiled forms).
pub mod gemv_quantized;

// Quantized patch-embed (block-scaled im2col + matmul).
pub mod patch_embed_block_scaled;
pub mod patch_embed_mma_block_scaled;

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Matrix-multiply kernels — the gemm family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): dense GEMM / GEMV (+ masked,
//! +axpy), the batched QKV / 4-way projection forms, patch-embed (im2col +
//! matmul), and the MLX `steel/` tiled-GEMM templates.
//!
//! This is the **dense** half of the family, migrated from the legacy `mlx/` +
//! `ffai/` split. The quantized matmuls (`*_q8`/`q4`/`block_scaled_qmm`/
//! `quantized_*`/`fp_quantized_*` and the batched `*_qgemv`/`*_qmm` forms) are
//! the quantized form of these ops and land here in a follow-up pass; the
//! format-axis fold (plan §7) is deferred.

pub mod dense;
pub mod gemv;
pub mod gemv_axpy_inplace;
pub mod gemv_masked;
pub mod patch_embed;
pub mod patch_embed_mma;
pub mod steel;

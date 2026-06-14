//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! FFAI / model-specific kernels.
//!
//! Kernels here are ports from FFAI / mlx-swift-lm / ekryski's `mlx` fork
//! that don't have a matching template in mainline MLX at the pinned
//! commit (see `metaltile-std/build.rs` — `MLX_COMMIT`). They register
//! a `BenchSpec` so `tile build` / `tile inspect` can find them, but
//! the spec uses `shapes: &[]` and `dispatch: BenchDispatch::Generic`,
//! so `tile bench` skips them (no MLX side-by-side, no GPU shapes).
//!
//! Correctness for these kernels is validated end-to-end in FFAI's
//! integration tests against real models. Once a kernel has been
//! verified there, the shape spec / bench dispatch can be added back
//! here so `tile bench` can track it for regressions — and if its MLX
//! counterpart lands in mainline at a future pin, the file moves to
//! `mlx/`.

pub mod aura_dequant_rotated;
pub mod aura_encode;
// batched_* projection GEMV/GEMM + dequant_gemv migrated to kernels/gemm/.
pub mod dequant_gather;
pub mod dequant_gather_block_scaled;
pub mod dsv4_fp8_block_dequant;
pub mod dsv4_mxfp4_dequant;
pub mod ffai_dequant_q4;
// gemm_q4_mpp / gemm_q8 / gemm_q8_mpp + gemv_q8 (split by family) → kernels/.
pub mod gguf_dequant_iq2_xxs;
pub mod gguf_dequant_iq2_xxs_raw;
pub mod gguf_dequant_q2_k;
pub mod gguf_dequant_q8_0;
pub mod gguf_iq2_xxs_extract_qs;
// moe family (moe* · dsv4_router_topk · dequant_gemv_expert_indexed*) →
// kernels/moe/. patch_embed_block_scaled / patch_embed_mma_block_scaled →
// kernels/gemm/.

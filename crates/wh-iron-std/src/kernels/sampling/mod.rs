//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Sampling kernels — the sampling family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): the logits→token pipeline.
//! Softmax and sort (from the legacy `mlx/`), the fused categorical sampler,
//! and the logits processors / masks (top-k, top-p, min-p, temperature,
//! repetition penalty) from the legacy `iron/`.

pub mod categorical_sample;
pub mod gpu_pipeline;
pub mod logits_min_p;
pub mod logits_processors;
pub mod logits_top_p;
pub mod logits_topk;
pub mod logits_topk_tiled;
pub mod softmax;
pub mod sort;

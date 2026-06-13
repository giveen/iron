//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Mixture-of-experts kernels — the moe family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). Migrated ahead of the full moe
//! wave: the batched Q4 expert-gather projections (up / down / weighted-sum)
//! and the on-device router pre-score. The remaining moe_* kernels still live
//! in `ffai/` and land here when the moe family is consolidated.

pub mod gather_q4;
pub mod sigmoid_bias;

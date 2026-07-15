//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Vision front-end kernels — the vision family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): image resize+normalize (+bicubic),
//! im2col patch extraction (+ interleaved), patch unfold (non-square grids),
//! 2-D positional-embedding add, NHWC avg-pool, token-major transpose, and luma
//! frame differencing. Migrated from the legacy `ffai/`.

pub mod avg_pool2d_nhwc;
pub mod broadcast_affine;
pub mod frame_diff_luma;
pub mod im2col_patch;
pub mod im2col_patch_interleaved;
pub mod patch_unfold;
pub mod patch_unfold_qwen;
pub mod pos_emb_2d_add;
pub mod resize_normalize;
pub mod transpose_th;

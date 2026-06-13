//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! KV-cache kernels — the kv_cache family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): the cache-update path
//! (single + batched), KV quantization / dequantization (incl. fp8), and the
//! FFT used by the STT front-end (`mt_fft` + Bluestein non-pow2 stages).
//! Migrated from the legacy `mlx/` (`fft`) + `ffai/` split.

pub mod cache;
pub mod fft;
pub mod update_many;

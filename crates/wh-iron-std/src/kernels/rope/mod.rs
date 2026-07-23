//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Rotary position embedding (RoPE) kernels — the rope family, migrated from
//! the legacy `mlx/` + `iron/` split (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`). All kernels carry the `iron_`
//! prefix. The position-batched kernels take a per-row/per-token position plus
//! an outer grid axis, so decode is just the single-row case (no separate
//! decode kernel):
//!   - `base`         — `iron_rope` (MLX `rope.metal` reference parity)
//!   - `rope_banded`  — `iron_rope_banded` (frequency-band scaling; decode = T=1)
//!   - `partial_rope` — `iron_partial_rope` (rotates tail dims only; decode = n_tokens=1)
//!   - `rope_2d`      — `iron_rope_2d` (2D positional / vision M-RoPE)
//!   - `rope_yarn`    — `iron_rope_yarn` (YaRN context extension)

pub mod base;
pub mod partial_rope;
pub mod rope_2d;
pub mod rope_banded;
pub mod rope_yarn;

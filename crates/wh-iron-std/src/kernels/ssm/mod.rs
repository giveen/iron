//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! State-space-model kernels — the ssm family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): the Mamba-2 selective-scan
//! decode step (`iron_ssm_step`, + `_a2d` for 2-D A_log, + `_grouped` for the
//! MLX-aligned reduction form), the SSD chunked-scan ops (`iron_ssd_*`), Mamba
//! input-projection split + gated group RMSNorm, the gated-delta-net family
//! (+ prep / chunk / gate-beta / wy / replay), and record/replay. Migrated
//! from `iron/`.
//! (The depthwise causal conv1d that precedes the scan lives in
//! `convolution/conv1d_causal.rs`.)

pub mod gated_delta;
pub mod gated_delta_gate_beta;
pub mod gated_delta_packed_prework;
pub mod gated_delta_prep;
pub mod gated_delta_prep_chunk;
pub mod gated_delta_qknorm_prepass;
pub mod gated_delta_qwen35_decode_fused;
pub mod gated_delta_replay;
pub mod gated_delta_wy;
pub mod gated_delta_wy_plan;
pub mod gated_delta_wy_scan;
pub mod scan;
pub mod ssm_replay;

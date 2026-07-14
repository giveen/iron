//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Hyper-connection kernels — manifold-constrained hyper-connections (mHC):
//! dynamic residual mixing (collapse / expand) and the Sinkhorn dynamic-mix
//! split. A distinct architectural mechanism, not attention or normalization.

pub mod mhc;
pub mod mhc_sinkhorn_split;

//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Probe kernels — `#[kernel]`s whose purpose is to validate a codegen path or
//! HW intrinsic end-to-end (not production work). Named `mt_<thing>_probe`; this
//! module sits at the crate root, outside the `kernels/<family>/` tree (see the
//! style guide §1). Distinct from `*_smoke` *tests*, which stay test-side.

pub mod mma_layout_probe;
pub mod mpp_matmul_probe;
pub mod simdgroup_load_probe;

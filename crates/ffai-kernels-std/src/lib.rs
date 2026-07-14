//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! FFAI Kernels kernel standard library: kernel definitions and bench helpers.
//!
//! `ffai-kernels-std` provides the kernel definitions (`#[kernel]` / `#[bench]` /
//! `#[test_kernel]`) and the shared bench-setup utilities used by the kernels.

pub mod kernels;
pub mod probe;

// Re-export the kernel inventories from the harness registry. The `#[kernel]` /
// `#[bench]` / `#[test_kernel]` registrations live in this crate's `kernels`
// modules; importing these accessors via `ffai_kernels_std` (rather than
// `ffai-kernels`) pulls the std rlib into a downstream link, which is what
// retains those inventory statics.
pub use ffai_kernels::harness::registry::{all_benches, all_kernels, all_tests};
pub mod utils;

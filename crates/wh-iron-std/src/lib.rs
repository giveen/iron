//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Iron kernel standard library: kernel definitions and bench helpers.
//!
//! `wh-iron-std` provides the kernel definitions (`#[kernel]` / `#[bench]` /
//! `#[test_kernel]`) and the shared bench-setup utilities used by the kernels.

pub mod kernels;
pub mod probe;

// Re-export the kernel inventories from the harness registry. The `#[kernel]` /
// `#[bench]` / `#[test_kernel]` registrations live in this crate's `kernels`
// modules; importing these accessors via `wh_iron_std` (rather than
// `wh-iron`) pulls the std rlib into a downstream link, which is what
// retains those inventory statics.
pub use wh_iron::harness::registry::{all_benches, all_kernels, all_tests};
pub mod utils;

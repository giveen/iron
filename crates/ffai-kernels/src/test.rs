//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Author-facing surface for `#[bench]` / `#[test_kernel]` setups.
//!
//! Glob-import this in a kernel file to bring the builder types and `DType`
//! into scope:
//!
//! ```ignore
//! pub use ffai_kernels::test::*;
//!
//! #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
//! fn test_my_kernel(dt: DType) -> TestSetup { /* ... */ }
//! ```

pub use ffai_kernels_core::{DType, ir::KernelMode};

pub use crate::harness::{
    bench::{BenchBuffer, BenchSetup, ConstValue, Grid, KernelBench, RefKernel},
    test::{KernelTest, TestBuffer, TestSetup},
};

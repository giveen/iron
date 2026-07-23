//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Compile-fail tests for the `#[kernel]` proc-macro.
//!
//! Each fixture under `tests/error/` is a minimal `#[kernel]` body that
//! triggers a specific diagnostic. The paired `.stderr` golden pins the
//! exact compiler error text so the macro's user-facing messages don't
//! silently regress.
//!
//! Refresh after intentional message changes:
//!
//!   TRYBUILD=overwrite cargo test -p wh-iron --test compile_fail

#[test]
fn kernel_macro_diagnostics() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/error/*.rs");
}

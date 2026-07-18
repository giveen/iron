//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! `#[kernel]` rejects a bare identifier that was never declared via a
//! `let`-binding, loop variable, `#[constexpr]` param, tensor param, or a
//! DSL built-in scalar. This mirrors a real bug: writing `sg` without
//! `let sg = simd_group_id();` used to silently resolve to `ValueId::new(0)`
//! (the kernel's first SSA value) and compile into a wrong-but-passing
//! kernel, only caught by a GPU oracle mismatch.

use ffai_kernels::prelude::*;

#[kernel]
fn kernel_with_undeclared_identifier(out: Tensor<f32>) {
    let idx = program_id(0);
    store(out[idx], sg);
}

fn main() {}

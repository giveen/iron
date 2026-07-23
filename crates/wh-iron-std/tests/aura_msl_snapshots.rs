//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Golden MSL snapshots for the AURA kernel family.
//!
//! Sibling to `wh-iron-codegen/tests/msl_snapshots.rs` — that file
//! pins MSL for hand-built IR fixtures; this file pins MSL for the real
//! `#[kernel]`-generated AURA kernels in `wh-iron-std::iron`. The
//! codegen-crate snapshot test can't reach these kernels (it does not
//! depend on `wh-iron-std`), so the AURA emit-path goldens live here
//! where `kernel_ir_for` is in scope.
//!
//! Each kernel is run through the full `MslGenerator` pipeline and the
//! emitted MSL is pinned via `insta::assert_snapshot!`. Any codegen
//! change — op lowering, preamble emission, scheduling, vectorization,
//! fusion — surfaces as a reviewable text diff instead of a silent
//! drift. They also catch the empty-body MSL hazard (a header with no
//! body would show up as a one-line diff) that `make emit-all`'s
//! `xcrun metal` smoke pass does not.
//!
//! One representative dtype / bit-width / dim per kernel — the goal is
//! to pin the emit path, not enumerate every monomorphization. The
//! per-kernel GPU correctness tests under `tests/aura_*_gpu_*.rs` carry
//! the numerical contract.
//!
//! Refresh after an intentional codegen change:
//!   cargo insta test --accept -p wh-iron-std --test aura_msl_snapshots
//! Or interactively:
//!   cargo insta review

use insta::assert_snapshot;
use wh_iron::{
    codegen::{MslGenerator, msl::MslConfig},
    core::{dtype::DType, ir::KernelMode},
};
use wh_iron_std::kernels::{
    quant::{
        aura_dequant_rotated::iron_aura_dequant_rotated_int4,
        aura_encode::iron_aura_encode_int4,
    },
    sdpa::{
        aura_flash_p1::iron_aura_flash_p1_kb4_vb2_d128,
        aura_flash_pass2::iron_aura_flash_pass2_d128,
        aura_score::iron_aura_score_int4,
        aura_value::iron_aura_value_int4,
    },
};

/// Generate MSL for an AURA kernel's IR with its declared kernel mode.
/// `kernel_ir_for` returns the bare IR; the mode (Reduction / Grid3D)
/// is carried in the kernel's `BenchSpec` and must be set on the IR
/// before codegen so `emit_reduce` / Grid3D dispatch lowers correctly.
fn aura_msl(kernel_ir: wh_iron::core::ir::Kernel, mode: KernelMode) -> String {
    let mut kernel = kernel_ir;
    kernel.mode = mode;
    MslGenerator::new(MslConfig::default())
        .generate(&kernel)
        .expect("AURA kernel must codegen cleanly")
}

/// `iron_aura_encode` — fused L2-norm + rotation + Lloyd-Max quantize +
/// bit-pack. Reduction mode (`simd_sum` over the rotated coordinates).
#[test]
fn iron_aura_encode_int4_f32_msl() {
    let msl = aura_msl(iron_aura_encode_int4::kernel_ir_for(DType::F32), KernelMode::Reduction);
    assert_snapshot!(msl);
}

/// `iron_aura_dequant_rotated` — bulk unpack + de-rotate of a packed AURA
/// K/V slab. Grid3D mode (one thread per packed word).
#[test]
fn iron_aura_dequant_rotated_int4_f32_msl() {
    let msl =
        aura_msl(iron_aura_dequant_rotated_int4::kernel_ir_for(DType::F32), KernelMode::Grid3D);
    assert_snapshot!(msl);
}

/// `iron_aura_score` — per-token Q·K score against the packed AURA cache.
/// Reduction mode.
#[test]
fn iron_aura_score_int4_f32_msl() {
    let msl = aura_msl(iron_aura_score_int4::kernel_ir_for(DType::F32), KernelMode::Reduction);
    assert_snapshot!(msl);
}

/// `iron_aura_value` — softmax-weighted accumulation of the packed V cache.
/// Grid3D mode.
#[test]
fn iron_aura_value_int4_f32_msl() {
    let msl = aura_msl(iron_aura_value_int4::kernel_ir_for(DType::F32), KernelMode::Grid3D);
    assert_snapshot!(msl);
}

/// `iron_aura_flash_p1` — flash-attention pass 1 over the packed cache
/// (kb4 / vb2 / d128 recipe). Grid3D mode.
#[test]
fn iron_aura_flash_p1_kb4_vb2_d128_f32_msl() {
    let msl =
        aura_msl(iron_aura_flash_p1_kb4_vb2_d128::kernel_ir_for(DType::F32), KernelMode::Grid3D);
    assert_snapshot!(msl);
}

/// `iron_aura_flash_pass2` — flash-attention pass 2 reduction (d128 recipe).
/// Reduction mode; storage in bf16, online softmax in fp32.
#[test]
fn iron_aura_flash_pass2_d128_bf16_msl() {
    let msl =
        aura_msl(iron_aura_flash_pass2_d128::kernel_ir_for(DType::BF16), KernelMode::Reduction);
    assert_snapshot!(msl);
}

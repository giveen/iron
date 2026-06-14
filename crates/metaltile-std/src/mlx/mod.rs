//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MLX-compared kernels.
//!
//! Every kernel in this submodule has (or can have) a side-by-side
//! correctness/perf comparison against an MLX reference kernel — the
//! benches embed MLX's `.metal` source via `metal_file = "..."` and
//! dispatch the MLX kernel through `compile_with_bool_constants` / a
//! constructed kernel name.
//!
//! When a kernel can't be directly compared today (MLX template not
//! shipped at the pinned commit, or the comparison isn't wired yet)
//! but the implementation faithfully mirrors MLX semantics and is
//! expected to wire up eventually, it lives in `ffai/` until the
//! comparison lands.

// block_scaled_dequant → quant family (migrated later); block_scaled_moe →
// kernels/moe/. The quantized matmuls moved to kernels/gemm/.
pub mod block_scaled_dequant;
pub mod steel;

// `conv.rs` and `shared.rs` are placeholder/stale stubs left over from
// the old `metaltile-bench` crate. They reference `crate::runner` which
// lives in `metaltile-cli`, so they don't compile — kept on disk for
// the kernel docs / future-work notes but intentionally not declared
// here. Delete or port when those kernels land in the #[kernel] DSL.
// `fft.rs` and `fence.rs` are now real `#[kernel]` ports (declared
// above / in `ffai/`).

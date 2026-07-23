//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//! DSL-facing type system: the types kernel authors write directly.
//!
//! These are the surface-level building blocks:
//! - [`DType`] — element data types (`f32`, `f16`, `bf16`, …)
//! - [`Shape`] / [`Dim`] — tensor shapes with compile-time dim tracking
//! - [`ConstExpr`] — constexpr values resolved at kernel compile time

pub mod constexpr;
pub mod dtype;
pub mod shape;

pub use constexpr::ConstExpr;
pub use dtype::DType;
pub use shape::{Dim, DimExpr, Shape, tile};

//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//! Caching layer: PSO compilation and MSL source generation.

#[cfg(target_os = "macos")]
pub(crate) mod msl_cache;
#[cfg(any(target_os = "macos", test))]
pub(crate) mod pso_cache;

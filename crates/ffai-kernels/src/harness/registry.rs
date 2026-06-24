//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Bench and test inventory registries for the harness runner.
//!
//! This module owns `inventory::collect!` for [`KernelBenchEntry`] and
//! [`KernelTestEntry`], and exposes the typed accessors used by the runner.
//!
//! [`KernelEntry`] (kernel IR builders) and its `collect!` call live in
//! `ffai_kernels_core::registry` because `ffai-kernels-codegen` depends on core
//! but cannot depend on this facade crate (circular dependency). The
//! `all_kernels()` accessor is re-exported here for convenience.

use crate::harness::{bench::KernelBenchEntry, test::KernelTestEntry};

// `collect!` must be in the same crate as the type definition.
// Both entry types are defined in this (`ffai-kernels`) crate.
inventory::collect!(KernelBenchEntry);
inventory::collect!(KernelTestEntry);

/// Iterate all registered bench definitions, sorted alphabetically by name.
///
/// Called by the runner after all `inventory::submit!` calls have fired at
/// link time. No other module should call `inventory::iter` for this type.
pub fn all_benches() -> impl Iterator<Item = &'static KernelBenchEntry> {
    let mut entries: Vec<_> = inventory::iter::<KernelBenchEntry>.into_iter().collect();
    entries.sort_unstable_by_key(|e| e.bench().name());
    entries.into_iter()
}

/// Iterate all registered test definitions, sorted alphabetically by name.
///
/// Called by the runner after all `inventory::submit!` calls have fired at
/// link time. No other module should call `inventory::iter` for this type.
pub fn all_tests() -> impl Iterator<Item = &'static KernelTestEntry> {
    let mut entries: Vec<_> = inventory::iter::<KernelTestEntry>.into_iter().collect();
    entries.sort_unstable_by_key(|e| e.test().name());
    entries.into_iter()
}

/// Iterate all registered kernel IR builders.
///
/// Re-exported from `ffai_kernels_codegen::kernel_registry`. The `tile` CLI
/// never calls this — it only runs in the `__tile_runner` subprocess.
pub use ffai_kernels_codegen::{KernelEntry, all_kernels};

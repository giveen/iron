//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//! CLI subcommand modules.
//!
//! Each subcommand will eventually implement the [`IronCommand`] trait so that
//! they can be dispatched uniformly by `Harness`/`ProjectRunner`.  The trait
//! is defined here; implementations are migrated incrementally in later steps.

pub mod bench;
pub mod build;
pub mod clean;
pub mod config;
pub mod device;
pub mod diff;
pub mod init;
pub mod inspect;
pub mod snap;
pub mod test;
pub mod update;

use crate::{CliError, harness::Harness};

/// Common interface for all `iron` subcommands.
///
/// Implementors receive a shared `Harness` (config + runner handle) and return
/// a `Result` so that the main dispatch loop can handle errors uniformly.
pub trait IronCommand {
    /// Execute the subcommand.  Returns `Ok(())` on success.
    fn run(&self, harness: &Harness) -> Result<(), CliError>;
}

// Re-export the concrete command types so callers can import from `cmd`.
pub use bench::BenchCommand;
pub use build::BuildCommand;
pub use init::InitCommand;
pub use inspect::InspectCommand;
pub use test::TestCommand;

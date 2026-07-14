//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
use thiserror::Error;

/// Canonical exit codes for the `ffaik` binary.
///
/// Mirrors the pattern from `foundry_cli::ExitCode` so CI pipelines can
/// distinguish test failures from build failures from regressions.
#[repr(i32)]
pub enum FFAIExitCode {
    /// All kernels passed / command succeeded.
    Success = 0,
    /// One or more `#[test_kernel]` checks failed.
    TestFailure = 1,
    /// Compilation or build step failed.
    BuildFailure = 2,
    /// `ffaik diff` detected a performance regression beyond the threshold.
    Regression = 3,
    /// `ffai.toml` parsing or configuration error.
    ConfigError = 10,
}

#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("metal compile failed: {0}")]
    MetalCompile(String),

    #[error("GPU runner initialization failed: {0}")]
    GpuInit(String),

    #[error("subprocess failed: {0}")]
    Subprocess(String),

    #[error("one or more tests failed")]
    TestFailure,

    #[error("build failed")]
    BuildFailure,

    #[error("performance regression detected")]
    Regression,

    #[error("{0}")]
    Other(String),
}

impl CliError {
    /// Map this error to its canonical process exit code.
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::TestFailure => FFAIExitCode::TestFailure as i32,
            CliError::BuildFailure | CliError::MetalCompile(_) => FFAIExitCode::BuildFailure as i32,
            CliError::Regression => FFAIExitCode::Regression as i32,
            _ => 1,
        }
    }
}

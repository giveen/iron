//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//! Runtime errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum IronError {
    #[error("metal error: {0}")]
    Metal(String),

    #[error("no Metal device found")]
    NoDevice,

    #[error("kernel compilation failed: {0}")]
    Compilation(String),

    #[error("buffer allocation failed: {0}")]
    Buffer(String),

    #[error("dispatch failed: {0}")]
    Dispatch(String),

    #[error("core error: {0}")]
    Core(#[from] wh_iron_core::error::Error),

    #[error("codegen error: {0}")]
    Codegen(#[from] wh_iron_codegen::error::Error),

    /// The device cannot satisfy a kernel-launch requirement its
    /// architecture caps below (e.g. requesting >48KB dynamic shared
    /// memory on a pre-Volta GPU where `cuFuncSetAttribute` rejects the
    /// opt-in). Surfaced *before* launch with a clear reason rather than
    /// letting a cryptic `cuLaunchKernel: invalid argument` escape.
    #[error("device capability: {0}")]
    DeviceCapability(String),

    #[error("not implemented on this platform")]
    UnsupportedPlatform,

    #[error("metal device creation failed")]
    DeviceCreation,

    #[error("metal command queue creation failed")]
    QueueCreation,

    #[error("MSL library compilation failed: {0}")]
    MslCompilation(String),

    #[error("kernel function '{name}' not found in compiled library")]
    FunctionNotFound { name: String },

    #[error("compute pipeline creation failed for '{name}': {reason}")]
    PipelineCreation { name: String, reason: String },

    #[error("buffer allocation failed for '{param}': {reason}")]
    BufferAllocation { param: String, reason: String },

    #[error("buffer size mismatch for '{param}': expected {expected} bytes, got {actual}")]
    BufferSize { param: String, expected: usize, actual: usize },

    #[error("mutex poisoned: {0}")]
    LockPoisoned(String),
}

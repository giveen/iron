//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Iron runtime: GPU dispatch and buffer management.
//!
//! This crate handles the runtime execution of compiled Iron kernels:
//! - Metal device and command queue management
//! - Pipeline state compilation and caching
//! - Buffer allocation and transfer

pub mod buffer;
mod cache;
pub mod context;
mod device;
mod dispatch;
pub mod error;

pub use context::{Context, DispatchResult, DispatchSpec, ResidentBuffer};
#[cfg(feature = "cuda")]
pub use device::cuda::{CudaDevice, CudaFunction, CudaModule, DeviceBuffer};
pub use device::gpu_family::GpuFamily;
#[cfg(feature = "hip")]
pub use device::hip::{HipBuffer, HipDevice, HipKernel, HipModuleHandle};
#[cfg(feature = "vulkan")]
pub use device::vulkan::{
    BatchDispatch,
    VulkanBuffer,
    VulkanDevice,
    VulkanPipeline,
    VulkanRawBuffer,
    compile_glsl_to_spv,
};
pub use error::IronError;

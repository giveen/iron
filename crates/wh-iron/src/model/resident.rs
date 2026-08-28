//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Resident/cached bindings: compiled kernels and uploaded weights reused
//! across decode steps instead of JIT-compiling and re-uploading each time.

use std::collections::BTreeMap;

use wh_iron_core::ir::Kernel;

/// Cached compiled kernel + its launch geometry.
#[derive(Debug, Clone)]
pub struct ResidentKernel {
    pub kernel: Kernel,
    pub block: [u32; 3],
    pub grid: [u32; 3],
    pub shared_bytes: u32,
}

/// Cache of resident kernels by name.
#[derive(Debug, Default)]
pub struct ResidentCache {
    kernels: BTreeMap<String, ResidentKernel>,
}

impl ResidentCache {
    pub fn new() -> Self { Self::default() }

    pub fn insert(&mut self, kernel: ResidentKernel) {
        self.kernels.insert(kernel.kernel.name.clone(), kernel);
    }

    pub fn get(&self, name: &str) -> Option<&ResidentKernel> {
        self.kernels.get(name)
    }

    pub fn len(&self) -> usize { self.kernels.len() }
    pub fn is_empty(&self) -> bool { self.kernels.is_empty() }
}

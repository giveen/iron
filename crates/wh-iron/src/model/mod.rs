//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! High-level model runtime: checkpoint loading, KV-cache management,
//! prefill/decode orchestration, and resident kernel/weight caching.
//!
//! This is the inference layer above raw `Context::dispatch` /
//! `CudaDevice::run_kernel`. It owns the weight tensors, KV cache,
//! and the generate loop so users do not have to hand-craft every
//! dispatch.

pub mod checkpoint;
pub mod forward;
pub mod kv;
pub mod resident;
pub mod sampler;

use std::collections::BTreeMap;

use wh_iron_core::ir::Kernel;

/// Minimal model description used to allocate caches and route kernels.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: wh_iron_core::dtype::DType,
}

impl ModelInfo {
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn kv_heads(&self) -> usize { self.num_kv_heads }
    pub fn kv_dim(&self) -> usize { self.num_kv_heads * self.head_dim }
    pub fn q_dim(&self) -> usize { self.num_heads * self.head_dim }
}

/// One transformer-layer step: inputs, expected outputs, metadata.
#[derive(Debug, Clone)]
pub struct LayerStep {
    pub name: &'static str,
    pub mode: forward::ForwardMode,
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub kernel: Kernel,
}

/// Forward pass state: scratch buffers + outputs accumulated across layers.
#[derive(Debug, Clone, Default)]
pub struct ForwardState {
    pub buffers: BTreeMap<String, Vec<u8>>,
}

impl ForwardState {
    pub fn seed(&mut self, name: impl Into<String>, bytes: Vec<u8>) {
        self.buffers.insert(name.into(), bytes);
    }

    pub fn borrow(&self, name: &str) -> Option<&[u8]> {
        self.buffers.get(name).map(|b| b.as_slice())
    }

    pub fn set_output(&mut self, name: impl Into<String>, bytes: Vec<u8>) {
        self.buffers.insert(name.into(), bytes);
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenerateRequest {
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub finished: bool,
    pub prompt_len: usize,
}

/// Compute dispatch geometry for a single attention layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerGeometry {
    pub prefill_grid: [u32; 3],
    pub prefill_tpg: [u32; 3],
    pub decode_grid: [u32; 3],
    pub decode_tpg: [u32; 3],
}

impl LayerGeometry {
    pub fn from_model(_info: &ModelInfo, tpg: usize) -> Self {
        let tpg_u = tpg as u32;
        Self {
            prefill_grid: [1, 1, 1],
            prefill_tpg: [tpg_u, 1, 1],
            decode_grid: [1, 1, 1],
            decode_tpg: [tpg_u, 1, 1],
        }
    }
}

/// Resident compiled kernel handle. Thin wrapper so the model runtime
/// can keep compiled kernels + their param buffers alive across many
/// decode steps.
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

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

pub use forward::{ForwardMode, ForwardPlan, ForwardState, GenerateRequest, GenerateResult, AnySampler, SamplerBackend};
pub use kv::{KvCache, KvLayout};
pub use resident::{ResidentCache, ResidentKernel};
pub use sampler::{SampleConfig, Sampler, Tokenizer, TokenizerInner};

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

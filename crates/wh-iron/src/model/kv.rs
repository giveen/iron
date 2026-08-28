//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! KV-cache allocator: persistent device buffers across decode steps,
//! with MQA/GQA-aware layout handling.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use wh_iron_runtime::CudaDevice;

#[derive(Debug, Clone, Copy)]
pub struct KvLayout {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
}

impl KvLayout {
    pub fn bytes_per_token(&self, dtype: usize) -> usize {
        self.num_kv_heads * self.head_dim * dtype * 2
    }

    pub fn layer_bytes(&self, dtype: usize) -> usize {
        self.max_seq_len * self.bytes_per_token(dtype)
    }

    pub fn total_bytes(&self, dtype: usize) -> usize {
        self.num_layers * self.layer_bytes(dtype)
    }
}

#[cfg(feature = "cuda")]
pub struct KvCache {
    _dev: Arc<CudaDevice>,
    pub layout: KvLayout,
    #[allow(dead_code)]
    ptrs: Vec<usize>,
    #[allow(dead_code)]
    lens: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl KvCache {
    /// Best-effort alloc; if the device cannot satisfy the full cache,
    /// returns `None` so callers can fall back to host-backed or sliced caches.
    pub fn allocate(dev: Arc<CudaDevice>, layout: KvLayout, dtype: usize) -> Option<Self> {
        let mut ptrs = Vec::with_capacity(layout.num_layers);
        let mut lens = Vec::with_capacity(layout.num_layers);
        for _ in 0..layout.num_layers {
            let len = layout.layer_bytes(dtype);
            if len == 0 {
                continue;
            }
            match unsafe { dev.alloc(len) } {
                Ok(b) => {
                    ptrs.push(b.device_ptr() as usize);
                    lens.push(b.len());
                }
                Err(_) => return None,
            }
        }
        Some(Self { _dev: dev, layout, ptrs, lens })
    }

    pub fn layout(&self) -> KvLayout { self.layout }
}

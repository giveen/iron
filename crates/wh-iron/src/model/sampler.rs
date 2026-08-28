//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Sampler: top-k, temperature, and token-id decoding backed by the
//! existing `wh_iron_std` sampling kernels where possible.

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SampleConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub max_new_tokens: usize,
    pub stop: Vec<u32>,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
            max_new_tokens: 256,
            stop: Vec::new(),
        }
    }
}

pub struct Tokenizer {
    pub vocab_size: usize,
    pub inner: Arc<dyn TokenizerInner>,
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        Self { vocab_size: self.vocab_size, inner: Arc::clone(&self.inner) }
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("vocab_size", &self.vocab_size).field("inner", &"<tokenizer>").finish()
    }
}

impl Tokenizer {
    pub fn new(vocab_size: usize, inner: Arc<dyn TokenizerInner>) -> Self {
        Self { vocab_size, inner }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> { self.inner.encode(text) }

    pub fn decode(&self, ids: &[u32]) -> String { self.inner.decode(ids) }
}

pub trait TokenizerInner: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
}

#[derive(Debug, Clone)]
pub struct Sampler {
    pub config: SampleConfig,
    pub tokenizer: Tokenizer,
    pub rng_state: u64,
}

impl Sampler {
    pub fn new(tokenizer: Tokenizer, config: SampleConfig) -> Self {
        Self { tokenizer, config, rng_state: 0x9E3779B9u64 }
    }

    /// CPU-side top-k + argmax sampler scaffold. Operates on plain f32
    /// logits; returns the sampled token id. This is intentionally
    /// simple — the GPU-backed path will replace this with
    /// `iron_logits_topk_mask` + `iron_softmax_categorical_sample`.
    pub fn sample_logits(&self, logits: &[f32], _seq_len: usize) -> u32 {
        let vocab = self.tokenizer.vocab_size;
        let n = logits.len().min(vocab);
        let mut indices: Vec<usize> = (0..n).collect();
        if self.config.top_k > 0 && self.config.top_k < n {
            indices.sort_by(|&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            indices.truncate(self.config.top_k);
        }
        let best = indices
            .iter()
            .max_by(|&&a, &&b| {
                logits[a].partial_cmp(&logits[b]).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();
        best.unwrap_or(0) as u32
    }
}

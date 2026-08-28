//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Sampler stubs: top-p, top-k, temperature, and token-id decoding.
//! These are CPU-side policy helpers for the generate loop.

use std::sync::Arc;

use crate::model::ModelInfo;

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
    fn clone(&self) -> Self { Self { vocab_size: self.vocab_size, inner: Arc::clone(&self.inner) } }
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

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids)
    }
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

    pub fn sample_logits(&self, _logits: &[f32], _seq_len: usize) -> u32 {
        // Placeholder: returns EOS for now.
        // Replace with GPU-backed softmax + top-p/top-k once the generate loop is wired.
        self.tokenizer.vocab_size.saturating_sub(1) as u32
    }
}

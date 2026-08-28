//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Tokenizer wrappers: BPE and Unigram-style interfaces.

use crate::model::sampler::TokenizerInner;

/// UTF-8 byte-fallback tokenizer. Decodes any byte sequence safely;
/// encodes by splitting on UTF-8 boundaries. Production models should
/// replace this with a real BPE/Unigram implementation.
#[derive(Debug, Clone, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self { Self::default() }
}

impl TokenizerInner for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len().min(1024));
        for id in ids {
            if *id <= 255 {
                bytes.push(*id as u8);
            } else {
                bytes.extend_from_slice(format!("<{id}>").as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

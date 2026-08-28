//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Checkpoint loader scaffold: safetensors + GGUF weight ingestion into
//! device buffers. This module intentionally avoids depending on external
//! runtime format crates until the project explicitly opts in; the public
//! API is fixed so the rest of the inference stack can be built now.

use std::collections::BTreeMap;

use wh_iron_core::dtype::DType;

/// Loaded tensor slice: host bytes + static shape.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl TensorSlice {
    pub fn new(name: impl Into<String>, dtype: DType, shape: impl Into<Vec<usize>>, bytes: Vec<u8>) -> Self {
        Self { name: name.into(), dtype, shape: shape.into(), bytes }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty() || self.element_count() == 0
    }
}

/// Model weights + metadata parsed from a checkpoint file or directory.
#[derive(Debug, Clone, Default)]
pub struct Checkpoint {
    pub tensors: BTreeMap<String, TensorSlice>,
    pub metadata: BTreeMap<String, String>,
}

impl Checkpoint {
    pub fn new() -> Self { Self::default() }

    pub fn insert_tensor(&mut self, tensor: TensorSlice) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    pub fn get(&self, name: &str) -> Option<&TensorSlice> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

/// Checkpoint source kinds supported by the loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointKind {
    Safetensors,
    Gguf,
}

/// Checkpoint load result.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("unsupported checkpoint format for path '{0}'")]
    UnsupportedFormat(String),
    #[error("missing required tensor '{0}'")]
    MissingTensor(String),
    #[error("shape mismatch for '{name}': expected {expected:?}, got {got:?}")]
    ShapeMismatch { name: String, expected: Vec<usize>, got: Vec<usize> },
    #[error("io error: {0}")]
    Io(String),
}

impl From<std::io::Error> for CheckpointError {
    fn from(value: std::io::Error) -> Self {
        CheckpointError::Io(value.to_string())
    }
}

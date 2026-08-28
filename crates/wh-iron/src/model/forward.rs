//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Prefill/decode orchestrator for the CUDA runtime path. Sequences
//! registered kernels over the existing `run_kernel` device API.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::model::{LayerGeometry, ModelInfo, SampleConfig};

#[cfg(feature = "cuda")]
use crate::model::ResidentCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    Prefill,
    Decode,
}

/// One transformer-layer step: inputs, expected outputs, metadata.
#[derive(Debug, Clone)]
pub struct LayerStep {
    pub name: String,
    pub mode: ForwardMode,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub kernel: wh_iron_core::ir::Kernel,
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

#[derive(Debug)]
pub struct ForwardPlan<'a> {
    pub info: &'a ModelInfo,
    pub geom: LayerGeometry,
    pub steps: Vec<LayerStep>,
}

impl<'a> ForwardPlan<'a> {
    pub fn new(info: &'a ModelInfo, steps: Vec<LayerStep>) -> Self {
        let geom = LayerGeometry::from_model(info, 32);
        Self { info, geom, steps }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
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

pub trait SamplerBackend: Send + Sync {
    fn sample_logits(&self, logits: &[f32], config: &SampleConfig) -> Result<u32, String>;
}

#[derive(Clone)]
pub struct AnySampler(Arc<dyn SamplerBackend>);

impl AnySampler {
    pub fn new<S: SamplerBackend + 'static>(inner: S) -> Self {
        Self(Arc::new(inner))
    }

    pub fn sample_logits(&self, logits: &[f32], config: &SampleConfig) -> Result<u32, String> {
        self.0.sample_logits(logits, config)
    }
}

impl std::fmt::Debug for AnySampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnySampler").finish()
    }
}

/// CUDA-backed generate loop scaffold. Uses raw device kernels until
/// the model-specific kernel set is wired.
#[cfg(feature = "cuda")]
pub struct CudaGenerator<'a> {
    pub dev: &'a wh_iron_runtime::CudaDevice,
    #[allow(dead_code)]
    pub info: ModelInfo,
    pub cache: ResidentCache,
    pub state: ForwardState,
    pub sampler: AnySampler,
}

#[cfg(feature = "cuda")]
impl<'a> CudaGenerator<'a> {
    pub fn new(
        dev: &'a wh_iron_runtime::CudaDevice,
        info: ModelInfo,
        sampler: AnySampler,
    ) -> Self {
        Self {
            dev,
            info,
            cache: ResidentCache::new(),
            state: ForwardState::default(),
            sampler,
        }
    }

    /// Prime the prompt embeddings or token ids into the forward state.
    pub fn set_input_ids(&mut self, ids: &[u32]) {
        let bytes = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        self.state.seed("input_ids", bytes);
    }

    /// Append generated token ids to the existing input sequence.
    pub fn append_ids(&mut self, ids: &[u32]) {
        let mut existing = self
            .state
            .borrow("input_ids")
            .map(|b| b.to_vec())
            .unwrap_or_default();
        existing.extend(ids.iter().flat_map(|id| id.to_le_bytes()));
        self.state.seed("input_ids", existing);
    }

    /// Run one forward plan and update the generator state in place.
    /// Returns the concatenated output bytes from the final step, if any.
    pub fn run_plan(&mut self, plan: &ForwardPlan<'_>) -> Result<Vec<u8>, String> {
        let mut local_state = self.state.clone();
        let mut last_out = Vec::new();
        for step in &plan.steps {
            let resident = self
                .cache
                .get(&step.name)
                .ok_or_else(|| format!("missing resident kernel: {}", step.name))?;
            let mut buffers = BTreeMap::new();
            for input in &step.inputs {
                let data = local_state
                    .borrow(input)
                    .ok_or_else(|| format!("missing input buffer: {}", input))?;
                buffers.insert(input.clone(), data.to_vec());
            }
            let outputs = self
                .dev
                .run_kernel(&resident.kernel, &buffers, resident.block, resident.grid)
                .map_err(|e| e.to_string())?;
            last_out.clear();
            for output in &step.outputs {
                let bytes = outputs.get(output).cloned().unwrap_or_default();
                local_state.set_output(output.clone(), bytes.clone());
                last_out = bytes;
            }
        }
        self.state = local_state;
        Ok(last_out)
    }

    /// Generate tokens for the request.
    pub fn generate(&mut self, mut req: GenerateRequest) -> Result<GenerateResult, String> {
        let prompt_len = req.prompt_ids.len();
        let stop = std::mem::take(&mut req.stop);
        self.set_input_ids(&req.prompt_ids);
        let mut generated = Vec::new();
        for _ in 0..req.max_new_tokens {
            let token = self.sample_next(&req)?;
            generated.push(token);
            self.append_ids(&[token]);
            if stop.contains(&token) {
                req.max_new_tokens = 0;
                break;
            }
        }
        Ok(GenerateResult { tokens: generated, finished: req.max_new_tokens == 0, prompt_len })
    }

    fn sample_next(&mut self, req: &GenerateRequest) -> Result<u32, String> {
        let logits = self.state.borrow("logits").ok_or("missing logits")?;
        let vals: Vec<f32> =
            logits.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        self.sampler.sample_logits(&vals, &SampleConfig {
            temperature: req.temperature,
            top_k: 0,
            top_p: req.top_p,
            min_p: 0.0,
            max_new_tokens: 0,
            stop: Vec::new(),
        })
    }
}

//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! CUDA model-runtime smoke: prove the new `wh_iron::model` façade
//! integrates with `CudaDevice::run_kernel` by dispatching one real
//! kernel from the registered corpus through the scaffold types.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::sync::Arc;

use wh_iron::{
    core::dtype::DType,
    model::{
        checkpoint::{Checkpoint, TensorSlice},
        forward::{CudaGenerator, ForwardPlan, GenerateRequest, LayerStep, ForwardMode},
        ForwardState,
        ResidentCache,
        Sampler,
        Tokenizer,
        TokenizerInner,
    },
    CudaDevice,
};
use wh_iron_std::all_tests;

struct DummyTokenizer;
impl TokenizerInner for DummyTokenizer {
    fn encode(&self, _text: &str) -> Vec<u32> { Vec::new() }
    fn decode(&self, _ids: &[u32]) -> String { String::new() }
}

fn leak_static(s: impl Into<String>) -> String {
    s.into()
}

#[test]
fn cuda_model_scaffold_smoke() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    let entry = all_tests().next().expect("at least one registered test kernel");
    let setup = entry.test().setup(DType::F32);

    let mut buffers = BTreeMap::new();
    for inp in setup.inputs() {
        buffers.insert(inp.name().to_string(), inp.data().to_vec());
    }
    for (k, v) in setup.constexprs() {
        buffers.insert(k.clone(), v.to_le_bytes());
    }

    let grid = setup.grid();
    let raw_out = dev.run_kernel(setup.kernel(), &buffers, grid.grid, grid.tpg).expect("raw CUDA run_kernel");

    let mut state = ForwardState::default();
    for (name, bytes) in &buffers {
        state.seed(name.clone(), bytes.clone());
    }

    let mut cache = ResidentCache::new();
    cache.insert(wh_iron::model::ResidentKernel { kernel: setup.kernel().clone(), block: grid.tpg, grid: grid.grid, shared_bytes: 0 });

    let tokenizer = wh_iron::model::Tokenizer::new(1024, Arc::new(DummyTokenizer));
    let mut runner = CudaGenerator::new(
        &dev,
        wh_iron::model::ModelInfo {
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 32,
            vocab_size: 1024,
            num_layers: 1,
            max_seq_len: 2048,
            dtype: DType::F32,
        },
        wh_iron::model::AnySampler::new(wh_iron::model::Sampler::new(tokenizer, Default::default())),
    );
    runner.state = state.clone();

    let req = GenerateRequest { prompt_ids: vec![], max_new_tokens: 1, ..GenerateRequest::default() };
    let _ = runner.generate(req);

    assert!(!raw_out.is_empty());
}

#[test]
fn cuda_model_run_plan_smoke() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    let entry = all_tests().next().expect("at least one registered test kernel");
    let setup = entry.test().setup(DType::F32);

    let grid = setup.grid();
    let mut buffers = BTreeMap::new();
    for inp in setup.inputs() {
        buffers.insert(inp.name().to_string(), inp.data().to_vec());
    }
    for (k, v) in setup.constexprs() {
        buffers.insert(k.clone(), v.to_le_bytes());
    }

    let mut state = ForwardState::default();
    for (name, bytes) in &buffers {
        state.seed(name.clone(), bytes.clone());
    }

    let mut cache = ResidentCache::new();
    cache.insert(wh_iron::model::ResidentKernel {
        kernel: setup.kernel().clone(),
        block: grid.tpg,
        grid: grid.grid,
        shared_bytes: 0,
    });

    let tokenizer = wh_iron::model::Tokenizer::new(1024, Arc::new(DummyTokenizer));
    let mut runner = CudaGenerator::new(
        &dev,
        wh_iron::model::ModelInfo {
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 32,
            vocab_size: 1024,
            num_layers: 1,
            max_seq_len: 2048,
            dtype: DType::F32,
        },
        wh_iron::model::AnySampler::new(wh_iron::model::Sampler::new(
            tokenizer,
            Default::default(),
        )),
    );
    runner.state = state;
    runner.cache = cache;

    let info = runner.info.clone();
    let output_names = if setup.expected().is_empty() {
        vec!["out".to_string()]
    } else {
        setup.expected().iter().map(|b| b.name().to_string()).collect()
    };
    let input_names: Vec<String> = setup.inputs().iter().map(|b| b.name().to_string()).collect();
    let step_name = setup.kernel().name.clone();
    let plan = ForwardPlan::new(&info, vec![LayerStep {
        name: step_name,
        mode: ForwardMode::Decode,
        inputs: input_names,
        outputs: output_names,
        kernel: setup.kernel().clone(),
    }]);

    let out = runner.run_plan(&plan).expect("run_plan");
    assert!(!out.is_empty(), "run_plan produced empty output");
}

#[test]
fn cuda_checkpoint_upload_smoke() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    let payload: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let bytes = payload.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let mut ckpt = Checkpoint::new();
    ckpt.insert_tensor(TensorSlice::new(
        "small.weight",
        DType::F32,
        vec![8, 8],
        bytes.clone(),
    ));

    let tensor = ckpt.get("small.weight").expect("tensor exists");
    let buf = dev.upload(tensor.bytes.as_slice()).expect("upload tensor bytes");
    let mut host = vec![0u8; tensor.bytes.len()];
    dev.download(&buf, &mut host).expect("download tensor bytes");

    assert_eq!(host, bytes);
}

#[test]
fn cuda_gpu_sampler_smoke() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    let sampler = wh_iron_std::kernels::sampling::gpu_pipeline::GpuSampler::new(&dev);
    let mut logits = vec![0.0f32; 1024];
    logits[42] = 8.0;
    logits[7] = 4.0;

    let config = wh_iron::model::SampleConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        ..wh_iron::model::SampleConfig::default()
    };

    let token = sampler.sample_logits(&logits, &config).expect("gpu sample");
    assert_eq!(token, 42);
}

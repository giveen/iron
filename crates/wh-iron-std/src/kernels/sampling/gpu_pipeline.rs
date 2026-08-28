//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//!
//! CUDA-backed sampling pipeline: chains the existing logits kernels
//! (`iron_logits_temperature`, `iron_logits_topk_mask`,
//! `iron_logits_top_p_mask`) and returns the argmax token id over the
//! masked logits. This keeps the initial path minimal while still
//! exercising real GPU filters instead of the CPU-only placeholder.

use std::collections::BTreeMap;

use wh_iron::{core::dtype::DType, CudaDevice};
use wh_iron::test::KernelMode;

use crate::kernels::sampling::{
    logits_top_p::iron_logits_top_p_mask,
    logits_topk::iron_logits_topk_mask,
    logits_processors::iron_logits_temperature,
};

/// Thin GPU sampler that runs the existing logits-mask kernels on a
/// `CudaDevice` and returns the argmax token id over the masked logits.
pub struct GpuSampler<'a> {
    dev: &'a CudaDevice,
}

impl<'a> GpuSampler<'a> {
    pub fn new(dev: &'a CudaDevice) -> Self {
        Self { dev }
    }

    pub fn sample_logits(&self, logits: &[f32], config: &wh_iron::model::sampler::SampleConfig) -> Result<u32, String> {
        let n = logits.len();
        let mut current = logits.to_vec();

        if config.temperature != 1.0 {
            current = self.run_temperature(&current, config.temperature)?;
        }
        if config.top_k > 0 && config.top_k < n {
            let threshold = self.kth_largest(&current, config.top_k);
            current = self.run_topk_mask(&current, threshold)?;
        }
        if config.top_p > 0.0 && config.top_p < 1.0 {
            current = self.run_topp_mask(&current, config.top_p)?;
        }

        current
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| "empty logits after masking".to_string())
    }

    fn run_temperature(&self, logits: &[f32], temperature: f32) -> Result<Vec<f32>, String> {
        let n = logits.len();
        let kernel = iron_logits_temperature::kernel_ir_for(DType::F32);
        let mut buffers = BTreeMap::new();
        buffers.insert("inp".to_string(), Self::pack_f32(logits));
        buffers.insert("out".to_string(), vec![0u8; n * 4]);
        buffers.insert("temperature".to_string(), temperature.to_le_bytes().to_vec());

        let grid = Self::grid_1d(n, 256);
        let block = [256, 1, 1];
        let outputs = self.dev.run_kernel(&kernel, &buffers, grid, block).map_err(|e| e.to_string())?;

        outputs.get("out")
            .cloned()
            .ok_or_else(|| "missing temperature output".to_string())
            .map(|data| Self::unpack_f32(&data))
    }

    fn run_topk_mask(&self, logits: &[f32], threshold: f32) -> Result<Vec<f32>, String> {
        let n = logits.len();
        let kernel = iron_logits_topk_mask::kernel_ir_for(DType::F32);
        let mut buffers = BTreeMap::new();
        buffers.insert("inp".to_string(), Self::pack_f32(logits));
        buffers.insert("out".to_string(), vec![0u8; n * 4]);
        buffers.insert("threshold".to_string(), threshold.to_le_bytes().to_vec());

        let grid = Self::grid_1d(n, 256);
        let block = [256, 1, 1];
        let outputs = self.dev.run_kernel(&kernel, &buffers, grid, block).map_err(|e| e.to_string())?;

        outputs.get("out")
            .cloned()
            .ok_or_else(|| "missing top-k output".to_string())
            .map(|data| Self::unpack_f32(&data))
    }

    fn run_topp_mask(&self, logits: &[f32], top_p: f32) -> Result<Vec<f32>, String> {
        let n = logits.len();
        let mut kernel = iron_logits_top_p_mask::kernel_ir_for(DType::F32);
        kernel.mode = KernelMode::Reduction;
        let mut buffers = BTreeMap::new();
        buffers.insert("inp".to_string(), Self::pack_f32(logits));
        buffers.insert("out".to_string(), vec![0u8; n * 4]);
        buffers.insert("n".to_string(), (n as u32).to_le_bytes().to_vec());
        buffers.insert("top_p".to_string(), top_p.to_le_bytes().to_vec());

        let grid = Self::grid_1d(n, 256);
        let block = [256, 1, 1];
        let outputs = self.dev.run_kernel(&kernel, &buffers, grid, block).map_err(|e| e.to_string())?;

        outputs.get("out")
            .cloned()
            .ok_or_else(|| "missing top-p output".to_string())
            .map(|data| Self::unpack_f32(&data))
    }

    fn kth_largest(&self, logits: &[f32], k: usize) -> f32 {
        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted[k - 1]
    }

    fn grid_1d(n: usize, tpg: u32) -> [u32; 3] {
        let blocks = (n + tpg as usize - 1) / tpg as usize;
        [blocks as u32, 1, 1]
    }

    fn pack_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn unpack_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}


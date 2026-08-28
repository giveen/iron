//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//!
//! Checkpoint round-trip smoke: tensor -> device upload -> download -> compare.

#![cfg(feature = "cuda")]

use wh_iron::{core::dtype::DType, CudaDevice};
use wh_iron::model::checkpoint::{Checkpoint, TensorSlice};

#[test]
fn cuda_checkpoint_upload_smoke() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    let payload: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let bytes = payload.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let mut ckpt = Checkpoint::new();
    ckpt.insert_tensor(TensorSlice::new("small.weight", DType::F32, vec![8, 8], bytes.clone()));

    let tensor = ckpt.get("small.weight").expect("tensor exists");
    let buf = dev.upload(tensor.bytes.as_slice()).expect("upload tensor bytes");
    let mut host = vec![0u8; tensor.bytes.len()];
    dev.download(&buf, &mut host).expect("download tensor bytes");

    assert_eq!(host, bytes);
}

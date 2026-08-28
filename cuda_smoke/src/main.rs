//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! CUDA KV-cache smoke: run the registered `test_kv_cache_update` corpus
//! entry on CUDA through `CudaDevice::run_kernel`, comparing to the same
//! CPU oracle the Metal harness uses.

use std::collections::BTreeMap;

use half::{bf16, f16};
use wh_iron_core::dtype::DType;
use wh_iron_runtime::CudaDevice;
use wh_iron_std::{all_tests, test::DType as StdDType};

fn read_raw_f32(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                f16::from_bits(bits).to_f32()
            })
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                bf16::from_bits(bits).to_f32()
            })
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        DType::I8 => bytes.iter().take(n).map(|&b| b as i8 as f32).collect(),
        DType::U8 => bytes.iter().take(n).map(|&b| b as f32).collect(),
        _ => vec![0.0; n],
    }
}

fn main() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping smoke");
        std::process::exit(0);
    };
    println!(
        "CUDA device OK, compute capability {}.{}",
        dev.compute_capability().0,
        dev.compute_capability().1
    );

    let mut ran = 0usize;
    for entry in all_tests() {
        let t = entry.test();
        if t.name() != "test_kv_cache_update" {
            continue;
        }
        ran += 1;
        for &dt in t.dtypes() {
            let setup = t.setup(dt);
            let tol = t.tolerance(dt);
            let kernel = setup.kernel();

            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for inp in setup.inputs() {
                buffers.insert(inp.name().to_string(), inp.data().to_vec());
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes().to_vec());
            }

            let grid = setup.grid();
            let label = format!("{} [{:?}]", t.name(), dt);

            let outputs = dev
                .run_kernel(kernel, &buffers, grid.grid, grid.tpg)
                .expect("run_kernel");
            let mut worst = 0.0f32;
            for exp in setup.expected() {
                let got_bytes = outputs.get(exp.name()).expect("missing output");
                let n = exp.len();
                let got = read_raw_f32(got_bytes, exp.dtype(), n);
                let want = read_raw_f32(exp.data(), exp.dtype(), n);
                worst = worst
                    .max(got.iter().zip(want).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max));
            }
            if worst > tol {
                panic!("{label}: max|Δ|={worst:.3e} > {tol:.3e}");
            }
            println!("{label}: max|Δ|={worst:.3e}");
        }
    }

    if ran == 0 {
        panic!("registered test `test_kv_cache_update` not found in corpus");
    }
    println!("PASS");
}

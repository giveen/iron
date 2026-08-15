//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Scoped CUDA-backend run of just the GDN (Gated DeltaNet) kernel
//! family's registered `#[test_kernel]` corpus — every kernel in
//! `kernels/ssm/gated_delta*.rs`, including the Qwen3.6-27B-shape
//! fixtures added alongside `GDN_PREFILL_CONTRACT.md`.
//!
//! `tests/cuda_kernel_corpus.rs` already runs the FULL registered corpus
//! (thousands of kernel×dtype combos) on this same CUDA device; this file
//! exists purely so a GDN-focused check runs in seconds instead of minutes
//! — the corpus test is a good "did anything regress anywhere" sweep, this
//! one is the fast loop for this validation pass specifically. Same
//! `CudaDevice::run_kernel` + `wh_iron_std::all_tests()` machinery, filtered
//! by name.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use wh_iron::{CodegenError, CudaDevice, IronError, core::dtype::DType};

/// Pre-existing CUDA-only issues in the WY plan/scan pipeline, found by this
/// scoped run but NOT introduced by (or fixed in) the Qwen3.6-27B shape
/// validation pass this file belongs to — that pipeline isn't wired into
/// any production call site (see `GDN_PREFILL_CONTRACT.md` §1/§5), so
/// blocking this pass on them would hold the actually-needed
/// `iron_gated_delta_prep_chunk` validation hostage to an unrelated,
/// deeper CUDA-codegen investigation. Mirrors `tests/cuda_kernel_corpus.rs`'s
/// own `KNOWN_HARD` mechanism (same purpose: track a real gap without
/// making every future `cargo test` run red for it). See
/// `GDN_PREFILL_CONTRACT.md` §7 for the full writeup and the follow-up
/// task raised for both.
///
/// NOTE: the `iron_gdn_wy_scan` single-Dv-tile `state_out` mismatch (and
/// its `debug_*` isolation fixtures) formerly listed here is FIXED — root
/// cause was `coop_tile_load_a`/`coop_tile_run` (CUDA codegen embeds its
/// own `__syncthreads()`) called inside a per-SG runtime `if` whose
/// predicate differs *by warp*, a divergent-barrier hazard invisible to
/// `--tool racecheck`/`initcheck` but confirmed via `--tool synccheck`
/// ("Barrier error: Divergent thread(s) in block"). Fixed in
/// `gated_delta_wy_scan.rs` by running the MMA unconditionally for every
/// SG (inputs are already zero-padded, output already masked on
/// consumption, so this was always safe — see that file's module doc).
const KNOWN_HARD: &[&str] = &[];

fn known_hard(label: &str) -> bool { KNOWN_HARD.iter().any(|k| label.contains(k)) }

/// Mirrors `tests/cuda_kernel_corpus.rs::is_unsupported` exactly: a
/// `DeviceCapability` error means the codegen DOES cover the kernel but
/// this specific GPU physically cannot run it (e.g. GB10's 48 KB static /
/// ~99 KB dynamic shared-memory cap on a kernel sized for Apple's larger
/// budget) — that's an environment limit, not a correctness bug, and the
/// corpus harness does not count it as a failure. Reusing the same rule
/// here (rather than treating every `Err` as fail) keeps this scoped
/// check's pass/fail semantics identical to the full corpus run it's a
/// faster stand-in for.
fn is_unsupported(e: &IronError) -> bool {
    matches!(e, IronError::Codegen(CodegenError::UnsupportedOp(_)) | IronError::DeviceCapability(_))
}

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
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        _ => vec![0.0; n],
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| if x == y || (x.is_nan() && y.is_nan()) { 0.0 } else { (x - y).abs() })
        .fold(0.0f32, f32::max)
}

#[test]
fn gdn_family_scoped_cuda_corpus() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping GDN scoped corpus check");
        return;
    };

    let (mut pass, mut fail, mut unsupported, mut known) = (0u32, 0u32, 0u32, 0u32);
    let mut failures: Vec<String> = Vec::new();
    let mut unsupported_names: Vec<String> = Vec::new();
    let mut known_names: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    for entry in wh_iron_std::all_tests() {
        let t = entry.test();
        if !(t.name().contains("gated_delta") || t.name().contains("gdn_")) {
            continue;
        }
        for &dt in t.dtypes() {
            let setup = t.setup(dt);
            if setup.ref_setup().is_some() {
                continue; // GPU-vs-GPU ref setups need a second dispatch; none in this family today.
            }
            let tol = t.tolerance(dt);
            let kernel = setup.kernel();

            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for inp in setup.inputs() {
                buffers.insert(inp.name().to_string(), inp.data().to_vec());
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes());
            }
            let grid = setup.grid();
            let label = format!("{} [{dt}]", t.name());
            names.push(label.clone());

            match dev.run_kernel(kernel, &buffers, grid.grid, grid.tpg) {
                Ok(outputs) => {
                    let mut worst = 0.0f32;
                    for exp in setup.expected() {
                        let Some(got_bytes) = outputs.get(exp.name()) else {
                            worst = f32::INFINITY;
                            break;
                        };
                        let n = exp.len();
                        let got = read_raw_f32(got_bytes, exp.dtype(), n);
                        let want = read_raw_f32(exp.data(), exp.dtype(), n);
                        worst = worst.max(max_abs_diff(&got, &want));
                    }
                    if (worst as f64) <= tol {
                        pass += 1;
                    } else if known_hard(&label) {
                        known += 1;
                        known_names.push(format!("{label}: max|Δ|={worst:.3e} > {tol:.3e}"));
                    } else {
                        fail += 1;
                        failures.push(format!("MISMATCH {label}: max|Δ|={worst:.3e} > {tol:.3e}"));
                    }
                },
                Err(e) =>
                    if known_hard(&label) {
                        known += 1;
                        known_names.push(format!("{label}: {e}"));
                    } else if is_unsupported(&e) {
                        unsupported += 1;
                        unsupported_names.push(format!("{label}: {e}"));
                    } else {
                        fail += 1;
                        failures.push(format!("ERROR {label}: {e}"));
                    },
            }
        }
    }

    eprintln!("=== GDN family scoped CUDA corpus ===");
    eprintln!(
        "PASS={pass}  KNOWN_HARD={known}  UNSUPPORTED={unsupported}  FAIL={fail}  (of {} kernel×dtype entries)",
        names.len()
    );
    if !unsupported_names.is_empty() {
        eprintln!("--- unsupported (device capability / codegen gap, not a failure) ---");
        for n in &unsupported_names {
            eprintln!("  ~ {n}");
        }
    }
    if !known_names.is_empty() {
        eprintln!("--- known hard (tracked, see GDN_PREFILL_CONTRACT.md §7) ---");
        for n in &known_names {
            eprintln!("  ? {n}");
        }
    }
    if !failures.is_empty() {
        eprintln!("--- failures ({}) ---", failures.len());
        for f in &failures {
            eprintln!("  ✗ {f}");
        }
    }
    assert!(pass > 0, "no GDN kernels matched/passed on CUDA — filter or pipeline broken");
    assert!(fail == 0, "{fail} GDN CUDA failures (unexpected — not in KNOWN_HARD)");
}

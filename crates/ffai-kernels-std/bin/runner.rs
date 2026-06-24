//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `__tile_runner` entry point for the ffai-kernels workspace.
//!
//! User projects get their own copy scaffolded by `tile init`. This copy
//! serves the ffai-kernels workspace itself (e.g. `make bench` / `make test`).

// Force the linker to include all `inventory::submit!` statics from the
// ffai-kernels-std library so that kernel/bench/test registrations are populated.
extern crate ffai_kernels_std;

fn main() {
    let args = match ffai_kernels::runner::RunnerArgs::from_env_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("__tile_runner: {e}");
            std::process::exit(2);
        },
    };
    std::process::exit(if ffai_kernels::runner::RunnerHarness::run(&args) { 0 } else { 1 });
}

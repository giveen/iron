//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! `__iron_runner` entry point for the wh-iron workspace.
//!
//! User projects get their own copy scaffolded by `iron init`. This copy
//! serves the wh-iron workspace itself (e.g. `make bench` / `make test`).

// Force the linker to include all `inventory::submit!` statics from the
// wh-iron-std library so that kernel/bench/test registrations are populated.
extern crate wh_iron_std;

fn main() {
    let args = match wh_iron::runner::RunnerArgs::from_env_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("__iron_runner: {e}");
            std::process::exit(2);
        },
    };
    std::process::exit(if wh_iron::runner::RunnerHarness::run(&args) { 0 } else { 1 });
}

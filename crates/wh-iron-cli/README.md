# wh-iron-cli

Iron Kernels CLI — benchmark, test, and inspect GPU kernels.
The `iron` binary is the primary developer tool for the Iron Kernels project:
run performance benchmarks against MLX, compile kernels to inspect generated
MSL, emit kernel packages via `iron build --emit`, and manage regression baselines.

This is a binary crate only — it has no library API. All functionality is
exposed through subcommands of the `iron` binary.

## Position in the pipeline

```
wh-iron (facade) ──┐
wh-iron-core       │
wh-iron-codegen    ├──► wh-iron-cli (this crate) ──► terminal / JSON / files
wh-iron-runtime    │         iron binary
wh-iron-std ───────┘
```

The CLI sits at the top of the stack, consuming every other crate.
It's the only crate in the workspace that exercises the full
compile→dispatch→measure loop end-to-end.

## Quick start

```sh
# Install the iron binary
cargo install --path crates/wh-iron-cli

# Run the full benchmark suite (requires macOS + Metal)
iron bench

# Compile all kernels and report errors
iron build

# Emit kernel package (metallib + sources + Swift wrappers)
iron build --emit all -o /tmp/kernel-pkg

# Inspect one kernel's IR and generated MSL
iron inspect --kernel iron_rms_norm

# Show GPU device info
iron device

# Save current bench results as a baseline
iron snap -o baseline.json

# Compare current bench results to a saved baseline
iron diff baseline.json
```

Subcommand-specific help:

```sh
iron bench --help
iron build --help
iron inspect --help
```

## Crate contents

| Module | Purpose |
|---|---|
| `cmd` | Subcommand dispatch: `bench`, `build`, `inspect`, `device`, `snap`, `diff` |
| `cmd::bench` | Full benchmark suite: Iron Kernels vs MLX reference kernels |
| `cmd::build` | Compile all kernels to MSL, report errors, and emit artifacts (`--emit msl,metallib,swift,ir,all`) |
| `cmd::inspect` | Print IR and/or MSL for a single kernel |
| `cmd::device` | Show GPU device info and supported Metal features |
| `cmd::snap` | Save benchmark results as a JSON regression baseline |
| `cmd::diff` | Compare current benchmark results to a saved baseline |
| `term` | Terminal styling: colored output, bold text |
| `suite_printer` | Benchmark results terminal rendering |
| `error` | `CliError` enum for CLI-specific failures |
| `git` | Git working-tree checks (dirty detection, baseline-ref resolution) |

## API reference

### Subcommands

| Command | Purpose |
|---|---|
| `iron bench` | Run the benchmark suite (latency, GFLOP/s, %-peak, bottleneck). `--filter <op>` to narrow; `--backend metal\|cuda\|hip\|vulkan` to pick a device; an optional metal reference runs side-by-side where a bench defines one. |
| `iron test` | Run the `#[test_kernel]` GPU correctness suite — each kernel's output vs its CPU oracle within tolerance. `--filter` / `--backend` as above. |
| `iron build` | Compile all registered kernels and report errors. Use `--emit msl,metallib,swift,ir,all -o <dir>` to write artifacts: `.metal` sources, `kernels.metallib`, `IronKernels.swift` wrappers, and `manifest.json`. |
| `iron inspect --kernel <name>` | Print the IR (SSA-form) and/or generated MSL for one kernel. Use `--ir` for IR only, `--msl` for MSL only. |
| `iron device` | Show GPU device info: name, feature set, supported language version, max threadgroup size. |
| `iron snap -o <file>` | Save current bench results as a JSON regression baseline file. |
| `iron diff <file>` | Compare current bench results to a saved baseline. Reports regressions. |
| `iron clean` | Remove build artifacts and cached baselines. |
| `iron config` | Print the effective merged config (defaults → `iron.toml` → `IRON_*` env → flags). |
| `iron init` | Scaffold a new Iron Kernels kernel project. |
| `iron update` | Self-update the `iron` binary. |
| `iron completions <shell>` | Generate shell completion scripts (bash / zsh / fish). |

### Installation

```sh
cargo install --path crates/wh-iron-cli
```

The binary is named `iron`. After installation it's available on your `$PATH`.

This crate is not published to crates.io (`publish = false`). It's a
project-internal developer tool, not a library.

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `wh-iron` | Facade re-exports (macros, `Context`) used by bench/inspect |
| `wh-iron-core` | IR types for kernel iteration and inspect output |
| `wh-iron-codegen` | MSL generation for build, inspect, and bench dispatch |
| `wh-iron-runtime` | GPU dispatch, PSO compilation, buffer management |
| `wh-iron-std` | `BenchSpec` registry via `inventory`, op catalog, benchmark shapes |

### External

| Crate | Role |
|---|---|
| `clap` | CLI argument parsing and subcommand dispatch |
| `serde` / `serde_json` | Serialize/deserialize snap/diff baseline files |
| `anstyle` / `anstream` | ANSI terminal coloring |
| `tracing` / `tracing-subscriber` | Diagnostics and structured logging |
| `objc2` / `objc2-metal` / `objc2-foundation` | Metal GPU API bindings (macOS only, cfg-gated) |

## MSRV / platform

**macOS is required** for GPU commands (`bench`, `device`).
All Metal API calls are cfg-gated behind `target_os = "macos"`.
On other platforms these commands return errors or zero-stub output.

`build` and `inspect` work on any platform — they only need the
compiler crates, not the GPU runtime.

Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New subcommand:** Create `src/cmd/<name>.rs` with the subcommand logic.
  Add `pub mod <name>;` to `src/cmd/mod.rs`. Add a variant to the `Command`
  enum in `src/main.rs` and a match arm in `main()`.

- **New benchmark output format:** `src/cmd/bench.rs` — extend the output
  rendering or add a `--format` flag.

- **Tests to update:** Integration tests in `src/cmd/`. Run `iron bench`
  on macOS to verify no regressions.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`wh-iron-std` README](../wh-iron-std/README.md) — the `BenchSpec` registry and op catalog this CLI exercises
- [`wh-iron-runtime` README](../wh-iron-runtime/README.md) — the GPU dispatch layer used by `runner`

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
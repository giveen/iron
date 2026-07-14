# ffai-kernels

Rust DSL for writing GPU kernels — write once, run fast. This is the user-facing
facade crate: add `ffai-kernels` to your `Cargo.toml`, import `ffai-kernels::prelude::*`,
annotate functions with `#[kernel]`, and dispatch them on the GPU with a few lines
of Rust. Metal (Apple Silicon) is the default and most mature backend; CUDA, HIP,
and Vulkan are feature-gated in the lower crates.

The crate re-exports the compiler, runtime, and macro crates under one namespace so
you never need to depend on `ffai-kernels-core`, `ffai-kernels-codegen`, or the others
directly unless you are writing tooling or compiler extensions. Beyond the
re-exports it hosts `harness/` (the `#[kernel]` / `#[bench]` / `#[test_kernel]`
registries) and `runner/` (the `__ffai_runner` engine — `ffaik` runs GPU work in a
spawned subprocess that streams results back as `ProtocolMessage` JSON lines).

## Position in the pipeline

```
        ┌──────────────────────────────┐
        │  ffai-kernels (this crate)       │
        │  user-facing facade           │
        │                              │
        │  use ffai-kernels::prelude::*;   │
        │  #[kernel]                    │
        │  kernel::launch(&ctx)         │
        └──────────┬───────────────────┘
                   │ re-exports
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
ffai-kernels-core  ffai-kernels-macros  ffai-kernels-codegen  ffai-kernels-runtime
   (IR types)    (#[kernel])      (MSL lowering)     (GPU dispatch)
```

`ffai-kernels` is the only crate end users depend on. It re-exports the DSL macros,
placeholder types, IR/codegen modules, and runtime entry points under flat paths
like `ffai-kernels::kernel`, `ffai-kernels::core`, `ffai-kernels::codegen`, and
`ffai-kernels::Context`.

## Quick start

```rust
use ffai-kernels::prelude::*;

#[kernel]
fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
    let idx = program_id::<0>();
    store(c[idx], load(a[idx]) + load(b[idx]));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new()?;
    let n = 256usize;
    let a: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..n).flat_map(|_| (1.0f32).to_le_bytes()).collect();
    let c = vec![0u8; n * 4];

    let result = vector_add::launch(&ctx)
        .input("a", a)
        .input("b", b)
        .input("c", c)
        .dispatch()?;

    let out: Vec<f32> = result.outputs["c"]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("out[0] = {}", out[0]); // 1.0
    Ok(())
}
```

To inspect the generated MSL directly without dispatching:

```rust
use ffai-kernels::codegen::msl::MslGenerator;

let msl = MslGenerator::default().generate(&vector_add::kernel_ir())?;
println!("{msl}");
```

## Crate contents

| Module | Purpose |
|---|---|
| `prelude` | Comprehensive single import: `#[kernel]` DSL stubs, all IR types, all macros, runtime bindings, and codegen entry points — re-exports **everything** public from all sub-crates |
| `codegen` | Re-export of `ffai_kernels_codegen` — MSL generation and optimization passes |
| `core` | Re-export of `ffai_kernels_core` — IR types, DType, Shape, ConstExpr |

## API reference

### Prelude

`use ffai-kernels::prelude::*;` brings **everything public from all sub-crates** into scope:

**Macros (from `ffai-kernels-macros`):**

| Macro | Kind | What it does |
|---|---|---|
| `#[kernel]` | attribute | Transforms a Rust function into IR + host-side `LaunchBuilder` |
| `#[bench]` | attribute | Registers a `BenchSetup`-returning fn for `ffaik bench` via `inventory::submit!` (separate attribute; the bench name is the function name) |
| `#[test_kernel]` | attribute | Registers a `TestSetup`-returning fn for `ffaik test` |
| `#[constexpr]` | attribute | Marks a kernel parameter as a compile-time constant |
| `#[scalar]` | attribute | Marks a `Tensor` parameter for `constant T&` lowering in MSL |
| `#[strided]` | attribute | Marks a `Tensor` parameter for strided lowering (shape + stride arrays emitted) |
| `shape!(…)` | function-like | Constructs a `Shape` from dimension expressions |
| `tile!(…)` | function-like | Constructs a 2D tile shape |

**IR types (from `ffai-kernels-core`):**

| Type | Kind | Purpose |
|---|---|---|
| `Tensor<T, S>` | struct | Placeholder type for kernel signatures — zero-sized marker |
| `DType` | enum | Numeric type: F32, F16, BF16, I32, U32, etc. |
| `Shape` | struct | Compile-time dimension tracking |
| `Dim` | enum | A single dimension: `Known`, `ConstExpr`, `Any` |
| `DimExpr` | enum | Dimension expression for indexing |
| `ConstExpr` | struct | Named compile-time constant |
| `ConstExprValues` | struct | Resolved constexpr values for a kernel launch |
| `Kernel` | struct | Complete kernel IR (params, body, blocks) |
| `KernelMode` | enum | Dispatch shape hint: Elementwise, Reduction, Tile2D, SimdGroup2D, Grid3D |
| `Op` | enum | A single IR operation (all variants) |
| `Block` | struct | Basic block: sequence of ops |
| `ValueId` | struct | Unique SSA value identifier |
| `BlockId` | struct | Unique block identifier |
| `VarId` | struct | Unique loop-variable identifier |
| `Param` | struct | Kernel parameter metadata |
| `TypedSlot` | struct | Typed hole for inline MSL |
| `UnaryOpKind` | enum | Unary math: Exp, Log, Sqrt, Cos, Sin, etc. |
| `BinOpKind` | enum | Binary math: Add, Sub, Mul, Max, Min, CmpLt, etc. |
| `ActKind` | enum | Activation: Silu, Gelu, Relu, Tanh, Sigmoid |
| `ReduceKind` | enum | Reduction: Sum, Max, Min, Mean, Product |
| `AtomicKind` | enum | Atomic: Add, Max, Min, And, Or, Xor |
| `AtomicScope` | enum | Memory scope: Device, Threadgroup |
| `CoopTileScope` | enum | Execution scope: SimdGroup, Threadgroup |
| `CoopTileAccMode` | enum | Accumulation: Overwrite, MultiplyAccumulate |
| `IndexExpr` | enum | Index expression for loads/stores |
| `KernelCallArg` | enum | Argument to a cross-kernel call |
| `KernelEntry` | struct | Registry entry for cross-kernel inlining |
| `Error` | enum | Core error type |
| `GpuFamily` | enum | Apple GPU family (Apple7–Apple10) |
| `IdCounter` | struct | Unique ID generator |

**Runtime (from `ffai-kernels-runtime`):**

| Type | Purpose |
|---|---|
| `Context` | Metal GPU device and command queue |
| `DispatchResult` | Output buffers after a kernel run |
| `DispatchSpec` | Input buffer spec for the dispatch pipeline |
| `ResidentBuffer` | A Metal buffer managed by the context |
| `FFAIError` | Top-level runtime error |

| Function | Purpose |
|---|---|
| `start_gpu_trace()` | Start Xcode GPU frame capture |
| `stop_gpu_trace()` | Stop Xcode GPU frame capture |

**Codegen (from `ffai-kernels-codegen`):**

| Type | Purpose |
|---|---|
| `MslGenerator` | Converts kernel IR to Metal Shading Language |
| `TileSchedule` | Codegen tile schedule configuration |
| `CodegenError` | Codegen error type (aliased from `ffai_kernels_codegen::Error`) |

| Function | Purpose |
|---|---|
| `generator_for_mode(KernelMode)` | Select the right MSL generator for a kernel mode |

**DSL function stubs (panic if called outside `#[kernel]`):**

| Function | Purpose |
|---|---|
| `program_id::<AXIS>()` | Current thread/program ID along a grid axis |
| `load(tensor[idx])` | Load a value from a tensor index expression |
| `store(tensor[idx], value)` | Store a value into a tensor index expression |
| `dot(a, b)` | Dot product placeholder for tiled kernels |

**Unary math (recognized by the body parser):**
`exp`, `log`, `sqrt`, `rsqrt`, `abs`, `silu`, `gelu`, `relu`, `tanh`, `sigmoid`, `sin`, `cos`, `ceil`, `floor`, `recip`

### Re-exports

Directly accessible from `ffai-kernels::`:

| Path | What it re-exports |
|---|---|
| `ffai-kernels::kernel` | `#[kernel]` proc-macro attribute |
| `ffai-kernels::bench` / `ffai-kernels::test_kernel` | `#[bench]` / `#[test_kernel]` proc-macro attributes |
| `ffai-kernels::constexpr` | `#[constexpr]` proc-macro attribute |
| `ffai-kernels::scalar` | `#[scalar]` proc-macro attribute |
| `ffai-kernels::strided` | `#[strided]` proc-macro attribute |
| `ffai-kernels::shape` | `shape!` proc-macro |
| `ffai-kernels::tile` | `tile!` proc-macro |
| `ffai-kernels::codegen` | `ffai_kernels_codegen` crate (MSL generator, optimization passes) |
| `ffai-kernels::CodegenError` | `ffai_kernels_codegen::error::Error` |
| `ffai-kernels::core` | `ffai_kernels_core` crate (IR, DType, Shape) |
| `ffai-kernels::Context` | `ffai_kernels_runtime::Context` — GPU device + command queue |
| `ffai-kernels::DispatchResult` | `ffai_kernels_runtime::DispatchResult` — output buffers after a kernel run |
| `ffai-kernels::FFAIError` | `ffai_kernels_runtime::FFAIError` — top-level runtime error |
| `ffai-kernels::Tensor` | `prelude::Tensor` — placeholder tensor type |
| `ffai-kernels::VERSION` | Crate version string constant |
| `ffai-kernels::version()` | Returns `VERSION` |

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `ffai-kernels-core` | Re-exported as `ffai-kernels::core`; provides IR types and DType for the prelude |
| `ffai-kernels-macros` | Re-exported as individual proc macros (`kernel`, `constexpr`, `scalar`, `strided`, `shape`, `tile`) |
| `ffai-kernels-codegen` | Re-exported as `ffai-kernels::codegen`; provides MSL generation for inspection |
| `ffai-kernels-runtime` | Re-exported as `Context`, `DispatchResult`, `FFAIError`; provides GPU dispatch |

### External

None — all external dependencies come transitively through the internal crates.

## MSRV / platform

The facade crate itself has no platform gating. The runtime (`Context::new()`)
requires macOS + Metal; codegen and IR introspection work on any host.

Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New re-export:** `src/lib.rs` — add `pub use ffai_kernels_<crate>::<Item>;` with a doc comment.
- **New prelude item:** `src/prelude.rs` — add the type stub, function stub, or re-export, with a doc comment.
- **New DSL intrinsic:** `src/prelude.rs` — add a `pub fn` stub that panics, then add recognition in `ffai-kernels-macros/src/body_parser.rs`.
- **Tests to update:** Doc-tests in `src/lib.rs`.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`ffai-kernels-core` README](../ffai-kernels-core/README.md) — the IR types prelude re-exports
- [`ffai-kernels-macros` README](../ffai-kernels-macros/README.md) — how `#[kernel]` transforms your function
- [`ffai-kernels-codegen` README](../ffai-kernels-codegen/README.md) — the MSL generator behind `ffai-kernels::codegen`
- [`ffai-kernels-runtime` README](../ffai-kernels-runtime/README.md) — the `Context` and dispatch lifecycle
- [Crate docs on docs.rs](https://docs.rs/ffai-kernels)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).

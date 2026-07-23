# wh-iron

Rust DSL for writing GPU kernels — write once, run fast. This is the user-facing
facade crate: add `wh-iron` to your `Cargo.toml`, import `wh-iron::prelude::*`,
annotate functions with `#[kernel]`, and dispatch them on the GPU with a few lines
of Rust. Metal (Apple Silicon) is the default and most mature backend; CUDA, HIP,
and Vulkan are feature-gated in the lower crates.

The crate re-exports the compiler, runtime, and macro crates under one namespace so
you never need to depend on `wh-iron-core`, `wh-iron-codegen`, or the others
directly unless you are writing tooling or compiler extensions. Beyond the
re-exports it hosts `harness/` (the `#[kernel]` / `#[bench]` / `#[test_kernel]`
registries) and `runner/` (the `__iron_runner` engine — `iron` runs GPU work in a
spawned subprocess that streams results back as `ProtocolMessage` JSON lines).

## Position in the pipeline

```
        ┌──────────────────────────────┐
        │  wh-iron (this crate)       │
        │  user-facing facade           │
        │                              │
        │  use wh-iron::prelude::*;   │
        │  #[kernel]                    │
        │  kernel::launch(&ctx)         │
        └──────────┬───────────────────┘
                   │ re-exports
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
wh-iron-core  wh-iron-macros  wh-iron-codegen  wh-iron-runtime
   (IR types)    (#[kernel])      (MSL lowering)     (GPU dispatch)
```

`wh-iron` is the only crate end users depend on. It re-exports the DSL macros,
placeholder types, IR/codegen modules, and runtime entry points under flat paths
like `wh-iron::kernel`, `wh-iron::core`, `wh-iron::codegen`, and
`wh-iron::Context`.

## Quick start

```rust
use wh-iron::prelude::*;

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
use wh-iron::codegen::msl::MslGenerator;

let msl = MslGenerator::default().generate(&vector_add::kernel_ir())?;
println!("{msl}");
```

## Crate contents

| Module | Purpose |
|---|---|
| `prelude` | Comprehensive single import: `#[kernel]` DSL stubs, all IR types, all macros, runtime bindings, and codegen entry points — re-exports **everything** public from all sub-crates |
| `codegen` | Re-export of `wh_iron_codegen` — MSL generation and optimization passes |
| `core` | Re-export of `wh_iron_core` — IR types, DType, Shape, ConstExpr |

## API reference

### Prelude

`use wh-iron::prelude::*;` brings **everything public from all sub-crates** into scope:

**Macros (from `wh-iron-macros`):**

| Macro | Kind | What it does |
|---|---|---|
| `#[kernel]` | attribute | Transforms a Rust function into IR + host-side `LaunchBuilder` |
| `#[bench]` | attribute | Registers a `BenchSetup`-returning fn for `iron bench` via `inventory::submit!` (separate attribute; the bench name is the function name) |
| `#[test_kernel]` | attribute | Registers a `TestSetup`-returning fn for `iron test` |
| `#[constexpr]` | attribute | Marks a kernel parameter as a compile-time constant |
| `#[scalar]` | attribute | Marks a `Tensor` parameter for `constant T&` lowering in MSL |
| `#[strided]` | attribute | Marks a `Tensor` parameter for strided lowering (shape + stride arrays emitted) |
| `shape!(…)` | function-like | Constructs a `Shape` from dimension expressions |
| `tile!(…)` | function-like | Constructs a 2D tile shape |

**IR types (from `wh-iron-core`):**

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

**Runtime (from `wh-iron-runtime`):**

| Type | Purpose |
|---|---|
| `Context` | Metal GPU device and command queue |
| `DispatchResult` | Output buffers after a kernel run |
| `DispatchSpec` | Input buffer spec for the dispatch pipeline |
| `ResidentBuffer` | A Metal buffer managed by the context |
| `IronError` | Top-level runtime error |

| Function | Purpose |
|---|---|
| `start_gpu_trace()` | Start Xcode GPU frame capture |
| `stop_gpu_trace()` | Stop Xcode GPU frame capture |

**Codegen (from `wh-iron-codegen`):**

| Type | Purpose |
|---|---|
| `MslGenerator` | Converts kernel IR to Metal Shading Language |
| `TileSchedule` | Codegen tile schedule configuration |
| `CodegenError` | Codegen error type (aliased from `wh_iron_codegen::Error`) |

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

Directly accessible from `wh-iron::`:

| Path | What it re-exports |
|---|---|
| `wh-iron::kernel` | `#[kernel]` proc-macro attribute |
| `wh-iron::bench` / `wh-iron::test_kernel` | `#[bench]` / `#[test_kernel]` proc-macro attributes |
| `wh-iron::constexpr` | `#[constexpr]` proc-macro attribute |
| `wh-iron::scalar` | `#[scalar]` proc-macro attribute |
| `wh-iron::strided` | `#[strided]` proc-macro attribute |
| `wh-iron::shape` | `shape!` proc-macro |
| `wh-iron::tile` | `tile!` proc-macro |
| `wh-iron::codegen` | `wh_iron_codegen` crate (MSL generator, optimization passes) |
| `wh-iron::CodegenError` | `wh_iron_codegen::error::Error` |
| `wh-iron::core` | `wh_iron_core` crate (IR, DType, Shape) |
| `wh-iron::Context` | `wh_iron_runtime::Context` — GPU device + command queue |
| `wh-iron::DispatchResult` | `wh_iron_runtime::DispatchResult` — output buffers after a kernel run |
| `wh-iron::IronError` | `wh_iron_runtime::IronError` — top-level runtime error |
| `wh-iron::Tensor` | `prelude::Tensor` — placeholder tensor type |
| `wh-iron::VERSION` | Crate version string constant |
| `wh-iron::version()` | Returns `VERSION` |

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `wh-iron-core` | Re-exported as `wh-iron::core`; provides IR types and DType for the prelude |
| `wh-iron-macros` | Re-exported as individual proc macros (`kernel`, `constexpr`, `scalar`, `strided`, `shape`, `tile`) |
| `wh-iron-codegen` | Re-exported as `wh-iron::codegen`; provides MSL generation for inspection |
| `wh-iron-runtime` | Re-exported as `Context`, `DispatchResult`, `IronError`; provides GPU dispatch |

### External

None — all external dependencies come transitively through the internal crates.

## MSRV / platform

The facade crate itself has no platform gating. The runtime (`Context::new()`)
requires macOS + Metal; codegen and IR introspection work on any host.

Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New re-export:** `src/lib.rs` — add `pub use wh_iron_<crate>::<Item>;` with a doc comment.
- **New prelude item:** `src/prelude.rs` — add the type stub, function stub, or re-export, with a doc comment.
- **New DSL intrinsic:** `src/prelude.rs` — add a `pub fn` stub that panics, then add recognition in `wh-iron-macros/src/body_parser.rs`.
- **Tests to update:** Doc-tests in `src/lib.rs`.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`wh-iron-core` README](../wh-iron-core/README.md) — the IR types prelude re-exports
- [`wh-iron-macros` README](../wh-iron-macros/README.md) — how `#[kernel]` transforms your function
- [`wh-iron-codegen` README](../wh-iron-codegen/README.md) — the MSL generator behind `wh-iron::codegen`
- [`wh-iron-runtime` README](../wh-iron-runtime/README.md) — the `Context` and dispatch lifecycle
- [Crate docs on docs.rs](https://docs.rs/wh-iron)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).

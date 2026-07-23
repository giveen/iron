# Acknowledgements

As with all open source, Iron stands on the work of others, we want to formally acknowledge that work here.

## Individual Contributors

If you wish to be acknowledged for your contributions, please list your name
with a short description of your contribution(s) below. For example:

- Jane Smith: Added the `foo` and `bar` ops.

Iron Kernels was developed with contributions from the following individuals:

- Tom Turney (@TheTom): Contributed many kernels, including Flash-Attention-2 SDPA, simdgroup-matrix quantised GEMM/GEMV, MoE + gated-delta-network kernels, sampling + logits processors, fused activations, ICB `_record` codegen, and kernel-pack infrastructure. He spearheaded adding GPU correctness tests and generally boosting overall test and CI infrastructure for improved regression testing.
- Eric Kryski (@ekryski): Contributed most of the kernels, including int4, int8, bf16 and fp16 completeness, AURA, image and audio kernels. Additionally, helped improve CI, added the initial documentation and performed and catalogued the kernel audit against upstream sources with verification against real model workloads.

<a href="https://github.com/thewafflehaus/iron/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thewafflehaus/iron&anon=0&columns=20&max=100&r=true" />
</a>

## Special Acknowledgements
Iron Kernels started out as a collaboration on the [MetalTile](https://github.com/0xClandestine/metaltile) project. We want to acknowledge that [@0xClandestine](https://github.com/0xClandestine) had the initial idea of the Rust DSL → Metal MSL and put together the first proof of concept. In addition they made significant contributions to what has now evolved to be the Rust kernel DSL used today. Many thanks to [@0xClandestine](https://github.com/0xClandestine).

In addition, we would like to acknowledge [@Ambisphaeric](https://github.com/Ambisphaeric) for their initial PM efforts, testing, bench running and some kernel tuning efforts while the project was called MetalTile.

Finally we would like to acknowledge the efforts of the following people that we have seen make incredible contributions to pushing open source AI forward:

- [@spiritbuun](https://github.com/spiritbuun)
- [@bstnxbt](https://github.com/bstnxbt)
- [@Blaizzy](https://github.com/Blaizzy)
- [@lucasnewman](https://github.com/lucasnewman)
- [@beshkenadze](https://github.com/beshkenadze)

Give them all a follow!

## Prior Art
When we were targeting Metal, Iron Kernels' benchmark suite and kernel library initially stood on the shoulders of the MLX ecosystem. As we now support multiple GPU platform targets, multiple models and have done further bug fixes, performance tuning, and optimizations it has diverged significantly. Nevertheless, a portion of the `wh-iron-std` kernels are ports, re-implementations or improvements of kernels from the following projects:

- [**ekryski/mlx**](https://github.com/ekryski/mlx) (`alpha`) — primary source for reference implementation of optimized kernels, including Iron extensions: gated-delta, SSM replay, AURA codec.
- [**ekryski/mlx-audio-swift**](https://github.com/ekryski/mlx-audio-swift) (`alpha`) — primary source for reference implementation of optimized kernels for audio models.
- [**ekryski/mlx-swift-lm**](https://github.com/ekryski/mlx-swift-lm) (`alpha`) — primary source for reference implementation of optimized kernels for speculative decoding, batch decoding, prefill caching.
- [**TheTom/turboquant_plus**](https://github.com/TheTom/turboquant_plus) - implementation ideas and benchmarking for turboquant KV compression kernels, long context performance improvements and general llama.cpp model performance comparisons.
- [**bstnxbt/dflash-mlx**](https://github.com/bstnxbt/dflash-mlx) - implementation ideas and benchmarking reference for draft model speculative decoding techniques.
- [**ml-explore/mlx**](https://github.com/ml-explore/mlx) — primary source for mainstream kernels for benchmarking against
- [**ml-explore/mlx-lm**](https://github.com/ml-explore/mlx-lm) — reference for GatedDeltaNet step semantics and benchmarking against vlm kernels.
- [**Blaizzy/mlx-audio-swift**](https://github.com/Blaizzy/mlx-audio-swift) — reference for benchmarking against audio kernels.

We are grateful to the MLX team at Apple and the broader MLX community for their work in pushing local AI on Apple Silicon forward.

After starting on this project we became aware of [cuda-oxide](https://github.com/NVlabs/cuda-oxide) from NVIDIA's labs. While similar in concept, we started with Metal output and quickly surpassed the breadth and depth of kernels implemented in cuda-oxide. After having higher performance kernels than MLX on Apple Silicon across various model architectures and workloads, we modified the DSL to support CUDA, AMD and VULKAN shader code surpassing what was available in any other project at the time. In addition, we feel, humbly, that our implementation is simpler and has less moving parts. Regardless, we wanted to acknowledge prior art we became aware of and would also like to thank NVIDIA for all they have done to push the AI frontier forward.

## Third-Party Software

Iron Kernels leverages several third-party libraries. Their repositories and licenses are listed below.

- **objc2 / objc2-metal / objc2-foundation** — Safe Rust bindings to Apple's Objective-C runtime, Metal GPU API, and Foundation frameworks. Used for all Metal device, command queue, buffer, and pipeline state objects. [MIT](https://github.com/madsmtm/objc2)
- **syn / quote / proc-macro2** — Proc-macro parsing and token-stream generation, used by all Iron Kernels macros. [MIT / Apache-2.0](https://github.com/dtolnay/syn)
- **clap** — Command-line argument parsing for the `iron` binary. [MIT / Apache-2.0](https://github.com/clap-rs/clap)
- **half** — Host-side `f16` and `bf16` types used in bench and buffer utilities. [MIT / Apache-2.0](https://github.com/starkat99/half-rs)
- **bytemuck** — Safe byte-reinterpretation for buffer uploads and downloads. [MIT / Apache-2.0 / Zlib](https://github.com/Lokathor/bytemuck)
- **inventory** — Compile-time kernel registry powering `iron build` and `#[kernel(bench(...))]`. [MIT / Apache-2.0](https://github.com/dtolnay/inventory)
- **serde / serde_json** — Serialisation for bench snapshots, baseline files, and IR manifests. [MIT / Apache-2.0](https://github.com/serde-rs/serde)
- **tracing / tracing-subscriber** — Structured diagnostics and event logging. [MIT](https://github.com/tokio-rs/tracing)
- **thiserror** — Error type derivation across Iron Kernels crates. [MIT / Apache-2.0](https://github.com/dtolnay/thiserror)
- **anstyle / anstream** — ANSI terminal colour output in the `iron` CLI. [MIT / Apache-2.0](https://github.com/rust-cli/anstyle)
- **smallvec** — Inline-allocated small vectors used in hot IR paths. [MIT / Apache-2.0](https://github.com/servo/rust-smallvec)
- **rustc-hash** — Fast non-cryptographic hashing for IR maps and pass data structures. [MIT / Apache-2.0](https://github.com/rust-lang/rustc-hash)

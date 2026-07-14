# Individual Contributors

If you wish to be acknowledged for your contributions, please list your name
with a short description of your contribution(s) below. For example:

- Jane Smith: Added the `foo` and `bar` ops.

FFAI Kernels was developed with contributions from the following individuals:

- Tom Turney (@TheTom): Contributed many kernels, including Flash-Attention-2 SDPA, simdgroup-matrix quantised GEMM/GEMV, MoE + gated-delta-network kernels, sampling + logits processors, fused activations, ICB `_record` codegen, and kernel-pack infrastructure. He spearheaded adding GPU correctness tests and generally boosting overall test and CI infrastructure for improved regression testing.
- Eric Kryski (@ekryski): Contributed most of the kernels, including int4, int8, bf16 and fp16 completeness, AURA, image and audio kernels. Additionally, helped improve CI, added the initial documentation and performed and catalogued the kernel audit against upstream sources with verification against real model workloads.

<a href="https://github.com/thewafflehaus/ffai-kernels/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thewafflehaus/ffai-kernels&anon=0&columns=20&max=100&r=true" />
</a>

## Special Acknowledgements
FFAI Kernels started out as a collaboration on the [MetalTile](https://github.com/0xClandestine/metaltile) project. We want to acknowledge that [@0xClandestine](https://github.com/0xClandestine) had the initial idea of the Rust DSL → Metal MSL and put together the first proof of concept. In addition they made significant contributions to what has now evolved to be the Rust kernel DSL used today. Many thanks to [@0xClandestine](https://github.com/0xClandestine).

In addition, we would like to acknowledge [@Ambisphaeric](https://github.com/Ambisphaeric) for their initial PM efforts, testing, bench running and some kernel tuning efforts while the project was called MetalTile.

Give them both a follow!

### Third-Party Software

FFAI Kernels leverages several third-party libraries. Their repositories and licenses are listed below.

- **objc2 / objc2-metal / objc2-foundation** — Safe Rust bindings to Apple's Objective-C runtime, Metal GPU API, and Foundation frameworks. Used for all Metal device, command queue, buffer, and pipeline state objects. [MIT](https://github.com/madsmtm/objc2)
- **syn / quote / proc-macro2** — Proc-macro parsing and token-stream generation, used by all FFAI Kernels macros. [MIT / Apache-2.0](https://github.com/dtolnay/syn)
- **clap** — Command-line argument parsing for the `ffaik` binary. [MIT / Apache-2.0](https://github.com/clap-rs/clap)
- **half** — Host-side `f16` and `bf16` types used in bench and buffer utilities. [MIT / Apache-2.0](https://github.com/starkat99/half-rs)
- **bytemuck** — Safe byte-reinterpretation for buffer uploads and downloads. [MIT / Apache-2.0 / Zlib](https://github.com/Lokathor/bytemuck)
- **inventory** — Compile-time kernel registry powering `ffaik build` and `#[kernel(bench(...))]`. [MIT / Apache-2.0](https://github.com/dtolnay/inventory)
- **serde / serde_json** — Serialisation for bench snapshots, baseline files, and IR manifests. [MIT / Apache-2.0](https://github.com/serde-rs/serde)
- **tracing / tracing-subscriber** — Structured diagnostics and event logging. [MIT](https://github.com/tokio-rs/tracing)
- **thiserror** — Error type derivation across FFAI Kernels crates. [MIT / Apache-2.0](https://github.com/dtolnay/thiserror)
- **anstyle / anstream** — ANSI terminal colour output in the `ffaik` CLI. [MIT / Apache-2.0](https://github.com/rust-cli/anstyle)
- **smallvec** — Inline-allocated small vectors used in hot IR paths. [MIT / Apache-2.0](https://github.com/servo/rust-smallvec)
- **rustc-hash** — Fast non-cryptographic hashing for IR maps and pass data structures. [MIT / Apache-2.0](https://github.com/rust-lang/rustc-hash)

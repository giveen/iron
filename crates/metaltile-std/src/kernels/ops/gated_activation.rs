//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gated MLP activations — `activation(gate) * up`.
//!
//! The SwiGLU-family fused activation used in modern transformer MLPs: given
//! the two halves of an MLP's gate/up projections (`w_gate·x`, `w_up·x`), apply
//! an activation to the gate and multiply by up in ONE elementwise pass — saves
//! a full-tensor read-modify-write versus separate `silu` + `mul` launches (the
//! intermediate stays in registers). One file covers the variants:
//!
//!   - `mt_swiglu` — `silu(gate) · up` (Llama/Qwen/Gemma/Mistral)
//!   - `mt_clamped_swiglu` — `clip(silu(gate), L) · clip(up, ±L)`; a per-layer
//!     activation-clip limit (gpt-oss, StepFun Step-3). `limit <= 0` ⇒ plain SwiGLU
//!   - `mt_fused_gate_gelu` — `gelu_approx(gate) · up`
//!   - `mt_fused_gate_clipped_swiglu` — GPT-OSS clipped: clamp `±7`,
//!     `sigmoid(1.702·g)` gate, `+1` up bias
//!
//! All are one-thread-per-output elementwise kernels computed in f32. The plain
//! ungated `silu` activation lives in `ops/unary.rs` (`mt_silu`); `mt_swiglu`
//! cross-kernel-calls it (inlined at codegen by `KernelInlinePass`).
//!
//! MLX reference: `mx.fast.swiglu` + `fused_gate_activation.metal`
//! (`apply_gate<activation_type ∈ {0,1,2}>`; type 0 = silu = `mt_swiglu`).

use metaltile::kernel;

// ── SwiGLU: silu(gate) · up ───────────────────────────────────────────────────

#[kernel]
pub fn mt_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // Cross-kernel call: KernelInlinePass splices mt_silu's scalar body here.
    // mt_silu's input-param load is replaced by g (already f32), so silu runs in
    // f32. Future fusion passes can identify the (silu, mul) → swiglu pattern.
    let s = mt_silu(g);
    store(out[idx], (s * u).cast::<T>());
}

// ── Clamped SwiGLU: clip(silu(gate), L) · clip(up, ±L) ────────────────────────

// Bare `#[kernel]` with a runtime `limit: f32` constexpr. `limit <= 0` collapses
// to plain SwiGLU (the per-layer dispatch reaches for this only on marked layers).
#[kernel]
pub fn mt_clamped_swiglu<T>(
    gate: Tensor<T>,
    up: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] limit: f32,
) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // silu(g) = g * sigmoid(g). Free-function `exp` + f32 literals so the binding
    // isn't elided (the method form `(-g).exp()` nested in a larger expr drops
    // out of codegen — see moe_router_sqrtsoftplus).
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - g));
    let s_raw = g * sig;
    // Clip via `select`, not `min`/`max`: the DSL's min/max overloads are
    // ambiguous on mixed int/float operands. silu's upper tail is clipped
    // one-sided; `up` two-sided.
    let active = limit > 0.0f32;
    let neg = 0.0f32 - limit;
    let s_clipped = select(active, select(s_raw > limit, limit, s_raw), s_raw);
    let u_hi = select(u > limit, limit, u);
    let u_lo = select(u_hi < neg, neg, u_hi);
    let u_clipped = select(active, u_lo, u);
    store(out[idx], (s_clipped * u_clipped).cast::<T>());
}

// ── Fused gate-GELU: gelu_approx(gate) · up ───────────────────────────────────

/// GELU via the tanh approximation MLX uses — matches `gelu_approx_act` in
/// `fused_gate_activation.metal`. Computed in f32 so the cubic + tanh keep
/// precision regardless of `T`.
#[kernel]
pub fn mt_fused_gate_gelu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // gelu_approx(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    let x3 = g * g * g;
    let inner = 0.7978845608f32 * (g + 0.044715f32 * x3);
    let act = 0.5f32 * g * (1.0f32 + tanh(inner));
    store(out[idx], (act * u).cast::<T>());
}

// ── Fused gate clipped-SwiGLU (GPT-OSS): clamp ±7, sigmoid(1.702·g), +1 up ─────

/// The GPT-OSS variant: both halves clamped to `[-7, 7]`; gate uses
/// `sigmoid(1.702·g)`; up carries a `+1` bias: `g·sigmoid(1.702·g)·(u + 1)`.
/// Matches `clipped_swiglu` in `fused_gate_activation.metal`. Clamp is composed
/// from two `select`s (the DSL has no `clamp` builtin).
#[kernel]
pub fn mt_fused_gate_clipped_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    let g_raw = load(gate[idx]).cast::<f32>();
    let u_raw = load(up[idx]).cast::<f32>();
    let g_hi = select(g_raw > 7.0f32, 7.0f32, g_raw);
    let g = select(g_hi < (0.0f32 - 7.0f32), 0.0f32 - 7.0f32, g_hi);
    let u_hi = select(u_raw > 7.0f32, 7.0f32, u_raw);
    let u = select(u_hi < (0.0f32 - 7.0f32), 0.0f32 - 7.0f32, u_hi);
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - 1.702f32 * g));
    let act = g * sig * (u + 1.0f32);
    store(out[idx], act.cast::<T>());
}

/// Correctness for the gated activations. Oracles mirror each kernel exactly on
/// dtype-rounded inputs, computed in f32.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{mt_clamped_swiglu, mt_fused_gate_clipped_swiglu, mt_fused_gate_gelu, mt_swiglu};
    use crate::utils::{pack_f32, unpack_f32};

    // ── mt_swiglu ─────────────────────────────────────────────────────────────

    fn swiglu_setup(n: usize, dt: DType) -> TestSetup {
        let gate: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.2 - 1.0).collect();
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let u_dt = unpack_f32(&pack_f32(&up, dt), dt);
        let expected: Vec<f32> =
            g_dt.iter().zip(&u_dt).map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();
        TestSetup::new(mt_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(&up, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_swiglu(dt: DType) -> TestSetup { swiglu_setup(1024, dt) }

    // ── mt_clamped_swiglu ─────────────────────────────────────────────────────

    fn clamped_setup(n: usize, limit: f32, dt: DType) -> TestSetup {
        let gate: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.2 - 1.0).collect();
        let g_dt = unpack_f32(&pack_f32(&gate, dt), dt);
        let u_dt = unpack_f32(&pack_f32(&up, dt), dt);
        let expected: Vec<f32> = g_dt
            .iter()
            .zip(&u_dt)
            .map(|(&g, &u)| {
                let s = g / (1.0 + (-g).exp()); // silu(g) = g * sigmoid(g)
                let (s_c, u_c) =
                    if limit > 0.0 { (s.min(limit), u.max(-limit).min(limit)) } else { (s, u) };
                s_c * u_c
            })
            .collect();
        TestSetup::new(mt_clamped_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(&up, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("limit", limit)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_clamped_swiglu_active(dt: DType) -> TestSetup { clamped_setup(1024, 7.0, dt) }

    /// `limit == 0` collapses to plain SwiGLU — equivalence with `mt_swiglu` is
    /// the invariant the per-layer dispatch ships on.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_clamped_swiglu_zero_limit_equals_plain(dt: DType) -> TestSetup {
        clamped_setup(1024, 0.0, dt)
    }

    // ── mt_fused_gate_{gelu,clipped_swiglu} ───────────────────────────────────

    fn inputs(n: usize, dt: DType) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Range spans beyond +/-7 so clipped_swiglu's clamp is exercised.
        let gate: Vec<f32> = (0..n).map(|i| (i % 17) as f32 - 8.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i % 13) as f32 - 6.0).collect();
        let g = unpack_f32(&pack_f32(&gate, dt), dt);
        let u = unpack_f32(&pack_f32(&up, dt), dt);
        (gate, up, g, u)
    }

    fn build(
        kernel: metaltile::core::ir::Kernel,
        gate: &[f32],
        up: &[f32],
        expected: &[f32],
        dt: DType,
    ) -> TestSetup {
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("gate", pack_f32(gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack_f32(up, dt), dt))
            .input(TestBuffer::zeros("out", gate.len(), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(expected, dt), dt))
            .grid_1d(gate.len(), 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_fused_gate_gelu(dt: DType) -> TestSetup {
        let (gate, up, g, u) = inputs(512, dt);
        const C: f32 = 0.797_884_6; // sqrt(2/pi)
        let expected: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| 0.5 * g * (1.0 + (C * (g + 0.044715 * g * g * g)).tanh()) * u)
            .collect();
        build(mt_fused_gate_gelu::kernel_ir_for(dt), &gate, &up, &expected, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_fused_gate_clipped_swiglu(dt: DType) -> TestSetup {
        let (gate, up, g, u) = inputs(512, dt);
        let expected: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| {
                let g = g.clamp(-7.0, 7.0);
                let u = u.clamp(-7.0, 7.0);
                g * (1.0 / (1.0 + (-1.702 * g).exp())) * (u + 1.0)
            })
            .collect();
        build(mt_fused_gate_clipped_swiglu::kernel_ir_for(dt), &gate, &up, &expected, dt)
    }
}

/// Benchmarks for the gated activations (1D elementwise, 3 streams).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{mt_clamped_swiglu, mt_fused_gate_clipped_swiglu, mt_fused_gate_gelu, mt_swiglu};

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_swiglu(dt: DType) -> BenchSetup {
        let n = 1024 * 1024usize;
        BenchSetup::new(mt_swiglu::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_clamped_swiglu(dt: DType) -> BenchSetup {
        let n = 1024 * 1024usize;
        BenchSetup::new(mt_clamped_swiglu::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("limit", 7.0f32)
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    fn fb(kernel: metaltile::core::ir::Kernel, dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(kernel)
            .buffer(BenchBuffer::random("gate", n, dt))
            .buffer(BenchBuffer::random("up", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gelu(dt: DType) -> BenchSetup { fb(mt_fused_gate_gelu::kernel_ir_for(dt), dt) }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_clipped(dt: DType) -> BenchSetup {
        fb(mt_fused_gate_clipped_swiglu::kernel_ir_for(dt), dt)
    }
}

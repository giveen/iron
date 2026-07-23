//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! MoE router pre-score (sigmoid + correction-bias), kept on-device so the
//! whole router (pre-score -> top-k -> gather) avoids a host round-trip.
//!
//! Candidate score producer for **Hy3** (`hy_v3`: `moe_router_use_sigmoid`,
//! `moe_router_enable_expert_bias`) when the reference path is
//! "top-k by biased, weight by unbiased" (NemotronH / DSv3-noaux style).
//! Confirm against Hy3 golden logits before Iron wires this path exclusively
//! (see `docs/specs/HY3_MOE_PREP.md`).

use wh_iron::kernel;

/// MoE router pre-scores (NemotronH / DeepSeek-V3 noaux, sigmoid variant):
/// `unbiased[i] = sigmoid(logit[i])`, `biased[i] = unbiased[i] + e_score_correction_bias[i]`.
/// Feeds `iron_moe_router_topk_biased` (top-k by biased, weights from unbiased) so the whole
/// router stays ON-DEVICE — no per-MoE-layer dl(gate)+host-topk+up(idx) sync round-trip.
#[kernel]
pub fn iron_moe_sigmoid_bias(
    logits: Tensor<f32>,
    bias: Tensor<f32>,
    mut unbiased: Tensor<f32>,
    mut biased: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let s = 1.0f32 / (1.0f32 + exp(0.0f32 - load(logits[i])));
        store(unbiased[i], s);
        store(biased[i], s + load(bias[i]));
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_moe_sigmoid_bias;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n_experts: usize) -> TestSetup {
        let dt = DType::F32;
        let logits: Vec<f32> = (0..n_experts).map(|i| (i % 31) as f32 * 0.1 - 1.5).collect();
        let bias: Vec<f32> = (0..n_experts).map(|i| (i % 7) as f32 * 0.05 - 0.15).collect();
        let l_dt = unpack_f32(&pack_f32(&logits, dt), dt);
        let b_dt = unpack_f32(&pack_f32(&bias, dt), dt);
        let unbiased: Vec<f32> = l_dt.iter().map(|&l| 1.0_f32 / (1.0 + (-l).exp())).collect();
        let biased: Vec<f32> = unbiased.iter().zip(&b_dt).map(|(&u, &b)| u + b).collect();
        TestSetup::new(iron_moe_sigmoid_bias::kernel_ir())
            .input(TestBuffer::from_vec("logits", pack_f32(&logits, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("unbiased", n_experts, dt))
            .input(TestBuffer::zeros("biased", n_experts, dt))
            .constexpr("n", n_experts as u32)
            .expect(TestBuffer::from_vec("unbiased", pack_f32(&unbiased, dt), dt))
            .expect(TestBuffer::from_vec("biased", pack_f32(&biased, dt), dt))
            .grid_1d(n_experts, 64)
    }

    /// DeepSeek-V3 / NemotronH expert count.
    #[test_kernel(dtypes = [f32], tol = [2e-4])]
    fn test_moe_sigmoid_bias_dsv3(_dt: DType) -> TestSetup { setup(256) }

    /// Hy3 / hy_v3 — 192 routed experts.
    #[test_kernel(dtypes = [f32], tol = [2e-4])]
    fn test_moe_sigmoid_bias_hy3(_dt: DType) -> TestSetup { setup(192) }
}

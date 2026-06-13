//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MoE router pre-score (sigmoid + correction-bias), kept on-device so the
//! whole router (pre-score -> top-k -> gather) avoids a host round-trip.

use metaltile::kernel;

/// MoE router pre-scores (NemotronH / DeepSeek-V3 noaux, sigmoid variant):
/// `unbiased[i] = sigmoid(logit[i])`, `biased[i] = unbiased[i] + e_score_correction_bias[i]`.
/// Feeds `mt_dsv4_router_topk` (top-k by biased, weights from unbiased) so the whole
/// router stays ON-DEVICE — no per-MoE-layer dl(gate)+host-topk+up(idx) sync round-trip.
#[kernel]
pub fn mt_moe_sigmoid_bias(
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

//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Scaled-dot-product attention — the sdpa family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): the bidirectional / decode /
//! multi / prefill paths across head-dims (d64..d512) and patterns (relpos,
//! windowed, conformer, 2-pass, batched, sink), the quantized + block-scaled
//! flash forms, the AURA compressed-domain attention, and the DSv4 compressed
//! sparse-attention (CSA) decode + Lightning Indexer. Migrated from `mlx/` +
//! `ffai/`; model names dropped (dsv4 → csa/indexer/compressor).

pub mod attn_head_gate;
pub mod aura_flash_p1;
pub mod aura_flash_pass2;
pub mod aura_flash_sdpa;
pub mod aura_score;
pub mod aura_value;
pub mod compressor_pool;
pub mod csa_sdpa_decode;
pub mod flash_block_scaled_sdpa;
pub mod flash_quantized_sdpa;
pub mod indexer_score;
pub mod indexer_topk;
pub mod scaled_dot_product_attention;
pub mod sdpa_bidirectional;
pub mod sdpa_bidirectional_d128_relpos;
pub mod sdpa_bidirectional_windowed;
pub mod sdpa_decode;
pub mod sdpa_decode_2pass;
pub mod sdpa_decode_batched;
pub mod sdpa_decode_d512_sink;
pub mod sdpa_decode_sink_buf;
pub mod sdpa_multi;
pub mod sdpa_multi_d256;
pub mod sdpa_prefill_d512_sink;
pub mod sdpa_rel_pos_conformer;
pub mod sdpa_vector;
pub mod steel_attn;

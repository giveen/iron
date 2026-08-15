// Hand-written dispatch table (F-85 round-6 marlin revival; round-12 added
// group_blocks==4/8). Unlike the upstream `generate_kernels.py`-emitted
// selector, this covers only the config this runtime actually instantiates:
// symmetric u4 (kU4B8, GPTQ-style bias-8 encoding) activations f16, weight
// scales f16, group_size in {32,64,128} (group_blocks in {2,4,8} — 32
// matches this codebase's own Q4_K sub-block granularity after the
// symmetric requant bridge, see marlin_shim.cu; 64/128 are round-12's
// bytes-per-token levers, same requant bridge parameterized on group_size),
// M<=8 (m_block_size_8=true, thread_m_blocks=1) — the small-M spec-decode
// verify GEMM shape. The three (threads, thread_n_blocks, thread_k_blocks)
// tuples below are exactly `small_batch_thread_configs` from marlin_mm.cu,
// so `determine_exec_config`'s auto-tune loop always finds a match.
//
// group_blocks==4/8 entries are NOT new kernel builds — `sm80_kernel_
// float16_u4b8_float16.cu` (generate_kernels.py output, unmodified) already
// carries `Marlin<..., true, 4, 4, false>` and `Marlin<..., true, 4, 8,
// false>` template instantiations for all three thread configs (it emits
// the full upstream group_blocks sweep, this file just never dispatched to
// them). Confirmed via `marlin_template.h`'s group_blocks handling being
// generic (both the `>= thread_k_blocks` and `< thread_k_blocks` branches
// are `if constexpr`-selected, no group_blocks==2-specific code path) —
// see the round-12 report for the numerics-gate results per arm.
//
// Extend this table (and the corresponding template instantiation .cu under
// this directory) before relying on any other m_block_size_8/group_blocks
// combination — an unmatched config silently falls through to
// `MarlinDefault` (a no-op kernel) in `get_marlin_kernel`.
// DEBUG (round-6 correctness triage, temporary): non-m8 (standard 16-row)
// dispatch entries at thread_m_blocks==1, used only to A/B the m_block_size_8
// path against the mainline marlin path with the same dense-random harness.
if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 4 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 256, 1, 8, 8, false, 4, -1, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 256, 1, 8, 8, true, 4, -1, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 4 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 8, 4, true, 4, -1, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 4, 8, true, 4, -1, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 256, 1, 8, 8, true, 4, 2, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 4 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 8, 4, true, 4, 2, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 4, 8, true, 4, 2, false>;
// round-12: group_size=64 (group_blocks=4) — g=64 arm.
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 256, 1, 8, 8, true, 4, 4, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 4 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 8, 4, true, 4, 4, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 4, 8, true, 4, 4, false>;
// round-12: group_size=128 (group_blocks=8) — g=128 arm.
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>;
else if (a_type == marlin_types::kFloat16 && b_type == marlin_types::kU4B8 && c_type == marlin_types::kFloat16 && s_type == marlin_types::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 4 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<marlin_types::kFloat16.id(), marlin_types::kU4B8.id(), marlin_types::kFloat16.id(), marlin_types::kFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>;

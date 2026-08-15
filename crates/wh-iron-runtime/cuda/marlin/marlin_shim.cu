// C-ABI shim: exposes the marlin small-M W4A16 GEMM + repack to Rust FFI
// without marlin_types::ScalarType in the signature.
//
// F-85 round-6: this file replaces the earlier NVFP4-only shim (archived on
// `backup/marlin-recovery`) with the symmetric U4 (kU4B8, GPTQ-style bias-8
// encoding) f16-activation / f16-scale path used as the spec-decode verify
// GEMM. Kernel-format provenance: "marlin" and "GPTQ" name a kernel/pack
// family, not a project dependency — this file is a from-scratch instance
// of that family (marlin_mm.cu / kernel.h / marlin_template.h are unmodified
// copies of the archived csrc; the format is documented in dequant.h).
//
// *** STATUS (2026-08-15, round-7): NUMERICS FIXED, caller must call
// ffai_marlin_permute_scales() once per weight after ffai_marlin_repack().
// Round-6 left two independent bugs, both now root-caused with decisive
// evidence (compute-sanitizer + isolation probes) and fixed:
//
// 1. Harmless-in-isolation WAR shared-memory hazard (marlin_template.h,
//    fixed with one __syncthreads() before thread_block_reduce() -- see
//    that comment for detail). Confirmed via `compute-sanitizer --tool
//    racecheck`: 0 hazards post-fix vs 1 (1024 lanes) pre-fix. Verified
//    NOT to be the correctness wall on its own (numerics were bit-identical
//    before/after this fix in isolation) -- kept because it is a genuine
//    hazard, not because it explains the error.
// 2. THE actual correctness wall: b_scales was never column-permuted.
//    marlin's epilogue reads the per-output-column scale out of shared
//    memory via a lane-index-derived offset that assumes the classic
//    GPTQ-Marlin "8x8 transpose" scale-column permutation (public format
//    lore -- see marlin_permute_scales_kernel below for the exact
//    formula). Round-6 shipped weight repack but no scale-side
//    counterpart, so raw un-permuted [num_groups, size_n] scales were fed
//    straight to a kernel that silently misselects which column's scale
//    applies to which output -- correct only when the scale happens to be
//    uniform within a permuted group (why the round-6 uniform-code /
//    single-nonzero spike probes passed) or the error is masked, wrong by
//    5x-500x relative error otherwise on real per-group-varying scales,
//    with NO smooth growth vs K (a wrongly-scaled-column bug, not an
//    accumulation-precision one -- confirmed via a K=64..4096 sweep at
//    fixed M=8,N=128 that stayed uniformly large across the whole range).
//    Decisive isolation: (a) forcing all scales to one constant value
//    collapsed max_abs from ~1-3 to ~0.005-0.03 (the quant-noise floor);
//    (b) applying the marlin scale permutation to real per-group-varying
//    scales collapsed error to that same floor at every K tested.
//    `use_fp16_accum` was independently ruled out by disassembling the
//    linked kernel object: 6120/6120 MMA instructions are
//    `HMMA.16816.F32` (f32 accumulate); zero f16-accumulate variants.
//
// Gate before re-wiring into Rust FFI (butter): dense-random unit test
// green at <=1e-2 max_abs (not max_rel -- individual near-zero reference
// cells inflate relative error harmlessly; use max_abs vs ref_absmax) on
// all four target shapes, M in {1,2,3,4,5,8}.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "scalar_type.hpp"
namespace marlin_moe_wna16 {
void marlin_mm(const void*,const void*,void*,void*,void*,void*,void*,void*,void*,void*,
               void*,void*,void*,void*,void*,void*,int,int,int,bool,int,int,int,void*,
               marlin_types::ScalarType const&,marlin_types::ScalarType const&,marlin_types::ScalarType const&,marlin_types::ScalarType const&,
               bool,bool,bool,bool,int,int,int,cudaStream_t,int,int,int,int,bool,bool,bool);
}
void run_repack(const unsigned*,unsigned*,int,int,int,int,cudaStream_t);

// On-device build of marlin routing arrays from per-"expert" offsets
// (off[n_exp+1]), removing a per-call host download+loop+upload+sync.
// sorted_token_ids = per-expert contiguous [off[e]..off[e+1]) padded to
// moe_block with sentinel=mt; expert_ids per block; num_tokens_past_padded =
// total padded length. For the dense (non-MoE) verify GEMM this is called
// with n_exp=1, off={0,M}, mt=M — i.e. one "expert" owning all M rows.
__global__ void marlin_routing_kernel(const int* off, int n_exp, int blk, int mt,
                                      int* stid, int* eid, int* ntpp) {
  extern __shared__ int sh[];
  int* bpe = sh; int* bstart = sh + n_exp;
  int e = threadIdx.x;
  if (e < n_exp) { int cnt = off[e+1]-off[e]; bpe[e] = (cnt + blk - 1) / blk; }
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int i = 0; i < n_exp; i++) { bstart[i] = acc; acc += bpe[i]; }
    *ntpp = acc * blk;
  }
  __syncthreads();
  if (e < n_exp) {
    int cnt = off[e+1]-off[e]; int lo = off[e];
    for (int b = 0; b < bpe[e]; b++) {
      int bi = bstart[e] + b; eid[bi] = e;
      for (int j = 0; j < blk; j++) { int r = b*blk + j; stid[bi*blk + j] = (r < cnt) ? (lo + r) : mt; }
    }
  }
}

// F-85 round-7: the marlin tensor-core epilogue reads a per-output-column
// scale straight out of shared memory using a lane-index-derived offset
// that assumes the scale row has already been reordered into marlin's
// classic "8x8 transpose" column layout (public GPTQ-Marlin format lore,
// not project-specific -- the same permutation used by the family's
// upstream host-side `marlin_permute_scales` glue, which lives outside
// this csrc and was never ported here). Round-6 shipped weight repack
// (`ffai_marlin_repack`) but no scale-side counterpart, so raw
// [num_groups, size_n] group scales were fed straight to a kernel that
// expects them column-permuted -- silently correct whenever the scale
// value happens to be uniform across a permuted group (why the round-6
// single-nonzero / uniform-code probes passed) and silently wrong by
// 10x-500x relative error otherwise (confirmed via a CPU-side permute
// A/B: applying this exact permutation before upload drops max_abs error
// on dense-random inputs to the same ~1e-2 quantization-noise floor as
// the uniform-scale control, at every K from 64 to 4096).
//
// Permutation: for each group row, split size_n into 64-wide chunks; for
// each chunk, out[c*64 + i*8 + j] = in[c*64 + i + 8*j] for i,j in [0,8)
// (equivalently out[c*64+k] = in[c*64 + k/8 + 8*(k%8)]).
__global__ void marlin_permute_scales_kernel(const __half* in, __half* out,
                                             int num_groups, int size_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_groups * size_n;
  if (idx >= total) return;
  int g = idx / size_n;
  int n = idx % size_n;
  int c = n / 64;
  int k = n % 64;
  int src_k = (k / 8) + 8 * (k % 8);
  out[g * size_n + n] = in[g * size_n + c * 64 + src_k];
}

extern "C" {

// Repack one weight's GPTQ-packed [size_k/8, size_n] u32 weights (u4, K-major
// packing, 8 values/u32) into the marlin tile layout. num_bits=4 is hardwired
// in `run_repack` (repack_standalone.cu) — matches our u4b8 GEMM below.
void ffai_marlin_repack(const unsigned* b_q_weight, unsigned* out, int size_k, int size_n,
                        int sms, int max_shared_mem, cudaStream_t stream) {
  run_repack(b_q_weight, out, size_k, size_n, sms, max_shared_mem, stream);
}

void ffai_marlin_build_routing(const int* off, int n_exp, int blk, int mt,
                               int* stid, int* eid, int* ntpp, cudaStream_t stream) {
  marlin_routing_kernel<<<1, n_exp, 2*n_exp*sizeof(int), stream>>>(off, n_exp, blk, mt, stid, eid, ntpp);
}

// One-time (per weight-load) scale-column permutation companion to
// ffai_marlin_repack -- see marlin_permute_scales_kernel above for why
// this is required. `in`/`out` are [num_groups, size_n] f16 device
// buffers; `out` may not alias `in` (the permutation is not in-place
// safe -- each output chunk reads all 64 source columns of its chunk).
void ffai_marlin_permute_scales(const void* in, void* out, int num_groups,
                                int size_n, cudaStream_t stream) {
  int total = num_groups * size_n;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  marlin_permute_scales_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(in), reinterpret_cast<__half*>(out),
      num_groups, size_n);
}

// Small-M symmetric-u4 (kU4B8) W4A16 GEMM: A [M,K] f16 activations,
// B_repacked = ffai_marlin_repack() output of a [K/8,N] u32 GPTQ-packed u4b8
// weight, b_scales = [K/group_size, N] f16 per-group scales, C = [M,N] f16
// out. `sorted_token_ids`/`expert_ids`/`num_tokens_past_padded` come from
// `ffai_marlin_build_routing` called once with a trivial single-"expert"
// [0, M) range (this GEMM has no MoE routing of its own — it is marlin_mm's
// generic grouped-GEMM driver run with num_experts=1, top_k=1). `workspace`
// = int32 locks buffer (size >= min(N/64 * (padded_M/8), sms*4)); `c_tmp` =
// f32 scratch (size >= min(N*padded_M, sms*4*8*max_thread_n), *2 since
// moe_block==8) — see get_kernel_cache_size()/ops_full.cu sizing in the
// archived csrc for the exact formulas; sized on the Rust side to match.
//
// F-85 round-10 (lever 1): `thread_k`/`thread_n`/`blocks_per_sm` are now
// caller-supplied instead of hardwired -1/-1/-1 ("auto"). Passing -1 still
// falls through to marlin_mm's auto-config search (kept for callers that
// don't have a cached config yet / debugging), but the decode hot path
// (qwen35.rs) resolves these ONCE per weight class at load time via
// `ffai_marlin_pick_config` (see marlin_mm.cu) and passes the resolved
// triple on every call, skipping the auto path's per-call
// cudaFuncGetAttributes probe loop (round-9's census-vs-isolated-
// microbench ~15ms/step integration gap).
void ffai_marlin_gemm_u4b8_f16(
    const void* A, const void* B_repacked, void* C, void* C_tmp,
    const void* b_scales,
    const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_past_padded,
    void* workspace,
    int prob_m, int prob_n, int prob_k, int num_groups, int group_size,
    int sms, int use_fp32_reduce,
    int thread_k, int thread_n, int blocks_per_sm, cudaStream_t stream) {
  marlin_moe_wna16::marlin_mm(
    A, B_repacked, C, C_tmp, /*bias*/nullptr, /*a_scales*/nullptr, (void*)b_scales, /*global_scale*/nullptr,
    /*zp*/nullptr, /*g_idx*/nullptr, /*perm*/nullptr, /*a_tmp*/nullptr,
    (void*)sorted_token_ids, (void*)expert_ids, (void*)num_tokens_past_padded, /*topk_weights*/nullptr,
    /*moe_block*/8, /*num_experts*/1, /*top_k*/1, /*mul_topk_weights*/false,
    prob_m, prob_n, prob_k, workspace,
    marlin_types::kFloat16, marlin_types::kU4B8, marlin_types::kFloat16, marlin_types::kFloat16,
    /*has_bias*/false, /*has_act_order*/false, /*is_k_full*/true, /*has_zp*/false,
    num_groups, group_size, /*dev*/0, stream, thread_k, thread_n, sms,
    blocks_per_sm, /*use_atomic_add*/false, use_fp32_reduce != 0, /*is_zp_float*/false);
}

}  // extern "C"

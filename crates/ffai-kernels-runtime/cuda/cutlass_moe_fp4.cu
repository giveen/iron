#include <cstdio>
// CUTLASS grouped block-scaled NVFP4 MoE GEMM, extern "C" entry for the
// ffai-kernels-runtime FFI (sm_120a/sm_121a only; AOT-built when CUTLASS_DIR set).
//
// out[t,n](f16) = sum_k A[t,k] * W[eid(t)][n,k]
//   A  = sorted-token activations, packed e2m1 [mt, K/2] bytes row-major
//   SFA= per-group ue4m3 scale blocks (canonical 512B-block swizzle, one
//        16-elem K-block per scale); group g's blob starts at SFA+sfa_off[g]
//        and is laid out for the GROUP-LOCAL row index (M_g rows pad to 128).
//   W  = contiguous packed e2m1 expert slab [n_exp, N, K/2] bytes (W[n,k]
//        row-major per expert == ColumnMajor [K,N] for the GEMM's B operand)
//   SFB= per-expert ue4m3 scale slab [n_exp, ceil(N/128)*512*ceil(K/64)] bytes
//   D  = f16 out [mt, N] (plain LinearCombination epilogue, alpha=1 beta=0 —
//        no SFD output fusion; the result feeds relu2 / scatter in f16)
//
// Sorted tokens: group g owns a contiguous row range of `group_rows[g]` rows;
// W[expert_ids[g]] is its weight slab. All per-group pointer/stride/layout
// arrays are built host-side here and shipped in ONE device blob (graph-safety
// device-side build is a follow-up; host-side first per the integration plan).

#include "cutlass/cutlass.h"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using ElementInput = cutlass::float_e2m1_t;

// A: activations, nvfp4 (e2m1 + ue4m3 block-16 SF), RowMajor [M,K]
using ElementA   = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

// B: per-expert weights, nvfp4, ColumnMajor [K,N] (== W[n,k] row-major)
using ElementB   = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C/D: f16 out, plain LinearCombination (no block-scaled output fusion)
using ElementD   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128,_128,_256>;
using ClusterShape       = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ════════════════════════════════════════════════════════════════════════════
// W4A8 (fp8 acts e4m3 × fp4 weights e2m1, mxf8f6f4 mixed block-scaled) — quality
// fix: fp8 acts = 256 levels vs fp4's 16, at the fast mxf8f6f4 MMA rate.
// ════════════════════════════════════════════════════════════════════════════
namespace w4a8 {
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;   // fp8 acts, per-32 ue8m0 SF
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
using ElementB   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;   // fp4 weights, per-32 ue8m0 SF
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128,_128,_128>;
using ClusterShape       = Shape<_1,_1,_1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace w4a8
extern "C" size_t moe_w4a8_compile_check() {
    return sizeof(typename w4a8::Gemm::GemmKernel) + sizeof(typename w4a8::CollectiveMainloop);
}


namespace w8a8 {
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;   // fp8 acts, per-32 ue8m0 SF
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;   // fp8 weights, per-32 ue8m0 SF (W8A8 near-lossless)
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;
using ElementD   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128,_128,_128>;
using ClusterShape       = Shape<_1,_1,_1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace w8a8
extern "C" size_t moe_w8a8_compile_check() {
    return sizeof(typename w8a8::Gemm::GemmKernel) + sizeof(typename w8a8::CollectiveMainloop);
}

using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

int queried_sm_count() {
    static int sm_count = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        return cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    }();
    return sm_count;
}

// ───────────── fused-activation epilogue (NEMOTRON_FUSE_UPACT) ──────────────
// Fold relu²·(1/256) + NVFP4 block-scale-quant INTO GEMM1's CUTLASS epilogue so
// GEMM1 emits (e2m1 D, ue4m3 SFD) DIRECTLY — deleting the separate relu2+amax+
// block-quant passes between the up- and down-GEMMs. The output (bp,bsf) is laid
// out bit-identically to what the plain GEMM2 _run reads as its (A,SFA).
//
// SquaredReLU: max(0,v)²·(1/256). BOTH a scalar T and the Array<T,N,true> partial
// spec are mandatory — Sm90Compute instantiates ComputeFn<Array<float,4>>; a
// scalar-only functor fails to compile.
constexpr int SFVecSize = 16;

template <class T>
struct SquaredReLU {
    static const bool kIsHeavy = false;
    CUTLASS_HOST_DEVICE T operator()(T const& v) const {
        cutlass::maximum<T> mx; T r = mx(v, T(0)); return r * r * T(1.0 / 256.0);
    }
};
template <class T, int N>
struct SquaredReLU<cutlass::Array<T, N, true>> {
    static const bool kIsHeavy = false;
    CUTLASS_HOST_DEVICE cutlass::Array<T, N, true>
    operator()(cutlass::Array<T, N, true> const& v) const {
        cutlass::maximum<cutlass::Array<T, N, true>> mx;
        cutlass::multiplies<cutlass::Array<T, N, true>> mul;
        cutlass::Array<T, N, true> r  = mx(v, T(0));
        cutlass::Array<T, N, true> sq = mul(r, r);
        return mul(sq, T(1.0 / 256.0));
    }
};

// GEMM1 with NVFP4 block-scaled output fusion. Same A/B operands, tile/cluster/
// schedule as the plain GEMM (above); only the epilogue changes: ElementD =
// e2m1, AlignmentC=AlignmentD=32, LinCombEltActBlockScaleFactor<SquaredReLU>.
namespace g1f {
    using ElementD   = cutlass::float_e2m1_t;
    using ElementC   = cutlass::half_t;
    using ElementSFType = cutlass::float_ue4m3_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    constexpr int AlignmentC = 32, AlignmentD = 32;
    using FusionOp = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
        SquaredReLU, SFVecSize, ElementD, ElementAccumulator, ElementSFType,
        cutlass::layout::RowMajor, ElementC, ElementAccumulator>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutCTag*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto, FusionOp>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA,
        ElementB, LayoutBTag*, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
} // namespace g1f

// ───────────── amax-in-epilogue GEMM1 (NEMOTRON_AMAX_EPI) ───────────────────
// GEMM1 emits f16 D = SquaredReLU(acc)·(1/256) (the relu²'d up_out / a2) AND a
// PER-GROUP amax of those activated values via an Sm90ScalarReduction node. The
// down-quant then reads a2 ONCE (no separate amax scan): the per-group amax (max
// over groups = per-tensor global) feeds the existing NVFP4 block-quant. D is the
// down-quant's input directly (relu² already applied), so the GEMM2-input is
// produced bit-identically to the 2-pass (relu2+amax+quant) path.
//
// L-stride EVT (Stride<_0,_0,int>, dScalar L-stride 1) gives true per-group amax
// on the grouped sm_120a path (validated: rel_err 0 vs separate-pass reference).
namespace g1a {
    namespace fus = cutlass::epilogue::fusion;
    static constexpr auto RS = cutlass::FloatRoundStyle::round_to_nearest;
    using ElementD   = cutlass::half_t;   // relu²'d a2 out (same dtype as plain GEMM)
    using ElementC   = cutlass::half_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using ElementAmax = float;
    using AmaxStride  = cute::Stride<cute::_0, cute::_0, int>;
    using EVTAmax =
        fus::Sm90EVT<
            fus::Sm90ScalarReduction<fus::detail::amax, cutlass::atomic_maximum,
                ElementAmax, ElementAccumulator, RS, AmaxStride>,
            fus::Sm90EVT<
                fus::Sm90Compute<SquaredReLU, ElementD, ElementAccumulator, RS>,
                fus::Sm90LinearCombination<ElementAccumulator, ElementAccumulator,
                    ElementC, ElementAccumulator, RS> > >;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutCTag*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto, EVTAmax>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA,
        ElementB, LayoutBTag*, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
} // namespace g1a

} // namespace

// Returns 0 on success. group_rows/expert_ids/sfa_off are HOST arrays
// (n_groups each); sfa_off[g] = BYTE offset of group g's SF blob inside SFA.
// alpha_vec: optional DEVICE float[n_groups] of per-group output scales
// (act_global * expert_global, folding both operands' per-tensor globals back
// in); null = alpha 1.
extern "C" int moe_grouped_gemm_cutlass_fp4(
    const void* A, const void* SFA, const void* B, const void* SFB, void* D,
    const int* group_rows, const int* expert_ids, const long long* sfa_off,
    const void* alpha_vec, int n_groups, int N, int K, void* stream_v)
{
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (K % 32 != 0 || N % 32 != 0) return 20; // e2m1 TMA alignment (32 elems)
    const size_t w_slab_bytes  = (size_t)N * (size_t)K / 2;
    const size_t sfb_exp_bytes = (size_t)((N + 127) / 128) * 512 * (size_t)((K + 63) / 64);

    // ── per-group host arrays ───────────────────────────────────────────────
    std::vector<UnderlyingProblemShape> ps_h(n_groups);
    std::vector<const ElementInput*> pA_h(n_groups);
    std::vector<const ElementInput*> pB_h(n_groups);
    std::vector<const ElementSF*> pSFA_h(n_groups);
    std::vector<const ElementSF*> pSFB_h(n_groups);
    std::vector<ElementD*> pD_h(n_groups);
    std::vector<StrideA> dA_h(n_groups);
    std::vector<StrideB> dB_h(n_groups);
    std::vector<StrideD> dD_h(n_groups);
    std::vector<LayoutSFA> lSFA_h(n_groups);
    std::vector<LayoutSFB> lSFB_h(n_groups);
    long rowoff = 0;
    for (int g = 0; g < n_groups; ++g) {
        const int m = group_rows[g];
        const long eid = expert_ids[g];
        ps_h[g]   = {m, N, K};
        pA_h[g]   = (const ElementInput*)((const uint8_t*)A + rowoff * (K / 2));
        pSFA_h[g] = (const ElementSF*)((const uint8_t*)SFA + sfa_off[g]);
        pB_h[g]   = (const ElementInput*)((const uint8_t*)B + eid * w_slab_bytes);
        pSFB_h[g] = (const ElementSF*)((const uint8_t*)SFB + eid * sfb_exp_bytes);
        pD_h[g]   = (ElementD*)((uint8_t*)D + rowoff * (long)N * 2);
        dA_h[g]   = cutlass::make_cute_packed_stride(StrideA{}, {m, K, 1});
        dB_h[g]   = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        dD_h[g]   = cutlass::make_cute_packed_stride(StrideD{}, {m, N, 1});
        lSFA_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, N, K, 1));
        lSFB_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, N, K, 1));
        rowoff += m;
    }

    // ── ship everything in ONE device blob (16B-aligned sections) ──────────
    auto sec = [](size_t bytes) { return (bytes + 15) & ~(size_t)15; };
    const size_t off_ps   = 0;
    const size_t off_pA   = off_ps   + sec(n_groups * sizeof(UnderlyingProblemShape));
    const size_t off_pB   = off_pA   + sec(n_groups * sizeof(void*));
    const size_t off_pSFA = off_pB   + sec(n_groups * sizeof(void*));
    const size_t off_pSFB = off_pSFA + sec(n_groups * sizeof(void*));
    const size_t off_pD   = off_pSFB + sec(n_groups * sizeof(void*));
    const size_t off_dA   = off_pD   + sec(n_groups * sizeof(void*));
    const size_t off_dB   = off_dA   + sec(n_groups * sizeof(StrideA));
    const size_t off_dD   = off_dB   + sec(n_groups * sizeof(StrideB));
    const size_t off_lSFA = off_dD   + sec(n_groups * sizeof(StrideD));
    const size_t off_lSFB = off_lSFA + sec(n_groups * sizeof(LayoutSFA));
    const size_t off_pAl  = off_lSFB + sec(n_groups * sizeof(LayoutSFB));
    const size_t blob_bytes = off_pAl + sec(n_groups * sizeof(float*));

    std::vector<uint8_t> staging(blob_bytes, 0);
    std::memcpy(staging.data() + off_ps,   ps_h.data(),   n_groups * sizeof(UnderlyingProblemShape));
    std::memcpy(staging.data() + off_pA,   pA_h.data(),   n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pB,   pB_h.data(),   n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pSFA, pSFA_h.data(), n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pSFB, pSFB_h.data(), n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pD,   pD_h.data(),   n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_dA,   dA_h.data(),   n_groups * sizeof(StrideA));
    std::memcpy(staging.data() + off_dB,   dB_h.data(),   n_groups * sizeof(StrideB));
    std::memcpy(staging.data() + off_dD,   dD_h.data(),   n_groups * sizeof(StrideD));
    std::memcpy(staging.data() + off_lSFA, lSFA_h.data(), n_groups * sizeof(LayoutSFA));
    std::memcpy(staging.data() + off_lSFB, lSFB_h.data(), n_groups * sizeof(LayoutSFB));
    if (alpha_vec) {
        // per-group alpha POINTER array (values stay device-side, addresses
        // are host-computable from the device base — no extra sync).
        std::vector<const float*> pAl_h(n_groups);
        for (int g = 0; g < n_groups; ++g) pAl_h[g] = (const float*)alpha_vec + g;
        std::memcpy(staging.data() + off_pAl, pAl_h.data(), n_groups * sizeof(float*));
    }

    uint8_t* blob = nullptr;
    uint8_t* work = nullptr;
    int rc = 0;
#define MT_CUDA_CK(call) do { if ((call) != cudaSuccess) { rc = 3; goto cleanup; } } while (0)
    MT_CUDA_CK(cudaMallocAsync((void**)&blob, blob_bytes, stream));
    MT_CUDA_CK(cudaMemcpyAsync(blob, staging.data(), blob_bytes, cudaMemcpyHostToDevice, stream));

    {
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = queried_sm_count();

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {n_groups, (UnderlyingProblemShape*)(blob + off_ps), ps_h.data()},
            {(const ElementA::DataType**)(blob + off_pA),  (StrideA*)(blob + off_dA),
             (const ElementB::DataType**)(blob + off_pB),  (StrideB*)(blob + off_dB),
             (const ElementSF**)(blob + off_pSFA), (LayoutSFA*)(blob + off_lSFA),
             (const ElementSF**)(blob + off_pSFB), (LayoutSFB*)(blob + off_lSFB)},
            {{}, // fusion args set below
             nullptr, (StrideC*)(blob + off_dD),                 // C unused (beta=0)
             (ElementD**)(blob + off_pD), (StrideD*)(blob + off_dD)},
            hw_info
        };
        if (alpha_vec) {
            args.epilogue.thread.alpha = 0.0f; // ignored when ptr_array set
            args.epilogue.thread.alpha_ptr_array = (const float* const*)(blob + off_pAl);
            args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 1};
        } else {
            args.epilogue.thread.alpha = 1.0f;
        }
        args.epilogue.thread.beta = 0.0f;

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) { rc = 10; goto cleanup; }
        size_t ws = Gemm::get_workspace_size(args);
        if (ws) MT_CUDA_CK(cudaMallocAsync((void**)&work, ws, stream));
        cutlass::Status st = gemm.initialize(args, work, stream);
        if (st != cutlass::Status::kSuccess) { rc = 1; goto cleanup; }
        st = gemm.run(stream);
        if (st != cutlass::Status::kSuccess) { rc = 2; goto cleanup; }
    }
#undef MT_CUDA_CK

cleanup:
    if (blob) cudaFreeAsync(blob, stream);
    if (work) cudaFreeAsync(work, stream);
    return rc;
}



// ════════════════════════════════════════════════════════════════════════════
// W4A8 host-descriptor grouped GEMM: A=fp8 e4m3 (mx, per-32 ue8m0 SF), B=fp4 e2m1
// (mx, per-32 ue8m0 SF), mxf8f6f4 mixed MMA. Mirrors moe_grouped_gemm_cutlass_fp4
// but with w4a8::Gemm + fp8 A stride (K elems/row, not K/2). sfa_off[] = per-group
// byte offsets into SFA; sfb_exp_bytes = per-expert SFB stride (both computed by the
// fp8 act-quant / mxfp4 weight-pack, which know the per-32 ue8m0 layout).
// ════════════════════════════════════════════════════════════════════════════
// W4A8 fp8 (e4m3) activation quant: per-32 MX block, ue8m0 SF in cutlass SFA layout.
// x [mt,K] half row-major (gathered) -> out [mt,K] e4m3 + sf ue8m0 (SFA swizzle).
namespace w4a8q {
using BSC2 = typename w4a8::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
template<class LSFA>
__global__ void actquant_kernel(const __half* __restrict__ x,
    cutlass::float_e4m3_t* __restrict__ out,
    cutlass::float_ue8m0_t* __restrict__ sf,
    LSFA layout_sfa, int mt, int K) {
  int row = blockIdx.x;
  if (row >= mt) return;
  int KB = K / 32;
  for (int kb = threadIdx.x; kb < KB; kb += blockDim.x) {
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) { float v = __half2float(x[(size_t)row*K + kb*32 + j]); amax = fmaxf(amax, fabsf(v)); }
    cutlass::float_ue8m0_t sfv(amax / 448.0f);
    float inv = 1.0f / fmaxf(float(sfv), 1e-30f);
    #pragma unroll
    for (int j = 0; j < 32; ++j) { float v = __half2float(x[(size_t)row*K + kb*32 + j]); out[(size_t)row*K + kb*32 + j] = cutlass::float_e4m3_t(v * inv); }
    sf[cute::crd2idx(cute::make_coord(row, kb*32, 0), layout_sfa)] = sfv;  // K-coord is in ELEMENTS (32 share one SF cell)
  }
}
} // namespace w4a8q
// W4A8 mxfp4 weight pack: f16 weight [n_exp,N,K] -> e2m1 packed [n_exp,N,K/2]
// + ue8m0 per-32 SF in cutlass SFB layout (per expert). amax/6 (e2m1 max=6).
namespace w4a8q {
template<class LSFB>
__global__ void packw_kernel(const __half* __restrict__ w,
    uint8_t* __restrict__ outp, cutlass::float_ue8m0_t* __restrict__ sf,
    LSFB layout_sfb, int n_exp, int N, int K, long long sfb_exp_elems) {
  int e = blockIdx.z; int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (e >= n_exp || n >= N) return;
  int KB = K / 32;
  const __half* wr = w + ((size_t)e * N + n) * K;
  uint8_t* op = outp + ((size_t)e * N + n) * (K / 2);
  cutlass::float_ue8m0_t* sfe = sf + (size_t)e * sfb_exp_elems;
  for (int kb = threadIdx.x; kb < KB; kb += blockDim.x) {
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) { float v = __half2float(wr[kb*32 + j]); amax = fmaxf(amax, fabsf(v)); }
    cutlass::float_ue8m0_t sfv(amax / 6.0f);
    float inv = 1.0f / fmaxf(float(sfv), 1e-30f);
    #pragma unroll
    for (int j = 0; j < 32; j += 2) {
      uint8_t lo = cutlass::float_e2m1_t(__half2float(wr[kb*32+j])   * inv).storage & 0xF;
      uint8_t hi = cutlass::float_e2m1_t(__half2float(wr[kb*32+j+1]) * inv).storage & 0xF;
      op[(kb*32 + j) / 2] = lo | (hi << 4);
    }
    sfe[cute::crd2idx(cute::make_coord(n, kb*32, 0), layout_sfb)] = sfv;  // K-coord is in ELEMENTS
  }
}
} // namespace w4a8q
// Returns the per-expert SFB element count (so the caller sizes the SF buffer + stride).
extern "C" long long w4a8_packw(const void* w, void* outp, void* sf, int n_exp, int N, int K, void* stream_v) {
  cudaStream_t stream = (cudaStream_t)stream_v;
  auto layout_sfb = w4a8q::BSC2::tile_atom_to_shape_SFB(cute::make_shape(1, N, K, 1));
  long long sfb_exp_elems = (long long)cute::cosize(layout_sfb);
  dim3 grid((N + 7) / 8, 1, n_exp); dim3 block(32, 8);
  w4a8q::packw_kernel<<<grid, block, 0, stream>>>(
      (const __half*)w, (uint8_t*)outp, (cutlass::float_ue8m0_t*)sf, layout_sfb, n_exp, N, K, sfb_exp_elems);
  return sfb_exp_elems;  // bytes (ue8m0 = 1 byte)
}

// GROUP-AWARE: per expert g, quantize its m_g rows into its own SFA section at
// sfa_off[g] (atom-padded), using that group's m_g layout — matches the grouped
// GEMM's per-group SFA read (pSFA[g] = SFA + sfa_off[g], lSFA[g] = layout(m_g)).
extern "C" void w4a8_actquant(const void* x, void* out, void* sf,
    const int* group_rows, const long long* sfa_off, int n_groups, int N, int K, void* stream_v) {
  cudaStream_t stream = (cudaStream_t)stream_v;
  long rowoff = 0;
  for (int g = 0; g < n_groups; ++g) {
    int m = group_rows[g];
    if (m > 0) {
      auto layout_sfa = w4a8q::BSC2::tile_atom_to_shape_SFA(cute::make_shape(m, N, K, 1));
      const __half* xg = (const __half*)x + (size_t)rowoff * K;
      cutlass::float_e4m3_t* og = (cutlass::float_e4m3_t*)out + (size_t)rowoff * K;
      cutlass::float_ue8m0_t* sfg = (cutlass::float_ue8m0_t*)((uint8_t*)sf + sfa_off[g]);
      dim3 grid(m), block(256);
      w4a8q::actquant_kernel<<<grid, block, 0, stream>>>(xg, og, sfg, layout_sfa, m, K);
    }
    rowoff += m;
  }
}


// ════════════════════════════════════════════════════════════════════════════
// W8A8 fp8 (e4m3) weight pack + act quant — both per-32 MX (ue8m0 SF). Mirrors
// w4a8q but B is fp8 (1 byte/elem, amax/448) not fp4. SF swizzle from w8a8's
// Sm1xxBlkScaledConfig (cutlass handles the layout).
// ════════════════════════════════════════════════════════════════════════════
namespace w8a8q {
using BSC2 = typename w8a8::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
template<class LSFA>
__global__ void actquant_kernel(const __half* __restrict__ x,
    cutlass::float_e4m3_t* __restrict__ out,
    cutlass::float_ue8m0_t* __restrict__ sf,
    LSFA layout_sfa, int mt, int K) {
  int row = blockIdx.x;
  if (row >= mt) return;
  int KB = K / 32;
  for (int kb = threadIdx.x; kb < KB; kb += blockDim.x) {
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) { float v = __half2float(x[(size_t)row*K + kb*32 + j]); amax = fmaxf(amax, fabsf(v)); }
    cutlass::float_ue8m0_t sfv(amax / 448.0f);
    float inv = 1.0f / fmaxf(float(sfv), 1e-30f);
    #pragma unroll
    for (int j = 0; j < 32; ++j) { float v = __half2float(x[(size_t)row*K + kb*32 + j]); out[(size_t)row*K + kb*32 + j] = cutlass::float_e4m3_t(v * inv); }
    sf[cute::crd2idx(cute::make_coord(row, kb*32, 0), layout_sfa)] = sfv;
  }
}
template<class LSFB>
__global__ void packw_kernel(const __half* __restrict__ w,
    cutlass::float_e4m3_t* __restrict__ outp, cutlass::float_ue8m0_t* __restrict__ sf,
    LSFB layout_sfb, int n_exp, int N, int K, long long sfb_exp_elems) {
  int e = blockIdx.z; int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (e >= n_exp || n >= N) return;
  int KB = K / 32;
  const __half* wr = w + ((size_t)e * N + n) * K;
  cutlass::float_e4m3_t* op = outp + ((size_t)e * N + n) * K;   // fp8: K elems/row
  cutlass::float_ue8m0_t* sfe = sf + (size_t)e * sfb_exp_elems;
  for (int kb = threadIdx.x; kb < KB; kb += blockDim.x) {
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) { float v = __half2float(wr[kb*32 + j]); amax = fmaxf(amax, fabsf(v)); }
    cutlass::float_ue8m0_t sfv(amax / 448.0f);
    float inv = 1.0f / fmaxf(float(sfv), 1e-30f);
    #pragma unroll
    for (int j = 0; j < 32; ++j) { op[kb*32 + j] = cutlass::float_e4m3_t(__half2float(wr[kb*32+j]) * inv); }
    sfe[cute::crd2idx(cute::make_coord(n, kb*32, 0), layout_sfb)] = sfv;
  }
}
} // namespace w8a8q
extern "C" long long w8a8_packw(const void* w, void* outp, void* sf, int n_exp, int N, int K, void* stream_v) {
  cudaStream_t stream = (cudaStream_t)stream_v;
  auto layout_sfb = w8a8q::BSC2::tile_atom_to_shape_SFB(cute::make_shape(1, N, K, 1));
  long long sfb_exp_elems = (long long)cute::cosize(layout_sfb);
  dim3 grid((N + 7) / 8, 1, n_exp); dim3 block(32, 8);
  w8a8q::packw_kernel<<<grid, block, 0, stream>>>(
      (const __half*)w, (cutlass::float_e4m3_t*)outp, (cutlass::float_ue8m0_t*)sf, layout_sfb, n_exp, N, K, sfb_exp_elems);
  return sfb_exp_elems;
}
extern "C" void w8a8_actquant(const void* x, void* out, void* sf,
    const int* group_rows, const long long* sfa_off, int n_groups, int N, int K, void* stream_v) {
  cudaStream_t stream = (cudaStream_t)stream_v;
  long rowoff = 0;
  for (int g = 0; g < n_groups; ++g) {
    int m = group_rows[g];
    if (m > 0) {
      auto layout_sfa = w8a8q::BSC2::tile_atom_to_shape_SFA(cute::make_shape(m, N, K, 1));
      const __half* xg = (const __half*)x + (size_t)rowoff * K;
      cutlass::float_e4m3_t* og = (cutlass::float_e4m3_t*)out + (size_t)rowoff * K;
      cutlass::float_ue8m0_t* sfg = (cutlass::float_ue8m0_t*)((uint8_t*)sf + sfa_off[g]);
      dim3 grid(m), block(256);
      w8a8q::actquant_kernel<<<grid, block, 0, stream>>>(xg, og, sfg, layout_sfa, m, K);
    }
    rowoff += m;
  }
}

extern "C" int moe_grouped_gemm_w4a8(
    const void* A, const void* SFA, const void* B, const void* SFB, void* D,
    const int* group_rows, const int* expert_ids, const long long* sfa_off,
    long long sfb_exp_bytes, const void* alpha_vec,
    int n_groups, int N, int K, void* stream_v)
{
    using G   = w4a8::Gemm;
    using PS  = typename w4a8::ProblemShape::UnderlyingProblemShape;
    using SA  = typename G::GemmKernel::InternalStrideA;
    using SB  = typename G::GemmKernel::InternalStrideB;
    using SC  = typename G::GemmKernel::InternalStrideC;
    using SD  = typename G::GemmKernel::InternalStrideD;
    using LSFA= typename G::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LSFB= typename G::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using ESF = typename G::GemmKernel::CollectiveMainloop::ElementSF;
    using BSC = typename G::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using EA  = typename w4a8::ElementA::DataType;   // float_e4m3_t
    using EB  = typename w4a8::ElementB::DataType;   // float_e2m1_t
    using ED  = w4a8::ElementD;                      // half_t
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (K % 32 != 0 || N % 32 != 0) return 20;
    const size_t w_slab_bytes = (size_t)N * (size_t)K / 2;   // fp4 weight slab

    std::vector<PS>  ps_h(n_groups);
    std::vector<const EA*> pA_h(n_groups);
    std::vector<const EB*> pB_h(n_groups);
    std::vector<const ESF*> pSFA_h(n_groups), pSFB_h(n_groups);
    std::vector<ED*> pD_h(n_groups);
    std::vector<SA> dA_h(n_groups); std::vector<SB> dB_h(n_groups); std::vector<SD> dD_h(n_groups);
    std::vector<LSFA> lSFA_h(n_groups); std::vector<LSFB> lSFB_h(n_groups);
    long rowoff = 0;
    for (int g = 0; g < n_groups; ++g) {
        const int m = group_rows[g]; const long eid = expert_ids[g];
        ps_h[g]   = {m, N, K};
        pA_h[g]   = (const EA*)((const uint8_t*)A + (size_t)rowoff * K);          // fp8: K bytes/row
        pSFA_h[g] = (const ESF*)((const uint8_t*)SFA + sfa_off[g]);
        pB_h[g]   = (const EB*)((const uint8_t*)B + (size_t)eid * w_slab_bytes);
        pSFB_h[g] = (const ESF*)((const uint8_t*)SFB + (size_t)eid * (size_t)sfb_exp_bytes);
        pD_h[g]   = (ED*)((uint8_t*)D + (size_t)rowoff * (long)N * 2);
        dA_h[g]   = cutlass::make_cute_packed_stride(SA{}, {m, K, 1});
        dB_h[g]   = cutlass::make_cute_packed_stride(SB{}, {N, K, 1});
        dD_h[g]   = cutlass::make_cute_packed_stride(SD{}, {m, N, 1});
        lSFA_h[g] = BSC::tile_atom_to_shape_SFA(cute::make_shape(m, N, K, 1));
        lSFB_h[g] = BSC::tile_atom_to_shape_SFB(cute::make_shape(m, N, K, 1));
        rowoff += m;
    }
    auto sec = [](size_t b){ return (b + 15) & ~(size_t)15; };
    const size_t o_ps=0;
    const size_t o_pA=o_ps+sec(n_groups*sizeof(PS));
    const size_t o_pB=o_pA+sec(n_groups*sizeof(void*));
    const size_t o_pSFA=o_pB+sec(n_groups*sizeof(void*));
    const size_t o_pSFB=o_pSFA+sec(n_groups*sizeof(void*));
    const size_t o_pD=o_pSFB+sec(n_groups*sizeof(void*));
    const size_t o_dA=o_pD+sec(n_groups*sizeof(void*));
    const size_t o_dB=o_dA+sec(n_groups*sizeof(SA));
    const size_t o_dD=o_dB+sec(n_groups*sizeof(SB));
    const size_t o_lSFA=o_dD+sec(n_groups*sizeof(SD));
    const size_t o_lSFB=o_lSFA+sec(n_groups*sizeof(LSFA));
    const size_t o_pAl=o_lSFB+sec(n_groups*sizeof(LSFB));
    const size_t blob_bytes=o_pAl+sec(n_groups*sizeof(float*));
    std::vector<uint8_t> stg(blob_bytes,0);
    std::memcpy(stg.data()+o_ps,ps_h.data(),n_groups*sizeof(PS));
    std::memcpy(stg.data()+o_pA,pA_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pB,pB_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pSFA,pSFA_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pSFB,pSFB_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pD,pD_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_dA,dA_h.data(),n_groups*sizeof(SA));
    std::memcpy(stg.data()+o_dB,dB_h.data(),n_groups*sizeof(SB));
    std::memcpy(stg.data()+o_dD,dD_h.data(),n_groups*sizeof(SD));
    std::memcpy(stg.data()+o_lSFA,lSFA_h.data(),n_groups*sizeof(LSFA));
    std::memcpy(stg.data()+o_lSFB,lSFB_h.data(),n_groups*sizeof(LSFB));
    if (alpha_vec){ std::vector<const float*> al(n_groups); for(int g=0;g<n_groups;++g) al[g]=(const float*)alpha_vec+g; std::memcpy(stg.data()+o_pAl,al.data(),n_groups*sizeof(float*)); }
    uint8_t* blob=nullptr; uint8_t* work=nullptr; int rc=0;
#define MT_CK(c) do{ if((c)!=cudaSuccess){rc=3;goto cleanup;} }while(0)
    MT_CK(cudaMallocAsync((void**)&blob,blob_bytes,stream));
    MT_CK(cudaMemcpyAsync(blob,stg.data(),blob_bytes,cudaMemcpyHostToDevice,stream));
    {
        cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=queried_sm_count();
        typename G::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {n_groups,(PS*)(blob+o_ps),ps_h.data()},
            {(const EA**)(blob+o_pA),(SA*)(blob+o_dA),
             (const EB**)(blob+o_pB),(SB*)(blob+o_dB),
             (const ESF**)(blob+o_pSFA),(LSFA*)(blob+o_lSFA),
             (const ESF**)(blob+o_pSFB),(LSFB*)(blob+o_lSFB)},
            {{}, nullptr,(SC*)(blob+o_dD),(ED**)(blob+o_pD),(SD*)(blob+o_dD)},
            hw
        };
        if (alpha_vec){ args.epilogue.thread.alpha=0.0f; args.epilogue.thread.alpha_ptr_array=(const float* const*)(blob+o_pAl); args.epilogue.thread.dAlpha={cute::_0{},cute::_0{},1}; }
        else { args.epilogue.thread.alpha=1.0f; }
        args.epilogue.thread.beta=0.0f;
        G gemm;
        if (gemm.can_implement(args)!=cutlass::Status::kSuccess){ rc=10; goto cleanup; }
        size_t ws=G::get_workspace_size(args);
        if (ws) MT_CK(cudaMallocAsync((void**)&work,ws,stream));
        if (gemm.initialize(args,work,stream)!=cutlass::Status::kSuccess){ rc=1; goto cleanup; }
        if (gemm.run(stream)!=cutlass::Status::kSuccess){ rc=2; goto cleanup; }
    }
#undef MT_CK
cleanup:
    if (blob) cudaFreeAsync(blob,stream);
    if (work) cudaFreeAsync(work,stream);
    return rc;
}


extern "C" int moe_grouped_gemm_w8a8(
    const void* A, const void* SFA, const void* B, const void* SFB, void* D,
    const int* group_rows, const int* expert_ids, const long long* sfa_off,
    long long sfb_exp_bytes, const void* alpha_vec,
    int n_groups, int N, int K, void* stream_v)
{
    using G   = w8a8::Gemm;
    using PS  = typename w8a8::ProblemShape::UnderlyingProblemShape;
    using SA  = typename G::GemmKernel::InternalStrideA;
    using SB  = typename G::GemmKernel::InternalStrideB;
    using SC  = typename G::GemmKernel::InternalStrideC;
    using SD  = typename G::GemmKernel::InternalStrideD;
    using LSFA= typename G::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LSFB= typename G::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using ESF = typename G::GemmKernel::CollectiveMainloop::ElementSF;
    using BSC = typename G::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using EA  = typename w8a8::ElementA::DataType;   // float_e4m3_t
    using EB  = typename w8a8::ElementB::DataType;   // float_e4m3_t (fp8)
    using ED  = w8a8::ElementD;                      // half_t
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (K % 32 != 0 || N % 32 != 0) return 20;
    const size_t w_slab_bytes = (size_t)N * (size_t)K;       // fp8 weight slab (1 byte/elem)

    std::vector<PS>  ps_h(n_groups);
    std::vector<const EA*> pA_h(n_groups);
    std::vector<const EB*> pB_h(n_groups);
    std::vector<const ESF*> pSFA_h(n_groups), pSFB_h(n_groups);
    std::vector<ED*> pD_h(n_groups);
    std::vector<SA> dA_h(n_groups); std::vector<SB> dB_h(n_groups); std::vector<SD> dD_h(n_groups);
    std::vector<LSFA> lSFA_h(n_groups); std::vector<LSFB> lSFB_h(n_groups);
    long rowoff = 0;
    for (int g = 0; g < n_groups; ++g) {
        const int m = group_rows[g]; const long eid = expert_ids[g];
        ps_h[g]   = {m, N, K};
        pA_h[g]   = (const EA*)((const uint8_t*)A + (size_t)rowoff * K);          // fp8: K bytes/row
        pSFA_h[g] = (const ESF*)((const uint8_t*)SFA + sfa_off[g]);
        pB_h[g]   = (const EB*)((const uint8_t*)B + (size_t)eid * w_slab_bytes);
        pSFB_h[g] = (const ESF*)((const uint8_t*)SFB + (size_t)eid * (size_t)sfb_exp_bytes);
        pD_h[g]   = (ED*)((uint8_t*)D + (size_t)rowoff * (long)N * 2);
        dA_h[g]   = cutlass::make_cute_packed_stride(SA{}, {m, K, 1});
        dB_h[g]   = cutlass::make_cute_packed_stride(SB{}, {N, K, 1});
        dD_h[g]   = cutlass::make_cute_packed_stride(SD{}, {m, N, 1});
        lSFA_h[g] = BSC::tile_atom_to_shape_SFA(cute::make_shape(m, N, K, 1));
        lSFB_h[g] = BSC::tile_atom_to_shape_SFB(cute::make_shape(m, N, K, 1));
        rowoff += m;
    }
    auto sec = [](size_t b){ return (b + 15) & ~(size_t)15; };
    const size_t o_ps=0;
    const size_t o_pA=o_ps+sec(n_groups*sizeof(PS));
    const size_t o_pB=o_pA+sec(n_groups*sizeof(void*));
    const size_t o_pSFA=o_pB+sec(n_groups*sizeof(void*));
    const size_t o_pSFB=o_pSFA+sec(n_groups*sizeof(void*));
    const size_t o_pD=o_pSFB+sec(n_groups*sizeof(void*));
    const size_t o_dA=o_pD+sec(n_groups*sizeof(void*));
    const size_t o_dB=o_dA+sec(n_groups*sizeof(SA));
    const size_t o_dD=o_dB+sec(n_groups*sizeof(SB));
    const size_t o_lSFA=o_dD+sec(n_groups*sizeof(SD));
    const size_t o_lSFB=o_lSFA+sec(n_groups*sizeof(LSFA));
    const size_t o_pAl=o_lSFB+sec(n_groups*sizeof(LSFB));
    const size_t blob_bytes=o_pAl+sec(n_groups*sizeof(float*));
    std::vector<uint8_t> stg(blob_bytes,0);
    std::memcpy(stg.data()+o_ps,ps_h.data(),n_groups*sizeof(PS));
    std::memcpy(stg.data()+o_pA,pA_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pB,pB_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pSFA,pSFA_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pSFB,pSFB_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_pD,pD_h.data(),n_groups*sizeof(void*));
    std::memcpy(stg.data()+o_dA,dA_h.data(),n_groups*sizeof(SA));
    std::memcpy(stg.data()+o_dB,dB_h.data(),n_groups*sizeof(SB));
    std::memcpy(stg.data()+o_dD,dD_h.data(),n_groups*sizeof(SD));
    std::memcpy(stg.data()+o_lSFA,lSFA_h.data(),n_groups*sizeof(LSFA));
    std::memcpy(stg.data()+o_lSFB,lSFB_h.data(),n_groups*sizeof(LSFB));
    if (alpha_vec){ std::vector<const float*> al(n_groups); for(int g=0;g<n_groups;++g) al[g]=(const float*)alpha_vec+g; std::memcpy(stg.data()+o_pAl,al.data(),n_groups*sizeof(float*)); }
    uint8_t* blob=nullptr; uint8_t* work=nullptr; int rc=0;
#define MT_CK(c) do{ if((c)!=cudaSuccess){rc=3;goto cleanup;} }while(0)
    MT_CK(cudaMallocAsync((void**)&blob,blob_bytes,stream));
    MT_CK(cudaMemcpyAsync(blob,stg.data(),blob_bytes,cudaMemcpyHostToDevice,stream));
    {
        cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=queried_sm_count();
        typename G::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {n_groups,(PS*)(blob+o_ps),ps_h.data()},
            {(const EA**)(blob+o_pA),(SA*)(blob+o_dA),
             (const EB**)(blob+o_pB),(SB*)(blob+o_dB),
             (const ESF**)(blob+o_pSFA),(LSFA*)(blob+o_lSFA),
             (const ESF**)(blob+o_pSFB),(LSFB*)(blob+o_lSFB)},
            {{}, nullptr,(SC*)(blob+o_dD),(ED**)(blob+o_pD),(SD*)(blob+o_dD)},
            hw
        };
        if (alpha_vec){ args.epilogue.thread.alpha=0.0f; args.epilogue.thread.alpha_ptr_array=(const float* const*)(blob+o_pAl); args.epilogue.thread.dAlpha={cute::_0{},cute::_0{},1}; }
        else { args.epilogue.thread.alpha=1.0f; }
        args.epilogue.thread.beta=0.0f;
        G gemm;
        if (gemm.can_implement(args)!=cutlass::Status::kSuccess){ rc=10; goto cleanup; }
        size_t ws=G::get_workspace_size(args);
        if (ws) MT_CK(cudaMallocAsync((void**)&work,ws,stream));
        if (gemm.initialize(args,work,stream)!=cutlass::Status::kSuccess){ rc=1; goto cleanup; }
        if (gemm.run(stream)!=cutlass::Status::kSuccess){ rc=2; goto cleanup; }
    }
#undef MT_CK
cleanup:
    if (blob) cudaFreeAsync(blob,stream);
    if (work) cudaFreeAsync(work,stream);
    return rc;
}

// ───────────────────────── device-side descriptor build ─────────────────────
// prepare(): one-time per (n_groups,N,K[,D-base...]) — allocates a persistent
// device blob + workspace, fills every M-INDEPENDENT section host-side once
// (expert ptrs, strides, SF layouts at worst-case M extent, alpha ptr array),
// initializes the GEMM with host_problem_shapes=nullptr, and returns a handle.
// run(): per call — ONE small kernel derives ps/pA/pSFA/pD from the DEVICE
// offsets array (no download, no host build, no allocs), then gemm.run().
// Graph-safe: fixed launch geometry, fixed pointers, stream-ordered only.

namespace {

struct Fp4GroupedHandle {
    Gemm gemm;
    uint8_t* blob = nullptr;
    uint8_t* work = nullptr;
    int n_groups = 0, N = 0, K = 0;
    // section offsets (same layout as the one-shot path)
    size_t off_ps, off_pA, off_pB, off_pSFA, off_pSFB, off_pD, off_pAl;
};

} // namespace

// one thread per group: derive the M-dependent sections from device offsets.
// FILE SCOPE (not anon-namespace: nvcc's cudafe stub collides __global__
// symbols in anon namespaces with CUTLASS's own). The problem-shape entry is
// written as 3 contiguous ints (asserted == UnderlyingProblemShape below).
// sfa blocks are laid out densely: group g's blob starts at
// (sum over j<g of ceil(M_j/128)) * 512 * ceil(K/64) bytes.
__global__ void mt_fp4_fill_group_args(
    const unsigned* __restrict__ off,   // [n_groups+1] device row offsets
    const uint8_t* A, const uint8_t* SFA, uint8_t* D,
    int* ps3,                            // [n_groups*3] (M,N,K) triples
    const void** pA, const void** pSFA, void** pD,
    int n_groups, int N, int K)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const int m = (int)(off[g + 1] - off[g]);
    ps3[g * 3 + 0] = m; ps3[g * 3 + 1] = N; ps3[g * 3 + 2] = K;
    pA[g] = A + (size_t)off[g] * (K / 2);
    pD[g] = D + (size_t)off[g] * (size_t)N * 2;
    // dense prefix of ceil(M_j/128) — n_groups is small (<=130), linear scan
    size_t blk = 0;
    for (int j = 0; j < g; ++j) blk += (size_t)((off[j + 1] - off[j] + 127) / 128);
    pSFA[g] = SFA + blk * 512 * (size_t)((K + 63) / 64);
}

static_assert(sizeof(UnderlyingProblemShape) == 3 * sizeof(int),
    "GroupProblemShape underlying entry must be 3 contiguous ints");

extern "C" void* moe_grouped_gemm_cutlass_fp4_prepare(
    const void* B, const void* SFB, const void* alpha_vec,
    int n_groups, int N, int K, int max_m_total)
{
    if (K % 32 != 0 || N % 32 != 0) return nullptr;
    auto* h = new Fp4GroupedHandle();
    h->n_groups = n_groups; h->N = N; h->K = K;
    const size_t w_slab_bytes  = (size_t)N * (size_t)K / 2;
    const size_t sfb_exp_bytes = (size_t)((N + 127) / 128) * 512 * (size_t)((K + 63) / 64);

    auto sec = [](size_t bytes) { return (bytes + 15) & ~(size_t)15; };
    h->off_ps   = 0;
    h->off_pA   = h->off_ps   + sec(n_groups * sizeof(UnderlyingProblemShape));
    h->off_pB   = h->off_pA   + sec(n_groups * sizeof(void*));
    h->off_pSFA = h->off_pB   + sec(n_groups * sizeof(void*));
    h->off_pSFB = h->off_pSFA + sec(n_groups * sizeof(void*));
    h->off_pD   = h->off_pSFB + sec(n_groups * sizeof(void*));
    const size_t off_dA   = h->off_pD + sec(n_groups * sizeof(void*));
    const size_t off_dB   = off_dA   + sec(n_groups * sizeof(StrideA));
    const size_t off_dD   = off_dB   + sec(n_groups * sizeof(StrideB));
    const size_t off_lSFA = off_dD   + sec(n_groups * sizeof(StrideD));
    const size_t off_lSFB = off_lSFA + sec(n_groups * sizeof(LayoutSFA));
    h->off_pAl  = off_lSFB + sec(n_groups * sizeof(LayoutSFB));
    const size_t blob_bytes = h->off_pAl + sec(n_groups * sizeof(float*));

    // host-fill every M-independent section once. SF layouts use the
    // WORST-CASE M extent (max_m_total): per-128-row-block strides are
    // M-independent and the tile scheduler bounds reads by the device
    // problem shapes, so an over-sized extent is safe.
    std::vector<uint8_t> staging(blob_bytes, 0);
    {
        std::vector<const ElementInput*> pB_h(n_groups);
        std::vector<const ElementSF*> pSFB_h(n_groups);
        std::vector<StrideA> dA_h(n_groups);
        std::vector<StrideB> dB_h(n_groups);
        std::vector<StrideD> dD_h(n_groups);
        std::vector<LayoutSFA> lSFA_h(n_groups);
        std::vector<LayoutSFB> lSFB_h(n_groups);
        for (int g = 0; g < n_groups; ++g) {
            pB_h[g]   = (const ElementInput*)((const uint8_t*)B + (size_t)g * w_slab_bytes);
            pSFB_h[g] = (const ElementSF*)((const uint8_t*)SFB + (size_t)g * sfb_exp_bytes);
            dA_h[g]   = cutlass::make_cute_packed_stride(StrideA{}, {max_m_total, K, 1});
            dB_h[g]   = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            dD_h[g]   = cutlass::make_cute_packed_stride(StrideD{}, {max_m_total, N, 1});
            lSFA_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m_total, N, K, 1));
            lSFB_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(max_m_total, N, K, 1));
        }
        std::memcpy(staging.data() + h->off_pB,   pB_h.data(),   n_groups * sizeof(void*));
        std::memcpy(staging.data() + h->off_pSFB, pSFB_h.data(), n_groups * sizeof(void*));
        std::memcpy(staging.data() + off_dA,   dA_h.data(),   n_groups * sizeof(StrideA));
        std::memcpy(staging.data() + off_dB,   dB_h.data(),   n_groups * sizeof(StrideB));
        std::memcpy(staging.data() + off_dD,   dD_h.data(),   n_groups * sizeof(StrideD));
        std::memcpy(staging.data() + off_lSFA, lSFA_h.data(), n_groups * sizeof(LayoutSFA));
        std::memcpy(staging.data() + off_lSFB, lSFB_h.data(), n_groups * sizeof(LayoutSFB));
        if (alpha_vec) {
            std::vector<const float*> pAl_h(n_groups);
            for (int g = 0; g < n_groups; ++g) pAl_h[g] = (const float*)alpha_vec + g;
            std::memcpy(staging.data() + h->off_pAl, pAl_h.data(), n_groups * sizeof(float*));
        }
    }
    { size_t fmb=0,tmb=0; cudaMemGetInfo(&fmb,&tmb); cudaError_t er0=cudaMalloc((void**)&h->blob, blob_bytes);
      if (er0 != cudaSuccess) { fprintf(stderr,"[MTDIAG plain.prepare] blob cudaMalloc(%zu) FAIL: %s; free=%zuMiB N=%d K=%d ng=%d\n", blob_bytes, cudaGetErrorString(er0), fmb>>20, N, K, n_groups); delete h; return nullptr; } }
    { cudaError_t emc=cudaMemcpy(h->blob, staging.data(), blob_bytes, cudaMemcpyHostToDevice);
      if (emc != cudaSuccess) {
        cudaStreamCaptureStatus cs=cudaStreamCaptureStatusNone; cudaError_t eq=cudaStreamIsCapturing((cudaStream_t)0,&cs);
        fprintf(stderr,"[MTDIAG plain.prepare] blob memcpy FAIL: %s (err=%d); nullstream_capture_status=%d (q_err=%d) blob_bytes=%zu\n", cudaGetErrorString(emc), (int)emc, (int)cs, (int)eq, blob_bytes);
        cudaGetLastError(); // clear sticky
        cudaFree(h->blob); delete h; return nullptr;
      } }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = queried_sm_count();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {n_groups, (UnderlyingProblemShape*)(h->blob + h->off_ps), nullptr},
        {(const ElementA::DataType**)(h->blob + h->off_pA),  (StrideA*)(h->blob + off_dA),
         (const ElementB::DataType**)(h->blob + h->off_pB),  (StrideB*)(h->blob + off_dB),
         (const ElementSF**)(h->blob + h->off_pSFA), (LayoutSFA*)(h->blob + off_lSFA),
         (const ElementSF**)(h->blob + h->off_pSFB), (LayoutSFB*)(h->blob + off_lSFB)},
        {{},
         nullptr, (StrideC*)(h->blob + off_dD),
         (ElementD**)(h->blob + h->off_pD), (StrideD*)(h->blob + off_dD)},
        hw_info
    };
    if (alpha_vec) {
        args.epilogue.thread.alpha = 0.0f;
        args.epilogue.thread.alpha_ptr_array = (const float* const*)(h->blob + h->off_pAl);
        args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 1};
    } else {
        args.epilogue.thread.alpha = 1.0f;
    }
    args.epilogue.thread.beta = 0.0f;

    if (h->gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr,"[MTDIAG plain.prepare] can_implement FAIL N=%d K=%d ng=%d max_m=%d\n", N, K, n_groups, max_m_total);
        cudaFree(h->blob); delete h; return nullptr;
    }
    size_t ws = Gemm::get_workspace_size(args);
    { size_t fmb=0,tmb=0; cudaMemGetInfo(&fmb,&tmb);
      if (ws && cudaMalloc((void**)&h->work, ws) != cudaSuccess) {
        fprintf(stderr,"[MTDIAG plain.prepare] workspace cudaMalloc(%zu) FAIL free=%zuMiB\n", ws, fmb>>20);
        cudaFree(h->blob); delete h; return nullptr;
      } }
    if (h->gemm.initialize(args, h->work) != cutlass::Status::kSuccess) {
        fprintf(stderr,"[MTDIAG plain.prepare] initialize FAIL ws=%zu\n", ws);
        cudaFree(h->blob); if (h->work) cudaFree(h->work); delete h; return nullptr;
    }
    return h;
}

// Per-call: fill M-dependent sections from DEVICE offsets, then run. A/SFA/D
// must be the SAME base pointers across calls if used under graph capture.
extern "C" int moe_grouped_gemm_cutlass_fp4_run(
    void* handle, const void* A, const void* SFA, void* D,
    const void* off_dev, void* stream_v)
{
    auto* h = (Fp4GroupedHandle*)handle;
    if (!h) return 1;
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    int blocks = (h->n_groups + threads - 1) / threads;
    mt_fp4_fill_group_args<<<blocks, threads, 0, stream>>>(
        (const unsigned*)off_dev,
        (const uint8_t*)A, (const uint8_t*)SFA, (uint8_t*)D,
        (int*)(h->blob + h->off_ps),
        (const void**)(h->blob + h->off_pA),
        (const void**)(h->blob + h->off_pSFA),
        (void**)(h->blob + h->off_pD),
        h->n_groups, h->N, h->K);
    if (cudaGetLastError() != cudaSuccess) return 3;
    return h->gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 2;
}

extern "C" void moe_grouped_gemm_cutlass_fp4_release(void* handle)
{
    auto* h = (Fp4GroupedHandle*)handle;
    if (!h) return;
    if (h->blob) cudaFree(h->blob);
    if (h->work) cudaFree(h->work);
    delete h;
}

// ═══════════════ FUSED-ACTIVATION GEMM1 (NEMOTRON_FUSE_UPACT) ═══════════════
// Same device-descriptor prepared-handle shape as the plain path, but the
// epilogue is LinCombEltActBlockScaleFactor<SquaredReLU>: GEMM1 emits e2m1 D +
// ue4m3 SFD directly. The down-quant pass (relu2+amax+block-quant) is removed
// from the Rust caller; (D,SFD) feed straight into the down-GEMM as (A,SFA).
//
// SFD per-group offset: blk*512*ceil(N/64) with blk = Σ_{j<g} ceil(M_j/128) —
// the SAME formula the plain _run uses to read SFA, so (D,SFD) is a drop-in for
// the down-GEMM's input (proven bit-identical in the standalone harness).
namespace {

struct Fp4FusedHandle {
    g1f::Gemm gemm;
    uint8_t* blob = nullptr;
    uint8_t* work = nullptr;
    const float* norm_const = nullptr; // device ptr to gs (1/256)
    int n_groups = 0, N = 0, K = 0;
    size_t off_ps, off_pA, off_pB, off_pSFA, off_pSFB, off_pD, off_pSFD, off_pAl;
    typename g1f::Gemm::Arguments args; // saved for optional per-run re-initialize
};

} // namespace

// one thread per group: derive M-dependent sections (ps, pA, pSFA, pD, pSFD)
// from the device row offsets. pD/pSFD are the fused output (e2m1 + ue4m3).
__global__ void mt_fp4_fill_group_args_fused(
    const unsigned* __restrict__ off,   // [n_groups+1] device row offsets
    const uint8_t* A, const uint8_t* SFA, uint8_t* D, uint8_t* SFD,
    int* ps3,
    const void** pA, const void** pSFA, void** pD, void** pSFD,
    int n_groups, int N, int K)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const int m = (int)(off[g + 1] - off[g]);
    ps3[g * 3 + 0] = m; ps3[g * 3 + 1] = N; ps3[g * 3 + 2] = K;
    pA[g]  = A   + (size_t)off[g] * (K / 2);
    pD[g]  = D   + (size_t)off[g] * (size_t)(N / 2);   // e2m1: N/2 bytes/row
    size_t blk = 0;
    for (int j = 0; j < g; ++j) blk += (size_t)((off[j + 1] - off[j] + 127) / 128);
    pSFA[g] = SFA + blk * 512 * (size_t)((K + 63) / 64);
    pSFD[g] = SFD + blk * 512 * (size_t)((N + 63) / 64); // matches down-GEMM SFA read
}

// prepare once per (weight slab, n_groups, N, K). norm_constant_ptr = device
// ptr to the static gs (= 1/256), used as the per-tensor dequant scale folded
// into the stored output SF. max_m_total sizes the worst-case SF extents.
extern "C" void* moe_grouped_gemm_cutlass_fp4_FUSEDACT_prepare(
    const void* B, const void* SFB, const void* alpha_vec, const void* norm_constant_ptr,
    int n_groups, int N, int K, int max_m_total)
{
    if (K % 32 != 0 || N % 32 != 0) return nullptr;
    auto* h = new Fp4FusedHandle();
    h->n_groups = n_groups; h->N = N; h->K = K;
    h->norm_const = (const float*)norm_constant_ptr;
    const size_t w_slab_bytes  = (size_t)N * (size_t)K / 2;
    const size_t sfb_exp_bytes = (size_t)((N + 127) / 128) * 512 * (size_t)((K + 63) / 64);

    auto sec = [](size_t bytes) { return (bytes + 15) & ~(size_t)15; };
    h->off_ps   = 0;
    h->off_pA   = h->off_ps   + sec(n_groups * sizeof(UnderlyingProblemShape));
    h->off_pB   = h->off_pA   + sec(n_groups * sizeof(void*));
    h->off_pSFA = h->off_pB   + sec(n_groups * sizeof(void*));
    h->off_pSFB = h->off_pSFA + sec(n_groups * sizeof(void*));
    h->off_pD   = h->off_pSFB + sec(n_groups * sizeof(void*));
    h->off_pSFD = h->off_pD   + sec(n_groups * sizeof(void*));
    const size_t off_dA   = h->off_pSFD + sec(n_groups * sizeof(void*));
    const size_t off_dB   = off_dA   + sec(n_groups * sizeof(g1f::StrideA));
    const size_t off_dD   = off_dB   + sec(n_groups * sizeof(g1f::StrideB));
    const size_t off_lSFA = off_dD   + sec(n_groups * sizeof(g1f::StrideD));
    const size_t off_lSFB = off_lSFA + sec(n_groups * sizeof(g1f::LayoutSFA));
    h->off_pAl  = off_lSFB + sec(n_groups * sizeof(g1f::LayoutSFB));
    const size_t blob_bytes = h->off_pAl + sec(n_groups * sizeof(float*));

    std::vector<uint8_t> staging(blob_bytes, 0);
    {
        std::vector<const ElementInput*> pB_h(n_groups);
        std::vector<const g1f::ElementSF*> pSFB_h(n_groups);
        std::vector<g1f::StrideA> dA_h(n_groups);
        std::vector<g1f::StrideB> dB_h(n_groups);
        std::vector<g1f::StrideD> dD_h(n_groups);
        std::vector<g1f::LayoutSFA> lSFA_h(n_groups);
        std::vector<g1f::LayoutSFB> lSFB_h(n_groups);
        for (int g = 0; g < n_groups; ++g) {
            pB_h[g]   = (const ElementInput*)((const uint8_t*)B + (size_t)g * w_slab_bytes);
            pSFB_h[g] = (const g1f::ElementSF*)((const uint8_t*)SFB + (size_t)g * sfb_exp_bytes);
            dA_h[g]   = cutlass::make_cute_packed_stride(g1f::StrideA{}, {max_m_total, K, 1});
            dB_h[g]   = cutlass::make_cute_packed_stride(g1f::StrideB{}, {N, K, 1});
            dD_h[g]   = cutlass::make_cute_packed_stride(g1f::StrideD{}, {max_m_total, N, 1});
            lSFA_h[g] = g1f::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m_total, N, K, 1));
            lSFB_h[g] = g1f::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(max_m_total, N, K, 1));
        }
        std::memcpy(staging.data() + h->off_pB,   pB_h.data(),   n_groups * sizeof(void*));
        std::memcpy(staging.data() + h->off_pSFB, pSFB_h.data(), n_groups * sizeof(void*));
        std::memcpy(staging.data() + off_dA,   dA_h.data(),   n_groups * sizeof(g1f::StrideA));
        std::memcpy(staging.data() + off_dB,   dB_h.data(),   n_groups * sizeof(g1f::StrideB));
        std::memcpy(staging.data() + off_dD,   dD_h.data(),   n_groups * sizeof(g1f::StrideD));
        std::memcpy(staging.data() + off_lSFA, lSFA_h.data(), n_groups * sizeof(g1f::LayoutSFA));
        std::memcpy(staging.data() + off_lSFB, lSFB_h.data(), n_groups * sizeof(g1f::LayoutSFB));
        if (alpha_vec) {
            std::vector<const float*> pAl_h(n_groups);
            for (int g = 0; g < n_groups; ++g) pAl_h[g] = (const float*)alpha_vec + g;
            std::memcpy(staging.data() + h->off_pAl, pAl_h.data(), n_groups * sizeof(float*));
        }
    }
    if (cudaMalloc((void**)&h->blob, blob_bytes) != cudaSuccess) { delete h; return nullptr; }
    if (cudaMemcpy(h->blob, staging.data(), blob_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = queried_sm_count();
    typename g1f::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {n_groups, (UnderlyingProblemShape*)(h->blob + h->off_ps), nullptr},
        {(const ElementA::DataType**)(h->blob + h->off_pA),  (g1f::StrideA*)(h->blob + off_dA),
         (const ElementB::DataType**)(h->blob + h->off_pB),  (g1f::StrideB*)(h->blob + off_dB),
         (const g1f::ElementSF**)(h->blob + h->off_pSFA), (g1f::LayoutSFA*)(h->blob + off_lSFA),
         (const g1f::ElementSF**)(h->blob + h->off_pSFB), (g1f::LayoutSFB*)(h->blob + off_lSFB)},
        {{},
         nullptr, (g1f::StrideC*)(h->blob + off_dD),
         (g1f::ElementD**)(h->blob + h->off_pD), (g1f::StrideD*)(h->blob + off_dD)},
        hw_info
    };
    // CRITICAL: any per-group alpha (alpha_ptr_array OR alpha_ptr+stride) SILENTLY
    // DROPS the block-scale-factor (SFD) output store in this CUTLASS version
    // (empirically verified: per-group alpha => SF all-zero; scalar alpha => SF
    // written ~1M nonzero). So the fused up-GEMM MUST run with SCALAR alpha=1.
    // The per-group up scale (act_global*up_expert_global) is instead folded into
    // the DOWN-GEMM's per-group alpha as alpha_u[g]^2 (the squared-relu squares it)
    // by the Rust caller via fp4_group_alpha_fused_down. alpha_vec is unused here.
    (void)alpha_vec;
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    // block-scaled output fusion: per-group SFD ptr array + PER-GROUP norm_constant.
    // norm_constant_ptr = device float[n_groups] gs_pe (gs_pe[e]=ugw[e]^2/256);
    // dNormConst L-stride 1 makes the epilogue read gs_pe[g] per group. This is
    // the per-expert output block-scale normalization that lets the fused GEMM1
    // run with SCALAR alpha=1 (required for the SFD store to write) while still
    // emitting correctly-ranged ue4m3 SFs.
    args.epilogue.thread.block_scale_factor_ptr = (g1f::ElementSFType**)(h->blob + h->off_pSFD);
    args.epilogue.thread.norm_constant_ptr = h->norm_const;
    args.epilogue.thread.dNormConst = {cute::_0{}, cute::_0{}, 1};
    h->args = args; // save for per-run re-initialize (refreshes fusion ptr snapshot)

    if (h->gemm.can_implement(args) != cutlass::Status::kSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }
    size_t ws = g1f::Gemm::get_workspace_size(args);
    if (ws && cudaMalloc((void**)&h->work, ws) != cudaSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }
    if (h->gemm.initialize(args, h->work) != cutlass::Status::kSuccess) {
        cudaFree(h->blob); if (h->work) cudaFree(h->work); delete h; return nullptr;
    }
    return h;
}

// per-call: fill M-dependent sections from DEVICE offsets, run. D = e2m1 out
// (mt*N/2 bytes), SFD = ue4m3 out (worst-case (mt/128+n_groups)*512*ceil(N/64)).
extern "C" int moe_grouped_gemm_cutlass_fp4_FUSEDACT_run(
    void* handle, const void* A, const void* SFA, void* D, void* SFD,
    const void* off_dev, void* stream_v)
{
    auto* h = (Fp4FusedHandle*)handle;
    if (!h) return 1;
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    int blocks = (h->n_groups + threads - 1) / threads;
    mt_fp4_fill_group_args_fused<<<blocks, threads, 0, stream>>>(
        (const unsigned*)off_dev,
        (const uint8_t*)A, (const uint8_t*)SFA, (uint8_t*)D, (uint8_t*)SFD,
        (int*)(h->blob + h->off_ps),
        (const void**)(h->blob + h->off_pA),
        (const void**)(h->blob + h->off_pSFA),
        (void**)(h->blob + h->off_pD),
        (void**)(h->blob + h->off_pSFD),
        h->n_groups, h->N, h->K);
    if (cudaGetLastError() != cudaSuccess) return 3;
    // DIAGNOSTIC (MT_FUSE_REINIT=1): re-initialize after the fill kernel so the
    // fusion block_scale_factor_ptr snapshot reflects the now-populated pSFD blob.
    // Confirms the init-time-snapshot theory (NOT graph-safe).
    static int reinit = -1;
    if (reinit < 0) { const char* e = getenv("MT_FUSE_REINIT"); reinit = (e && e[0]=='1') ? 1 : 0; }
    if (reinit) {
        if (h->gemm.initialize(h->args, h->work, stream) != cutlass::Status::kSuccess) return 4;
    }
    return h->gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 2;
}

extern "C" void moe_grouped_gemm_cutlass_fp4_FUSEDACT_release(void* handle)
{
    auto* h = (Fp4FusedHandle*)handle;
    if (!h) return;
    if (h->blob) cudaFree(h->blob);
    if (h->work) cudaFree(h->work);
    delete h;
}

// ═══════════════ AMAX-IN-EPILOGUE GEMM1 (NEMOTRON_AMAX_EPI) ══════════════════
// Same device-descriptor prepared-handle shape as the plain path: GEMM1 emits
// f16 D = SquaredReLU(acc)·(1/256) (relu²'d a2) AND, via the Sm90ScalarReduction
// EVT node, a per-group amax (float[n_groups]) of those activated values. The
// down-quant reads a2 ONCE — the per-group amax (its max = the per-tensor global)
// replaces the separate amax scan. d_amax is a PERSISTENT device buffer (stable
// address → graph-safe); the Rust caller zeroes it before each run (the build
// uses -DCUTLASS_SKIP_REDUCTION_INIT=1, so the kernel does NOT self-init it).
namespace {

struct Fp4AmaxHandle {
    g1a::Gemm gemm;
    uint8_t* blob = nullptr;
    uint8_t* work = nullptr;
    float* d_amax = nullptr;   // device float[n_groups], caller-owned, zeroed per run
    int n_groups = 0, N = 0, K = 0;
    size_t off_ps, off_pA, off_pB, off_pSFA, off_pSFB, off_pD, off_pAl;
};

} // namespace

extern "C" void* moe_grouped_gemm_cutlass_fp4_AMAX_prepare(
    const void* B, const void* SFB, const void* d_amax,
    int n_groups, int N, int K, int max_m_total)
{
    if (K % 32 != 0 || N % 32 != 0) return nullptr;
    auto* h = new Fp4AmaxHandle();
    h->n_groups = n_groups; h->N = N; h->K = K;
    h->d_amax = (float*)d_amax;
    const size_t w_slab_bytes  = (size_t)N * (size_t)K / 2;
    const size_t sfb_exp_bytes = (size_t)((N + 127) / 128) * 512 * (size_t)((K + 63) / 64);

    auto sec = [](size_t bytes) { return (bytes + 15) & ~(size_t)15; };
    h->off_ps   = 0;
    h->off_pA   = h->off_ps   + sec(n_groups * sizeof(UnderlyingProblemShape));
    h->off_pB   = h->off_pA   + sec(n_groups * sizeof(void*));
    h->off_pSFA = h->off_pB   + sec(n_groups * sizeof(void*));
    h->off_pSFB = h->off_pSFA + sec(n_groups * sizeof(void*));
    h->off_pD   = h->off_pSFB + sec(n_groups * sizeof(void*));
    const size_t off_dA   = h->off_pD + sec(n_groups * sizeof(void*));
    const size_t off_dB   = off_dA   + sec(n_groups * sizeof(g1a::StrideA));
    const size_t off_dD   = off_dB   + sec(n_groups * sizeof(g1a::StrideB));
    const size_t off_lSFA = off_dD   + sec(n_groups * sizeof(g1a::StrideD));
    const size_t off_lSFB = off_lSFA + sec(n_groups * sizeof(g1a::LayoutSFA));
    h->off_pAl  = off_lSFB + sec(n_groups * sizeof(g1a::LayoutSFB));
    const size_t blob_bytes = h->off_pAl + sec(n_groups * sizeof(float*));

    std::vector<uint8_t> staging(blob_bytes, 0);
    {
        std::vector<const ElementInput*> pB_h(n_groups);
        std::vector<const g1a::ElementSF*> pSFB_h(n_groups);
        std::vector<g1a::StrideA> dA_h(n_groups);
        std::vector<g1a::StrideB> dB_h(n_groups);
        std::vector<g1a::StrideD> dD_h(n_groups);
        std::vector<g1a::LayoutSFA> lSFA_h(n_groups);
        std::vector<g1a::LayoutSFB> lSFB_h(n_groups);
        for (int g = 0; g < n_groups; ++g) {
            pB_h[g]   = (const ElementInput*)((const uint8_t*)B + (size_t)g * w_slab_bytes);
            pSFB_h[g] = (const g1a::ElementSF*)((const uint8_t*)SFB + (size_t)g * sfb_exp_bytes);
            dA_h[g]   = cutlass::make_cute_packed_stride(g1a::StrideA{}, {max_m_total, K, 1});
            dB_h[g]   = cutlass::make_cute_packed_stride(g1a::StrideB{}, {N, K, 1});
            dD_h[g]   = cutlass::make_cute_packed_stride(g1a::StrideD{}, {max_m_total, N, 1});
            lSFA_h[g] = g1a::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m_total, N, K, 1));
            lSFB_h[g] = g1a::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(max_m_total, N, K, 1));
        }
        std::memcpy(staging.data() + h->off_pB,   pB_h.data(),   n_groups * sizeof(void*));
        std::memcpy(staging.data() + h->off_pSFB, pSFB_h.data(), n_groups * sizeof(void*));
        std::memcpy(staging.data() + off_dA,   dA_h.data(),   n_groups * sizeof(g1a::StrideA));
        std::memcpy(staging.data() + off_dB,   dB_h.data(),   n_groups * sizeof(g1a::StrideB));
        std::memcpy(staging.data() + off_dD,   dD_h.data(),   n_groups * sizeof(g1a::StrideD));
        std::memcpy(staging.data() + off_lSFA, lSFA_h.data(), n_groups * sizeof(g1a::LayoutSFA));
        std::memcpy(staging.data() + off_lSFB, lSFB_h.data(), n_groups * sizeof(g1a::LayoutSFB));
    }
    if (cudaMalloc((void**)&h->blob, blob_bytes) != cudaSuccess) { delete h; return nullptr; }
    if (cudaMemcpy(h->blob, staging.data(), blob_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = queried_sm_count();
    typename g1a::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {n_groups, (UnderlyingProblemShape*)(h->blob + h->off_ps), nullptr},
        {(const ElementA::DataType**)(h->blob + h->off_pA),  (g1a::StrideA*)(h->blob + off_dA),
         (const ElementB::DataType**)(h->blob + h->off_pB),  (g1a::StrideB*)(h->blob + off_dB),
         (const g1a::ElementSF**)(h->blob + h->off_pSFA), (g1a::LayoutSFA*)(h->blob + off_lSFA),
         (const g1a::ElementSF**)(h->blob + h->off_pSFB), (g1a::LayoutSFB*)(h->blob + off_lSFB)},
        {{},
         nullptr, (g1a::StrideC*)(h->blob + off_dD),
         (g1a::ElementD**)(h->blob + h->off_pD), (g1a::StrideD*)(h->blob + off_dD)},
        hw_info
    };
    // EVT thread args (nested tuple) — exact validated-scaffold form:
    //   tree = Sm90EVT<ScalarReduction, Sm90EVT<Compute<SquaredReLU>, LinComb>>
    //   inner Sm90LinearCombination : {{alpha=1},{alpha_ptr=null},{}}, beta {}, ...
    //   outer Sm90ScalarReduction   : { d_amax(float* len n_groups), identity 0,
    //                                   dScalar L-stride 1 } → group g writes d_amax[g]
    args.epilogue.thread = {
        { { { {0.0f}, {nullptr}, {} }, {}, { { {1.0f}, {nullptr}, {} }, {}, {} }, {} }, {} },
        { h->d_amax, 0.0f, {cute::_0{}, cute::_0{}, 1} }
    };

    if (h->gemm.can_implement(args) != cutlass::Status::kSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }
    size_t ws = g1a::Gemm::get_workspace_size(args);
    if (ws && cudaMalloc((void**)&h->work, ws) != cudaSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }
    if (h->gemm.initialize(args, h->work) != cutlass::Status::kSuccess) {
        cudaFree(h->blob); if (h->work) cudaFree(h->work); delete h; return nullptr;
    }
    return h;
}

// per-call: fill M-dependent sections from DEVICE offsets, run. D = f16 a2 out
// (mt*N*2 bytes, relu²'d). The caller MUST cudaMemsetAsync(d_amax, 0,
// n_groups*sizeof(float), stream) before this call (SKIP_REDUCTION_INIT build).
extern "C" int moe_grouped_gemm_cutlass_fp4_AMAX_run(
    void* handle, const void* A, const void* SFA, void* D,
    const void* off_dev, void* stream_v)
{
    auto* h = (Fp4AmaxHandle*)handle;
    if (!h) return 1;
    cudaStream_t stream = (cudaStream_t)stream_v;
    // REQUIRED (SKIP_REDUCTION_INIT build): zero the per-group amax before the
    // atomicMax accumulation. Stream-ordered → captured into a CUDA graph.
    if (cudaMemsetAsync(h->d_amax, 0, (size_t)h->n_groups * sizeof(float), stream) != cudaSuccess) return 4;
    int threads = 128;
    int blocks = (h->n_groups + threads - 1) / threads;
    mt_fp4_fill_group_args<<<blocks, threads, 0, stream>>>(
        (const unsigned*)off_dev,
        (const uint8_t*)A, (const uint8_t*)SFA, (uint8_t*)D,
        (int*)(h->blob + h->off_ps),
        (const void**)(h->blob + h->off_pA),
        (const void**)(h->blob + h->off_pSFA),
        (void**)(h->blob + h->off_pD),
        h->n_groups, h->N, h->K);
    if (cudaGetLastError() != cudaSuccess) return 3;
    return h->gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 2;
}

extern "C" void moe_grouped_gemm_cutlass_fp4_AMAX_release(void* handle)
{
    auto* h = (Fp4AmaxHandle*)handle;
    if (!h) return;
    if (h->blob) cudaFree(h->blob);
    if (h->work) cudaFree(h->work);
    delete h;
}

#else // !CUTLASS_ARCH_MMA_SM120_SUPPORTED && !CUTLASS_ARCH_MMA_SM121_SUPPORTED

extern "C" int moe_grouped_gemm_cutlass_fp4(
    const void*, const void*, const void*, const void*, void*,
    const int*, const int*, const long long*, const void*, int, int, int, void*)
{
    return 100; // built without sm_120a/sm_121a block-scaled mma support
}

extern "C" void* moe_grouped_gemm_cutlass_fp4_prepare(
    const void*, const void*, const void*, int, int, int, int) { return nullptr; }
extern "C" int moe_grouped_gemm_cutlass_fp4_run(
    void*, const void*, const void*, void*, const void*, void*) { return 100; }
extern "C" void moe_grouped_gemm_cutlass_fp4_release(void*) {}

extern "C" void* moe_grouped_gemm_cutlass_fp4_FUSEDACT_prepare(
    const void*, const void*, const void*, const void*, int, int, int, int) { return nullptr; }
extern "C" int moe_grouped_gemm_cutlass_fp4_FUSEDACT_run(
    void*, const void*, const void*, void*, void*, const void*, void*) { return 100; }
extern "C" void moe_grouped_gemm_cutlass_fp4_FUSEDACT_release(void*) {}

extern "C" void* moe_grouped_gemm_cutlass_fp4_AMAX_prepare(
    const void*, const void*, const void*, int, int, int, int) { return nullptr; }
extern "C" int moe_grouped_gemm_cutlass_fp4_AMAX_run(
    void*, const void*, const void*, void*, const void*, void*) { return 100; }
extern "C" void moe_grouped_gemm_cutlass_fp4_AMAX_release(void*) {}

#endif

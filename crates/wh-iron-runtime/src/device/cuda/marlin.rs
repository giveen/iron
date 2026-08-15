//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Marlin U4B8 (symmetric, bias-8) small-M W4A16 GEMM entry points for
//! `CudaDevice`. F-85 round-8: wires the round-7-verified-correct
//! (`../../../cuda/marlin/marlin_shim.cu`, max_rel <=5.3e-4 vs f64 ref on
//! all 4 dense-verify-GEMM shapes x M in {1..8}) csrc into real Rust FFI.
//! AOT-linked (`cfg(have_marlin)` via build.rs, opt-in behind
//! `IRON_MARLIN_BUILD=1` -- no external SDK dependency, the csrc is fully
//! in-tree). Kept in a dedicated module, mirroring `nvfp4_moe.rs`, so the
//! marlin surface stays separate from the core device lifecycle in `mod.rs`.
#![allow(clippy::too_many_arguments)]

#[cfg(have_marlin)]
use std::os::raw::{c_int, c_void};

use super::{CUdeviceptr, CudaDevice};
use crate::error::IronError;

impl CudaDevice {
    /// Repack one weight's GPTQ-packed `[size_k/8, size_n]` u32 (u4, K-major,
    /// 8 values/word) weights into the marlin tile layout (`[size_k/16,
    /// size_n*16/8]` u32 out). One-time, per weight-load.
    pub fn marlin_repack(
        &self,
        b_q_weight: CUdeviceptr,
        out: CUdeviceptr,
        size_k: usize,
        size_n: usize,
        sms: i32,
        max_shared_mem: i32,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_marlin)]
        {
            unsafe extern "C" {
                fn ffai_marlin_repack(
                    b_q_weight: *const c_void,
                    out: *mut c_void,
                    size_k: c_int,
                    size_n: c_int,
                    sms: c_int,
                    max_shared_mem: c_int,
                    stream: *mut c_void,
                );
            }
            unsafe {
                ffai_marlin_repack(
                    b_q_weight as *const c_void,
                    out as *mut c_void,
                    size_k as c_int,
                    size_n as c_int,
                    sms,
                    max_shared_mem,
                    self.stream as *mut c_void,
                );
            }
            Ok(())
        }
        #[cfg(not(have_marlin))]
        {
            let _ = (b_q_weight, out, size_k, size_n, sms, max_shared_mem);
            Err(IronError::Dispatch(
                "marlin_repack: runtime built without marlin (set IRON_MARLIN_BUILD=1)".into(),
            ))
        }
    }

    /// One-time scale-column permutation companion to [`marlin_repack`]
    /// (round-7's correctness fix -- see the marlin_shim.cu status header).
    /// `in_`/`out` are `[num_groups, size_n]` f16 device buffers; `out` may
    /// not alias `in_`.
    pub fn marlin_permute_scales(
        &self,
        in_: CUdeviceptr,
        out: CUdeviceptr,
        num_groups: usize,
        size_n: usize,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_marlin)]
        {
            unsafe extern "C" {
                fn ffai_marlin_permute_scales(
                    in_: *const c_void,
                    out: *mut c_void,
                    num_groups: c_int,
                    size_n: c_int,
                    stream: *mut c_void,
                );
            }
            unsafe {
                ffai_marlin_permute_scales(
                    in_ as *const c_void,
                    out as *mut c_void,
                    num_groups as c_int,
                    size_n as c_int,
                    self.stream as *mut c_void,
                );
            }
            Ok(())
        }
        #[cfg(not(have_marlin))]
        {
            let _ = (in_, out, num_groups, size_n);
            Err(IronError::Dispatch(
                "marlin_permute_scales: runtime built without marlin (set IRON_MARLIN_BUILD=1)"
                    .into(),
            ))
        }
    }

    /// On-device build of marlin routing arrays from per-"expert" offsets.
    /// For the dense (non-MoE) verify/decode GEMM, called with `n_exp=1`,
    /// `off={0,M}`, `mt=M` -- one "expert" owning all M rows.
    pub fn marlin_build_routing(
        &self,
        off: CUdeviceptr,
        n_exp: usize,
        blk: usize,
        mt: usize,
        stid: CUdeviceptr,
        eid: CUdeviceptr,
        ntpp: CUdeviceptr,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_marlin)]
        {
            unsafe extern "C" {
                fn ffai_marlin_build_routing(
                    off: *const c_int,
                    n_exp: c_int,
                    blk: c_int,
                    mt: c_int,
                    stid: *mut c_int,
                    eid: *mut c_int,
                    ntpp: *mut c_int,
                    stream: *mut c_void,
                );
            }
            unsafe {
                ffai_marlin_build_routing(
                    off as *const c_int,
                    n_exp as c_int,
                    blk as c_int,
                    mt as c_int,
                    stid as *mut c_int,
                    eid as *mut c_int,
                    ntpp as *mut c_int,
                    self.stream as *mut c_void,
                );
            }
            Ok(())
        }
        #[cfg(not(have_marlin))]
        {
            let _ = (off, n_exp, blk, mt, stid, eid, ntpp);
            Err(IronError::Dispatch(
                "marlin_build_routing: runtime built without marlin (set IRON_MARLIN_BUILD=1)"
                    .into(),
            ))
        }
    }

    /// Small-M symmetric-u4 (kU4B8) W4A16 GEMM: `a` f16 `[m,k]` activations,
    /// `b_repacked` = [`marlin_repack`] output of a `[k/8,n]` u32 GPTQ-packed
    /// u4b8 weight, `b_scales` = [`marlin_permute_scales`] output (permuted
    /// `[k/group_size,n]` f16 per-group scales), `c` = f16 out `[m,n]`.
    /// `sorted_token_ids`/`expert_ids`/`num_tokens_past_padded` come from
    /// [`marlin_build_routing`] called once with a trivial single-"expert"
    /// `[0,m)` range. `workspace` = int32 locks buffer, `c_tmp` = f32
    /// scratch -- see the marlin_shim.cu doc comment / round-7 probe harness
    /// for the exact sizing formulas (mirrored in `wh-butter-ops`' caller).
    pub fn marlin_gemm_u4b8_f16(
        &self,
        a: CUdeviceptr,
        b_repacked: CUdeviceptr,
        c: CUdeviceptr,
        c_tmp: CUdeviceptr,
        b_scales: CUdeviceptr,
        sorted_token_ids: CUdeviceptr,
        expert_ids: CUdeviceptr,
        num_tokens_past_padded: CUdeviceptr,
        workspace: CUdeviceptr,
        prob_m: usize,
        prob_n: usize,
        prob_k: usize,
        num_groups: usize,
        group_size: usize,
        sms: i32,
        use_fp32_reduce: bool,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_marlin)]
        {
            unsafe extern "C" {
                fn ffai_marlin_gemm_u4b8_f16(
                    a: *const c_void,
                    b_repacked: *const c_void,
                    c: *mut c_void,
                    c_tmp: *mut c_void,
                    b_scales: *const c_void,
                    sorted_token_ids: *const c_void,
                    expert_ids: *const c_void,
                    num_tokens_past_padded: *const c_void,
                    workspace: *mut c_void,
                    prob_m: c_int,
                    prob_n: c_int,
                    prob_k: c_int,
                    num_groups: c_int,
                    group_size: c_int,
                    sms: c_int,
                    use_fp32_reduce: c_int,
                    stream: *mut c_void,
                );
            }
            unsafe {
                ffai_marlin_gemm_u4b8_f16(
                    a as *const c_void,
                    b_repacked as *const c_void,
                    c as *mut c_void,
                    c_tmp as *mut c_void,
                    b_scales as *const c_void,
                    sorted_token_ids as *const c_void,
                    expert_ids as *const c_void,
                    num_tokens_past_padded as *const c_void,
                    workspace as *mut c_void,
                    prob_m as c_int,
                    prob_n as c_int,
                    prob_k as c_int,
                    num_groups as c_int,
                    group_size as c_int,
                    sms,
                    if use_fp32_reduce { 1 } else { 0 },
                    self.stream as *mut c_void,
                );
            }
            Ok(())
        }
        #[cfg(not(have_marlin))]
        {
            let _ = (
                a,
                b_repacked,
                c,
                c_tmp,
                b_scales,
                sorted_token_ids,
                expert_ids,
                num_tokens_past_padded,
                workspace,
                prob_m,
                prob_n,
                prob_k,
                num_groups,
                group_size,
                sms,
                use_fp32_reduce,
            );
            Err(IronError::Dispatch(
                "marlin_gemm_u4b8_f16: runtime built without marlin (set IRON_MARLIN_BUILD=1)"
                    .into(),
            ))
        }
    }
}

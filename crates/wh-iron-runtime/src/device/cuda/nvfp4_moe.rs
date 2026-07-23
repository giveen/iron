//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! NVFP4 / FP8 cutlass grouped-MoE GEMM entry points for `CudaDevice`.
//!
//! Carved out of the kitchen-sink CUDA work and re-architected onto the current
//! backend: additive `impl CudaDevice` methods wrapping the AOT-compiled cutlass
//! grouped kernels (`cuda/cutlass_moe_fp4.cu`, gated behind `CUTLASS_DIR` /
//! `cfg(have_cutlass)` in build.rs) plus the cuBLASLt fp4/fp8 GEMM escape hatch.
//! Kept in a dedicated module so the NVFP4 surface stays separate from the core
//! device lifecycle in `mod.rs`.
//!
//! Two lints are relaxed module-wide to match the C side: `too_many_arguments`
//! (the GEMM/MoE wrappers mirror the cutlass / cuBLASLt C ABI parameter lists)
//! and `non_snake_case` (a few entry points embed the cutlass kernel variant
//! tags FUSEDACT / AMAX verbatim so the Rust name maps 1:1 to the CUDA variant).
#![allow(clippy::too_many_arguments, non_snake_case)]

use std::{
    os::raw::{c_int, c_void},
    ptr,
};

use super::{CUBLASLT_WORKSPACE_BYTES, CUdeviceptr, CudaDevice, cu_check, ffi::*};
use crate::error::IronError;

impl CudaDevice {
    /// Lazily create the cublasLt handle + fixed device workspace. Returns
    /// `(handle, workspace_ptr)`.
    fn lt_beta_zero_ptr(&self) -> Result<CUdeviceptr, IronError> {
        let mut g = self.lt_beta_zero.lock().unwrap();
        if *g == 0 {
            let p = self.alloc_raw(4)?;
            cu_check(
                unsafe { cuMemsetD8Async(p, 0, 4, self.stream) },
                "cuMemsetD8Async(lt_beta_zero)",
            )?;
            *g = p as usize;
        }
        Ok(*g as CUdeviceptr)
    }

    /// CUTLASS grouped block-scaled NVFP4 MoE GEMM (AOT-linked, sm_120a/121a).
    /// `a` = packed e2m1 sorted-token activations `[mt, K/2]` bytes, `sfa` =
    /// per-group ue4m3 scale pool (group g's blob at byte `sfa_off[g]`, laid out
    /// for the group-LOCAL row), `w`/`sfw` = packed e2m1 + scale expert slabs
    /// (`[n_exp, N*K/2]` / `[n_exp, ceil(N/128)*512*ceil(K/64)]` bytes), `c` =
    /// f16 out `[mt, N]`. `group_rows`/`expert_ids`/`sfa_off` are HOST slices.
    /// `out[t,n] = Σ_k a[t,k]·w[eid][n,k]` with per-16-block scales on both
    /// operands. Errors if the runtime was built without CUTLASS.
    #[allow(clippy::too_many_arguments)] // mirrors the C entry point's signature
    pub fn moe_grouped_cutlass_fp4(
        &self,
        a: CUdeviceptr,
        sfa: CUdeviceptr,
        w: CUdeviceptr,
        sfw: CUdeviceptr,
        c: CUdeviceptr,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
        alpha_vec: CUdeviceptr, // device f32[n_groups] per-group scales, 0 = none
        n: usize,
        k: usize,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4(
                    a: *const c_void,
                    sfa: *const c_void,
                    w: *const c_void,
                    sfw: *const c_void,
                    c: *mut c_void,
                    group_rows: *const c_int,
                    expert_ids: *const c_int,
                    sfa_off: *const i64,
                    alpha_vec: *const c_void,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                ) -> c_int;
            }
            if group_rows.len() != expert_ids.len() || group_rows.len() != sfa_off.len() {
                return Err(IronError::Dispatch(
                    "moe_grouped_cutlass_fp4: group_rows/expert_ids/sfa_off len mismatch".into(),
                ));
            }
            let r = unsafe {
                moe_grouped_gemm_cutlass_fp4(
                    a as *const c_void,
                    sfa as *const c_void,
                    w as *const c_void,
                    sfw as *const c_void,
                    c as *mut c_void,
                    group_rows.as_ptr(),
                    expert_ids.as_ptr(),
                    sfa_off.as_ptr(),
                    alpha_vec as *const c_void,
                    group_rows.len() as c_int,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(IronError::Dispatch(format!(
                    "moe_grouped_gemm_cutlass_fp4 failed: code {r}"
                )));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (a, sfa, w, sfw, c, group_rows, expert_ids, sfa_off, alpha_vec, n, k);
            Err(IronError::Dispatch(
                "moe_grouped_cutlass_fp4: runtime built without CUTLASS (set CUTLASS_DIR)".into(),
            ))
        }
    }

    /// W4A8 grouped GEMM (fp8 e4m3 acts × fp4 e2m1 weights, mxf8f6f4). Mirrors
    /// moe_grouped_cutlass_fp4 + sfb_exp_bytes (per-expert SFB stride from w4a8_packw_run).
    pub fn moe_grouped_cutlass_w4a8(
        &self,
        a: CUdeviceptr,
        sfa: CUdeviceptr,
        w: CUdeviceptr,
        sfw: CUdeviceptr,
        c: CUdeviceptr,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
        sfb_exp_bytes: i64,
        alpha_vec: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_w4a8(
                    a: *const c_void,
                    sfa: *const c_void,
                    w: *const c_void,
                    sfw: *const c_void,
                    c: *mut c_void,
                    group_rows: *const c_int,
                    expert_ids: *const c_int,
                    sfa_off: *const i64,
                    sfb_exp_bytes: i64,
                    alpha_vec: *const c_void,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                ) -> c_int;
            }
            if group_rows.len() != expert_ids.len() || group_rows.len() != sfa_off.len() {
                return Err(IronError::Dispatch("moe_grouped_cutlass_w4a8: len mismatch".into()));
            }
            let r = unsafe {
                moe_grouped_gemm_w4a8(
                    a as *const c_void,
                    sfa as *const c_void,
                    w as *const c_void,
                    sfw as *const c_void,
                    c as *mut c_void,
                    group_rows.as_ptr(),
                    expert_ids.as_ptr(),
                    sfa_off.as_ptr(),
                    sfb_exp_bytes,
                    alpha_vec as *const c_void,
                    group_rows.len() as c_int,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(IronError::Dispatch(format!("moe_grouped_gemm_w4a8 failed: {r}")));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (
                a,
                sfa,
                w,
                sfw,
                c,
                group_rows,
                expert_ids,
                sfa_off,
                sfb_exp_bytes,
                alpha_vec,
                n,
                k,
            );
            Err(IronError::Dispatch("moe_grouped_cutlass_w4a8: built without CUTLASS".into()))
        }
    }

    /// W4A8 fp8 act-quant: x[mt,K] half -> out[mt,K] e4m3 + sf ue8m0 (per-32, group-aware SFA).
    /// group_rows / sfa_off are HOST arrays (per-group SFA section offsets, atom-padded).
    pub fn w4a8_actquant_run(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        sf: CUdeviceptr,
        group_rows: &[i32],
        sfa_off: &[i64],
        n: usize,
        k: usize,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn w4a8_actquant(
                    x: *const c_void,
                    out: *mut c_void,
                    sf: *mut c_void,
                    group_rows: *const c_int,
                    sfa_off: *const ::std::os::raw::c_longlong,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                );
            }
            let n_groups = group_rows.len() as c_int;
            unsafe {
                w4a8_actquant(
                    x as *const c_void,
                    out as *mut c_void,
                    sf as *mut c_void,
                    group_rows.as_ptr(),
                    sfa_off.as_ptr() as *const ::std::os::raw::c_longlong,
                    n_groups,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                );
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (x, out, sf, group_rows, sfa_off, n, k);
            Err(IronError::Dispatch("w4a8_actquant: built without CUTLASS".into()))
        }
    }

    /// W4A8 mxfp4 weight pack: w[n_exp,N,K] half -> outp e2m1 + sf ue8m0 (per-32, SFB).
    /// Returns the per-expert SFB element count (bytes; size the SF buffer + stride).
    pub fn w4a8_packw_run(
        &self,
        w: CUdeviceptr,
        outp: CUdeviceptr,
        sf: CUdeviceptr,
        n_exp: usize,
        n: usize,
        k: usize,
    ) -> Result<i64, IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn w4a8_packw(
                    w: *const c_void,
                    outp: *mut c_void,
                    sf: *mut c_void,
                    n_exp: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                ) -> i64;
            }
            Ok(unsafe {
                w4a8_packw(
                    w as *const c_void,
                    outp as *mut c_void,
                    sf as *mut c_void,
                    n_exp as c_int,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                )
            })
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (w, outp, sf, n_exp, n, k);
            Err(IronError::Dispatch("w4a8_packw: built without CUTLASS".into()))
        }
    }

    pub fn moe_grouped_cutlass_w8a8(
        &self,
        a: CUdeviceptr,
        sfa: CUdeviceptr,
        w: CUdeviceptr,
        sfw: CUdeviceptr,
        c: CUdeviceptr,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
        sfb_exp_bytes: i64,
        alpha_vec: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_w8a8(
                    a: *const c_void,
                    sfa: *const c_void,
                    w: *const c_void,
                    sfw: *const c_void,
                    c: *mut c_void,
                    group_rows: *const c_int,
                    expert_ids: *const c_int,
                    sfa_off: *const i64,
                    sfb_exp_bytes: i64,
                    alpha_vec: *const c_void,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                ) -> c_int;
            }
            if group_rows.len() != expert_ids.len() || group_rows.len() != sfa_off.len() {
                return Err(IronError::Dispatch("moe_grouped_cutlass_w8a8: len mismatch".into()));
            }
            let r = unsafe {
                moe_grouped_gemm_w8a8(
                    a as *const c_void,
                    sfa as *const c_void,
                    w as *const c_void,
                    sfw as *const c_void,
                    c as *mut c_void,
                    group_rows.as_ptr(),
                    expert_ids.as_ptr(),
                    sfa_off.as_ptr(),
                    sfb_exp_bytes,
                    alpha_vec as *const c_void,
                    group_rows.len() as c_int,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(IronError::Dispatch(format!("moe_grouped_gemm_w8a8 failed: {r}")));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (
                a,
                sfa,
                w,
                sfw,
                c,
                group_rows,
                expert_ids,
                sfa_off,
                sfb_exp_bytes,
                alpha_vec,
                n,
                k,
            );
            Err(IronError::Dispatch("moe_grouped_cutlass_w8a8: built without CUTLASS".into()))
        }
    }

    /// W4A8 fp8 act-quant: x[mt,K] half -> out[mt,K] e4m3 + sf ue8m0 (per-32, group-aware SFA).
    /// group_rows / sfa_off are HOST arrays (per-group SFA section offsets, atom-padded).
    pub fn w8a8_actquant_run(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        sf: CUdeviceptr,
        group_rows: &[i32],
        sfa_off: &[i64],
        n: usize,
        k: usize,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn w8a8_actquant(
                    x: *const c_void,
                    out: *mut c_void,
                    sf: *mut c_void,
                    group_rows: *const c_int,
                    sfa_off: *const ::std::os::raw::c_longlong,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                );
            }
            let n_groups = group_rows.len() as c_int;
            unsafe {
                w8a8_actquant(
                    x as *const c_void,
                    out as *mut c_void,
                    sf as *mut c_void,
                    group_rows.as_ptr(),
                    sfa_off.as_ptr() as *const ::std::os::raw::c_longlong,
                    n_groups,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                );
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (x, out, sf, group_rows, sfa_off, n, k);
            Err(IronError::Dispatch("w8a8_actquant: built without CUTLASS".into()))
        }
    }

    /// W4A8 mxfp4 weight pack: w[n_exp,N,K] half -> outp e2m1 + sf ue8m0 (per-32, SFB).
    /// Returns the per-expert SFB element count (bytes; size the SF buffer + stride).
    pub fn w8a8_packw_run(
        &self,
        w: CUdeviceptr,
        outp: CUdeviceptr,
        sf: CUdeviceptr,
        n_exp: usize,
        n: usize,
        k: usize,
    ) -> Result<i64, IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn w8a8_packw(
                    w: *const c_void,
                    outp: *mut c_void,
                    sf: *mut c_void,
                    n_exp: c_int,
                    n: c_int,
                    k: c_int,
                    stream: *mut c_void,
                ) -> i64;
            }
            Ok(unsafe {
                w8a8_packw(
                    w as *const c_void,
                    outp as *mut c_void,
                    sf as *mut c_void,
                    n_exp as c_int,
                    n as c_int,
                    k as c_int,
                    self.stream as *mut c_void,
                )
            })
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (w, outp, sf, n_exp, n, k);
            Err(IronError::Dispatch("w8a8_packw: built without CUTLASS".into()))
        }
    }

    /// Persistent-handle variant of the CUTLASS grouped NVFP4 GEMM: descriptors
    /// are derived ON DEVICE from the group-offsets buffer each call (one tiny
    /// fill kernel + gemm.run) — no host build, no per-call allocs, graph-safe.
    /// `prepare` once per (weight slab, n_groups, N, K); `run` per call.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped_cutlass_fp4_prepare(
        &self,
        w: CUdeviceptr,
        sfw: CUdeviceptr,
        alpha_vec: CUdeviceptr,
        n_groups: usize,
        n: usize,
        k: usize,
        max_m_total: usize,
    ) -> Result<u64, IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_prepare(
                    b: *const c_void,
                    sfb: *const c_void,
                    alpha_vec: *const c_void,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    max_m_total: c_int,
                ) -> *mut c_void;
            }
            let h = unsafe {
                moe_grouped_gemm_cutlass_fp4_prepare(
                    w as *const c_void,
                    sfw as *const c_void,
                    alpha_vec as *const c_void,
                    n_groups as c_int,
                    n as c_int,
                    k as c_int,
                    max_m_total as c_int,
                )
            };
            if h.is_null() {
                return Err(IronError::Dispatch("cutlass_fp4_prepare failed".into()));
            }
            Ok(h as u64)
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (w, sfw, alpha_vec, n_groups, n, k, max_m_total);
            Err(IronError::Dispatch("built without CUTLASS (set CUTLASS_DIR)".into()))
        }
    }

    /// Per-call run against a prepared handle. `off_dev` = device u32
    /// `[n_groups+1]` row offsets (the router's expert offsets).
    pub fn moe_grouped_cutlass_fp4_run(
        &self,
        handle: u64,
        a: CUdeviceptr,
        sfa: CUdeviceptr,
        d_out: CUdeviceptr,
        off_dev: CUdeviceptr,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_run(
                    handle: *mut c_void,
                    a: *const c_void,
                    sfa: *const c_void,
                    d: *mut c_void,
                    off_dev: *const c_void,
                    stream: *mut c_void,
                ) -> c_int;
            }
            let r = unsafe {
                moe_grouped_gemm_cutlass_fp4_run(
                    handle as *mut c_void,
                    a as *const c_void,
                    sfa as *const c_void,
                    d_out as *mut c_void,
                    off_dev as *const c_void,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(IronError::Dispatch(format!("cutlass_fp4_run failed: code {r}")));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (handle, a, sfa, d_out, off_dev);
            Err(IronError::Dispatch("built without CUTLASS (set CUTLASS_DIR)".into()))
        }
    }

    /// Fused-activation variant of [`moe_grouped_cutlass_fp4_prepare`]: GEMM1's
    /// epilogue folds in relu²·(1/256) + NVFP4 block-scale-quant, so the GEMM
    /// emits e2m1 D + ue4m3 SFD directly (the down-GEMM's input). `norm_constant`
    /// = device f32 ptr to the static gs (1/256). Gated behind NEMOTRON_FUSE_UPACT.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped_cutlass_fp4_FUSEDACT_prepare(
        &self,
        w: CUdeviceptr,
        sfw: CUdeviceptr,
        alpha_vec: CUdeviceptr,
        norm_constant: CUdeviceptr,
        n_groups: usize,
        n: usize,
        k: usize,
        max_m_total: usize,
    ) -> Result<u64, IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_FUSEDACT_prepare(
                    b: *const c_void,
                    sfb: *const c_void,
                    alpha_vec: *const c_void,
                    norm_constant: *const c_void,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    max_m_total: c_int,
                ) -> *mut c_void;
            }
            let h = unsafe {
                moe_grouped_gemm_cutlass_fp4_FUSEDACT_prepare(
                    w as *const c_void,
                    sfw as *const c_void,
                    alpha_vec as *const c_void,
                    norm_constant as *const c_void,
                    n_groups as c_int,
                    n as c_int,
                    k as c_int,
                    max_m_total as c_int,
                )
            };
            if h.is_null() {
                return Err(IronError::Dispatch("cutlass_fp4_FUSEDACT_prepare failed".into()));
            }
            Ok(h as u64)
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (w, sfw, alpha_vec, norm_constant, n_groups, n, k, max_m_total);
            Err(IronError::Dispatch("built without CUTLASS (set CUTLASS_DIR)".into()))
        }
    }

    /// Per-call run against a fused-activation handle. Emits `d_out` (e2m1 packed
    /// `[mt, N/2]`) + `sfd_out` (ue4m3 swizzled SF blob, same layout the plain
    /// down-GEMM reads as SFA). `off_dev` = device u32 `[n_groups+1]` row offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped_cutlass_fp4_FUSEDACT_run(
        &self,
        handle: u64,
        a: CUdeviceptr,
        sfa: CUdeviceptr,
        d_out: CUdeviceptr,
        sfd_out: CUdeviceptr,
        off_dev: CUdeviceptr,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_FUSEDACT_run(
                    handle: *mut c_void,
                    a: *const c_void,
                    sfa: *const c_void,
                    d: *mut c_void,
                    sfd: *mut c_void,
                    off_dev: *const c_void,
                    stream: *mut c_void,
                ) -> c_int;
            }
            let r = unsafe {
                moe_grouped_gemm_cutlass_fp4_FUSEDACT_run(
                    handle as *mut c_void,
                    a as *const c_void,
                    sfa as *const c_void,
                    d_out as *mut c_void,
                    sfd_out as *mut c_void,
                    off_dev as *const c_void,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(IronError::Dispatch(format!(
                    "cutlass_fp4_FUSEDACT_run failed: code {r}"
                )));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (handle, a, sfa, d_out, sfd_out, off_dev);
            Err(IronError::Dispatch("built without CUTLASS (set CUTLASS_DIR)".into()))
        }
    }

    /// Amax-in-epilogue variant of [`moe_grouped_cutlass_fp4_prepare`]: GEMM1
    /// emits f16 D = relu²·(1/256) (the a2 / up_out) AND a per-group amax of
    /// those activated values into `d_amax` (device `f32[n_groups]`, persistent
    /// → graph-safe; max over groups = the per-tensor global for the down-quant).
    /// The down-quant then reads a2 ONCE (no separate amax scan). Gated behind
    /// NEMOTRON_AMAX_EPI. Output is bit-identical to the 2-pass relu2+amax+quant.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped_cutlass_fp4_AMAX_prepare(
        &self,
        w: CUdeviceptr,
        sfw: CUdeviceptr,
        d_amax: CUdeviceptr,
        n_groups: usize,
        n: usize,
        k: usize,
        max_m_total: usize,
    ) -> Result<u64, IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_AMAX_prepare(
                    b: *const c_void,
                    sfb: *const c_void,
                    d_amax: *const c_void,
                    n_groups: c_int,
                    n: c_int,
                    k: c_int,
                    max_m_total: c_int,
                ) -> *mut c_void;
            }
            let h = unsafe {
                moe_grouped_gemm_cutlass_fp4_AMAX_prepare(
                    w as *const c_void,
                    sfw as *const c_void,
                    d_amax as *const c_void,
                    n_groups as c_int,
                    n as c_int,
                    k as c_int,
                    max_m_total as c_int,
                )
            };
            if h.is_null() {
                return Err(IronError::Dispatch("cutlass_fp4_AMAX_prepare failed".into()));
            }
            Ok(h as u64)
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (w, sfw, d_amax, n_groups, n, k, max_m_total);
            Err(IronError::Dispatch("built without CUTLASS (set CUTLASS_DIR)".into()))
        }
    }

    /// Per-call run against an amax-in-epilogue handle. Emits `d_out` (f16 a2,
    /// `[mt, N]`) + writes the per-group amax into the handle's `d_amax`. The
    /// caller MUST zero `d_amax` before each call (the build uses
    /// `-DCUTLASS_SKIP_REDUCTION_INIT=1`, so the kernel does not self-init it).
    /// `off_dev` = device u32 `[n_groups+1]` row offsets.
    pub fn moe_grouped_cutlass_fp4_AMAX_run(
        &self,
        handle: u64,
        a: CUdeviceptr,
        sfa: CUdeviceptr,
        d_out: CUdeviceptr,
        off_dev: CUdeviceptr,
    ) -> Result<(), IronError> {
        self.ensure_current();
        #[cfg(have_cutlass)]
        {
            unsafe extern "C" {
                fn moe_grouped_gemm_cutlass_fp4_AMAX_run(
                    handle: *mut c_void,
                    a: *const c_void,
                    sfa: *const c_void,
                    d: *mut c_void,
                    off_dev: *const c_void,
                    stream: *mut c_void,
                ) -> c_int;
            }
            let r = unsafe {
                moe_grouped_gemm_cutlass_fp4_AMAX_run(
                    handle as *mut c_void,
                    a as *const c_void,
                    sfa as *const c_void,
                    d_out as *mut c_void,
                    off_dev as *const c_void,
                    self.stream as *mut c_void,
                )
            };
            if r != 0 {
                return Err(IronError::Dispatch(format!("cutlass_fp4_AMAX_run failed: code {r}")));
            }
            Ok(())
        }
        #[cfg(not(have_cutlass))]
        {
            let _ = (handle, a, sfa, d_out, off_dev);
            Err(IronError::Dispatch("built without CUTLASS (set CUTLASS_DIR)".into()))
        }
    }

    /// Block-scaled NVFP4 GEMM via cuBLASLt (Blackwell tensor cores):
    ///   `out[m,n](f16) = X[m,k](e2m1, vec16 ue4m3 scales) · W[n,k]^T(e2m1, vec16 ue4m3)`
    /// Operands are packed 2 elems/byte row-major; scale tensors use the
    /// 512-byte-block swizzled layout (one ue4m3 per 16 elements along K):
    ///   sf_off(r, kb) = (r/128)*512*ceil(KB/4) + (kb/4)*512 + (r%32)*16
    ///                 + ((r/32)%4)*4 + (kb%4)
    /// Measured 240-306 TFLOP/s on GB10 at dense-projection shapes — ~4x the
    /// f16 path — with max_rel 5e-4 vs an exact dequant reference.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublaslt_fp4(
        &self,
        x: CUdeviceptr,    // [m, k/2] packed e2m1 activation
        x_sf: CUdeviceptr, // activation scales, swizzled
        w: CUdeviceptr,    // [n, k/2] packed e2m1 weight
        w_sf: CUdeviceptr, // weight scales, swizzled
        out: CUdeviceptr,  // [m, n] row-major result (f16, or f32 when out_f32)
        m: usize,
        n: usize,
        k: usize,
        out_f32: bool,
        d_scale: CUdeviceptr, // device f32 applied to D (per-tensor global fold); 0 = none
    ) -> Result<(), IronError> {
        self.ensure_current();
        let (lt, workspace) = self.cublaslt_ctx()?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let use_dev_alpha = d_scale != 0;
        let beta_zero = if use_dev_alpha { self.lt_beta_zero_ptr()? } else { 0 };
        unsafe {
            let mut desc: cublasLtMatmulDesc_t = ptr::null_mut();
            let s = cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            if s != CUBLAS_STATUS_SUCCESS {
                return Err(IronError::Dispatch(format!("cublasLtMatmulDescCreate: {s}")));
            }
            if use_dev_alpha {
                let pm: c_int = CUBLASLT_POINTER_MODE_DEVICE;
                cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_POINTER_MODE,
                    &pm as *const c_int as *const c_void,
                    std::mem::size_of::<c_int>(),
                );
            }
            let opt = CUBLAS_OP_T;
            let opn = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                &opt as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &opn as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            // Block-scale modes + pointers. With the col-major reorder A=W and
            // B=X, so the A scale is the WEIGHT scale and B the activation.
            let sm: c_int = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                &sm as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                &sm as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &w_sf as *const CUdeviceptr as *const c_void,
                std::mem::size_of::<CUdeviceptr>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &x_sf as *const CUdeviceptr as *const c_void,
                std::mem::size_of::<CUdeviceptr>(),
            );
            // A=W [n,k] rm == [k,n] cm (ld=k); B=X [m,k] rm == [k,m] cm (ld=k);
            // D [m,n] rm == [n,m] cm (ld=n). lds are in ELEMENTS (4-bit ok: k%32==0).
            let mut a_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut b_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut d_l: cublasLtMatrixLayout_t = ptr::null_mut();
            cublasLtMatrixLayoutCreate(&mut a_l, CUDA_R_4F_E2M1, k as u64, n as u64, k as i64);
            cublasLtMatrixLayoutCreate(&mut b_l, CUDA_R_4F_E2M1, k as u64, m as u64, k as i64);
            cublasLtMatrixLayoutCreate(
                &mut d_l,
                if out_f32 { CUDA_R_32F } else { CUDA_R_16F },
                n as u64,
                m as u64,
                n as i64,
            );

            let mut pref: cublasLtMatmulPreference_t = ptr::null_mut();
            cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_bytes: usize = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            );
            let red_mask: u32 = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                &red_mask as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            );

            let mut result = cublasLtMatmulHeuristicResult_t::default();
            let mut returned: c_int = 0;
            let hs = cublasLtMatmulAlgoGetHeuristic(
                lt,
                desc,
                a_l,
                b_l,
                d_l,
                d_l,
                pref,
                1,
                &mut result,
                &mut returned,
            );
            if hs != CUBLAS_STATUS_SUCCESS || returned < 1 {
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(a_l);
                cublasLtMatrixLayoutDestroy(b_l);
                cublasLtMatrixLayoutDestroy(d_l);
                cublasLtMatmulDescDestroy(desc);
                return Err(IronError::Dispatch(format!(
                    "cublasLt-fp4: no algo (m={m} n={n} k={k} status={hs} returned={returned})"
                )));
            }
            let alpha_arg: *const c_void = if use_dev_alpha {
                d_scale as usize as *const c_void
            } else {
                &alpha as *const f32 as *const c_void
            };
            let beta_arg: *const c_void = if use_dev_alpha {
                beta_zero as usize as *const c_void
            } else {
                &beta as *const f32 as *const c_void
            };
            let mm = cublasLtMatmul(
                lt,
                desc,
                alpha_arg,
                w,
                a_l,
                x,
                b_l,
                beta_arg,
                out,
                d_l,
                out,
                d_l,
                result.algo.as_ptr(),
                workspace,
                ws_bytes,
                self.stream,
            );
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(a_l);
            cublasLtMatrixLayoutDestroy(b_l);
            cublasLtMatrixLayoutDestroy(d_l);
            cublasLtMatmulDescDestroy(desc);
            if mm != CUBLAS_STATUS_SUCCESS {
                return Err(IronError::Dispatch(format!(
                    "cublasLtMatmul(fp4) failed: status {mm} (m={m} n={n} k={k})"
                )));
            }
        }
        Ok(())
    }

    /// FP8 (e4m3) GEMM via cuBLASLt with per-tensor f32 DEVICE scale
    /// pointers (dequant scales; D = scaleX*scaleW * X·Wᵀ). TN layout —
    /// the FP8-required form. ~2x the f16 rate on GB10, far gentler
    /// quantization than FP4 (the vendor recipe for this model family runs
    /// the Mamba/shared/o-proj GEMMs at FP8).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cublaslt_fp8(
        &self,
        x: CUdeviceptr,       // [m, k] e4m3 row-major activation
        x_scale: CUdeviceptr, // f32 scalar (device)
        w: CUdeviceptr,       // [n, k] e4m3 row-major weight
        w_scale: CUdeviceptr, // f32 scalar (device)
        out: CUdeviceptr,     // [m, n] row-major result
        m: usize,
        n: usize,
        k: usize,
        out_f32: bool,
    ) -> Result<(), IronError> {
        self.ensure_current();
        let (lt, workspace) = self.cublaslt_ctx()?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            let mut desc: cublasLtMatmulDesc_t = ptr::null_mut();
            let s = cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            if s != CUBLAS_STATUS_SUCCESS {
                return Err(IronError::Dispatch(format!("cublasLtMatmulDescCreate: {s}")));
            }
            let opt = CUBLAS_OP_T;
            let opn = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                &opt as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &opn as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            );
            // Per-tensor f32 scales (default SCALAR mode). ptr==0 => skip
            // (caller folds per-channel/per-token scales as a post-pass).
            // With the col-major reorder A=W and B=X.
            if w_scale != 0 {
                cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &w_scale as *const CUdeviceptr as *const c_void,
                    std::mem::size_of::<CUdeviceptr>(),
                );
            }
            if x_scale != 0 {
                cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &x_scale as *const CUdeviceptr as *const c_void,
                    std::mem::size_of::<CUdeviceptr>(),
                );
            }
            let mut a_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut b_l: cublasLtMatrixLayout_t = ptr::null_mut();
            let mut d_l: cublasLtMatrixLayout_t = ptr::null_mut();
            cublasLtMatrixLayoutCreate(&mut a_l, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64);
            cublasLtMatrixLayoutCreate(&mut b_l, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64);
            cublasLtMatrixLayoutCreate(
                &mut d_l,
                if out_f32 { CUDA_R_32F } else { CUDA_R_16F },
                n as u64,
                m as u64,
                n as i64,
            );

            let mut pref: cublasLtMatmulPreference_t = ptr::null_mut();
            cublasLtMatmulPreferenceCreate(&mut pref);
            let ws_bytes: usize = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            );
            let red_mask: u32 = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                &red_mask as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            );
            let mut result = cublasLtMatmulHeuristicResult_t::default();
            let mut returned: c_int = 0;
            let hs = cublasLtMatmulAlgoGetHeuristic(
                lt,
                desc,
                a_l,
                b_l,
                d_l,
                d_l,
                pref,
                1,
                &mut result,
                &mut returned,
            );
            if hs != CUBLAS_STATUS_SUCCESS || returned < 1 {
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(a_l);
                cublasLtMatrixLayoutDestroy(b_l);
                cublasLtMatrixLayoutDestroy(d_l);
                cublasLtMatmulDescDestroy(desc);
                return Err(IronError::Dispatch(format!(
                    "cublasLt-fp8: no algo (m={m} n={n} k={k} status={hs} returned={returned})"
                )));
            }
            let mm = cublasLtMatmul(
                lt,
                desc,
                &alpha as *const f32 as *const c_void,
                w,
                a_l,
                x,
                b_l,
                &beta as *const f32 as *const c_void,
                out,
                d_l,
                out,
                d_l,
                result.algo.as_ptr(),
                workspace,
                ws_bytes,
                self.stream,
            );
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(a_l);
            cublasLtMatrixLayoutDestroy(b_l);
            cublasLtMatrixLayoutDestroy(d_l);
            cublasLtMatmulDescDestroy(desc);
            if mm != CUBLAS_STATUS_SUCCESS {
                return Err(IronError::Dispatch(format!(
                    "cublasLtMatmul(fp8) failed: status {mm} (m={m} n={n} k={k})"
                )));
            }
        }
        Ok(())
    }

    /// Zero-fill `len` bytes at `ptr`, stream-ordered on the device stream.
    /// No host staging buffer and no stream drain — the cheap way to seed an
    /// accumulator (vs uploading a host zero buffer, which for a [s,hid] f32
    /// accumulator is a multi-MB pageable H2D copy per call).
    pub fn memset_zero_raw(&self, ptr: CUdeviceptr, len: usize) -> Result<(), IronError> {
        self.ensure_current();
        if ptr == 0 || len == 0 {
            return Ok(());
        }
        cu_check(unsafe { cuMemsetD8Async(ptr, 0, len, self.stream) }, "cuMemsetD8Async(zero)")
    }
}

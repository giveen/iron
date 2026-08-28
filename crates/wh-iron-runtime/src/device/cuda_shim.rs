//! SPDX-License-Identifier: Apache-2.0
//! Minimal C ABI shim for Butter/Swift CUDA integration.

use std::ffi::{c_char, c_int, c_void};
use std::slice;
use std::sync::OnceLock;

use super::cuda::{CudaDevice, CudaFunction, CudaModule, DeviceBuffer};

#[repr(C)]
pub struct OpaqueDevice(*mut c_void);
#[repr(C)]
pub struct OpaqueModule(*mut c_void);
#[repr(C)]
pub struct OpaqueFunction(*mut c_void);
#[repr(C)]
pub struct OpaqueBuffer(*mut c_void);

static DEVICE: OnceLock<Option<CudaDevice>> = OnceLock::new();

fn device() -> Option<&'static CudaDevice> {
    DEVICE.get_or_init(|| CudaDevice::create().ok().flatten()).as_ref()
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_init() -> *mut OpaqueDevice {
    device().map(|_| std::ptr::NonNull::<OpaqueDevice>::new(std::ptr::NonNull::dangling().as_ptr()).as_ptr()).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_alloc(_: *mut OpaqueDevice, bytes: usize) -> *mut OpaqueBuffer {
    device()?.alloc(bytes).ok().map(|b| Box::into_raw(Box::new(b)) as *mut OpaqueBuffer).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_free_buffer(_: *mut OpaqueDevice, buf: *mut OpaqueBuffer) {
    if !buf.is_null() {
        let _ = Box::from_raw(buf as *mut DeviceBuffer);
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_upload(_: *mut OpaqueDevice, buf: *mut OpaqueBuffer, data: *const u8, bytes: usize) -> c_int {
    let dev = match device() { Some(d) => d, None => return -1 };
    let buf = &*(buf as *mut DeviceBuffer);
    let slice = slice::from_raw_parts(data, bytes);
    dev.upload(slice).map(|_| 0).unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_download(_: *mut OpaqueDevice, buf: *mut OpaqueBuffer, out: *mut u8, bytes: usize) -> c_int {
    let dev = match device() { Some(d) => d, None => return -1 };
    let buf = &*(buf as *mut DeviceBuffer);
    let out = slice::from_raw_parts_mut(out, bytes);
    dev.download(buf, out).map(|_| 0).unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_compile(_: *mut OpaqueDevice, src: *const c_char, name: *const c_char) -> *mut OpaqueModule {
    let dev = match device() { Some(d) => d, None => return std::ptr::null_mut() };
    let src = match CString::from_raw(src as *mut c_char).into_string() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    let name = match CString::from_raw(name as *mut c_char).into_string() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    dev.compile(&src, &name).ok().map(|m| Box::into_raw(Box::new(m)) as *mut OpaqueModule).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_free_module(_: *mut OpaqueDevice, mod_: *mut OpaqueModule) {
    if !mod_.is_null() {
        let _ = Box::from_raw(mod_ as *mut CudaModule);
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_get_function(_: *mut OpaqueDevice, mod_: *mut OpaqueModule, name: *const c_char) -> *mut OpaqueFunction {
    let mod_ = &*(mod_ as *mut CudaModule);
    let name = match CStr::from_ptr(name).to_str() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    mod_.function(name).ok().map(|f| Box::into_raw(Box::new(f)) as *mut OpaqueFunction).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_launch_1d(_: *mut OpaqueDevice, func: *mut OpaqueFunction, blocks: u32, threads: u32, args: *mut *mut c_void) -> c_int {
    let dev = match device() { Some(d) => d, None => return -1 };
    let func = &*(func as *mut CudaFunction);
    let args = slice::from_raw_parts_mut(args, 4);
    dev.launch_1d(func, blocks, threads, args).map(|_| 0).unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn cuda_shim_synchronize(_: *mut OpaqueDevice) -> c_int {
    device().map(|d| d.synchronize().map(|_| 0).unwrap_or(-1)).unwrap_or(-1)
}

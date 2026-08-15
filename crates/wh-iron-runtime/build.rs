//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Link config for the optional GPU backends (CUDA, HIP, Vulkan).
//!
//! Per-feature; each enabled feature emits its own `cargo:rustc-link-*`
//! directives. macOS without any feature builds the Metal path unchanged.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_cutlass)");
    println!("cargo:rustc-check-cfg=cfg(have_marlin)");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        cuda();
    }
    if std::env::var("CARGO_FEATURE_HIP").is_ok() {
        hip();
    }
    if std::env::var("CARGO_FEATURE_VULKAN").is_ok() {
        vulkan();
    }
}

/// HIP / ROCm linker setup. Windows ships `amdhip64.lib` and `hiprtc.lib`
/// under `<ROCm>/lib`; Linux distros ship them under `<rocm>/lib` or
/// `/opt/rocm/lib`. `HIP_PATH` is the canonical env var; `ROCM_PATH`
/// covers Linux installs that prefer that name.
fn hip() {
    let hip_root =
        std::env::var("HIP_PATH").or_else(|_| std::env::var("ROCM_PATH")).unwrap_or_else(|_| {
            if cfg!(windows) {
                r"C:\Program Files\AMD\ROCm\7.1".to_string()
            } else {
                "/opt/rocm".to_string()
            }
        });

    for sub in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={hip_root}/{sub}");
    }

    // `amdhip64` is the import lib on Windows and the shared object on
    // Linux; rustc emits the right form for the target platform.
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hiprtc");
}

/// Vulkan linker setup. The SDK installs `vulkan-1.lib` and `shaderc*.lib`
/// under `<VULKAN_SDK>/Lib` on Windows, `<sdk>/lib` on Linux.
fn vulkan() {
    let vk_sdk = std::env::var("VULKAN_SDK").unwrap_or_else(|_| {
        if cfg!(windows) { r"C:\VulkanSDK\1.4.350.0".to_string() } else { "/usr".to_string() }
    });

    for sub in ["Lib", "lib", "lib64"] {
        println!("cargo:rustc-link-search=native={vk_sdk}/{sub}");
    }

    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
        // shaderc_combined is the all-in-one static lib that bundles
        // libshaderc + glslang + SPIRV-Tools — simpler than wiring up the
        // shared `shaderc_shared.dll` plus its dep chain.
        println!("cargo:rustc-link-lib=static=shaderc_combined");
    } else {
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rustc-link-lib=dylib=shaderc_shared");
    }
}

fn cuda() {
    let cuda_root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // `libnvrtc` lives in the toolkit lib dir; `libcuda` (driver) link stub
    // lives under `lib64/stubs`. Both 64-bit dir names are tried.
    for sub in ["lib64", "lib64/stubs", "lib", "lib/stubs", "lib/x64"] {
        println!("cargo:rustc-link-search=native={cuda_root}/{sub}");
    }
    for p in ["/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
        println!("cargo:rustc-link-search=native={p}");
    }

    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=cuda");
    // cuBLAS tensor-core GEMM escape hatch (Path A): ships in the CUDA toolkit.
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Optional: AOT-compile the CUTLASS grouped-MoE GEMM when CUTLASS_DIR points
    // at a CUTLASS checkout. nvcc → static lib → link; emits cfg(have_cutlass)
    // so the FFI/runtime path compiles only when present. Skipped (no-op) on any
    // build without CUTLASS_DIR, so Mac/Metal and CUTLASS-less CUDA boxes are
    // unaffected.
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=NEMOTRON_CUTLASS_ARCH");
    if let Ok(cutlass_dir) = std::env::var("CUTLASS_DIR") {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
        let arch = std::env::var("NEMOTRON_CUTLASS_ARCH").unwrap_or_else(|_| "sm_121a".to_string());
        let nvcc = format!("{cuda_root}/bin/nvcc");
        // Block-scaled mma availability define for the FP4 grouped kernel:
        // mirror the arch the objects are built for (sm_120a / sm_121a).
        let mma_define = if arch.contains("121") {
            Some("-DCUTLASS_ARCH_MMA_SM121_SUPPORTED=1")
        } else if arch.contains("120") {
            Some("-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1")
        } else {
            None
        };
        let compile = |src_name: &str, extra: &[&str]| -> String {
            let src = format!("{}/cuda/{src_name}.cu", env!("CARGO_MANIFEST_DIR"));
            println!("cargo:rerun-if-changed={src}");
            let obj = format!("{out_dir}/{src_name}.o");
            let status = std::process::Command::new(&nvcc)
                .args([
                    "-O3",
                    "-std=c++17",
                    &format!("-arch={arch}"),
                    "--expt-relaxed-constexpr",
                    "-Xcompiler",
                    "-fPIC",
                ])
                .args(extra)
                .args([
                    "-I",
                    &format!("{cutlass_dir}/include"),
                    "-I",
                    &format!("{cutlass_dir}/tools/util/include"),
                ])
                .args(["-c", &src, "-o", &obj])
                .status()
                .unwrap_or_else(|_| panic!("nvcc invocation for {src_name}.cu failed to start"));
            assert!(status.success(), "nvcc failed to compile {src_name}.cu");
            obj
        };
        let obj = compile("cutlass_moe", &[]);
        // -DCUTLASS_SKIP_REDUCTION_INIT=1: the amax-in-epilogue GEMM1 variant
        // (NEMOTRON_AMAX_EPI) zeroes its per-group amax buffer on-stream before
        // each run, so the epilogue must NOT auto-init the ScalarReduction output.
        let mut fp4_extra: Vec<&str> =
            vec!["--expt-extended-lambda", "-DCUTLASS_SKIP_REDUCTION_INIT=1"];
        if let Some(d) = mma_define {
            fp4_extra.push(d);
        }
        let obj_fp4 = compile("cutlass_moe_fp4", &fp4_extra);
        let lib = format!("{out_dir}/libcutlass_moe.a");
        let ar = std::process::Command::new("ar")
            .args(["crs", &lib, &obj, &obj_fp4])
            .status()
            .expect("ar failed to start");
        assert!(ar.success(), "ar failed to archive cutlass_moe objects");
        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static=cutlass_moe");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-cfg=have_cutlass");
    }

    // Optional: AOT-compile the in-tree marlin U4B8 small-M GEMM (F-85 round-8)
    // when IRON_MARLIN_BUILD is set. nvcc -rdc each .cu, device-link, ar into a
    // static lib; emits cfg(have_marlin). Additive + opt-in like the CUTLASS
    // block above: default builds (Mac/Metal, CUTLASS-less/marlin-less CUDA
    // hosts) are unaffected. Unlike CUTLASS this needs no external SDK dir —
    // the csrc is fully in-tree under cuda/marlin/ (repack_standalone.cu,
    // marlin_mm.cu, the single u4b8/f16 kernel instantiation, marlin_shim.cu).
    println!("cargo:rerun-if-env-changed=IRON_MARLIN_BUILD");
    println!("cargo:rerun-if-env-changed=IRON_MARLIN_ARCH");
    if std::env::var("IRON_MARLIN_BUILD").is_ok() {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
        let arch = std::env::var("IRON_MARLIN_ARCH").unwrap_or_else(|_| "sm_121a".to_string());
        let nvcc = format!("{cuda_root}/bin/nvcc");
        let md = format!("{}/cuda/marlin", env!("CARGO_MANIFEST_DIR"));
        let base: Vec<String> = vec![
            "-O3".into(),
            "-std=c++17".into(),
            format!("-arch={arch}"),
            "--expt-relaxed-constexpr".into(),
            "-Xcompiler".into(),
            "-fPIC".into(),
        ];
        let inc: Vec<String> = vec![
            "-I".into(),
            format!("{md}/csrc"),
            "-I".into(),
            format!("{md}/csrc/core"),
            "-I".into(),
            format!("{md}/csrc/quantization/marlin"),
            "-I".into(),
            format!("{md}/csrc/moe/marlin_moe_wna16"),
        ];
        let srcs = [
            format!("{md}/csrc/moe/marlin_moe_wna16/sm80_kernel_float16_u4b8_float16.cu"),
            format!("{md}/csrc/moe/marlin_moe_wna16/marlin_mm.cu"),
            format!("{md}/csrc/quantization/marlin/repack_standalone.cu"),
            format!("{md}/marlin_shim.cu"),
        ];
        let mut objs: Vec<String> = Vec::new();
        for (i, src) in srcs.iter().enumerate() {
            println!("cargo:rerun-if-changed={src}");
            let obj = format!("{out_dir}/marlin_{i}.o");
            let st = std::process::Command::new(&nvcc)
                .args(&base)
                .args(&inc)
                .args(["-dc", src.as_str(), "-o", obj.as_str()])
                .status()
                .expect("nvcc marlin compile failed to start");
            assert!(st.success(), "nvcc failed on {src}");
            objs.push(obj);
        }
        let dlink = format!("{out_dir}/marlin_dlink.o");
        let mut dl = std::process::Command::new(&nvcc);
        dl.arg(format!("-arch={arch}")).arg("-rdc=true").arg("-dlink");
        for o in &objs {
            dl.arg(o);
        }
        dl.arg("-o").arg(&dlink);
        assert!(
            dl.status().expect("nvcc -dlink failed to start").success(),
            "marlin device-link failed"
        );
        let lib = format!("{out_dir}/libmarlin.a");
        let mut arc = std::process::Command::new("ar");
        arc.arg("crs").arg(&lib);
        for o in &objs {
            arc.arg(o);
        }
        arc.arg(&dlink);
        assert!(arc.status().expect("ar failed to start").success(), "ar libmarlin failed");
        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static=marlin");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-cfg=have_marlin");
    }
}

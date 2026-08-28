use wh_iron::{
    codegen::{CodegenBackend, CudaGenerator},
    core::{dtype::DType, ir::KernelMode},
    CudaDevice,
};
use wh_iron_std::kernels::kv_cache::cache::iron_kv_cache_update;

fn main() {
    let Ok(Some(dev)) = CudaDevice::create() else {
        eprintln!("No CUDA device found");
        std::process::exit(1);
    };
    println!("CUDA device OK, compute capability {}.{}", dev.compute_capability().0, dev.compute_capability().1);

    let dt = DType::F32;
    let kernel = iron_kv_cache_update::kernel_ir_for(dt);
    let mut k = kernel;
    k.mode = KernelMode::Grid3D;

    let src = match CudaGenerator::new().generate(&k) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("codegen failed: {e}");
            std::process::exit(2);
        }
    };
    println!("generated {} bytes of CUDA C++", src.len());

    let module = match dev.compile(&src, "iron_kv_cache_update") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("NVRTC compile failed: {e}");
            std::process::exit(3);
        }
    };
    println!("compile: ok");
    let func = match module.function("iron_kv_cache_update") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("function lookup failed: {e}");
            std::process::exit(4);
        }
    };
    println!("function: ok");

    let n_kv_heads = 4usize;
    let head_dim = 16usize;
    let max_seq = 8usize;
    let position = 3usize;
    let n_elems = n_kv_heads * head_dim;

    let src_data: Vec<f32> = (0..n_elems).map(|i| 10.0 + i as f32).collect();
    let mut cache_data = vec![999.0f32; n_kv_heads * max_seq * head_dim];

    let src_buf = match dev.upload(bytemuck::cast_slice(&src_data)) {
        Ok(b) => b,
        Err(e) => { eprintln!("upload src failed: {e}"); std::process::exit(5); }
    };
    let cache_buf = match dev.upload(bytemuck::cast_slice(&cache_data)) {
        Ok(b) => b,
        Err(e) => { eprintln!("upload cache failed: {e}"); std::process::exit(6); }
    };

    let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
    args.push(&src_buf.device_ptr() as *const _ as *mut _);
    args.push(&cache_buf.device_ptr() as *const _ as *mut _);
    let head_dim_ptr = &(head_dim as u32) as *const u32 as *const _ as *mut _;
    let max_seq_ptr = &(max_seq as u32) as *const u32 as *const _ as *mut _;
    let position_ptr = &(position as u32) as *const u32 as *const _ as *mut _;
    args.push(head_dim_ptr as *mut _);
    args.push(max_seq_ptr as *mut _);
    args.push(position_ptr as *mut _);

    match dev.launch_1d(func, 1, n_elems as u32, args) {
        Ok(()) => println!("launch: ok"),
        Err(e) => { eprintln!("launch failed: {e}"); std::process::exit(7); }
    }
    dev.synchronize().ok();

    let mut got = vec![0u8; n_kv_heads * max_seq * head_dim * 4];
    match dev.download(&cache_buf, &mut got) {
        Ok(()) => (),
        Err(e) => { eprintln!("download failed: {e}"); std::process::exit(8); }
    }
    let got_f32: Vec<f32> = bytemuck::cast_slice(&got).to_vec();

    let mut worst = 0.0f32;
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let dst = h * max_seq * head_dim + position * head_dim + d;
            let src_idx = h * head_dim + d;
            let expected = 10.0 + src_idx as f32;
            let diff = (got_f32[dst] - expected).abs();
            worst = worst.max(diff);
            if diff > 1e-6 {
                eprintln!("mismatch h={h} d={d}: got {} expected {}", got_f32[dst], expected);
            }
        }
    }
    println!("max|Δ| = {worst:.3e}");
    if worst < 1e-6 {
        println!("PASS");
    } else {
        println!("FAIL");
        std::process::exit(9);
    }
}

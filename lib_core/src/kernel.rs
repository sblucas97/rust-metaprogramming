use crate::cuda_vec::CudaVec;
use crate::ffi;

pub trait KernelName {
    fn kernel_name() -> &'static str;
}

pub fn spawn<K: KernelName>(a_h: CudaVec<f32>, b_h: CudaVec<f32>, r_h: &mut Vec<f32>) {
    let name = K::kernel_name();

    // ── compile ───────────────────────────────────────────────────────────────
    let so_name = format!("kernel_and_launcher_generated_for_{name}.so");

    let output = std::process::Command::new("nvcc")
        .args([
            "-arch=sm_86",
            &format!("generated_{name}.cu"),
            &format!("generated_{name}_launcher.cu"),
            "-o",
            &so_name,
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "-x",
            "cu",
        ])
        .output()
        .expect("failed to run nvcc — is it on PATH?");

    if !output.stdout.is_empty() {
        eprintln!("nvcc stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        eprintln!("nvcc stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() {
        panic!(
            "nvcc compilation failed (exit {:?}) for kernel `{name}`",
            output.status.code()
        );
    }

    // ── load & launch ─────────────────────────────────────────────────────────
    let so_path = std::env::current_dir().unwrap().join(&so_name);
    eprintln!("spawn: loading {}", so_path.display());

    unsafe {
        // Allocate result buffer on the device.
        let mut result_device_ptr: *mut f32 = std::ptr::null_mut();
        ffi::cuda_allocate(&mut result_device_ptr as *mut *mut f32);

        // Dynamically resolve the launcher symbol.
        type LaunchFn =
            unsafe extern "C" fn(*mut f32, *mut f32, *mut f32);

        let lib = libloading::Library::new(&so_path)
            .unwrap_or_else(|e| panic!("failed to load `{}`: {e}", so_path.display()));

        let symbol_name = format!("launch_generated_{name}\0");
        let launch: libloading::Symbol<LaunchFn> = lib
            .get(symbol_name.as_bytes())
            .unwrap_or_else(|e| panic!("symbol `launch_generated_{name}` not found: {e}"));

        launch(
            a_h.get_device_ptr() as *mut f32,
            b_h.get_device_ptr() as *mut f32,
            result_device_ptr,
        );

        // Copy result back to host.
        let result_len = a_h.len();
        r_h.resize(result_len, 0.0_f32);
        ffi::cuda_copy_to_host(r_h.as_mut_ptr(), result_device_ptr);

        ffi::cuda_free(result_device_ptr);
    }
}
//! Minimal helpers to load a PTX file, resolve a kernel, push parameters, and launch on the default stream.
//!
//! Launching is done with a closure so each call site can chain `.arg(...)` directly — no tuple + macro machinery.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, DeviceRepr, DriverError},
    nvrtc::Ptx,
};

use crate::{CudaVec};

pub use cudarc::driver::LaunchConfig;
pub use cudarc::driver::PushKernelArg;

// -----------------------------------------------------------------------------
// `spawn!` needs one place to turn `&CudaVec<T>` into a device pointer word and
// `&scalar` into a by-value CUDA parameter. Rust cannot overload free functions on
// unrelated types without a trait or macro at the type level.
// -----------------------------------------------------------------------------

pub trait KernelArg {
    type Output: DeviceRepr;
    fn into_kernel_arg(self) -> Self::Output;
}

impl<T: DeviceRepr + Copy> KernelArg for &T {
    type Output = T;
    fn into_kernel_arg(self) -> Self::Output {
        *self
    }
}

impl<T> KernelArg for &CudaVec<T> {
    type Output = u64;
    fn into_kernel_arg(self) -> Self::Output {
        self.get_device_ptr() as u64
    }
}

impl<T> KernelArg for &mut CudaVec<T> {
    type Output = u64;
    fn into_kernel_arg(self) -> Self::Output {
        self.get_device_ptr() as u64
    }
}

#[inline]
pub fn kernel_arg<T: KernelArg>(value: T) -> T::Output {
    value.into_kernel_arg()
}

// Lazily created GPU context
static CUDA_CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();

// Cache loaded modules
static MODULES: OnceLock<Mutex<HashMap<String, Arc<CudaModule>>>> = OnceLock::new();

/// Launch a PTX kernel using a &str path
pub fn launch_generated_ptx<F>(
    ptx_path: &str,
    cfg: LaunchConfig,
    push_and_launch: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce(&Arc<CudaStream>, &CudaFunction, LaunchConfig) -> Result<(), DriverError>,
{
    // Get or create CUDA context
    let ctx = if let Some(ctx) = CUDA_CTX.get() {
        Arc::clone(ctx)
    } else {
        let created = CudaContext::new(0)?;
        CUDA_CTX.set(Arc::clone(&created)).ok();
        created
    };

    // Load or reuse PTX module
    let modules = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = modules.lock().unwrap();

    let module = if let Some(m) = guard.get(ptx_path) {
        Arc::clone(m)
    } else {
        let ptx_src = std::fs::read_to_string(ptx_path)?;
        let loaded = ctx.load_module(Ptx::from_src(ptx_src))?;
        guard.insert(ptx_path.to_string(), Arc::clone(&loaded));
        loaded
    };
    drop(guard);

    // Infer kernel name from PTX file stem
    let kernel_name = Path::new(ptx_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("PTX path has no UTF-8 file stem: {}", ptx_path))?;

    let kernel_name = kernel_name.strip_prefix("generated_").unwrap_or(kernel_name);

    // Load function
    let func = module.load_function(kernel_name)?;

    // Default stream
    let stream = ctx.default_stream();

    // Call user closure to push args & launch
    push_and_launch(&stream, &func, cfg)?;

    // Wait for GPU to finish
    stream.synchronize()?;

    Ok(())
}

// Load `ptx_path`, get the kernel named `kernel_name`, then run `push_and_launch`.
//
// `push_and_launch` must:
// 1. Call `stream.launch_builder(function)` to get a launch builder bound to that stream/kernel pair.
// 2. For each kernel parameter (in order), call `.arg(&value)` with a type that implements `DeviceRepr` 
//     on the GPU ABI (often `u64` for pointers, or plain scalars).
// 3. Call `unsafe { builder.launch(cfg) }` to enqueue the kernel on the stream.
//
// This function then waits for the stream to finish (`synchronize`) so errors surface here.
// pub fn launch_ptx_kernel<F, P, S>(
//     ptx_path: P,
//     kernel_name: S,
//     cfg: LaunchConfig,
//     push_and_launch: F,
// ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
// where
//     P: AsRef<Path>,
//     S: AsRef<str>,
//     F: FnOnce(&Arc<CudaStream>, &CudaFunction, LaunchConfig) -> Result<(), DriverError>,
// {
//     let path = ptx_path.as_ref();

//     // Reuse an existing context if another thread already initialized it.
//     let ctx: Arc<CudaContext> = if let Some(existing) = CUDA_CTX.get() {
//         Arc::clone(existing)
//     } else {
//         // First caller: create a context on device ordinal 0 (first GPU).
//         let created = match CudaContext::new(0) {
//             Ok(c) => c,
//             Err(e) => return Err(format!("CudaContext::new(0) failed: {e}").into()),
//         };
//         // Another thread might have won the race; `set` returns `Err` with the value we tried to insert.
//         match CUDA_CTX.set(created.clone()) {
//             Ok(()) => created,
//             Err(_) => CUDA_CTX
//                 .get()
//                 .map(Arc::clone)
//                 .ok_or_else(|| "CUDA_CTX unexpectedly empty after failed set".to_string())?,
//         }
//     };

//     // Serialize access to the path → module map (first load compiles/links PTX on the driver).
//     let modules = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
//     let mut guard = match modules.lock() {
//         Ok(g) => g,
//         Err(_) => return Err("MODULES mutex poisoned (another thread panicked while holding it)".into()),
//     };

//     let module: Arc<CudaModule> = if let Some(hit) = guard.get(path) {
//         Arc::clone(hit)
//     } else {
//         // Read the `.ptx` text from disk so we can hand it to the CUDA driver as module source.
//         let ptx_src = match std::fs::read_to_string(path) {
//             Ok(s) => s,
//             Err(e) => return Err(format!("failed to read PTX at {}: {e}", path.display()).into()),
//         };
//         // Ask the driver to load this PTX into the context (JIT if needed, depending on toolkit).
//         let loaded = match ctx.load_module(Ptx::from_src(ptx_src)) {
//             Ok(m) => m,
//             Err(e) => return Err(format!("ctx.load_module failed for {}: {e}", path.display()).into()),
//         };
//         guard.insert(path.to_path_buf(), Arc::clone(&loaded));
//         loaded
//     };
//     drop(guard);

//     // Look up the `__global__` entry symbol inside the module by name.
//     let function = match module.load_function(kernel_name.as_ref()) {
//         Ok(f) => f,
//         Err(e) => {
//             return Err(
//                 format!("module.load_function({:?}) failed: {e}", kernel_name.as_ref()).into(),
//             )
//         }
//     };

//     // Default stream for this context: operations are FIFO-ordered on it unless you use other streams.
//     let stream = ctx.default_stream();

//     // Caller fills in kernel arguments and submits the launch.
//     if let Err(e) = push_and_launch(&stream, &function, cfg) {
//         return Err(format!("kernel launch submission failed: {e}").into());
//     }

//     // Block the CPU until everything queued on this stream (including the kernel) has completed.
//     match stream.synchronize() {
//         Ok(()) => Ok(()),
//         Err(e) => Err(format!("stream.synchronize after kernel failed: {e}").into()),
//     }
// }

// /// Like [`launch_ptx_kernel`], but the entry name is inferred from a `generated_<kernel>.ptx` filename
// /// (stem without `generated_` prefix).
// pub fn launch_generated_ptx<F, P>(
//     ptx_path: P,
//     cfg: LaunchConfig,
//     push_and_launch: F,
// ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
// where
//     P: AsRef<Path>,
//     F: FnOnce(&Arc<CudaStream>, &CudaFunction, LaunchConfig) -> Result<(), DriverError>,
// {
//     let path = ptx_path.as_ref();
//     let stem = path
//         .file_stem()
//         .and_then(|s| s.to_str())
//         .ok_or_else(|| format!("PTX path has no UTF-8 file stem: {}", path.display()))?;
//     let kernel_name = stem.strip_prefix("generated_").unwrap_or(stem);
//     if kernel_name.is_empty() {
//         return Err(format!("empty kernel name after stripping prefix from {}", path.display()).into());
//     }
//     launch_ptx_kernel(path, kernel_name, cfg, push_and_launch)
// }

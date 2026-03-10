#[link(name = "lib_core")]
unsafe extern "C" {
    // fn add(a: i32, b: i32) -> i32;
    fn allocate_gpu_mem(a_d: *mut *mut f32);
    fn free_gpu_mem(device_data: *mut f32);
    fn copy_to_gpu(a_d: *mut f32, a_h: *const f32);
    fn copy_from_gpu(result_h: *mut f32, result_d: *mut f32);
    // fn launch_kernel(a_d: *mut f32, b_d: *mut f32, result_d: *mut f32);
}

// pub fn custom_add(a: i32, b: i32) -> i32 {
//     unsafe { add(a, b) }
// }

pub fn custom_allocate_gpu_mem(a_d: *mut *mut f32) {
    unsafe { allocate_gpu_mem(a_d) }
}

pub fn custom_copy_to_gpu(a_d: *mut f32, a_h: *const f32) {
    unsafe { copy_to_gpu(a_d, a_h) }
}

pub fn custom_copy_from_gpu(result_h: *mut f32, result_d: *mut f32) {
    unsafe { copy_from_gpu(result_h, result_d) }
}

// pub fn custom_launch_kernel(a_d: *mut f32, b_d: *mut f32, result_d: *mut f32) {
//     unsafe { launch_kernel(a_d, b_d, result_d) }
// }

pub fn custom_free_gpu_mem(device_data: *mut f32) {
    unsafe { free_gpu_mem(device_data) }
}

#[derive(Debug, Clone)]
pub struct CudaVec<T> {
    data: Vec<T>,
    device_ptr: *mut T
}

impl<T> CudaVec<T> {
    pub fn new(data: Vec<T>) -> Self {

        let vec_type = std::any::type_name::<T>();
        match vec_type {
            "f32" => {
                let mut device_ptr: *mut f32 = std::ptr::null_mut();           
                custom_allocate_gpu_mem(&mut device_ptr as *mut *mut f32);
                if !data.is_empty() {
                    println!("Copying to GPU");
                    custom_copy_to_gpu(device_ptr, data.as_ptr() as *mut f32);
                }

                Self {
                    data: data,
                    device_ptr: device_ptr as *mut T
                }
            }

            _ => {
                panic!("Error: Type {} not supported", vec_type);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_device_ptr(&self) -> *mut T {
        self.device_ptr
    }
} 

impl<T> Drop for CudaVec<T> {
    fn drop(&mut self) {
        if !self.device_ptr.is_null() {
            let vec_type = std::any::type_name::<T>();
            
            match vec_type {
                "f32" => {
                    println!("Dropping CudaVec<{}>: Freeing GPU memory at {:?}", vec_type, self.device_ptr);
                    custom_free_gpu_mem(self.device_ptr as *mut f32);
                }
            
                _ => {
                    println!("Dropping CudaVec<{}>: No custom cleanup needed.", vec_type);
                }
            }
        } else {
            println!("Dropping CudaVec<{}>: Pointer was null, no memory to free.", std::any::type_name::<T>());
        }
    }
}

pub trait KernelName {
    fn kernel_name() -> &'static str;
}

pub fn spawn<K: KernelName>(a_h: CudaVec<f32>, b_h: CudaVec<f32>, r_h: &mut Vec<f32>) {
    let function_name = K::kernel_name();
    let output_lib_path_generated = format!("kernel_and_launcher_generated_for_{function_name}.so");
    
    let output = std::process::Command::new("nvcc")
        .arg("-arch=sm_86")
        .arg(format!("generated_{function_name}.cu"))
        .arg(format!("generated_{function_name}_launcher.cu"))
        .arg("-o")
        .arg(&output_lib_path_generated)
        .arg("--shared")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-x")
        .arg("cu")
        .output()
        .expect("Failed to execute nvcc");

    if !output.stdout.is_empty() {
        eprintln!("NVCC stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        eprintln!("NVCC stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() {
        panic!("nvcc compilation failed with exit code: {:?}", output.status.code());
    }

    let absolute_path = std::env::current_dir()
        .unwrap()
        .join(&output_lib_path_generated);

    eprintln!("Loading library from: {}", absolute_path.display());

    unsafe {
        let mut result_device_ptr: *mut f32 = std::ptr::null_mut();
        crate::custom_allocate_gpu_mem(&mut result_device_ptr as *mut *mut f32);

        type LaunchKernelFuncFloat = unsafe extern "C" fn(
            *mut std::os::raw::c_float,
            *mut std::os::raw::c_float,
            *mut std::os::raw::c_float
        );

        let lib = libloading::Library::new(&absolute_path)
            .expect("Failed to load library");

        let launch_fn_name = format!("launch_generated_{function_name}");
        let symbol_name = std::ffi::CString::new(launch_fn_name)
            .expect("Failed to create CString");

        let launch_kernel_symbol: libloading::Symbol<LaunchKernelFuncFloat> =
            lib.get(symbol_name.as_bytes_with_nul())
                .expect("Failed to get symbol");

        launch_kernel_symbol(
            a_h.get_device_ptr() as *mut std::os::raw::c_float,
            b_h.get_device_ptr() as *mut std::os::raw::c_float,
            result_device_ptr as *mut std::os::raw::c_float,
        );

        let result_size = a_h.len();
        r_h.resize(result_size, 0.0);
        crate::custom_copy_from_gpu(r_h.as_mut_ptr(), result_device_ptr);
        crate::custom_free_gpu_mem(result_device_ptr);
    }
}
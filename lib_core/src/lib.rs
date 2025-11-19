#[link(name = "lib_core")]
unsafe extern "C" {
    fn add(a: i32, b: i32) -> i32;
    fn allocate_gpu_mem(a_d: *mut *mut f32);
    fn free_gpu_mem(device_data: *mut f32);
    fn copy_to_gpu(a_d: *mut f32, a_h: *const f32);
    fn copy_from_gpu(result_h: *mut f32, result_d: *mut f32);
    fn launch_kernel(a_d: *mut f32, b_d: *mut f32, result_d: *mut f32);
}

pub fn custom_add(a: i32, b: i32) -> i32 {
    unsafe { add(a, b) }
}

pub fn custom_allocate_gpu_mem(a_d: *mut *mut f32) {
    unsafe { allocate_gpu_mem(a_d) }
}

pub fn custom_copy_to_gpu(a_d: *mut f32, a_h: *const f32) {
    unsafe { copy_to_gpu(a_d, a_h) }
}

pub fn custom_copy_from_gpu(result_h: *mut f32, result_d: *mut f32) {
    unsafe { copy_from_gpu(result_h, result_d) }
}

pub fn custom_launch_kernel(a_d: *mut f32, b_d: *mut f32, result_d: *mut f32) {
    unsafe { launch_kernel(a_d, b_d, result_d) }
}

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

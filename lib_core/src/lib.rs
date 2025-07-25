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
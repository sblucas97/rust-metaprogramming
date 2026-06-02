#[link(name = "lib_core")]
unsafe extern "C" {
    fn allocate_gpu_mem(a_d: *mut *mut f32, n: usize);
    fn free_gpu_mem(device_data: *mut f32);
    fn copy_to_gpu(a_d: *mut f32, a_h: *const f32, n: usize);
    fn copy_from_gpu(result_h: *mut f32, result_d: *mut f32, n: usize);
}

pub fn cuda_allocate(a_d: *mut *mut f32, n: usize) {
    unsafe { allocate_gpu_mem(a_d, n) }
}

pub fn cuda_copy_to_device(a_d: *mut f32, a_h: *const f32, n: usize) {
    unsafe { copy_to_gpu(a_d, a_h, n) }
}

pub fn cuda_copy_to_host(result_h: *mut f32, result_d: *mut f32, n: usize) {
    unsafe { copy_from_gpu(result_h, result_d, n) }
}

pub fn cuda_free(device_data: *mut f32) {
    unsafe { free_gpu_mem(device_data) }
}


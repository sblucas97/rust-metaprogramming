use std::ffi::c_void;

// The C side (c_src/cuda.cu) is element-type-agnostic: every function takes a
// byte count. These wrappers keep the element-count interface callers think in
// and do the len -> bytes conversion in exactly one place.

#[link(name = "lib_core")]
unsafe extern "C" {
    fn allocate_gpu_mem(p: *mut *mut c_void, nbytes: usize);
    fn free_gpu_mem(p: *mut c_void);
    fn copy_to_gpu(dst_d: *mut c_void, src_h: *const c_void, nbytes: usize);
    fn copy_from_gpu(dst_h: *mut c_void, src_d: *const c_void, nbytes: usize);
}

fn nbytes<T>(len: usize) -> usize {
    len.checked_mul(size_of::<T>())
        .expect("CudaVec byte size overflows usize")
}

pub fn cuda_allocate<T>(p: &mut *mut T, len: usize) {
    unsafe { allocate_gpu_mem(p as *mut *mut T as *mut *mut c_void, nbytes::<T>(len)) }
}

pub fn cuda_copy_to_device<T>(dst_d: *mut T, src_h: *const T, len: usize) {
    unsafe { copy_to_gpu(dst_d as *mut c_void, src_h as *const c_void, nbytes::<T>(len)) }
}

pub fn cuda_copy_to_host<T>(dst_h: *mut T, src_d: *mut T, len: usize) {
    unsafe { copy_from_gpu(dst_h as *mut c_void, src_d as *mut c_void, nbytes::<T>(len)) }
}

pub fn cuda_free<T>(p: *mut T) {
    unsafe { free_gpu_mem(p as *mut c_void) }
}

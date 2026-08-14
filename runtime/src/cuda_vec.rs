use std::ops::IndexMut;
use crate::ffi;

/// Element types that may live in a `CudaVec`.
///
/// Sealed marker: only plain numeric scalars whose bytes can be memcpy'd
/// between host and device verbatim. Deliberately not implemented for `bool`
/// (not a DSL vector element) or arbitrary `T` (which could smuggle heap
/// pointers onto the GPU).
pub trait DeviceScalar: sealed::Sealed + Copy + 'static {}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_device_scalar {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl DeviceScalar for $t {}
    )*};
}

impl_device_scalar!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[derive(Debug, Clone)]
pub struct CudaVec<T> {
    data: Vec<T>,
    device_ptr: *mut T
}

impl<T> IndexMut<usize> for CudaVec<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}

impl<T> std::ops::Index<usize> for CudaVec<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<T> IndexMut<u64> for CudaVec<T> {
    fn index_mut(&mut self, idx: u64) -> &mut Self::Output {
        &mut self.data[idx as usize]
    }
}

impl<T> std::ops::Index<u64> for CudaVec<T> {
    type Output = T;
    fn index(&self, idx: u64) -> &Self::Output {
        &self.data[idx as usize]
    }
}

// Constructors and transfers require a device-copyable scalar; everything that
// only touches the host copy stays available for any T.
impl<T: DeviceScalar> CudaVec<T> {
    pub fn new_empty(data: Vec<T>, size: usize) -> Self {
        let mut device_ptr: *mut T = std::ptr::null_mut();
        ffi::cuda_allocate(&mut device_ptr, size);
        Self { data, device_ptr }
    }

    pub fn new(data: Vec<T>) -> Self {
        let mut device_ptr: *mut T = std::ptr::null_mut();
        ffi::cuda_allocate(&mut device_ptr, data.len());
        if !data.is_empty() {
            ffi::cuda_copy_to_device(device_ptr, data.as_ptr(), data.len());
        }
        Self { data, device_ptr }
    }

    pub fn copy_from_device(&mut self) {
        if self.data.is_empty() {
            return;
        }
        ffi::cuda_copy_to_host(self.data.as_mut_ptr(), self.device_ptr, self.data.len());
    }
}

impl<T> CudaVec<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_device_ptr(&self) -> *mut T {
        self.device_ptr
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Drop for CudaVec<T> {
    fn drop(&mut self) {
        // Null (zero-length vec) is fine: cudaFree(nullptr) is a no-op.
        ffi::cuda_free(self.device_ptr);
    }
}

use std::ops::IndexMut;
use crate::ffi;

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

impl<T> CudaVec<T> {
    pub fn new_empty(data: Vec<T>, size: usize) -> Self {

        let vec_type = std::any::type_name::<T>();
        match vec_type {
            "f32" => {
                let mut device_ptr: *mut f32 = std::ptr::null_mut();           
                ffi::cuda_allocate(&mut device_ptr as *mut *mut f32, size);
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

    pub fn new(data: Vec<T>) -> Self {

        let vec_type = std::any::type_name::<T>();
        match vec_type {
            "f32" => {
                let mut device_ptr: *mut f32 = std::ptr::null_mut();           
                ffi::cuda_allocate(&mut device_ptr as *mut *mut f32, data.len());
                if !data.is_empty() {
                    ffi::cuda_copy_to_device(
                        device_ptr, 
                        data.as_ptr() as *mut f32,
                        data.len()
                    );
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

    pub fn copy_from_device(&mut self) {
        let vec_type = std::any::type_name::<T>();
        match vec_type {
            "f32" => {
                ffi::cuda_copy_to_host(
                    self.data.as_mut_ptr() as *mut f32,
                    self.get_device_ptr() as *mut f32,
                    self.data.len()
                );
            }
            _ => panic!("copy_from_device: type {} not supported", vec_type),
        }
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
        if self.device_ptr.is_null() {
            eprintln!(
                "CudaVec<{}>: device pointer was null, nothing to free",
                std::any::type_name::<T>()
            );
            return;
        }

        let vec_type = std::any::type_name::<T>();
        match vec_type {
            "f32" => {
                ffi::cuda_free(self.device_ptr as *mut f32);
            }
            _ => {
                eprintln!("CudaVec<{}>: no GPU cleanup needed on drop", vec_type);
            }
        }
    }
}
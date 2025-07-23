use std::ffi::CString;

#[link(name = "lib_core")]
unsafe extern "C" {
    fn add(a: i32, b: i32) -> i32;
}

pub fn custom_add(a: i32, b: i32) -> i32 {
    unsafe { add(a, b) }
}

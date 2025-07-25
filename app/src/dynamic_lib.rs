use libloading::{Library, Symbol};

pub struct DynamicLib {
    _lib: &'static Library, // store a reference, not the value
    // pub mult: Symbol<'static, unsafe extern "C" fn(i32, i32) -> i32>,
    pub launch_kernel_2: Symbol<'static, unsafe extern "C" fn(*mut f32, *mut f32, *mut f32)>,
}

impl DynamicLib {
    pub unsafe fn new(lib_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let lib = Box::leak(Box::new(Library::new(lib_path)?)); // leak to extend lifetime
        // let mult = lib.get::<unsafe extern "C" fn(i32, i32) -> i32>(b"mult")?;
        let launch_kernel_2 = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, *mut f32)>(b"launch_kernel_2")?;
        Ok(DynamicLib {
            _lib: lib, // just store the reference
            // mult,
            launch_kernel_2,
        })
    }
}
//! Host <-> device round-trips for CudaVec across element types. These need a
//! working GPU, so they are `#[ignore]`d by default:
//!
//!     cargo test -p runtime -- --ignored
//!
//! A wrong element-size computation in the byte-based FFI shows up here as a
//! truncated or garbage tail after the copy back.

use runtime::cuda_vec::CudaVec;

fn roundtrip<T>(data: Vec<T>)
where
    T: runtime::cuda_vec::DeviceScalar + PartialEq + std::fmt::Debug,
{
    let expected = data.clone();
    let mut vec = CudaVec::new(data);
    // Clobber the host copy so the assert can only pass if the bytes really
    // travelled to the device and back.
    for slot in vec.as_mut_slice() {
        *slot = expected[0];
    }
    vec.copy_from_device();
    assert_eq!(vec.as_slice(), expected.as_slice());
}

#[test]
#[ignore = "requires GPU"]
fn roundtrip_f32() {
    roundtrip((0..1024).map(|i| i as f32 * 0.5).collect::<Vec<f32>>());
}

#[test]
#[ignore = "requires GPU"]
fn roundtrip_f64() {
    roundtrip((0..1024).map(|i| i as f64 * 0.25).collect::<Vec<f64>>());
}

#[test]
#[ignore = "requires GPU"]
fn roundtrip_u8() {
    roundtrip((0..1024u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>());
}

#[test]
#[ignore = "requires GPU"]
fn roundtrip_i16() {
    roundtrip((0..1024).map(|i| (i - 512) as i16).collect::<Vec<i16>>());
}

#[test]
#[ignore = "requires GPU"]
fn roundtrip_u32() {
    roundtrip((0..1024u32).map(|i| i * 3).collect::<Vec<u32>>());
}

#[test]
#[ignore = "requires GPU"]
fn roundtrip_i64() {
    roundtrip((0..1024i64).map(|i| i - 512).collect::<Vec<i64>>());
}

#[test]
#[ignore = "requires GPU"]
fn zero_length_vec_constructs_and_drops() {
    let mut vec = CudaVec::<f32>::new(Vec::new());
    assert!(vec.is_empty());
    assert!(vec.get_device_ptr().is_null());
    vec.copy_from_device(); // no-op, must not error
    // drop runs cuda_free(null), which is a no-op
}

#[test]
#[ignore = "requires GPU"]
fn new_empty_allocates_without_copy() {
    let mut vec = CudaVec::<u16>::new_empty(vec![0u16; 64], 64);
    assert!(!vec.get_device_ptr().is_null());
    assert_eq!(vec.len(), 64);
    vec.copy_from_device();
    assert_eq!(vec.len(), 64);
}

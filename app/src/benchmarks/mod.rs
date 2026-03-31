pub mod julia_k;
pub mod mm;
pub mod vector_sum_k;

use std::time::Instant;

pub fn run_timed<T, F: FnOnce() -> T>(name: &str, f: F) -> T {
    let start = Instant::now();
    let result = f();
    println!("[{}] elapsed: {}ms", name, start.elapsed().as_millis());
    result
}

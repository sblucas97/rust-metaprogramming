pub mod julia;
pub mod mm;
pub mod vector_sum;

use std::time::Instant;

pub fn run_timed<T, F: FnOnce() -> T>(name: &str, f: F) -> T {
    let start = Instant::now();
    let result = f();
    println!("[{}] elapsed: {}ms", name, start.elapsed().as_millis());
    result
}

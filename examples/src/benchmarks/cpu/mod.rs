//! Pure-Rust sequential CPU implementations of the benchmarks (single threaded,
//! no parallelism at all).
//! These do not use the custom compiler at all -- they exist as a third
//! reference point next to the DSL ("rust-gpu") and hand-written CUDA.
//!
//! Every module here mirrors its DSL counterpart's arithmetic exactly, reuses
//! the same input generator where the DSL side has one, and times only the
//! compute -- the DSL side times just the spawn!, so allocation and host/device
//! transfers are outside the measurement on both sides.
pub mod julia;
pub mod mm;
pub mod nbodies;
pub mod nearest_neighbor;
pub mod raytracer;
pub mod ripple;
pub mod vector_sum;

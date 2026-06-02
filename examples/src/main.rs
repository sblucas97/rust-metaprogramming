mod benchmarks;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);


    benchmarks::vector_sum::run(size);
    // benchmarks::mm::run(size);
    // benchmarks::julia::run(size);
    // benchmarks::raytracer::run(size);
    // benchmarks::ripple::run(size);

    // benchmarks::tc::run();

    let mut x: i32 = 20;
    let mut y: i32 = 15;
    let r1: &mut i32 = &mut x;
    let r2: &mut i32 = &mut y;

    *r1 = 21;
    *r2 = 16;

    let a = *r1;
    let b = *r2;

}
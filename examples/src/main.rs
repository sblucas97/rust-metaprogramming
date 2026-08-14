mod benchmarks;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // `demo`                    -> run everything, default sizes
    // `demo 4096`               -> run everything, size 4096 (clamped per-bench as before)
    // `demo nbodies`            -> run only nbodies, default size
    // `demo nbodies 4096`       -> run only nbodies, size 4096
    // `demo nbodies_cpu 4096`   -> run the pure-Rust sequential CPU version instead
    let (bench, size_arg) = match args.get(1).map(String::as_str) {
        Some(name) if name.parse::<usize>().is_err() => (Some(name), args.get(2)),
        _ => (None, args.get(1)),
    };

    let size: usize = size_arg
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

    match bench {
        Some("vector_sum") => {
            benchmarks::vector_sum::run(size);
        }
        Some("mm") => {
            benchmarks::mm::run(size);
        }
        Some("julia") => {
            benchmarks::julia::run(size);
        }
        Some("raytracer") => {
            benchmarks::raytracer::run(size);
        }
        Some("ripple") => {
            benchmarks::ripple::run(size);
        }
        Some("nearest_neighbor") => {
            benchmarks::nearest_neighbor::run(size);
        }
        Some("nbodies") => {
            benchmarks::nbodies::run(size);
        }
        // Pure-Rust sequential CPU counterparts -- same arithmetic, no custom compiler.
        Some("vector_sum_cpu") => {
            benchmarks::cpu::vector_sum::run(size);
        }
        Some("mm_cpu") => {
            benchmarks::cpu::mm::run(size);
        }
        Some("julia_cpu") => {
            benchmarks::cpu::julia::run(size);
        }
        Some("raytracer_cpu") => {
            benchmarks::cpu::raytracer::run(size);
        }
        Some("ripple_cpu") => {
            benchmarks::cpu::ripple::run(size);
        }
        Some("nearest_neighbor_cpu") => {
            benchmarks::cpu::nearest_neighbor::run(size);
        }
        Some("nbodies_cpu") => {
            benchmarks::cpu::nbodies::run(size);
        }
        Some(other) => {
            eprintln!(
                "unknown benchmark `{other}` (expected one of: vector_sum, mm, julia, raytracer, \
                 ripple, nearest_neighbor, nbodies -- each also available with a `_cpu` suffix)"
            );
            std::process::exit(1);
        }
        None => {
            benchmarks::vector_sum::run(size);
            benchmarks::mm::run(size.min(1024));
            benchmarks::julia::run(size.min(1024));
            benchmarks::raytracer::run(size.min(1024));
            benchmarks::ripple::run(size.min(1024));
            benchmarks::nearest_neighbor::run(size);
            benchmarks::nbodies::run(size.min(4096));
        }
    }
}

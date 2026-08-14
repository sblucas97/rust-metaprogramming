// TEMPORARY diagnostic: breaks the "elapsed" number reported by the DSL benchmarks
// into its phases (context init, PTX JIT, H2D, kernel, D2H) so we can see what the
// rust-gpu column is actually measuring. Delete after use.
use std::time::Instant;

use runtime::launch::{LaunchConfig, PushKernelArg};
use runtime::CudaVec;

macro_rules! phase {
    ($label:expr, $body:expr) => {{
        let t = Instant::now();
        let out = $body;
        println!("  {:<28} {:>10.3} ms", $label, t.elapsed().as_secs_f64() * 1000.0);
        out
    }};
}

fn launch(ptx: &str, cfg: LaunchConfig, args: &[u64], scalars: &[f32], layout: &[u8]) {
    // layout: 0 = next u64 arg, 1 = next f32 scalar
    runtime::launch::launch_generated_ptx(ptx, cfg, |stream, func, cfg| {
        let mut builder = stream.launch_builder(func);
        let (mut ai, mut si) = (0usize, 0usize);
        // Keep values alive for the duration of the builder.
        let u64s: Vec<u64> = args.to_vec();
        let f32s: Vec<f32> = scalars.to_vec();
        for k in layout {
            match k {
                0 => {
                    builder.arg(&u64s[ai]);
                    ai += 1;
                }
                _ => {
                    builder.arg(&f32s[si]);
                    si += 1;
                }
            }
        }
        unsafe { builder.launch(cfg).map(|_| ()) }
    })
    .expect("launch failed");
}

fn probe_nearest_neighbor(n: usize) {
    println!("nearest_neighbor n={n}");
    let ptx = concat!(env!("CARGO_MANIFEST_DIR"), "/generated_euclid.ptx");

    let locations = phase!("host gen input", {
        let mut state: u32 = 7919;
        let mut v = Vec::with_capacity(n * 2);
        for _ in 0..n {
            for range in [180.0f32, 360.0f32] {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = (state >> 8) % 32767;
                v.push(x as f32 / 32767.0 * range - range / 2.0);
            }
        }
        v
    });
    let zeros = phase!("host alloc output vec", vec![0.0f32; n]);

    let total = Instant::now();
    // First cudaMalloc pays for CUDA runtime primary-context creation.
    let d_loc = phase!("H2D input (+ctx init)", CudaVec::new(locations));
    let mut d_dst = phase!("H2D output zeros", CudaVec::new(zeros));

    let threads: u32 = 128;
    let blocks: u32 = (n as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let args = [d_loc.get_device_ptr() as u64, d_dst.get_device_ptr() as u64, n as u64];
    phase!("kernel #1 (+PTX JIT)", launch(ptx, cfg, &args, &[30.0, 90.0], &[0, 0, 0, 1, 1]));
    phase!("kernel #2 (warm)", launch(ptx, cfg, &args, &[30.0, 90.0], &[0, 0, 0, 1, 1]));
    phase!("kernel #3 (warm)", launch(ptx, cfg, &args, &[30.0, 90.0], &[0, 0, 0, 1, 1]));
    phase!("D2H result", d_dst.copy_from_device());
    println!("  {:<28} {:>10.3} ms", "TOTAL (benchmark region)", total.elapsed().as_secs_f64() * 1000.0);
}

fn probe_raytracer(dim: usize) {
    println!("raytracer dim={dim}");
    let ptx = concat!(env!("CARGO_MANIFEST_DIR"), "/generated_raytracing.ptx");

    let mut state: u32 = 313;
    let mut rnd = |x: f32| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = ((state >> 1) % 32767) + 1;
        x * v as f32 / 32767.0
    };
    let (radius, sum) = (160.0f32, 20.0f32);
    let mut spheres = Vec::with_capacity(140);
    for _ in 0..20 {
        spheres.push(rnd(1.0));
        spheres.push(rnd(1.0));
        spheres.push(rnd(1.0));
        spheres.push(rnd(radius) + sum);
        spheres.push(rnd(dim as f32) - dim as f32 / 2.0);
        spheres.push(rnd(dim as f32) - dim as f32 / 2.0);
        spheres.push(rnd(256.0) - 128.0);
    }
    let img = phase!("host alloc output vec", vec![0.0f32; dim * dim * 4]);

    let total = Instant::now();
    let d_sph = phase!("H2D spheres (+ctx init)", CudaVec::new(spheres));
    let mut d_img = phase!("H2D image zeros", CudaVec::new(img));

    let block: u32 = 16;
    let g = (dim as u32).div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (g, g, 1),
        block_dim: (block, block, 1),
        shared_mem_bytes: 0,
    };
    let args = [
        d_sph.get_device_ptr() as u64,
        d_img.get_device_ptr() as u64,
        dim as u64,
        dim as u64,
    ];
    phase!("kernel #1 (+PTX JIT)", launch(ptx, cfg, &args, &[], &[0, 0, 0, 0]));
    phase!("kernel #2 (warm)", launch(ptx, cfg, &args, &[], &[0, 0, 0, 0]));
    phase!("kernel #3 (warm)", launch(ptx, cfg, &args, &[], &[0, 0, 0, 0]));
    phase!("D2H image", d_img.copy_from_device());
    println!("  {:<28} {:>10.3} ms", "TOTAL (benchmark region)", total.elapsed().as_secs_f64() * 1000.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("raytracer");
    let size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    match which {
        "nearest_neighbor" => probe_nearest_neighbor(size),
        "raytracer" => probe_raytracer(size),
        other => eprintln!("unknown probe: {other}"),
    }
}

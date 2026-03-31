use lib::{cuda_module, spawn};
use lib_core::CudaVec;

/// Simple LCG matching the seed/range from the Elixir reference:
/// rnd(x) = x * randint(1, 32767) / 32767
fn rnd(x: f32, state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let v = ((*state >> 1) % 32767) + 1;
    x * v as f32 / 32767.0
}

fn generate_spheres(dim: usize) -> Vec<f32> {
    let (radius, sum) = match dim {
        256  => (20.0f32,  5.0f32),
        1024 => (80.0f32,  20.0f32),
        2048 => (120.0f32, 20.0f32),
        _    => (160.0f32, 20.0f32),
    };

    let mut state: u32 = 313;
    let mut spheres = Vec::with_capacity(20 * 7);
    for _ in 0..20 {
        spheres.push(rnd(1.0, &mut state));                        // r
        spheres.push(rnd(1.0, &mut state));                        // g
        spheres.push(rnd(1.0, &mut state));                        // b
        spheres.push(rnd(radius, &mut state) + sum);               // radius
        spheres.push(rnd(dim as f32, &mut state) - dim as f32 / 2.0); // x
        spheres.push(rnd(dim as f32, &mut state) - dim as f32 / 2.0); // y
        spheres.push(rnd(256.0, &mut state) - 128.0);             // z
    }
    spheres
}

pub fn run(dim: usize) -> CudaVec<f32> {
    let spheres: CudaVec<f32> = CudaVec::new(generate_spheres(dim));
    let mut image: CudaVec<f32> = CudaVec::new(vec![0.0f32; dim * dim * 4]);

    let block: u32 = 16;
    let grid_x = (dim as u32 + block - 1) / block;
    let grid_y = (dim as u32 + block - 1) / block;

    spawn!(
        raytracer_kernel::raytracing,
        (grid_x, grid_y, 1),
        (block, block, 1),
        spheres,
        image,
        dim as u64,
        dim as u64
    );
    image.copy_from_device();
    image
}

#[cuda_module]
pub mod raytracer_kernel {
    use lib_core::CudaVec;
    use lib::kernel;

    #[kernel]
    pub fn raytracing(spheres: &CudaVec<f32>, image: &mut CudaVec<f32>, width: u64, height: u64) {
        let x: u64 = threadIdx.x + blockIdx.x * blockDim.x;
        let y: u64 = threadIdx.y + blockIdx.y * blockDim.y;
        if x < width && y < height {
            let offset: u64 = x + y * width;

            let ox: f32 = x as f32 - width  as f32 / 2.0f32;
            let oy: f32 = y as f32 - height as f32 / 2.0f32;

            let mut r: f32 = 0.0f32;
            let mut g: f32 = 0.0f32;
            let mut b: f32 = 0.0f32;
            let mut maxz: f32 = -99999.0f32;

            for i in (0u64, 20u64).step_by(1u64) {
                let sphere_radius: f32 = spheres[i * 7 + 3];
                let dx: f32 = ox - spheres[i * 7 + 4];
                let dy: f32 = oy - spheres[i * 7 + 5];
                let mut n: f32 = 0.0f32;
                let mut t: f32 = -99999.0f32;

                if (dx * dx + dy * dy) < (sphere_radius * sphere_radius) {
                    let dz: f32 = sqrtf(sphere_radius * sphere_radius - dx * dx - dy * dy);
                    n = dz / sqrtf(sphere_radius * sphere_radius);
                    t = dz + spheres[i * 7 + 6];
                } else {
                    t = -99999.0f32;
                    n = 0.0f32;
                }

                if t > maxz {
                    let fscale: f32 = n;
                    r    = spheres[i * 7 + 0] * fscale;
                    g    = spheres[i * 7 + 1] * fscale;
                    b    = spheres[i * 7 + 2] * fscale;
                    maxz = t;
                }
            }

            image[offset * 4 + 0] = r * 255.0f32;
            image[offset * 4 + 1] = g * 255.0f32;
            image[offset * 4 + 2] = b * 255.0f32;
            image[offset * 4 + 3] = 255.0f32;
        }
    }
}

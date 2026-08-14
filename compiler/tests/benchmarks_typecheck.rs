//! Regression suite: every benchmark kernel in examples/src/benchmarks must
//! lower and type-check. Kernel bodies are mirrored here verbatim so the suite
//! runs without a GPU or nvcc.

use compiler::{context::Context, lower::lower_fn, type_checker::type_check, types::TypeError};

fn assert_typechecks(item: syn::ItemFn) {
    let name = item.sig.ident.to_string();
    let func = lower_fn(&item).unwrap_or_else(|e| panic!("`{name}` failed to lower: {e}"));
    let mut ctx = Context::new();
    if let Err(e) = type_check(&func, &mut ctx) {
        panic!("`{name}` failed to type-check: {e:?}");
    }
}

#[test]
fn vector_sum_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn add_vectors(a: &CudaVec<f32>, b: &CudaVec<f32>, result: &mut CudaVec<f32>, n: u64) {
            let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
            result[idx] = a[idx] + b[idx];
        }
    });
}

#[test]
fn mm_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn mm(a: &CudaVec<f32>, b: &CudaVec<f32>, c: &mut CudaVec<f32>, m: u64, n: u64, k: u64) {
            let row: u64 = blockIdx.y * blockDim.y + threadIdx.y;
            let col: u64 = blockIdx.x * blockDim.x + threadIdx.x;
            let mut sum: f32 = 0.0f32;
            if col < k && row < m {
                for i in (0u64, n).step_by(1u64) {
                    sum = sum + a[row * n + i] * b[i * k + col];
                }
                c[row * k + col] = sum;
            }
        }
    });
}

#[test]
fn julia_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn julia_kernel(ptr: &mut CudaVec<f32>, dim: u64) {
            let x: u64 = blockIdx.x;
            let y: u64 = blockIdx.y;
            if x < dim && y < dim {
                let offset: u64 = x + y * dim;
                let scale: f32 = 0.1f32;
                let jx: f32 = scale * (dim - x) as f32 / dim as f32;
                let jy: f32 = scale * (dim - y) as f32 / dim as f32;
                let cr: f32 = (0.0f32 - 0.8f32);
                let ci: f32 = 0.156f32;
                let mut ar: f32 = jx;
                let mut ai: f32 = jy;
                let mut julia_value: f32 = 1.0f32;
                let mut escaped: u32 = 0;
                for _i in (0u64, 200u64).step_by(1u64) {
                    if escaped == 0 {
                        let nar: f32 = ((ar * ar) - (ai * ai)) + cr;
                        let nai: f32 = ((ai * ar) + (ar * ai)) + ci;
                        if ((nar * nar) + (nai * nai)) > 1000.0f32 {
                            julia_value = 0.0f32;
                            escaped = 1;
                        }
                        if escaped == 0 {
                            ar = nar;
                            ai = nai;
                        }
                    }
                }
                ptr[offset * 4 + 0] = 255.0f32 * julia_value;
                ptr[offset * 4 + 1] = 0.0f32;
                ptr[offset * 4 + 2] = 0.0f32;
                ptr[offset * 4 + 3] = 255.0f32;
            }
        }
    });
}

#[test]
fn raytracer_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn raytracing(spheres: &CudaVec<f32>, image: &mut CudaVec<f32>, width: u64, height: u64) {
            let x: u64 = threadIdx.x + blockIdx.x * blockDim.x;
            let y: u64 = threadIdx.y + blockIdx.y * blockDim.y;
            if x < width && y < height {
                let offset: u64 = x + y * width;
                let ox: f32 = x as f32 - width as f32 / 2.0f32;
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
                        r = spheres[i * 7 + 0] * fscale;
                        g = spheres[i * 7 + 1] * fscale;
                        b = spheres[i * 7 + 2] * fscale;
                        maxz = t;
                    }
                }
                image[offset * 4 + 0] = r * 255.0f32;
                image[offset * 4 + 1] = g * 255.0f32;
                image[offset * 4 + 2] = b * 255.0f32;
                image[offset * 4 + 3] = 255.0f32;
            }
        }
    });
}

#[test]
fn ripple_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn ripple_kernel(ptr: &mut CudaVec<f32>, dim: u64, ticks: f32) {
            let x: u64 = blockIdx.x;
            let y: u64 = blockIdx.y;
            if x < dim && y < dim {
                let offset: u64 = x + y * dim;
                let dim_f: f32 = dim as f32;
                let fx: f32 = 0.5f32 * x as f32 - dim_f / 15.0f32;
                let fy: f32 = 0.5f32 * y as f32 - dim_f / 15.0f32;
                let d: f32 = sqrtf(fx * fx + fy * fy);
                let grey: f32 = floorf(
                    128.0f32 + 127.0f32 * cosf(d / 10.0f32 - ticks / 7.0f32) / (d / 10.0f32 + 1.0f32),
                );
                ptr[offset * 4 + 0] = grey;
                ptr[offset * 4 + 1] = grey;
                ptr[offset * 4 + 2] = grey;
                ptr[offset * 4 + 3] = 255.0f32;
            }
        }
    });
}

#[test]
fn nearest_neighbor_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn euclid(
            d_locations: &CudaVec<f32>,
            d_distances: &mut CudaVec<f32>,
            num_records: u64,
            lat: f32,
            lng: f32,
        ) {
            let global_id: u64 = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
            let ilat: u64 = 2 * global_id;
            let ilng: u64 = 2 * global_id + 1;
            if global_id < num_records {
                let dlat: f32 = lat - d_locations[ilat];
                let dlng: f32 = lng - d_locations[ilng];
                d_distances[global_id] = sqrtf(dlat * dlat + dlng * dlng);
            }
        }
    });
}

#[test]
fn nbodies_force_kernel_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn gpu_n_bodies(p: &mut CudaVec<f32>, dt: f32, n: u64, softening: f32) {
            let i: u64 = blockDim.x * blockIdx.x + threadIdx.x;
            if i < n {
                let mut fx: f32 = 0.0f32;
                let mut fy: f32 = 0.0f32;
                let mut fz: f32 = 0.0f32;
                for j in (0u64, n).step_by(1u64) {
                    let dx: f32 = p[6 * j] - p[6 * i];
                    let dy: f32 = p[6 * j + 1] - p[6 * i + 1];
                    let dz: f32 = p[6 * j + 2] - p[6 * i + 2];
                    let dist_sqr: f32 = dx * dx + dy * dy + dz * dz + softening;
                    let inv_dist: f32 = 1.0f32 / sqrtf(dist_sqr);
                    let inv_dist3: f32 = inv_dist * inv_dist * inv_dist;
                    fx = fx + dx * inv_dist3;
                    fy = fy + dy * inv_dist3;
                    fz = fz + dz * inv_dist3;
                }
                p[6 * i + 3] = p[6 * i + 3] + dt * fx;
                p[6 * i + 4] = p[6 * i + 4] + dt * fy;
                p[6 * i + 5] = p[6 * i + 5] + dt * fz;
            }
        }
    });
}

#[test]
fn nbodies_integrate_kernel_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn gpu_integrate(p: &mut CudaVec<f32>, dt: f32, n: u64) {
            let i: u64 = blockDim.x * blockIdx.x + threadIdx.x;
            if i < n {
                p[6 * i] = p[6 * i] + p[6 * i + 3] * dt;
                p[6 * i + 1] = p[6 * i + 1] + p[6 * i + 4] * dt;
                p[6 * i + 2] = p[6 * i + 2] + p[6 * i + 5] * dt;
            }
        }
    });
}

// Not (yet) a real benchmark: pins the extended scalar set working end-to-end
// from surface syntax -- f64 vectors, signed locals, negation, mixed-rank
// widening -- through lowering and the type checker.
#[test]
fn extended_scalar_kernel_typechecks() {
    assert_typechecks(syn::parse_quote! {
        pub fn axpy_f64(a: &CudaVec<f64>, out: &mut CudaVec<f64>, n: u64) {
            let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
            if idx < n {
                let sign: i32 = -1;
                let wide: i64 = sign as i64;
                let scale: f64 = 2.5f64 * wide as f64;
                out[idx] = a[idx] * scale;
            }
        }
    });
}

// The old examples/src/benchmarks/tc.rs smoke test: deliberately ill-typed,
// must be rejected at the offending let.
#[test]
fn ill_typed_kernel_is_rejected() {
    let item: syn::ItemFn = syn::parse_quote! {
        pub fn add_vectors() {
            let x: u64 = 10;
            let y: u64 = 12.0;
            let c: u64 = x + y;
        }
    };
    let func = lower_fn(&item).expect("lowers fine; the type error comes later");
    let mut ctx = Context::new();
    assert_eq!(
        type_check(&func, &mut ctx),
        Err(TypeError::LetTypeMismatch {
            name: "y".into(),
            expected: "U64".into(),
            found: "F32".into(),
        })
    );
}

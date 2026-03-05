use std::fs::OpenOptions;
use std::io::Write;

use crate::helpers;

pub fn gen_kernel(name: &String) {
    // let file_name = format!("generated_{name}.cu");

    // let kernel = format!(r#"
    //     __global__ void {name}(const float *a, const float *b, float *result) {{
    //         int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //         if (idx < ROWS * COLS) {{
    //             result[idx] = a[idx] + b[idx];
    //         }}
    //     }}
    // "#);

    // let mut file = OpenOptions::new()
    //     .create(true)
    //     .write(true)
    //     .truncate(true)
    //     .open(&file_name)
    //     .expect("Failed to create output file");

    // writeln!(file, "{}", helpers::get_headers()).unwrap();
    // writeln!(file, "{}", helpers::gen_contants()).unwrap();
    // writeln!(file, "{}", kernel).unwrap();
}

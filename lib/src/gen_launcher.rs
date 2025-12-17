use std::fs::OpenOptions;
use std::io::Write;

use crate::helpers;

pub fn gen_launcher(name: &String) {
    let file_name = format!("generated_{name}_launcher.cu");

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&file_name)
        .expect("Failed to create output file");

     writeln!(file, "{}", helpers::get_headers()).unwrap();
     let incluce_kernel_header = format!("#include \"generated_{name}.h\"");
     writeln!(file, "{}", incluce_kernel_header).unwrap();
     writeln!(file, "{}", helpers::gen_contants()).unwrap();

    let launcher = format!(r#"
extern "C" void launch_{name}() {{
    int total = ROWS * COLS;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    {name}<<<grid_size, block_size>>>();
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");
    cudaDeviceSynchronize();
}}
    "#);
     writeln!(file, "{}", launcher).unwrap();
}

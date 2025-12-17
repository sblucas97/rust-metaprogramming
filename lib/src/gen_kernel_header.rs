use std::fs::OpenOptions;
use std::io::Write;

use crate::helpers;

pub fn gen_kernel_header(name: &String) {
    let file_name = format!("generated_{name}.h");

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&file_name)
        .expect("Failed to create output file");

     writeln!(file, "{}", helpers::get_header_init_headers()).unwrap();
     writeln!(file, "{}", helpers::gen_contants()).unwrap();
     let fn_name = format!(r#"
        __global__ void {name}();
     "#);
     writeln!(file, "{}", fn_name).unwrap();
     writeln!(file, "{}", helpers::get_header_end_headers()).unwrap();
}

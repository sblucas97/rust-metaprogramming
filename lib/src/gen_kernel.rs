use std::fs::OpenOptions;
use std::io::Write;

use crate::helpers;

pub fn gen_kernel(name: &String) {
    let file_name = format!("generated_{name}.cu");

    let kernel = format!(r#"
        __global__ void {name}() {{
            fprintf("hello world");
        }}
    "#);

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&file_name)
        .expect("Failed to create output file");

     writeln!(file, "{}", helpers::get_headers()).unwrap();
     writeln!(file, "{}", kernel).unwrap();
}

pub fn get_headers() -> String {
    let stdio_header = r#"
    #include <cuda_runtime.h>
    #include <iostream>
    "#;

    return stdio_header.to_string();
}

pub fn gen_contants() -> String {
    let define_block = r#"
        #define ROWS 100
        #define COLS 100
    "#;

    return define_block.to_string();
}

pub fn get_header_init_headers() -> String {
    r#"
    #ifndef KERNEL_H
    #define KERNEL_H
    "#.to_string()
}

pub fn get_header_end_headers() -> String {
    r#"
    #endif
    "#.to_string()
}

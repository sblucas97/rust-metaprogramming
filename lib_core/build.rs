use std::process::Command;
use std::env;
use std::path::PathBuf;


fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let add_o = out_dir.join("add.o");
    let cuda_o = out_dir.join("cuda.o");
    let lib_path = out_dir.join("liblib_core.a");

    // Compile add.c using cc
    Command::new("cc")
        .args(&["-c", "src/c_src/add.c", "-o", add_o.to_str().unwrap(), "-fPIC"])
        .status()
        .expect("failed to compile add.c");

    // Compile cuda.cu using nvcc
    Command::new("nvcc")
        .args(&["-c", "src/c_src/cuda.cu", "-o", cuda_o.to_str().unwrap(), "-arch=sm_86", "-Xcompiler", "-fPIC"])
        .status()
        .expect("failed to compile cuda.cu");

    // Create a single static lib containing both
    Command::new("ar")
        .args(&["crus", lib_path.to_str().unwrap(), add_o.to_str().unwrap(), cuda_o.to_str().unwrap()])
        .status()
        .expect("failed to create liblib_core.a");

    // Link the static lib
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.8/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=lib_core");
    // Rerun if files change
    println!("cargo:rerun-if-changed=src/c_src/add.c");
    println!("cargo:rerun-if-changed=src/c_src/cuda.cu");
    println!("cargo:rerun-if-changed=src/c_src/lib.h");
}

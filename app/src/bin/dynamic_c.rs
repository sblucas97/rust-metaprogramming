use libloading::{Library, Symbol};
use std::{fs::write, process::Command};

// file created to run using cargo run --bin dynamic_c
fn main() {
    let new_c_function = r#"
        #include<stdio.h>

        int mult(int a, int b) { return a * b; } 

        int main() {
            return 0;
        }
    "#;

    let file_path = "dynamic.c";

    write(file_path, new_c_function).expect("Failed to create file");

    // compile to shared object
    let so_path = "./libdyn.so";
    let output = Command::new("gcc")
        .args(["-shared", "-fPIC", file_path, "-o", so_path])
        .output()
        .expect("Failed to compile shared object");

    if !output.status.success() {
        eprintln!("GCC Error: {}", String::from_utf8_lossy(&output.stderr));
        return;
    }

    unsafe {
        let lib = Library::new(so_path).expect("Failed to load lib");
        let mult: Symbol<unsafe extern "C" fn(i32, i32) -> i32> =
            lib.get(b"mult").expect("Failed to find symbol");
        
        let result = mult(12, 12);
        println!("{}", result);
    }
}
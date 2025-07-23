/*use std::fs;
use std::process::Command;
use std::path::Path;
use lib::my_macro;
*/
mod rust_macros;

fn main() {
    

    let r = lib_core::custom_add(300, 400);

    println!("result {}", r);
    /*
    custom_for!(
        let x = 0; x < 10; x += 1 => {
            println!("from outside")
        }
    );
    custom_for!(let x = 10; x > 0; x -= 1);
    */

    /*
    let total_sum = custom_reduce!([1, 2, 3, 10, 20], item => acc + item, 1555);  
    let total_product = custom_reduce!([1, 2, 3], item => acc * item, 1);  
    println!("{}", total_sum);
    println!("{}", total_product);
    */

    /*let filename = "runtime_generated.c";
    let binary = "./runtime_generated";

    my_macro! {
        xx a : 10;
        xx b : 10;
        xx r : a + b;
        print("%d", r);

        xx a1 : 20;
        xx b1 : 30;
        xx r2 : a1 * b1;
        print("%d", r2);
    };

    let compile_status = Command::new("gcc")
        .arg(filename)
        .arg("-o")
        .arg(binary)
        .status()
        .expect("Failed to run gcc");

    if !compile_status.success() {
        panic!("GCC compilation failed");
    }

    if !Path::new(binary).exists() {
        panic!("Compile binary not found");
    }

    println!("Running generated C program");
    let run_status = Command::new(binary)
        .status()
        .expect("Failed to run C binary");

    if !run_status.success() {
        panic!("C binary ran with a non-zero exit code");
    }
*/
}

fn main() {
    cc::Build::new()
        .include("src/c_src")
        .files([
            "src/c_src/add.c",
        ])
        .compile("lib_core");

    println!("cargo:rerun-if-changed=src/c_src/lib.h");
    println!("cargo:rerun-if-changed=src/c_src/add.c");
}

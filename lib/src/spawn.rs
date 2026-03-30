use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Path, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

struct SpawnInput {
    kernel: Path,
    args: Punctuated<Expr, Token![,]>,
}

impl Parse for SpawnInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kernel: Path = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let args = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;

        Ok(SpawnInput { kernel, args })
    }
}

pub fn spawn_impl(input: TokenStream) -> TokenStream {
    let SpawnInput { kernel, args } = parse_macro_input!(input as SpawnInput);
    let kernel_name = kernel.segments.last().unwrap().ident.to_string();
    let arg_pushes = args.into_iter().enumerate().map(|(i, arg)| {
        let ident = format_ident!("arg_{i}");
        quote! {
            let #ident = lib_core::launch::kernel_arg(&(#arg));
            builder.arg(&#ident);
        }
    });

    let ptx_file = format!("generated_{}.ptx", kernel_name);

    let expanded = quote! {
        {
            let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/", #ptx_file);

            let cfg = lib_core::launch::LaunchConfig {
                grid_dim: ((100 * 100 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            lib_core::launch::launch_generated_ptx(ptx_path, cfg, |stream, func, cfg| {
                use lib_core::launch::PushKernelArg;

                let mut builder = stream.launch_builder(func);

                #( #arg_pushes )*

                unsafe { builder.launch(cfg).map(|_| ()) }
            }).expect("kernel launch failed");
        }
    };

    expanded.into()

    // quote! {
    //     {
    //         let name = #kernel_str;
    //         let so_name = format!("kernel_and_launcher_generated_for_{name}.so");

    //         let output = std::process::Command::new("nvcc")
    //             .args([
    //                 "-arch=sm_86",
    //                 &format!("generated_{name}.cu"),
    //                 &format!("generated_{name}_launcher.cu"),
    //                 "-o", &so_name,
    //                 "--shared",
    //                 "-Xcompiler", "-fPIC",
    //                 "-x", "cu",
    //             ])
    //             .output()
    //             .expect("failed to run nvcc — is it on PATH?");

    //         if !output.stdout.is_empty() {
    //             eprintln!("nvcc stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    //         }
    //         if !output.stderr.is_empty() {
    //             eprintln!("nvcc stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    //         }
    //         if !output.status.success() {
    //             panic!(
    //                 "nvcc compilation failed (exit {:?}) for kernel `{name}`",
    //                 output.status.code()
    //             );
    //         }


        
    //         let so_path = std::env::current_dir().unwrap().join(&so_name);
    //         eprintln!("spawn: loading {}", so_path.display());

    //         unsafe {
    //             // type LaunchFn = unsafe extern "C" fn( #( #fn_arg_types ),* );

    //             let lib = libloading::Library::new(&so_path)
    //                 .unwrap_or_else(|e| panic!("failed to load `{}`: {e}", so_path.display()));

    //             let symbol_name = format!("launch_generated_{name}\0");
    //             let launch: libloading::Symbol<LaunchFn> = lib
    //                 .get(symbol_name.as_bytes())
    //                 .unwrap_or_else(|e| panic!("symbol `launch_generated_{name}` not found: {e}"));

    //             launch( #( #fn_arg_values ),* );
    //         }

    //     }
    // }
    // .into()
}
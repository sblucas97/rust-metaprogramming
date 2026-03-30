use proc_macro::TokenStream;
use std::collections::HashMap;
use quote::quote;
use syn::{
    {parse::Parse, parse::ParseStream, parse_macro_input},
    Ident, LitInt, Token
};

use crate::helpers;

struct CudaModuleArgs {
    rows: Option<u64>,
    cols: Option<u64>,
}

impl Parse for CudaModuleArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut rows = None;
        let mut cols = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _eq: Token![=] = input.parse()?;
            let val: LitInt = input.parse()?;

            match key.to_string().as_str() {
                "ROWS" => rows = Some(val.base10_parse()?),
                "COLS" => cols = Some(val.base10_parse()?),
                other  => return Err(syn::Error::new(key.span(),
                    format!("unknown cuda_module param `{other}`"))),
            }

            // optional trailing comma
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(CudaModuleArgs { rows, cols })
    }
}

pub fn cuda_module_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = syn::parse::<CudaModuleArgs>(attr).unwrap();

    let module = parse_macro_input!(item as syn::ItemMod);

    let items = match &module.content {
        Some((_, items)) => items.clone(),
        None => return quote! { #module }.into(),
    };

    // collect device functions
    let device_fns: HashMap<String, syn::ItemFn> = items.iter()
        .filter_map(|item| {
            if let syn::Item::Fn(func) = item {
                let has_attr = func.attrs.iter().any(|a| {
                    a.path().segments.last()
                     .map(|s| s.ident == "device_function")
                     .unwrap_or(false)
                });
                if has_attr { return Some((func.sig.ident.to_string(), func.clone())); }
            }
            None
        })
        .collect();

    // collect kernel functions and generate .cu files now, while we have device_fns
    for item in &items {
        if let syn::Item::Fn(func) = item {
            let has_kernel = func.attrs.iter().any(|a| {
                a.path().segments.last()
                 .map(|s| s.ident == "kernel")
                 .unwrap_or(false)
            });
            if has_kernel {
                helpers::gen_kernel(
                    &TokenStream::new(), 
                    func.clone(),
                    device_fns.clone(),
                    args.rows,
                    args.cols,
                );

                let name = func.sig.ident.to_string();
                let ptx_name = format!("generated_{name}.ptx");
                let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
                    .unwrap_or_else(|_| std::env::current_dir().unwrap().to_string_lossy().to_string());
                let ptx_path = std::path::Path::new(&manifest_dir).join(&ptx_name);
                let ptx_path_str = ptx_path.to_string_lossy().to_string();

                let output = std::process::Command::new("nvcc")
                    .args([
                        "-arch=sm_86",
                        "-ptx",
                        &format!("generated_{name}.cu"),
                        "-o",
                        &ptx_path_str,
                    ])
                    .output()
                    .expect("failed to run nvcc — is it on PATH?");

                if !output.stdout.is_empty() {
                    eprintln!("nvcc stdout:\n{}", String::from_utf8_lossy(&output.stdout));
                }
                if !output.stderr.is_empty() {
                    eprintln!("nvcc stderr:\n{}", String::from_utf8_lossy(&output.stderr));
                }
                if !output.status.success() {
                    panic!(
                        "nvcc PTX compilation failed (exit {:?}) for kernel `{name}`",
                        output.status.code()
                    );
                }

                eprintln!("spawn: generated PTX {}", ptx_path.display());
            }
        }
    }

    let mod_name  = &module.ident;
    let mod_vis   = &module.vis;

    quote! {
        #mod_vis mod #mod_name {
            #( #items )*
        }
    }
    .into()
}
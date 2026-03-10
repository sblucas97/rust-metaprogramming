extern crate proc_macro;

use std::sync::Mutex;
use once_cell::sync::Lazy;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;
use std::collections::HashMap;

mod helpers;

static DEVICE_FUNCTIONS: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[proc_macro_attribute]
pub fn cuda_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let module = parse_macro_input!(item as syn::ItemMod);

    let items = match &module.content {
        Some((_, items)) => items.clone(),
        None => return quote! { #module }.into(),
    };

    let device_fns: HashMap<String, String> = items.iter()
        .filter_map(|item| {
            if let syn::Item::Fn(func) = item {
                let has_device_fn = func.attrs.iter().any(|attr| {
                    attr.path().segments.last()
                        .map(|s| s.ident == "device_function")
                        .unwrap_or(false)
                });
                if has_device_fn {
                    // serialize to string
                    return Some((
                        func.sig.ident.to_string(),
                        quote! { #func }.to_string()
                    ));
                }
            }
            None
        })
        .collect();

    *DEVICE_FUNCTIONS.lock().unwrap() = device_fns;

    quote! { #module }.into()
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as syn::ItemFn);

    let device_fns: HashMap<String, syn::ItemFn> = DEVICE_FUNCTIONS.lock().unwrap()
        .iter()
        .filter_map(|(name, src)| {
            let item_fn: syn::ItemFn = syn::parse_str(src).ok()?;
            Some((name.clone(), item_fn))
        })
        .collect();


    helpers::gen_kernel(&attr, input_fn.clone(), device_fns);
    helpers::gen_launcher(&attr, input_fn.clone());

    let sig = &input_fn.sig;
    let function_name_ident = &input_fn.sig.ident;

    let expanded = quote! {
        pub mod #function_name_ident {
            use super::*;
            use lib_core::KernelName;

            #sig {
                let _guard = guard_rt::KernelContextGuard::new();
                panic!("Kernel function '{}' cannot be called directly on CPU. Use spawn::<{}::Marker>() instead.",
                       stringify!(#function_name_ident),
                       stringify!(#function_name_ident));
            }

            pub struct Marker;

            impl KernelName for Marker {
                fn kernel_name() -> &'static str {
                    stringify!(#function_name_ident)
                }
            }
        }
    };

    expanded.into()
}

#[proc_macro_attribute]
pub fn device_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as syn::ItemFn);

    let sig = &input_fn.sig;
    let block = &input_fn.block;
    let vis = &input_fn.vis;
    let attrs = &input_fn.attrs;
    let name = &input_fn.sig.ident;

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            guard_rt::CONTEXT_ACTIVE.with(|flag: &std::cell::Cell<bool>| {
                if !flag.get() {
                    panic!(
                        "`{}` can only be called inside a #[kernel] function",
                        stringify!(#name)
                    );
                }
            });
            #block
        }
    };

    expanded.into()
}
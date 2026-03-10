extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{quote};
use syn::{parse_macro_input};

mod helpers;

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as syn::ItemFn);
    helpers::gen_kernel(&attr, input_fn.clone());
    helpers::gen_launcher(&attr, input_fn.clone());

    let sig = &input_fn.sig;
    let function_name_ident = &input_fn.sig.ident;

    let expanded = quote! {

        pub mod #function_name_ident {
            use super::*;
            
            #sig {
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
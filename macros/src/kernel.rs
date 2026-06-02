use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

pub fn kernel_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as syn::ItemFn);
    let sig = &input_fn.sig;
    let function_name_ident = &input_fn.sig.ident;

    quote! {
        pub mod #function_name_ident {
            use super::*;
            #sig {
                panic!("Kernel '{}' cannot be called directly on CPU.",
                       stringify!(#function_name_ident));
            }
        }
    }
    .into()
}
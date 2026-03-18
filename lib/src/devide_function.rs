use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;


pub fn device_function_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as syn::ItemFn);

    let sig = &input_fn.sig;
    let block = &input_fn.block;
    let vis = &input_fn.vis;
    let attrs = &input_fn.attrs;
    let name = &input_fn.sig.ident;

    // extract return type to generate a default return value
    let default_return = match &input_fn.sig.output {
        syn::ReturnType::Default => quote! {},
        syn::ReturnType::Type(_, ty) => quote! {
            let default: #ty = Default::default();
            default
        },
    };

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
            #default_return
        }
    };

    expanded.into()
}
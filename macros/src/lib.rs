extern crate proc_macro;

use proc_macro::TokenStream;

mod spawn;
mod cuda_module;
mod kernel;
mod devide_function;
mod helpers;

#[proc_macro_attribute]
pub fn cuda_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    cuda_module::cuda_module_impl(_attr, item).into()
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::kernel_impl(attr, item).into()
}

#[proc_macro_attribute]
pub fn device_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    devide_function::device_function_impl(_attr, item).into()
}

#[proc_macro]
pub fn spawn(input: TokenStream) -> TokenStream {
    spawn::spawn_impl(input).into()
}
extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Literal, Group, Ident, Punct, Spacing, Span, Delimiter};
use quote::quote;

mod helpers;

#[proc_macro]
pub fn my_macro(input: TokenStream) -> TokenStream {
    let mut output = TokenStream::new();
    let mut tokens = input.into_iter().peekable();

    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Ident(ident) if ident.to_string() == "xx" => {
                let new_ident = Ident::new("int", ident.span());
                output.extend([TokenTree::Ident(new_ident)]);
            }

            TokenTree::Punct(punct) if punct.as_char() == ':' => {
                let new_punct = Punct::new('=', Spacing::Alone);
                output.extend([TokenTree::Punct(new_punct)]);
            }

            
            TokenTree::Ident(ident) if ident.to_string() == "print" => {
                output.extend([TokenTree::Ident(Ident::new("printf", ident.span()))]);

                if let Some(TokenTree::Group(group)) = tokens.next() {
                    // Just forward the parentheses and their contents
                    let new_group = Group::new(Delimiter::Parenthesis, group.stream());
                    output.extend([TokenTree::Group(new_group)]);
                } else {
                    panic!("Expected group after print");
                }
            }

            _ => {
                output.extend([token]);
            }
        }
    }

    let mut code_str = helpers::get_headers();
    code_str.push_str(&helpers::init_main_function());
    code_str.push_str(&output.to_string());
    code_str.push_str(&helpers::close_main_function());
    code_str = code_str.replace(";", ";\n");

    
    // write dsl to c file
    let quote = quote! {
        use std::fs;
        use std::path::PathBuf;

        let mut output_path = PathBuf::from("runtime_generated.c");
        let code = #code_str;
        fs::write(&output_path, code).expect("Failed to write to output_path from macro");
    };
    

    return quote.into();
}

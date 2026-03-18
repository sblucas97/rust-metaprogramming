use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Path,
    parse::{Parse, ParseStream},
    parse_macro_input,
    Expr, Token, Type,
};

enum ArgKind {
    /// plain expr — CudaVec input, calls .get_device_ptr()
    Input  { value: Expr },
    /// `> expr: T` — host Vec<T>, macro allocates GPU buf, copies back, frees
    Result { value: Expr, inner_ty: Type },
    /// plain `expr: T` — scalar/primitive, passed as-is with explicit type
    Scalar { value: Expr, ty: Type },
}

struct SpawnInput {
    kernel: Path,
    args: Vec<ArgKind>,
}

impl Parse for SpawnInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kernel: Path = input.parse()?;
        let mut args = Vec::new();

        while input.peek(Token![,]) {
            let _comma: Token![,] = input.parse()?;

            // `> expr: T` — result
            if input.peek(Token![>]) {
                let _gt: Token![>]    = input.parse()?;
                let value: Expr       = input.parse()?;
                let _colon: Token![:] = input.parse()?;
                let inner_ty: Type    = input.parse()?;
                args.push(ArgKind::Result { value, inner_ty });
                continue;
            }

            // peek ahead: if `expr :` follows, it's a scalar with explicit type
            // otherwise it's a bare CudaVec input
            let value: Expr = input.parse()?;
            if input.peek(Token![:]) {
                let _colon: Token![:]     = input.parse()?;
                let ty: Type              = input.parse()?;
                args.push(ArgKind::Scalar { value, ty });
            } else {
                args.push(ArgKind::Input { value });
            }
        }


        let result_count = args.iter()
            .filter(|a| matches!(a, ArgKind::Result { .. }))
            .count();

        if result_count != 1 {
            return Err(input.error(format!(
                "spawn! requires exactly one `>` result argument, found {result_count}"
            )));
        }

        Ok(SpawnInput { kernel, args })
    }
}

pub fn spawn_impl(input: TokenStream) -> TokenStream {
    let SpawnInput { kernel, args } = parse_macro_input!(input as SpawnInput);

    let kernel_str = kernel.segments.last()
        .map(|s| s.ident.to_string())
        .unwrap_or_else(|| panic!("invalid kernel path"));

    let mut fn_arg_types  = Vec::new();
    let mut fn_arg_values = Vec::new();
    let mut pre_launch    = Vec::new();
    let mut post_launch   = Vec::new();

    // grab the first input CudaVec for the resize length
    let first_input = args.iter()
        .find_map(|a| match a {
            ArgKind::Input { value } => Some(value),
            _ => None,
        })
        .expect("spawn! requires at least one input CudaVec argument");

    for arg in &args {
        match arg {
            ArgKind::Input { value } => {
                fn_arg_types.push(quote!  { *mut f32 });
                fn_arg_values.push(quote! { #value.get_device_ptr() });
            }

            ArgKind::Result { value, inner_ty } => {
                pre_launch.push(quote! {
                    let mut __result_device_ptr: *mut #inner_ty = std::ptr::null_mut();
                    lib_core::ffi::cuda_allocate(
                        &mut __result_device_ptr as *mut *mut #inner_ty,
                        #first_input.len()
                    );
                });

                fn_arg_types.push(quote!  { *mut #inner_ty });
                fn_arg_values.push(quote! { __result_device_ptr });

                post_launch.push(quote! {
                    #value.resize(#first_input.len(), <#inner_ty>::default());
                    lib_core::ffi::cuda_copy_to_host(
                        #value.as_mut_ptr(), 
                        __result_device_ptr,
                        #value.len()
                    );
                    lib_core::ffi::cuda_free(__result_device_ptr);
                });
            }

            ArgKind::Scalar { value, ty } => {
                fn_arg_types.push(quote!  { #ty });
                fn_arg_values.push(quote! { #value });
            }
        }
    }

    quote! {
        {
            let name = #kernel_str;
            let so_name = format!("kernel_and_launcher_generated_for_{name}.so");

            let output = std::process::Command::new("nvcc")
                .args([
                    "-arch=sm_86",
                    &format!("generated_{name}.cu"),
                    &format!("generated_{name}_launcher.cu"),
                    "-o", &so_name,
                    "--shared",
                    "-Xcompiler", "-fPIC",
                    "-x", "cu",
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
                    "nvcc compilation failed (exit {:?}) for kernel `{name}`",
                    output.status.code()
                );
            }

            // allocate result device buffer
            #( #pre_launch )*

            // load & launch
            let so_path = std::env::current_dir().unwrap().join(&so_name);
            eprintln!("spawn: loading {}", so_path.display());

            unsafe {
                type LaunchFn = unsafe extern "C" fn( #( #fn_arg_types ),* );

                let lib = libloading::Library::new(&so_path)
                    .unwrap_or_else(|e| panic!("failed to load `{}`: {e}", so_path.display()));

                let symbol_name = format!("launch_generated_{name}\0");
                let launch: libloading::Symbol<LaunchFn> = lib
                    .get(symbol_name.as_bytes())
                    .unwrap_or_else(|e| panic!("symbol `launch_generated_{name}` not found: {e}"));

                launch( #( #fn_arg_values ),* );
            }

            // copy result back to host + free device buffer
            #( #post_launch )*
        }
    }
    .into()
}
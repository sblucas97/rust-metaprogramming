use std::collections::HashMap;

use crate::{
    codegen, context::Context, diagnostics::{DiagKind, Diagnostic}, lower, type_checker, types::TypeError
};

pub struct CompiledKernel {
    pub cuda: String,
    pub name: String,
}

pub fn compile_kernel(
    item: &syn::ItemFn,
    device_fns: HashMap<String, syn::ItemFn>,
    rows: Option<u64>,
    cols: Option<u64>
) -> Result<CompiledKernel, Vec<Diagnostic>> {
    let lower = lower::lower_fn(item).unwrap();
    dbg_stage("1 lower", &lower);

    let mut ctx = Context::new();
    // lower should have in the output the param name and types
    // having this it should traverse adding the items to the ctx and them check the body
    // right now its checking just the body
    type_checker::type_check(&lower, &mut ctx).map_err(|e| vec![Diagnostic {
        msg: type_error_message(e),
        span: None,
        kind: DiagKind::Type,
    }])?;

    let fn_name = item.sig.ident.to_string();
    let cuda = codegen::gen_kernel(item.clone(), fn_name.clone(), device_fns, rows, cols);
    dbg_stage("3 codegen", &cuda);
    Ok(CompiledKernel { cuda, name: fn_name })
}

fn type_error_message(e: TypeError) -> String {
    match e {
        TypeError::UnknownVariable(name) => format!("unknown variable `{name}`"),
        TypeError::TypeMismatch { expected, found } => {
            format!("type mismatch: expected `{expected}`, found `{found}`")
        },
        TypeError::InvalidAssignmentTarget => "invalid assignment target".into(),
        TypeError::InvalidIndexing => "invalid indexing: expected CudaVec<T>[u64]".into(),
        TypeError::InvalidCudaVecSize => "CudaVec size must be u64".into(),
        TypeError::InvalidFieldProperty => "invalid field access".into(),
    }
}

fn dbg_stage(name: &str, val: &impl std::fmt::Debug) {
    if std::env::var("DEBUG_ACTIVE").is_ok() {
        eprintln!("\n=== [{name}] ===\n{val:#?}");
    }
}
use syn::{BinOp, Expr as SynExpr, Field, ItemFn, Pat, Stmt as SynStmt};
use quote::{ToTokens, quote};

use crate::{
    types::{Type},
    ast::{Expr, Function, Stmt, Param}
};

pub fn lower_fn(item: &ItemFn) -> Result<Function, String> {
    let name = item.sig.ident.to_string();

    let mut params = Vec::new();
    for arg in &item.sig.inputs {
        params.push(lower_param(arg)?);
    }

    let mut body = Vec::new();
    for stmt in &item.block.stmts {
        body.push(lower_stmt(stmt)?);
    }

    Ok(Function { name, params, body })
}

fn lower_param(arg: &syn::FnArg) -> Result<Param, String> {
    match arg {
        syn::FnArg::Typed(pat_type) => {
            let name = match &*pat_type.pat {
                syn::Pat::Ident(id) => id.ident.to_string(),
                _ => return Err("unsupported param pattern".into()),
            };

            let ty = lower_type(&pat_type.ty)?;

            Ok(Param { name, ty })
        }

        syn::FnArg::Receiver(_) => {
            Err("self receiver not supported".into())
        }
    }
}

fn lower_type(ty: &syn::Type) -> Result<Type, String> {
    match ty {
        syn::Type::Reference(type_ref) => {
            let mutable = type_ref.mutability.is_some();

            Ok(Type::Ref {
                mutable,
                inner: Box::new(lower_type(&type_ref.elem)?),
            })
        }

        syn::Type::Path(type_path) => {
            let segment = type_path
                .path
                .segments
                .last()
                .ok_or("missing type segment")?;

            match segment.ident.to_string().as_str() {
                "f32" => Ok(Type::F32),
                "u64" => Ok(Type::U64),
                "u32" => Ok(Type::U32),
                "CudaVec" => {
                    match &segment.arguments {
                        syn::PathArguments::AngleBracketed(args) => {
                            let first = args.args.first()
                                .ok_or("CudaVec missing generic")?;

                            match first {
                                syn::GenericArgument::Type(inner) => {
                                    Ok(Type::CudaVec(
                                        Box::new(lower_type(inner)?)
                                    ))
                                }
                                _ => Err("unsupported CudaVec generic".into()),
                            }
                        }

                        _ => Err("CudaVec requires generic".into()),
                    }
                }

                other => Err(format!("unsupported type: {}", other)),
            }
        }

        _ => Err("unsupported type".into()),
    }
}

fn lower_stmt(stmt: &SynStmt) -> Result<Stmt, String> {
    match stmt {
        SynStmt::Local(local) => {
            let name = match &local.pat {
                Pat::Type(pat_type) => match &*pat_type.pat {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    _ => return Err("unsupported pattern".into()),
                },
                _ => return Err("unsupported pattern".into()),
            };

            let init = local.init.as_ref()
                .ok_or("missing initializer")?;

            let value = lower_expr(&init.expr)?;

            Ok(Stmt::Let { name, value })
        }

        SynStmt::Expr(expr, _) => {
            Ok(Stmt::Expr(lower_expr(expr)?))
        }

        _ => Err("unsupported statement".into()),
    }
}

fn lower_expr(expr: &SynExpr) -> Result<Expr, String> {
    match expr {
        SynExpr::Paren(p) => lower_expr(&p.expr),

        SynExpr::Lit(lit) => match &lit.lit {
            syn::Lit::Float(f) => Ok(Expr::LiteralF32(f.base10_parse().unwrap())),
            syn::Lit::Int(i) => Ok(Expr::LiteralU64(i.base10_parse().unwrap())),
            _ => Err("unsupported literal".into()),
        },

        SynExpr::Path(p) => {
            let ident = p.path.segments.last().unwrap().ident.to_string();
            Ok(Expr::Var(ident))
        }

        SynExpr::Binary(b) => match &b.op {
            BinOp::Add(_) => Ok(Expr::Add(
                Box::new(lower_expr(&b.left)?),
                Box::new(lower_expr(&b.right)?),
            )),
            BinOp::Mul(_) => Ok(Expr::Mul(
                Box::new(lower_expr(&b.left)?),
                Box::new(lower_expr(&b.right)?)
            )), 
            _ => 
            {
                let tokens = quote! { #b };
                println!("\n#####\n#####\n{}\n######\n######\n", tokens.to_string());
                Err("unsupported binary op".into())
            },
        },

        SynExpr::Assign(a) => {
            Ok(Expr::Assign {
                target: Box::new(lower_expr(&a.left)?),
                value: Box::new(lower_expr(&a.right)?),
            })
        },

        SynExpr::Index(i) => Ok(Expr::Index {
            target: Box::new(lower_expr(&i.expr)?),
            index: Box::new(lower_expr(&i.index)?),
            }
        ),
        
        SynExpr::Field(f) => {
            let member = match &f.member {
                syn::Member::Named(ident) => ident.to_string(),
                _ => panic!("Unnamed member not supported."),
            };

            Ok(Expr::Field { base: Box::new(lower_expr(&f.base)?), member: member })
        }

        e => {

            let tokens = quote! { #e };
            println!("\n#####\n#####\n{}\n######\n######\n", tokens.to_string());
            Err(format!("unsupported expression: {} ", expr.to_token_stream()))
        },
    }
}

// #[cfg(test)]
// mod tests {
//     use syn::{Expr as SynExpr};

//     use crate::{    
//         ast::Expr,
//         lower_expr::lower_expr
//     };

//     #[test]
//     fn should_map_literal_float() {
//         SynExpr::ExprLit::
//     }

// }
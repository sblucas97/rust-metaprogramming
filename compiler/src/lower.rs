use syn::{BinOp, Expr as SynExpr, ItemFn, Pat, Stmt as SynStmt};
use quote::{ToTokens, quote};

use crate::{
    ast::{Expr, ExprKind, Function, NodeId, Param, Stmt}, types::Type
};

pub fn lower_fn(item: &ItemFn) -> Result<Function, String> {
    Lowerer::new().lower_fn(item)
}

struct Lowerer {
    next_id: u32
}

impl Lowerer {
    fn new() -> Self {
        Self { next_id: 0 }
    }

    fn fresh(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    fn mk(&mut self, kind: ExprKind) -> Expr {
        Expr { id: self.fresh(), kind }
    }

    pub fn lower_fn(&mut self, item: &ItemFn) -> Result<Function, String> {
        let name = item.sig.ident.to_string();

        let mut params = Vec::new();
        for arg in &item.sig.inputs {
            params.push(self.lower_param(arg)?);
        }

        let mut body = Vec::new();
        for stmt in &item.block.stmts {
            body.push(self.lower_stmt(stmt)?);
        }

        Ok(Function { name, params, body })
    }

    fn lower_param(&mut self, arg: &syn::FnArg) -> Result<Param, String> {
        match arg {
            syn::FnArg::Typed(pat_type) => {
                let name = match &*pat_type.pat {
                    syn::Pat::Ident(id) => id.ident.to_string(),
                    _ => return Err("unsupported param pattern".into()),
                };

                let ty = self.lower_type(&pat_type.ty)?;

                Ok(Param { name, ty })
            }

            syn::FnArg::Receiver(_) => {
                Err("self receiver not supported".into())
            }
        }
    }

    fn lower_type(&mut self, ty: &syn::Type) -> Result<Type, String> {
        match ty {
            syn::Type::Reference(type_ref) => {
                let mutable = type_ref.mutability.is_some();

                Ok(Type::Ref {
                    mutable,
                    inner: Box::new(self.lower_type(&type_ref.elem)?),
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
                                            Box::new(self.lower_type(inner)?)
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

    fn lower_stmt(&mut self, stmt: &SynStmt) -> Result<Stmt, String> {
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

                let value = self.lower_expr(&init.expr)?;

                Ok(Stmt::Let { name, value })
            }

            SynStmt::Expr(expr, _) => {
                Ok(Stmt::Expr(self.lower_expr(expr)?))
            }

            _ => Err("unsupported statement".into()),
        }
    }

    fn lower_expr(&mut self, expr: &SynExpr) -> Result<Expr, String> {
        let kind: ExprKind = match expr {
            SynExpr::Paren(p) => return self.lower_expr(&p.expr),

            SynExpr::Lit(lit) => match &lit.lit {
                syn::Lit::Float(f) => ExprKind::LiteralF32(f.base10_parse().unwrap()),
                syn::Lit::Int(i) => ExprKind::LiteralU64(i.base10_parse().unwrap()),
                _ => return Err("unsupported literal".into()),
            },

            SynExpr::Path(p) => {
                let ident = p.path.segments.last().unwrap().ident.to_string();
                ExprKind::Var(ident)
            },

            SynExpr::Binary(b) => match &b.op {
                BinOp::Add(_) => ExprKind::Add(
                    Box::new(self.lower_expr(&b.left)?),
                    Box::new(self.lower_expr(&b.right)?),
                ),
                BinOp::Mul(_) => ExprKind::Mul(
                    Box::new(self.lower_expr(&b.left)?),
                    Box::new(self.lower_expr(&b.right)?)
                ), 
                _ => 
                {
                    let tokens = quote! { #b };
                    println!("\n#####\n#####\n{}\n######\n######\n", tokens.to_string());
                    return Err("unsupported binary op".into())
                },
            },

            SynExpr::Assign(a) => {
                ExprKind::Assign {
                    target: Box::new(self.lower_expr(&a.left)?),
                    value: Box::new(self.lower_expr(&a.right)?),
                }
            },

            SynExpr::Index(i) => ExprKind::Index {
                target: Box::new(self.lower_expr(&i.expr)?),
                index: Box::new(self.lower_expr(&i.index)?),
            },
            
            SynExpr::Field(f) => {
                let member = match &f.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    _ => panic!("Unnamed member not supported."),
                };

                ExprKind::Field { base: Box::new(self.lower_expr(&f.base)?), member: member }
            }

            e => {

                let tokens = quote! { #e };
                println!("\n#####\n#####\n{}\n######\n######\n", tokens.to_string());
                return Err(format!("unsupported expression: {} ", expr.to_token_stream()))
            },
        };

        Ok(self.mk(kind))
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
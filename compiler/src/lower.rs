use syn::{BinOp as SynBinOp, Expr as SynExpr, ItemFn, Pat, Stmt as SynStmt, UnOp as SynUnOp};
use quote::ToTokens;

use crate::{
    ast::{BinOp, Expr, ExprKind, Function, NodeId, Param, Stmt, UnOp},
    types::Type,
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

        let body = self.lower_block(&item.block.stmts)?;

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
                    "bool" => Ok(Type::Bool),
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

    fn lower_block(&mut self, stmts: &[SynStmt]) -> Result<Vec<Stmt>, String> {
        stmts.iter().map(|stmt| self.lower_stmt(stmt)).collect()
    }

    fn lower_stmt(&mut self, stmt: &SynStmt) -> Result<Stmt, String> {
        match stmt {
            SynStmt::Local(local) => {
                let (name, mutable, ty) = match &local.pat {
                    Pat::Type(pat_type) => {
                        let (name, mutable) = match &*pat_type.pat {
                            Pat::Ident(ident) => {
                                (ident.ident.to_string(), ident.mutability.is_some())
                            }
                            _ => return Err("unsupported let pattern".into()),
                        };
                        (name, mutable, self.lower_type(&pat_type.ty)?)
                    }
                    Pat::Ident(ident) => {
                        return Err(format!(
                            "let `{}` requires an explicit type annotation",
                            ident.ident
                        ));
                    }
                    _ => return Err("unsupported let pattern".into()),
                };

                let init = local.init.as_ref()
                    .ok_or_else(|| format!("let `{name}` is missing an initializer"))?;

                let value = self.lower_expr(&init.expr)?;

                Ok(Stmt::Let { name, ty, mutable, value })
            }

            SynStmt::Expr(SynExpr::If(expr_if), _) => self.lower_if(expr_if),

            SynStmt::Expr(SynExpr::ForLoop(for_loop), _) => self.lower_for(for_loop),

            SynStmt::Expr(expr, _) => {
                Ok(Stmt::Expr(self.lower_expr(expr)?))
            }

            other => Err(format!(
                "unsupported statement: {}",
                other.to_token_stream()
            )),
        }
    }

    fn lower_if(&mut self, expr_if: &syn::ExprIf) -> Result<Stmt, String> {
        let cond = self.lower_expr(&expr_if.cond)?;
        let then_block = self.lower_block(&expr_if.then_branch.stmts)?;

        let else_block = match &expr_if.else_branch {
            None => None,
            Some((_, else_expr)) => match else_expr.as_ref() {
                SynExpr::Block(block) => Some(self.lower_block(&block.block.stmts)?),
                SynExpr::If(nested) => Some(vec![self.lower_if(nested)?]),
                other => {
                    return Err(format!(
                        "unsupported else branch: {}",
                        other.to_token_stream()
                    ));
                }
            },
        };

        Ok(Stmt::If { cond, then_block, else_block })
    }

    // Only `for var in (start, end).step_by(step) { ... }` is supported.
    fn lower_for(&mut self, for_loop: &syn::ExprForLoop) -> Result<Stmt, String> {
        let var = match &*for_loop.pat {
            Pat::Ident(ident) => ident.ident.to_string(),
            _ => return Err("unsupported for-loop pattern".into()),
        };

        let method_call = match &*for_loop.expr {
            SynExpr::MethodCall(mc) if mc.method == "step_by" => mc,
            other => {
                return Err(format!(
                    "unsupported for-loop range (expected `(start, end).step_by(step)`): {}",
                    other.to_token_stream()
                ));
            }
        };

        let (start, end) = match &*method_call.receiver {
            SynExpr::Tuple(tuple) if tuple.elems.len() == 2 => (
                self.lower_expr(&tuple.elems[0])?,
                self.lower_expr(&tuple.elems[1])?,
            ),
            other => {
                return Err(format!(
                    "unsupported for-loop bounds (expected `(start, end)` tuple): {}",
                    other.to_token_stream()
                ));
            }
        };

        let step = match method_call.args.first() {
            Some(step_expr) if method_call.args.len() == 1 => self.lower_expr(step_expr)?,
            _ => return Err("step_by expects exactly one argument".into()),
        };

        let body = self.lower_block(&for_loop.body.stmts)?;

        Ok(Stmt::For { var, start, end, step, body })
    }

    fn lower_expr(&mut self, expr: &SynExpr) -> Result<Expr, String> {
        let kind: ExprKind = match expr {
            SynExpr::Paren(p) => return self.lower_expr(&p.expr),

            SynExpr::Lit(lit) => match &lit.lit {
                syn::Lit::Float(f) => {
                    ExprKind::LiteralF32(f.base10_parse().map_err(|e| e.to_string())?)
                }
                syn::Lit::Int(i) => {
                    let value: u64 = i.base10_parse().map_err(|e| e.to_string())?;
                    match i.suffix() {
                        "" => ExprKind::LiteralInt(value),
                        "u64" => ExprKind::LiteralTypedInt(value, Type::U64),
                        "u32" => ExprKind::LiteralTypedInt(value, Type::U32),
                        other => {
                            return Err(format!("unsupported integer suffix: {other}"));
                        }
                    }
                }
                _ => return Err("unsupported literal".into()),
            },

            SynExpr::Path(p) => {
                let ident = p.path.segments.last()
                    .ok_or("empty path expression")?
                    .ident
                    .to_string();
                ExprKind::Var(ident)
            },

            SynExpr::Binary(b) => {
                let op = match &b.op {
                    SynBinOp::Add(_) => BinOp::Add,
                    SynBinOp::Sub(_) => BinOp::Sub,
                    SynBinOp::Mul(_) => BinOp::Mul,
                    SynBinOp::Div(_) => BinOp::Div,
                    SynBinOp::Lt(_) => BinOp::Lt,
                    SynBinOp::Le(_) => BinOp::Le,
                    SynBinOp::Gt(_) => BinOp::Gt,
                    SynBinOp::Ge(_) => BinOp::Ge,
                    SynBinOp::Eq(_) => BinOp::Eq,
                    SynBinOp::Ne(_) => BinOp::Ne,
                    SynBinOp::And(_) => BinOp::And,
                    SynBinOp::Or(_) => BinOp::Or,
                    other => {
                        return Err(format!(
                            "unsupported binary op: {}",
                            other.to_token_stream()
                        ));
                    }
                };
                ExprKind::Binary {
                    op,
                    lhs: Box::new(self.lower_expr(&b.left)?),
                    rhs: Box::new(self.lower_expr(&b.right)?),
                }
            },

            SynExpr::Unary(u) => match &u.op {
                SynUnOp::Neg(_) => ExprKind::Unary {
                    op: UnOp::Neg,
                    expr: Box::new(self.lower_expr(&u.expr)?),
                },
                other => {
                    return Err(format!(
                        "unsupported unary op: {}",
                        other.to_token_stream()
                    ));
                }
            },

            SynExpr::Cast(c) => ExprKind::Cast {
                expr: Box::new(self.lower_expr(&c.expr)?),
                ty: self.lower_type(&c.ty)?,
            },

            SynExpr::Call(call) => {
                let func = match &*call.func {
                    SynExpr::Path(p) => p.path.segments.last()
                        .ok_or("empty call path")?
                        .ident
                        .to_string(),
                    other => {
                        return Err(format!(
                            "unsupported call target: {}",
                            other.to_token_stream()
                        ));
                    }
                };

                let mut args = Vec::new();
                for arg in &call.args {
                    args.push(self.lower_expr(arg)?);
                }

                ExprKind::Call { func, args }
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
                    _ => return Err("unnamed field access not supported".into()),
                };

                ExprKind::Field { base: Box::new(self.lower_expr(&f.base)?), member }
            }

            e => {
                return Err(format!(
                    "unsupported expression: {}",
                    e.to_token_stream()
                ));
            },
        };

        Ok(self.mk(kind))
    }
}

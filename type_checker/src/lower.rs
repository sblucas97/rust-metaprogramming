use syn::{ItemFn, Stmt as SynStmt, Expr as SynExpr, BinOp, Pat};
use quote::ToTokens;

use crate::{
    ast::{Expr, Function, Stmt}
};

pub fn lower_fn(item: &ItemFn) -> Result<Function, String> {
    let name = item.sig.ident.to_string();

    let mut body = Vec::new();

    for stmt in &item.block.stmts {
        body.push(lower_stmt(stmt)?);
    }

    Ok(Function { name, body })
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
            _ => Err("unsupported binary op".into()),
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

        _ => Err(format!("unsupported expression: {}", expr.to_token_stream())),
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
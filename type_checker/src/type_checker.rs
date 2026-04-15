use crate:: {
    ast::{Expr, Function, Stmt},
    context::Context,
    types::{Type, TypeError}
};

pub fn type_check(func: &Function, ctx: &mut Context) -> Result<Type, TypeError> {
    let mut last_ty = Type::Unit;
    for stmt in &func.body {
        last_ty = type_check_stmt(stmt, ctx)?;
    }
    Ok(last_ty)
}

fn type_check_stmt(stmt: &Stmt, ctx: &mut Context) -> Result<Type, TypeError> {
    match stmt {
        Stmt::Let { name, value } => {
            let value_type = type_check_expr(value, ctx)?;
            ctx.insert(name.clone(), value_type);
            Ok(Type::Unit)
        }
        Stmt::Expr(expr) => type_check_expr(expr, ctx),
    }
}

fn type_check_expr(expr: &Expr, ctx: &mut Context) -> Result<Type, TypeError> {
    match expr {
        Expr::LiteralF32(_) => {
            return Ok(Type::F32)
        }

        Expr::LiteralU64(_) => {
            return Ok(Type::U64)
        }

        Expr::Var(name) => {
            return ctx.get(name)
                .cloned()
                .ok_or(TypeError::UnknownVariable(name.clone()))
        }

        Expr::Add(left, right) => {
            let ty1 = type_check_expr(left, ctx)?;
            let ty2 = type_check_expr(right, ctx)?;

            match (&ty1, &ty2) {
                (Type::F32, Type::F32) => {
                    return Ok(Type::F32)
                },
                (Type::U64, Type::U64) => {
                    return Ok(Type::U64)
                },
                _ => {
                    return Err(TypeError::TypeMismatch { 
                        expected: format!("{:?} + {:?}", ty1.clone(), ty1.clone()), 
                        found:  format!("{:?} + {:?}", ty1.clone(), ty2.clone())
                    })
                }
            }
        }

        /**
         *  Γ ⊢ size_expr : U64
            ---------------------------
            Γ ⊢ CudaVec(size_expr) : CudaVec<F32>
         */
        Expr::CudaVec(size_expr) => {
            let ty = type_check_expr(size_expr, ctx)?;

            match ty {
                Type::U64 => Ok(Type::CudaVec(Box::new(Type::F32))),
                _ => Err(TypeError::InvalidCudaVecSize)
            }
        }

        /**
         *  Γ ⊢ target : CudaVec<T>
            Γ ⊢ index  : U64
            ---------------------------
            Γ ⊢ target[index] : T
         */
        Expr::Index {target, index } => {
            let ty_target = type_check_expr(target, ctx)?;
            let ty_index = type_check_expr(index, ctx)?;

            match(ty_target, ty_index) {
                (Type::CudaVec(inner), Type::U64) => Ok(*inner),
                _ => Err(TypeError::InvalidIndexing)
            }
        }

        Expr::Assign { target, value} => {
            let var_name = match &**target {
                Expr::Var(name) => name,
                // only {x = ....} valid for now
                // others like x[i] = a[i] * 2 not yet
                _ => return Err(TypeError::InvalidAssignmentTarget)
            };

            let value_type = type_check_expr(value, ctx)?;

            match ctx.get(var_name) {
                // If var already exists, check types are equal
                Some(existing_type) => {
                    if *existing_type == value_type {
                        return Ok(Type::Unit);
                    } else {
                        
                        return Err(TypeError::TypeMismatch { 
                            expected: format!("{:?}", existing_type), 
                            found: format!("{:?}", value_type) 
                        });
                    }
                }
                None => {
                    // Allowing implicit declaration
                    ctx.insert(var_name.clone(), value_type);
                    return Ok(Type::Unit);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Expr, Function, Stmt},
        context::Context,
        type_checker::type_check,
        types::{Type, TypeError}
    };

    fn func(stmts: Vec<Stmt>) -> Function {
        Function { name: "test".into(), body: stmts }
    }

    // CudaVec(10)
    #[test]
    fn should_create_cudavec_from_size() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::CudaVec(Box::new(Expr::LiteralU64(10))))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(
            result,
            Ok(Type::CudaVec(Box::new(Type::F32)))
        )
    }

    // CudaVec(10.0)
    #[test]
    fn should_fail_cudavec_with_non_integer_size() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::CudaVec(Box::new(Expr::LiteralF32(10.0))))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(
            result,
            Err(TypeError::InvalidCudaVecSize)
        )
    }

    // let x = CudaVec(10);
    #[test]
    fn should_assign_cuda_vec() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Let {
            name: "x".into(),
            value: Expr::CudaVec(Box::new(Expr::LiteralU64(10))),
        }]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn should_assign_new_variable() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::Assign {
            target: Box::new(Expr::Var("x".into())),
            value: Box::new(Expr::LiteralF32(1.0)),
        })]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn should_fail_assignment_type_mismatch() {
        let mut ctx = Context::new();
        ctx.insert("x", Type::F32);

        let f = func(vec![Stmt::Expr(Expr::Assign {
            target: Box::new(Expr::Var("x".into())),
            value: Box::new(Expr::LiteralU64(10)),
        })]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(
            result, 
            Err(TypeError::TypeMismatch { expected: "F32".into(), found: "U64".into() })
        )
    }

    #[test]
    fn should_fail_unknown_variable() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::Var("y".into()))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(result, Err(TypeError::UnknownVariable("y".into())));
    }

    #[test]
    fn should_pass_known_variable() {
        let mut ctx = Context::new();
        ctx.insert("y", Type::F32);

        let f = func(vec![Stmt::Expr(Expr::Var("y".into()))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(result, Ok(Type::F32));
    }

    #[test]
    fn should_add_two_f32() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::Add(
            Box::new(Expr::LiteralF32(1.0)),
            Box::new(Expr::LiteralF32(2.0)),
        ))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(result, Ok(Type::F32));
    }

    #[test]
    fn should_add_two_u64() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::Add(
            Box::new(Expr::LiteralU64(1)),
            Box::new(Expr::LiteralU64(2)),
        ))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(result, Ok(Type::U64));
    }

    #[test]
    fn should_fail_add_f32_with_u64() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::Add(
            Box::new(Expr::LiteralF32(1.0)),
            Box::new(Expr::LiteralU64(2)),
        ))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(
            result, 
            Err(TypeError::TypeMismatch { expected: "F32 + F32".into(), found: "F32 + U64".into() })
        )
    }

    #[test]
    fn should_fail_add_u64_with_f32() {
        let mut ctx = Context::new();

        let f = func(vec![Stmt::Expr(Expr::Add(
            Box::new(Expr::LiteralU64(1)),
            Box::new(Expr::LiteralF32(2.0)),
        ))]);

        let result = type_check(&f, &mut ctx);

        assert_eq!(
            result, 
            Err(TypeError::TypeMismatch { expected: "U64 + U64".into(), found: "U64 + F32".into() })
        )
    }
}
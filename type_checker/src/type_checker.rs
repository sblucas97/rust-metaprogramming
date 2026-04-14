use crate:: {
    ast::Expr,
    context::Context,
    types::{Type, TypeError}
};

pub fn type_check(expr: &Expr, ctx: &mut Context) -> Result<Type, TypeError> {
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
            let ty1 = type_check(left, ctx)?;
            let ty2 = type_check(right, ctx)?;

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

        Expr::Assing { target, value} => {
            let var_name = match &**target {
                Expr::Var(name) => name,
                // only {x = ....} valid for now
                // others like x[i] = a[i] * 2 not yet
                _ => return Err(TypeError::InvalidAssignmentTarget)
            };

            let value_type = type_check(value, ctx)?;

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
        ast::Expr,
        context::Context,
        type_checker::type_check,
        types::{Type, TypeError}
    };
    
    #[test]
    fn should_assing_new_variable() {
        let mut ctx = Context::new();

        let expr = Expr::Assing {
            target: Box::new(Expr::Var("x".into())),
            value: Box::new(Expr::LiteralF32(1.0)),
        };

        let result = type_check(&expr, &mut ctx);

        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn should_fail_assignment_type_mismatch() {
        let mut ctx = Context::new();
        ctx.insert("x", Type::F32);

        let expr = Expr::Assing {
            target: Box::new(Expr::Var("x".into())),
            value: Box::new(Expr::LiteralU64(10)),
        };

        let result = type_check(&expr, &mut ctx);

        assert_eq!(
            result, 
            Err(TypeError::TypeMismatch { expected: "F32".into(), found: "U64".into() })
        )
    }

    #[test]
    fn should_fail_unknown_variable() {
        let mut ctx = Context::new();

        let expr = Expr::Var("y".into());

        let result = type_check(&expr, &mut ctx);

        assert_eq!(result, Err(TypeError::UnknownVariable("y".into())));
    }

    #[test]
    fn should_pass_known_variable() {
        let mut ctx = Context::new();
        ctx.insert("y", Type::F32);

        let expr = Expr::Var("y".into());

        let result = type_check(&expr, &mut ctx);

        assert_eq!(result, Ok(Type::F32));
    }

    #[test]
    fn should_add_two_f32() {
        let mut ctx = Context::new();
        
        let expr = Expr::Add(
            Box::new(Expr::LiteralF32(1.0)),
            Box::new(Expr::LiteralF32(2.0)),
        );

        let result = type_check(&expr, &mut ctx);

        assert_eq!(result, Ok(Type::F32));
    }

    #[test]
    fn should_add_two_u64() {
        let mut ctx = Context::new();
        
        let expr = Expr::Add(
            Box::new(Expr::LiteralU64(1)),
            Box::new(Expr::LiteralU64(2)),
        );

        let result = type_check(&expr, &mut ctx);

        assert_eq!(result, Ok(Type::U64));
    }

    #[test]
    fn should_fail_add_f32_with_u64() {
        let mut ctx = Context::new();
        
        let expr = Expr::Add(
            Box::new(Expr::LiteralF32(1.0)),
            Box::new(Expr::LiteralU64(2)),
        );

        let result = type_check(&expr, &mut ctx);

        assert_eq!(
            result, 
            Err(TypeError::TypeMismatch { expected: "F32 + F32".into(), found: "F32 + U64".into() })
        )
    }

    #[test]
    fn should_fail_add_u64_with_f32() {
        let mut ctx = Context::new();
        
        let expr = Expr::Add(
            Box::new(Expr::LiteralU64(1)),
            Box::new(Expr::LiteralF32(2.0)),
        );

        let result = type_check(&expr, &mut ctx);

        assert_eq!(
            result, 
            Err(TypeError::TypeMismatch { expected: "U64 + U64".into(), found: "U64 + F32".into() })
        )
    }
}
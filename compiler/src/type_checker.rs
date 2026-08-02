use crate::{
    ast::{Expr, ExprKind, Function, Stmt, UnOp},
    context::Context,
    types::{Type, TypeError}
};

const BUILTIN_FUNCTIONS: &[(&str, &[Type], Type)] = &[
    ("sqrtf", &[Type::F32], Type::F32),
    ("cosf", &[Type::F32], Type::F32),
    ("sinf", &[Type::F32], Type::F32),
    ("floorf", &[Type::F32], Type::F32),
];

pub fn type_check(func: &Function, ctx: &mut Context) -> Result<Type, TypeError> {
    // Params are immutable bindings; mutability of pointees lives in Ref types.
    for param in &func.params {
        ctx.insert(param.name.clone(), param.ty.clone(), false);
    }

    type_check_block(&func.body, ctx)
}

fn type_check_block(stmts: &[Stmt], ctx: &mut Context) -> Result<Type, TypeError> {
    let mut last_ty = Type::Unit;
    for stmt in stmts {
        last_ty = type_check_stmt(stmt, ctx)?;
    }
    Ok(last_ty)
}

fn type_check_stmt(stmt: &Stmt, ctx: &mut Context) -> Result<Type, TypeError> {
    match stmt {
        Stmt::Let { name, ty, mutable, value } => {
            let value_type = type_check_expr(value, ctx, Some(ty))?;
            if !coerces_to(&value_type, ty) {
                return Err(TypeError::LetTypeMismatch {
                    name: name.clone(),
                    expected: format!("{ty:?}"),
                    found: format!("{value_type:?}"),
                });
            }
            ctx.insert(name.clone(), ty.clone(), *mutable);
            Ok(Type::Unit)
        }

        Stmt::If { cond, then_block, else_block } => {
            let cond_ty = type_check_expr(cond, ctx, Some(&Type::Bool))?;
            if cond_ty != Type::Bool {
                return Err(TypeError::ConditionNotBool(format!("{cond_ty:?}")));
            }

            ctx.push_scope();
            let then_result = type_check_block(then_block, ctx);
            ctx.pop_scope();
            then_result?;

            if let Some(else_block) = else_block {
                ctx.push_scope();
                let else_result = type_check_block(else_block, ctx);
                ctx.pop_scope();
                else_result?;
            }

            Ok(Type::Unit)
        }

        Stmt::For { var, start, end, step, body } => {
            for bound in [start, end, step] {
                let ty = type_check_expr(bound, ctx, Some(&Type::U64))?;
                if !coerces_to(&ty, &Type::U64) {
                    return Err(TypeError::TypeMismatch {
                        expected: "U64 loop bound".into(),
                        found: format!("{ty:?}"),
                    });
                }
            }

            ctx.push_scope();
            ctx.insert(var.clone(), Type::U64, false);
            let body_result = type_check_block(body, ctx);
            ctx.pop_scope();
            body_result?;

            Ok(Type::Unit)
        }

        Stmt::Expr(expr) => type_check_expr(expr, ctx, None),
    }
}

fn type_check_expr(
    expr: &Expr,
    ctx: &mut Context,
    expected: Option<&Type>,
) -> Result<Type, TypeError> {
    match &expr.kind {
        ExprKind::LiteralF32(_) => Ok(Type::F32),

        // Γ ⊢ n : τ   if τ is the expected integer type, defaulting to U64
        ExprKind::LiteralInt(_) => match expected {
            Some(ty) if ty.is_integer() => Ok(ty.clone()),
            _ => Ok(Type::U64),
        },

        ExprKind::LiteralTypedInt(_, ty) => Ok(ty.clone()),

        ExprKind::Var(name) => {
            if let Some(binding) = ctx.get(name) {
                return Ok(binding.ty.clone());
            }

            if let Some(ty) = builtin_type(name) {
                return Ok(ty);
            }

            Err(TypeError::UnknownVariable(name.clone()))
        }

        // Arithmetic:  Γ ⊢ lhs : τ   Γ ⊢ rhs : τ   τ numeric  ⟹  lhs op rhs : τ
        // Comparison:  Γ ⊢ lhs : τ   Γ ⊢ rhs : τ   τ numeric  ⟹  lhs op rhs : Bool
        // Logical:     Γ ⊢ lhs : Bool   Γ ⊢ rhs : Bool        ⟹  lhs op rhs : Bool
        // U32 widens to U64 on either side.
        ExprKind::Binary { op, lhs, rhs } => {
            if op.is_logical() {
                for side in [lhs, rhs] {
                    let ty = type_check_expr(side, ctx, Some(&Type::Bool))?;
                    if ty != Type::Bool {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("Bool {op} Bool"),
                            found: format!("{ty:?}"),
                        });
                    }
                }
                return Ok(Type::Bool);
            }

            let operand_expected = if op.is_arithmetic() { expected } else { None };
            let lhs_ty = type_check_expr(lhs, ctx, operand_expected)?;
            let rhs_ty = type_check_expr(rhs, ctx, Some(&lhs_ty))?;

            let unified = unify_numeric(&lhs_ty, &rhs_ty).ok_or_else(|| {
                TypeError::TypeMismatch {
                    expected: format!("{lhs_ty:?} {op} {lhs_ty:?}"),
                    found: format!("{lhs_ty:?} {op} {rhs_ty:?}"),
                }
            })?;

            if op.is_comparison() {
                Ok(Type::Bool)
            } else {
                Ok(unified)
            }
        }

        // Γ ⊢ e : F32  ⟹  -e : F32   (no signed integers in the DSL)
        ExprKind::Unary { op: UnOp::Neg, expr: inner } => {
            let ty = type_check_expr(inner, ctx, Some(&Type::F32))?;
            if ty != Type::F32 {
                return Err(TypeError::TypeMismatch {
                    expected: "F32".into(),
                    found: format!("-{ty:?}"),
                });
            }
            Ok(Type::F32)
        }

        // Γ ⊢ e : τ₁   τ₁, τ₂ numeric  ⟹  e as τ₂ : τ₂
        ExprKind::Cast { expr: inner, ty } => {
            let from = type_check_expr(inner, ctx, None)?;
            if !from.is_numeric() || !ty.is_numeric() {
                return Err(TypeError::InvalidCast {
                    from: format!("{from:?}"),
                    to: format!("{ty:?}"),
                });
            }
            Ok(ty.clone())
        }

        ExprKind::Call { func, args } => {
            let (_, param_tys, ret_ty) = BUILTIN_FUNCTIONS
                .iter()
                .find(|(name, _, _)| name == func)
                .ok_or_else(|| TypeError::UnknownFunction(func.clone()))?;

            if args.len() != param_tys.len() {
                return Err(TypeError::ArityMismatch {
                    func: func.clone(),
                    expected: param_tys.len(),
                    found: args.len(),
                });
            }

            for (arg, param_ty) in args.iter().zip(param_tys.iter()) {
                let arg_ty = type_check_expr(arg, ctx, Some(param_ty))?;
                if !coerces_to(&arg_ty, param_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: format!("{func}({param_ty:?})"),
                        found: format!("{func}({arg_ty:?})"),
                    });
                }
            }

            Ok(ret_ty.clone())
        }

        ExprKind::Field { base, member } => {
            let tbase = type_check_expr(base, ctx, None)?;
            match (deref(&tbase), member.as_str()) {
                (Type::Dim3, "x") | (Type::Dim3, "y") | (Type::Dim3, "z") => Ok(Type::U32),
                _ => Err(TypeError::InvalidFieldProperty),
            }
        }

        // Γ ⊢ size_expr : U64
        // ---------------------------
        // Γ ⊢ CudaVec(size_expr) : CudaVec<F32>
        ExprKind::CudaVec(size_expr) => {
            let ty = type_check_expr(size_expr, ctx, Some(&Type::U64))?;

            match ty {
                Type::U64 => Ok(Type::CudaVec(Box::new(Type::F32))),
                _ => Err(TypeError::InvalidCudaVecSize)
            }
        }

        // Γ ⊢ target : CudaVec<T> (through & / &mut)
        // Γ ⊢ index  : U32 | U64
        // ---------------------------
        // Γ ⊢ target[index] : T
        ExprKind::Index { target, index } => {
            let target_ty = type_check_expr(target, ctx, None)?;
            let index_ty = type_check_expr(index, ctx, Some(&Type::U64))?;

            let element_ty = match deref(&target_ty) {
                Type::CudaVec(inner) => inner.as_ref().clone(),
                _ => return Err(TypeError::InvalidIndexing),
            };

            if !index_ty.is_integer() {
                return Err(TypeError::InvalidIndexing);
            }

            Ok(element_ty)
        }

        ExprKind::Assign { target, value } => match &target.kind {
            // x = e  — x must be declared and `mut`
            ExprKind::Var(name) => {
                let binding = ctx
                    .get(name)
                    .ok_or_else(|| TypeError::UnknownVariable(name.clone()))?
                    .clone();

                if !binding.mutable {
                    return Err(TypeError::NotMutable(name.clone()));
                }

                let value_ty = type_check_expr(value, ctx, Some(&binding.ty))?;
                if !coerces_to(&value_ty, &binding.ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: format!("{:?}", binding.ty),
                        found: format!("{value_ty:?}"),
                    });
                }

                Ok(Type::Unit)
            }

            // v[i] = e — v must be a writable CudaVec place
            ExprKind::Index { target: base, index } => {
                let base_name = match &base.kind {
                    ExprKind::Var(name) => name.clone(),
                    _ => return Err(TypeError::InvalidAssignmentTarget),
                };

                let binding = ctx
                    .get(&base_name)
                    .ok_or_else(|| TypeError::UnknownVariable(base_name.clone()))?
                    .clone();

                let (writable, element_ty) = match &binding.ty {
                    Type::CudaVec(inner) => (binding.mutable, inner.as_ref().clone()),
                    Type::Ref { mutable, inner } => match inner.as_ref() {
                        Type::CudaVec(element) => (*mutable, element.as_ref().clone()),
                        _ => return Err(TypeError::InvalidIndexing),
                    },
                    _ => return Err(TypeError::InvalidIndexing),
                };

                if !writable {
                    return Err(TypeError::NotMutable(base_name));
                }

                let index_ty = type_check_expr(index, ctx, Some(&Type::U64))?;
                if !index_ty.is_integer() {
                    return Err(TypeError::InvalidIndexing);
                }

                let value_ty = type_check_expr(value, ctx, Some(&element_ty))?;
                if !coerces_to(&value_ty, &element_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: format!("{element_ty:?}"),
                        found: format!("{value_ty:?}"),
                    });
                }

                Ok(Type::Unit)
            }

            _ => Err(TypeError::InvalidAssignmentTarget),
        },
    }
}

// `found` is usable where `expected` is required: equal types, or U32 → U64 widening.
fn coerces_to(found: &Type, expected: &Type) -> bool {
    found == expected || (*found == Type::U32 && *expected == Type::U64)
}

// Same numeric type, or widened to U64 when U32 and U64 meet.
fn unify_numeric(lhs: &Type, rhs: &Type) -> Option<Type> {
    if !lhs.is_numeric() || !rhs.is_numeric() {
        return None;
    }
    if lhs == rhs {
        return Some(lhs.clone());
    }
    if lhs.is_integer() && rhs.is_integer() {
        return Some(Type::U64);
    }
    None
}

// Indexing and field access look through references.
fn deref(ty: &Type) -> &Type {
    match ty {
        Type::Ref { inner, .. } => deref(inner),
        other => other,
    }
}

fn builtin_type(name: &str) -> Option<Type> {
    match name {
        "blockIdx" => Some(Type::Dim3),
        "threadIdx" => Some(Type::Dim3),
        "blockDim" => Some(Type::Dim3),
        "gridDim" => Some(Type::Dim3),
        "warpSize" => Some(Type::U32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use crate::{context::Context, lower::lower_fn, type_checker::type_check, types::{Type, TypeError}};

    // Parse a kernel-shaped fn, lower it, and type-check it.
    fn check(tokens: proc_macro2::TokenStream) -> Result<Type, TypeError> {
        let item: syn::ItemFn = syn::parse2(tokens).expect("test source must parse");
        let func = lower_fn(&item).expect("test source must lower");
        let mut ctx = Context::new();
        type_check(&func, &mut ctx)
    }

    // ---- positive ----

    #[test]
    fn params_are_in_scope() {
        let result = check(quote! {
            fn f(n: u64) {
                let x: u64 = n;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn thread_index_u32_widens_to_u64() {
        let result = check(quote! {
            fn f() {
                let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn index_read_through_shared_ref() {
        let result = check(quote! {
            fn f(a: &CudaVec<f32>) {
                let x: f32 = a[0u64];
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn index_read_with_u32_index() {
        let result = check(quote! {
            fn f(a: &CudaVec<f32>) {
                let x: f32 = a[threadIdx.x];
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn index_assign_through_mut_ref() {
        let result = check(quote! {
            fn f(a: &CudaVec<f32>, result: &mut CudaVec<f32>) {
                let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
                result[idx] = a[idx] + a[idx];
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn mut_var_can_be_reassigned() {
        let result = check(quote! {
            fn f() {
                let mut x: f32 = 0.0f32;
                x = 1.0f32;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn if_with_bool_condition() {
        let result = check(quote! {
            fn f(n: u64) {
                if threadIdx.x < n {
                    let x: f32 = 1.0f32;
                }
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn for_loop_var_usable_as_index() {
        let result = check(quote! {
            fn f(a: &CudaVec<f32>, out: &mut CudaVec<f32>, n: u64) {
                for i in (0u64, n).step_by(1u64) {
                    out[i] = a[i];
                }
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn numeric_casts() {
        let result = check(quote! {
            fn f(n: u64) {
                let x: f32 = n as f32;
                let y: u32 = x as u32;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn builtin_call_sqrtf() {
        let result = check(quote! {
            fn f() {
                let x: f32 = sqrtf(2.0f32);
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn comparison_yields_bool() {
        let result = check(quote! {
            fn f() {
                let b: bool = 1.0f32 < 2.0f32;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn logical_and_of_comparisons() {
        let result = check(quote! {
            fn f(n: u64, m: u64) {
                let b: bool = n < m && m < n;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn polymorphic_int_literal_checks_against_u32() {
        let result = check(quote! {
            fn f() {
                let mut escaped: u32 = 0;
                escaped = 1;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    #[test]
    fn negated_float_literal() {
        let result = check(quote! {
            fn f() {
                let x: f32 = -99999.0f32;
            }
        });
        assert_eq!(result, Ok(Type::Unit));
    }

    // ---- negative ----

    #[test]
    fn let_rejects_float_for_u64_annotation() {
        let result = check(quote! {
            fn f() {
                let y: u64 = 12.0;
            }
        });
        assert_eq!(
            result,
            Err(TypeError::LetTypeMismatch {
                name: "y".into(),
                expected: "U64".into(),
                found: "F32".into(),
            })
        );
    }

    #[test]
    fn assign_to_immutable_var_fails() {
        let result = check(quote! {
            fn f() {
                let x: f32 = 0.0f32;
                x = 1.0f32;
            }
        });
        assert_eq!(result, Err(TypeError::NotMutable("x".into())));
    }

    #[test]
    fn index_write_through_shared_ref_fails() {
        let result = check(quote! {
            fn f(a: &CudaVec<f32>) {
                a[0u64] = 1.0f32;
            }
        });
        assert_eq!(result, Err(TypeError::NotMutable("a".into())));
    }

    #[test]
    fn non_bool_condition_fails() {
        let result = check(quote! {
            fn f() {
                if 1 {
                    let x: f32 = 1.0f32;
                }
            }
        });
        assert_eq!(result, Err(TypeError::ConditionNotBool("U64".into())));
    }

    #[test]
    fn mixed_f32_u64_addition_fails() {
        let result = check(quote! {
            fn f(n: u64) {
                let x: f32 = 1.0f32 + n;
            }
        });
        assert!(matches!(result, Err(TypeError::TypeMismatch { .. })));
    }

    #[test]
    fn unknown_variable_fails() {
        let result = check(quote! {
            fn f() {
                let x: f32 = y;
            }
        });
        assert_eq!(result, Err(TypeError::UnknownVariable("y".into())));
    }

    #[test]
    fn unknown_function_fails() {
        let result = check(quote! {
            fn f() {
                let x: f32 = frobnicate(1.0f32);
            }
        });
        assert_eq!(result, Err(TypeError::UnknownFunction("frobnicate".into())));
    }

    #[test]
    fn call_arity_mismatch_fails() {
        let result = check(quote! {
            fn f() {
                let x: f32 = sqrtf(1.0f32, 2.0f32);
            }
        });
        assert_eq!(
            result,
            Err(TypeError::ArityMismatch {
                func: "sqrtf".into(),
                expected: 1,
                found: 2,
            })
        );
    }

    #[test]
    fn cast_to_bool_fails() {
        let result = check(quote! {
            fn f(n: u64) {
                let b: bool = n as bool;
            }
        });
        assert_eq!(
            result,
            Err(TypeError::InvalidCast {
                from: "U64".into(),
                to: "Bool".into(),
            })
        );
    }

    #[test]
    fn assign_to_undeclared_var_fails() {
        let result = check(quote! {
            fn f() {
                x = 1.0f32;
            }
        });
        assert_eq!(result, Err(TypeError::UnknownVariable("x".into())));
    }

    #[test]
    fn block_scoped_local_is_not_visible_outside() {
        let result = check(quote! {
            fn f(n: u64) {
                if n < 1u64 {
                    let x: f32 = 1.0f32;
                }
                let y: f32 = x;
            }
        });
        assert_eq!(result, Err(TypeError::UnknownVariable("x".into())));
    }

    #[test]
    fn for_loop_var_not_visible_after_loop() {
        let result = check(quote! {
            fn f(n: u64) {
                for i in (0u64, n).step_by(1u64) {
                    let x: u64 = i;
                }
                let y: u64 = i;
            }
        });
        assert_eq!(result, Err(TypeError::UnknownVariable("i".into())));
    }
}

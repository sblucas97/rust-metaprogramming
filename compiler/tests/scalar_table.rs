//! Pins the shared scalar table in types.rs: lowering (rust name -> Type) and
//! codegen (rust name -> CUDA C name) both read it, so these tests are what
//! keeps the DSL's type tables from drifting apart again.

use compiler::types::{cuda_name, scalar_from_name, Type, SCALARS};

#[test]
fn every_scalar_row_is_consistent() {
    for (rust_name, ty, c_name) in SCALARS {
        assert_eq!(
            scalar_from_name(rust_name).as_ref(),
            Some(ty),
            "scalar_from_name({rust_name})"
        );
        assert_eq!(cuda_name(rust_name), Some(*c_name), "cuda_name({rust_name})");
        assert!(ty.is_numeric(), "{ty:?} in SCALARS must be numeric");
    }
}

#[test]
fn expected_scalar_set_is_exactly_covered() {
    let names: Vec<&str> = SCALARS.iter().map(|(n, _, _)| *n).collect();
    assert_eq!(
        names,
        ["f32", "f64", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"]
    );
}

#[test]
fn non_scalars_are_not_in_the_table() {
    for name in ["bool", "usize", "isize", "f16", "char", "CudaVec"] {
        assert_eq!(scalar_from_name(name), None, "{name} must not be a scalar");
        assert_eq!(cuda_name(name), None, "{name} must have no CUDA name");
    }
}

// Exhaustiveness guard: adding a Type variant fails this match until the
// author decides whether it belongs in SCALARS (and updates the sets below).
#[test]
fn every_type_variant_is_classified() {
    let in_table = |ty: &Type| SCALARS.iter().any(|(_, t, _)| t == ty);

    let probe = |ty: Type| match ty {
        Type::F32
        | Type::F64
        | Type::U8
        | Type::U16
        | Type::U32
        | Type::U64
        | Type::I8
        | Type::I16
        | Type::I32
        | Type::I64 => assert!(in_table(&ty), "{ty:?} is a scalar and belongs in SCALARS"),
        Type::Bool | Type::Unit | Type::CudaVec(_) | Type::Dim3 | Type::Ref { .. } => {
            assert!(!in_table(&ty), "{ty:?} must not be in SCALARS")
        }
    };

    probe(Type::F32);
    probe(Type::F64);
    probe(Type::U8);
    probe(Type::U16);
    probe(Type::U32);
    probe(Type::U64);
    probe(Type::I8);
    probe(Type::I16);
    probe(Type::I32);
    probe(Type::I64);
    probe(Type::Bool);
    probe(Type::Unit);
    probe(Type::CudaVec(Box::new(Type::F32)));
    probe(Type::Dim3);
    probe(Type::Ref { mutable: false, inner: Box::new(Type::F32) });
}

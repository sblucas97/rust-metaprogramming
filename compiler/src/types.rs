#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    Unit,
    CudaVec(Box<Type>),
    Dim3,
    Ref {
        mutable: bool,
        inner: Box<Type>
    }
}

/// The numeric scalars the DSL supports, as (rust name, Type, CUDA C name).
///
/// Single source of truth shared by lowering (rust name -> Type) and codegen
/// (rust name -> CUDA C name), so the two can't drift apart. `bool` is kept
/// out on purpose: it is a valid scalar for locals/conditions but not a
/// CudaVec element, and codegen special-cases it.
pub const SCALARS: &[(&str, Type, &str)] = &[
    ("f32", Type::F32, "float"),
    ("f64", Type::F64, "double"),
    ("u8", Type::U8, "uint8_t"),
    ("u16", Type::U16, "uint16_t"),
    ("u32", Type::U32, "uint32_t"),
    ("u64", Type::U64, "uint64_t"),
    ("i8", Type::I8, "int8_t"),
    ("i16", Type::I16, "int16_t"),
    ("i32", Type::I32, "int32_t"),
    ("i64", Type::I64, "int64_t"),
];

pub fn scalar_from_name(name: &str) -> Option<Type> {
    SCALARS
        .iter()
        .find(|(rust_name, _, _)| *rust_name == name)
        .map(|(_, ty, _)| ty.clone())
}

pub fn cuda_name(name: &str) -> Option<&'static str> {
    SCALARS
        .iter()
        .find(|(rust_name, _, _)| *rust_name == name)
        .map(|(_, _, c_name)| *c_name)
}

impl Type {
    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    pub fn is_signed_int(&self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }

    pub fn is_unsigned_int(&self) -> bool {
        matches!(self, Type::U8 | Type::U16 | Type::U32 | Type::U64)
    }

    pub fn is_integer(&self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Bit width used for integer-widening decisions; None for non-integers.
    pub fn int_rank(&self) -> Option<u8> {
        match self {
            Type::U8 | Type::I8 => Some(8),
            Type::U16 | Type::I16 => Some(16),
            Type::U32 | Type::I32 => Some(32),
            Type::U64 | Type::I64 => Some(64),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TypeError {
    UnknownVariable(String),
    TypeMismatch {
        expected: String,
        found: String,
    },
    InvalidAssignmentTarget,
    InvalidIndexing,
    InvalidCudaVecSize,
    InvalidFieldProperty,
    NotMutable(String),
    UnknownFunction(String),
    ArityMismatch {
        func: String,
        expected: usize,
        found: usize,
    },
    ConditionNotBool(String),
    InvalidCast {
        from: String,
        to: String,
    },
    LetTypeMismatch {
        name: String,
        expected: String,
        found: String,
    },
}

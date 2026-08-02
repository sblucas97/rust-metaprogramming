#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32,
    U64,
    U32,
    Bool,
    Unit,
    CudaVec(Box<Type>),
    Dim3,
    Ref {
        mutable: bool,
        inner: Box<Type>
    }
}

impl Type {
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::F32 | Type::U32 | Type::U64)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Type::U32 | Type::U64)
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

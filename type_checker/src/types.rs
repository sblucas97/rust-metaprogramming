#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32,
    U64,
    Bool,
    Unit,
}

#[derive(Debug, PartialEq)]
pub enum TypeError {
    UnknownVariable(String),
    TypeMismatch {
        expected: String,
        found: String,
    },
    InvalidAssignmentTarget,
}
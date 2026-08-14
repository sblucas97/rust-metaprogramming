use crate::types::Type;

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty: Type,
        mutable: bool,
        value: Expr,
    },
    If {
        cond: Expr,
        then_block: Vec<Stmt>,
        else_block: Option<Vec<Stmt>>,
    },
    // for var in (start, end).step_by(step) { body }
    For {
        var: String,
        start: Expr,
        end: Expr,
        step: Expr,
        body: Vec<Stmt>,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinOp {
    pub fn is_arithmetic(self) -> bool {
        matches!(self, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
    }

    pub fn is_comparison(self) -> bool {
        matches!(self, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne)
    }

    pub fn is_logical(self) -> bool {
        matches!(self, BinOp::And | BinOp::Or)
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    // Unsuffixed float literal: polymorphic, checks against an expected float
    // type when one is known, defaults to F32. Stored as f64 so f64-typed
    // literals lose no precision.
    LiteralFloat(f64),
    // Suffixed float literal (0.0f32 / 0.0f64): fixed type.
    LiteralTypedFloat(f64, Type),
    // Unsuffixed integer literal: polymorphic, checks against an expected
    // integer type when one is known, defaults to U64.
    LiteralInt(u64),
    // Suffixed integer literal (0u32 / 0i64 / ...): fixed type.
    LiteralTypedInt(u64, Type),

    CudaVec(Box<Expr>),

    Var(String),

    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },

    Cast {
        expr: Box<Expr>,
        ty: Type,
    },

    Call {
        func: String,
        args: Vec<Expr>,
    },

    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },

    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },

    Field {
        base: Box<Expr>,
        member: String,
    }
}

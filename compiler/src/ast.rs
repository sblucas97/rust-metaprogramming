use crate::types::Type;


#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        value: Expr,
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

#[derive(Debug, Clone)]
pub enum Expr {
    LiteralF32(f32),
    LiteralU64(u64),

    CudaVec(Box<Expr>),

    Var(String),

    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),

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
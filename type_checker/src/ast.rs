#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        value: Expr,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Expr {
    LiteralF32(f32),
    LiteralU64(u64),

    CudaVec(Box<Expr>),

    Var(String),

    Add(Box<Expr>, Box<Expr>),

    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },

    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    }

}
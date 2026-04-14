#[derive(Debug, Clone)]
pub enum Expr {
    LiteralF32(f32),
    LiteralU64(u64),

    Var(String),

    Add(Box<Expr>, Box<Expr>),

    Assing {
        target: Box<Expr>,
        value: Box<Expr>,
    }
}
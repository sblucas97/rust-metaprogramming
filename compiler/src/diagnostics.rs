use proc_macro2::Span;

pub struct Diagnostic { pub msg: String, pub span: Option<Span>, pub kind: DiagKind }
pub enum DiagKind { Parse, Type, Codegen }
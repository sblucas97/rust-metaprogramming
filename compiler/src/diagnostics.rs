use proc_macro2::Span;

#[derive(Debug)]
pub struct Diagnostic { pub msg: String, pub span: Option<Span>, pub kind: DiagKind }

#[derive(Debug)]
pub enum DiagKind { Parse, Type, Codegen }
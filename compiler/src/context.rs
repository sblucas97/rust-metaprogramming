use std::collections::HashMap;

use crate::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub ty: Type,
    pub mutable: bool,
}

// Lexically scoped typing context: a stack of frames, innermost last.
#[derive(Debug)]
pub struct Context {
    frames: Vec<HashMap<String, Binding>>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            frames: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.frames.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        assert!(self.frames.len() > 1, "cannot pop the outermost scope");
        self.frames.pop();
    }

    pub fn insert(&mut self, name: impl Into<String>, ty: Type, mutable: bool) {
        self.frames
            .last_mut()
            .expect("context has at least one frame")
            .insert(name.into(), Binding { ty, mutable });
    }

    pub fn get(&self, name: &str) -> Option<&Binding> {
        self.frames.iter().rev().find_map(|frame| frame.get(name))
    }
}

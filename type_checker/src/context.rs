use std::collections::HashMap;

use crate::types::Type;

#[derive(Debug, Default)]
pub struct Context {
    variables: HashMap<String, Type>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, ty: Type) {
        self.variables.insert(name.into(), ty);
    }

    pub fn get(&self, name: &str) -> Option<&Type> {
        self.variables.get(name)
    }
}
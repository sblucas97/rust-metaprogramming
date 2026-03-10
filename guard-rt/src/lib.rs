use std::cell::Cell;

thread_local! {
    pub static CONTEXT_ACTIVE: Cell<bool> = Cell::new(false);
}

pub struct KernelContextGuard;

impl KernelContextGuard {
    pub fn new() -> Self {
        CONTEXT_ACTIVE.with(|f: &Cell<bool>| f.set(true));
        KernelContextGuard
    }
}

impl Drop for KernelContextGuard {
    fn drop(&mut self) {
        CONTEXT_ACTIVE.with(|f: &Cell<bool>| f.set(false));
    }
}
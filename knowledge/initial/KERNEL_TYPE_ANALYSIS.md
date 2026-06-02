# Type Analysis: Vector Sum Kernel

Detailed trace of how descend would type-check your `add_vectors` kernel using descend's internal type structures.

## The Kernel

```rust
#[kernel]
pub fn add_vectors(
    a: &CudaVec<f32>,
    b: &CudaVec<f32>,
    result: &mut CudaVec<f32>,
    n: u64
) {
    let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = a[idx] + b[idx];
}
```

---

## Stage 1: Lower (syn::ItemFn → Function)

```rust
// INPUT: syn::ItemFn
fn add_vectors(
    a: &CudaVec<f32>,
    b: &CudaVec<f32>,
    result: &mut CudaVec<f32>,
    n: u64
) { ... }

// OUTPUT: Function (descend's internal AST)
Function {
    name: "add_vectors",
    params: vec![
        Param { name: "a",      ty: Type::Ref { ... } },
        Param { name: "b",      ty: Type::Ref { ... } },
        Param { name: "result", ty: Type::Ref { ... } },
        Param { name: "n",      ty: Type::Scalar(U64) },
    ],
    body: vec![
        Stmt::Let { name: "idx", value: Expr::BinOp(Mul, ...) },
        Stmt::Expr { Expr::IdxAssign { target: Var("result"), idx: Var("idx"), value: Add(...) } },
    ]
}
```

---

## Stage 2: Expand Types (lower_type)

### Parameter `a: &CudaVec<f32>`

```rust
// Rust syntax: &CudaVec<f32> (immutable borrow)
// Descend equivalent: &shrd gpu.global [f32; n]

Type::Ref(RefDty {
    rgn: Provenance::Value("a"),        // lifetime / provenance
    own: Ownership::Shrd,                // &(read-only) = shared
    mem: Memory::GpuGlobal,              // lives in GPU global memory
    dty: Box::new(DataTy::new(
        DataTyKind::Array(
            Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::F32))),
            Nat::Ident("n")              // size parameterized by n
        )
    ))
})
```

**Why this type?**
- `&` (immutable) → `Ownership::Shrd` (shareable, multiple readers OK)
- `CudaVec<f32>` living on GPU → `Memory::GpuGlobal` (not CPU memory)
- `CudaVec` is really `[T; n]` internally, so `Array(F32, n)`

### Parameter `result: &mut CudaVec<f32>`

```rust
// Rust syntax: &mut CudaVec<f32> (mutable borrow)
// Descend equivalent: &uniq gpu.global [f32; n]

Type::Ref(RefDty {
    rgn: Provenance::Value("result"),   // lifetime
    own: Ownership::Uniq,                // &mut = unique (exclusive access)
    mem: Memory::GpuGlobal,              // GPU memory
    dty: Box::new(DataTy::new(
        DataTyKind::Array(
            Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::F32))),
            Nat::Ident("n")
        )
    ))
})
```

**Why Uniq?**
- `&mut` in Rust = exclusive access = `Uniq` in descend
- Only one writer at a time; incompatible with any other simultaneous access to the same location

### Parameter `n: u64`

```rust
Type::new(TyKind::Data(Box::new(
    DataTy::new(DataTyKind::Scalar(ScalarTy::U64))
)))
```

---

## Stage 3: Build Function Type (FnTy)

```rust
FnTy {
    generics: vec![],  // No generic params declared
    generic_exec: None,  // No generic exec parameter
    
    param_sigs: vec![
        ParamSig {
            exec_expr: ExecExpr { base: GpuGrid(64, 1024), path: [] },
            ty: Ty::new(TyKind::Data(RefDty { Shrd, gpu.global, [F32; n] }))
        },
        ParamSig {
            exec_expr: ExecExpr { base: GpuGrid(64, 1024), path: [] },
            ty: Ty::new(TyKind::Data(RefDty { Shrd, gpu.global, [F32; n] }))
        },
        ParamSig {
            exec_expr: ExecExpr { base: GpuGrid(64, 1024), path: [] },
            ty: Ty::new(TyKind::Data(RefDty { Uniq, gpu.global, [F32; n] }))
        },
        ParamSig {
            exec_expr: ExecExpr { base: GpuGrid(64, 1024), path: [] },
            ty: Ty::new(TyKind::Data(Scalar(U64)))
        },
    ],
    
    exec: ExecExpr { base: GpuGrid(64, 1024), path: [] },
    ret_ty: Box::new(Ty::new(TyKind::Data(Scalar(Unit)))),
    nat_constrs: vec![],
}
```

**Observation:** All parameters have `exec: GpuGrid` because they are kernel *parameters*, bound at the kernel boundary. The *body* will re-type them as `GpuThread` implicitly.

---

## Stage 4: Type-Check Expression (ty_check_expr)

### Initial Context Setup

```rust
TyCtx {
    frames: vec![
        Frame {
            bindings: vec![
                FrameEntry::Var(IdentTyped {
                    ident: "a",
                    ty: Ty::Data(RefDty { Shrd, gpu.global, [F32; n] }),
                    mutbl: Mutability::Const,
                    exec: ExecExpr { base: GpuThread, path: [] },  // ← Re-typed to thread level
                }),
                FrameEntry::Var(IdentTyped {
                    ident: "b",
                    ty: Ty::Data(RefDty { Shrd, gpu.global, [F32; n] }),
                    mutbl: Mutability::Const,
                    exec: ExecExpr { base: GpuThread, path: [] },
                }),
                FrameEntry::Var(IdentTyped {
                    ident: "result",
                    ty: Ty::Data(RefDty { Uniq, gpu.global, [F32; n] }),
                    mutbl: Mutability::Mut,
                    exec: ExecExpr { base: GpuThread, path: [] },
                }),
                FrameEntry::Var(IdentTyped {
                    ident: "n",
                    ty: Ty::Data(Scalar(U64)),
                    mutbl: Mutability::Const,
                    exec: ExecExpr { base: GpuThread, path: [] },
                }),
                FrameEntry::Var(IdentTyped {
                    ident: "blockIdx",
                    ty: Ty::Data(Dim3),
                    mutbl: Mutability::Const,
                    exec: ExecExpr { base: GpuGrid, path: [] },  // ← Grid-level built-in
                }),
                FrameEntry::Var(IdentTyped {
                    ident: "threadIdx",
                    ty: Ty::Data(Dim3),
                    mutbl: Mutability::Const,
                    exec: ExecExpr { base: GpuThread, path: [] },  // ← Thread-level built-in
                }),
                FrameEntry::Var(IdentTyped {
                    ident: "blockDim",
                    ty: Ty::Data(Dim3),
                    mutbl: Mutability::Const,
                    exec: ExecExpr { base: GpuThread, path: [] },  // Available at thread level
                }),
            ]
        }
    ]
}
```

### Expression 1: `blockIdx.x`

```
Type inference:
  blockIdx: Ty::Data(Dim3)  [from context]
  .x: field access on Dim3 → U32
  
Result: Ty::Data(Scalar(U32))
```

### Expression 2: `blockDim.x`

```
Type inference:
  blockDim: Ty::Data(Dim3)
  .x: U32
  
Result: Ty::Data(Scalar(U32))
```

### Expression 3: `blockIdx.x * blockDim.x`

```
Binary op: Mul
  Left: Scalar(U32)
  Right: Scalar(U32)
  → Scalar(U32)
  
Result: Ty::Data(Scalar(U32))
```

### Expression 4: `threadIdx.x`

```
Result: Ty::Data(Scalar(U32))
```

### Expression 5: `blockIdx.x * blockDim.x + threadIdx.x`

```
Binary op: Add
  Left: Scalar(U32)
  Right: Scalar(U32)
  → Scalar(U32)
  
Result: Ty::Data(Scalar(U32))
```

### Statement: `let idx: u64 = ...`

```rust
// Cast U32 to U64 (implicit in actual code, but type checker handles)
// Let statement adds to context:

FrameEntry::Var(IdentTyped {
    ident: "idx",
    ty: Ty::Data(Scalar(U64)),
    mutbl: Mutability::Const,
    exec: ExecExpr { base: GpuThread, path: [] },  // Created in thread body
})
```

---

## Stage 5: Index Access `a[idx]`

```rust
// Expression: Index { target: Var("a"), index: Var("idx") }
// Type check:

1. Type of target `a`:
   Ty::Data(RefDty { Shrd, gpu.global, Array(F32, n) })
   
2. Type of index `idx`:
   Ty::Data(Scalar(U64))
   
3. Type rule for indexing:
   target: Ref(dty @ mem), index: U64 → dty
   → F32
   
4. Borrow checking:
   - Call access_safety_check(ctx, PlaceExpr::Index(Var("a"), Var("idx")))
   
   a) narrowing_check:
      - "a" bound at exec: GpuThread
      - active exec: GpuThread
      - GpuThread is prefix of GpuThread? YES ✓
   
   b) access_conflict_check:
      - AccessCtx currently empty (first access)
      - No conflicts ✓
   
   c) borrow_check:
      - Create Loan { place_expr: a[idx], own: Shrd }
      - Insert into AccessCtx
   
5. Result:
   Type::Scalar(F32)
   Loan recorded: { place_expr: a[idx], own: Shrd }
```

---

## Stage 6: Index Access `b[idx]`

```rust
// Same as a[idx] but for b

1. Type: Scalar(F32) ✓
2. Narrowing: GpuThread vs GpuThread ✓
3. Conflict check:
   - AccessCtx has: Loan { a[idx], Shrd }
   - New access: b[idx]
   - Different ident ("b" vs "a") → NO CONFLICT ✓
4. Borrow check:
   - Record Loan { place_expr: b[idx], own: Shrd }
5. Result: F32
```

---

## Stage 7: Addition `a[idx] + b[idx]`

```rust
Binary op: Add
  Left: Scalar(F32)
  Right: Scalar(F32)
  
Match rule: (F32, F32) → F32

Type: Scalar(F32)
```

---

## Stage 8: Index Assign `result[idx] = ...`

```rust
// Expression: IdxAssign { target: Var("result"), idx: Var("idx"), value: Add(...) }

1. Type of target `result`:
   Ty::Data(RefDty { Uniq, gpu.global, Array(F32, n) })
   
2. Type of index `idx`:
   Ty::Data(Scalar(U64))
   
3. Type of value:
   Ty::Data(Scalar(F32))
   
4. Type rule for index assignment:
   target: Ref(Array(T, n) @ mem), index: U64, value: T → Unit
   → Unit ✓
   
5. Borrow checking for assignment target:
   - Call access_safety_check(ctx, PlaceExpr::Index(Var("result"), Var("idx")))
   
   a) narrowing_check:
      - "result" bound at: GpuThread
      - active exec: GpuThread
      - GpuThread ⊆ GpuThread? YES ✓
   
   b) access_conflict_check:
      - AccessCtx has:
        * Loan { a[idx], Shrd }
        * Loan { b[idx], Shrd }
      - New access: result[idx] with ownership Uniq
      - Check: is result[idx] in conflict with a[idx]?
        * Different ident → NO CONFLICT ✓
      - Check: is result[idx] in conflict with b[idx]?
        * Different ident → NO CONFLICT ✓
   
   c) borrow_check:
      - Record Loan { place_expr: result[idx], own: Uniq }
      - Insert into AccessCtx
   
6. After assignment:
   - Mark result[idx] as written to
   - Type: Unit
```

---

## Final Context State

```rust
TyCtx {
    frames: vec![
        Frame {
            bindings: vec![
                // Original params (re-typed to GpuThread)
                FieldTyped { "a",      RefDty { Shrd, gpu.global, [F32; n] }, GpuThread },
                FieldTyped { "b",      RefDty { Shrd, gpu.global, [F32; n] }, GpuThread },
                FieldTyped { "result", RefDty { Uniq, gpu.global, [F32; n] }, GpuThread },
                FieldTyped { "n",      U64, GpuThread },
                
                // Built-ins
                FieldTyped { "blockIdx", Dim3, GpuGrid },
                FieldTyped { "threadIdx", Dim3, GpuThread },
                FieldTyped { "blockDim", Dim3, GpuThread },
                
                // Local binding
                FieldTyped { "idx", U64, GpuThread },
                
                // Provenance mappings
                PrvMapping { prv: "a", loans: { Loan { a[idx], Shrd } } },
                PrvMapping { prv: "b", loans: { Loan { b[idx], Shrd } } },
                PrvMapping { prv: "result", loans: { Loan { result[idx], Uniq } } },
            ]
        }
    ]
}

AccessCtx {
    ctx: {
        Loan { a[idx], Shrd },
        Loan { b[idx], Shrd },
        Loan { result[idx], Uniq },
    }
}
```

---

## Safety Properties Verified

✅ **No Data Races:**
- `a[idx]` and `b[idx]` are both reads (`Shrd`) — multiple readers OK
- `result[idx]` is a write (`Uniq`) — exclusive access verified
- Each thread has unique `idx` (implicit from CUDA execution model)
- No two threads write to the same `result[idx]`

✅ **No Memory Aliasing:**
- `a`, `b`, `result` are all on `GpuGlobal` — consistent memory space
- No CPU pointer can alias them (would need `CpuMem` type)

✅ **Narrowing Valid:**
- All accesses happen at `GpuThread` scope
- Bindings (params) are available at `GpuThread` or broader
- No thread tries to read grid-level bindings

---

## Codegen Input (Typed AST)

After type checking, codegen receives:

```rust
Function {
    name: "add_vectors",
    params: [
        (a, RefDty { Shrd, gpu.global, [F32; n] }),
        (b, RefDty { Shrd, gpu.global, [F32; n] }),
        (result, RefDty { Uniq, gpu.global, [F32; n] }),
        (n, U64),
    ],
    body: [
        Let { idx: U64, value: ... },  // Type attached
        IdxAssign {
            target: Var("result"),
            idx: Var("idx"),
            value: BinOp(Add, Index(a, idx), Index(b, idx))
        }
    ]
    // Each Expr node has .ty = Some(Type)
}
```

Codegen knows the types of every expression and can emit type-safe CUDA without further checking.

---

## Summary Table

| Item | Type | Ownership | Memory | Exec | Notes |
|------|------|-----------|--------|------|-------|
| `a` | `[f32; n]` | `Shrd` | `gpu.global` | `GpuThread` | Immutable, shareable |
| `b` | `[f32; n]` | `Shrd` | `gpu.global` | `GpuThread` | Immutable, shareable |
| `result` | `[f32; n]` | `Uniq` | `gpu.global` | `GpuThread` | Mutable, exclusive |
| `n` | `u64` | — | — | `GpuThread` | Plain scalar |
| `idx` | `u64` | — | — | `GpuThread` | Computed per-thread |
| `a[idx]` | `f32` | `Shrd` | `gpu.global` | `GpuThread` | Loan recorded |
| `b[idx]` | `f32` | `Shrd` | `gpu.global` | `GpuThread` | Loan recorded |
| `a[idx]+b[idx]` | `f32` | — | — | `GpuThread` | Temporary value |
| `result[idx]` | `f32` | `Uniq` | `gpu.global` | `GpuThread` | Write target |

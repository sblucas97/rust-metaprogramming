# The Descend Type System: From Current to Data-Race-Free

A step-by-step guide to understanding descend's type system and how to build it in your project.

---

## Part 1: What You Have Now

### Current Type System (Simple)

```rust
// In type_checker/src/types.rs
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32,
    U64,
    U32,
    Bool,
    Unit,
    CudaVec(Box<Type>),
    Dim3,
    Ref { mutable: bool, inner: Box<Type> }
}
```

**What this represents:**
- Basic scalar types (`F32`, `U64`, etc.)
- Collections (`CudaVec<T>`)
- References (`&T` or `&mut T` distinguished only by `mutable: bool`)

**What's missing:**
- Memory location information (is this on CPU or GPU?)
- Ownership semantics (beyond mutable/immutable)
- Provenance tracking (which reference is this?)
- Execution context (where can this binding be used?)

### Current Type Checking (Simple)

```rust
// In type_checker/src/type_checker.rs
fn type_check_expr(expr: &Expr, ctx: &mut Context) -> Result<Type, TypeError> {
    match expr {
        Expr::LiteralF32(_) => Ok(Type::F32),
        Expr::Var(name) => ctx.get(name).cloned().ok_or(...),
        Expr::Add(left, right) => {
            let ty1 = type_check_expr(left, ctx)?;
            let ty2 = type_check_expr(right, ctx)?;
            match (&ty1, &ty2) {
                (Type::F32, Type::F32) => Ok(Type::F32),
                _ => Err(...)
            }
        }
        // ... more cases
    }
}
```

**What this does:**
- Bottom-up type inference (compute type of operands, then combine)
- Simple unification (types must match exactly)
- No data-race checking

**What's missing:**
- No loan tracking
- No ownership enforcement
- No execution context tracking
- No memory space checking

---

## Part 2: Descend's Type Hierarchy

### Full Type Representation

```
Type
 └─ TyKind
     ├─ Data(DataTy)
     │   └─ DataTyKind
     │       ├─ Scalar(ScalarTy)        ← f32, u64, i32, etc.
     │       ├─ Atomic(AtomicTy)        ← atomicAdd-safe types
     │       ├─ Array(DataTy, Nat)      ← [T; n]
     │       ├─ Tuple(Vec<DataTy>)      ← (T1, T2, ...)
     │       ├─ Struct(StructDecl)      ← custom structs
     │       ├─ At(DataTy, Memory)      ← T @ gpu.global (T at location)
     │       ├─ Ref(RefDty)             ← &r ω m T (reference)
     │       ├─ RawPtr(DataTy)          ← unsafe *T
     │       └─ Dead(DataTy)            ← type marked as moved/killed
     │
     └─ FnTy                             ← function type (not used in kernels yet)
         ├─ generics: Vec<IdentKinded>
         ├─ param_sigs: Vec<ParamSig>
         └─ ret_ty: Type

RefDty (the reference type — the star of the show)
 ├─ rgn: Provenance              ← Which reference? ("a", "b", "result")
 ├─ own: Ownership               ← Shrd or Uniq? (Multiple readers or exclusive?)
 ├─ mem: Memory                  ← Where does it live? (CPU or GPU?)
 └─ dty: Box<DataTy>             ← What does it point to?

Ownership
 ├─ Shrd                          ← &T (read-only, shareable)
 └─ Uniq                          ← &mut T (exclusive, writable)

Memory
 ├─ CpuMem                        ← CPU system memory
 ├─ GpuGlobal                     ← GPU global memory (slow, persistent)
 ├─ GpuShared                     ← GPU shared memory (fast, block-scoped)
 ├─ GpuLocal                      ← GPU local memory (registers)
 └─ Ident(String)                ← Generic parameter (e.g., <m: mem>)

Provenance
 ├─ Value(String)                ← Named: "a", "b", "result"
 └─ Ident(Ident)                 ← Generic: <'r: prv>

ScalarTy
 ├─ Unit, U8, U32, U64, I32, I64, F32, F64, Bool, Gpu
 └─ (defines primitive values)

AtomicTy
 ├─ AtomicU32, AtomicI32
 └─ (thread-safe primitives for histogram, etc.)

Nat (natural numbers, for array sizes)
 ├─ Lit(u64)                     ← Concrete: 42, 1024
 ├─ Ident(String)                ← Symbolic: "n", "m"
 └─ BinOp(Box<Nat>, Op, Box<Nat>) ← Expressions: n + 1, n / 2
```

---

## Part 3: The Three Axes of Safety

Descend's type system has **three independent dimensions** for safety:

### Axis 1: Ownership (Who Can Write?)

```
Ownership::Shrd
├─ Multiple threads can read
├─ No thread can write
├─ Rules:
│   ├─ Shrd + Shrd = OK (many readers)
│   ├─ Shrd + Uniq = ERROR (reader + writer conflict)
│   └─ Uniq + Uniq = ERROR (writers conflict)
└─ Example: &CudaVec<f32> in add_vectors → Shrd

Ownership::Uniq
├─ Exactly one thread can have access
├─ That thread can read and write
├─ Rules:
│   ├─ Uniq + anything = ERROR (exclusive access violated)
│   └─ Two uses of same Uniq reference to same place = ERROR
└─ Example: &mut CudaVec<f32> in add_vectors → Uniq
```

**What each prevents:**

```
Scenario 1: Two threads read from a[0]
  Thread 0: Loan { a[0], Shrd }
  Thread 1: Loan { a[0], Shrd }
  Result: ✓ OK (Shrd + Shrd is safe)

Scenario 2: One thread reads, another writes to result[0]
  Thread 0: Loan { result[0], Uniq }
  Thread 1: Loan { result[0], Uniq }
  Result: ✗ ERROR (Uniq conflict)

Scenario 3: Same thread reads then writes to result[0]
  Loan 1: { result[0], Uniq }
  Loan 2: { result[0], Uniq }
  Result: ✗ ERROR (Can't hold two exclusive loans)
```

### Axis 2: Memory (Where Does It Live?)

```
Memory::CpuMem
├─ Lives in CPU system RAM
├─ CPU can access directly
├─ GPU must load/store via transfer
└─ Example: Vec<f32> on host

Memory::GpuGlobal
├─ Lives in GPU global memory (DRAM-like)
├─ All GPU threads can access
├─ All GPU blocks can access
├─ CPU must transfer to/from
└─ Example: CudaVec<f32> allocated on device

Memory::GpuShared
├─ Lives in GPU shared memory (fast, limited)
├─ Only threads in same block can access
├─ Shared between threads in block
├─ Block-scoped lifetime
└─ Example: __shared__ T in CUDA

Memory::GpuLocal
├─ Lives in GPU local memory (registers/local)
├─ Only one thread can access
├─ Thread-scoped lifetime
└─ Example: Local variable in kernel
```

**What each prevents:**

```
Scenario 1: Aliasing through different memory spaces
  CPU code: let cpu_ptr = vec.as_ptr();
  GPU code: result[0] = ...;
  
  CPU type: RefDty { own: Uniq, mem: CpuMem, ... }
  GPU type: RefDty { own: Uniq, mem: GpuGlobal, ... }
  
  Result: ✗ ERROR (same data, different memory spaces)
  Prevention: Memory annotation catches cross-space aliasing

Scenario 2: CPU-GPU coherence violation
  GPU writes to result (GpuGlobal)
  CPU reads from result without sync
  
  Type system can't prevent this (needs runtime sync)
  But Memory annotation documents the constraint
```

### Axis 3: Execution Context (Where Is This Binding?)

```
ExecExpr tracks:
├─ BaseExec
│   ├─ CpuThread
│   ├─ GpuGrid(Dim, Dim)
│   └─ Ident (generic parameter)
│
└─ Path extensions (narrowing)
    ├─ TakeRange
    ├─ ForAll(DimCompo)
    ├─ ToWarps
    └─ ToThreads(DimCompo)

Example narrowing chain:
  GpuGrid(64, 1024)
    └─ ForAll block in grid
       └─ GpuBlock(64, 1024)
          └─ ForAll thread in block
             └─ GpuThread
```

**What each prevents:**

```
Scenario 1: Thread reading grid-level exclusive data
  Parameter: result: &mut CudaVec<f32>
    exec: GpuGrid         ← Bound at kernel boundary
    own: Uniq             ← Exclusive access
  
  In kernel body (GpuThread level):
    result[idx] = ...     ← Trying to narrow Uniq from GpuGrid to GpuThread
  
  Result: ✗ ERROR (narrowing_check fails)
  Prevention: Each thread cannot claim exclusive access to grid-level data

Scenario 2: Block-level data safely accessed by threads
  (Hypothetically, with sched construct)
  
  block_data: &mut T
    exec: GpuBlock        ← Bound at block level
    own: Uniq             ← Exclusive to this block
  
  In block body (GpuThread level):
    block_data[threadIdx] = ...  ← Each thread accesses different index
  
  Result: ✓ OK (narrowing valid, different indices = no conflict)
  Prevention: Within same block, threads can safely share exclusive block-level data
```

---

## Part 4: Type Inference Rules

### Rule 1: Literal Types

```
────────────
Γ ⊢ 42 : u64

────────────
Γ ⊢ 3.14 : f32

────────────
Γ ⊢ () : unit
```

**In code:**
```rust
Expr::LiteralU64(_) => Ok(Type::new(TyKind::Data(DataTy::new(
    DataTyKind::Scalar(ScalarTy::U64)
))))
```

### Rule 2: Variable Lookup

```
x: T ∈ Γ
────────
Γ ⊢ x : T
```

**In code:**
```rust
Expr::Var(name) => {
    ctx.get(name)
        .cloned()
        .ok_or(TypeError::UnknownVariable(name.clone()))
}
```

### Rule 3: Binary Operations (Same Type)

```
Γ ⊢ e1 : f32    Γ ⊢ e2 : f32
────────────────────────────
Γ ⊢ e1 + e2 : f32
```

**In code:**
```rust
Expr::Add(left, right) => {
    let ty1 = type_check_expr(left, ctx)?;
    let ty2 = type_check_expr(right, ctx)?;
    match (&ty1, &ty2) {
        (Type::F32, Type::F32) => Ok(Type::F32),
        (Type::U64, Type::U64) => Ok(Type::U64),
        _ => Err(TypeError::TypeMismatch { ... })
    }
}
```

### Rule 4: Indexing (Extract Element)

```
Γ ⊢ arr : ref (array T n) @ m    Γ ⊢ idx : u64
──────────────────────────────────────────────
Γ ⊢ arr[idx] : T
```

**In descend's terms:**
```
If arr has type RefDty { own, mem, dty: Array(T, n) }
and idx has type U64
then arr[idx] has type T
```

**In code:**
```rust
Expr::Index { target, index } => {
    let ty_target = type_check_expr(target, ctx)?;
    let ty_index = type_check_expr(index, ctx)?;
    
    match (ty_target, ty_index) {
        (Type::Ref(RefDty { dty: Array(inner, _), .. }), Type::U64) => {
            Ok(*inner)  // Return inner type
        }
        _ => Err(TypeError::InvalidIndexing)
    }
}
```

### Rule 5: Borrow (Create Reference)

```
Γ ⊢ p : place    Γ ⊢ access_safety_check(p, ω) : OK
────────────────────────────────────────────────────
Γ ⊢ &ω p : ref ω m T
```

**In code (simplified):**
```rust
Expr::Ref { prv, own, place_expr } => {
    let place_type = type_check_place(place_expr, ctx)?;
    
    // Check ownership safety
    let loans = access_safety_check(ctx, place_expr)?;
    
    // Create reference type
    Ok(Type::Ref(RefDty {
        rgn: Provenance::Value(prv.clone()),
        own,
        mem: place_type.mem,
        dty: Box::new(place_type),
    }))
}
```

### Rule 6: Assignment (Introduce Binding)

```
Γ ⊢ e : T
──────────────────────
Γ, x : T ⊢ rest
```

**In code:**
```rust
Stmt::Let { name, value } => {
    let value_type = type_check_expr(value, ctx)?;
    ctx.insert(name.clone(), value_type);
    type_check_rest(ctx)
}
```

---

## Part 5: Type Checking Flow

### The Pipeline

```
1. LOWER: syn::ItemFn → Function
   Input:  #[kernel] pub fn add_vectors(a: &CudaVec<f32>, ...)
   Process: Extract params, stmts, return type
   Output:  Function { name, params: [Param { name, ty: Type }], body }

2. TYPE_CHECK: Function → Typed Function
   Input:  Function (untyped stmts and exprs)
   Process:
     a. Build initial TyCtx from params
     b. For each stmt, call type_check_stmt
     c. For each expr, call type_check_expr
     d. Mutate Expr.ty = Some(Type)
     e. Check borrow safety on memory accesses
   Output: Function (every Expr has .ty)

3. CODEGEN: Typed Function → .cu file
   Input:  Function with types attached
   Process: Walk typed AST, emit CUDA
   Output:  .cu file

4. NVCC: .cu → .ptx
   Input:  .cu file
   Output: .ptx binary
```

### Phase 1: Lower Type

```rust
// Input: syn::Type::Reference
// Example: &CudaVec<f32>

match ty {
    syn::Type::Reference(type_ref) => {
        let mutable = type_ref.mutability.is_some();  // true if &mut
        let inner = lower_type(&type_ref.elem)?;      // lower CudaVec<f32>
        
        Ok(Type::Ref {
            mutable,
            inner: Box::new(inner)
        })
    }
}

// Output: Type::Ref { mutable: false, inner: CudaVec(F32) }
```

**Current limitation:** Only captures mutability, loses memory and ownership info.

### Phase 2: Expand to RefDty

```rust
// Input: Type::Ref { mutable, inner }
// Process: Infer ownership, memory, provenance from context

fn expand_to_reftdy(
    ty: Type,
    param_name: &str,      // "a", "b", "result"
    is_gpu_kernel: bool,   // running on GPU?
) -> Type {
    match ty {
        Type::Ref { mutable, inner } => {
            Type::Ref(RefDty {
                rgn: Provenance::Value(param_name.to_string()),
                own: if mutable { Uniq } else { Shrd },
                mem: if is_gpu_kernel { GpuGlobal } else { CpuMem },
                dty: Box::new(inner),
            })
        }
        _ => ty
    }
}

// Output:
// &CudaVec<f32>       → RefDty { Shrd, GpuGlobal, [F32; n] }
// &mut CudaVec<f32>   → RefDty { Uniq, GpuGlobal, [F32; n] }
```

### Phase 3: Track Execution Context

```rust
// Input: Parameter with type
// Output: IdentTyped with exec

IdentTyped {
    ident: "a",
    ty: RefDty { Shrd, GpuGlobal, [F32; n] },
    mutbl: Mutability::Const,
    exec: ExecExpr {                          // ← New!
        base: BaseExec::GpuGrid(64, 1024),    // Kernel boundary
        path: []
    }
}

// In body, re-type to thread level:
IdentTyped {
    ident: "a",
    ty: RefDty { Shrd, GpuGlobal, [F32; n] },
    mutbl: Mutability::Const,
    exec: ExecExpr {                          // ← Changed!
        base: BaseExec::GpuThread,            // Thread-level execution
        path: []
    }
}
```

### Phase 4: Borrow Check

```rust
fn access_safety_check(ctx: &BorrowCheckCtx, p: &PlaceExpr) -> Result<HashSet<Loan>> {
    // 1. Narrowing check
    narrowing_check(ctx, p)?;
    
    // 2. Access conflict check
    access_conflict_check(ctx, p)?;
    
    // 3. Borrow check (record loan)
    let loans = borrow_check(ctx, p)?;
    
    Ok(loans)
}

fn access_conflict_check(ctx: &BorrowCheckCtx, p: &PlaceExpr) -> Result<()> {
    for existing_loan in ctx.access_ctx.hash_set() {
        if possible_conflict(ctx.nat_ctx, ctx.own, p, existing_loan)? {
            return Err(BorrowingError::Conflict { ... });
        }
    }
    Ok(())
}

fn possible_conflict(
    nat_ctx: &NatCtx,
    new_own: Ownership,
    new_place: &PlaceExpr,
    existing: &Loan,
) -> Result<bool> {
    // Check if places could alias
    let (new_ident, new_path) = new_place.as_ident_and_path();
    let (exist_ident, exist_path) = existing.place_expr.as_ident_and_path();
    
    if new_ident != exist_ident {
        return Ok(false);  // Different variables, no conflict
    }
    
    // Same variable — check indices
    match (&new_own, &existing.own) {
        (Shrd, Shrd) => Ok(false),           // Multiple readers OK
        (Uniq, _) | (_, Uniq) => Ok(true),  // Writer involved — conflict!
    }
}
```

---

## Part 6: Migration Path From Your Current System

### Phase 0: Foundation (Weeks 1-2)

**Goal:** Prepare the type system infrastructure.

1. **Add Span support to types**
   ```rust
   pub struct Ty {
       pub kind: TyKind,
       pub span: Option<Span>,  // ← New
   }
   ```

2. **Split types into categories**
   ```rust
   pub enum TyKind {
       Data(Box<DataTy>),
       FnTy(Box<FnTy>),
   }
   ```

3. **Add Mutability to Stmt::Let**
   ```rust
   Stmt::Let {
       name: String,
       mutbl: Mutability,  // ← New
       ty: Option<Ty>,     // ← New
       value: Expr,
   }
   ```

### Phase 1: Ownership & Memory (Weeks 3-4)

**Goal:** Capture ownership and memory information on references.

1. **Define Ownership and Memory**
   ```rust
   pub enum Ownership { Shrd, Uniq }
   pub enum Memory { CpuMem, GpuGlobal, GpuShared, GpuLocal }
   pub enum Provenance { Value(String), Ident(Ident) }
   ```

2. **Rewrite Ref variant**
   ```rust
   DataTyKind::Ref(Box<RefDty>)
   
   pub struct RefDty {
       pub rgn: Provenance,
       pub own: Ownership,
       pub mem: Memory,
       pub dty: Box<DataTy>,
   }
   ```

3. **Update lower_type to infer ownership**
   ```rust
   fn lower_type(ty: &syn::Type) -> Result<Type> {
       match ty {
           syn::Type::Reference(type_ref) => {
               let own = if type_ref.mutability.is_some() {
                   Ownership::Uniq
               } else {
                   Ownership::Shrd
               };
               // Default to GpuGlobal for CudaVec in kernels
               let mem = Memory::GpuGlobal;
               // ...
           }
       }
   }
   ```

### Phase 2: Execution Resources (Weeks 5-6)

**Goal:** Track which scope each binding lives in.

1. **Define ExecTy and ExecExpr**
   ```rust
   pub enum ExecTyKind {
       CpuThread,
       GpuThread,
       GpuBlock(Dim),
       GpuGrid(Dim, Dim),
   }
   
   pub struct ExecExpr {
       pub base: BaseExec,
       pub path: Vec<ExecPathElem>,
   }
   ```

2. **Update IdentTyped to track exec**
   ```rust
   pub struct IdentTyped {
       pub ident: Ident,
       pub ty: Ty,
       pub mutbl: Mutability,
       pub exec: ExecExpr,  // ← New
   }
   ```

3. **In type_check_global_fun_def, set execution context**
   ```rust
   let kernel_exec = ExecExpr::new(BaseExec::GpuGrid(...));
   let body_exec = ExecExpr::new(BaseExec::GpuThread);
   
   // Re-type parameters with body_exec
   for param in params {
       ctx.append_ident_typed(IdentTyped {
           ident: param.name,
           exec: body_exec.clone(),  // ← Re-typed
           ..
       });
   }
   ```

### Phase 3: Context Stack (Weeks 7-8)

**Goal:** Split flat HashMap into structured contexts.

1. **Build GlobalCtx**
   ```rust
   pub struct GlobalCtx {
       fn_sigs: HashMap<String, FnTy>,
       struct_decls: HashMap<String, StructDecl>,
       built_ins: HashMap<String, FnTy>,
   }
   ```

2. **Build TyCtx with frame stack**
   ```rust
   pub struct TyCtx {
       frames: Vec<Frame>,
   }
   
   pub struct Frame {
       bindings: Vec<FrameEntry>,
   }
   
   pub enum FrameEntry {
       Var(IdentTyped),
       ExecMapping(Ident, ExecExpr),
       PrvMapping { prv: String, loans: HashSet<Loan> },
   }
   ```

3. **Build AccessCtx**
   ```rust
   pub struct AccessCtx {
       ctx: HashSet<Loan>,
   }
   ```

4. **Migrate type_check_expr to use these**
   ```rust
   fn type_check_expr(
       ctx: &mut ExprTyCtx,  // Contains all four contexts
       expr: &mut Expr
   ) -> Result<Ty>
   ```

### Phase 4: Borrow Checking (Weeks 9-10)

**Goal:** Implement loan tracking and conflict detection.

1. **Define Loan**
   ```rust
   pub struct Loan {
       pub place_expr: PlaceExpr,
       pub own: Ownership,
   }
   ```

2. **Define Place for normalized place expressions**
   ```rust
   pub struct Place {
       pub ident: Ident,
       pub path: Vec<PathElem>,
   }
   
   pub enum PathElem {
       Proj(usize),
       FieldProj(Ident),
       Idx(Nat),
   }
   ```

3. **Implement access_safety_check**
   ```rust
   pub fn access_safety_check(
       ctx: &BorrowCheckCtx,
       p: &PlaceExpr,
   ) -> Result<HashSet<Loan>> {
       narrowing_check(ctx, p)?;
       access_conflict_check(ctx, p)?;
       borrow_check(ctx, p)
   }
   ```

4. **Wire into ty_check_expr**
   ```rust
   Expr::Index { target, index } => {
       // ... existing type inference ...
       
       // NEW: Check borrow safety
       let loans = access_safety_check(ctx, place_expr)?;
       ctx.access_ctx.insert(loans);
       
       Ok(result_type)
   }
   ```

### Phase 5: Parallel Constructs (Weeks 11-12)

**Goal:** Add sched or implicit parallel-for semantics.

(Optional for MVP — simpler to start with implicit kernel-body parallelism)

---

## Part 7: Summary: Three Axes Together

### How the Three Axes Work Together

```
Example 1: Reading shared input
Type: RefDty { Shrd, GpuGlobal, [F32; n] }
Exec: GpuThread

Access 1 (Thread 0): a[0] → Loan { a[0], Shrd }
Access 2 (Thread 1): a[0] → Loan { a[0], Shrd }

Conflict check:
  Ownership: Shrd + Shrd = OK ✓
  Place: a[0] same for both ✓
  Exec: Both GpuThread, both accessing same index ✓
  Result: ✅ SAFE

Example 2: Writing unique output
Type: RefDty { Uniq, GpuGlobal, [F32; n] }
Exec: GpuThread

Access 1 (Thread 0): result[0] → Loan { result[0], Uniq }
Access 2 (Thread 1): result[1] → Loan { result[1], Uniq }

Conflict check:
  Ownership: Uniq (exclusive) — need to check places
  Place: result[0] vs result[1] → Different indices ✓
  Exec: Both GpuThread but different indices ✓
  Result: ✅ SAFE

Example 3: Re-reading same reference (naive)
Type: RefDty { Uniq, GpuGlobal, [F32; n] }

Access 1: result[0] = 5 → Loan { result[0], Uniq }
Access 2: x = result[0] + 1 → Tries to add Loan { result[0], Uniq }

Conflict check:
  Ownership: Uniq + Uniq → CONFLICT ✗
  Place: Same place result[0] ✗
  Exec: Same thread ✗
  Result: ❌ ERROR
```

---

## Part 8: Key Differences From Current System

| Feature | Current | Descend |
|---------|---------|---------|
| **Reference Type** | `Ref { mutable, inner }` | `Ref(RefDty { rgn, own, mem, dty })` |
| **Ownership** | Implicit (mutable/immutable) | Explicit (`Shrd`/`Uniq`) |
| **Memory** | Implicit (assumed GPU) | Explicit (`CpuMem`, `GpuGlobal`, etc.) |
| **Provenance** | None | Named (`Value("a")`, etc.) |
| **Execution** | Not tracked | Tracked in `IdentTyped.exec` |
| **Context** | Flat `HashMap` | Stack of `Frame`s with multiple entry types |
| **Loan Tracking** | None | `AccessCtx` + `PrvMapping` |
| **Conflict Detection** | None | `access_conflict_check` |
| **Narrowing Check** | None | `narrowing_check` |
| **Data Race Safety** | Best effort (no guarantee) | Proven (complete check) |

---

## Part 9: Building Confidence

### Start Small

**Week 1 MVP: Just ownership**
- Add `Ownership { Shrd, Uniq }` to `RefDty`
- Update lowering to infer from `&` vs `&mut`
- Add simple `access_conflict_check` that forbids two `Uniq` loans
- Test on existing kernel (should still pass)

**Week 2 MVP: Add execution context**
- Add `exec: ExecExpr` to `IdentTyped`
- Thread exec through type checking
- Implement stub `narrowing_check` (accept everything for now)
- Add test kernel that would fail without narrowing (to verify stub works later)

**Week 3 MVP: Add borrow tracking**
- Implement `Place` + `Loan` structures
- Implement real `access_conflict_check`
- Wire into `Index` and `Assign` expressions
- Test with the "two threads same index" racy kernel — should now reject it

### Validation Kernels

Create these test kernels to validate each phase:

```rust
// Phase 1: Ownership — should PASS
#[kernel]
fn safe_reads(a: &CudaVec<f32>, b: &CudaVec<f32>) {
    let idx = ...;
    let x = a[idx];
    let y = b[idx];  // Two Shrd accesses, OK
}

// Phase 1: Ownership — should FAIL
#[kernel]
fn racy_writes(result: &mut CudaVec<f32>) {
    result[0] = 5.0;
    result[0] = 10.0;  // Two Uniq accesses to same place, ERROR
}

// Phase 2: Execution — should PASS
#[kernel]
fn safe_narrow(a: &CudaVec<f32>) {
    let idx = ...;
    let x = a[idx];  // Narrowing from kernel param OK
}

// Phase 2: Execution — should FAIL (with full narrowing check)
// (Harder to construct without sched construct)

// Phase 3: Full — should PASS
#[kernel]
fn safe_add(a: &CudaVec<f32>, b: &CudaVec<f32>, result: &mut CudaVec<f32>) {
    let idx = ...;
    result[idx] = a[idx] + b[idx];
}

// Phase 3: Full — should FAIL
#[kernel]
fn racy_hist(input: &CudaVec<u32>, hist: &mut CudaVec<u32>) {
    let val = input[threadIdx.x];
    hist[0] = hist[0] + val;  // All threads write same index, ERROR
}
```

---

## Conclusion: Your Roadmap

The journey from your current simple type system to descend's race-free system has **5 major phases**, each adding one key capability:

0. **Foundation** — Types with span, mutability tracking, structure
1. **Ownership** — Distinguish `Shrd` from `Uniq`, detect exclusive access violations
2. **Execution** — Track where bindings live (grid/block/thread), enable narrowing
3. **Contexts** — Split flat context into structured frames + loan tracking
4. **Borrow Check** — Implement full `access_safety_check` with all three checks
5. **Parallel** — (Optional) Add explicit sched construct for clearer semantics

Each phase builds on the last and adds ~2 weeks of work. By **week 10**, you'll have a functional type system that rejects data races in GPU kernels.

The hardest part isn't the implementation — it's the conceptual shift from "best effort" type checking to "proven safe" race-free code. Once you have ownership + execution + loan tracking, the rest is careful bookkeeping.

Start with Phase 0 + Phase 1 (weeks 1-4). Build the infrastructure and get comfortable with `Ownership` and `RefDty`. The rest flows naturally from there.

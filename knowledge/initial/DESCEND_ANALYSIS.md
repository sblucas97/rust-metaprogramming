# Descend: Type-Checker Architecture & a Roadmap for `rust-metaprogramming`

This document distills how [`descend`](https://github.com/descend-lang/descend) implements its
compile-time data-race-free type system, maps each piece to the current
`rust-metaprogramming/type_checker/` crate, and lays out the work needed to evolve our checker into
one that can actually reject racy GPU kernels at compile time.

The goal we are working toward: **a Rust attribute macro (`#[kernel]` / `#[cuda_module]`) that
fails compilation when the body of a kernel could exhibit a data race on `CudaVec` or any other
shared memory.**

---

## 1. How descend builds its type checker

Descend is its own surface language, so its compiler is a full pipeline. The shape of that pipeline
is what we care about — not the syntax.

```
source (.desc)
   │  parser/             (PEG, produces ast::CompilUnit)
   ▼
ast::CompilUnit
   │  ty_check::ty_check  (mutates AST, attaches Ty to every Expr)
   ▼
typed AST
   │  codegen::gen        (translates to cu_ast, then prints CUDA C++)
   ▼
CUDA C++
```

### 1.1 One AST + auxiliary typing-context data

`src/ast/mod.rs` defines the *single* program AST (~2000 LOC of enums/structs). `src/ast/internal.rs`
is **not** a second AST — it is a module of auxiliary data structures that only the type checker
ever creates or reads: `Frame`, `FrameEntry`, `IdentTyped`, `Place`, `PlaceCtx`, `Loan`,
`PrvMapping`. They encode *typing-context state* (live bindings, in-flight loans, dead places), not
program structure.

The distinction matters:

- The program AST (`PlaceExpr`, `ExprKind`, `DataTy`) is what the parser produces and codegen
  consumes. It lives for the whole compile.
- The context data (`Frame`, `Loan`, `Place`) is *transient*: created during checking, reset
  between sub-expressions, never read by codegen. `Place` and `PlaceCtx` are normalized
  projections *derived from* `PlaceExpr` — same information, restructured so the borrow checker can
  answer "is loan A a path-prefix of place B?" in a single match.

Calling `internal.rs` "a parallel AST" would be wrong — it is closer to the typing-context module
that lives alongside the AST, not a second representation of the program.

We have a faint shadow of this structure today:
- `type_checker/src/ast.rs` is the surface program AST (`Function`, `Stmt`, `Expr`).
- `type_checker/src/context.rs` is a flat `HashMap<String, Type>` — the start of a context
  module, but without frame discipline, loans, or exec tracking.

### 1.2 The type system: kinds and types

Descend separates **kinds** (`Nat`, `Memory`, `DataTy`, `Provenance`) from **types**:

```rust
pub enum TyKind { Data(Box<DataTy>), FnTy(Box<FnTy>) }

pub enum DataTyKind {
    Scalar(ScalarTy), Atomic(AtomicTy),
    Array(Box<DataTy>, Nat), ArrayShape(Box<DataTy>, Nat),
    Tuple(Vec<DataTy>), Struct(Box<StructDecl>),
    At(Box<DataTy>, Memory),          // dty stored *in* a specific memory
    Ref(Box<RefDty>),                 // &r ω m dty  — provenance, ownership, memory, pointee
    RawPtr(Box<DataTy>), Dead(Box<DataTy>),
    Ident(Ident),
}

pub struct RefDty { rgn: Provenance, own: Ownership, mem: Memory, dty: Box<DataTy> }
pub enum Ownership { Shrd, Uniq }     // exactly the "many readers XOR one writer" axiom
pub enum Memory  { CpuMem, GpuGlobal, GpuShared, GpuLocal, Ident(Ident) }
pub enum Provenance { Value(String), Ident(Ident) }  // lifetimes
```

Three things are doing the heavy lifting for data-race prevention:

1. **`Ownership::{Shrd, Uniq}`** on every reference type — Rust's borrow rule, lifted into the
   surface language.
2. **`Memory`** annotated on every reference — a kernel parameter typed
   `&uniq gpu.global [f32; n]` cannot accidentally be aliased by a CPU pointer.
3. **`Provenance`** — named lifetime regions whose live loans live in the typing context.

### 1.3 The execution-resource system (the unique-to-GPU part)

This is what we have *nothing* of today and is the single biggest idea to steal.

```rust
pub enum ExecTyKind {
    CpuThread, GpuThread, GpuWarp,
    GpuBlock(Dim), GpuGrid(Dim, Dim),
    GpuToThreads(Dim, Box<ExecTy>),
    GpuThreadGrp(Dim), GpuWarpGrp(Nat), GpuBlockGrp(Dim, Dim), Any,
}

pub enum BaseExec      { Ident(Ident), CpuThread, GpuGrid(Dim, Dim) }
pub enum ExecPathElem  { TakeRange(Box<TakeRange>), ForAll(DimCompo),
                         ToWarps, ToThreads(DimCompo) }
pub struct ExecExpr    { /* base + path of ExecPathElem */ ... }
```

Every binding in the typing context (`IdentTyped { ident, ty, mutbl, exec }`) is tagged with the
**execution resource** that owns it. A `let` inside `sched g in groups to block in grid { ... }`
records the bound variable under the inner `exec` (the per-block thread). That is the trick that
lets descend statically refuse `&uniq` access from a thread to memory owned by a different thread.

### 1.4 The contexts threaded through checking

`ty_check/ctxs.rs` defines four cooperating contexts:

| Context        | Holds                                                     | Lifetime           |
| -------------- | --------------------------------------------------------- | ------------------ |
| `GlobalCtx`    | Function & struct decls, signatures of built-ins          | Whole compile unit |
| `KindCtx`      | Type / nat / mem / provenance identifiers + outlives rels | Per function       |
| `TyCtx`        | Stack of `Frame`s; each `Frame` is `Vec<FrameEntry>`      | Per expression     |
| `AccessCtx`    | `HashSet<Loan>` accumulated *during* current sub-expr     | Per expression     |

A `FrameEntry` is one of: a typed binding, an `ExecMapping`, or a `PrvMapping { prv, loans }`.
This is critical — *loans live in the typing context tied to a named provenance*. Borrow check
becomes "does the loan I want to record conflict with anything stored under any active
provenance?".

### 1.5 The pass itself

`ty_check::ty_check` (entry point in `src/ty_check/mod.rs`) walks the AST and **mutates each
`Expr.ty`** to attach an inferred `Ty`. The kernel of the recursion is `ty_check_expr` — a big
match on `ExprKind` dispatching to per-form helpers (`ty_check_let`, `ty_check_app`,
`ty_check_borrow`, `ty_check_assign_place`, `ty_check_sched`, `ty_check_app_kernel`, …).

The data-race-relevant calls converge on `borrow_check::access_safety_check`, which performs:

1. **`narrowing_check`** — a unique borrow is only legal if the active `ExecExpr` is *more
   specific* (a path extension) than the binding's `exec`. This is what prevents two threads from
   getting `&uniq` to the same memory.
2. **`access_conflict_check`** — scans `AccessCtx` for an overlapping place expression with
   incompatible ownership.
3. **`borrow_check`** — records the new loan against its provenance.

The pre-declared built-ins (`exec`, `to_view_mut`, `group_mut`, `gpu_alloc_copy`, atomics, …) are
listed in `pre_decl.rs` as fully-typed `FnTy`s. The checker treats them as if they were user
functions; their signatures encode the safety contract.

### 1.6 Codegen is *after* checking

`codegen::gen` runs only if `ty_check` succeeded. It receives the typed AST and emits CUDA. The
type system therefore doesn't have to be "best effort" — by construction, anything that reaches
the printer is race-free.

---

## 2. Mapping descend → our current code

| Descend                                        | rust-metaprogramming today                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| `parser::parse`                                | `lower::lower_fn` (we lower `syn::ItemFn`, we don't parse strings)       |
| `ast::CompilUnit` / `Item::FunDef`             | `ast::Function`                                                          |
| `ast::ExprKind` (40+ variants)                 | `ast::Expr` (8 variants)                                                 |
| `ast::DataTy` / `DataTyKind`                   | `types::Type` (8 variants, no memory / no ownership / no lifetime)       |
| `ast::RefDty { rgn, own, mem, dty }`           | `Type::Ref { mutable, inner }` — drops *provenance and memory*           |
| `ast::ExecTy` / `ExecExpr`                     | **missing** — closest analog is the runtime `(grid_dim, block_dim)`      |
| `ast/internal.rs` — context data (`Frame`, `FrameEntry`) | `Context { variables: HashMap<String, Type> }` — same role, simpler shape |
| `ast/internal.rs` — `Place` / `PathElem` / `Loan` (derived from `PlaceExpr`, not a second AST) | **missing** |
| `ast/internal.rs` — `PrvMapping`               | **missing**                                                              |
| `ty_check::ctxs::{GlobalCtx, KindCtx, TyCtx,   AccessCtx}` | single `Context`                                              |
| `ty_check::ty_check_expr` + helpers            | `type_checker::type_check_expr` (one ~150-line `match`)                  |
| `ty_check::borrow_check::access_safety_check`  | **missing**                                                              |
| `ty_check::pre_decl` (signatures of built-ins) | `builtin_type()` (5 hard-coded names → `Type::Dim3 / U32`)               |
| `codegen::gen`                                 | `lib::helpers::gen_kernel` + `nvcc -ptx`                                 |

The macro pipeline in `lib/src/cuda_module.rs` calls `lower_fn` → `type_check` → `gen_kernel`,
which is the *right* shape — it mirrors descend's `parser → ty_check → codegen`. What we are
missing is the entire **typed-context + borrow-check layer**.

---

## 3. What you need to build for data-race prevention

A useful mental model: the checker today proves "the program is well-typed in a sequential
arithmetic sense". To prove "the program is well-typed under SIMT execution" you need three new
axes on every type-bearing thing in the AST:

1. **Ownership** (`Shrd` / `Uniq`) — already half there as `Type::Ref { mutable }`. Rename and
   commit to it.
2. **Memory space** (`CpuMem`, `GpuGlobal`, `GpuShared`, `GpuLocal`) — currently implicit. Every
   pointer/reference and every allocation site has to carry this.
3. **Execution resource** (`grid`, `block`, `warp`, `thread`, or an identifier bound by a
   parallel-for construct). Every typing-context entry must record which exec it was bound under.

Concretely you need:

### 3.1 Richer AST/type structures
- Replace `Type::Ref { mutable, inner }` with something like:
  ```rust
  Type::Ref { own: Ownership, mem: Memory, prv: Provenance, inner: Box<Type> }
  ```
- Add `Type::Array(Box<Type>, Nat)` (descend's `Nat` is a small symbolic-arithmetic enum — start
  with a `usize` literal + an `Ident(String)` variant; you only need evaluation later).
- Add `Type::At(Box<Type>, Memory)` so that an allocation expression can say "this `[f32; n]` lives
  in `gpu.global`".
- Add `Type::Atomic(AtomicTy)` so `atomicAdd`-style built-ins have a real signature.
- Add an `ExecTy` enum (start tiny: `CpuThread`, `GpuGrid`, `GpuBlock`, `GpuThread`).

### 3.2 A typing-context data module
- A module `type_checker/src/context.rs` (or `ctxs/`) holding `Place`, `PathElem`, `Loan`,
  `PrvMapping`, `Frame`, `FrameEntry`, `IdentTyped { ident, ty, mutbl, exec }`.
- These are **not** a second AST — they are auxiliary state the checker creates while walking the
  program AST. The program AST never holds them; codegen never reads them.
- Don't add this module until you actually need ≥3 of `{Place, Loan, Frame, IdentTyped}`. Until
  then, growing the existing `context.rs` is fine. The split is just basic hygiene — different
  lifetime, different audience — not a mandatory second IR.

### 3.3 Typed contexts
Split `Context` into:
- `GlobalCtx` — function signatures (kernel, device functions, built-ins).
- `KindCtx` — generic params bound at the kernel signature (`<n: nat>`, `<m: mem>`, `<r: prv>`).
- `TyCtx` — stack of frames; entries are bindings, exec mappings, or provenance mappings.
- `AccessCtx` — set of loans accumulated within the current expression, reset between sequenced
  expressions.

### 3.4 Borrow checking
- A `borrow_check.rs` with `access_safety_check(ctx, place_expr)` returning a `HashSet<Loan>`.
- It must implement at minimum:
  1. **Uniqueness within an exec**: only one `Uniq` loan per place per active exec; many `Shrd`
     allowed.
  2. **Narrowing**: a `Uniq` access from a thread-level exec into memory bound at grid level is
     fine; the reverse is not. (Mirror `narrowing_check` from descend.)
  3. **Aliasing across threads**: two distinct iterations of a `sched`/parallel-for cannot both
     hold a `Uniq` loan into the same array index.

### 3.5 A `sched`-equivalent construct
You currently have a `for i in (idx, n).step_by(stride)` loop in `vector_sum_k.rs`. That is
sequential per-thread. For real race detection you need a *parallel-for* the checker recognises
(call it `sched!` or treat the body of `#[kernel]` as the implicit `sched` over the launch grid).
Whatever syntax you pick, the checker must:
- Push a new `ExecExpr` onto the active exec stack.
- Re-type the bindings introduced inside as `exec = inner_thread`.
- Reject `&uniq` accesses that escape that exec.

### 3.6 A pre-declared built-ins table
- Move the 5 lines of `builtin_type()` into a `pre_decl.rs` keyed by name, returning `FnTy`-like
  signatures. Add `CudaVec::index`, `CudaVec::index_mut`, atomics, `syncthreads`, etc.
- `blockIdx / threadIdx / blockDim / gridDim` should remain values but their `Dim3` type should be
  refined to know which exec produced them — so reading `threadIdx.x` outside a kernel body fails.

### 3.7 Errors with provenance
- Replace stringly-typed `TypeError::TypeMismatch { expected: String, found: String }` with
  structured variants. Add `Span` to every AST node (descend uses `Span` from `parser/source.rs`;
  in our case it's `proc_macro2::Span` which is free since we're lowering from `syn`).

---

## 4. Suggested structural updates to the workspace

Today:

```
type_checker/
  src/
    ast.rs           ← Expr/Stmt/Function (surface)
    types.rs         ← Type, TypeError
    context.rs       ← flat HashMap context
    type_checker.rs  ← one big function
    lower.rs         ← syn::ItemFn → ast::Function
lib/                  ← proc-macros (cuda_module, kernel, spawn, …)
lib_core/             ← runtime (CudaVec, ffi, launch)
guard-rt/             ← (empty)
app/                  ← examples / benches
```

Proposed:

```
type_checker/
  src/
    lib.rs
    ast/
      mod.rs          ← Expr, Stmt, Function, Block, Pattern, Sched
      internal.rs     ← Place, PathElem, Frame, FrameEntry, Loan, PrvMapping
      span.rs         ← re-export proc_macro2::Span
    types/
      mod.rs          ← Ty, TyKind, DataTy, RefDty, Memory, Ownership, Provenance
      exec.rs         ← ExecTy, ExecExpr, BaseExec, ExecPathElem
      nat.rs          ← Nat (start: Lit(u64) | Ident(String) | BinOp)
    ctxs/
      mod.rs          ← re-exports
      global.rs       ← GlobalCtx (fn signatures, struct decls)
      kind.rs         ← KindCtx (generic params)
      ty.rs           ← TyCtx (frame stack)
      access.rs       ← AccessCtx (loan set)
    ty_check/
      mod.rs          ← entry point ty_check(&mut Function) -> Result<(), TyError>
      expr.rs         ← ty_check_expr dispatch
      borrow.rs       ← access_safety_check, narrowing_check, conflict_check
      pre_decl.rs     ← built-in function signatures
      error.rs        ← TyError with Span
    lower/
      mod.rs          ← syn::ItemFn → ast::Function (the proc-macro side)
guard-rt/             ← either delete or repurpose as "safe runtime wrappers"
                        used by the macros' expansion (Send/Sync newtypes)
```

Note that `ast/internal.rs` in the proposed layout is **not** a second program AST — it is the
typing-context data module (frames, loans, places). The program AST lives in `ast/mod.rs` alone.
This naming follows descend's convention and is a reasonable choice, but `ctxs/internal.rs` or
simply a larger `context.rs` would serve the same purpose.

Two other things worth noting:

- **Keep `lower` inside `type_checker`** rather than in `lib`. The macro crate (`lib/`) must stay
  thin because anything it touches becomes a `proc-macro` dependency. Today `cuda_module.rs`
  already does `use type_checker::{lower::lower_fn, …}`, which is the right direction.
- **Empty `guard-rt`** has the right name for the runtime safety guards — newtypes around raw
  pointers that statically encode which memory space they live in, fed back into the macro
  expansion. Worth keeping; not worth filling until the checker can actually emit calls to it.

---

## 5. Build plan

A linear roadmap that stays shippable at every step. Each phase ends with passing tests and a
demoable kernel.

### Phase 0 — clean-up groundwork (small, do first)
1. Move `lower.rs` tests back in and convert to `Span`-aware errors using `proc_macro2::Span`.
2. Introduce `Ty { kind: TyKind, span: Option<Span> }` and migrate the printer/tests. No new
   semantics yet — just preparing the shape.
3. Replace `Stmt::Let { name, value }` with `Stmt::Let { name, mutbl: Mutability, ty: Option<Ty>,
   value: Expr }` to capture `let` vs `let mut`.

### Phase 1 — memory & ownership on references
4. Add `Memory`, `Ownership`, `Provenance` enums in `types/`.
5. Rewrite `Type::Ref` to `Type::Ref(RefDty)` carrying `(own, mem, prv, inner)`.
6. Update `lower::lower_type` to read `&CudaVec<f32>` as `&shrd gpu.global CudaVec<f32>` and
   `&mut CudaVec<f32>` as `&uniq gpu.global CudaVec<f32>` by default — sufficient for the existing
   `add_vectors` kernel.
7. Update `type_check_expr` so `Index`/`Field` propagate the inner pointee type instead of
   erasing memory/ownership.

### Phase 2 — execution resources
8. Add `ExecTy`, `ExecExpr`, `BaseExec`, `ExecPathElem` (start with only `GpuGrid`, `GpuBlock`,
   `GpuThread` and the `ToThreads` path elem). No `Warp` / `TakeRange` yet — they aren't needed
   for the first kernel.
9. Tag every `Function` with its `exec: ExecExpr` (kernels = `GpuGrid` at the boundary, body
   types check under `GpuThread` because CUDA-style kernels are implicitly per-thread).
10. Extend `FrameEntry` to include `ExecMapping`; thread the active `ExecExpr` through
    `type_check_expr` so every binding is recorded with its `exec`.

### Phase 3 — split contexts
11. Implement `GlobalCtx`, `KindCtx`, `TyCtx` (frame stack), `AccessCtx`.
12. Migrate `type_check_expr` to take `&mut ExprTyCtx { gl, kind, ty, access, exec, … }` instead
    of `&mut Context`.
13. Move `builtin_type` into `pre_decl.rs` and grow it: `CudaVec::index`, `CudaVec::index_mut`,
    `blockIdx/threadIdx/blockDim/gridDim` with proper `Dim3` exec scoping, `__syncthreads`,
    `atomicAdd(&mut Atomic<u32>, u32) -> u32`.

### Phase 4 — the loan model & borrow checking
14. Add `internal::{Place, PathElem, Loan, PrvMapping}` and the methods on `TyCtx` to insert /
    extend / query loan sets.
15. Implement `borrow_check::access_safety_check` with `narrowing_check` +
    `access_conflict_check`. Initial scope: scalar places and array-index places, no `View` /
    `Sched` yet.
16. Wire `ty_check_assign_place`, `ty_check_idx_assign`, and `ty_check_borrow` to call
    `access_safety_check` and update `AccessCtx`.
17. **Validation kernel**: write a deliberately racy kernel
    (`result[0] = a[idx] + b[idx]` where the LHS index is a constant) and confirm the checker
    rejects it.

### Phase 5 — parallel constructs
18. Decide the syntax for parallelism inside `#[kernel]`. Two options to choose between:
    - Implicit: the entire `#[kernel] fn body` is already inside `sched x in grid to thread`,
      no explicit construct needed. Simplest, matches CUDA.
    - Explicit: add a `sched!` macro that mirrors descend's `sched g in groups to block in grid`.
      Closer to descend, lets you nest grid/block/thread cleanly.
19. Implement `ty_check_sched`: push exec, type-check body, pop exec, garbage-collect loans
    bound to popped frames.
20. **Validation kernel**: re-express vector_sum cleanly and confirm a two-thread aliasing bug is
    rejected.

### Phase 6 — sharpening
21. Add `Nat` evaluation (`NatCtx`) so index conflicts can be decided when both sides are
    constant or symbolically equal.
22. Add `Atomic<T>` and verify a histogram-style kernel (which would race without atomics)
    *only* type-checks when the accumulator is `Atomic<u32>`.
23. Surface real errors through `proc_macro2::Diagnostic` so the IDE underlines the offending
    expression instead of `panic!`-ing in the macro.

---

## 6. Concrete deltas to the existing files

Highest-leverage edits if you want to start moving today, in priority order:

1. `type_checker/src/types.rs` — add `Memory`, `Ownership`, `Provenance`, replace the `Ref`
   variant. Everything else cascades from this.
2. `type_checker/src/context.rs` — convert `Context` into a stack of `Frame`s. Even before
   splitting into four contexts, frame discipline is what lets you scope `exec` and loans
   correctly.
3. `type_checker/src/lower.rs` — `lower_type` currently silently drops mutability/memory.
   Make it surface that information; it's the only entry point translating Rust types into our
   type system.
4. `type_checker/src/type_checker.rs` — split into `expr.rs`, `borrow.rs`, `pre_decl.rs`. Even a
   purely mechanical split makes the next phase tractable.
5. `lib/src/cuda_module.rs` — once the checker can fail with a `Span`, replace the `println!` +
   `panic!` flow with `syn::Error::new(span, msg).to_compile_error()` so build failures look
   native.

---

## 7. What we deliberately do *not* copy from descend

- The PEG parser. We get a parse tree for free from `syn`. Lowering is enough.
- The full `Nat` symbolic arithmetic. Start with literals + idents; add what we need when we hit
  it.
- The kinding machinery for `Memory`/`Provenance` generics across function calls. Hard-code
  `gpu.global` everywhere until we have multiple memory spaces in real programs.
- The codegen pipeline. We already emit `.cu` via `helpers::gen_kernel` and let `nvcc` produce
  PTX; we don't need a CUDA AST printer of our own.

The smallest viable "data-race-free" version is: **ownership + memory + exec + frame stack +
narrowing check**. Everything else in descend is sharpening, and can be added once the core proof
holds.

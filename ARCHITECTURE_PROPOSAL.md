# Project Architecture Proposal — A Data-Race-Free CUDA DSL in Rust

> Status: proposal / refactor plan. Targets the current code in `type_checker/`,
> `lib/`, `lib_core/`, `guard-rt/`, `app/`.
>
> Goal: a DSL where the user writes a kernel in a higher-level Rust-like language,
> we **lower → type-check → data-race-check → codegen CUDA → compile (nvcc, at Rust
> compile time) → spawn at runtime**. The static safety layer follows Descend
> (places, loans, regions/provenance, frames, execution resources), but trimmed to
> a minimal core we can grow.

---

## 0. First, the "two ASTs" question

**Descend uses one AST, not two.** Type checking and data-race checking are the
**same pass** over the **same tree**, sharing contexts. What can look like a second
"structure" is really two things:

1. **Side contexts** carried *next to* the AST during checking:
   - `TyCtx` — a stack of **frames** (bindings) plus a `provenance → {loans}` map,
   - `AccessCtx` — the **loans** accumulated for the current access,
   - `KindCtx` — generic/region variables and their outlives relations,
   - the active **`ExecExpr`** — "who is running this" (grid/block/thread).

2. A **normalized view of place expressions** (`Place` + `PlaceCtx`). Place
   expressions are a *sub-grammar* of the AST (`x`, `*p`, `p[i]`, `p.f`, …). They get
   normalized into `(root_ident, path_of_projections)` so the borrow checker can
   compare them **syntactically** ("is this loan a prefix of that access?"). This is
   a derived index, not a separate tree.

So the design below is: **one IR, decorated and validated by successive passes, then
consumed by codegen.** That is both simpler than two trees and closer to what Descend
(and rustc's HIR + typeck side-tables) actually do.

---

## 1. The single most important refactor

Today there are effectively **two disconnected front-ends**:

| Path | Input | Output | Used for |
| --- | --- | --- | --- |
| `type_checker::lower` → `type_checker::type_check` | `syn` AST | your `ast::Expr` IR + a type | **prints result, then discarded** |
| `lib::helpers::Generator` | `syn` AST (again) | CUDA C string | **the actual codegen** |

`cuda_module_impl` runs the type checker for its side effects (a `println!`) and then
generates CUDA straight from `syn`, *ignoring* the checked IR. That means:

- the type checker can never *inform* codegen (no type-driven lowering),
- the data-race layer would have nowhere to plug in (codegen doesn't see places/loans),
- the two front-ends drift (they already support different expression sets).

**The refactor: one IR is the spine.** Everything hangs off it:

```
syn AST ──lower──► IR ──typeck──► IR(+types) ──borrowck──► IR(validated) ──codegen──► CUDA C
                                                                                      │
                                                                              nvcc -ptx (compile time)
                                                                                      │
                                                                          spawn!(...) at runtime
```

`lib/` (the proc-macro crate) becomes a thin shell that calls a `pipeline` function in
the compiler crate and writes/compiles the result. All the interesting logic moves into
a normal library crate that is **unit-testable without proc-macro machinery**.

---

## 2. Guiding principles

1. **One IR, many passes.** Each pass either *decorates* the IR (types, places, loans)
   or *rejects* it (errors). Passes never re-parse `syn`.
2. **Side-tables keyed by `NodeId`, not tree rewrites.** Don't build a new tree per
   pass. Give every IR node a stable `NodeId` and keep `HashMap<NodeId, _>` side-tables
   (types, place-normalizations, loans). This is how you add analyses later without
   touching the AST definition. (rustc does exactly this with HIR + `TypeckResults`.)
3. **Codegen consumes the IR, never `syn`.** The only place `syn` appears is `lower`.
4. **Minimal-but-shaped.** Define the *shape* of places/loans/regions/frames/exec now,
   even if the first implementation handles only the trivial cases. Each section below
   marks the MVP and the growth path so the types don't need breaking changes later.
5. **Errors are data.** A single `Diagnostic` type with spans, returned (not `panic!`),
   so the macro can emit real `compile_error!`s pointing at user code.

---

## 3. Crate layout

Keep the existing crates; repurpose `type_checker` into the real compiler and slim down
`lib`. (Renaming `type_checker` → `kc` / `compiler` is optional and cosmetic; the README
of this proposal calls it **the compiler crate**.)

```
app/         user-facing examples + the runtime-side API (CudaVec, spawn!) consumers
lib/         proc-macros ONLY: #[cuda_module] #[kernel] #[device_function] spawn!
             → thin: parse syn, call compiler::pipeline, write+nvcc, emit tokens
lib_core/    runtime: launch config, PTX loading, FFI, CudaVec device buffer
guard-rt/    runtime guard (CONTEXT_ACTIVE) so device fns can't run on the CPU
compiler/    (was type_checker) the whole frontend: lower, IR, typeck, borrowck, codegen
```

Why move codegen out of `lib/` into `compiler/`: proc-macro crates can't be unit-tested
normally and can't be depended on by test harnesses. A plain library crate can. You want
your type system and race checker to have a fat test suite.

---

## 4. Module layout inside `compiler/`

```
compiler/src/
  lib.rs            re-exports + pipeline entry
  pipeline.rs       orchestrates: lower → typeck → borrowck → codegen; collects diagnostics

  ast/
    mod.rs          NodeId, the IR enums, re-exports
    expr.rs         Expr, Stmt, Function, Param   (your current ast.rs, + NodeId)
    place.rs        PlaceExpr (sub-grammar) + Place/PlaceCtx (normalized)   ← NEW
    ty.rs           Type / DataTy and type-level kinds (your current types.rs, grown)

  lower/
    mod.rs
    expr.rs         syn::Expr  → ast::Expr      (your current lower.rs, split up)
    ty.rs           syn::Type  → ast::Type
    item.rs         syn::ItemFn → ast::Function

  typeck/
    mod.rs
    check.rs        type_check pass            (your current type_checker.rs)
    env.rs          type environment           (your current context.rs, becomes Frame-based)
    builtins.rs     blockIdx/threadIdx/... and intrinsics

  borrowck/         ← the data-race detector (the Descend layer)
    mod.rs
    place.rs        place normalization + prefix/overlap tests
    loan.rs         Loan, Ownership
    region.rs       Provenance/region, outlives relation
    frame.rs        Frame stack + flow-sensitive TyCtx, prv→loans map
    exec.rs         ExecResource (grid/block/thread), narrowing
    check.rs        the borrow/race-check pass: Γ ⊢ e ⊣ Γ'

  codegen/          ← moved out of lib/helpers.rs, now consumes the IR
    mod.rs
    cuda.rs         IR → CUDA C
    types.rs        Type → C type mapping (uint32_t, float, …)

  diagnostics.rs    Diagnostic { message, span, kind }, Span, error kinds
```

The dependency direction is strictly downward: `lower` → `ast`; `typeck`/`borrowck` →
`ast` + `diagnostics`; `codegen` → `ast` + typeck results; `pipeline` ties them; `lib`
depends only on `compiler::pipeline`.

---

## 5. The IR (`ast/`)

Keep your current enums; add a `NodeId` so passes can attach results without rewriting
the tree.

```rust
// ast/mod.rs
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

// every expression carries an id + (later) a span
#[derive(Debug, Clone)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind,
    // pub span: Span,   // add when you wire diagnostics to real syn spans
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    LiteralF32(f32),
    LiteralU64(u64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Assign { target: Box<Expr>, value: Box<Expr> },
    Index  { target: Box<Expr>, index: Box<Expr> },
    Field  { base: Box<Expr>, member: String },
    CudaVec(Box<Expr>),
    // GROWTH: Ref { own: Ownership, place: Box<Expr> }   // explicit borrow &/&uniq
    // GROWTH: Sched { ... } / Split { ... }              // parallel scheduling
    // GROWTH: If, For, Call, Deref
}
```

> **Why a wrapper struct instead of decorating the enum variants?** A single `id`/`span`
> on the wrapper means every pass attaches data by `NodeId` without each variant carrying
> optional fields. Today your `Expr` is a bare enum — adding the wrapper is the one
> upfront change that pays off for every later analysis.

**Side-tables (the decorations), produced by passes and consumed downstream:**

```rust
// typeck output
pub struct TypeckResults {
    pub node_types: HashMap<NodeId, Type>,   // type of every expr
    pub fn_ty: Type,                          // type of the kernel
}

// borrowck output (data-race layer)
pub struct BorrowckResults {
    pub places: HashMap<NodeId, Place>,       // normalized place for place-exprs
    pub loans:  HashMap<NodeId, Vec<Loan>>,   // loans created/required at a node
}
```

Codegen reads `TypeckResults` (e.g. to pick the right C type, or to know a `Var` is a
`CudaVec` pointer) instead of re-deriving types from `syn`.

> **Alternative considered:** a separate "typed AST" (`thir`) built by cloning the tree
> with `Type` baked into each node (this is closer to what the Descend repo does with its
> `ty: Option<Ty>` fields). It's fine, but for a project this size the side-table approach
> means you never write a second tree definition or a tree-to-tree transform. Recommend
> side-tables; revisit only if you start doing heavy IR rewriting (e.g. view desugaring).

---

## 6. The type system (`ast/ty.rs`, `typeck/`)

### MVP (basically what you have, lightly reorganized)

Your `Type` enum already has `F32/U64/U32/Bool/Unit/CudaVec/Dim3/Ref`. Keep it. The
`type_check` pass stays a fold returning `Result<Type, Diagnostic>`, and the environment
(`context.rs`) keeps mapping names → types.

The one structural change: **`Ref` should eventually carry the Descend trio** so the data
race layer has somewhere to read from. Stub the fields now, ignore them in the MVP type
checker, fill them in when borrowck lands:

```rust
// ast/ty.rs
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32, U64, U32, Bool, Unit,
    Dim3,
    CudaVec(Box<Type>),
    Ref(RefTy),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefTy {
    pub own: Ownership,          // Shrd | Uniq      (was: `mutable: bool`)
    pub mem: MemSpace,           // CpuMem | GpuGlobal | GpuShared | GpuLocal
    pub prv: Provenance,         // region/lifetime this ref lives in
    pub pointee: Box<Type>,
}
```

> `mutable: bool` → `own: Ownership` is the same information in the vocabulary the race
> checker needs (`&uniq` vs `&`). `mem`/`prv` can default to `GpuGlobal` / a fresh region
> during lowering until you actually enforce them.

### Evolution

- **Kinds.** Descend separates *data types* from *type-level naturals/memory/provenance*
  (the `Kind` system: `Nat`, `Memory`, `Provenance`, `DataTy`). Introduce a `DataTy` vs
  `Type` split only when you need generic array sizes (`[T; N]` with `N` a `Nat`). Until
  then, `CudaVec<T>` with a runtime length is enough.
- **Arrays with static size** (`[f32; 1024]`, shared memory `tmp`): add `Type::Array(Box<Type>, Nat)`.
  This is the gateway to checking `__shared__` buffers like the transpose example.
- **Atomics** (`AtomicU32`): a distinct data type so codegen emits `atomicAdd` and the race
  checker treats them as always-safe concurrent writes.
- **Subtyping / coercion:** currently `Add` requires identical operand types. When you add
  literals-as-polymorphic or `u32→u64` widening, put it behind an explicit `coerce(a, b)`
  helper rather than sprinkling matches.

---

## 7. The data-race layer (`borrowck/`) — the Descend core, minimized

This is the heart of the proposal. The point is to **define the five abstractions now**
(place, loan, region/provenance, frame, exec resource) with shapes that can grow, and
implement only the trivial behavior first. Each subsection: **what it is → minimal Rust →
how it grows.**

### 7.1 Places (`borrowck/place.rs`)

**What:** the unique syntactic name of a memory location. Used so two accesses can be
compared *syntactically* — overlap ⇒ potential conflict.

```rust
// the sub-grammar of Expr that names memory
pub enum PlaceExpr {
    Var(String),
    Deref(Box<PlaceExpr>),          // *p
    Field(Box<PlaceExpr>, String),  // p.f
    Index(Box<PlaceExpr>, Box<Expr>)// p[i]
    // GROWTH: Select(Box<PlaceExpr>, ExecResource) // p[[e]] — per-thread slice
    // GROWTH: View(Box<PlaceExpr>, View)           // reshape/reorder
}

// normalized form: a root + a path of projections (for syntactic comparison)
pub struct Place { pub root: String, pub path: Vec<Proj> }
pub enum Proj { Deref, Field(String), Index(IndexKey) }

// IndexKey is what makes the race check tractable: classify the index
pub enum IndexKey {
    Const(u64),          // a[0]
    ThreadLinear,        // a[global_thread_id]  → each thread distinct ⇒ safe
    Other(NodeId),       // anything we can't prove distinct ⇒ conservative
}
```

Key operation: `fn overlaps(a: &Place, b: &Place) -> bool` (prefix comparison, with
`Index(ThreadLinear)` vs `Index(ThreadLinear)` treated as **disjoint** because distinct
threads hit distinct elements).

**MVP:** handle `Var`, `Index` with `IndexKey` classification. That alone distinguishes
the safe `a[threadIdx.x] = …` from the racy `a[0] = …` / `a[const] = …` — i.e. a real,
if narrow, data-race check.

**Growth:** add `Select`/`View` (Descend §3.2) to support grouping, halving, transposed
access patterns. The normalization (`Place` + path) is the stable part; you're only adding
`Proj` variants.

### 7.2 Loans & ownership (`borrowck/loan.rs`)

**What:** a record that a place is currently borrowed, and how (shared vs unique). The
classic rule: **shared XOR unique** — many `&`, or exactly one `&uniq`, never both.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ownership { Shrd, Uniq }

pub struct Loan {
    pub place: Place,
    pub own: Ownership,
    // GROWTH: pub exec: ExecResource,  // which resource holds it (§7.5)
}

// conflict test
pub fn conflicts(a: &Loan, b: &Loan) -> bool {
    overlaps(&a.place, &b.place)
        && (a.own == Ownership::Uniq || b.own == Ownership::Uniq)
}
```

**MVP:** track loans on kernel parameters (the only borrowed things initially): a `&uniq`
output, several `&` inputs. Reject a `&uniq` access that overlaps any other live access.

**Growth:** loans gain an `exec` field so the checker can say "this `&uniq` from the grid
is fine to *share among blocks*, but not among *threads*" — that's the narrowing rule (7.5).

### 7.3 Regions / provenance (`borrowck/region.rs`)

**What:** the "lifetime" a reference belongs to. A provenance variable `'r` maps to the set
of loans created within it; outlives constraints (`'a: 'b`) order them. On the CPU this is
exactly Rust lifetimes; Descend keeps it unchanged.

```rust
pub enum Provenance {
    Var(String),     // abstract region 'r (from a signature)
    Fresh(u32),      // inferred at a borrow site
}

// lives in the context: which loans flow into each region
pub type PrvMap = HashMap<Provenance, HashSet<Loan>>;

// outlives relation 'a : 'b
pub struct Outlives(pub Provenance, pub Provenance);
```

**MVP:** one implicit region per kernel; every borrow joins it; no outlives reasoning. You
get correct *intra-kernel* checking without lifetime inference.

**Growth:** real region inference + an outlives graph when references start being *stored*
(in structs, returned, passed to device functions). This is the piece you can defer longest
because a flat kernel body rarely needs it.

### 7.4 Frames & flow-sensitive context (`borrowck/frame.rs`)

**What:** the typing context `Γ` — a **stack of frames**, each an *ordered* list of bindings.
"Flow-sensitive" means the judgement is `Γ ⊢ e ⊣ Γ'`: checking an expression can *change*
the context (a binding becomes "dead" after a move, a new loan is recorded). This is what
your current `Context` (a flat `HashMap`) will grow into.

```rust
pub struct Binding {
    pub name: String,
    pub ty: Type,
    pub mutbl: bool,
    pub state: BindingState,          // Live | Dead (moved-out)
    // GROWTH: pub exec: ExecResource, // the resource that owns this binding
}
pub enum BindingState { Live, Dead }

pub struct Frame(pub Vec<Binding>);

pub struct TyCtx {
    pub frames: Vec<Frame>,   // push on block/scope entry, pop on exit
    pub prv: PrvMap,          // provenance → loans (7.3)
}
```

> Your `context.rs` is a `HashMap<String, Type>`. The migration is: wrap entries in
> `Binding`, make it a `Vec` inside a `Frame`, and keep a `Vec<Frame>` stack. The
> `get`/`insert` API can stay; lookups walk frames top-down.

**MVP:** a single frame for the kernel params + body lets. No `Dead` tracking (no moves
yet). This is a near-drop-in upgrade of today's `Context`.

**Growth:** push/pop frames for nested scopes (`sched`/`split` bodies, `if`/`for` blocks),
`Dead` marking for move semantics, and the `exec` field for narrowing.

### 7.5 Execution resources (`borrowck/exec.rs`) — what makes it *GPU*

**What:** *who* is running the code — the grid → block → thread hierarchy as a first-class
value. This is Descend's key addition over Oxide. Ownership/borrowing is performed *by an
execution resource*; the **narrowing** rule says a more-specific resource (a thread) may
read a borrow created by a less-specific one (the grid), but not vice-versa, and a `&uniq`
held at grid level must be *narrowed* to a thread before a thread may write through it.

```rust
pub enum ExecResource {
    CpuThread,
    GpuGrid { blocks: Dim, threads: Dim },
    Forall(Box<ExecResource>, DimSel),   // sched: run over all sub-resources
    Split  { base: Box<ExecResource>, at: Nat, dim: DimSel, half: Half }, // split
}
pub enum DimSel { X, Y, Z }
pub enum Half { Fst, Snd }

// the narrowing test: can `active` legally access a loan made by `holder`?
pub fn narrowable(holder: &ExecResource, active: &ExecResource) -> bool {
    // active must be a sub-resource (more specific) of holder:
    // holder.path is a prefix of active.path
    todo!()
}
```

**MVP:** two levels — `GpuGrid` (kernel boundary, owns the params) and an implicit
per-thread resource for the body. Narrowing reduces to: "a write must happen at thread
level through a thread-distinct index." Combined with `IndexKey::ThreadLinear` (7.1) this
is enough to accept `out[tid] = a[tid] + b[tid]` and reject `out[0] = …`.

**Growth:** model `sched`/`split` so blocks and warps appear; then you can check the
transpose example's `__shared__` buffer (a block-level resource) and `__syncthreads()`
(must execute at block level). This is where the DSL surface syntax (`sched`, `split`)
and this enum co-evolve.

### 7.6 The pass itself (`borrowck/check.rs`)

A flow-sensitive fold mirroring `typeck`, threading `TyCtx` and an `AccessCtx`
(the loans for the *current* access, reset between sequenced statements):

```rust
pub fn race_check(
    func: &Function,
    tys: &TypeckResults,
    ctx: &mut TyCtx,
) -> Result<BorrowckResults, Diagnostic> {
    // for each statement: check_expr, then drain AccessCtx (sequence point)
    // at each place access:
    //   1. classify into a Place (7.1)
    //   2. narrowing_check against the active ExecResource (7.5)
    //   3. conflict-check the new Loan against live loans (7.2)
    //   4. record the Loan
}
```

This is intentionally the **same shape** as `type_check` so the two can later be **fused
into one traversal** (as Descend does) if you want to avoid two passes — but keeping them
separate first is easier to build and test.

---

## 8. Codegen (`codegen/`) — consume the IR

Move `lib/helpers.rs` into `compiler/src/codegen/cuda.rs` and change its input from
`syn::ItemFn` to `&Function` (the IR) + `&TypeckResults`. Concretely:

- `gen_kernel_arguments` reads `Param { name, ty }` from the IR instead of re-walking
  `syn::FnArg`. The C-type mapping uses `Type` (you already have the `Ref{own}` info, so
  `Uniq → T*`, `Shrd → const T*` falls out cleanly).
- `gen_expr` matches on `ExprKind` instead of `syn::Expr`. You lose nothing — every
  `syn` case you handle already has (or should have) an IR counterpart. Cases the IR
  doesn't model yet (`if`, `for`, `call`, `cast`) become IR growth items, *shared* by the
  type checker and codegen instead of living only in codegen.
- Codegen runs **only after** typeck and borrowck succeed, so it can `unwrap`/assume
  well-formed input and stay simple.

> Net effect: the expression set the user can write is defined in exactly one place (the
> IR + lower), and all three of {typeck, borrowck, codegen} agree on it by construction.

The proc-macro side (`lib/cuda_module.rs`) keeps the nvcc invocation and PTX path logic;
it just calls `compiler::pipeline::compile_kernel(func) -> Result<String /*cuda*/, Vec<Diagnostic>>`
and writes the string.

---

## 9. The pipeline (`compiler/src/pipeline.rs`)

```rust
pub struct CompiledKernel { pub cuda: String, pub name: String }

pub fn compile_kernel(item: &syn::ItemFn) -> Result<CompiledKernel, Vec<Diagnostic>> {
    let func = lower::item::lower_fn(item)?;          // syn → IR
    let tys  = typeck::check::type_check(&func)?;     // IR → types (decorate)
    let _br  = borrowck::check::race_check(&func, &tys)?; // data-race validation
    let cuda = codegen::cuda::gen(&func, &tys);       // IR → CUDA C
    Ok(CompiledKernel { cuda, name: func.name.clone() })
}
```

`lib/cuda_module.rs` becomes: for each `#[kernel]` fn → `compile_kernel` → on `Err`, emit
`compile_error!` with the diagnostics' spans → on `Ok`, write `generated_{name}.cu`, run
`nvcc -ptx`, keep emitting the runtime module. Device functions feed in as additional IR
`Function`s (lower them too, type-check them, make them callable from kernels).

---

## 10. Diagnostics (`compiler/src/diagnostics.rs`)

Replace the mix of `Result<_, String>` (lower), `TypeError` enum (typeck), and `panic!`
(codegen) with one type carrying a `proc_macro2::Span` so errors point at the user's source:

```rust
pub struct Diagnostic { pub msg: String, pub span: Option<Span>, pub kind: DiagKind }
pub enum DiagKind { Parse, Type, Race, Codegen }
```

Thread a `span` from `syn` nodes through `lower` onto each `Expr` (the `span` field stubbed
in §5). Even a single span on the offending node turns "type error printed to stdout at
build time" into a red squiggle in the user's editor — a large UX jump for a DSL.

---

## 11. Incremental migration plan

Ordered so the project stays compiling and the type checker stops being decorative as early
as possible.

1. **Wrap the IR with `NodeId` + span fields** (§5). Update `lower` to assign ids. No
   behavior change. *(small, mechanical)*
2. **Move codegen into the compiler crate and point it at the IR** (§8). This is the big
   win: codegen now flows from the checked tree. Delete the duplicate `syn`-walking.
   Anything the IR can't represent yet, add to the IR + lower (not to codegen alone).
3. **Make typeck results feed codegen** (`TypeckResults`, §5/§6). C types come from `Type`.
4. **Reshape `Context` → `TyCtx`/`Frame`** (§7.4) — drop-in, still a flat single frame.
5. **Introduce `Place`/`IndexKey` + a minimal `race_check`** that only classifies indices
   and rejects non-thread-distinct writes to `&uniq` params (§7.1, §7.6). First real,
   testable data-race check.
6. **Add `Ownership`/`MemSpace`/`Provenance` to `RefTy`** and a two-level `ExecResource`
   with `narrowable` (§6, §7.5). Now `&uniq gpu.global` is enforced.
7. **Grow surface syntax** (`Ref`, `sched`, `split`, `if`, `for`, arrays, shared memory)
   one construct at a time, each touching IR + lower + typeck + borrowck + codegen together.
8. **Span-based diagnostics + `compile_error!`** (§10) once the passes return `Diagnostic`.

Steps 1–3 are pure refactor (no new features) and already make the architecture correct.
Steps 4–6 are the minimal data-race core. Step 7 is open-ended feature growth on a stable
spine.

---

## 12. Testing strategy

- **Unit-test the compiler crate directly** (now possible because it's not the proc-macro
  crate): build `Function` IR by hand (your `type_checker.rs` tests already do this) and
  assert on `type_check`, `race_check`, and `codegen` outputs.
- **Golden CUDA tests:** IR (or a tiny DSL snippet) → expected `.cu` string. Catches codegen
  regressions cheaply.
- **Race-check tables:** a list of `(kernel_body, expect: Accept | Reject(reason))`,
  seeded from `knowledge/initial/DATA_RACE_EXAMPLES.md`. This is your correctness oracle for
  the whole safety layer — grow it alongside §7.
- **`trybuild`** for the proc-macro surface: assert that bad kernels produce the right
  `compile_error!`.

---

## 13. Open decisions (worth settling before step 5)

1. **Typed-AST vs side-tables.** This doc recommends side-tables keyed by `NodeId` (§5). If
   you anticipate heavy IR-to-IR rewriting (e.g. desugaring Descend *views* into index
   arithmetic), a dedicated typed/lowered IR (`THIR`/`MIR`-style) may pay off instead.
2. **Fused vs separate typeck/borrowck.** Separate is easier to build and test (recommended
   first); Descend fuses them into one `ty_check`. The §7.6 pass is shaped so fusing later
   is mechanical.
3. **How much region inference.** A single implicit kernel region (§7.3 MVP) covers flat
   kernels. Full provenance inference is only needed once references are stored/returned —
   defer until a feature demands it.
4. **Index distinctness proof.** `IndexKey::ThreadLinear` (§7.1) is a syntactic
   approximation ("the index is `threadIdx`-derived and bijective"). Deciding how much
   arithmetic to prove distinct (`tid`, `tid+c`, `tid*2`, `bid*blockDim+tid`) sets how many
   real kernels pass. Start with "linear in exactly one thread coordinate."

---

## Appendix — mapping to your `knowledge/` notes

| Abstraction here | Descend / Oxide source in `knowledge/` |
| --- | --- |
| `Place` / `PlaceExpr` (§7.1) | `oxide/place-oxide.md`, `descend/descend-from-oxide.md` §4 (select/views) |
| `Loan` / `Ownership` (§7.2) | `oxide/loan-oxide.md`, `oxide/ownership-oxide.md`, `oxide/ownership-safety-oxide.md` |
| `Provenance` / outlives (§7.3) | `oxide/region-oxide.md`, `oxide/outlives-oxide.md`, `initial/PROVENANCE_EXPLAINED.md` |
| `Frame` / flow typing (§7.4) | `oxide/frame-oxide.md`, `oxide/flow-typing-oxide.md` |
| `ExecResource` / narrowing (§7.5) | `descend/descend-from-oxide.md` §2–3, §5 |
| pipeline & contexts (§9, §0) | `initial/DESCEND_ARCHITECTURE.md`, `initial/DESCEND_ANALYSIS.md` |
| type system (§6) | `initial/TYPE_SYSTEM_EXPLAINED.md`, `initial/KERNEL_TYPE_ANALYSIS.md` |

---

# Part II — Worked examples & the data structures at every stage

This part takes **one kernel** and shows the *actual internal value* produced by each
pass, then shows the same machinery **rejecting** bad kernels, and finally gives a
**per-step print checkpoint** so each piece you implement has a visible "is it right?"
signal. The printed blocks are written the way a `{:#?}` / custom pretty-printer would
render the structs from Part I — they are what you should literally `println!` while
building each stage.

> Convention used below: `n<k>` is the `NodeId(k)` of an IR node. Exact numbering depends
> on your traversal order; the relationships are what matter.

---

## 14. The running kernel (the happy path)

```rust
#[cuda_module]
mod vectors {
    #[kernel]
    fn add_vectors(a: &CudaVec<f32>, b: &CudaVec<f32>, out: &mut CudaVec<f32>) {
        let i = threadIdx.x;     // i : u32, thread-derived
        out[i] = a[i] + b[i];
    }
}
```

Target CUDA (what codegen must eventually emit):

```c
extern "C" __global__ void add_vectors(const float *a, const float *b, float *out) {
    uint32_t i = threadIdx.x;
    out[i] = a[i] + b[i];
}
```

> Note on `u32` vs `u64`: CUDA thread indices are `unsigned int` (`u32`). The current
> `type_checker` hard-codes `U64` for `Index`/`CudaVec` size; this example assumes the
> small coercion flagged in §6 (let `Index` accept `U32 | U64`). Keep that in mind — it is
> step 0 of making this example type-check.

### Stage 1 — Lowered AST (`lower`, migration step 1)

`lower_fn` turns the `syn` tree into the IR. Every node gets a `NodeId`.

```text
Function {
  name: "add_vectors",
  params: [
    Param { name: "a",   ty: Ref { own: Shrd, mem: GpuGlobal, prv: Fresh(0), pointee: CudaVec(F32) } },
    Param { name: "b",   ty: Ref { own: Shrd, mem: GpuGlobal, prv: Fresh(1), pointee: CudaVec(F32) } },
    Param { name: "out", ty: Ref { own: Uniq, mem: GpuGlobal, prv: Fresh(2), pointee: CudaVec(F32) } },
  ],
  body: [
    Let {
      name: "i",
      value: Expr { id: n2, kind: Field {
                base:   Expr { id: n1, kind: Var("threadIdx") },
                member: "x" } },
    },
    Expr( Expr { id: n13, kind: Assign {
            target: Expr { id: n5, kind: Index {
                      target: Expr { id: n3,  kind: Var("out") },
                      index:  Expr { id: n4,  kind: Var("i") } } },
            value:  Expr { id: n12, kind: Add(
                      Expr { id: n8,  kind: Index {
                               target: Expr { id: n6,  kind: Var("a") },
                               index:  Expr { id: n7,  kind: Var("i") } } },
                      Expr { id: n11, kind: Index {
                               target: Expr { id: n9,  kind: Var("b") },
                               index:  Expr { id: n10, kind: Var("i") } } } ) } } } ),
  ],
}
```

**How you know it's right:** the tree round-trips — every `syn` node you support appears
exactly once, params carry the `Ref{own,...}` you expect (`out` is `Uniq`, `a`/`b` are
`Shrd`), and `mutable: bool` is gone (replaced by `own`). This is the *only* place `syn`
is read.

### Stage 2 — Type check (`typeck`, migration steps 3–4)

The pass folds the tree, writes one entry per node into `node_types`, and threads the
environment. Print the resulting `TypeckResults`:

```text
TypeckResults {
  node_types: {
    n1  Var("threadIdx")  : Dim3,
    n2  Field .x          : U32,        // Dim3.x  ⇒ U32   (binds i : U32)
    n3  Var("out")        : Ref{Uniq, CudaVec(F32)},
    n4  Var("i")          : U32,
    n5  Index out[i]      : F32,        // CudaVec(F32)[U32]  ⇒ F32
    n6  Var("a")          : Ref{Shrd, CudaVec(F32)},
    n7  Var("i")          : U32,
    n8  Index a[i]        : F32,
    n9  Var("b")          : Ref{Shrd, CudaVec(F32)},
    n10 Var("i")          : U32,
    n11 Index b[i]        : F32,
    n12 Add(a[i], b[i])   : F32,        // F32 + F32 ⇒ F32  ✓
    n13 Assign            : Unit,       // out[i] : F32  ==  rhs : F32  ✓
  },
  fn_ty: Unit,
}
```

**How you know it's right:** every node has a type, the `Add` matched `F32 + F32`, and the
`Assign` saw `lhs == rhs`. If you swap `b[i]` for `i`, `n12` becomes `Add(F32, U32)` and the
pass returns `Diagnostic{ kind: Type, msg: "F32 + U32", span: <n12> }` *instead of* a table
— that is your first red/green signal.

### Stage 3 — Frame / context after checking (`borrowck::frame`, migration step 4)

`Context` → `TyCtx`. After processing params + the `let`, the single frame holds:

```text
TyCtx {
  frames: [
    Frame([
      Binding { name: "a",   ty: Ref{Shrd, CudaVec(F32)}, mutbl: false, state: Live, exec: Grid },
      Binding { name: "b",   ty: Ref{Shrd, CudaVec(F32)}, mutbl: false, state: Live, exec: Grid },
      Binding { name: "out", ty: Ref{Uniq, CudaVec(F32)}, mutbl: true,  state: Live, exec: Grid },
      Binding { name: "i",   ty: U32, mutbl: false, state: Live, exec: Thread,
                idx_class: ThreadLinear },   // ← KEY: i is tainted "thread-derived"
    ]),
  ],
  prv: { /* filled in Stage 6 */ },
}
```

> **The crucial bit is `idx_class: ThreadLinear` on `i`.** When the pass checked
> `let i = threadIdx.x`, it recognized the RHS as a thread coordinate and recorded that `i`
> is *thread-linear*. This taint is what lets Stage 5 classify `out[i]` as thread-distinct.
> Concretely: keep an `idx_env: HashMap<String, IndexKey>` next to the frame; `threadIdx.x`
> seeds `i ↦ ThreadLinear`; any later `let j = i + 1` would propagate to `Other`.

**How you know it's right:** `out`/`a`/`b` are owned by `Grid` (params), `i` lives at
`Thread`, and `i` carries `ThreadLinear`. Print the frame after each statement; the taint
appearing on `i` is the signal.

### Stage 4 — Places (`borrowck::place`, migration step 5)

For each **place expression** (memory-naming node), normalize to `Place`. The index vars
are resolved through `idx_env`:

```text
normalize(n5  out[i]) = Place { root: "out", path: [ Index(ThreadLinear) ] }
normalize(n8  a[i])   = Place { root: "a",   path: [ Index(ThreadLinear) ] }
normalize(n11 b[i])   = Place { root: "b",   path: [ Index(ThreadLinear) ] }
```

Overlap tests (the syntactic comparison the whole race check rests on):

```text
overlaps(out[ThreadLinear], a[ThreadLinear]) = false   // different roots
overlaps(out[ThreadLinear], out[ThreadLinear]) = false  // SAME place, but ThreadLinear
                                                        // ⇒ distinct threads ⇒ disjoint
```

**How you know it's right:** `i` resolved to `ThreadLinear` (not `Other`), and two
`out[ThreadLinear]` places report **disjoint** — that single rule is the reason "every
thread writes its own element" is safe. Print each `(NodeId → Place)` and the result of the
two `overlaps` calls above.

### Stage 5 — Loans (`borrowck::loan`, migration step 5)

Evaluating the statement `out[i] = a[i] + b[i]` produces, at this sequence point, an access
set (the `AccessCtx`):

```text
AccessCtx (loans for the current statement) = [
  Loan { place: a[ThreadLinear],   own: Shrd },   // read  (rhs)
  Loan { place: b[ThreadLinear],   own: Shrd },   // read  (rhs)
  Loan { place: out[ThreadLinear], own: Uniq },   // write (lhs)
]
```

Pairwise `conflicts` (overlap && some-uniq):

```text
conflicts(a,Shrd ; b,Shrd)     = false   // different roots
conflicts(out,Uniq ; a,Shrd)   = false   // different roots
conflicts(out,Uniq ; b,Shrd)   = false   // different roots
```

**How you know it's right:** no pair conflicts. If you (incorrectly) gave `a` and `out` the
same provenance/root, `conflicts(out,Uniq ; a,Shrd)` would flip to `true`. Print the loan
set + the conflict matrix.

### Stage 6 — Regions / provenance (`borrowck::region`, migration step 6)

Each `Loan` is filed under the region of the reference it came through. MVP keeps the
per-param `Fresh(k)` regions from lowering (you could collapse to one):

```text
prv (Provenance → {Loan}) = {
  Fresh(0) /* a   */ : { Loan{ a[ThreadLinear],   Shrd } },
  Fresh(1) /* b   */ : { Loan{ b[ThreadLinear],   Shrd } },
  Fresh(2) /* out */ : { Loan{ out[ThreadLinear], Uniq } },
}
```

**How you know it's right:** the `Uniq` loan sits alone in `out`'s region; the `Shrd` loans
sit in distinct regions. (Region reasoning only starts *doing work* once references are
stored or returned — §7.3 — so for a flat kernel this print is mostly a bookkeeping check.)

### Stage 7 — Execution resource & narrowing (`borrowck::exec`, migration step 6)

```text
active_exec at kernel entry = GpuGrid { blocks: <dyn>, threads: <dyn> }
active_exec inside body      = Thread(of grid)      // MVP: body runs per-thread implicitly
```

At the write `out[i] = …` the checker asks **narrowing**: `out` is owned by `Grid`, written
by `Thread`.

```text
narrowable(holder = Grid, active = Thread) = true     // Thread is a sub-resource of Grid
```

**How you know it's right:** `narrowable(Grid, Thread) == true`, and the reverse
`narrowable(Thread, Grid) == false`. Print the active exec at each statement and the
narrowing result at each access.

### Stage 8 — The data-race decision (`borrowck::check`, migration steps 5–6)

The MVP rule, stated exactly, combining Stages 4–7:

```text
for each Loan L in AccessCtx, made at active_exec = Thread:
  (R1 narrowing)   require narrowable(owner_exec(L.place), active_exec)        else REJECT
  (R2 distinct)    if L.own == Uniq:
                      require L.place is thread-distinct under active_exec
                      i.e. L.place.path contains Index(ThreadLinear)           else REJECT
  (R3 conflict)    for each other live Loan L': if conflicts(L, L')            then REJECT
accept
```

Run for `add_vectors`:

```text
[race] out[ThreadLinear] Uniq  : R1 narrow(Grid→Thread)=ok  R2 thread-distinct=ok  R3 no-conflict=ok
[race] a[ThreadLinear]   Shrd  : R1 ok  R2 n/a(shrd)  R3 ok
[race] b[ThreadLinear]   Shrd  : R1 ok  R2 n/a(shrd)  R3 ok
[race] RESULT: ACCEPT  ✓  → kernel is data-race free, proceeding to codegen
```

**How you know it's right:** `ACCEPT`, and codegen then emits the CUDA in §14.

---

## 15. The same machinery rejecting bad kernels

Each case below changes **one line** of the running kernel and shows **which stage fires**
and **what prints**. This is exactly your regression oracle (§12 "Race-check tables").

### 15.1 Constant index — parallel write race (caught at Stage 8, rule R2)

```rust
out[0] = a[i] + b[i];       // every thread writes element 0
```

```text
normalize(out[0]) = Place { root: "out", path: [ Index(Const(0)) ] }
AccessCtx = [ Loan{out[Const(0)], Uniq}, Loan{a[ThreadLinear], Shrd}, Loan{b[ThreadLinear], Shrd} ]

[race] out[Const(0)] Uniq : R1 narrow ok ; R2 thread-distinct? path has no Index(ThreadLinear) → FAIL
[race] RESULT: REJECT
Diagnostic {
  kind: Race,
  span: <n13 / the assignment>,
  msg: "data race: `out[0]` is written by a unique borrow at thread level, but the index \
        is not thread-distinct — every thread writes the same element",
}
```

### 15.2 Non-distinct derived index — conservative race (Stage 8, rule R2)

```rust
let i = threadIdx.x;
out[i / 2] = a[i] + b[i];    // threads 2k and 2k+1 collide
```

```text
idx_env: i ↦ ThreadLinear, but (i / 2) is not a bare thread coord ⇒ classify as Other(n_div)
normalize(out[i/2]) = Place { root: "out", path: [ Index(Other(n_div)) ] }

[race] out[Other] Uniq : R2 thread-distinct? Other ≠ ThreadLinear → FAIL
[race] RESULT: REJECT
Diagnostic {
  kind: Race,
  msg: "data race: cannot prove `out[i / 2]` selects a distinct element per thread \
        (index is not thread-linear); refusing to allow the unique write",
}
```

> This is the conservative case from open-decision §13.4: until the index analysis is
> smarter, anything that isn't provably bijective-in-one-thread-coord is `Other` and a
> unique write through it is rejected. Tightening `IndexKey` classification later turns some
> of these from REJECT into ACCEPT *without touching any other stage*.

### 15.3 Type error in the body (caught earlier, at Stage 2)

```rust
out[i] = a[i] + i;          // F32 + U32
```

```text
[typeck] n12 Add : checking F32 + U32
[typeck] RESULT: type error (pass aborts before places/loans ever run)
Diagnostic {
  kind: Type,
  span: <n12>,
  msg: "type mismatch in `+`: expected `F32 + F32`, found `F32 + U32`",
}
```

This shows the **pass ordering** doing its job: typeck rejects before borrowck, so the race
checker only ever sees well-typed trees.

### 15.4 Loan conflict — shared XOR unique (Stage 8, rule R3) *(growth tier)*

Once explicit borrows of locals exist (`Ref`/`Deref` in the IR — §5 growth), this is the
classic aliasing rejection:

```rust
let r = &uniq out;          // unique borrow of out
let s = &out;               // shared borrow of the same place — overlaps r
s[i] = r[i];                // both loans live at the same sequence point
```

```text
AccessCtx = [ Loan{out, Uniq /*via r*/}, Loan{out, Shrd /*via s*/} ]
conflicts(out,Uniq ; out,Shrd) = overlaps(out,out)=true && (Uniq||..)=true → true

[race] RESULT: REJECT
Diagnostic {
  kind: Race,
  msg: "cannot borrow `out` as shared because it is also borrowed as unique",
}
```

This case needs nothing new in the *checker* — only the IR/lower support for `&`/`&uniq`.
It's listed here to show the loan machinery you build in step 5 already covers it.

---

## 16. Valid / invalid catalogue

| # | Kernel body (given `let i = threadIdx.x;`) | Verdict | Caught by | Why |
| --- | --- | --- | --- | --- |
| V1 | `out[i] = a[i] + b[i];` | ✅ accept | — | thread-distinct unique write, shared reads |
| V2 | `out[i] = a[i] * 2.0 + b[i];` | ✅ accept | — | same shape; literal `2.0 : F32` |
| V3 | `let t = a[i]; out[i] = t + b[i];` | ✅ accept | — | `t : F32` local, no extra borrow conflict |
| I1 | `out[0] = a[i] + b[i];` | ❌ reject | Stage 8 / R2 | const index ⇒ all threads alias `out[0]` |
| I2 | `out[i / 2] = a[i] + b[i];` | ❌ reject | Stage 8 / R2 | index not provably thread-distinct (`Other`) |
| I3 | `out[i] = a[i] + i;` | ❌ reject | Stage 2 | `F32 + U32` type mismatch |
| I4 | `out[i] = c[i] + b[i];` (`c` undeclared) | ❌ reject | Stage 2 | unknown variable `c` |
| I5 | `let r = &uniq out; let s = &out; s[i]=r[i];` | ❌ reject | Stage 8 / R3 | shared-XOR-unique violated *(growth)* |
| I6 | `out[i] = a[j];` (`j` block-shared, not thread-distinct) | ❌ reject | Stage 8 / R2 | read pattern not thread-distinct *(growth, needs `sched`)* |

Seed `compiler/tests/race_table.rs` with these rows: each is `(body, Accept | Reject(stage))`.
When you implement a stage, the rows it owns flip from "not yet checked" to the right verdict.

---

## 17. Incremental checkpoints — what should print when each piece works

This maps every migration step (§11) to a concrete `println!` and the pass/fail signal, so
**no step is invisible**. Build top-to-bottom; each checkpoint should light up before moving on.

| Step | You implemented | Print this | "It's right" looks like |
| --- | --- | --- | --- |
| 1 | `NodeId` wrapper + `lower` assigns ids | `println!("{:#?}", func)` | the Stage 1 tree, every node has a distinct `id` |
| 2 | codegen reads IR (not `syn`) | the generated `.cu` string | byte-identical to §14's CUDA |
| 3 | `TypeckResults.node_types` | `println!("{:#?}", tys.node_types)` | the Stage 2 table; mismatch ⇒ `Diagnostic{Type}` |
| 4 | `Context` → `TyCtx`/`Frame` + `idx_env` | `println!("{:#?}", ctx)` after each stmt | Stage 3 frame; `i` shows `idx_class: ThreadLinear` |
| 5a | `Place` normalization | `println!("{} → {:?}", node, place)` | Stage 4 places; `out[i] → out[ThreadLinear]` |
| 5b | `overlaps` | print the 2 calls in Stage 4 | `out[TL]` vs `out[TL]` ⇒ **false** (disjoint) |
| 5c | `Loan` + `AccessCtx` | `println!("{:#?}", access_ctx)` | Stage 5 loan set per statement |
| 5d | `conflicts` + rule R2/R3 | the `[race] …` lines | V1 ⇒ `ACCEPT`; I1/I2 ⇒ `REJECT` with reason |
| 6a | `RefTy{own,mem,prv}` + `prv` map | `println!("{:#?}", ctx.prv)` | Stage 6 region map |
| 6b | `ExecResource` + `narrowable` | the `[exec] …` lines | `narrowable(Grid,Thread)=true`, reverse `false` |
| 7 | new surface syntax (`if`/`for`/`sched`…) | re-print Stages 1–8 for the new kernel | each new node flows through all stages |
| 8 | `Diagnostic` + `compile_error!` | the macro emits a real compiler error | I1–I4 fail the *build* with a span, not stdout |

### A single debug switch

Wire one env var so you can watch the whole pipeline while building it:

```rust
// pipeline.rs
fn dbg_stage(name: &str, val: &impl std::fmt::Debug) {
    if std::env::var("DSL_DEBUG").is_ok() {
        eprintln!("\n=== [{name}] ===\n{val:#?}");
    }
}

pub fn compile_kernel(item: &syn::ItemFn) -> Result<CompiledKernel, Vec<Diagnostic>> {
    let func = lower::item::lower_fn(item)?;          dbg_stage("1 lower", &func);
    let tys  = typeck::check::type_check(&func)?;     dbg_stage("2 typeck", &tys);
    let br   = borrowck::check::race_check(&func, &tys)?;  dbg_stage("3 borrowck", &br);
    let cuda = codegen::cuda::gen(&func, &tys);       dbg_stage("4 codegen", &cuda);
    Ok(CompiledKernel { cuda, name: func.name.clone() })
}
```

Running `DSL_DEBUG=1 cargo build` then prints, in order, every structure in §§14 Stage 1–8
for each kernel in your `app/` — which is exactly the incremental visibility you asked for:
implement a stage, rebuild, watch its block appear and match the reference above.

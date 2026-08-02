# Evolve the type checker to validate all benchmarks (+ 2 new gpotion ports)

## Context

The `rust-metaprogramming` project compiles a Rust-embedded GPU DSL to CUDA: `syn → lower (IR) → type_check → codegen (syn walk) → nvcc`. Today the type checker is not load-bearing and cannot even validate the one live benchmark:

- **Params are never inserted into `Context`** — every kernel body referencing `a`, `result`, `n` fails with `UnknownVariable`.
- **`Stmt::Let` discards the type annotation** (`ast.rs` has no `ty` field), so `let y: u64 = 12.0` type-checks the wrong way (caught only later, or not at all).
- **`Index`/`Assign` don't see through `Ref`** — params are `Type::Ref{inner: CudaVec}` but the Index rule matches bare `CudaVec`.
- **`result[idx] = …` is rejected** (`InvalidAssignmentTarget`: only `Var` targets allowed), so even `vector_sum` fails.
- The IR models only `Add/Mul/Assign/Index/Field/Var/2 literals`, but the benchmarks (mm, julia, raytracer, ripple) use `if`, `for`, sub, div, comparisons, `&&`, casts, `mut`, unary neg, and `sqrtf/cosf/floorf` — those exist only in the syn-based codegen.
- Type errors are `eprintln!`'d by `#[cuda_module]` and the build continues.

Goal (user-confirmed scope): grow AST/lower/type_checker so that **all existing benchmarks type-check**, port **nearest_neighbor and nbodies from gpotion** as new benchmarks, make type errors **fail the build** (`compile_error!`), fix incorrect existing rules, and add **unit tests**. Codegen keeps walking `syn` unchanged (IR-driven codegen is explicitly out of scope — ARCHITECTURE_PROPOSAL steps 2–3 come later).

## Design decisions

- **Consolidate binary ops**: replace `ExprKind::Add/Mul` with `Binary { op: BinOp, lhs, rhs }`, `BinOp = Add|Sub|Mul|Div|Lt|Le|Gt|Ge|Eq|Ne|And|Or`. Avoids 12 copies of the same match. Codegen is unaffected (walks syn).
- **Integer literals are polymorphic**: replace `LiteralU64(u64)` with `LiteralInt(u64)`. An int literal checks against an expected type (`U32`/`U64`) when one is known (let annotation, other operand), defaults to `U64`. Suffixed literals (`0u64`, `1u32`) lower to a fixed type.
- **Widening coercion `U32 → U64`** in binary ops, let-init, index positions, and assignments (one direction only). Needed because `blockIdx.x` etc. are `U32` (CUDA reality) but the DSL uses `u64` locals — e.g. `let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x`. (Matches ARCHITECTURE_PROPOSAL §6 `coerce()` note.)
- **Auto-deref for `Ref`**: indexing/assignment through `&CudaVec<T>` / `&mut CudaVec<T>` unwraps the ref; writes through an index require `Ref{mutable: true}` (or an owned `CudaVec`). Reads are allowed through both.
- **Mutability tracked per binding**: `let mut` recorded; assigning to a non-`mut` local or writing through a shared ref is a new `TypeError::NotMutable`. Remove the current implicit-declaration-on-assign behavior (benchmarks always declare first).
- **Scoped context**: `Context` becomes a frame stack (`push_scope`/`pop_scope`) so locals declared inside `if`/`for` blocks don't leak (julia/raytracer declare inside blocks). This is proposal §11 step 4's "single flat frame first" grown minimally — no ownership/provenance yet.

## Changes by file

### 1. `compiler/src/ast.rs`
- `Stmt::Let { name, ty: Type, mutable: bool, value }` (lower already requires the annotation).
- New `Stmt` variants: `If { cond: Expr, then_block: Vec<Stmt>, else_block: Option<Vec<Stmt>> }`, `For { var: String, start: Expr, end: Expr, step: Expr, body: Vec<Stmt> }` (matches the `(start, end).step_by(step)` surface form).
- `ExprKind`: replace `Add`/`Mul`/`LiteralU64` per design above; add `Unary { op: UnOp /*Neg*/, expr }`, `Cast { expr, ty: Type }`, `Call { func: String, args: Vec<Expr> }`.

### 2. `compiler/src/types.rs`
- New `TypeError` variants: `NotMutable(String)`, `UnknownFunction(String)`, `ArityMismatch { func, expected, found }`, `ConditionNotBool(String)`, `InvalidCast { from, to }`, `LetTypeMismatch { name, expected, found }`. Keep the existing ones.
- `Type` unchanged (no `Array`/memory spaces in this pass).

### 3. `compiler/src/context.rs`
- `Context { frames: Vec<HashMap<String, Binding>> }`, `Binding { ty: Type, mutable: bool }`.
- API: `insert(name, ty, mutable)`, `get(&name) -> Option<&Binding>` (lookup walks frames innermost-out), `push_scope()`, `pop_scope()`. Update existing callers.

### 4. `compiler/src/lower.rs`
- Lower the remaining syn constructs the benchmarks use: `Sub/Div`, all comparisons, `&&`/`||`, unary `-`, `as` casts (`ExprCast`), `if`/`else` (`ExprIf` as `Stmt::If`), `for i in (a, b).step_by(s)` (`ExprForLoop` → `Stmt::For`), calls (`ExprCall` with a plain path → `ExprKind::Call`), `let mut`, int-literal suffixes (`0u64`/`0u32` → fixed type; unsuffixed → polymorphic `LiteralInt`).
- Capture the let annotation into `Stmt::Let.ty` via existing `lower_type`.
- Return proper `Err` (no `eprintln!` + generic error) with the offending construct named.

### 5. `compiler/src/type_checker.rs` — the core
- **At entry**: push a frame and insert every `func.params` binding (`Ref{mutable}` params recorded as immutable bindings whose pointee mutability lives in the `Ref` type; scalar params immutable).
- Rules (existing ones fixed, new ones added):
  - `Let`: check `value` against annotation with expected-type propagation for `LiteralInt` + `U32→U64` widening; mismatch → `LetTypeMismatch`. Insert with `mutable`.
  - `Binary` arithmetic (`Add/Sub/Mul/Div`): both operands same numeric type after literal-inference/widening → that type.
  - Comparisons: same numeric type (after widening) → `Bool`. `And/Or`: `Bool × Bool → Bool`.
  - `Unary::Neg`: `F32 → F32` (int neg rejected — DSL has no signed ints).
  - `If`: cond must be `Bool` (`ConditionNotBool`); each block checked in its own scope; type is `Unit`.
  - `For`: `start`/`end`/`step` must be `U64` (widened); loop var bound as immutable `U64` in the body scope; type `Unit`.
  - `Cast`: numeric→numeric only (`F32/U32/U64`), else `InvalidCast`.
  - `Call`: builtin table `sqrtf/cosf/sinf/floorf: (F32) -> F32` — arity + arg types checked; unknown name → `UnknownFunction`.
  - `Index` (fixed): target auto-derefs `Ref` to `CudaVec(T)`; index `U32` or `U64` → `T`.
  - `Assign` (fixed): target may be `Var` (must exist + be `mut`, value must match binding type — no implicit declaration) **or** `Index` (base must be writable: `Ref{mutable: true, CudaVec}` or owned `CudaVec`; element type must match value).
  - `Field`: keep `Dim3.{x,y,z} → U32`; keep builtins (`blockIdx` etc. → `Dim3`, `warpSize → U32`).

### 6. Pipeline + macro: fail the build
- `compiler/src/pipeline.rs`: stop `.unwrap()`ing `lower_fn` — map lowering errors into `Diagnostic { kind: Parse }`; skip codegen when diagnostics exist.
- `macros/src/cuda_module.rs`: on `Err(diags)`, emit `compile_error!("<kernel name>: <all messages joined>")` tokens instead of `eprintln!` + continue (keep the eprintln for DEBUG_ACTIVE if desired).

### 7. Benchmarks (`examples/src/benchmarks/`)
- **Re-enable** `mm`, `julia`, `raytracer`, `ripple` in `mod.rs` and `main.rs` (they must now pass the load-bearing checker; fix any kernel that legitimately trips a new rule, e.g. missing `mut`).
- **Delete `tc.rs`** (deliberately ill-typed; it would now break the build) — its case becomes a negative unit test.
- **New `nearest_neighbor.rs`** (port of gpotion `benchmarks/nearest_neighbor.ex`): `euclid` kernel — 2D grid `globalId`, guard `if global_id < num_records`, `sqrtf` distance into `d_distances`. Host `run()` generates random lat/lng pairs, validates against a CPU loop (follow `vector_sum.rs` run/validate pattern, spawn via `spawn!`).
- **New `nbodies.rs`** (port of gpotion `benchmarks/nbodies.ex`): two kernels in one `#[cuda_module]` — `gpu_n_bodies` (O(n²) force loop: `for j in (0u64, n).step_by(1u64)`, `1.0f32 / sqrtf(dist_sqr)`, reads+writes `p: &mut CudaVec<f32>`) and `gpu_integrate` (position update). Host `run()` iterates a few timesteps and validates against a CPU reference.
- Reference CUDA for perf comparison optionally added under `benchmarks/cuda/` (copy from gpotion's `benchmarks/cuda/nearest_neighbor.cu`, `nbodies.cu`) — optional, Makefile already has the pattern.

### 8. Unit tests
- `compiler/src/type_checker.rs` `#[cfg(test)]` module (replace the commented-out one, updated to the `Expr{id,kind}`/new-AST shape; add a small `expr(kind)` builder helper for `NodeId`s):
  - Positive: params in scope; let with matching annotation; `U32→U64` widening (`let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x`); index read through `&CudaVec`; index-assign through `&mut CudaVec`; `mut` var reassign; if with bool cond; for-loop var usable as index; casts; `sqrtf` call; comparisons → Bool; polymorphic int literal against `u32`.
  - Negative: `let y: u64 = 12.0` (the tc.rs case) → `LetTypeMismatch`; assign to non-`mut` → `NotMutable`; index-write through `&CudaVec` (shared) → `NotMutable`; `if 1 {}` → `ConditionNotBool`; `F32 + U64` mismatch; unknown var; unknown function; wrong arity; `U64 → Bool` cast; assign to undeclared var (implicit-decl removed); block-scoped local not visible after the block.
- `compiler/tests/benchmarks_typecheck.rs` (integration, GPU-free): for each benchmark kernel, embed its source as `syn::parse_quote!` `ItemFn`, run `lower_fn` + `type_check`, assert `Ok` — covering vector_sum, mm, julia, raytracer, ripple, euclid, gpu_n_bodies, gpu_integrate; plus the tc.rs body asserting the exact error. This pins "all benchmarks type-check" as a regression suite. (Requires `lower_fn`/`type_check` to stay `pub` — they are.)

## Verification

1. `cargo test -p compiler` — all unit + integration tests pass; **no GPU/nvcc needed** (checker is pure).
2. `cargo build` in the workspace — `#[cuda_module]` now runs lower→typecheck→codegen→nvcc for all 7 kernels (nvcc 12.0 is present at `/usr/bin/nvcc`); a build success proves every benchmark passes the load-bearing checker.
3. Negative build check: temporarily re-introduce a `tc.rs`-style kernel and confirm `cargo build` fails with the `compile_error!` message, then remove it (the same case is permanently covered by the unit test).
4. If a GPU is available: `cargo run -p examples` (or `make -C examples run`) executes each benchmark's `run()` with its CPU validation, including the two new ports.

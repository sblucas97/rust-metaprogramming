# Building a basic type checker — *"the smallest set of Oxide rules you can actually run"*

The other docs explain *what* Oxide is. This one is a **build plan**: the minimal subset of
constructs to implement first, the **typing rule** for each one (transcribed/adapted from
[`oxide-the-essence-of-rust.pdf`](oxide-the-essence-of-rust.pdf), Fig. 1 syntax + Fig. 4–5 rules),
and a **concrete algorithm** for turning those rules into code. It is the bridge between the theory
in [[flow-typing-oxide|flow-sensitive typing]] / [[ownership-safety-oxide|ownership safety]] and a
`check(expr) -> Result<Ty, …>` function you can compile.

> **Scope.** Oxide deliberately leaves out **type inference** — every binding in the paper is
> annotated (`let x : τˢⁱ = e`), so the *semantics* of borrowing isn't tangled with unification
> (paper §7, [`OXIDE_ANALYSIS.md`](OXIDE_ANALYSIS.md) §7). A Rust-like DSL usually *wants*
> inference, so treat that as a **later layer** bolted on top of the checker described here — get
> the flow-sensitive skeleton working with annotations first.

---

## 0. The one decision that shapes everything: `Γ` in, `Γ′` out

Before any single rule, internalise the judgment shape (this is the whole architecture):

```text
Σ ; Δ ; Θ ; Γ  ⊢  e : τ  ⇒  Γ′
```

The piece that makes Oxide *Oxide* is the **`⇒ Γ′`**: checking an expression returns not just a
type but an **updated environment** (see [[flow-typing-oxide|flow-typing]]). Moves, borrows, and
drops are all *edits* to `Γ`. **In code, this becomes a `&mut Env`** — you thread one mutable
environment through the recursion, and "returning `Γ′`" is just "having mutated `env` by the time
you return." Get this right and the rest is mechanical.

For the **first milestone** you can ignore `Σ` (globals), `Δ` (type/region vars), and `Θ`
(temporaries) entirely, and shrink `Γ` to *"a stack of `name → type` bindings"* with a `†` dead
marker. Loans and [[region-oxide|regions]] are the **second** layer.

---

## 1. The build order

Implement in this sequence — each step exercises the machinery the next one needs:

| # | Construct | What it forces you to build | Rule |
| - | --------- | --------------------------- | ---- |
| **1** | **Variables + `let`** | the `&mut Env` threading + copy/move + `†` dead tracking | `T-Move`, `T-Copy`, `T-Let` |
| **2** | **`if`** | environment **snapshot + join** (`⊎`) | `T-Branch` |
| **3** | **Indexing** `p[e]` | [[place-oxide\|place]] expressions; whole-array ownership | `T-Index` (adapted) |
| **4** | **`for`** | the **loop-invariance** check (body leaves `Γ` as it found it) | `T-For` (adapted) |

**Milestone A** = steps 1–2 with *no loans at all*, just move tracking. **Milestone B** = add
[[loan-oxide|loans]] / [[region-oxide|regions]] and the [[ownership-safety-oxide|ownership-safety]]
judgment, exactly as the separate borrow checker it is in `rustc`.

---

## 2. The rules

Each rule is given in **two tiers**: the **basic tier** (Milestone A — move tracking only, what you
code first) and a note on what the **full Oxide rule** (Fig. 5) adds. Symbols follow the repo
convention ([`OXIDE_ANALYSIS.md`](OXIDE_ANALYSIS.md) §0.5): `†τ` = [[types-oxide|dead type]],
`⊎` = per-region union join, `ω ∈ {shrd, uniq}`.

### 2.1 Variables — `T-Move` / `T-Copy`

A variable use is the *only* place the copy-vs-move decision is made, and it's the smallest rule
that already needs `⇒ Γ′`.

```text
 Γ(π) = τ      copyable(τ)
─────────────────────────────         (T-Copy)   — Γ unchanged
 Γ ⊢ π : τ  ⇒  Γ

 Γ(π) = τ      noncopyable(τ)
─────────────────────────────         (T-Move)   — π is now DEAD
 Γ ⊢ π : τ  ⇒  Γ[ π ↦ τ† ]
```

A later use of `π` looks it up, finds `τ†`, and fails — that *is* "use of moved value"
([[ownership-oxide|ownership]], [[types-oxide|the `†τ` type]]).

> **Full rule.** `T-Move` (Fig. 5) additionally demands `π` be **uniq-safe** via the
> [[ownership-safety-oxide|ownership-safety]] judgment (`⊢_uniq π ⇒ {uniq π}`) and **sized**;
> `T-Copy` works on a place *expression* `p` (so it can copy through `*x`) and demands `shrd`-safety.
> Those checks belong to Milestone B.

### 2.2 `let` — `T-Let`

Bind, check the body under the extended environment, drop the binding on the way out.

```text
 Γ ⊢ e₁ : τ₁  ⇒  Γ₁          Γ₁, x:τ₁ ⊢ e₂ : τ₂  ⇒  Γ₂
──────────────────────────────────────────────────────────   (T-Let, basic)
 Γ ⊢ let x = e₁; e₂ : τ₂  ⇒  Γ₂ ∖ x
```

Note how `e₁`'s **output** `Γ₁` is the **input** for `e₂` — that ordering is what makes
`let b = a; let c = a;` fail on line 2 ([[flow-typing-oxide|flow-typing]]).

> **Full rule.** The real `T-Let` runs `gc-loans` (non-lexical drop), rewrites `e₁`'s type to the
> annotation, and — the RAII twist — **requires `x`'s type be dead (`τˢᴰ`) by the end of `e₂`**
> ([`OXIDE_ANALYSIS.md`](OXIDE_ANALYSIS.md) §4.4). For Milestone A, "drop `x`" is just removing it
> from the top frame.

### 2.3 `if` — `T-Branch`

Both arms are checked under the **same** input environment; their outputs are **joined**.

```text
 Γ ⊢ e₁ : bool ⇒ Γ₁
 Γ₁ ⊢ e₂ : τ ⇒ Γ₂        Γ₁ ⊢ e₃ : τ ⇒ Γ₃        Γ₂ ⊎ Γ₃ = Γ′
──────────────────────────────────────────────────────────────   (T-Branch)
 Γ ⊢ if e₁ { e₂ } else { e₃ } : τ  ⇒  Γ′
```

The **join `⊎`** is the subtle part and the source of many real borrow errors:

- a place is **live** in `Γ′` only if it is live in **both** `Γ₂` and `Γ₃` (dead-if-either-moved);
- a [[region-oxide|region]]'s loan set in `Γ′` is the **union** of the two arms' (the conservative
  over-approximation — see [[flow-typing-oxide|flow-typing]] "joining two output contexts").

Both arms must agree on a common result type `τ` (in full Oxide via region rewriting,
[`OXIDE_ANALYSIS.md`](OXIDE_ANALYSIS.md) §4.5; for Milestone A, plain type equality). An `if`
without `else` is the same rule with `e₃ = ()` and `τ = unit`.

### 2.4 Indexing — `T-Index` (adapted)

In Oxide's grammar (Fig. 1) `p[e]` is an expression and `&r ω p[e]` is a borrow-at-index.
The defining fact (paper p. 7; [[place-oxide|place]] doc): **indexing is *not* a place** — an
arbitrary index can't be told apart statically, so it takes ownership of the **whole array**.
That's why you can split a tuple two ways but never an array.

```text
 Γ ⊢ p : [τ; n]  ⇒  Γ₁        Γ₁ ⊢ e : u32  ⇒  Γ₂        copyable(τ)
─────────────────────────────────────────────────────────────────────   (T-Index, basic)
 Γ ⊢ p[e] : τ  ⇒  Γ₂
```

The `copyable(τ)` premise encodes Rust's "**cannot move out of indexed content**": reading
`arr[i]` by value is only allowed when the element copies (otherwise it would leave a hole in the
array). Moving a non-`Copy` element needs `&uniq p[e]` instead. Bounds are checked at runtime, not
here.

> **Full rule.** A general `Index<Idx, Output=τ>` form (traits) is out of Oxide's scope; ownership
> safety for `&r ω p[e]` checks the loan against the *entire* array place, never `p[i]` alone.

### 2.5 `for` — `T-For` (adapted)

A loop body must be **idempotent on the environment**: whatever it leaves in `Γ` must match what it
started with, or iteration 2 would see a moved-out value. This is exactly the "use of moved value:
value moved here, in previous iteration of loop" error.

```text
 Γ ⊢ e₁ : [τ; n]  ⇒  Γ₁          Γ₁, x:τ ⊢ e₂ : unit  ⇒  Γ₂          Γ₂ ∖ x  =  Γ₁
─────────────────────────────────────────────────────────────────────────────────────   (T-For)
 Γ ⊢ for x in e₁ { e₂ } : unit  ⇒  Γ₁
```

The premise `Γ₂ ∖ x = Γ₁` is the **invariance check**: drop the loop variable, then demand the
body's output environment equals its input. The loop as a whole yields `unit` and leaves `Γ₁`.

> **Adapted.** Fig. 5 in the paper shows `if`/`seq`/`let` explicitly; `for`/`while` are in the
> grammar (Fig. 1) and follow this same invariance discipline. Start with concrete iterables
> (`[τ; n]`); generalise to an `IntoIterator`-style protocol only once traits exist.

---

## 3. The algorithm

The mapping from rules to code is mechanical once you commit to **`&mut Env` = `⇒ Γ′`**.

### 3.1 The data types (basic tier)

```rust
/// τ — the type grammar (Milestone A subset; add `Ref { region, own, pointee }` in B)
enum Ty {
    Bool, U32, Unit,
    Tuple(Vec<Ty>),
    Array(Box<Ty>, usize),
    Dead(Box<Ty>),            // †τ — a moved-out value
}

/// Γ — an ordered stack of frames (FILO; see the [[frame-oxide]] doc)
struct Env { frames: Vec<Frame> }
struct Frame {
    bindings: HashMap<String, Ty>,
    // loans: HashMap<Region, LoanSet>,   // <- Milestone B
}
```

`copyable(τ)` = the base types (`Bool`, `U32`, `Unit`) and tuples/arrays of copyable types; a
`Ref` with `ω = shrd` is copyable, `uniq` is not (added in B).

### 3.2 The shape of every rule

```rust
/// `Γ ⊢ e : τ ⇒ Γ′`  becomes  `check(e, &mut env) -> Result<Ty, TypeError>`
///   • the returned `Ty`        is the `τ`
///   • the mutation of `env`    is the `⇒ Γ′`
fn check(e: &Expr, env: &mut Env) -> Result<Ty, TypeError> {
    match e {
        Expr::Var(x)               => check_var(x, env),
        Expr::Let { x, e1, e2 }    => check_let(x, e1, e2, env),
        Expr::If { c, then, els }  => check_if(c, then, els, env),
        Expr::Index { p, idx }     => check_index(p, idx, env),
        Expr::For { x, iter, body }=> check_for(x, iter, body, env),
        // …literals, tuples, etc.
    }
}
```

**Reading a rule into code:** premises *above the line* become statements that run **in order**
(each may mutate `env`, threading `Γ → Γ₁ → Γ₂`); the conclusion's type is the **return value**;
the conclusion's `Γ′` is **the state of `env` when you return**.

### 3.3 Per-construct (each is a direct transcription of §2)

**Variables** — `T-Move`/`T-Copy`. Look up; copy leaves `env` alone, move writes `†τ`:

```rust
fn check_var(x: &str, env: &mut Env) -> Result<Ty, TypeError> {
    let ty = env.lookup(x).ok_or(TypeError::Unbound)?;
    if let Ty::Dead(_) = ty { return Err(TypeError::UseAfterMove); } // Γ(π) = τ†
    if copyable(&ty) {
        Ok(ty)                                  // T-Copy: Γ ⇒ Γ
    } else {
        env.mark_dead(x);                       // T-Move: Γ ⇒ Γ[π ↦ τ†]
        Ok(ty)
    }
}
```

**`let`** — push, check body, pop:

```rust
fn check_let(x: &str, e1: &Expr, e2: &Expr, env: &mut Env) -> Result<Ty, TypeError> {
    let t1 = check(e1, env)?;          //  Γ ⊢ e₁ : τ₁ ⇒ Γ₁   (env now Γ₁)
    env.bind(x, t1);                   //  Γ₁, x:τ₁
    let t2 = check(e2, env)?;          //  … ⊢ e₂ : τ₂ ⇒ Γ₂   (env now Γ₂)
    env.unbind(x);                     //  Γ₂ ∖ x
    Ok(t2)
}
```

**`if`** — the only rule needing a **snapshot** (because both arms start from `Γ₁`) and a **join**:

```rust
fn check_if(c: &Expr, then: &Expr, els: &Expr, env: &mut Env) -> Result<Ty, TypeError> {
    expect(check(c, env)?, Ty::Bool)?;          //  Γ ⊢ e₁ : bool ⇒ Γ₁   (env = Γ₁)

    let mut env_then = env.clone();             //  fork Γ₁
    let t2 = check(then, &mut env_then)?;       //  Γ₁ ⊢ e₂ : τ ⇒ Γ₂

    let mut env_else = env.clone();             //  fork Γ₁ again
    let t3 = check(els, &mut env_else)?;        //  Γ₁ ⊢ e₃ : τ ⇒ Γ₃

    let t = unify(t2, t3)?;                     //  τ₂ = τ₃
    *env = join(env_then, env_else);            //  Γ₂ ⊎ Γ₃ = Γ′
    Ok(t)
}
```

with the join implementing §2.3 (dead-if-either, regions unioned):

```rust
fn join(a: Env, b: Env) -> Env {
    // per binding: live in result  ⟺  live in BOTH a and b; else mark †
    // (Milestone B) per region: loan set = a.loans(r) ∪ b.loans(r)
}
```

**Indexing** — array in, `u32` index, element out (with the copyable guard):

```rust
fn check_index(p: &Expr, idx: &Expr, env: &mut Env) -> Result<Ty, TypeError> {
    let Ty::Array(elem, _) = check(p, env)? else { return Err(TypeError::NotIndexable) };
    expect(check(idx, env)?, Ty::U32)?;         //  Γ₁ ⊢ e : u32 ⇒ Γ₂
    if !copyable(&elem) { return Err(TypeError::MoveOutOfIndex); }  // can't move out of array
    Ok(*elem)
}
```

**`for`** — check iterable, run the body once under a fork, enforce invariance:

```rust
fn check_for(x: &str, iter: &Expr, body: &Expr, env: &mut Env) -> Result<Ty, TypeError> {
    let Ty::Array(elem, _) = check(iter, env)? else { return Err(TypeError::NotIterable) };
    let before = env.clone();                   //  Γ₁

    let mut body_env = env.clone();
    body_env.bind(x, *elem);
    expect(check(body, &mut body_env)?, Ty::Unit)?;
    body_env.unbind(x);                         //  Γ₂ ∖ x

    if body_env != before {                     //  invariance:  Γ₂ ∖ x = Γ₁
        return Err(TypeError::MovedInLoopBody);
    }
    Ok(Ty::Unit)                                //  env stays Γ₁
}
```

### 3.4 Why `clone()` is the honest implementation of the rules

The forks in `if`/`for` look wasteful, but they are *exactly* what the rules say: `Γ₁` is consumed
by **two** independent derivations. `rustc` avoids the copies with a more clever dataflow lattice,
but a clone-and-join is the faithful, debuggable starting point — optimise only once Milestone A is
correct. The same goes for `for`: the body is type-checked **once** (a loop is checked, not run),
and the invariance check is what stands in for "every iteration."

---

## 4. The trap to avoid

The biggest mistake porting these rules is making `check` take `&Env` (read-only) and *return* a
new `Env`, then forgetting to thread it — you silently lose moves and borrows across statements.
**Make it `&mut Env` and let the type system force you to thread it.** Every place you'd write
`⇒ Γ′` in the paper is a place the borrow checker of *your host language* (Rust) will make you
account for the mutation. Lean on that.

Second trap: trying to do **loans and inference at the same time as move tracking**. They are
separate layers for the same reason `rustc` separates typeck from borrowck. Ship Milestone A
(move tracking + join + invariance) first; it is independently testable and is already most of the
"feel" of the borrow checker.

---

## 5. The one-paragraph summary

Build the checker in four steps — **variables + `let`**, then **`if`**, then **indexing**, then
**`for`** — and implement the Oxide judgment `Γ ⊢ e : τ ⇒ Γ′` as a function
`check(e, &mut Env) -> Result<Ty, _>`, where the **returned type is `τ`** and the **mutation of
`env` is `⇒ Γ′`** ([[flow-typing-oxide|flow-typing]]). Variables decide copy-vs-move (`T-Copy`
leaves `Γ`, `T-Move` writes `†τ`); `let` threads `e₁`'s output into `e₂` and drops the binding;
`if` (`T-Branch`) **forks** the environment for each arm and **joins** the results (`⊎`:
dead-if-either, [[region-oxide|regions]] unioned); indexing takes ownership of the **whole array**
and forbids moving a non-`Copy` element out; `for` checks the body once and demands it leave `Γ`
**unchanged** (the loop-invariance that catches "moved in previous iteration"). Do all of this with
**move tracking only** as Milestone A, ignoring [[loan-oxide|loans]], [[region-oxide|regions]], and
inference — then add the [[ownership-safety-oxide|ownership-safety]] judgment and annotation-free
inference as separate layers on top, exactly as Oxide (paper §7) and `rustc` keep them separate.
```

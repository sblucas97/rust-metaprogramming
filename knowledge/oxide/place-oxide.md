# Place — *"a name for a slot of memory"*

A **place** is a static path from a variable to a specific piece of data. Think of it as the answer to *"which box am I talking about?"*

```rust
let mut pt = Point(6, 9);
```

After this line, these are all valid places:

| Place  | Refers to                  |
| ------ | -------------------------- |
| `pt`   | the whole `Point`          |
| `pt.0` | the first field (the `6`)  |
| `pt.1` | the second field (the `9`) |

The key property: **two places either overlap or they don't, and you can decide this statically.**

- `pt` and `pt.0` overlap — one contains the other.
- `pt.0` and `pt.1` do **not** — they're disjoint.

This is how the checker can allow you to borrow `pt.0` and `pt.1` simultaneously: they name different boxes.

> **Restriction:** Places are *not* allowed to go through pointer dereferences (`*x`). The distinction matters because `*x` can alias anything depending on what `x` points to — you can't tell statically just from the path. `pt.0` always means the first field of `pt`, no ambiguity.

---

## What's a pointer dereference?

A **pointer** (or reference) is a value that holds an *address* — it tells you where some other data lives, rather than being the data itself.

```rust
let x = 5;
let r = &x;   // r is a reference: it holds the address of x
```

Here `r` doesn't contain `5`. It contains something like *"the data at memory location `0x7ffe…`"*. To get the actual `5`, you **dereference** `r` — written `*r`. The `*` means *"follow the address and look at what's there."*

```rust
let value = *r;   // *r dereferences r → reads the 5 that r points to
```

So `*x` = *"go to wherever `x` points, and that's the slot I'm talking about."*

---

## Why places stop at dereferences

Recall the key property of a place: two places either overlap or they don't, and you can decide this **statically** (just by looking at the path, without running the program).

That works for field paths like `pt.0` and `pt.1` because the path itself fully determines the memory:

```text
pt.0  →  always the first field of pt. Period.
pt.1  →  always the second field of pt. Period.
```

Disjoint, guaranteed, forever. The compiler knows this from the text alone.

Now look at a dereference:

```rust
let a = Point(6, 9);
let b = Point(1, 2);
let mut x = &a;     // x points to a
// ... later ...
x = &b;             // now x points to b
```

What does the place `*x` refer to? It depends on **what `x` happens to hold at runtime** — sometimes `a`, sometimes `b`. The path `*x` doesn't pin down a specific box.

Worse, consider:

```rust
let p = &mut pt.0;
let q = &mut pt.1;
*p   // is this pt.0? pt.1? something else entirely?
```

Two different pointers `*p` and `*q` might point to the same memory (aliasing) or different memory — you can't tell by reading the path. That's exactly the static-decidability property that places are built to guarantee, and `*x` breaks it.

### The summary

| Expression | Memory it names                  | Known statically?            |
| ---------- | -------------------------------- | ---------------------------- |
| `pt`       | the whole `Point`                | ✅ yes                        |
| `pt.0`     | first field                      | ✅ yes                        |
| `pt.1`     | second field                     | ✅ yes                        |
| `*x`       | whatever `x` points to right now | ❌ no — could alias anything  |

A place is the subset of path expressions where overlap is decidable purely from the syntax. The moment you go through a `*`, you've left that world — reasoning about `*x` requires reasoning about *values* (what does `x` hold?), which is a job for the borrow checker / provenance analysis, not the place machinery itself.

That's the whole reason for the restriction: places are deliberately the *"easy, statically-obvious"* fragment, and dereferences are precisely what's **not** statically obvious.

---

## Reading the place out of an expression

Now the practical skill: **given a Rust expression, what place does it denote?** Every expression falls into one of three buckets.

| Bucket               | What it is                                                        | Has a place? |
| -------------------- | ---------------------------------------------------------------- | ------------ |
| **Place** (pure)     | a root variable + zero or more *field* projections, no `*`       | ✅ a place    |
| **Place expression** | a path that goes through a deref (`*`) or a dynamic index `[i]`  | ⚠️ a *place expression*, not a pure place |
| **Value expression** | produces a fresh temporary — literals, arithmetic, calls, ctors, borrows | ❌ no place   |

Throughout, assume this setup:

```rust
struct Point(i32, i32);
struct Line { start: Point, end: Point }

let n   = 5;
let mut pt = Point(6, 9);
let ln  = Line { start: Point(0, 0), end: Point(3, 4) };
let r   = &pt;          // r  : &Point
let rr  = &r;           // rr : &&Point
let arr = [10, 20, 30];
let v   = vec![1, 2, 3];
let bx  = Box::new(Point(7, 8));
```

### The decision procedure

To find the place an expression denotes, peel it from the **outside in** and accumulate a path:

1. **Variable** `x` → the root of the place is `x`. *(Still a pure place.)*
2. **Field** `E.f` → find the place of `E`, then append the projection `.f`. *(Stays pure if `E` was pure.)*
3. **Deref** `*E` → find the place of `E`, then append a deref step. **You have now left the pure-place fragment** — the result is a place *expression*.
4. **Index** `E[i]` → find the place of `E`, append an index step. Dynamic `i` also leaves the pure-place fragment.
5. **Borrow** `&E` / `&mut E` → **not** a place. It's a *value* (a pointer) that *borrows* the place of `E`.
6. **Anything else** (literal, `a + b`, `f(...)`, `Point(..)`, `if`, block, closure) → **not** a place. A fresh temporary.

> **The auto-deref trap.** `E.f` where `E` is a reference is silently `(*E).f`, and `vec[i]` is silently `*Index::index(&vec, i)`. Rust inserts the `*` for you — so an expression that *looks* like a pure field/index path can actually be a place expression. Always ask: *is the receiver a reference or a smart pointer?* If so, there's a hidden deref.

### Worked classifications

| Expression          | Bucket           | Place / notes                                                            |
| ------------------- | ---------------- | ----------------------------------------------------------------------- |
| `n`                 | place            | `n`                                                                      |
| `pt`                | place            | `pt`                                                                     |
| `pt.0`              | place            | `pt.0`                                                                   |
| `ln.start`          | place            | `ln.start`                                                              |
| `ln.start.0`        | place            | `ln.start.0` — nested field projections, still fully static            |
| `ln.end.1`          | place            | `ln.end.1`                                                              |
| `*r`                | place expression | root `r`, then deref → resolves via `r`'s provenance (`{pt}`)           |
| `(*r).0`            | place expression | deref `r`, then field `.0`                                              |
| `r.0`               | place expression | **auto-deref!** sugar for `(*r).0` — *not* a pure place                 |
| `**rr`              | place expression | two derefs: `rr` → `&Point` → `Point`                                   |
| `bx.0`              | place expression | **auto-deref** through `Box`: `(*bx).0`                                 |
| `arr[1]`            | place expression | index step; not a static field path                                    |
| `v[0]`              | place expression | `*Index::index(&v, 0)` — deref + dynamic index                          |
| `5`                 | value expression | literal — temporary, no place                                          |
| `n + 1`             | value expression | arithmetic — temporary                                                  |
| `pt.0 + pt.1`       | value expression | the *operands* `pt.0`, `pt.1` are places; the **sum** is a temporary    |
| `Point(6, 9)`       | value expression | constructor — fresh value                                              |
| `&pt.0`             | value expression | a pointer value; the place it *borrows* is `pt.0`                       |
| `f(pt)`             | value expression | call result — temporary                                                |
| `if c { pt.0 } else { pt.1 }` | value expression | `if` yields a copied `i32` temporary, even though each arm reads a place |

### The two questions that settle every case

1. **Did I pass through a `*` (explicit or auto-inserted)?** → If yes, it's a place *expression*, not a pure place.
2. **Does the expression build a fresh value (math, call, constructor, borrow, literal)?** → If yes, there's no place at all; only its *sub-expressions* may be places.

If neither happened — you only walked a variable and field projections — you're holding a pure **place**, and its overlap with any other place is decidable on sight.

---

## Layering the borrow checker on top of places

Places only solve the easy half of the problem. The hard half — *"what does `*x` actually refer to?"* — is handled by giving every reference a **provenance**: a static over-approximation of the set of places it could possibly point to.

When the type checker sees `&pt.0`, it doesn't just produce *"a reference"*. It produces a reference whose type carries a provenance set:

```rust
let p = &mut pt.0;   // p : &mut {pt.0}  ← "p may point to pt.0"
let q = &mut pt.1;   // q : &mut {pt.1}  ← "q may point to pt.1"
```

Now the dereference `*p` is no longer a black box. The checker doesn't know the runtime address, but it knows the *set of places* `p` was allowed to come from. So when you write `*p`, it reasons: *"this touches some place in `{pt.0}`."* Likewise `*q` touches `{pt.1}`.

To decide whether `*p` and `*q` conflict, the checker intersects their provenance sets:

```text
{pt.0} ∩ {pt.1} = ∅   → disjoint → both borrows allowed
```

This is the crucial move. Places gave us disjointness for *paths*; provenance lifts that same disjointness test up to *dereferences*, by reducing *"do these pointers alias?"* to *"do their provenance sets overlap?"* — and overlap of place-sets is exactly the statically-decidable question places were built to answer.

### Where provenance comes from

Provenance is not magic; it flows through the program by a few rules:

- **Borrow** — `&place` starts a fresh provenance equal to `{place}`.

  ```rust
  let r = &x;        // r : &{x}
  ```

- **Reborrow / copy** — passing a reference around carries its provenance along.

  ```rust
  let s = r;         // s : &{x}   (same set)
  ```

- **Join at control flow** — if a reference can be one of several depending on a branch, its provenance is the *union* of the possibilities (the over-approximation).

  ```rust
  let m = if cond { &a } else { &b };   // m : &{a, b}
  ```

  Now `*m` is treated as possibly touching `a` **or** `b`, so it conflicts with a borrow of either.

- **Function boundaries** — provenance variables become the abstract regions in a function's signature. A signature like `fn first<'a>(p: &'a Point) -> &'a i32` says *"the returned reference's provenance is whatever the caller passed in"* — the caller substitutes its own place-set in at the call site.

### Why over-approximation is the right call

The set is always a *superset* of the real runtime targets. That makes the analysis **sound but conservative**:

- If the checker says two borrows are disjoint, they truly are — the sets didn't overlap, so no runtime address could coincide.
- The cost: it sometimes rejects a program that would have been fine — when the *possible* targets overlap even though the *actual* ones never would at runtime.

That's the standard static-analysis trade: refuse the ambiguous case rather than risk being wrong.

### The two layers together

| Layer                     | Question it answers                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------- |
| **Places**                | Do these two *paths* name overlapping memory? *(decided from syntax alone: `pt.0` vs `pt.1`)* |
| **Provenance / borrow checker** | Do these two *dereferences* alias? → reduce to: do their place-sets overlap? *(decided by intersecting provenance, built on places)* |

So places are the foundation and provenance is the floor built on top: every `*x` question is ultimately translated back down into a place-overlap question, which is the one thing the system knows how to decide statically.

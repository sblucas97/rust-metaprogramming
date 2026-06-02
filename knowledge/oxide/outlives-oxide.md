# Outlives & region subtyping — *"when can one reference stand in for another?"*

The last piece ties [[region-oxide|regions]] together. So far each reference had its own region. But programs constantly **mix** references — assign one to a variable expecting another, pass one to a function, merge two in an `if`. The question this doc answers:

> *When is a reference with region `'a` allowed where a reference with region `'b` is expected?*

The answer is **region subtyping**, and its classic name is the **outlives** relation, written `'a : 'b`.

---

## Two ways to read the same relation

The relation `'a : 'b` (*"`'a` outlives `'b`"*) has a *temporal* reading and a *set-of-loans* reading. They're the same constraint seen from two angles — exactly the [[region-oxide|"big conceptual shift"]] you already met.

| View                | `'a : 'b` means…                                            |
| ------------------- | ----------------------------------------------------------- |
| **Classic Rust** (time) | region `'a` lives *at least as long as* `'b`             |
| **Oxide** (sets of loans) | the constraint is about **loan-set containment** between regions |

This doc leans on the **set** view, because it's concrete and you already have it: a region *is* a set of loans, so relating two regions is just relating two sets.

---

## The core move: enlarging a region is always safe

Here's the key intuition. A region is an **over-approximation** of where a reference came from. Making that set **bigger** only makes the checker *more* conservative — it assumes the reference might alias *more* places than it really does. That's always sound; it can only ever reject more, never accept something unsafe.

So: **a reference whose region is a *smaller* loan-set can stand in wherever a *larger* loan-set is expected.**

```text
   { shrd pt }            ⊆            { shrd pt, shrd other_pt }
   (knows it's just pt)                (treated as "pt or other_pt")
        │                                          ▲
        └──────────  safe to use here  ────────────┘
```

Treating a `{ shrd pt }` reference as a `{ shrd pt, shrd other_pt }` reference is safe: you're being *extra* careful, guarding against a conflict with `other_pt` that can't actually happen. The reverse — treating a `{pt, other_pt}` reference as if it were only `{pt}` — would be **unsound** (you'd ignore a real possible alias). Subtyping only flows one way: **smaller set → larger set.**

---

## This is exactly why a join is a union

You already saw this without the name. When two branches produce references, the [[region-oxide|merged region is the union]]:

```rust
let m = if cond { &pt } else { &other_pt };
//  region of m = { shrd pt } ∪ { shrd other_pt }
```

Why the union? Because the result must be a region that **both** arms can subtype *into*. The smallest set containing both `{pt}` and `{other_pt}` is their union — and each arm's region is a subset of it, so each arm's reference legally "stands in" for the merged type. Region subtyping is the rule; **union is how you compute the least common supertype.**

```text
   { shrd pt }  ⊆  { shrd pt, shrd other_pt }  ⊇  { shrd other_pt }
       (then arm)            (type of m)            (else arm)
```

---

## Where outlives bounds show up: function signatures

The temporal reading surfaces in function signatures, where you sometimes must *state* an outlives constraint explicitly:

```rust
fn pick<'a, 'b>(x: &'a i32, y: &'b i32) -> &'a i32
    where 'a: 'b      // "'a outlives 'b"
{ x }
```

The bound `'a : 'b` promises the caller that the returned reference (region `'a`) is backed by loans that live at least as long as `'b`'s — so it's safe to use the result anywhere a `'b` reference would be valid. In set terms, it constrains how the caller's actual loan-sets may be substituted in: whatever fills `'a` must cover whatever fills `'b`. It's the same containment rule, written as a promise across the function boundary.

---

## How the checker uses it

Region subtyping is invoked at every point where a region in one type must match a region in another:

| Site                       | What must hold                                            |
| -------------------------- | -------------------------------------------------------- |
| **Assignment** `let y = x` | `x`'s region ⊆ `y`'s expected region                     |
| **Branch join** (`if`)     | each arm's region ⊆ the merged region (so merged = **union**) |
| **Function argument**      | the actual reference's region ⊆ the parameter's region   |
| **Function return**        | the returned region ⊆ what the signature promises        |

In every case the checker emits a **constraint** (`this set ⊆ that set`) and later solves all constraints together — picking the **smallest** regions that satisfy them, just as described in the [[region-oxide|region]] and [[loan-oxide|NLL]] discussions. Outlives/subtyping is *the form those constraints take.*

> **Connecting back:** the [[region-oxide|"regions only grow by constraint, never shrink to dodge a clash"]] note was foreshadowing this. The constraints are precisely these `⊆` (outlives) relations. The solver finds the minimal loan-sets respecting all of them; if even the minimal solution makes a [[loan-oxide|loan conflict]] live where it shouldn't be, you get a borrow error.

---

## The one-paragraph summary

**Region subtyping** answers when a reference with one [[region-oxide|region]] may be used where another is expected, and its classic name is the **outlives** relation `'a : 'b`. The sound direction is *smaller loan-set → larger*: enlarging a region only makes the checker more conservative, so a reference known to come from few origins can always stand in where more origins are assumed — the reverse is unsound. This is why a branch **join** takes the **union** (the least region both arms subtype into), and why function signatures sometimes carry explicit `'a: 'b` bounds (a promise that one region's loans cover another's). Operationally, every assignment, join, call, and return emits a `⊆` **constraint**, and the checker solves them all for the smallest [[region-oxide|regions]] that fit — which is exactly the constraint-solving that makes [[loan-oxide|non-lexical lifetimes]] and the whole borrow check fall out.

# Types — *"the static shape the checker gives every value"*

Throughout the other docs, the symbol `τ` (tau) kept appearing — in [[loan-oxide|loans]] (`x : τ`), in [[frame-oxide|frames]] (`ℱ, x : τ`), in the typing judgment (`Γ ⊢ e : τ ⇒ Γ′`). This doc finally says what `τ` actually *is*: the grammar of **types** in the model. A type is the static description the checker attaches to every value — its *shape*, plus, crucially, the **borrow information** baked right into it.

The headline: in Oxide, a reference type doesn't just say *"a reference."* It carries the [[region-oxide|region]] and [[loan-oxide|ownership]] **inside the type itself.** That's what makes the whole system work — the borrow facts travel *with* the type.

---

## The grammar of `τ`

```text
τ  ::=  base                  (i32, bool, char, …)
     |  &r ω τ                (a reference: region r, ownership ω, pointing at τ)
     |  (τ₁, τ₂, …)           (a tuple)
     |  τ₁ -ℱ→ τ₂             (a function/closure carrying a captured frame ℱ)
     |  †τ                    (a "dead" / moved-out type)
```

Let's take them one at a time.

| Type form  | Name              | Carries borrow info? |
| ---------- | ----------------- | -------------------- |
| `base`     | plain scalar      | no                   |
| `&r ω τ`   | **reference**     | **yes — region + ownership** |
| `(τ₁, τ₂)` | tuple / struct    | only via its fields  |
| `τ₁ -ℱ→ τ₂`| function / closure| yes — via captured frame `ℱ` |
| `†τ`       | dead / uninitialized | tracks [[ownership-oxide\|moves]] |

---

## Base types

The simplest types — `i32`, `bool`, `char`, and friends. Plain data, no references inside, nothing to borrow-check. These are also the [[ownership-oxide|`Copy`]] types: duplicating their bytes is harmless, so assigning one doesn't move it.

```rust
let n: i32 = 5;       // τ = i32
```

---

## The reference type `&r ω τ` — the heart of it

This is the type form that makes Oxide *Oxide*. A reference type has **three** parts welded together:

```text
        &   r   ω    τ
        │   │   │    └── the type it points AT (the pointee)
        │   │   └─────── ownership:  shrd or uniq      ← the LOAN doc
        │   └─────────── region / provenance: a set of loans   ← the REGION doc
        └─────────────── "this is a reference"
```

```rust
let x = &uniq pt.0;   //  x : &'r uniq i32     where 'r = { uniq pt.0 }
```

Everything you learned about [[loan-oxide|loans]] and [[region-oxide|regions]] lives *here*, in the type of every reference value. That's why [[flow-typing-oxide|flow-sensitive typing]] can carry borrow facts forward: the facts are **part of the type**, and types ride through the context `Γ`. When [[ownership-safety-oxide|ownership safety]] needs to know "which loans does `*x` touch?", it reads them straight off `x`'s type.

| Part | Question it answers                | Defined in            |
| ---- | ---------------------------------- | --------------------- |
| `r`  | *where could this point?*          | [[region-oxide\|region]]   |
| `ω`  | *read-only or exclusive?*          | [[loan-oxide\|loan]]       |
| `τ`  | *what's the shape of the target?*  | this doc (recursively) |

---

## Tuple (and struct) types

A tuple type is just a fixed sequence of component types. Its [[place-oxide|places]] are the field projections `.0`, `.1`, … you met in the very first doc.

```rust
let pt: (i32, i32) = (6, 9);     //  τ = (i32, i32)
//   place pt.0 : i32
//   place pt.1 : i32
```

The static **disjointness** of `pt.0` and `pt.1` — the thing that lets you borrow or move them independently — comes directly from the tuple type having two separate component slots.

---

## Function / closure types `τ₁ -ℱ→ τ₂`

A function takes a `τ₁` and returns a `τ₂`. A **closure** is a function that *also* drags along a captured [[frame-oxide|frame]] `ℱ` — the snapshot of borrows it closed over. That's exactly the suspended frame from the [[frame-oxide|frame]] doc, and it's written right into the arrow:

```text
i32  -ℱ→  i32        "a function i32 → i32 carrying captured frame ℱ"
```

Because `ℱ` is part of the closure's **type**, the borrows it captured stay alive in `Γ` as long as the closure value does — and [[ownership-safety-oxide|ownership safety]] knows to look inside it. The type is what keeps those captured loans on the books.

---

## The dead type `†τ` — moves, in the type system

When a value is [[ownership-oxide|moved out]] of a place, the place doesn't vanish — its type is marked **dead** (written `†τ` here, "was a `τ`, now moved-out"). Reading a place whose type is dead is the *"use of moved value"* error.

```rust
let a = Point(6, 9);   //  a : Point
let b = a;             //  moves a → b.   now  a : †Point   (dead)
let c = a.0;           //  ❌ a's type is dead
```

This is how [[flow-typing-oxide|flow-sensitive typing]] records a move: the `⇒ Γ′` step rewrites `a`'s entry from `a : Point` to `a : †Point`. **Partial moves** work because the deadness can land on a single field's type (`pt.0 : †T` while `pt.1 : T` stays live) — the same per-place granularity as borrows.

---

## Putting the grammar to work

A single richly-typed value shows every layer at once:

```rust
let r = &shrd pt;     //  r : &'r shrd (i32, i32)
//                            │   │     └──────── pointee: a tuple type
//                            │   └────────────── ownership (loan kind)
//                            └────────────────── region (set of loans)
```

Reading that type left to right *is* reading the borrow checker's knowledge about `r`: where it may point (`'r`), how it may touch it (`shrd`), and the shape it sees (`(i32, i32)`). The type **is** the borrow information.

---

## The one-paragraph summary

A **type** `τ` is the static shape the checker assigns every value, and the grammar is small: **base** scalars (`i32`, the [[ownership-oxide|`Copy`]] types), **references** `&r ω τ`, **tuples** `(τ₁, τ₂, …)`, **functions/closures** `τ₁ -ℱ→ τ₂`, and **dead** types `†τ` for [[ownership-oxide|moved-out]] places. The pivotal one is the reference type `&r ω τ`, which bakes the [[region-oxide|region]] *and* the [[loan-oxide|ownership]] **into the type itself** — so that [[flow-typing-oxide|flow-sensitive typing]] carries borrow facts forward simply by carrying types, and [[ownership-safety-oxide|ownership safety]] reads the loans an access needs straight off the type. Closures store their captured [[frame-oxide|frame]] in the arrow, keeping captured loans alive; moves rewrite a place's type to `†τ`, which is how the system tracks initialization. In short: in Oxide the type of a value *is* what the borrow checker knows about it.

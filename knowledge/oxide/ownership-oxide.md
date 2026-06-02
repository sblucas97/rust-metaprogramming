# Ownership — *"who is responsible for this value, and what happens when you give it away"*

Everything you've studied so far — [[place-oxide|places]], [[loan-oxide|loans]], [[region-oxide|regions]] — is the **borrowing** half of Rust. But borrowing only exists as an *alternative* to the thing it protects: **ownership**, and its central operation, the **move**.

This is the other half of the language. You can't really understand *why you'd borrow* until you understand *what borrowing lets you avoid* — namely, giving the value away for good.

> **The one-sentence idea:** every value has exactly **one owner**, and assigning or passing it **moves** ownership to a new owner — leaving the old place **dead** (unusable) unless the type opts out via `Copy`.

---

## A value has exactly one owner

In Rust, a value isn't floating in space — some place is **responsible for it** (responsible, ultimately, for freeing its memory). That place is its **owner**.

```rust
let a = Point(6, 9);   // `a` owns this Point
```

There is exactly one owner at a time. That single-owner rule is what lets Rust free memory deterministically with no garbage collector: when the owner goes out of scope, the value is dropped.

---

## Move — handing ownership over

When you assign a value somewhere new, ownership **moves**:

```rust
let a = Point(6, 9);
let b = a;            // ownership MOVES from a → b
```

After this line, `b` owns the `Point` and **`a` is dead** — it no longer owns anything. Using it is a compile error:

```rust
let c = a.0;          // ❌ ERROR: use of moved value `a`
```

No loan, no region, no borrow happened here. This is a completely separate mechanism from everything in the other docs: ownership was *transferred*, and the place `a` became **uninitialized / dead**.

| Place state | Meaning                              | Can you read it? |
| ----------- | ------------------------------------ | ---------------- |
| **live**    | the place currently owns a value     | ✅ yes            |
| **dead** (moved-out) | ownership was given away      | ❌ no — error     |

> **This is affine typing.** A value can be used (consumed) **at most once**. A move *consumes* the old place. The type checker tracks, for every [[place-oxide|place]], whether it's currently live or dead — and reading a dead place is rejected.

---

## Copy vs Move — why `let b = n` doesn't kill `n`

Not every assignment is a move. Small, plain-data types are **`Copy`**: assigning them *duplicates* the value instead of transferring it, so the original stays live.

```rust
let n = 5;
let m = n;            // n is Copy → m gets a COPY
let k = n + 1;        // ✅ n is still alive, never moved
```

```rust
let a = Point(6, 9);  // (suppose Point is NOT Copy)
let b = a;            // a is Move → ownership transferred
let k = a.0;          // ❌ a is dead
```

| Kind     | On assignment   | Original after | Examples                          |
| -------- | --------------- | -------------- | --------------------------------- |
| **Copy** | bit-for-bit duplicate | stays **live** | `i32`, `bool`, `char`, `&T` (shared refs) |
| **Move** | ownership transfer | becomes **dead** | `String`, `Vec`, `Box`, most structs |

The rule of thumb: a type is `Copy` only if duplicating its bytes is harmless — no heap buffer to double-free, no unique resource to alias. Anything that owns a resource is `Move`.

---

## Partial moves — places strike again

Because ownership is tracked **per [[place-oxide|place]]**, you can move *one field* out of a struct and keep using the rest. The disjointness of `pt.0` and `pt.1` that let you *borrow* them independently also lets you *move* them independently:

```rust
let pt = Point(big_owned_thing(), 9);

let taken = pt.0;     // moves ONLY pt.0 out → pt.0 is dead
let still = pt.1;     // ✅ pt.1 untouched, still live
let whole = pt;       // ❌ ERROR: can't move all of pt — pt.0 is already gone
```

This is the exact same static place-overlap reasoning from the [[place-oxide|place]] doc, now applied to *ownership state* instead of borrows. `pt.1` is a disjoint box, so moving `pt.0` leaves it alone; `pt` overlaps `pt.0`, so once `pt.0` is gone you can't move the whole.

---

## Reading a place requires it to be live

Tying it together: **any access** — a read, a move, or a [[loan-oxide|borrow]] — requires the place to currently be live. A moved-out place owns nothing, so there's nothing to read, move, or lend.

```text
move out of a place   →  place becomes dead
read / use a place    →  requires the place to be live
borrow a place        →  requires the place to be live (you can't lend what you don't have)
```

---

## Why borrowing exists at all

Now the whole picture clicks. Suppose you want a function to *look at* your `Point` without taking it away forever:

```rust
let pt = Point(6, 9);
let len = distance(pt);   // if this MOVES pt, you've lost it...
let x = pt.0;             // ❌ ...and this is now an error
```

Moving everything you want to use would be unbearable. **Borrowing is the escape hatch:** lend the value temporarily instead of giving it away.

```rust
let pt = Point(6, 9);
let len = distance(&pt);  // borrows pt — ownership stays home
let x = pt.0;             // ✅ pt is still yours
```

That's the relationship between the two halves:

| Operation     | What happens to ownership      | The place afterward |
| ------------- | ------------------------------ | ------------------- |
| **Move** `b = a` | transferred away               | **dead**            |
| **Borrow** `&a`  | stays put; a [[loan-oxide\|loan]] is recorded | still **live** (just temporarily lent) |

Borrowing is *"let me use it without owning it."* The entire loan/region/frame machinery exists to make that safe — to guarantee a borrow can't outlive or alias-corrupt the value its owner still holds.

---

## The one-paragraph summary

**Ownership** is the rule that every value has exactly one responsible owner, and a **move** (plain assignment or pass-by-value for non-`Copy` types) *transfers* that ownership, leaving the source [[place-oxide|place]] **dead** — using it afterward is an error. `Copy` types opt out by duplicating instead of transferring, so the original stays live; and because ownership is tracked per place, you can move or keep individual fields independently. This affine *"use it at most once"* discipline is the half of Rust that **[[loan-oxide|borrowing]] is built to avoid**: a borrow lets you touch a value *without* moving it, so its owner keeps the value while you work. Ownership says *whose value is it*; borrowing says *who may temporarily look at it* — together they're the whole safety story.

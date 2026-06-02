# Loan — *"a record that someone borrowed from a specific place"*

A **loan** is the borrow checker's note that says:

> *"A reference was created. It came from **this** place, and it was borrowed with **this** kind of ownership."*

If a place is the answer to *"which box?"*, a loan is the answer to *"who is currently holding a key to that box, and is it a read-only key or an exclusive one?"*

The borrow checker keeps a running set of these notes. Every time you write `&` or `&mut`, a note is added. When the reference dies, the note is removed. At every step, the checker scans the notes to make sure no two of them conflict — that scan is what enforces Rust's *"shared XOR mutable"* rule.

---

## The two pieces of a loan

A loan is just two things glued together:

```text
loan  =  ⟨ ownership qualifier ⟩  +  ⟨ place ⟩
```

### 1. The ownership qualifier `ω`

Every borrow is one of exactly two flavours, written `ω` (omega) in the formal notation, with `ω ∈ {shrd, uniq}`:

| Qualifier | Rust syntax | Meaning                          | Real-world analogy             |
| --------- | ----------- | -------------------------------- | ------------------------------ |
| `shrd`    | `&x`        | **shared** — read-only, many allowed | a *photocopy* of a document |
| `uniq`    | `&mut x`    | **unique** — exclusive, may mutate   | the *only key* to a room    |

- A **`shrd`** (shared) loan promises *"I won't change the data, and I'm fine with others also reading it."* You can hand out as many shared keys as you like — copies of a read-only document never disagree.
- A **`uniq`** (unique) loan promises *"I am the only one who can see or touch this right now."* Exactly one exclusive key can exist, and while it does, no other key — shared or unique — may.

> **Naming note.** Rust's surface syntax says *"mutable"* (`&mut`), but the deeper property the checker actually enforces is **uniqueness**, not mutation. That's why the formal model calls it `uniq`. Mutation is *allowed* because the reference is unique; uniqueness is the real invariant.

### 2. The place

The second half is the place the reference points at — `pt.0`, `ln.start`, etc. This is *what* got borrowed. Because places have a statically-decidable overlap relation, the checker can compare the places of two loans and know whether they touch the same memory.

### Putting them together — the formal notation

In the Oxide formalism a loan is written:

```text
ω p        — a place expression p tagged with an ownership qualifier ω
```

So `uniq pt.0` is the loan *"a unique borrow of `pt.0`."* That's the whole object.

---

## How a loan is born

When the type checker walks over a borrow expression, it mints a loan:

```rust
let x = &uniq pt.0;     // (oxide spelling of &mut pt.0)
```

The moment this line is checked, a loan comes into existence:

```text
uniq pt.0
```

This note now says: *"there is a unique borrow pointing at `pt.0`."* As long as the loan is **live**, the checker won't let anyone else touch `pt.0` — not a read, not a write, nothing.

Each borrow expression produces exactly one loan:

| Expression       | Loan minted   |
| ---------------- | ------------- |
| `&n`             | `shrd n`      |
| `&pt`            | `shrd pt`     |
| `&pt.0`          | `shrd pt.0`   |
| `&mut pt.1`      | `uniq pt.1`   |
| `&mut ln.start`  | `uniq ln.start` |
| `&ln.start.0`    | `shrd ln.start.0` |

---

## The conflict rule

A new loan is allowed only if it doesn't conflict with any loan already live. Two loans conflict when **their places overlap** *and* **at least one of them is `uniq`.**

| Existing \ New | `shrd`        | `uniq`        |
| -------------- | ------------- | ------------- |
| **`shrd`**     | ✅ fine        | ❌ **ERROR**   |
| **`uniq`**     | ❌ **ERROR**   | ❌ **ERROR**   |

Read it as:

- `shrd` + `shrd` → ✅ **fine** — many readers can coexist.
- `shrd` + `uniq` → ❌ **ERROR** — a reader and a writer at the same time.
- `uniq` + `uniq` → ❌ **ERROR** — two writers at the same time.

This is precisely *"shared **XOR** mutable"*: the only allowed combination involving the same memory is *all-shared*.

> **The overlap half matters too.** The table above only fires when the two places actually overlap. `uniq pt.0` and `uniq pt.1` are **both unique yet both fine**, because `pt.0` and `pt.1` are disjoint places — different boxes. This is exactly the [[place-oxide|place]] disjointness property doing the heavy lifting. Conflict = *overlapping places* **AND** *not both shared*.

### Worked examples

```rust
let r1 = &pt.0;        // loan: shrd pt.0
let r2 = &pt.0;        // loan: shrd pt.0   ✅ two readers of the same field
```

```rust
let a = &mut pt.0;     // loan: uniq pt.0
let b = &pt.0;         // loan: shrd pt.0   ❌ ERROR: read while a writer is live
```

```rust
let a = &mut pt.0;     // loan: uniq pt.0
let b = &mut pt.1;     // loan: uniq pt.1   ✅ disjoint places — no overlap
```

```rust
let whole = &mut pt;   // loan: uniq pt
let part  = &pt.0;     // loan: shrd pt.0   ❌ ERROR: pt CONTAINS pt.0, so they overlap
```

That last one is the subtle case: `pt` and `pt.0` aren't equal, but they **overlap** (one contains the other), so a `uniq` on the whole and *any* loan on a part collide.

---

## Loan liveness — when the note is torn up

A loan is **live** from the moment the borrow is created until the last point the reference is actually used. After that, the loan is dropped and the place is free again.

```rust
let mut pt = Point(6, 9);

let a = &mut pt.0;     // loan uniq pt.0 is born
*a += 1;               // last use of `a` — loan dies HERE
                       // (no note for pt.0 anymore)

let b = &pt.0;         // ✅ fine: the earlier uniq loan is already gone
```

This *"until the last use"* rule — not *"until the end of the enclosing scope"* — is what Rust calls **non-lexical lifetimes (NLL)**. The loan's lifetime is the region of code where the reference is genuinely needed, no longer. Shrinking loans to their real span is what lets the two borrows above coexist even though both names are still in scope.

> Think of liveness as the *duration the note stays pinned to the board*. The conflict scan only ever compares loans whose durations **overlap in time**. Two loans that never coexist — like `a` dying before `b` is born — can't conflict no matter what their places are.

So a conflict needs **both** kinds of overlap at once:

| Overlap in *space* (places touch) | Overlap in *time* (both live) | Result          |
| --------------------------------- | ----------------------------- | --------------- |
| no                                | —                             | ✅ never conflict |
| yes                               | no                            | ✅ never conflict |
| yes                               | yes (and not both `shrd`)     | ❌ **ERROR**     |

---

## How loans connect to places and provenance

The three concepts stack into one pipeline:

1. **[[place-oxide|Place]]** — *which box?* A static path (`pt.0`) whose overlap with other paths is decidable on sight.
2. **Loan** — *who holds a key, and what kind?* An ownership qualifier glued to a place: `uniq pt.0`.
3. **[[place-oxide|Provenance]]** — *where could this reference have come from?* The set of loans/places a reference is allowed to refer to, used to resolve dereferences (`*x`) back down to places.

When you dereference a reference, the checker looks at its **provenance** (the set of loans it carries), pulls out the underlying places, and runs the same conflict scan on those. So even `*x` — which isn't a pure place — gets checked by reducing it back to the loans it could possibly be.

```text
&mut pt.0   ──mints──▶   loan: uniq pt.0   ──recorded in──▶   the reference's provenance: {uniq pt.0}
                                                                        │
                                                              later, *x reads it back
                                                                        ▼
                                                       conflict scan over places {pt.0}
```

That's the full loop: a borrow mints a **loan**, the loan names a **place**, the place rides along in the reference's **provenance**, and every later access reduces back to a place-overlap check against all the live loans.

---

## The one-paragraph summary

A **loan** is a live note of the form `ω p` — an ownership qualifier (`shrd` = read-only/many, `uniq` = exclusive/one) attached to a [[place-oxide|place]]. A borrow expression mints one; the reference's last use kills it. Two loans conflict only when their places **overlap** *and* their lifetimes **overlap in time** *and* at least one is `uniq`. That single rule — checked by scanning the set of live loans — is the entire machinery behind Rust's *"shared XOR mutable"* guarantee.

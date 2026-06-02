# Ownership safety — *"the one rule that says yes or no to a borrow"*

You now have the four **nouns**: a [[place-oxide|place]] (which box), a [[loan-oxide|loan]] (a key + its kind), a [[region-oxide|region]] (the set of loans a reference might be), and a [[frame-oxide|frame]] (where all that is stored, stacked into the context `Γ`). **Ownership safety** is the **verb** that operates on them — the single judgment the checker runs every time you borrow or access something.

Every "conflict scan" the other docs kept hand-waving at *is* this judgment. Here it gets a name and a precise job.

---

## What it takes in, what it gives back

Ownership safety is a function with a very specific signature:

```text
GIVEN:   the context Γ        (all frames, all region→loan entries)
         an ownership ω        (shrd or uniq — how you want to touch it)
         a place expression p  (what you want to touch)

PRODUCE: either  ❌ REJECT          (some live loan conflicts)
         or      ✅ a set of loans  (the loans this access would create/need)
```

In words: *"I want to access place `p` with ownership `ω`. Is that safe given everything currently borrowed in `Γ` — and if so, what loans does it entail?"*

It does **two jobs at once**:

1. **Collect** the loans this access needs (walking through any [[region-oxide|regions]] it touches).
2. **Check** that none of those conflict with a loan already live anywhere in `Γ`.

---

## Job 1 — collecting the loans an access needs

For a pure [[place-oxide|place]] like `pt.0`, the loan it needs is obvious: `ω pt.0`. But if the place goes **through a reference** (`*x`), the access actually reaches *all the loans in that reference's [[region-oxide|region]]* — because `*x` could be any of them.

```rust
let p = &uniq pt.0;          // region of p:  { uniq pt.0 }
// ... access *p with shrd ...
//   → the loans this access touches = the places behind p's region = { pt.0 }
```

So "collecting" means: start at the place, follow any dereferences down into their regions, and gather the underlying [[place-oxide|places]] those regions stand for. This is exactly the *"reduce `*x` back to a place-overlap question"* move from the [[place-oxide|place]] and [[loan-oxide|loan]] docs — now stated as a step of the judgment.

---

## Job 2 — checking for a conflict in `Γ`

With the needed loans in hand, ownership safety scans **every region entry `r ↦ {…}` in every [[frame-oxide|frame]] of `Γ`** (including frames suspended inside closure types) and applies the [[loan-oxide|loan]] conflict rule to each existing loan:

> Two loans conflict when their **[[place-oxide|places]] overlap** *and* **at least one is `uniq`**.

| You want `ω` … | … and a live loan over an overlapping place is `shrd` | … is `uniq` |
| -------------- | ----------------------------------------------------- | ----------- |
| **`shrd`**     | ✅ fine (many readers)                                 | ❌ REJECT    |
| **`uniq`**     | ❌ REJECT                                              | ❌ REJECT    |

If *any* existing loan clashes, the whole judgment **rejects** — that's the compile error. If none do, it **succeeds** and returns the freshly-collected loans, ready to be written into a region.

---

## Worked example

```rust
let mut pt = Point(6, 9);

let a = &uniq pt.0;     // ownership-safety(Γ, uniq, pt.0):
                        //   needs { uniq pt.0 };  Γ has no live loans → ✅
                        //   → writes uniq pt.0 into a's region. Now live.

let b = &shrd pt.0;     // ownership-safety(Γ, shrd, pt.0):
                        //   needs { shrd pt.0 };  Γ has live `uniq pt.0`
                        //   pt.0 overlaps pt.0, existing is uniq → ❌ REJECT

let c = &shrd pt.1;     // ownership-safety(Γ, shrd, pt.1):
                        //   needs { shrd pt.1 };  live loan is uniq pt.0
                        //   pt.0 and pt.1 are DISJOINT → no conflict → ✅
```

Notice the judgment never reasons about *time* or *runtime values* — only about the **places** in `Γ` and their static overlap. That's the entire reason the [[place-oxide|place]] abstraction was built the way it was: it makes Job 2 a decidable, syntactic check.

---

## Where it sits in the borrow's lifecycle

Ownership safety is **step ①** of taking a borrow. The full sequence (from the [[frame-oxide|frame]] doc):

```text
&ω p
  │
  ▼
① ownership safety(Γ, ω, p)   ← THIS judgment: collect loans + scan Γ for conflicts
  │
  ├─ ❌ conflict found  → compile error
  │
  └─ ✅ no conflict     → returns the new loans
                           │
                           ▼
② write those loans into the region's entry in the current frame   ("now live")
                           │
                           ▼
③ later: gc-loans / frame pop clears them                          ("now dead")
```

So ownership safety is the **gatekeeper**: nothing becomes a live loan without passing it first.

---

## How it connects to ownership (moves)

A subtlety worth flagging: ownership safety guards **borrows**, but the [[ownership-oxide|move]] machinery guards **ownership transfer**. They share a precondition — *you can only borrow or move a place that is currently **live***. A moved-out (dead) place owns nothing, so ownership safety has nothing to lend. The two judgments are siblings: one keeps borrows from aliasing unsafely, the other keeps you from using a value you already gave away.

---

## The one-paragraph summary

**Ownership safety** is the core judgment of the borrow checker: given the context `Γ`, a desired ownership `ω`, and a [[place-oxide|place]] `p`, it (1) **collects** the [[loan-oxide|loans]] the access needs — following any `*` down into [[region-oxide|regions]] to recover the underlying places — and (2) **scans every loan in every [[frame-oxide|frame]] of `Γ`** for a conflict, where conflict means *overlapping places and at least one `uniq`*. If anything clashes it **rejects** (your compile error); otherwise it **returns the new loans** to be written into a region. It's the gatekeeper every borrow must pass, and it decides purely from the *static places* in `Γ` — never from runtime values — which is exactly what the place abstraction was designed to make possible.

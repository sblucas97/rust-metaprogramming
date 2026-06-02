# Frame — *"one layer of the call/scope stack"*

We've built the model bottom-up: a [[place-oxide|place]] names a box, a [[loan-oxide|loan]] ties an ownership kind to that box, and a [[region-oxide|region]] is the *set of loans* a reference might be. But one question was left dangling:

> *Where do all these regions and bindings actually **live** while the checker works?*

The answer is a **frame**. A frame is **one layer of the type checker's environment** — the bookkeeping for a single scope or function call. If a region is a set of loans, a frame is the **filing cabinet** that holds those region→loans entries (plus the variable types) for one level of the program.

Think of it exactly like a **stack frame** in a running program — but instead of holding runtime values, it holds the *type-checking facts* that are true inside one scope.

---

## What a frame holds

A frame (written `ℱ`) is a list of entries, and there are exactly **two kinds**:

```text
ℱ ::= •                 (empty frame)
    | ℱ, x : τ          (a variable binding and its current type)
    | ℱ, r ↦ { ℓ̄ }      (a region r mapped to its current set of loans)
```

| Entry          | Reads as                                 | Comes from        |
| -------------- | ---------------------------------------- | ----------------- |
| `x : τ`        | *"variable `x` currently has type `τ`"*  | a `let` binding   |
| `r ↦ { ℓ̄ }`    | *"region `r` currently holds these loans"* | a region declaration |

> **This is where regions physically live.** In the [[region-oxide|region]] doc we said a region *is* a set of loans but were vague about *where that set is stored*. Here's the answer: the `r ↦ { … }` entry **inside a frame** is the home of that set. When the region doc said *"`'r = { uniq pt.0 }`,"* it meant *"somewhere in the current frame there is an entry `'r ↦ { uniq pt.0 }`."*

### Adding entries

Two source-level constructs write into the **current** (top-of-stack) frame:

```rust
let x = ...;            // adds   x : τ        to the current frame
letrgn <r> { ... }      // adds   r ↦ { }      (a fresh, empty region) to the current frame
```

Both kinds of entry land in the *same* frame — the one for the scope you're currently inside.

---

## The stack `Γ` — frames piled up

A single frame describes one scope. A whole program is **nested** scopes, so the checker carries a **stack of frames** called the **stack typing**, written `Γ` (capital gamma). Frames are separated by `‡`:

```text
Γ  =  ℱ₁ ‡ ℱ₂ ‡ ℱ₃
      └┬─┘   └┬─┘   └┬─┘
   outermost  …    innermost (current)
```

**Order matters.** The checker reads frames in **FILO** order — *first-in, last-out* — exactly like a real call stack: the most recently entered scope is on top, and it's the first one consulted.

```text
                  ┌─────────────────────────────┐
   push on entry  │  ℱ₃   inner block  (current) │  ← top: looked at first
                  ├─────────────────────────────┤
                  │  ℱ₂   function body          │
                  ├─────────────────────────────┤
                  │  ℱ₁   outer scope            │  ← bottom: outermost
                  └─────────────────────────────┘
   pop on exit    (top frame removed when its scope ends)
```

---

## Popping a frame = end of a scope

When a `let`-scope or block ends, its frame is **popped** off `Γ` — and **all the loans living in that frame's regions die with it.**

```rust
let mut pt = Point(6, 9);
{
    let r = &pt.0;       // pushes nothing new, but writes loan `shrd pt.0`
                         //   into a region entry in the CURRENT frame
    // ... use r ...
}                        // block ends → its frame is POPPED
                         //   → the region entry goes away
                         //   → loan `shrd pt.0` dies
let w = &mut pt.0;       // ✅ fine: that loan is gone with the popped frame
```

> **How this relates to region GC.** The [[region-oxide|region]] doc described loans dying by *garbage collection* — *"the region name is no longer mentioned in any live type."* Frame-popping is the **coarser, scope-level** version of the same idea: when a frame leaves `Γ`, every region it owned (and every loan in them) is gone in one stroke. GC handles the fine-grained *within-a-scope* case (last use); frame-popping handles the *whole-scope-ended* case. Both end with the same result: the loan leaves `Γ`, so it can no longer conflict with anything.

---

## Closures — why frames are the interesting part

Everything above is tidy because scopes nest and pop in order. **Closures break that tidiness** — and frames are precisely the machinery that copes.

When a closure captures variables, it doesn't just grab values — it **packages up a snapshot of the current frame** and *suspends* it over the function arrow. Oxide writes a closure type as:

```text
τ₁  -ℱ→  τ₂        "a function from τ₁ to τ₂ that carries a captured frame ℱ"
```

That captured frame `ℱ` stays **alive inside the closure's type** for as long as the closure value lives — even after the scope it came from has ended.

```rust
let printer = {
    let r = &pt.0;          // loan `shrd pt.0` born in THIS inner frame
    move || use_ref(r)      // closure captures r → snapshots the frame,
                            //   suspends it in the closure's type
};                          // inner block ends, frame would normally pop...
                            //   ...but the loan is SUSPENDED in `printer`'s type,
                            //   so `shrd pt.0` is STILL live out here
printer();                  // uses r → the loan had better still be live. It is.
```

If frames simply vanished at scope exit, the loan `shrd pt.0` would die at the closing brace — but `printer` still holds `r` and will dereference it later. The suspended frame is what keeps that loan alive past its lexical scope.

> **The consequence for the checker:** ownership safety can't just scan the top frame, or even the linear stack. It must scan **every region in every frame — including the frames suspended inside closure types.** A loan does **not** die just because its original scope ended, if a closure is still holding a reference into it.

---

## How it all fits together — the complete machine

Now every layer has a home:

```text
Region 'r  ──owns──▶  { loan₁, loan₂, … }
                            │
                        each loan  =  (ownership ω, place π)
                                                    │
                                          place = a static path into a
                                          variable, no pointer derefs

   …and every region entry  r ↦ { ℓ̄ }  lives inside a FRAME,
   …and every frame stacks (FILO) into  Γ = ℱ₁ ‡ ℱ₂ ‡ … ,
   …including frames suspended inside closure types  ( τ -ℱ→ τ ).
```

The complete stack of concepts, one per layer:

| Layer                      | Object       | "Where it lives"                         |
| -------------------------- | ------------ | ---------------------------------------- |
| [[place-oxide\|Place]]     | `pt.0`       | a static path — pure syntax              |
| [[loan-oxide\|Loan]]       | `uniq pt.0`  | inside a region's loan-set               |
| [[region-oxide\|Region]]   | `{ uniq pt.0 }` | a `r ↦ {…}` entry inside a frame       |
| **Frame**                  | `ℱ`          | one layer of `Γ`                         |
| **Stack typing**           | `Γ`          | the whole context the checker carries    |

### A borrow, traced through `Γ`

```text
   &uniq pt.0
        │
        ▼
   ① ownership safety runs over place pt.0
        │   scans EVERY  r ↦ {…}  entry in EVERY frame of Γ
        │   (including closure-suspended frames)
        ▼
   ② no conflicting loan found anywhere in Γ
        │
        ▼
   ③ write the new loan  uniq pt.0  into region r's entry in the current frame
        │
        ▼
   "the reference is now LIVE"   ← that single write is what makes it live

   … later …

   ④ gc-loans clears r's loan set   (or its frame is popped)
        │
        ▼
   "the reference is now DEAD"
```

That single write into `Γ`'s entry for `r` **is** the reference coming to life; clearing that entry (by GC or by popping the frame) **is** it dying. The borrow checker is, at bottom, just **maintaining `Γ` and scanning it for conflicts** at every borrow.

---

## The one-paragraph summary

A **frame** `ℱ` is one layer of the checker's environment — a list of two kinds of entries: variable bindings `x : τ` and region→loan-set maps `r ↦ { ℓ̄ }`. Frames stack (FILO) into the **stack typing** `Γ = ℱ₁ ‡ ℱ₂ ‡ …`, the full context the checker carries; entering a scope pushes a frame, leaving it pops the frame and kills every loan in its regions. **Closures** complicate this by snapshotting the current frame and suspending it inside their type (`τ -ℱ→ τ`), which keeps captured loans alive past their lexical scope — so the ownership-safety check must scan **every region in every frame across all of `Γ`, suspended ones included.** Taking a borrow is then just: scan `Γ` for a conflicting loan over the [[place-oxide|place]], and if there's none, write the new [[loan-oxide|loan]] into the [[region-oxide|region]]'s entry — that write is the reference's whole life, and clearing it is its death.

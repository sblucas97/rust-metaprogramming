# Flow-sensitive typing — *"type-checking the program changes the context as it goes"*

Here's a structural fact that quietly underlies everything in the other docs, and explains *why order matters* in Rust: **type-checking an expression doesn't just produce a type — it produces an updated context.**

If you've seen type systems before, you're used to a fixed context `Γ` that every part of an expression reads from. Rust isn't like that. Its context **evolves** as the checker walks down the program, statement by statement. This is called **flow-sensitive typing** (or *type-and-effect* / *flow typing*).

---

## The judgment: `Γ` in, `Γ'` out

The typing judgment in Oxide threads the context through:

```text
Γ ⊢ e : τ ⇒ Γ′
└┬┘       └┬┘ └┬┘
context   type  UPDATED context
going in        coming out
```

Read it as: *"in context `Γ`, expression `e` has type `τ`, and checking it transforms the context into `Γ′`."*

That little `⇒ Γ′` is the whole point. A classic type system would just say `Γ ⊢ e : τ` — context in, type out, context unchanged. Rust's says the context **comes out different**, and the *next* statement is checked against that new context.

```text
Γ₀ ⊢ stmt₁ ⇒ Γ₁ ⊢ stmt₂ ⇒ Γ₂ ⊢ stmt₃ ⇒ Γ₃ ...
     └── each statement hands its updated context to the next ──┘
```

---

## Why borrowing needs it

A borrow on one line affects the next line precisely *because* it mutates the context:

```rust
let mut pt = Point(6, 9);     // Γ₀

let a = &uniq pt.0;           // checking this WRITES loan `uniq pt.0` into Γ
                              //   ⇒ Γ₁  (now contains the live loan)

let b = &shrd pt.0;           // checked against Γ₁ — which knows about the uniq loan
                              //   ⇒ ❌ conflict
```

If the context didn't carry forward, line 3 would have no idea line 2 ever borrowed anything. The [[loan-oxide|loan]] minted on line 2 lives **in the context**, and flow-sensitivity is what makes it visible to line 3. Every "now `Γ` contains the loan" in the [[frame-oxide|frame]] and [[ownership-safety-oxide|ownership-safety]] docs is one of these `⇒ Γ′` steps.

---

## Why moves need it

[[ownership-oxide|Ownership]] is the same story. Moving a value flips a [[place-oxide|place]] from *live* to *dead* — and that fact has to travel forward:

```rust
let a = Point(6, 9);          // Γ₀:  a is LIVE

let b = a;                    // moves a → b.  ⇒ Γ₁:  a is DEAD
                              //   the deadness is recorded IN the context

let c = a.0;                  // checked against Γ₁, which says a is dead
                              //   ⇒ ❌ use of moved value
```

The checker knows `a` is dead on line 3 only because line 2's `⇒ Γ′` wrote that into the context. Flow-sensitivity is *how move-tracking works at all*.

---

## Branches: joining two output contexts

What happens when control flow splits? Each arm produces its **own** output context, and the checker **joins** them into one before continuing:

```rust
let x = if cond {
    &shrd pt          //  ⇒ Γ_then  (region of x = { shrd pt })
} else {
    &shrd other_pt    //  ⇒ Γ_else  (region of x = { shrd other_pt })
};
//  join(Γ_then, Γ_else)  ⇒  region of x = { shrd pt, shrd other_pt }
```

The **join** is exactly the [[region-oxide|region]]-union you already saw: where the two branches disagree about a region, the merged context takes the **union** (the conservative over-approximation). Likewise, a place is treated as **dead** after the `if` if *either* arm moved it. Joining is how flow-sensitivity copes with not knowing which branch ran.

---

## Where region GC fits in

Between these `⇒ Γ′` steps is also where [[region-oxide|loans get garbage-collected]]. After a statement finishes and its references are no longer used, the transition to the next context is the moment a region whose name is no longer mentioned gets dropped:

```text
Γ ⊢ *a += 1 ⇒ Γ′      // a's last use; in forming Γ′, region of a is GC'd,
                       // so its loan is gone from Γ′ onward
```

So `Γ ⇒ Γ′` isn't only about *adding* facts (new loans, new deaths) — it's also where *stale* facts are swept away.

---

## The mental model

Picture the context as a **living document** the checker edits as it reads top-to-bottom:

| As the checker reads…   | …it edits the context by                          |
| ----------------------- | ------------------------------------------------- |
| a [[loan-oxide\|borrow]]      | **adding** a loan to a region                     |
| a [[ownership-oxide\|move]]   | **marking** a place dead                          |
| the **last use** of a ref | **GC-ing** the region (removing its loans)      |
| an `if` / branch join   | **merging** both arms' contexts (union regions, dead-if-either) |
| a scope **exit**        | **popping** the [[frame-oxide\|frame]] (dropping all its regions) |

Every rule in every other doc is one of these edits. Flow-sensitive typing is the *spine* they all attach to: the reason a fact established on one line is known on the next.

---

## The one-paragraph summary

**Flow-sensitive typing** means the typing judgment is `Γ ⊢ e : τ ⇒ Γ′` — checking an expression yields not just a type but an **updated context**, which the next statement is checked against. This is *why order matters* in Rust: a [[loan-oxide|borrow]] is visible to later lines because it was written into the context, and a [[ownership-oxide|moved-out]] place is rejected later because its deadness was recorded the same way. Branches each produce their own output context and are **joined** (union the [[region-oxide|regions]], treat a place as dead if either arm moved it), and the transitions between contexts are also where stale loans get [[region-oxide|garbage-collected]] and exited [[frame-oxide|frames]] get popped. Every rule in the other docs is an *edit* to this evolving context; flow-sensitivity is the spine that carries each fact forward from the line that established it to the line that depends on it.

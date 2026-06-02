# Region — *"the set of all places a reference might have come from"*

A **region** answers one question: *"where could this reference possibly point?"* And the answer is concrete — a region is **literally a set of [[loan-oxide|loans]].**

This is the third and final layer of the model. Stacking what we already have:

| Concept                  | Answers                                  | Shape                          |
| ------------------------ | ---------------------------------------- | ------------------------------ |
| [[place-oxide\|Place]]   | *which box?*                             | a static path — `pt.0`         |
| [[loan-oxide\|Loan]]     | *who holds a key, and what kind?*        | `ω` + place — `uniq pt.0`      |
| **Region**               | *which loans could this reference be?*   | a **set of loans** — `{ uniq pt.0 }` |

> **Naming bridge.** The thing the [[place-oxide|place]] and [[loan-oxide|loan]] docs kept calling **provenance** *is* a region. In the Oxide formalism these are the same object: a *provenance variable* (written `'r`) names a *region*, and the region is the set of loans it stands for. So whenever you read "provenance," think "region," and vice-versa.

---

## A region is a set of loans

Every reference type in Oxide carries a region name. The full anatomy of a reference type:

```rust
let x = &'r uniq pt;     //  x : &'r uniq Point
```

```text
        &  'r  uniq  Point
        │   │    │     └── the type pointed at
        │   │    └──────── ownership qualifier  (shrd / uniq)  ← from the loan doc
        │   └───────────── region / provenance name           ← THIS doc
        └───────────────── it's a reference
```

The name `'r` is just a label. What matters is the **set it denotes**:

```text
'r  =  { uniq pt }
```

Read that as: *"the reference `x` was born from exactly one loan — a unique borrow of `pt`."*

---

## The simple case: one origin

When a reference is created by a single borrow, its region holds exactly one loan — there's no ambiguity about where it came from.

```rust
let x = &'r uniq pt;     // 'r = { uniq pt }
```

One borrow ⇒ one loan ⇒ a one-element region. The reference *clearly* came from `pt`, full stop.

---

## After a branch: many origins

The interesting case is when the checker **can't tell statically** which borrow produced the reference — for example, the two arms of an `if`:

```rust
let x = if cond {
    &'a uniq pt
} else {
    &'a uniq other_pt
};
// 'a = { uniq pt, uniq other_pt }
```

At runtime `x` came from *one* of them — but the compiler doesn't know which. So it does the safe thing: the region is the **union** of both possibilities.

```text
'a  =  { uniq pt }  ∪  { uniq other_pt }  =  { uniq pt, uniq other_pt }
```

Now `'a` carries **two** loans, and the checker treats **both as active simultaneously**. If later code touches `pt` *or* `other_pt`, it conflicts with `*x`. This is the same conservative *over-approximation* you saw with provenance: the region is a **superset** of the true runtime origin, which keeps the analysis sound at the cost of occasionally rejecting a safe program.

> This is exactly why a region is a **set** and not a single loan: it has to be able to say *"one of these, I'm not sure which,"* and a set is how you spell that.

---

## How regions flow through the program

Regions aren't conjured per-expression; they propagate by a handful of rules (identical to the provenance-flow rules from the [[place-oxide|place]] doc — because they're the same thing):

| Event                | Effect on the region                                                |
| -------------------- | ------------------------------------------------------------------- |
| **Borrow** `&pt`     | seeds a fresh region with one loan: `{ shrd pt }` / `{ uniq pt }`   |
| **Reborrow / copy**  | the new reference carries the **same** region along                 |
| **Join** (branches)  | the region is the **union** of the incoming regions                 |
| **Function boundary**| the region becomes an abstract parameter `'a`; the caller plugs in its own loan-set |

```rust
let r = &pt;            // r : &'1 shrd Point      '1 = { shrd pt }
let s = r;              // s : &'1 shrd Point      same region '1 (carried along)
let m = if c { &pt } else { &other_pt };
                       // m : &'2 shrd Point      '2 = { shrd pt, shrd other_pt }
```

> **Regions only ever grow by constraint, never shrink to dodge a clash.** The checker collects constraints of the form *"region `'a` must contain at least these loans"* (every use of a reference forces loans into its region) and then solves for the **smallest** sets satisfying all of them. If even those minimal sets produce a conflict, it's a genuine error — exactly the same "minimal regions forced by usage" logic behind [[loan-oxide|non-lexical lifetimes]].

---

## The big conceptual shift

This is the part that re-wires how you think about Rust lifetimes:

| Mental model    | What `'a` means                                          |
| --------------- | -------------------------------------------------------- |
| **Classic Rust**| *"the span of code where the reference is alive"* (a *time interval*) |
| **Oxide**       | *"the set of loans/places this reference could have come from"* (a *set of origins*) |

The classic model thinks of `'a` as a **stretch of time**. Oxide flips it: `'a` is a **set of origins**. Same symbol `'a`, completely different mental object — a set, not an interval.

And the payoff is that **NLL falls out for free** from the second framing.

---

## How NLL falls out — region garbage collection

Recall from the [[loan-oxide|loan]] doc: a loan should stay live only until the reference's **last use**. In the classic model you compute that with a dedicated *liveness analysis* over spans of code.

In the region model you don't need a separate analysis at all. A loan lives inside a region; a region is named in a reference's **type**; so the rule is simply:

> A region's loans can be **garbage-collected** the moment its name `'r` no longer appears in the type of **any live variable**. No liveness pass — just *"is this region still mentioned anywhere?"*

Worked through:

```rust
let mut pt = Point(6, 9);

let x = &'r uniq pt.0;   // x : &'r uniq i32     'r = { uniq pt.0 }
*x += 1;                 // last use of x.
                         //   ↳ after this, NO live variable's type mentions 'r
                         //   ↳ region 'r is collected → loan `uniq pt.0` vanishes

let y = &pt.0;           // ✅ fine: no live loan on pt.0 anymore
```

Why this *is* NLL: the loan `uniq pt.0` is reachable only through `'r`, and `'r` appears only in `x`'s type. The instant `x` stops being used, `'r` drops out of the set of live types, the region is collected, and its loan disappears. "Region no longer mentioned" and "reference no longer used" are the *same moment* — so the loan dies exactly at last use, which is precisely what non-lexical lifetimes promise.

| Classic NLL                          | Oxide regions                                  |
| ------------------------------------ | ---------------------------------------------- |
| Run a liveness analysis over the CFG | None needed                                    |
| Loan dies when its span ends         | Loan dies when its region is no longer mentioned |
| `'a` = interval of program points    | `'a` = set of loans, GC'd by reachability      |

---

## The full pipeline, end to end

All three docs are one machine:

```text
  &mut pt.0
      │  (a borrow expression)
      ▼
  mints a LOAN ........................  uniq pt.0          ← ownership ω + place
      │
      ▼
  dropped into a REGION ...............  'r = { uniq pt.0 } ← the set of possible origins
      │                                                       (a.k.a. provenance)
      ▼
  carried in the reference's TYPE .....  x : &'r uniq i32
      │
      │  later you write *x ...
      ▼
  region expands to its loans → places   { pt.0 }
      │
      ▼
  CONFLICT SCAN against every live loan  (overlap? + not both shrd?)  ← from the loan doc
```

- A **[[place-oxide|place]]** says *which box*.
- A **[[loan-oxide|loan]]** ties an ownership kind to that box.
- A **region** collects the loans a reference might be, rides along in its type, and is garbage-collected when the type goes out of use — which is what makes the whole thing enforce *"shared XOR mutable"* with NLL precision, no separate liveness pass required.

---

## The one-paragraph summary

A **region** (a.k.a. **provenance**, written `'r`) is a **set of [[loan-oxide|loans]]** — the over-approximated set of origins a reference might have come from. A single borrow seeds a one-loan region; branches **union** regions together; uses force loans in and the checker solves for the smallest sets that satisfy every constraint. The deep shift is that `'r` is *not* a span of time but a *set of origins*, and that reframing makes **non-lexical lifetimes** automatic: a loan lives exactly as long as some live variable's type still mentions its region, so it's collected — and the borrow released — the instant the reference falls out of use.

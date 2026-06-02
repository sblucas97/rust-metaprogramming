# Oxide: How the Borrow Checker Is Turned Into a *Proof*

A reading of **"Oxide: The Essence of Rust"** (Weiss, Gierczak, Patterson, Ahmed —
arXiv:1903.00982v4, Oct 2021), distilling **how Oxide formalises the borrow checker, lifetimes,
and the core safety rules** as an inductive type system you can actually prove sound.

Where [`DESCEND_ANALYSIS.md`](DESCEND_ANALYSIS.md) and
[`TYPE_SYSTEM_EXPLAINED.md`](TYPE_SYSTEM_EXPLAINED.md) describe a *GPU* type system that bolts
ownership onto a real compiler, Oxide is the upstream idea: it is the first type system that
**fully captures Rust's notion of ownership and borrowing**, proved sound with conventional tools
(progress + preservation), *without* separation logic. Descend's `Ownership`/`Provenance`/`Loan`
machinery is a direct descendant of what Oxide formalises here.

> **How to read this document.** Each section pairs an *intuition* (for a reader starting from zero)
> with the *formal object* (the grammar, judgment, or inference rule) that the paper actually proves
> things about. If the symbols look intimidating, read §0.5 first — it teaches you to read every
> rule in the paper in about five minutes.

---

## 0. The one-sentence thesis

> Rust's borrow checker is really a system for **statically building a proof that every piece of
> memory is either *uniquely owned* (and so safe to mutate unguarded) or *collectively shared*
> (and so safe to read).** Oxide makes that proof an *inductive typing judgment* instead of a
> constraint-solver, and a *region* (set of loan origins) instead of a "lifetime."

Everything below is an unpacking of that sentence.

---

## 0.5 A five-minute primer on the notation (read this first)

The paper is written in the language of **type systems / programming-language theory**. Three
ideas are enough to read every rule:

**1. A *judgment* is a claim of the form "in this context, this thing is true."**
The universal shape of Oxide's main claim is:

```
Σ ; Δ ; Θ ; Γ  ⊢  e : τ  ⇒  Γ′
```

The symbol `⊢` is a **turnstile**, read "**entails**" or "**proves**." Everything to its *left* is
*context* (what you may assume). Everything to its *right* is the *thing being claimed*. So the line
above reads: *"under the global env Σ, type env Δ, temporary typing Θ, and stack typing Γ, the
expression `e` has type `τ`, and produces an updated stack typing `Γ′`."* (The four contexts are
defined in §4.2; for now just know `⊢` separates "givens" from "conclusion.")

**2. An *inference rule* is "if the things on top hold, the thing on the bottom holds."**
It is written as premises stacked over a horizontal line over a conclusion:

```
        premise₁        premise₂        premise₃
       ─────────────────────────────────────────   (RULE-NAME)
                       conclusion
```

This is just implication drawn vertically: *premise₁ and premise₂ and premise₃ ⟹ conclusion.*
A rule with **nothing above the line** is an **axiom** — it holds unconditionally. To *typecheck a
program* is to stack these rules into a tree (a **derivation**) whose leaves are axioms and whose
root is "the whole program is well-typed." **That derivation tree *is* the proof Oxide is talking
about.**

**3. A handful of recurring symbols.**

| Symbol | Name | Read it as |
| --- | --- | --- |
| `⊢` | turnstile | "proves / entails" |
| `∀` , `∃` | for-all, exists | "every…", "there is some…" |
| `∈` | element-of | "is a member of" |
| `∅` | empty set | "the empty set / no loans" |
| `∪` , `⊎` | union, disjoint-merge | "combine the sets" |
| `⇒ Γ′` | output | "…and afterwards the environment is `Γ′`" |
| `↦` | maps-to | "this key is associated with this value" |
| `τ` , `ρ` , `ω` , `π` | tau, rho, omega, pi | a *type*, a *region*, an *ownership qualifier*, a *place* |
| `Γ` , `Δ` , `Σ` , `Θ` | Gamma, Delta, Sigma, Theta | the four environments (see §4.2) |
| `τ†` | dagger | "this type is *dead* (its value was moved out)" |
| `p̄` , `ℓ̄` , `r̄` | overbar | "a sequence/set of `p`s, loans, regions" |

That is the whole alphabet. Two theorems use it (Progress and Preservation, §5); everything in
between is rules that build the derivation tree they reason about.

---

## 1. What problem Oxide solves

| Prior work | Limitation Oxide fixes |
| --- | --- |
| `rustc`'s real borrow checker | Implemented as **constraint generation + an algorithmic solver** — operational, hard to reason about, no clean spec. |
| RustBelt (`λ_Rust`) | Semantic soundness via **separation logic in Iris**; built on lifetime *logic*. Continuation-passing, MIR-level — far from source Rust, costly to extend. |
| Stacked Borrows | Models `unsafe` raw pointers — orthogonal to the *static* semantics of safe Rust. |

Oxide's bet: model the **source program** (syntax close to surface Rust, with fully-annotated
types so there's no inference to muddy the semantics), drop traits and concurrency, and show the
borrow checker can be understood as **conventional inference rules** with a **standard** progress
& preservation proof. The headline novelty is *non-lexical lifetimes done as ordinary typing*.

The paper lists **five contributions** (§1.2), and it's worth knowing which is the deep one:
(1) the first formal account of *safe, surface* Rust; (2) — *the heart* — an **inductive
definition of borrow checking as ordinary inference rules** rather than a constraint solver;
(3) the *region-based alias management* view (regions manage *aliasing*, not memory);
(4) the first **syntactic** type-safety result for Rust (Wright–Felleisen progress + preservation);
(5) a **tested semantics** validated against `rustc`'s own test suite. Contribution (4) is the
"it's a proof" claim, and (2) is what makes (4) possible at all.

---

## 2. The conceptual shift: from "lifetimes" to **regions**

This is the single most important idea in the paper.

```
Rust mental model               Oxide mental model
─────────────────               ──────────────────
'a is a span of CODE            'a / ϱ is a REGION: a set of LOANS
("lines where the ref           ("the set of places this reference
 is live")                        could possibly have come from")

lifetime ≈ region of            region ≈ static, abstract grouping of
the program text                objects in memory a reference may point to
```

**Starting from zero:** in surface Rust you learn that `'a` is a *lifetime* — vaguely, "how long
the reference lives." Oxide throws that intuition out and replaces it with something more concrete
and more checkable. A reference doesn't carry "how long it lives"; it carries **the set of places
it might have been born from.** Knowing the *origins* of a reference is exactly what you need to
ask "could this reference and that one touch the same memory?" — which is the only question the
borrow checker ever really asks.

A **loan** `ᵂp` is the state recorded in the borrow checker when a reference is created: a
**place** `p` (an abstract memory location — `x`, `pt.0`, `*y`) tagged with an **ownership
qualifier** `ω ∈ {shrd, uniq}`. A **region** is a *set of loans*. Here is the formal vocabulary
straight from the term syntax (Fig. 1):

```
Regions              ρ  ::= ϱ | r              (abstract ϱ, or concrete r)
Ownership Quals.     ω  ::= shrd | uniq
Loans                ℓ  ::= ᵂp                 (a place tagged with an ownership)
Paths                q  ::= ε | n.q            (ε = empty; n.q = field n then more path)
Places               π  ::= x.q                (a variable, then a path — NO dereference)
Place Exprs.         p  ::= x | *p | p.n       (places, PLUS dereference *p)
```

The distinction `π` (place) vs `p` (place expression) matters: a **place** is a path into a value
that *does not* go through a pointer (`x`, `pt.0`), whereas a **place expression** may dereference
(`*y`, `(*y).0`). Borrowing acts on place expressions `p`; ownership-per-path acts on places `π`.

Worked example (the loan sets are the borrow-checker state after each line):

```
let mut pt = Point(6, 9);
let x = &'x uniq pt;     // after this line:  'x ↦ { uniq pt }
let z = &'z uniq *x;     // after this line:  'z ↦ { uniq x , uniq *x }
```

Key reframing the paper calls **region-based alias management**: regions are *not* about managing
memory (as in classic region calculi — Tofte/Talpin, Grossman). They have **no influence over
allocation**. They exist **purely so the borrow checker can rule out bad aliasing patterns**. A
concrete region maps to *all* the possible origins of a reference; the checker reasons over those
origin-sets instead of over program points. This is what makes **non-lexical lifetimes fall out
for free** — there's no "where is this live" analysis, just "what loans are in this set, and do
they conflict."

> Rust has since moved the real compiler in this direction too: Polonius uses *origins*, which are
> essentially Oxide's regions. The paper (§4.1, §5) notes Oxide is the *type-system analogue* of
> Polonius's *constraint-solving* approach to the same idea.

---

## 3. The four ideas that make ownership safe

### 3.1 Ownership qualifiers — `shrd` vs `uniq` (aliasing, not mutation)

Oxide annotates **every** borrow with `shrd` or `uniq` instead of Rust's `&` / `&mut`:

```
&'a shrd pt     ←→  &pt        (many readers, no writer)
&'a uniq pt     ←→  &mut pt    (exactly one usable name)
```

**Why rename `mut` to `uniq`?** Because the borrow checker is *not* about mutation — it's about
**aliasing** (how many names can reach the same memory at once). Two facts the paper uses to make
the point: `&&mut u32` (a shared ref to a unique ref) *cannot* be mutated, while `&Cell<u32>` (a
shared ref to a `Cell`) *can* be mutated through `Cell::set`. So "can I mutate?" does not track the
`mut` keyword; "how many live names point here?" does. The renaming makes the *real* invariant
visible. The conflict axiom is:

```
shrd + shrd  =  OK     (many readers)
shrd + uniq  =  ERROR  (reader/writer conflict)
uniq + uniq  =  ERROR  (two writers / aliased unique)
```

This three-line table is literally what the ownership-safety judgment (§4.3) enforces: a `uniq`
use must find **no** live loan over the same memory; a `shrd` use must find **no live `uniq`**
loan. (Formally captured by the `(ω = uniq ∨ ω' = uniq) ⟹ π' # π` premise of O-SafePlace, where
`#` means "disjoint / does not overlap.")

### 3.2 Places & fine-grained ownership

A **place** `π` names a path into a value — a variable, a tuple projection `pt.0`, a struct field.
Ownership is *per-place*, so disjoint paths can each be uniquely borrowed:

```
let mut pt = Point(6, 9);
let x = &'x uniq pt.0;   // unique loan on pt.0
let y = &'y uniq pt.1;   // unique loan on pt.1
// no error — pt.0 and pt.1 don't overlap
```

Two places **overlap** when one is a prefix of the other (the longer names a sub-part of the
shorter). Formally the paper writes `π' # π` for "`π'` is disjoint from `π`" (the negation of
overlap), and overlap is exactly the test the conflict rules use. Crucially, **indexing/slicing
`p[e]` / `p[e1..e2]` are *not* places** — an arbitrary index expression can't be statically
distinguished, so indexing takes ownership of the *whole* array. That's why you can split a tuple
two ways but not an array.

> Zero-knowledge gloss: "prefix" here is literal path-prefix. `pt` is a prefix of `pt.0`, so they
> overlap (one contains the other). `pt.0` and `pt.1` share no prefix relationship, so they're
> disjoint — and the checker is happy to hand out a unique loan on each.

### 3.3 Reborrowing — the source of subtlety

```
let mut pt = Point(6, 9);
let x = &'x uniq pt.0;
let y = &'y uniq *x;     // reborrow through x
// can use y; cannot use x until y is dropped
```

A reborrow's region carries **more than one loan** (e.g. `'z ↦ { uniq x, uniq *x }`) because the
source reference may itself have lost precision. The borrow checker tracks a **borrow chain** and a
**reborrow exclusion list** `π̄` so that a reborrow doesn't falsely conflict with the very
reference it was borrowed *from*. This `π̄` is the second annotation on the ownership-safety
turnstile — `⊢ω^π̄` — and exists *only* to handle reborrowing (see §4.3).

### 3.4 Non-lexical lifetimes via **loan garbage collection**

NLL is not a liveness analysis in Oxide — it's a metafunction `gc-loans_Θ(·)` applied while
sequencing (`T-Seq`) and at `let` (`T-Let`). It **empties the loan set of any region that no longer
appears in any type** in the stack typing `Γ` or temporary typing `Θ`. Once a reference's region is
GC'd, later borrows stop conflicting with it — exactly the "early drop" NLL allows. The companion
**kill rules** `Γ ▷ *π` (`T-Assign`) erase reborrowing relationships that an assignment invalidates.

> Zero-knowledge gloss: "non-lexical lifetime" means *a borrow can end before its `}`-scope ends,*
> as soon as it's provably never used again. Oxide gets this without dataflow analysis: between two
> statements, it just garbage-collects any region whose name has vanished from all live types. A
> region with an empty loan set conflicts with nothing — so the borrow it represented is, for all
> the checker cares, over.

---

## 4. The type system, formally (the part you'd actually port)

### 4.1 Types are split on **two axes**: sized-ness and initialised-ness

This is the type grammar (Fig. 2). Read `::=` as "is one of" and `|` as "or":

```
Kinds          κ   ::= ★ | RGN | FRM          ← kinds of types, regions, frame-typings
Base           τᴮ  ::= bool | u32 | unit
Sized+Init     τˢⁱ ::= τᴮ | α | &ρ ω τˣⁱ | (τ₁ˢⁱ … τₙˢⁱ) | [τˢⁱ; n]
                       | Either<τ₁ˢⁱ, τ₂ˢⁱ> | ∀<φ̄,ϱ̄,ᾱ>(τ₁ˢⁱ…) -ᶲ→ τᵣˢⁱ where ϱ₁:ϱ₂
Maybe-Unsized  τˣⁱ ::= τˢⁱ | [τˢⁱ]            ← a ref may point at a slice
Dead           τˢᴰ ::= τˢⁱ† | (τ₁ˢᴰ …)        ← "use-once" moved values
Maybe-Dead     τˢˣ ::= τˢⁱ | τˢᴰ | (τ₁ˢˣ …)   ← partially-moved aggregates
Types          τ   ::= τˣⁱ | τˢˣ
```

Why each split exists (the paper devotes §3.2 to this; here's the zero-knowledge version):

- **sized / unsized** mirrors Rust's `Sized` marker. A `let` binding must be sized (it lives on the
  stack, which needs a known size); a *reference* may point to an unsized slice `[τ]`. References
  are always pointer-sized regardless of pointee — that's why `τˣⁱ` ("maybe-unsized") is allowed
  *behind* a `&` but a bare `let` needs `τˢⁱ`.
- **dead / maybe-dead** (the `†` dagger) is how Oxide models **move semantics as linearity**: a
  moved value's type is marked dead (`τˢⁱ†`) and can't be used again. *Maybe*-dead `τˢˣ` handles
  *partially*-moved aggregates — e.g. a pair `(dead, alive)` after one field has been moved out.
  This is the type-level shadow of Rust's "use of partially moved value" errors.
- A **reference type** is `&ρ ω τˣⁱ` — it carries the **region `ρ`** and **ownership `ω`** right
  in the type. (Compare Descend's `RefDty { rgn, own, mem, dty }` — same idea, plus a `mem` axis
  for GPU.)
- A **function type** `∀<…>(…) -ᶲ→ τ where ϱ₁:ϱ₂` carries three unusual things: a **frame
  expression `Φ`** *over the arrow* (recording what a closure captured), polymorphism over type
  *and* region *and* frame variables, and `where ϱ₁:ϱ₂` **outlives bounds** (Rust's where-clauses).

> Zero-knowledge gloss on kinds: a *kind* is "the type of a type." Just as `5` has type `u32`, the
> type `u32` has kind `★` ("it is an ordinary type"), a region has kind `RGN`, and a frame-typing
> has kind `FRM`. Kinds keep you from writing nonsense like using a region where a type is expected.

### 4.2 The environments threaded through every rule

The typing judgment is:

```
Σ ; Δ ; Θ ; Γ  ⊢  e : τ  ⇒  Γ′
└┬┘ └┬┘ └┬┘ └┬┘            └┬┘
 │   │   │   │             └ OUTPUT stack typing (the continuation's view)
 │   │   │   └ Γ  stack typing: ordered frames ℱ of (binding → type) and (region → loan set)
 │   │   └ Θ  temporary typing: types of already-checked parts of a product
 │   └ Δ  type env: type / region / frame variables + outlives relations
 └ Σ  global env: top-level function & struct signatures
```

Their grammars (Fig. 3):

```
Global Env        Σ  ::= • | Σ, ε                       ε = a function definition
Type Env          Δ  ::= • | Δ, α:★ | Δ, ϱ:RGN | Δ, φ:FRM | Δ, ϱ:>ϱ′
Continuation Typ. Θ  ::= • | Θ, τˢⁱ
Stack Typing      Γ  ::= • | Γ ‡ ℱ                      ‡ separates ordered frames
Frame Typing      ℱ  ::= • | ℱ, x:τˢˣ | ℱ, r ↦ {ℓ̄}    bindings AND region→loans
```

The two unusual contexts:

- **Output `Γ′`**: typing is **flow-sensitive**. Every rule *returns an updated stack typing*
  because moving, borrowing, dropping and assigning all mutate what's live. This threading (input
  `Γ` on the left of `⊢`, output `Γ′` after `⇒`) is how Oxide gets non-lexical behaviour without a
  separate dataflow pass — the dataflow *is* the typing.
- **Temporary typing `Θ`**: holds types of already-checked components of a tuple/sequence so that a
  product of two `uniq` references to the same data can be accepted exactly when Rust accepts it
  (the loans get GC'd between sequence points). Concretely, `Θ` keeps an earlier component's type
  "alive" so `gc-loans` doesn't wrongly delete the region it depends on while a *later* component
  is checked.

`Γ` being an **ordered stack of frames** (separated by `‡`) matters: closures capture moved values
into a frame suspended over the function arrow `-ᶲ→`, and the FILO (first-in-last-out) order is
what makes `let`/`letrgn` scoping and `T-Drop` ("resource acquisition is initialisation")
well-defined. The well-formedness judgment for `Γ` (Fig. WF-StackTyping in App. B.1) literally
requires every loan in every frame to refer to an in-scope, ownership-safe place — so a
*well-formed environment already encodes the borrow-checker invariant.*

### 4.3 The heart: the **ownership-safety** judgment

This is the rule the whole borrow checker converges on (Fig. 4 in the paper). Its form:

```
Δ ; Γ ; Θ  ⊢ᵂ^π̄  p  ⇒  { ᵂp̄′ }
```

Read: *"in environments Δ, Γ, Θ, with reborrow-exclusion-list `π̄`, it is safe to use place
expression `p` `ω`-ly, producing the borrow chain `{ ᵂp̄′ }` (the loans this use depends on)."*
The shorthand `⊢ω` (no list) means `⊢ω^•` (empty exclusion list).

- if `ω = uniq`: there must be **no live loan** (of any ownership) against the memory `p` covers.
- if `ω = shrd`: there must be **no live *unique* loan** against it.

Here are the three rules, transcribed from Fig. 4 and lightly de-sugared for readability (the
`explode(Γ)`/`#`/`≲` machinery is spelled out underneath):

```
                ∀ (r̄ ↦ {ℓ̄}) ∈ regions(Γ,Θ).
                   ( ∀ ᵂ′π″ ∈ {ℓ̄}.  (ω = uniq ∨ ω′ = uniq) ⟹ π″ # π )   -- no conflicting loan
                 ∨ ( the conflicting reference's place is already in π̄ )   -- ...unless excluded
               ──────────────────────────────────────────────────────────  (O-SafePlace)
                Δ ; Γ ; Θ  ⊢ω^π̄  π  ⇒  { ω π }


  Γ(π) = &r ω_π τ_π     Γ(r) = { ᵂ′p′ }ⁿ     excl = π̄ ∪ {prefixes π_j that π was reborrowed from}
  ω ≲ ω_π     ∀ i ∈ 1..n.  Δ ; Γ ; Θ  ⊢ω^excl  p′ᵢ  ⇒  { ω p′ᵢ }     (recurse into r's loans)
               ──────────────────────────────────────────────────────────  (O-Deref)
                Δ ; Γ ; Θ  ⊢ω^π̄  p□[*π]  ⇒  { ω p′₁ … ω p′ₙ , ω p□[*π] }


  Γ(π) = &ϱ ω_π τ_π     Δ ; Γ ⊢ω p□[*π] : τ     ω ≲ ω_π     (abstract region: NO loans to inspect)
               ──────────────────────────────────────────────────────────  (O-DerefAbs)
                Δ ; Γ ; Θ  ⊢ω^π̄  p□[*π]  ⇒  { ω p□[*π] }
```

What each piece means, in words:

| Symbol | Meaning |
| --- | --- |
| `regions(Γ,Θ)` | every `region ↦ loan-set` mapping currently live in the environment |
| `π″ # π` | "place `π″` is disjoint from `π`" (no prefix-overlap → no conflict) |
| `ω ≲ ω_π` | ownership "fits": `shrd ≲ shrd`, `shrd ≲ uniq`, `uniq ≲ uniq` (reflexive closure of `shrd ≲ uniq`) — you can't use a `shrd` ref `uniq`-ly |
| `p□[*π]` | a place expression decomposed as *context* `p□` around an innermost dereference `*π` |
| `π̄` | the **reborrow exclusion list** — places we must *not* count as conflicts because the new borrow descends *from* them |

| Rule | Does |
| --- | --- |
| **O-SafePlace** | base case: a place `π` is `ω`-safe if, in every region, every loan either doesn't conflict (disjoint, or both `shrd`) **or** the offending reference is on the exclusion list `π̄`. |
| **O-Deref** | dereference through a **concrete** region `r`: look up `r`'s loans in `Γ`, recurse into them, append their reborrow prefixes to the exclusion list, union the resulting borrow chains. This is where the borrow chain grows beyond a single loan. |
| **O-DerefAbs** | dereference through an **abstract** region (a `'a` parameter of the enclosing fn): there are no concrete loans to inspect, so just check the inner place expression is well-typed and safe. |

> This judgment *is* the formal content of "Rust's borrow checker." Descend's `access_safety_check`
> → `access_conflict_check` is the operational shadow of these three rules. The reason the *proof*
> in §5 is hard is that **this** invariant has to survive every step of execution — and closures
> can smuggle loans across function boundaries.

### 4.4 Selected typing rules (Fig. 5) and what each *proves*

Here are the rules transcribed from Fig. 5 (some side-conditions abbreviated; the framebox in the
paper highlights the expression being checked). Each is followed by the one-line invariant it
establishes.

**Constants** (axioms — nothing above the line; the environment passes through unchanged):

```
─────────────────────────────────────   (T-u32)
 Σ ; Δ ; Γ ; Θ  ⊢  n : u32  ⇒  Γ
```

**Move** — consuming a non-copyable value:

```
 Δ ; Γ ; Θ ⊢_uniq π ⇒ { uniq π }      Γ(π) = τˢⁱ      noncopyable_Σ τˢⁱ
──────────────────────────────────────────────────────────────────────   (T-Move)
 Σ ; Δ ; Γ ; Θ  ⊢  π : τˢⁱ  ⇒  Γ[ π ↦ τˢⁱ† ]                            → linearity
```
*Moving `π` requires `π` is uniq-safe, sized, initialised & noncopyable; it marks `π`'s type DEAD
(`τˢⁱ†`) in the output `Γ`, so any later use fails to typecheck.*

**Copy** — reading a copyable value (note: leaves `Γ` untouched, and works on a *place expression*
`p`, so it can copy through a dereference):

```
 Δ ; Γ ; Θ ⊢_shrd p ⇒ { ℓ̄ }      Σ ; Δ ⊢ p : τˢⁱ      copyable_Σ τˢⁱ
──────────────────────────────────────────────────────────────────────   (T-Copy)
 Σ ; Δ ; Γ ; Θ  ⊢  p : τˢⁱ  ⇒  Γ
```

**Borrow** — the *birth of a loan* (the only rule that adds to a region's loan set):

```
 Γ(r) = ∅      Γ ; Θ ⊢ r rnic      Δ ; Γ ; Θ ⊢ω p ⇒ { ℓ̄ }      Δ ; Γ ⊢ω p : τˣⁱ
─────────────────────────────────────────────────────────────────────────────────   (T-Borrow)
 Σ ; Δ ; Γ ; Θ  ⊢  &r ω p : &r ω τˣⁱ  ⇒  Γ[ r ↦ { ℓ̄ } ]                          → birth of a loan
```
*`&r ω p` requires region `r` is **fresh** (`Γ(r)=∅`) and **not in a closure** (`r rnic`), checks
ownership safety to get the borrow chain `{ℓ̄}`, computes the pointee type, and **records `r ↦ {ℓ̄}`**
in the output `Γ`. This single rule is "creating a reference."*

**LetRegion** — introducing a fresh empty concrete region:

```
 Σ ; Δ ; Γ, r ↦ {} ; Θ  ⊢  e : τˢⁱ  ⇒  Γ′, r ↦ {ℓ̄}
──────────────────────────────────────────────────────   (T-LetRegion)
 Σ ; Δ ; Γ ; Θ  ⊢  letrgn <r> { e } : τˢⁱ  ⇒  Γ′
```
*`letrgn<r>{e}` binds a fresh region `r ↦ {}` for the scope of `e`, then drops it on the way out.*

**Sequencing** — *non-lexical drop* happens here, via `gc-loans`:

```
 Σ ; Δ ; Γ ; Θ ⊢ e₁ : τ₁ˢⁱ ⇒ Γ₁      Σ ; Δ ; gc-loans_Θ(Γ₁) ; Θ ⊢ e₂ : τ₂ˢⁱ ⇒ Γ₂
────────────────────────────────────────────────────────────────────────────────────   (T-Seq)
 Σ ; Δ ; Γ ; Θ  ⊢  e₁ ; e₂ : τ₂ˢⁱ  ⇒  Γ₂                                            → non-lexical drop
```
*Type `e₂` under `e₁`'s **output** `Γ₁`, but first run `gc-loans` to empty the loan sets of regions
no longer mentioned in any type. That clearing is exactly NLL's "early drop."*

**Let** — binding + scoping + RAII:

```
 Σ ; Δ ; Γ ; Θ ⊢ e₁ : τ₁ˢⁱ ⇒ Γ₁      Δ ; Γ₁ ; Θ ⊢ τ₁ˢⁱ ⤳ τₐˢⁱ ⊣ Γ₁′    (rewrite to annotation)
 ∀ r ∈ free-regions(τₐˢⁱ). Γ₁′ ⊢ r rnrb       (annotated regions not reborrowed)
 Σ ; Δ ; gc-loans_Θ(Γ₁′, x:τₐˢⁱ) ; Θ ⊢ e₂ : τ₂ˢⁱ ⇒ Γ₂, x:τˢᴰ      (x must be DEAD by scope end)
──────────────────────────────────────────────────────────────────────────────────────   (T-Let)
 Σ ; Δ ; Γ ; Θ  ⊢  let x : τₐˢⁱ = e₁ ; e₂ : τ₂ˢⁱ  ⇒  Γ₂                               → NLL + RAII
```
*Binds `x`; runs `gc-loans`; **requires `x`'s type be dead (`τˢᴰ`) by the end of `e₂`** — i.e. `x`
must have been moved out or explicitly `T-Drop`ped. That requirement is the formalisation of RAII /
"resource acquisition is initialisation" and the FILO scope discipline.*

**Assignment** — type-check the RHS, run the **kill rules** `▷ *π`, check uniq-safety:

```
 Σ;Δ;Γ;Θ ⊢ e : τˣⁱ ⇒ Γ₁     Γ₁(π) = τˢˣ     (τˢˣ = &r ω τ′ ⟹ r is unique to π in Γ₁)
 Δ ; Γ₁ ▷ *π ; Θ ⊢ τˢˣ ⤳ τˣⁱ ⊣ Γ′     ( τˢˣ = τˢᴰ  ∨  Δ ; Γ′ ; Θ ⊢_uniq π ⇒ { uniq π } )
──────────────────────────────────────────────────────────────────────────────────────   (T-Assign)
 Σ ; Δ ; Γ ; Θ  ⊢  π := e : unit  ⇒  Γ′[ π ↦ τˣⁱ ]
```
*`π := e`: check `e`, ensure assigning to `π` is sound (either `π` is currently dead, or it's
uniq-safe), and run `Γ ▷ *π` to **erase stale reborrows** that the overwrite invalidates. There is
a sibling rule **T-AssignDeref** for assigning *through* a dereference (`p := e`), identical in
shape but using the conservative `+` region-rewriting mode.*

**Branch** — both arms under the same input `Γ`, outputs merged with `⊎`:

```
 Σ;Δ;Γ;Θ ⊢ e₁ : bool ⇒ Γ₁
 Σ;Δ;Γ₁;Θ ⊢ e₂ : τ₂ˢⁱ ⇒ Γ₂      Σ;Δ;Γ₁;Θ ⊢ e₃ : τ₃ˢⁱ ⇒ Γ₃      τ₂ˢⁱ ∨ τ₃ˢⁱ = τ′ˢⁱ
 Δ;Γ₂;Θ ⊢ τ₂ˢⁱ ⤳ τ′ˢⁱ ⊣ Γ₂′      Δ;Γ₃;Θ ⊢ τ₃ˢⁱ ⤳ τ′ˢⁱ ⊣ Γ₃′      Γ₂′ ⊎ Γ₃′ = Γ′
──────────────────────────────────────────────────────────────────────────────────────   (T-Branch)
 Σ;Δ;Γ;Θ ⊢ if e₁ { e₂ } else { e₃ } : τ′ˢⁱ ⇒ Γ′                                       → join points
```
*Both arms are checked under the *same* `Γ₁`; their result types are unified to a common supertype
`τ′ˢⁱ` via region rewriting; their output environments are merged with `⊎` (per-region union of
loan sets; bound-variable types must match). The union is what makes a region after an `if`
*approximate* both possible origins.*

**Closure** — captures free vars into a suspended frame (simplified; full rule in Fig. 5):

```
 free-vars(e) \ x̄ = x̄_f      free-nc-vars(e) \ x̄ = x̄_nc        (free vars, and non-copyable ones)
 ℱ_c = (captured regions ↦ their loans), (x̄_f : their types)    (the captured frame)
 Σ ; Δ ; Γ[ x̄_nc ↦ Γ(x̄_nc)† ] ‡ ℱ_c , x₁:τ₁ˢⁱ … xₙ:τₙˢⁱ  ⊢  e : τᵣˢⁱ  ⇒  Γ′ ‡ ℱ
──────────────────────────────────────────────────────────────────────────────────────   (T-Closure)
 Σ;Δ;Γ;Θ ⊢ |x₁:τ₁ˢⁱ … xₙ:τₙˢⁱ| → τᵣˢⁱ { e } : (τ₁ˢⁱ…τₙˢⁱ) -ℱ→ τᵣˢⁱ  ⇒  Γ′    → closures alias safely
```
*A closure captures its free variables into a **frame `ℱ`** suspended over the arrow `-ℱ→`, and
marks captured non-copyable vars **dead** in the outer `Γ`. The captured frame stays live (so its
loans are still checked by ownership safety) — which is why the paper says closures "interact with
every other rule" and are the hardest part of the soundness proof.*

The recurring pattern, now visible across all the rules: **a rule's *premises* are safety checks
(ownership safety / freshness / copyability / RAII), and its *conclusion threads an updated `Γ′`*
recording the loan/death it just created.** Safety is enforced at the introduction site; the
consequences are carried forward in the output environment.

### 4.5 Region rewriting & outlives (Fig. 6) — making branches and bindings agree

Because the two arms of an `if` may annotate references with *different* regions
(`&'a uniq u32` vs `&'b uniq u32`), Oxide needs a **region rewriting** judgment to coerce one
type's regions into another's:

```
 Δ ; Γ ; Θ  ⊢^μ  τ₁ ⤳ τ₂  ⊣  Γ′        ("rewrite τ₁'s regions into τ₂'s, outputting Γ′")
```

It is **reflexive and transitive** and recurses structurally; the only interesting base case is
references, which defer to the **outlives** judgment. The mode `μ ∈ {+, ⊞, =}` controls how
conservative the rewrite is (`+` = combine and update output; `⊞` = unrestricted combine, used at
application; `=` = check only, don't change the output). Selected rewriting rules:

```
──────────────────────────   (RR-Refl)        Δ;Γ;Θ⊢^μ τ₁⤳τ₂⊣Γ′   Δ;Γ′;Θ⊢^μ τ₂⤳τ₃⊣Γ″
 Δ;Γ;Θ ⊢^μ τ₁ ⤳ τ₁ ⊣ Γ                       ───────────────────────────────────────  (RR-Trans)
                                                    Δ;Γ;Θ ⊢^μ τ₁ ⤳ τ₃ ⊣ Γ″

 Δ;Γ;Θ ⊢^μ ρ₁ :> ρ₂ ⊣ Γ′      Δ;Γ′;Θ ⊢^μ τ₁ ⤳ τ₂ ⊣ Γ″
──────────────────────────────────────────────────────────   (RR-Reference)
 Δ;Γ;Θ ⊢^μ &ρ₁ ω τ₁ ⤳ &ρ₂ ω τ₂ ⊣ Γ″
```

The **outlives** judgment `Δ ; Γ ; Θ ⊢^μ ρ₁ :> ρ₂ ⊣ Γ′` replaces Rust's `'a: 'b` constraints
(read `:>` as "outlives"). Key rules from Fig. 6:

```
 ϱ₁:RGN ∈ Δ   ϱ₂:RGN ∈ Δ   ϱ₁:>ϱ₂ ∈ Δ
──────────────────────────────────────────   (OL-BothAbstract)   -- both abstract: read off Δ
 Δ;Γ;Θ ⊢^μ ϱ₁ :> ϱ₂ ⊣ Γ


 Γ ⊢ r₁ rnrb    Γ ⊢ r₂ rnrb    r₁ occurs before r₂ in Γ    {ℓ̄} = Γ(r₁) ∪ Γ(r₂)
──────────────────────────────────────────────────────────────────────────────   (OL-CombineConcrete)
 Δ;Γ;Θ ⊢⁺ r₁ :> r₂ ⊣ Γ[ r₂ ↦ {ℓ̄} ]            -- two concrete, neither reborrowed: union the loans


 ϱ:RGN ∈ Δ    r ∈ dom(Γ)
────────────────────────────   (OL-AbstractConcrete)   -- abstract ALWAYS outlives concrete
 Δ;Γ;Θ ⊢^μ ϱ :> r ⊣ Γ
```

Why **OL-AbstractConcrete** holds *unconditionally* is a genuinely clever point worth slowing down
on: an abstract region `ϱ` is bound at the *top-level function* signature, while a concrete region
`r` can only be created by a `letrgn` *inside* the body. When the function is eventually applied,
whatever concrete region gets substituted for `ϱ` necessarily **already existed before** `r` was
ever created (even for recursive calls) — so `ϱ` outlives `r` by construction. No premise needed.

The `where ϱ₁ : ϱ₂` bounds on function types are exactly Rust's where-clauses, and they're what
lets a polymorphic function reborrow from one of several reference parameters soundly.

---

## 5. Why it's a *proof*, not just a checker (§3.7)

Oxide proves **syntactic type safety** the textbook way (Wright–Felleisen progress + preservation),
which is only possible because the whole thing is an inductive judgment, not a solver.

> Zero-knowledge gloss: "type safety" = *a well-typed program never gets stuck in a meaningless
> state* (e.g. dereferencing a dangling pointer, mutating aliased memory). The classic recipe has
> two halves. **Progress**: a well-typed program is either finished (a value) or can take another
> step — it's never *stuck*. **Preservation** (a.k.a. *subject reduction*): if a well-typed program
> takes a step, the result is *still* well-typed. Run them together and you get an induction: a
> well-typed program steps to a well-typed program, which steps to a well-typed program, … forever
> or until it produces a value — and at no point is it stuck. That chain *is* the safety guarantee.

The three core results, stated as in the paper:

```
Lemma 3.1 (PROGRESS)
  If   Σ; •; Γ; Θ ⊢ e : τˢⁱ ⇒ Γ′   and   Σ ⊢ σ : Γ ,
  then either  e is a value,
       or       e is an abort!(…),
       or       ∃ σ′, e′.  Σ ⊢ (σ; e) → (σ′; e′).      (e can take a step)
```
*Reads: if `e` typechecks under a stack typing `Γ`, and we have an actual runtime stack `σ` that
**satisfies** `Γ` (written `Σ ⊢ σ : Γ` — every value on the stack has the type `Γ` ascribes it),
then `e` is done, has aborted, or can step. It is never stuck.* Proved by induction on the typing
derivation, leaning on a Canonical Forms lemma and on Lemma 3.2.

```
Lemma 3.2 (PLACE EXPRESSIONS REDUCE)
  If   Δ; Γ ⊢ω p : τˣⁱ   and   Σ ⊢ σ : Γ ,
  then σ ⊢ p ⇓ R ↦ V[v]   and   Δ; Γ; Θ ⊢ v : τˣⁱ ⇒ Γ.
```
*A place expression `p` always evaluates (`⇓`) to a referent `R` holding a value `v` of the right
type. This is the "canonical forms for moves/copies/borrows/assignment" lemma — it's what lets
Progress conclude that `*x`, `pt.0`, etc. actually resolve to something at runtime.*

```
Lemma 3.3 (PRESERVATION)
  If   Σ; •; Γ; Θ ⊢ e : τ₁ˢⁱ ⇒ Γ_f   and   Σ ⊢ σ : Γ   and   Σ; Γ ⊢ v̄ : Θ
       and   Σ ⊢ (σ; e) → (σ′; e′) ,
  then ∃ Γ_i.  Σ ⊢ σ′ : Γ_i   and   Σ; Γ_i ⊢ v̄ : Θ
       and     Σ; •; Γ_i; Θ ⊢ e′ : τ₂ˢⁱ ⇒ Γ_f′
       and     τ₂ˢⁱ ; Γ_f′ ⊢⁺ τ₁ˢⁱ ⤳ Γ_s   and   ∃ Γ_o.  Γ_f = Γ_s ⊎ Γ_o.
```
*Reads: if `e : τ₁ˢⁱ` and it steps to `e′`, then there is an intermediate stack typing `Γ_i`
describing the new stack `σ′`, the temporaries `v̄` still satisfy `Θ`, and **`e′` is still
well-typed** — at a possibly *more precise* type `τ₂ˢⁱ` (related to `τ₁ˢⁱ` by region rewriting) and
with an output environment that recombines (`⊎`) into the original `Γ_f`. The flexibility in the
type/environment is needed because a step into one branch of an `if` legitimately sharpens the
regions.* **AND — the part that matters for the borrow checker — all aliasing invariants (ownership
safety) are maintained across the step.**

The paper is candid that **preservation is the hard, interesting theorem**, not progress: to keep
values on the stack well-typed *as the program runs*, you must show that **every
environment-mutating judgment — region rewriting, gc-loans, the kill rules, assignment, closures —
preserves ownership safety.** That obligation is discharged by a family of supporting lemmas, all
of the same shape *"values stay well-typed after ⟨environment operation⟩"*:

```
Lemma 3.4  Values are well-typed after REGION REWRITING.        (App. E.13)
Lemma 3.5  Values are well-typed after DROP / GC-LOANS.         (App. E.25)
Lemma 3.6  Values are well-typed under WELL-TYPED EXTENSIONS.   (App. E.34)
Lemma 3.7  Values are well-typed after ASSIGNMENT.             (App. E.40)
Lemma 3.8  Values are well-typed under SAFE LOAN UPDATES.       (App. E.62)
```

Each ultimately reduces to the ownership-safety judgment being **robust under the loan and
reborrow-exclusion-list changes** that the operation makes. Closures are the worst offender (they
introduce suspended aliasing via captured frames), and the authors argue a Rust formalism *without*
closures would miss the essence precisely because closures interact with every other rule.

So the "proof of the borrow checker" is literally:

> **ownership safety is an invariant preserved by every step of a well-typed program** (Lemma 3.3,
> via 3.4–3.8) ⟹ no datum ever has a live `uniq` loan aliased by anything else ⟹ unguarded
> mutation is sound ⟹ **memory safety + data-race freedom.**

Conventional type safety (no stuck states) then falls out as a *corollary* of Progress (3.1) +
Preservation (3.3). The paper stresses that Preservation is the more interesting of the two,
because it is where the *aliasing* invariants — not just the typing — are shown to hold throughout
execution.

---

## 6. Operational semantics & the "tested semantics" (§3.6, §3.8)

**Operational semantics** (Figs. 8–9): small-step, Felleisen–Hieb evaluation contexts over
configurations `(σ; e)` where `σ` is an **ordered stack** of frames `ς`. The dynamic syntax adds
*referents* `R` (abstract addresses = a variable + offsets: `x | R.n | R[n] | R[n₁..n₂]`) and
pointer values `ptr R`. Place-expression evaluation is written `σ ⊢ p ⇓ R ↦ V[v]` ("`p` resolves to
referent `R`, which holds value `v` inside context `V`"). Selected reduction rules:

```
 σ ⊢ π ⇓ _ ↦ [v]
────────────────────────────────────   (E-Move)     -- blank the slot with `dead`, yield v
 Σ ⊢ (σ; π) → (σ[π ↦ dead]; v)


 σ ⊢ p ⇓ _ ↦ [v]
──────────────────────────   (E-Copy)               -- leave the slot, yield v
 Σ ⊢ (σ; p) → (σ; v)


 σ ⊢ p ⇓ R ↦ _[_]
──────────────────────────────   (E-Borrow)         -- make a pointer to the referent
 Σ ⊢ (σ; &r ω p) → (σ; ptr R)


 σ ⊢ p ⇓ R ↦ V[_]      R = R□[x]
─────────────────────────────────────────────────   (E-Assign)
 Σ ⊢ (σ; p := v) → (σ[x ↦ V[v]]; ())


──────────────────────────────────────────   (E-Let)        ─────────────────────────   (E-Shift)
 Σ ⊢ (σ; let x:τ = v; e) → (σ, x↦v; shift e)               Σ ⊢ (σ, x↦v; shift e) → (σ; e)


 x̄_f = free-vars(e)   x̄_nc = free-nc-vars(e)   ς_c = σ|x̄_f
──────────────────────────────────────────────────────────────────────────────   (E-Closure)
 Σ ⊢ (σ; |x₁…| → τ {e}) → (σ[x̄_nc↦dead]; ⟨ς_c, |x₁…| → τ {e}⟩)
```

The **ordered stack** (`E-Shift`, `E-Framed`) is what makes scoping sound at runtime: `let` steps
to a `shift e` administrative form that pops the binding when the scope ends, and `E-AppClosure`
steps a closure body to a `framed e` form that drops the captured stack frame afterward. These rely
crucially on `σ` being *ordered* — they always touch the most-recent entry.

**Tested semantics** (§3.8): there's no official Rust spec to prove equivalence against, so Oxide is
*validated empirically*. The authors built **Reducer** (a compiler from a subset of real Rust →
Oxide) and **OxideTC** (a typechecker implementing these rules), then ran Rust's **own** test
suites (Fig. 10):

```
PASSING (typecheck identically to rustc):
  borrowck  89    nll  119    = 208 tests, all agree with rustc
DISQUALIFIED (out of scope by design, 407 total across 20 categories):
  traits 93 · heap 63 · enums 50 · statics/consts 40 ·
  out-of-scope libs 40 · uninitialized vars 40 · misc 81
```

All 208 in-scope tests from the `borrowck` and `nll` suites produce the *same* accept/reject (and
the same error category) as `rustc`. That's the evidence the inductive rules really capture Rust's
borrow checker on the supported subset. Structs are supported by treating them as tagged tuples.
The annotation burden is minor: `&'a uniq x` appears in real Rust as `#[lft="a"] &mut x`, and fresh
local regions are auto-generated, so most expressions need no change.

---

## 7. What Oxide deliberately leaves out

| Omitted | Why |
| --- | --- |
| **Type inference** | All bindings are type-annotated, so the *semantics* of borrowing isn't tangled with unification. (The authors note their judgment is *nearly* a bidirectional *synthesis* judgment, so this could be relaxed.) |
| **Traits / typeclasses** | Well-described elsewhere; orthogonal to ownership. |
| **Concurrency primitives** | The claim is borrow checking is understandable *without* them — data-race freedom is a *consequence*, not an input. |
| **Heap allocation, statics, generics over memory** | Oxide uses an *abstract* notion of memory — no concrete layout decisions — to focus on aliasing. |
| **n-ary enums** | Binary `Either<…>` + `match` suffices to model tagged sums. |

The minimal essence the paper isolates: **ownership (`shrd`/`uniq`) + regions-as-loan-sets +
an ordered stack + the ownership-safety judgment + flow-sensitive output environments**. Everything
else is sharpening.

> A nice framing from §5: people call Rust "an affine language," but Oxide shows the *substructural*
> story is subtler. `T-Drop` is a *weakening*-like rule (drop a binding early) — but uniquely, the
> binding must still be *present* (with a dead type) because of the ordering requirement. `T-Copy`
> is a *contraction*-like rule (use a value many times). So Rust is affine-*ish* with an ordered
> twist, not textbook-affine.

---

## 8. Oxide → this project (the bridge to Descend)

| Oxide concept | Where it reappears in our notes / Descend |
| --- | --- |
| `ω ∈ {shrd, uniq}` | `Ownership::{Shrd, Uniq}` ([`DESCEND_ANALYSIS.md`](DESCEND_ANALYSIS.md) §1.2) |
| Region `ρ` = set of loans | `Provenance::{Value, Ident}` + `PrvMapping { prv, loans }` |
| Loan `ᵂp` | `Loan { place_expr, own }` |
| Place `π`, prefix-overlap (`#`) | `Place` / `PathElem`, `possible_conflict` |
| Ownership-safety judgment (Fig. 4) | `borrow_check::access_safety_check` (`narrowing + conflict + borrow`) |
| `gc-loans` / kill rules (`▷ *π`) | (not yet built) — the NLL machinery to add |
| Ordered stack of frames `Γ` | `TyCtx` = stack of `Frame`s; `FrameEntry::{Var, ExecMapping, PrvMapping}` |
| `&ρ ω τ` reference type | `RefDty { rgn, own, mem, dty }` — **plus a `Memory` axis** Oxide doesn't need |
| `where ϱ₁ : ϱ₂` outlives (Fig. 6) | not yet needed (we hard-code `gpu.global`) |

The crucial extra axis Descend/our project adds on top of Oxide is **execution resources**
(`grid`/`block`/`thread`) and **memory spaces** — Oxide proves *sequential* aliasing safety;
Descend lifts the same ownership-safety judgment to *SIMT* aliasing by also requiring a
**narrowing check** on the exec under which each loan was taken. In other words:

```
Oxide:    ownership safety over regions               ⟹ memory safety (sequential)
Descend:  ownership safety over regions
          + memory-space agreement
          + narrowing over execution resources        ⟹ data-race freedom (parallel)
```

If you read Oxide first, Descend's design stops looking like new invention and starts looking like
"Oxide's ownership-safety judgment, indexed by a GPU execution resource."

---

## 9. Bottom line

Oxide's contribution is **methodological**: it shows the borrow checker is not inherently a
constraint-solver or a separation-logic artifact — it is an ordinary, **inductively-defined type
system** whose central judgment (*ownership safety over regions*, Fig. 4) is a **provable invariant**
of execution (Lemma 3.3). "Lifetimes" become **regions = sets of loan origins**; "non-lexical"
becomes **garbage collection of loan sets between expressions** (`gc-loans` in T-Seq/T-Let);
"memory safety / no data races" becomes a **corollary of preservation**. That reframing — *region-
based alias management* — is the foundation everything in this repo's type-checker work is built on.

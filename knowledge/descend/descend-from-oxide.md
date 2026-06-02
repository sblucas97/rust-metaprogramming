# Descend — *"Oxide's borrow checker, extended to make GPU programming data-race-free"*

> Köpcke, Gorlatch & Steuwer, *Descend: A Safe GPU Systems Programming Language* (2023).
> *"Our type system is based on the formalization of Rust's type system in **Oxide**."*

You've learned the Oxide model from zero: [[place-oxide|places]], [[loan-oxide|loans]], [[region-oxide|regions/provenance]], [[frame-oxide|frames]], [[ownership-oxide|ownership]], and [[ownership-safety-oxide|ownership safety]]. **Descend** takes that exact type system and *extends* it so it can prove a **GPU kernel — running across thousands of parallel threads — is free of data races and synchronization bugs, at compile time.**

The paper is explicit that it builds on Oxide and adds three ingredients (Section 3): **execution resources**, **extended place expressions** (with *select* and *views*), and **memory address spaces** — all tied together by an **extended borrow check** whose new rule is called **narrowing**. This doc walks each one, grounded in the paper's own examples.

---

## 0. The problem Descend solves

The paper's opening example is a CUDA matrix transpose (Listing 1). Each thread copies elements into a `__shared__` buffer, synchronizes, then copies out:

```c
__global__ void transpose(const double *input, double *output) {
  __shared__ float tmp[1024];
  for (int j = 0; j < 32; j += 8)
    tmp[threadIdx.y+j *32+threadIdx.x] = input[...];   // line 5: subtle bug
  __syncthreads();
  for (int j = 0; j < 32; j += 8)
    output[...] = tmp[(threadIdx.x)*32+threadIdx.y+j]; }
```

> *"List­ing 1 contains a subtle bug: in line 5, `threadIdx.y+j` should be enclosed by parenthesis… As a result, a data race occurs as multiple threads will write uncoordinated into the same memory location."*

The paper notes Rust *would* reject this (`tmp` is mutated by parallel threads) — **but only because Rust has no way to reason about safe parallel array access at all**, so it rejects even the *correct* version. Descend's goal is to **statically tell the safe parallel access from the racy one.** Everything below is the machinery for that.

---

## 1. What Descend inherits from Oxide (unchanged)

The paper reuses Oxide's core wholesale and only re-presents what changes. From Section 3.3:

> *"On the CPU, Descend implements exactly the same rules as Rust."*
> *"…the treatment of lifetimes has been formalized in Oxide… we omit here the presentation of lifetime variables that each reference carries."*

So these stay exactly as you learned them:

| Oxide concept | Status in Descend |
| --- | --- |
| [[place-oxide\|Place]] expressions as unique syntactic names for memory | kept — and **extended** (§4) |
| [[ownership-oxide\|Ownership]] `uniq` / shared (`&uniq` vs `&`) | kept — and **refined by narrowing** (§5) |
| [[loan-oxide\|Loans]] & the shared-XOR-unique conflict rule | kept inside the extended borrow check (§8) |
| [[region-oxide\|Regions / provenance / lifetimes]] | **kept unchanged but not re-presented** — "formalized in Oxide" |
| [[frame-oxide\|Frame]] / context `Γ`, [[flow-typing-oxide\|flow-sensitive typing]] | kept — the judgement is still `Γ ⊢ … ⊣ Γ'` (§7) |

**Example.** A Descend reference is written just like Oxide's, with `&` for shared and `&uniq` for unique:

```text
input:  &      gpu.global [[f64;2048];2048]   // shared  (read-only)
output: &uniq  gpu.global [[f64;2048];2048]   // unique  (writable)
```

That's Oxide's `&r ω τ` — the only visible addition so far is `gpu.global` (memory, §6). Provenance `r` is still there; the paper simply doesn't reprint it.

---

## 2. Ownership, lifted from "one thread" to "execution resources"

In Oxide/Rust, exclusive ownership belongs to a single thread. Descend's key generalization (Section 3.3):

> *"In Descend, each execution resource, such as the grid or a block, might take ownership of a memory object. Analogously, execution resources might create references, i.e., they might borrow. … The parameters of a function are owned by the function's execution resource."*

So [[ownership-oxide|ownership]] and [[loan-oxide|borrowing]] are now performed *by* an execution resource (a grid, a block, a thread), not by "the thread." This is the hook that lets the borrow checker reason about parallelism — it can ask *which execution resource* holds a given [[loan-oxide|loan]].

**Example.** In `transpose`, the parameter `output: &uniq gpu.global …` is owned by the **grid** (the function's execution resource, declared in its signature). Before a single thread can write into it, that ownership must be *narrowed* down the hierarchy (§5).

---

## 3. Execution resources — the new "who runs this" (Section 3.1)

Oxide has no notion of *who* executes code; in a sequential program it's trivially "the thread." A GPU has a *hierarchy* of executors, and Descend makes it a first-class object — the **execution resource** `e` (Figure 2):

```text
e ::= cpu.thread
    | gpu.grid⟨d,d⟩                  -- the whole launch (blocks × threads dims)
    | e.forall(X|Y|Z)                -- schedule the same code over all sub-resources
    | e.split(η, [X|Y|Z]).[fst|snd]  -- split into two independent sub-groups
```

You navigate the grid → block → thread tree with two surface constructs: **`sched`** (forall) and **`split`**.

**Example (paper, Section 3.1).** A 3-D grid of 2×2×1 blocks, each 4×4×4 threads:

```text
fn foo(...) -[grid: gpu.Grid<XYZ<2,2,1>, XYZ<4,4,4>>]-> () {
  sched(X,Z) blocks in grd {       // schedule over blocks sharing X,Z coords
    split(Y) blocks at 1 {         // split the Y dimension into two subgroups
      fstBlock => ...
      sndBlock => ... } } }
```

The paper gives two purposes for execution resources: **(1)** check what runs on **CPU vs GPU**, and **(2)** check *which* part of the hierarchy runs an instruction — e.g. *"a barrier synchronisation must be executed inside a block."*

---

## 4. Place expressions — extended with *select* and *views* (Section 3.2)

This is the biggest change to a concept you already know. Oxide [[place-oxide|places]] are `x`, `p.fst`/`p.snd`, `*p`, `p[t]`. Descend **adds two forms** (Figure 3):

```text
p ::= x                  variable          ┐
    | p.fst | p.snd      projections       │ inherited from Oxide
    | *p                 dereference       │
    | p[t]               index             ┘
    | p[[e]]             select            ← NEW: pick the slice for exec resource e
    | p.v::⟨η̄,δ̄⟩(v)      view              ← NEW: reshape/reorder how the array is accessed
```

The reason this matters is the *Oxide* reason — places are compared **syntactically** to guarantee no two overlapping places are mutated at once:

> *"Place expressions [are] unique names for a region of memory. … This allows them to be compared syntactically … so the same memory location is never (mutably) accessed at the same time. Through this, it can be guaranteed that no data races occur."*

**Select `p[[e]]`** picks, for each sub-execution-resource, *its own* element of an array:

> *"The select expression selects memory for an execution resource from an array. … each sub-execution resource accesses one element, providing a safe concurrent array access."*

But select alone is rigid (you can't pick groups, or reorder). So Descend adds **views**.

**Views** reshape or reorder an array *without moving data* — they only remap *which executor touches which element*. The basic views and their types (Listing 3):

```text
split   : [[d;n]] -> ([[d;k]], [[d;n-k]])        where n >= k
group   : [[d;n]] -> [[ [[d;k]]; n/k]]           where n % k == 0
transpose: [[ [[d;n]]; m]] -> [[ [[d;m]]; n]]
reverse : [[d;n]] -> [[d;n]]
map     : (([[d1;n]])->[[d2;m]], [[ [[d1;n]];m]]) -> [[ [[d2;m]];n]]
```

Views **compose** into richer ones, e.g. `group_by_row`:

```text
view group_by_row<row_size, num_rows> = group::<row_size/num_rows>.map(transpose)
```

> *"Descend statically checks that accesses into views are safe… the borrow checking of Descend is capable to statically determine that the parallel write access into the shared temporary buffer and the output are safe."* — this is exactly what fixes the Listing 1 race.

**Example (paper, Figure 4 / Listing 2).** The full place expression a thread writes is built from views + select:

```text
input.group::<32,32>.transpose[[block]].group_by_row::<32,4>[[thread]][i]
```

Read it right-to-left as a pipeline: reshape the 32×32 array (`group`+`transpose`), let each **block** select its tile (`[[block]]`), regroup by row, let each **thread** select its row (`[[thread]]`), index element `i`. Because every step is a *syntactic* remapping, the checker can prove no two threads' place expressions overlap — *safe by construction*. Concretely, `x.split::<32>.fst` and `x.split::<32>.snd` are syntactically **distinct** (disjoint halves), while both overlap `x`.

---

## 5. Narrowing — the new borrow-check rule (Section 3.3)

Ownership of an array starts at the **grid** and must be *refined* as you descend the hierarchy. The paper:

> *"Narrowing describes how ownership and borrows are refined when navigating the execution hierarchy from grid, to blocks and threads. … the ownership of an array by a grid is narrowed to the grid's blocks (each block a distinct part), [then] further to the block's threads."*

The rule it enforces: **a `uniq` (exclusive) access is only legal if each execution resource ends up with its own *distinct* part.** If a refinement would give *every* sibling resource exclusive access to the *same* memory, narrowing is **violated**.

**Example — two violations and the fix (paper kernel, Section 3.3):**

```text
fn kernel(arr: &uniq gpu.global [f32; 1024])
        -[grd: gpu.Grid<X<32>, X<32>>]-> () {
  sched(X) block in grd {
    let in_borrow = &uniq *arr;                   // ❌ narrowing violated:
                                                   //    each block would get uniq
                                                   //    write to the ENTIRE array
    sched(X) thread in block {
      let grp = &uniq arr.group::<32>[[thread]];   // ❌ narrowing violated:
                                                   //    selection done per block but
                                                   //    NOT for the block itself, so
                                                   //    threads of different blocks
                                                   //    hit the same memory
      arr.group::<32>[[block]][[thread]];          // ✅ correct narrowing:
    } } }                                          //    block selects its part, THEN
                                                   //    each thread selects an element
```

The first line dereferences the whole array uniquely inside every block → every block claims all of `arr`. The fix `arr.group::<32>[[block]][[thread]]` selects **per block first**, then **per thread** — disjoint all the way down. This is the [[place-oxide|place-disjointness]] idea from Oxide, now applied *across the execution hierarchy* via select.

---

## 6. Memory address spaces — *new in Descend vs. Oxide* (Section 3.4)

This is the axis you asked about, and the paper confirms it's an Oxide extension:

> *"The physically separated memories of CPU and GPU are reflected in the types of references for which Descend enforces that they are only dereferenced in the correct execution context."*
> *"In Descend, all references carry an address space."*

The address spaces (Figure 6, `μ`):

```text
μ ::= cpu.mem      -- CPU stack and heap
    | gpu.global   -- slow, large; accessible by every thread in the grid
    | gpu.shared   -- fast; accessible only within one block
    | m            -- a memory variable (polymorphism over address spaces)
```

The reference type grows from Oxide's `&|uniq| δ` to Descend's **`&|uniq| μ δ`** — a memory annotation `μ` slots in. References may only be dereferenced **in the matching execution context**, so a GPU reference can't be read on the CPU.

**Example 1 — wrong memory space (paper, Section 2.3).** Passing a CPU buffer where a GPU one is expected:

```text
copy_mem_to_host(d_vec, h_vec);
// error: mismatched types
//        expected reference to `gpu.global`, found reference to `cpu.mem`
```

**Example 2 — dereferencing CPU memory on the GPU (paper):**

```text
sched(X) thread in grid {
  (*vec)[[thread]] = 1.0
// error: cannot dereference `*vec` pointing to `cpu.mem`, executed by `gpu.Thread`
```

**Example 3 — allocation via boxed `@`-types (paper, Section 3.4).** A value of type `δ @ μ` is a smart pointer recording *which* address space it lives in (`T @ cpu.mem` ≈ Rust's `Box<T>`):

```text
{ let cpu_array: [i32;n] @ cpu.mem = CpuHeap::new([0;n]);
  { let global_array: [i32;n] @ gpu.global = GpuGlobal::alloc_copy(&cpu_array);
  } // free global_array  (smart pointer dropped at end of scope)
} // free cpu_array
```

> **Why Oxide didn't need this:** Oxide models a single sequential address space — there's only "memory," nothing to tag. Once CPU, global, and shared memories coexist, every reference needs a label saying which one it points into.

---

## 7. The extended type system (Section 4)

The formal pieces, all "based on Oxide" and extended.

**Types (`δ`, Figure 6):**

```text
δ ::= i32 | … | unit         scalar
    | (δ1, …, δn)             tuple
    | [δ; η] | [[δ; η]]       array  | array-view   ← view arrays needn't be contiguous
    | &|uniq| μ δ             reference   ← μ (memory) added vs Oxide
    | δ @ μ                   boxed / @-type (smart pointer + address space)
    | x                       type variable
```

**Execution levels (`ε`):** `cpu.Thread | gpu.Grid d d | gpu.Block d | gpu.Thread` — the *kind* of executor a function or statement runs at.

**The typing judgement (Section 4.3)** — note it is still [[flow-typing-oxide|flow-sensitive]], exactly like Oxide (`Γ` in, `Γ'` out), with new environments threaded through:

```text
Δ; Γg; Γl; Θ | ef:ε ; e | A ⊢ t : δ ⊣ Γ'l | A'
```

| Symbol | Holds | Oxide analog |
| --- | --- | --- |
| `Δ` | kinds of type variables | kinding |
| `Γg` | types of global functions | global env |
| `Γl` | local variables **and active borrows** | the [[frame-oxide\|frame stack `Γ`]] |
| `Θ` | temporary borrows | — |
| `ef:ε` | execution resource running the **function** + its level | — (new) |
| `e` | execution resource running the **current statement** | — (new) |
| `A` | **which execution resource accesses which place expressions** | — (new; the parallel guard) |
| `Γ'l`, `A'` | updated contexts out | the `⇒ Γ'` of [[flow-typing-oxide\|flow typing]] |

> *"the typing judgement is flow-sensitive… when accessing an owned value we are not allowed to access it again (as it has been moved) and therefore it is removed from the typing environment."* — that's Oxide's [[ownership-oxide|move]] tracking, verbatim.

The genuinely new environment is **`A`**, the *access environment*, keyed by **execution resource**. It records that exec `e` accessed place `p` (shared or unique) — and the cross-thread safety check reads it.

---

## 8. The extended borrow check — `access_safety_check` (Section 4.3, Figure 7)

Reads and writes go through two typing rules, each invoking the new check:

- **T-Read-By-Copy** — reading a *copyable* value `p`:
  `{ shrd pᵢ } = access_safety_check(p, shrd, …)`, requires `is_copyable(δ)`, then records the shared accesses into `A(e)`.
- **T-Write** — assignment `p = t`:
  `{ uniq pᵢ } = access_safety_check(p, uniq, …)`, then records the unique access into `A(e)`.

`access_safety_check` is **three logical steps** (the paper's own list):

1. **Narrowing check** (§5) — *"check if the place expression is accessed uniquely by multiple execution resources… each execution resource selects its own distinct part."*
2. **Access conflict check** — *"check that using a place expression in an execution resource does not conflict with previous accesses by other execution resources stored in the access mapping environment `A`."*
3. **Borrow checking** — *"the unchanged borrow checking as in Rust and as formalized in Oxide."* ← this is your [[ownership-safety-oxide|ownership-safety]]/[[loan-oxide|loan-conflict]] check, untouched.

Steps 1–2 are the new parallel guards; step 3 is pure Oxide.

**Example — a race the access-conflict check catches (paper `rev_per_block`, Section 2.2):**

```text
// CUDA: block_part[threadIdx.x] = block_part[blockDim.x-1 - threadIdx.x];
// Descend rejects the equivalent:
arr[[thread]] = arr.rev[[thread]];
// error: conflicting memory access
//        cannot select memory because a conflicting prior selection here
```

One thread reads element `n-1-i` while another writes element `i` of the *same* block array → the place expressions `arr[[thread]]` and `arr.rev[[thread]]` overlap across executors → **rejected**.

### Synchronization clears the access environment

Sometimes threads *should* revisit shared memory — after a barrier. The paper:

> *"On a synchronization, we remove from the mapping the previous accesses of threads in the block from `A`."*

So `sync` wipes the block's entries in `A`, legitimizing post-barrier access. And the barrier itself is checked via execution resources — it must be reached by **all** threads in a block:

**Example — misplaced barrier (paper, Section 2.2):**

```text
// error: barrier not allowed here
//   split(X) block at 32 { first_32_threads => { sync } … }
//   `sync` not performed by all threads in the block
```

A `sync` nested inside a `split` (so only some threads reach it) is forbidden — exactly the CUDA `if (threadIdx.x < 32) __syncthreads()` undefined-behavior bug, now a compile error.

---

## How it all maps back to Oxide

| Oxide | Descend | New? |
| --- | --- | --- |
| [[place-oxide\|place]] `x`, `.fst`, `*p`, `p[t]` | + **select `p[[e]]`** + **views `p.v`** | extended |
| [[ownership-oxide\|ownership]] `&`/`&uniq`, by one thread | owned/borrowed **by an execution resource**; refined by **narrowing** | extended |
| [[loan-oxide\|loan]] conflict (shared XOR unique) | unchanged — step 3 of `access_safety_check` | kept |
| [[region-oxide\|region/provenance]], lifetimes | unchanged — "formalized in Oxide", omitted from presentation | kept |
| [[frame-oxide\|frame]] `Γ`, [[flow-typing-oxide\|flow-sensitive]] `Γ⇒Γ'` | `Γl` + the new **access environment `A`** (per exec resource) | extended |
| reference type `&|uniq| δ` | `&|uniq| μ δ` — **memory address space `μ`** | extended |
| — | **execution resources** `e` and **levels** `ε` | brand new |

---

## The one-paragraph summary

**Descend** is Oxide's type system extended to make GPU programming data-race-free. It keeps Oxide's core untouched — [[place-oxide|place]] expressions compared syntactically, [[ownership-oxide|shared-XOR-unique]] ownership, [[loan-oxide|loans]], [[region-oxide|provenance/lifetimes]] ("formalized in Oxide," omitted from the paper), and the [[flow-typing-oxide|flow-sensitive]] judgement — and layers on three new ingredients: **execution resources** (the grid→block→thread tree, navigated with `sched`/`split`) that make ownership belong to *an executor* rather than *the thread*; **extended place expressions** with **select** `p[[e]]` (give each sub-executor its own element) and **views** (reshape/reorder access without moving data, `group`/`transpose`/`reverse`/…), so the checker can prove parallel accesses are disjoint *by construction*; and **memory address spaces** `μ` (`cpu.mem`/`gpu.global`/`gpu.shared`) carried on every reference (`&|uniq| μ δ`) and enforced so references are only dereferenced in the right execution context. The extended borrow check, `access_safety_check`, adds two parallel guards — **narrowing** (exclusive access must refine to *distinct* parts down the hierarchy) and an **access-conflict check** against a new per-execution-resource access environment `A` (cleared by `sync` after a barrier) — on top of Oxide's *unchanged* borrow check. The payoff is the matrix-transpose example: Descend statically accepts the correct parallel version and rejects the racy one — something neither CUDA nor plain Rust can do.

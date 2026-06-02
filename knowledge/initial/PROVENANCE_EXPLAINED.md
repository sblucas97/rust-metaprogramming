# Provenance: The Borrow Checker's Reference Identity

Understanding why descend uses `Provenance` in `RefDty` and how it prevents data races.

---

## What Is Provenance?

**Provenance** is a name or identity tag that tracks *which reference* a loan came from.

In Rust's type system:
```rust
fn example<'a>(r: &'a i32) { ... }
```

The lifetime `'a` is Rust's way of saying "this reference is from region `a`". In descend, `Provenance` serves the same role — it names the region a reference belongs to.

```rust
// Descend equivalent:
RefDty {
    rgn: Provenance::Value("a"),  // This ref belongs to region "a"
    own: Ownership::Shrd,
    mem: Memory::CpuMem,
    dty: Box::new(Ty::Scalar(I32))
}
```

---

## The Problem Provenance Solves

### Without Provenance (Naive Borrow Checker)

Imagine a type system that just tracks ownership without naming references:

```rust
let r1: &mut i32 = ...;  // Mutable reference
let r2: &mut i32 = ...;  // Another mutable reference

*r1 = 5;  // Write through r1
*r2 = 10; // Write through r2
let x = *r1;  // Read through r1 — might be wrong if r1 and r2 alias!
```

**Problem:** The checker doesn't know if `r1` and `r2` point to the same place. Are they the same reference? Two different references to the same data? The checker can't tell without names.

### With Provenance (Named References)

```rust
let r1: &'r1 mut i32 = ...;  // Provenance "r1"
let r2: &'r2 mut i32 = ...;  // Provenance "r2"

*r1 = 5;   // Loan from "r1"
*r2 = 10;  // Loan from "r2" — different provenance
let x = *r1;  // Reuse provenance "r1" — same region, OK
```

**Solution:** Each reference gets a unique provenance name. The checker can now ask:
- "Is this loan from the same reference as the last one?" → Look up the provenance
- "Can I reuse this reference?" → Same provenance = yes

---

## Provenance in Your Kernel

### The Vector Add Kernel

```rust
#[kernel]
pub fn add_vectors(
    a: &CudaVec<f32>,           // Provenance: "a"
    b: &CudaVec<f32>,           // Provenance: "b"
    result: &mut CudaVec<f32>   // Provenance: "result"
) {
    let idx: u64 = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = a[idx] + b[idx];
}
```

### How Provenance Is Used

**Stage 1: Build Context**

```rust
TyCtx {
    frames: [
        Frame {
            bindings: [
                IdentTyped {
                    ident: "a",
                    ty: RefDty {
                        rgn: Provenance::Value("a"),  // ← Named reference
                        own: Ownership::Shrd,
                        ...
                    },
                    exec: GpuThread
                },
                IdentTyped {
                    ident: "b",
                    ty: RefDty {
                        rgn: Provenance::Value("b"),  // ← Different name
                        own: Ownership::Shrd,
                        ...
                    },
                    exec: GpuThread
                },
                IdentTyped {
                    ident: "result",
                    ty: RefDty {
                        rgn: Provenance::Value("result"),  // ← Yet another
                        own: Ownership::Uniq,
                        ...
                    },
                    exec: GpuThread
                },
            ]
        }
    ]
}
```

**Stage 2: Create Provenance Mappings**

When type checking starts, the checker creates a `PrvMapping` for each provenance:

```rust
TyCtx {
    frames: [
        Frame {
            bindings: [
                // ... IdentTyped bindings ...
                
                PrvMapping {
                    prv: "a",
                    loans: HashSet::new()  // Initially empty
                },
                PrvMapping {
                    prv: "b",
                    loans: HashSet::new()
                },
                PrvMapping {
                    prv: "result",
                    loans: HashSet::new()
                },
            ]
        }
    ]
}
```

**Stage 3: Record Loans By Provenance**

When accessing `a[idx]`:

```rust
// Type of a:
RefDty {
    rgn: Provenance::Value("a"),
    own: Ownership::Shrd,
    ...
}

// Borrow checking creates:
Loan {
    place_expr: Index(Var("a"), Var("idx")),
    own: Ownership::Shrd
}

// And stores it in:
PrvMapping {
    prv: "a",
    loans: { Loan { a[idx], Shrd } }  // ← Tied to provenance "a"
}
```

When accessing `b[idx]`:

```rust
// Different provenance, different loan set:
PrvMapping {
    prv: "b",
    loans: { Loan { b[idx], Shrd } }  // ← Tied to provenance "b"
}
```

**Stage 4: Check Conflicts**

When accessing `result[idx]`:

```rust
// Check: Can I create a new loan on result?
access_safety_check(ctx, PlaceExpr::Index(Var("result"), Var("idx")))
  └─ access_conflict_check:
     - For each existing Loan in AccessCtx:
       * Loan { a[idx], Shrd } from provenance "a"
       * Loan { b[idx], Shrd } from provenance "b"
     - New access: result[idx] with Uniq from provenance "result"
     - Check conflicts:
       * a[idx] (different ident) vs result[idx] → ✓ no conflict
       * b[idx] (different ident) vs result[idx] → ✓ no conflict
       * BUT if result[idx] was accessed twice:
         └─ First loan from provenance "result": { result[idx], Uniq }
         └─ Second loan from same provenance "result": { result[idx], Uniq }
         └─ ✗ CONFLICT: Cannot have two Uniq loans from same provenance
```

---

## Key Insight: Provenance Enables Reference Reuse

### Same Provenance = Same Reference

```rust
// This is OK:
let r: &i32 = ...;
let x = *r;  // First use
let y = *r;  // Second use — same reference, same provenance

// Both uses share the same provenance "r"
// Loans don't accumulate; they're in the same region
```

**In descend's terms:**

```rust
IdentTyped {
    ident: "r",
    rgn: Provenance::Value("r"),
    ...
}

// Access 1: *r
access_safety_check(ctx, PlaceExpr::Deref(Var("r")))
  └─ Create Loan { *r, Shrd } in PrvMapping("r")

// Access 2: *r (same ident, same provenance)
access_safety_check(ctx, PlaceExpr::Deref(Var("r")))
  └─ Create Loan { *r, Shrd } in PrvMapping("r")
  └─ Same provenance, same place → merged or deduplicated
```

### Different Provenance = Different References

```rust
// This is NOT the same:
let r1: &i32 = ...;
let r2: &i32 = ...;
let x = *r1;
let y = *r2;

// r1 has provenance "r1", r2 has provenance "r2"
// Even if they point to the same data, they're tracked separately
```

---

## Why Provenance Matters for Preventing Data Races

### Scenario 1: Reusing the Same Reference (Safe)

```rust
let a: &CudaVec<f32> = input;

result[0] = a[0] + a[0];  // Use 'a' twice
```

**Type:**
```rust
a: RefDty {
    rgn: Provenance::Value("a"),
    own: Ownership::Shrd,
    ...
}
```

**Checking:**
```
Access 1: a[0]
  └─ Create Loan { a[0], Shrd } in PrvMapping("a")

Access 2: a[0]
  └─ Check: Can I loan from "a" again?
  └─ Previous loan from "a": Shrd
  └─ New loan: Shrd
  └─ Shrd + Shrd = OK ✓ (multiple readers allowed)
  └─ Same provenance reuse: OK ✓
```

**Result:** ✅ SAFE — Same reference can be read multiple times

---

### Scenario 2: Two Different Mutable References (Unsafe)

```rust
fn swap<'a, 'b>(r1: &'a mut i32, r2: &'b mut i32) {
    *r1 = *r2;  // Read from r2, write to r1
}
```

**Types:**
```rust
r1: RefDty {
    rgn: Provenance::Value("r1"),
    own: Ownership::Uniq,
    ...
}

r2: RefDty {
    rgn: Provenance::Value("r2"),
    own: Ownership::Uniq,
    ...
}
```

**Checking:**
```
Access 1: *r2 (read)
  └─ Create Loan { *r2, Uniq } in PrvMapping("r2")

Access 2: *r1 (write)
  └─ Create Loan { *r1, Uniq } in PrvMapping("r1")

Conflict check:
  └─ PrvMapping("r2") has Loan { *r2, Uniq }
  └─ PrvMapping("r1") has Loan { *r1, Uniq }
  └─ Different provenances, different places → ✓ No conflict
```

**Result:** ✅ SAFE (as long as they don't alias at runtime)

---

### Scenario 3: Single Mutable Reference Used Twice (Unsafe)

```rust
fn bad<'a>(r: &'a mut i32) {
    *r = 5;
    *r = *r + 1;  // Read and write to same place, same reference!
}
```

**Type:**
```rust
r: RefDty {
    rgn: Provenance::Value("r"),
    own: Ownership::Uniq,
    ...
}
```

**Checking:**
```
Access 1: *r = 5 (write)
  └─ Create Loan { *r, Uniq } in PrvMapping("r")

Access 2: *r + 1 (read)
  └─ Create Loan { *r, Uniq } in PrvMapping("r")
  └─ Same provenance!
  └─ Conflict check:
     - PrvMapping("r") already has Loan { *r, Uniq }
     - Trying to add another Loan { *r, Uniq }
     - Uniq + Uniq = ✗ CONFLICT
```

**Result:** ❌ ERROR — Cannot hold two exclusive loans to the same place from the same reference

---

## Provenance vs. Place

**Important distinction:**

| Concept | Purpose | Example |
|---------|---------|---------|
| **Provenance** | Which reference? | "a", "b", "result" |
| **Place** | Which location in memory? | `a[0]`, `a[1]`, `b[0]` |

You need **both** to prevent races:

```rust
// Scenario: Two threads access different indices of the same array
let a: &[i32] = ...;  // Provenance: "a"

Thread 0: a[0] = 5;   // Place: a[0], Provenance: "a"
Thread 1: a[1] = 10;  // Place: a[1], Provenance: "a"

// Same provenance (same reference)
// Different places (different indices)
// No conflict ✓
```

vs.

```rust
Thread 0: a[0] = 5;
Thread 0: a[0] = *a[0] + 1;  // Same place, same provenance, exclusive access

// Same provenance, same place
// Can't have two Uniq loans
// Conflict ✗
```

---

## Provenance in Your Type System Implementation

When you implement provenance tracking, you need:

### 1. Provenance in RefDty

```rust
pub struct RefDty {
    pub rgn: Provenance,        // ← Name of this reference
    pub own: Ownership,
    pub mem: Memory,
    pub dty: Box<DataTy>,
}

pub enum Provenance {
    Value(String),  // Named provenance from a binding (parameter, let)
    Ident(Ident),   // Generic provenance parameter
}
```

### 2. PrvMapping in Context

```rust
pub struct PrvMapping {
    pub prv: String,           // Which provenance?
    pub loans: HashSet<Loan>,  // Loans tied to this provenance
}
```

### 3. Lookup in Borrow Checker

```rust
fn access_safety_check(ctx: &BorrowCheckCtx, p: &PlaceExpr) -> Result<HashSet<Loan>> {
    // Get the provenance of the reference being accessed
    let provenance = get_provenance_of_place(p);
    
    // Get the loans already issued for this provenance
    let existing_loans = ctx.ty_ctx.loans_for_prv(&provenance)?;
    
    // Check for conflicts with existing loans from THIS provenance
    for existing_loan in existing_loans {
        if places_conflict(existing_loan.place, p) &&
           ownership_conflict(existing_loan.own, ctx.own) {
            return Err(Conflict);
        }
    }
    
    // Issue new loan tied to this provenance
    let new_loan = Loan { place_expr: p.clone(), own: ctx.own };
    ctx.ty_ctx.extend_loans_for_prv(&provenance, [new_loan])?;
    
    Ok(...)
}
```

---

## Summary: Why Provenance Matters

1. **Identity:** Names each reference so the checker knows which reference a loan came from

2. **Reusability:** The same reference can be used multiple times if the ownership allows it (e.g., `Shrd` can be reused infinitely)

3. **Conflict Detection:** The checker only looks for conflicts within loans from the *same* provenance (same reference), not across different provenances

4. **GPU Safety:** Combined with execution context, provenance prevents races by tracking which *reference* a thread is accessing and whether that reference is exclusively owned

5. **Correctness:** Without provenance, the checker couldn't tell if two accesses came from the same reference or different references — essential for sound borrow checking

# Data Race Examples: Kernels That Descend Would Reject

Simple racy kernels written in your actual syntax, with detailed analysis of why they race and how descend's type system catches them.

---

## Example 1: Two Threads Writing to Same Location (Classic Race)

### The Racy Code

```rust
#[kernel]
pub fn buggy_histogram(input: &CudaVec<u32>, hist: &mut CudaVec<u32>) {
    let idx: u32 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BUG: ALL threads write to hist[0]
    let val = input[idx];
    hist[0] = hist[0] + val;  // ← RACE! Multiple threads compete
}
```

### What's Wrong

Two threads (say thread 0 and thread 1) execute in parallel:

```
Thread 0:                    Thread 1:
load hist[0] → 42          load hist[0] → 42
add 5 → 47                 add 3 → 45
store hist[0] = 47        store hist[0] = 45  ← overwrites thread 0's result
```

Final result: `hist[0] = 45` (lost thread 0's contribution)

### How Descend Catches It

**Type of `hist`:**
```rust
RefDty {
    rgn: Provenance::Value("hist"),
    own: Ownership::Uniq,        // ← Unique (exclusive write)
    mem: Memory::GpuGlobal,
    dty: Array(U32, n)
}
```

**Access 1: `hist[0] =` (first operand of read)**
```
access_safety_check(ctx, Place::Index(Var("hist"), Lit(0)))
  ├─ narrowing_check: ✓ GpuThread can access GpuThread-bound binding
  ├─ access_conflict_check: AccessCtx is empty → ✓ no conflict
  └─ borrow_check: Create Loan { hist[0], Uniq }
```

**Access 2: `hist[0] + val` (read within addition)**
```
access_safety_check(ctx, Place::Index(Var("hist"), Lit(0)))
  ├─ narrowing_check: ✓ same as before
  ├─ access_conflict_check:
  │   AccessCtx has: Loan { hist[0], Uniq }
  │   New access: hist[0] with Uniq
  │   ✗ CONFLICT: Cannot have two Uniq loans on same place
  └─ ERROR: "Uniq loan conflict on hist[0]"
```

**Why This Is Racy:**
- The `Uniq` ownership says "only one active loan at a time"
- Reading `hist[0]` while holding a write loan to `hist[0]` is forbidden
- This prevents the scenario where two threads interleave reads/writes to the same location

**Result:** ❌ **REJECTED** — Compile error

---

## Example 2: Narrowing Violation (Thread Reading Grid-Level Data)

### The Racy Code

```rust
// Assume some grid-level state
let shared_scratch: CudaVec<f32> = /* allocated at grid level */;

#[kernel]
pub fn buggy_sched(input: &CudaVec<f32>, shared_scratch: &mut CudaVec<f32>) {
    let idx: u32 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // All threads write to shared_scratch (which is owned by the grid)
    shared_scratch[idx] = input[idx];
    
    // Implicit sync: __syncthreads()
    
    // Now thread 0 reads what thread 1 wrote
    // (and vice versa — they alias!)
    let val = shared_scratch[idx];
}
```

### What's Wrong

Without synchronization and clear ownership:
- Thread 0 writes to `shared_scratch[0]`
- Thread 1 writes to `shared_scratch[1]`
- Thread 0 then reads `shared_scratch[1]` (data that thread 1 just wrote)
- **Problem**: Thread 1 might not have written yet, or worse, thread 1 might write *again* while thread 0 is reading

This is a **narrowing violation** — a thread is trying to claim exclusive ownership of block-level data.

### How Descend Catches It

**Type of `shared_scratch` parameter:**
```rust
RefDty {
    rgn: Provenance::Value("shared_scratch"),
    own: Ownership::Uniq,        // Mutable reference
    mem: Memory::GpuGlobal,      // or GpuShared
    dty: Array(F32, n)
}
```

**At kernel boundary (GpuGrid level):**
```
FrameEntry::Var(IdentTyped {
    ident: "shared_scratch",
    exec: ExecExpr { base: GpuGrid, ... },  // ← Bound at grid level
    ty: RefDty { Uniq, gpu.global, [...] },
})
```

**In kernel body (implicit re-type to GpuThread level):**
```
FrameEntry::Var(IdentTyped {
    ident: "shared_scratch",
    exec: ExecExpr { base: GpuThread, ... },  // ← Trying to use at thread level
    ty: RefDty { Uniq, gpu.global, [...] },
})
```

**Access attempt: `shared_scratch[idx] = ...`**
```
access_safety_check(ctx, Place::Index(Var("shared_scratch"), Var("idx")))
  └─ narrowing_check:
     - Binding exec: GpuThread
     - Active exec: GpuThread
     - narrowable(GpuThread, GpuThread)?
       └─ exec_is_prefix_of: GpuThread ⊆ GpuThread? YES
       └─ BUT: Uniq borrow from grid level cannot be narrowed to thread level
            (Each thread would think it has exclusive access!)
       └─ ERROR: "Cannot narrow Uniq borrow from GpuGrid to GpuThread"
```

**Why This Fails:**
- `shared_scratch` is bound at `GpuGrid` (whole kernel)
- `Uniq` says "only one thread can write"
- Trying to use it at `GpuThread` scope would let *all* threads think they have exclusive access
- This would cause races between threads

**Result:** ❌ **REJECTED** — Narrowing check fails

---

## Example 3: Memory Space Aliasing (CPU Pointer + GPU Pointer)

### The Racy Code

```rust
#[kernel]
pub fn buggy_alias(input: &CudaVec<f32>, result: &mut CudaVec<f32>) {
    let idx: u32 = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = input[idx] * 2.0;
}

// Then in host code:
let mut h_result = result.as_slice();  // Get CPU pointer
h_result[0] = 999.0;  // Write from CPU
// Meanwhile, some GPU threads might also be writing to result[0]
```

### What's Wrong

- GPU thread writes to `result[0]`
- CPU simultaneously writes to `result[0]`
- The GPU sees old data, or CPU sees partially-written GPU data
- **Classic memory aliasing race**

### How Descend Catches It (Partially)

**Type of `result` parameter:**
```rust
RefDty {
    rgn: Provenance::Value("result"),
    own: Ownership::Uniq,
    mem: Memory::GpuGlobal,        // ← Key: on GPU, not CPU
    dty: Array(F32, n)
}
```

**In kernel:**
```
GPU thread accessing: Type = RefDty { Uniq, GpuGlobal, [...] }
```

**If someone tries to create a CPU reference:**
```
// Hypothetically, if the type system saw:
let cpu_ptr: &mut [f32] = h_result;  // Type would be RefDty { Uniq, CpuMem, [...] }
```

**Conflict:**
```
GPU thread: RefDty { Uniq, GpuGlobal, [...] }
CPU thread: RefDty { Uniq, CpuMem, [...] }

These are different Memory spaces but same underlying data!
Descend would catch this IF the type system tracked the same place
across memory spaces.
```

**Result:** ❌ **CAUGHT** (indirectly via memory space mismatch)

---

## Example 4: Correct Version (Safe)

### The Safe Code

```rust
#[kernel]
pub fn safe_vector_add(
    a: &CudaVec<f32>,           // Shrd, gpu.global
    b: &CudaVec<f32>,           // Shrd, gpu.global
    result: &mut CudaVec<f32>   // Uniq, gpu.global
) {
    let idx: u32 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread writes to a DIFFERENT index
    result[idx] = a[idx] + b[idx];
}
```

### Why This Is Safe

**Type analysis:**

| Access | Type | Ownership | Place | Thread 0 | Thread 1 | Conflict? |
|--------|------|-----------|-------|----------|----------|-----------|
| `a[idx]` | F32 | `Shrd` | `a[0]` | `a[0]` | `a[0]` | Shrd+Shrd=✓ |
| `b[idx]` | F32 | `Shrd` | `b[0]` | `b[0]` | `b[0]` | Shrd+Shrd=✓ |
| `result[idx]` | F32 | `Uniq` | `result[0]` | `result[0]` | `result[1]` | Uniq to different places=✓ |

**Key insight:** The `idx` is different for each thread (by CUDA semantics), so `result[0]` and `result[1]` are *different places*, not aliased.

**Access safety check for thread 0:**
```
AccessCtx = { Loan { a[0], Shrd }, Loan { b[0], Shrd }, Loan { result[0], Uniq } }

access_conflict_check:
  a[0] (Shrd) vs b[0] (Shrd) → ✓ both reads
  a[0] (Shrd) vs result[0] (Uniq) → ✓ different idents
  b[0] (Shrd) vs result[0] (Uniq) → ✓ different idents
```

**Access safety check for thread 1:**
```
AccessCtx = { Loan { a[0], Shrd }, Loan { b[0], Shrd }, Loan { result[0], Uniq } }
            + New loans from thread 1: { a[0], Shrd }, { b[0], Shrd }, { result[1], Uniq }

Conflict check:
  a[0] (Shrd) appears in both → ✓ Shrd is reentrant
  b[0] (Shrd) appears in both → ✓ Shrd is reentrant
  result[0] vs result[1] → ✓ different indices, no conflict
```

**Result:** ✅ **ACCEPTED** — All checks pass

---

## Example 5: Histogram With Atomics (Safe With Atomic Ownership)

### The Code (With Atomic Type)

```rust
#[kernel]
pub fn safe_histogram(input: &CudaVec<u32>, hist: &mut CudaVec<Atomic<u32>>) {
    let idx: u32 = blockIdx.x * blockDim.x + threadIdx.x;
    
    let val = input[idx];
    // atomicAdd is a special function that takes &mut Atomic<u32>
    atomicAdd(&mut hist[0], val);
}
```

### Why This Works

**Type of `hist[0]`:**
```rust
// When hist is CudaVec<Atomic<u32>>, indexing gives:
Type::Atomic(AtomicU32)  // Not a regular Uniq reference!
```

**Key: `atomicAdd` signature:**
```rust
pub fn atomicAdd(target: &mut Atomic<u32>, delta: u32) -> u32 {
    // Hardware atomic operation — no race!
}
```

**Access check:**
```
access_safety_check for atomicAdd(&mut hist[0], val):
  - hist[0] has type Atomic<u32>, not regular Uniq
  - Atomic types have special handling: many threads can "own" them
  - Type rule: Atomic<T> is inherently thread-safe
  - ✓ No conflict, no narrowing issue
```

**Result:** ✅ **ACCEPTED** — Atomic types bypass the Uniq exclusivity rule

---

## Summary Table

| Example | Issue | Ownership? | Memory? | Exec/Narrowing? | Descend Verdict |
|---------|-------|-----------|---------|-----------------|-----------------|
| **Example 1**: Two threads → `hist[0]` | Multiple Uniq loans | ✓ Caught | — | — | ❌ REJECT |
| **Example 2**: Thread reads grid-level | Narrowing violation | — | — | ✓ Caught | ❌ REJECT |
| **Example 3**: CPU + GPU alias | Memory space mismatch | — | ✓ Caught | — | ❌ REJECT |
| **Example 4**: Each thread → different index | None | ✓ Safe | ✓ Safe | ✓ Safe | ✅ ACCEPT |
| **Example 5**: Histogram with atomics | None (atomic bypass) | — | — | — | ✅ ACCEPT |

---

## How to Implement These Checks in Your Type Checker

To catch these races, you need (in order of importance):

### 1. **Ownership Tracking** (catches Example 1)
```rust
// In Type::Ref or RefDty:
pub enum Ownership { Shrd, Uniq }

// In access_conflict_check:
match (existing_loan.own, new_access.own) {
    (Shrd, Shrd) => Ok(()),           // ✓ Multiple readers
    (Uniq, _) | (_, Uniq) => Err(...) // ✗ Exclusive write violation
}
```

### 2. **Execution Context Tracking** (catches Example 2)
```rust
// In IdentTyped:
pub struct IdentTyped {
    ident: Ident,
    ty: Ty,
    mutbl: Mutability,
    exec: ExecExpr,  // ← WHERE is this binding from?
}

// In narrowing_check:
if new_access_exec.is_more_specific_than(binding_exec) {
    Ok(())  // ✓ Can narrow from coarse to fine
} else {
    Err(...)  // ✗ Cannot narrow from fine to coarse
}
```

### 3. **Memory Space Tracking** (catches Example 3)
```rust
// In Type or RefDty:
pub enum Memory { CpuMem, GpuGlobal, GpuShared, GpuLocal }

// On borrow creation:
// Ensure no Uniq borrow crosses memory boundaries
```

### 4. **Place Granularity** (needed for all)
```rust
// Distinguish between:
// - Place::Ident("hist")          ← whole array
// - Place::Index("hist", Lit(0))  ← specific element
// - Place::Index("hist", Var("i")) ← parametric (harder)

// Only flag conflict if same place AND incompatible ownership
```

---

## Key Takeaway

Descend's three-axis type system prevents data races by enforcing:

1. **Ownership**: `Shrd` allows concurrent reads; `Uniq` requires exclusive access
2. **Memory**: keeps CPU and GPU allocations separate
3. **Execution**: prevents threads from claiming grid-level or block-level exclusive ownership

Your implementation needs all three to reject real racy kernels.

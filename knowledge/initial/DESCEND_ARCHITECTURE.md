# Descend Type Checker Architecture

A visual guide to how descend's type system, contexts, and checks work together to prove data-race freedom.

## Complete Pipeline

```mermaid
graph LR
    A["Source Code<br/>.desc file"] -->|parser/| B["ast::CompilUnit<br/>Program AST"]
    B -->|ty_check::ty_check| C["Typed AST<br/>+ verified safety"]
    C -->|codegen::gen| D["CUDA C++"]
    
    style A fill:#e8e8e8
    style B fill:#e3f2fd
    style C fill:#c8e6c9
    style D fill:#fff9c4
```

## Type System Hierarchy

```mermaid
graph TD
    A["Type System"] --> B["TyKind"]
    A --> C["Kind (Nat, Memory, Provenance, DataTy)"]
    
    B -->|Data| D["DataTy"]
    B -->|Function| E["FnTy"]
    
    D --> D1["Scalar<br/>F32, I32, U64, etc"]
    D --> D2["Atomic<br/>AtomicU32, AtomicI32"]
    D --> D3["Array<br/>dty, Nat"]
    D --> D4["Tuple<br/>Vec DataTy"]
    D --> D5["Struct<br/>field list"]
    D --> D6["At<br/>dty @ Memory"]
    D --> D7["Ref<br/>&r ω m dty"]
    D --> D8["RawPtr"]
    D --> D9["Dead<br/>type marked killed"]
    
    D7 --> R["RefDty"]
    R --> R1["Provenance r<br/>Value(name) | Ident"]
    R --> R2["Ownership ω<br/>Shrd | Uniq"]
    R --> R3["Memory m<br/>CpuMem | GpuGlobal<br/>GpuShared | GpuLocal"]
    R --> R4["pointee dty"]
    
    E --> E1["generic params<br/>IdentKinded"]
    E --> E2["ParamSig[]<br/>exec, type"]
    E --> E3["return Type"]
    E --> E4["nat constraints"]
    
    style A fill:#9b59b6,color:#fff
    style D7 fill:#e74c3c,color:#fff
    style R1 fill:#f39c12,color:#fff
    style R2 fill:#3498db,color:#fff
    style R3 fill:#16a085,color:#fff
```

## Type Checking Contexts

```mermaid
graph TD
    A["ExprTyCtx<br/>Master context during<br/>expression checking"] --> B["GlobalCtx"]
    A --> C["KindCtx"]
    A --> D["TyCtx"]
    A --> E["AccessCtx"]
    A --> F["active ExecExpr"]
    
    B --> B1["Function declarations<br/>& signatures"]
    B --> B2["Struct declarations"]
    B --> B3["Pre-declared built-ins<br/>exec, atomicAdd, etc"]
    
    C --> C1["Generic params<br/>IdentKinded"]
    C --> C2["Provenance<br/>outlives relations<br/>PrvRel"]
    
    D --> D1["Stack of Frame"]
    D --> D2["Frame = Vec FrameEntry"]
    D --> D3["FrameEntry =<br/>Var(IdentTyped) |<br/>ExecMapping |<br/>PrvMapping"]
    
    D3 -->|Var| V["IdentTyped<br/>ident, ty, mutbl<br/>exec (KEY!)"]
    D3 -->|ExecMapping| EM["ident → ExecExpr"]
    D3 -->|PrvMapping| PM["prv name → HashSet Loan"]
    
    E --> E1["HashSet Loan"]
    E --> E2["accumulated during<br/>current sub-expr"]
    E --> E3["reset between<br/>sequenced exprs"]
    
    E2 --> EL["Loan<br/>place_expr, own"]
    
    F --> F1["ExecExpr<br/>where are we executing?<br/>grid, block, thread?"]
    
    style A fill:#9b59b6,color:#fff
    style V fill:#e74c3c,color:#fff
    style PM fill:#3498db,color:#fff
    style EL fill:#f39c12,color:#fff
    style F1 fill:#16a085,color:#fff
```

## Execution Context System

```mermaid
graph TD
    A["ExecExpr"] --> B["BaseExec"]
    A --> C["ExecPath"]
    
    B --> B1["Ident<br/>parameter binding"]
    B --> B2["CpuThread<br/>CPU execution"]
    B --> B3["GpuGrid<br/>grid of blocks"]
    
    C --> C1["TakeRange<br/>select range of dims"]
    C --> C2["ForAll<br/>iterate over dim"]
    C --> C3["ToWarps<br/>divide into warps"]
    C --> C4["ToThreads<br/>divide into threads"]
    
    style A fill:#16a085,color:#fff
    style B1 fill:#3498db,color:#fff
    style C2 fill:#f39c12,color:#fff
    
    subgraph Example["Example: Vector_Sum"]
        X["Grid(64, 1024)"] -->|ForAll block in grid| Y["Block(64, 1024)"]
        Y -->|ToThreads thread in block| Z["Thread(per-thread)"]
        Z --> W["Binding created here has<br/>exec = Thread(...)"]
    end
```

## Type Checking Pass

```mermaid
graph TD
    A["ty_check_global_fun_def<br/>entry point for kernel"] --> B["Build initial contexts"]
    
    B --> B1["GlobalCtx: functions,<br/>struct decls, built-ins"]
    B --> B2["KindCtx: generics &<br/>provenance relations"]
    B --> B3["TyCtx: push Frame for<br/>function params"]
    B --> B4["AccessCtx: empty HashSet"]
    B --> B5["ExecExpr: kernel's exec<br/>GpuGrid at boundary"]
    
    B1 --> C["For each param<br/>ParamDecl"]
    C --> C1["lower_type ParamDecl.ty"]
    C --> C2["Record in TyCtx as<br/>IdentTyped { ident, ty,<br/>mutbl, exec }"]
    
    C2 --> D["ty_check_expr<br/>dispatch on ExprKind"]
    
    D --> D1["Expr::PlaceExpr"]
    D --> D2["Expr::Block"]
    D --> D3["Expr::Let"]
    D --> D4["Expr::Assign"]
    D --> D5["Expr::Index"]
    D --> D6["Expr::Ref<br/>borrow operation"]
    D --> D7["Expr::Sched<br/>parallel for"]
    
    D1 -->|if place| E["ty_check_place<br/>read from memory"]
    D1 -->|if non-place| F["ty_check_non_place<br/>computed value"]
    
    E --> E1["call access_safety_check<br/>narrowing_check<br/>access_conflict_check<br/>borrow_check"]
    E1 --> E2["record Loan in AccessCtx"]
    
    D3 --> G["type_check_expr value"]
    G --> G1["TyCtx::insert name<br/>update Context"]
    
    D4 --> H["borrow_check target"]
    H --> H1["kill_place target<br/>mark as Dead"]
    
    D7 --> I["Push new exec onto stack"]
    I --> I1["type_check body"]
    I1 --> I2["Pop exec, garbage collect"]
    
    style A fill:#9b59b6,color:#fff
    style E1 fill:#e74c3c,color:#fff
    style E2 fill:#f39c12,color:#fff
    style H1 fill:#3498db,color:#fff
```

## Borrow Checking: The Data-Race Detector

```mermaid
graph TD
    A["access_safety_check<br/>ctx: BorrowCheckCtx<br/>p: PlaceExpr<br/>→ Result HashSet Loan"] --> B{unsafe_flag?}
    
    B -->|false| C["narrowing_check<br/>unique borrow OK?"]
    B -->|true| D["Skip to borrow_check"]
    
    C --> C1["Bind exec = ident_ty.exec"]
    C --> C2["Active exec = ctx.exec"]
    C --> C3["Is Bind narrowable to Active?<br/>Bind.path ⊆ Active.path?"]
    C3 -->|no| C4["ERROR:<br/>Cannot narrow:<br/>reading at grid level<br/>from thread scope"]
    C3 -->|yes| C5["OK: More specific exec<br/>can read from<br/>less specific"]
    
    C5 --> E["access_conflict_check"]
    E --> E1["For each Loan in AccessCtx"]
    E --> E2["Do place expressions overlap?<br/>Same ident?<br/>Same path?"]
    E2 -->|compatible owners<br/>Shrd + Shrd| E3["OK"]
    E2 -->|incompatible<br/>Uniq + anything| E4["ERROR:<br/>Conflict with<br/>existing access"]
    
    E3 --> F["borrow_check<br/>p.is_place?"]
    E4 --> FAIL["REJECT<br/>Compile error"]
    
    F -->|yes| G["ownership_safe_place<br/>insert Loan<br/>{place_expr: p,<br/>own: ctx.own}"]
    
    F -->|no| H["borrow_check deref<br/>p points through<br/>provenance r"]
    
    H --> H1["loans_for_prv r<br/>lookup in TyCtx"]
    H --> H2["For each in loans<br/>check narrowability"]
    
    H2 --> I["insert Loan<br/>with new ownership"]
    
    G --> J["Return HashSet Loan"]
    I --> J
    D --> J
    
    style A fill:#9b59b6,color:#fff
    style C fill:#e74c3c,color:#fff
    style E fill:#f39c12,color:#fff
    style C4 fill:#c0392b,color:#fff
    style E4 fill:#c0392b,color:#fff
    style J fill:#27ae60,color:#fff
```

## Place Expressions: Normalized for Borrow Checking

```mermaid
graph TD
    A["PlaceExpr<br/>from program<br/>x, *r, a[i], s.f, etc"] --> B["to_pl_ctx_and_most_specif_pl<br/>normalize"]
    
    B --> B1["PlaceCtx<br/>context with holes<br/>Deref | Proj n<br/>FieldProj f | Idx i<br/>Select exec | View v"]
    
    B --> B2["Place<br/>ident + path<br/>most specific"]
    
    B1 --> C["Used by borrow checker<br/>to ask:<br/>is this loan<br/>a prefix of that place?"]
    
    B2 --> D["Used to<br/>find the type of<br/>the place"]
    
    style A fill:#9b59b6,color:#fff
    style B1 fill:#3498db,color:#fff
    style B2 fill:#f39c12,color:#fff
```

## Data Race Prevention: The Full Picture

```mermaid
graph TD
    A["Three Axes of Safety"] --> B["1. Ownership"]
    A --> C["2. Memory Space"]
    A --> D["3. Execution Context"]
    
    B --> B1["&uniq can only have<br/>one active loan"]
    B --> B2["&shrd allows many<br/>simultaneous loans"]
    B --> B3["Checked by:<br/>access_conflict_check"]
    
    C --> C1["&uniq gpu.global<br/>cannot be aliased<br/>by cpu.mem pointer"]
    C --> C2["Each binding's memory<br/>is explicit in type"]
    C --> C3["Checked by:<br/>type system<br/>RefDty.mem"]
    
    D --> D1["&uniq from grid<br/>can safely be shared<br/>among blocks"]
    D --> D2["&uniq from block<br/>cannot be shared<br/>between threads"]
    D --> D3["Each binding's exec<br/>is tracked in<br/>IdentTyped.exec"]
    D --> D4["Checked by:<br/>narrowing_check"]
    
    B3 --> RACE["PREVENTS:<br/>two threads writing<br/>to same memory"]
    C3 --> ALIAS["PREVENTS:<br/>CPU aliasing GPU<br/>allocation"]
    D4 --> SCOPE["PREVENTS:<br/>thread reading memory<br/>from different thread"]
    
    style RACE fill:#c0392b,color:#fff
    style ALIAS fill:#c0392b,color:#fff
    style SCOPE fill:#c0392b,color:#fff
```

## Key Invariants

```mermaid
graph LR
    A["After ty_check succeeds:"] --> B["✓ Every Expr has type"]
    A --> C["✓ Every binding tracked<br/>with ownership"]
    A --> D["✓ Every memory access<br/>checked for conflicts"]
    A --> E["✓ Every exec context<br/>narrowing validated"]
    A --> F["✓ Codegen receives<br/>race-free AST"]
    
    style B fill:#27ae60,color:#fff
    style C fill:#27ae60,color:#fff
    style D fill:#27ae60,color:#fff
    style E fill:#27ae60,color:#fff
    style F fill:#27ae60,color:#fff
```

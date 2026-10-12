# LLM-Assisted JIT for AQP Middleware

## 1. Context

AQP_middleware splits complex join queries into sub-SQLs (via node-based,
relationship-center, or top-down strategies), executes each in DuckDB, and
optionally JIT-compiles operators within DuckDB at three levels: expr-jit,
operator-jit, pipeline-jit.

A fourth level, **sql-jit**, was attempted and failed (see `sql_jit_fail.md`).
This document explores using **LLM inference** to generate specialised code for
sub-queries, running it inside DuckDB's execution framework.

## 2. Previous SQL-JIT Failure (Level 4)

Compiled the entire probe-side execution into a single LLVM coordinator
function that replaced DuckDB's physical operators. Failed because:

1. **Negligible materialisation savings** — DuckDB's 2048-row vectorised
   chunks already fit L2; the register-vs-L2 gap is ~1-2 ns/access.
2. **Physical plan mismatch** — DuckDB's optimizer creates right-deep/bushy
   plans; the coordinator assumed left-deep probe chains. Compiled on only
   1/113 JOB queries.
3. **Memory-bound hash probes** — compiling the probe logic doesn't help when
   the bottleneck is pointer-chasing into the hash table.
4. **High complexity** — required `PhysicalAQPSubPlan`, `AQPSourceWrapper`,
   MetaPipeline wiring, etc.

## 3. Proposed New Approach: LLM-Generated Specialised Code

### 3.1 Key Difference from Failed SQL-JIT

The failed SQL-JIT tried to replace operators **inside** DuckDB's plan.
The new approach generates specialised **callback functions** that plug into
DuckDB's existing operator framework (same as pipeline-jit), but with
**data-adaptive decisions** informed by actual runtime cardinalities.

### 3.2 Inside-DuckDB Execution (Chosen Approach)

The LLM generates LLVM IR snippets or C++ functions against a minimal runtime:
- Arrow IPC reader (provided by the middleware runtime)
- Small hash-table library (direct-array, linear-probing, Robin Hood)
- SIMD primitives for filter evaluation

These plug into DuckDB's operator callbacks just like pipeline-jit does today.

Advantage over standalone binary: reuses DuckDB's hash tables, parallelism,
buffer management. No data serialisation overhead.

### 3.3 GPU for LLM Inference, CPU for Query Execution

- GPU runs LLM inference to generate specialised code (no query execution on GPU)
- CPU runs DuckDB on the current sub-query simultaneously
- Zero resource contention: GPU inference and CPU execution use independent hardware
- JOB queries are join-heavy (memory-bound), poor fit for GPU query execution anyway

### 3.4 Data Handoff

Arrow IPC over shared memory (`/dev/shm`). DuckDB has native Arrow export;
the middleware runtime provides an Arrow IPC reader. Zero-copy where possible.

## 4. What the LLM Can Optimise (from GenDB and BespokeOLAP analysis)

### 4.1 Filter

| Technique | Source | Inside DuckDB? |
|---|---|---|
| Direct lookup arrays indexed by domain value | GenDB | Yes — replace filter expression evaluator |
| Zone maps to skip 64K-row blocks | GenDB | Partially — DuckDB has its own zone maps |
| Bitmap acceleration from dimension tables | BespokeOLAP | Yes |
| Compiled LIKE patterns (Prefix/Suffix/Contains) | BespokeOLAP | Yes |

Current pipeline-jit already handles filters well (10-80x speedup observed).
LLM adds marginal gain here.

### 4.2 Aggregation (HIGH opportunity)

| Technique | Source | Inside DuckDB? |
|---|---|---|
| Direct-array when groups <= 256 | GenDB | Yes — biggest opportunity |
| Thread-local hash maps when groups > 256 | GenDB | Yes |
| Pre-sized containers from known cardinalities | BespokeOLAP | Yes |
| Custom hash functions for composite keys | GenDB | Yes |
| Kahan summation for FP accuracy | GenDB | Yes |

DuckDB uses generic hash aggregation for all group sizes. After a split,
temp table group cardinalities are **known exactly** — the LLM can generate
direct-array aggregation that's 10-50x faster for small group counts.

### 4.3 Hash Join — Build Side

| Technique | Source | Inside DuckDB? |
|---|---|---|
| Direct array when FK domain <= 100K | GenDB | No — can't replace DuckDB's Build |
| Semi-join via boolean mask | BespokeOLAP | No |
| Index with open-addressing + custom hash | GenDB | No |

DuckDB's `JoinHashTable::Build` is deeply integrated. Not a practical target.

### 4.4 Hash Join — Probe Side (MEDIUM opportunity)

| Technique | Source | Inside DuckDB? |
|---|---|---|
| O(1) direct-array probe for small build sides | GenDB | Yes — replace probe function |
| Bitmap pre-filtering of probe rows | BespokeOLAP | Yes |
| Prefetch distances tuned to hash table size | GenDB | Yes |
| Type-specialised hash functions | GenDB | Yes |

Pipeline-jit already hooks into hash probe. The LLM can improve by choosing
probe strategy based on build-side cardinality (which is known after earlier
sub-queries).

### 4.5 Projection

Marginal. Pipeline-jit and DuckDB already handle this efficiently.

### 4.6 Scan

Partially. For temp table scans (which splits create), the LLM could generate
specialised scans that skip null checks and use mmap + `MADV_SEQUENTIAL`.
DuckDB manages its own storage for base tables.

## 5. JOB Benchmark Analysis

### 5.1 Breakdown File Format

The breakdown CSVs record per-phase timing. Each iteration writes a different
number of columns depending on JIT level:

- **Pipeline-jit**: `[extract, gen_sql, jit_compile, execute, update_ir]` = 5 per iteration
- **No-JIT**: `[extract, gen_sql, execute, update_ir]` = 4 per iteration
- Both have 2 overhead columns (parse_sql, preprocess) and a variable-length final block.

Execution time and JIT compilation time are separate columns, allowing
independent measurement.

### 5.2 JIT Execution Results

Across all 113 JOB queries with node-based split + pipeline-jit (O1, auto SIMD),
separating execution time from compilation time:

| Category | Count | No-JIT exec (ms) | Pipeline exec (ms) | Notes |
|---|---|---|---|---|
| **JIT helps** | 206 iterations | 2813.7 | 10.8 | **260x** aggregate speedup |
| **JIT hurts** | 15 iterations | 15.5 | 174.5 | 11x aggregate slowdown |
| **JIT wasted** | 28 iterations | - | - | 637.8ms compile overhead on tiny sub-queries |

**Net execution savings: 2643.9ms** (ignoring compilation overhead).

The JIT execution itself is highly effective — the bottleneck is
**compilation overhead**, especially on the 28 iterations where JIT compiles
code for sub-queries that take < 0.5ms without JIT.

### 5.3 Where JIT Execution Hurts

The 15 regressions cluster at **iteration 5-6** in queries with many tables
(19a, 24a, 27a, 29a, etc.). These iterations typically contain hash-join-heavy
sub-queries where JIT-compiled probe logic is slower than DuckDB's
hand-optimised C++ hash table probing.

### 5.4 Dynamic IR Issue

All splitters re-optimise through DuckDB's optimizer each iteration:
- **TopDown**: explicit `ReOptimizeIR()` with actual temp table cardinalities
- **Node-based**: `RunMiddleOptimize()` each iteration
- **All**: `conn->Prepare(sql)` triggers DuckDB's full optimizer for each sub-SQL

This means future sub-SQL shapes are **not fully predictable** — the join order
and plan structure can change after each execution based on actual cardinalities.

## 6. Proposed Architecture

### 6.1 Latency-Hiding Pipeline

```
GPU:  [══ LLM generates specialised code for S(k)...S(n) ══════════]
CPU:  [DuckDB S1][DuckDB S2]...[DuckDB S(k-1)][S(k) with LLM code]...
       ↑ early sub-queries use            ↑ later sub-queries use
         existing pipeline-jit              LLM-generated callbacks
```

### 6.2 Speculative Pre-Generation

Since all splitters re-optimise per iteration, we cannot predict exact future
sub-SQLs. Strategy:

1. After the initial split, submit **predicted** sub-SQL shapes to GPU LLM
2. After each iteration, check if the **actual** sub-SQL matches a pre-generated
   binary (by join structure, column types, predicate shapes)
3. If matched, use LLM-generated code; otherwise fall back to pipeline-jit/DuckDB

Match rate should be high: re-optimisation mostly reorders joins but doesn't
change which tables are joined or predicate types.

### 6.3 Cost Model

Decide whether to use LLM-generated code based on:
1. **Is the code ready?** (LLM inference + compilation finished before this iteration)
2. **Is it a hash-join iteration?** (iter 5-6 pattern where pipeline-jit hurts)
3. **Is the expected execution benefit > overhead?** (skip for tiny sub-queries)

### 6.4 Target: Later Sub-Queries

- **Early sub-queries (S1-S3)**: small, fast. Pipeline-jit or no-jit is sufficient.
- **Later sub-queries (S(k)...S(n))**: join **temp tables** with **known exact
  cardinalities**. This is where:
  - DuckDB's optimizer may choose suboptimal plans (based on estimated stats)
  - The LLM can exploit actual cardinalities to pick better algorithms
  - Enough CPU time has elapsed to hide LLM inference + compile latency

## 7. What Breakdown Data is Needed

The current breakdown logs per-sub-query total time. To confirm which operators
benefit most from LLM-generated code, we need a **per-operator-type breakdown
within each sub-query**:

```
Sub-query S(i):
  scan_time:        X ms
  filter_time:      Y ms
  hash_build_time:  Z ms
  hash_probe_time:  W ms
  aggregate_time:   V ms
  projection_time:  U ms
```

Options to collect this:
1. DuckDB's `PRAGMA enable_profiling='json'` for operator-level timing
2. Instrument `RegisterJIT` to log which operators are JIT-compiled vs interpreted
3. Tag each sub-query with its operator composition

This would tell us whether the 15 regression iterations (iter 5-6) are
dominated by hash build, hash probe, or another operator, and which GenDB/
BespokeOLAP optimisations to apply.

## 8. LLM Inference Latency for Sub-SQLs

Sub-SQLs are much simpler than the original query:

| | Original JOB query | Sub-SQL after split |
|---|---|---|
| Tables | 5-17 | 2-3 (one base + one temp) |
| Joins | 4-28 | 1 (typically) |
| Predicates | 7-44 | 2-5 |
| Plan topology | right-deep / bushy | trivial (one join) |

BespokeOLAP generates 655-1794 lines of C++ for full JOB queries.
A sub-SQL with 1 join would require roughly 100-300 lines. Estimated
output tokens:

| Target | Lines | Tokens | Time @ 100 tok/s | Time @ 200 tok/s |
|---|---|---|---|---|
| Full query | 800-1800 | 3000-7000 | 30-70s | 15-35s |
| Sub-SQL | 100-300 | 500-1200 | 5-12s | 2.5-6s |

Sub-SQL generation is 3-6x faster than full-query generation. Still too slow
for online single-query use (total query execution is 50-300ms), but feasible
with:
- **Ahead-of-time generation** for known workloads
- **Batched inference** across sub-queries (submit all predicted shapes at once)
- **Template + LLM hybrid**: pre-built templates for common operators, LLM only
  fills in data-adaptive parameters (hash table variant, prefetch distance,
  aggregation strategy) — reducing output to ~50-100 tokens per sub-query

## 9. Caching Strategy: Specialisation vs. Generality

### 9.1 Shape Diversity in JOB

JOB uses 21 base tables with 185 unique table pairs. After node-based split,
each sub-query joins a subset of these. Within the same JOB template
(e.g., 17a-17f), queries share the same table set but differ in filter
constants. However, DuckDB's optimizer produces **different iteration counts**
for different filter selectivities:

- Only 7/33 templates have all variants with identical iteration counts
- The remaining 26 templates vary because different selectivities lead to
  different join orders and split decompositions

### 9.2 Caching Levels

| Level | Cache key | Reuse | Specialisation | Best for |
|---|---|---|---|---|
| **Operator-level** | (operator type, input column types) | Highest — few dozen unique operators | Lowest — generic over data distribution | Filters, projections |
| **Shape-level** | (table pair, join keys, projected columns) | Medium — up to 185 unique shapes | Medium — specialised to schema but not data | Hash probe functions |
| **Instance-level** | (shape + actual cardinalities + selectivities) | Lowest — unique per execution | Highest — data-adaptive algorithm selection | Aggregation, join strategy |

### 9.3 Recommended Balance for JOB

**Two-tier caching:**

1. **Shape-level cache for probe/filter functions** (pre-generated offline):
   - Cache key: (table pair, join key types, filter predicate types)
   - ~185 entries cover all JOB sub-queries
   - Generate once with LLM, compile to shared libraries, reuse across runs
   - Provides type-specialised hash functions, null-check elimination,
     LIKE pattern compilation
   - ~1.5-3x speedup from type specialisation alone

2. **Instance-level adaptation at runtime** (cheap parametric decisions):
   - After each sub-query executes, actual cardinalities are known
   - A lightweight decision function (not full LLM inference) picks:
     - Direct-array vs hash aggregation based on group count
     - Nested-loop vs hash join based on build-side size
     - Prefetch distance based on hash table size
   - These are parameter choices within pre-compiled code paths, not new
     code generation — zero inference cost at runtime

This avoids the worst of both extremes:
- Not fully general (shape-level code is type-specialised, not generic)
- Not fully instance-specific (no LLM call per execution)
- The "LLM" cost is amortised to offline generation of ~185 shape-level
  binaries, each ~100-300 lines of C++

### 9.4 Template + Parameterisation Architecture

The LLM generates a **general C++ template** per sub-query shape offline.
At runtime, the middleware fills in parameters — no LLM inference needed.

**Offline (LLM generates templates):**

For each sub-query shape (e.g., `title JOIN movie_info ON title.id =
movie_info.movie_id WHERE movie_info.info_type_id = ? AND title.production_year
> ?`), the LLM generates a C++ template function containing:

- **Fixed structure** (determined by schema, not data):
  - Column types, join key types, null-ability → compiled into the template
  - Hash function specialised to the join key type (e.g., identity hash for
    integer FK, CRC32 for composite keys)
  - Null-check elimination for NOT NULL columns
  - SIMD filter evaluation for known predicate shapes (equality, range, LIKE)
  - Projection as direct column copy (no type dispatch)

- **Parameterised slots** (filled at runtime):
  - `BUILD_CARD`: build-side cardinality → selects join strategy
    (direct-array if ≤ 100K, Robin Hood hash if > 100K)
  - `PROBE_CARD`: probe-side cardinality → sizes output buffer
  - `FILTER_VALUES`: literal filter constants (e.g., `info_type_id = 3`)
  - `HT_SIZE`: hash table bucket count (next power of 2 from BUILD_CARD)
  - `PREFETCH_DISTANCE`: tuned to hash table size (4 if HT fits L2, 8 if not)

**Runtime (zero LLM cost):**

```
1. Extract sub-query from IR
2. Compute shape key: (table pair, join key types, filter predicate types)
3. Look up shape key in template cache (hash map, O(1))
4. If hit:
   a. Read actual cardinalities from previous iteration's temp tables
   b. Fill parameter slots (BUILD_CARD, FILTER_VALUES, etc.)
   c. Call parameterised template function
   d. Result becomes next temp table
5. If miss:
   Fall back to pipeline-jit or DuckDB native execution
```

**Why this works for JOB:**

- ~185 unique table pairs → ~185 templates to pre-generate
- Each template is ~100-300 lines of C++ (~500-1200 LLM output tokens)
- Total offline generation: 185 × 5-12s = 15-37 minutes (one-time cost)
- Templates compile to shared libraries (~185 `.so` files, ~10-50 KB each)
- Runtime overhead: hash map lookup + parameter fill ≈ microseconds
- The structural optimisations (type-specialised hash, null elimination, SIMD
  filters) are baked into the template — the LLM's value-add is captured
  permanently
- Parametric adaptation (direct-array vs hash, prefetch distance) handles
  instance-level variation without new code

**Example template (pseudocode):**

```cpp
// Shape: title(int id, int production_year, ...) JOIN
//        movie_info(int movie_id, int info_type_id, varchar info, ...)
// Generated by LLM for shape key: (title, movie_info, id=movie_id)

void execute_title_movie_info(
    ArrowArray* build_table,    // movie_info (smaller side)
    ArrowArray* probe_table,    // title (larger side)
    int64_t build_card,         // PARAM: actual build cardinality
    int64_t probe_card,         // PARAM: actual probe cardinality
    int32_t filter_info_type,   // PARAM: e.g., 3
    int32_t filter_min_year,    // PARAM: e.g., 2005
    ArrowArray* output          // result buffer
) {
    // Strategy selection (parameterised, not compiled)
    if (build_card <= 100000) {
        // Direct-array join: O(1) probe, no hash collisions
        int32_t* lookup = calloc(MAX_MOVIE_ID, sizeof(int32_t));
        // ... build direct-array, probe, filter, project ...
    } else {
        // Robin Hood hash: specialised to int32 keys
        // ... build hash table, probe with prefetch, filter, project ...
    }
}
```

The LLM decides the code **structure** (direct-array threshold, hash function
choice, null handling, SIMD patterns). The runtime decides the code
**parameters** (which branch to take, what values to filter on).

## 10. SQL-Level Compilation Without LLM vs. sql_jit_fail.md

### 10.1 What sql_jit_fail.md Did

Compiled the **probe-side subtree of the original query's physical plan**
into a single LLVM coordinator function that replaced DuckDB's operators:

```
Original: DuckDB PhysicalPlan → [Scan → Filter → HashProbe → HashProbe → Project]
SQL-JIT:  DuckDB PhysicalPlan → [PhysicalAQPSubPlan(compiled coordinator)]
```

- Operated on the **full query** (7-17 tables, 4-28 joins)
- Had to match DuckDB's physical plan topology (right-deep/bushy)
- Required `PhysicalAQPSubPlan` to replace the probe-side subtree
- Required MetaPipeline dependency wiring for build-side hash tables
- Failed on 112/113 queries due to plan mismatch

### 10.2 What SQL-Level Compilation of Sub-SQLs Would Do

Compile a **complete standalone sub-SQL** (after splitting) into LLVM:

```
Sub-SQL: "SELECT t.id, ci.person_id FROM temp_1 JOIN cast_info ci ON ... WHERE ..."
Compile: LLVM function that scans temp_1, probes hash table, filters, projects
Execute: Call compiled function, result becomes next temp table
```

### 10.3 Why It's Fundamentally Different

| Aspect | sql_jit_fail.md | Sub-SQL compilation |
|---|---|---|
| **Input** | Full query (7-17 tables) | Sub-SQL (2-3 tables, 1 join) |
| **Plan topology** | Must handle right-deep/bushy | Trivial — one join, no topology issue |
| **Integration** | Replaces DuckDB's physical operators | Replaces the entire `ExecuteSQLandCreateTempTable` call |
| **Plan surgery** | Walks DuckDB's physical plan, matches IR stages to physical operators by table name | No plan surgery — generates a fresh standalone function from the sub-SQL's IR |
| **Build-side dependency** | Must wire into DuckDB's MetaPipeline for hash table builds | Builds its own hash table (or uses direct-array if cardinality is small) |
| **Cardinality knowledge** | Only has optimizer estimates | Has **actual** cardinalities from previous sub-query results |
| **Applicability** | 1/113 queries (plan mismatch on rest) | All sub-queries are candidates |

### 10.4 Why the Original Failure Modes Don't Apply

1. **Plan mismatch eliminated**: A sub-SQL has ~1 join. There is no left-deep
   vs right-deep issue — the plan is always `scan → probe → filter → project`.
   DuckDB's optimizer chooses the join order for each sub-SQL, and the compiled
   function implements exactly that single join.

2. **No physical plan surgery**: The compiled function is called **instead of**
   `ExecuteSQLandCreateTempTable`, not spliced into DuckDB's operator tree.
   DuckDB is not involved in execution of that sub-query at all.

3. **Materialisation cost is real this time**: The original sql_jit_fail.md
   eliminated inter-operator materialisation within DuckDB (L2-resident chunks
   → negligible). Here, the compiled function eliminates **DuckDB's entire
   execution stack**: SQL parsing, optimisation, plan building, operator
   dispatch, buffer management. For small temp tables, this overhead is a
   significant fraction of execution time.

4. **Hash probe remains memory-bound**: This limitation still applies. The
   compiled function's hash probe won't be faster than DuckDB's for large
   hash tables. The gain comes from:
   - Eliminating DuckDB's per-sub-query fixed overhead (~0.5-2ms)
   - Using **data-adaptive join strategies** (direct-array when build side
     is small, which DuckDB doesn't do)
   - Skipping null checks, type dispatch, and generic operator dispatch

### 10.5 Expected Gain

The 15 regression iterations (where pipeline-jit execution hurts) are at
iter 5-6 with ~10-16ms pipe exec vs ~0.5-1ms no-jit exec. These are caused
by JIT-compiled probe code being slower than DuckDB's native probe. A
sql-level compiled sub-SQL could:
- Use the same direct-array / hash probe strategy as the current JIT for
  filter-heavy sub-queries (where 260x speedup is already observed)
- Fall back to DuckDB's native hash probe for join-heavy sub-queries
  (avoiding the 15 regression iterations entirely)
- Eliminate DuckDB's per-sub-query overhead for trivially small sub-queries
  (the 28 "wasted JIT" iterations where compile >> execute)

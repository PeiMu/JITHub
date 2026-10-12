# Topdown Speculative JIT: Analysis and Implementation Plan

## Goal

Enable speculative JIT compilation for the **topdown** split strategy, matching the existing node-based spec-jit support. Currently, topdown only has limited Phase B speculation (query-jit only, est_rows >= 1000). Full Phase A speculation (overlapping Execute(i) with bg compile for i+1) is blocked because TopDownSplitter lacks a `PeekNextSubquery()` equivalent.

## Background: How Node-Based Spec-JIT Works

The existing spec-jit pipeline has two phases:

**Phase A** (early, overlaps Execute(i)):
- Triggered by the adapter's **post-Prepare hook** — fires after DuckDB's `Prepare(sub_sql_i)` but before `Execute`.
- `LaunchSpeculativeCompile()` calls `NodeBasedSplitter::PeekNextSubquery()` to predict iteration i+1's sub-IR.
- Generates speculative SQL, launches background JIT compilation on `jit_compile_pool_`.
- Guard at `ir_query_splitter.cpp:2335`: `if (config_.strategy != SplitStrategy::NODE_BASED) return;`

**Phase B** (late, after UpdateRemainingIR):
- `PrecomputeNextExtraction()` runs the real `SplitIR(i+1)` on updated IR.
- For node-based: compares Phase B SQL with Phase A's speculative SQL. If they match, keeps the Phase A compile. If not, learns the miss.
- For topdown: currently only launches a bg compile in a limited case (query-jit mode, est_rows >= 1000). See `ir_query_splitter.cpp:2793-2807`.

**Outcome check** (at iteration i+1's execution):
- `CheckSpeculativeResult()` (`ir_query_splitter.cpp:2982-3099`) compares:
  1. **SQL match**: speculative SQL vs actual SQL (zero-wait, catches wrong group selection)
  2. **Card guard**: if `assumed_card` vs actual temp cardinality diverges by >2x ratio, rejects (catches physical plan quality issues)
- On hit: use the bg-compiled plan (skip inline compile)
- On miss: fall through to inline TPDE recompile (`compensate_fast`) or interpret

**Counters** (per query, printed in `[AQP-SPECJIT] summary`):
- `hits`: speculation correct, no recompile (includes `not_ready` waits)
- `compensate_fast`: speculation wrong, inline TPDE recompile
- `misses`: SQL mismatch (subset of compensate causes)
- `card_misses`: cardinality guard failed (subset of compensate causes)
- `bg_errors`: background compile failed
- `not_ready`: subset of hits where main thread waited for bg compile

## Why TopDown Can't Do Phase A Today

`NodeBasedSplitter::PeekNextSubquery()` (`node_based_splitter.cpp:413-510`) works because:
1. DuckDB's QuerySplit pre-computes a queue of sub-plans (`subqueries_`).
2. Peeking at `subqueries_.front()` gives a reasonable prediction without re-running optimization.
3. It injects a speculative `LogicalColumnDataGet` (placeholder chunk) with estimated cardinality.

`TopDownSplitter` cannot peek because:
1. It uses an IR-native DP optimizer (`IRJoinOptimizer`) that runs on the **updated** remaining IR.
2. `PlanNext()` requires `table_card_[temp_idx]` — the exact cardinality of the just-executed temp.
3. After `UpdateRemainingIR()`, `planned_for_` is invalidated (set to `nullptr`), forcing a full DP re-run.
4. `TopDownSplitter` has no `PeekNextSubquery()` or `HasNextSubquery()` methods.

## Proposed Approach: Speculative PlanNext with Estimated Cardinality

### Flow

During Execute(i), before the actual temp cardinality is known:

1. Set `table_card_[temp_idx] = planned_card_` (the DP's estimated output for iteration i, already computed at zero cost in `SplitIR(i)` and stored in `extraction->estimated_rows`).
2. Build a speculative remaining IR by applying `UpdateRemainingIR` with estimated cardinality.
3. Run `PlanNext()` on this speculative IR to get `planned_group_` for iteration i+1.
4. Build speculative sub-IR via `BuildSubIRForCluster()`, generate SQL, launch bg compilation.
5. When Execute(i) completes and actual cardinality is known, the real `PlanNext(i+1)` runs.
6. `CheckSpeculativeResult()` compares speculative SQL vs real SQL. SQL match = hit; mismatch = compensate.

### Implementation Sketch

Add to `TopDownSplitter`:

```cpp
// Predict the next group using estimated cardinality from the current iteration.
// Returns the speculative sub-IR (caller generates SQL and launches bg compile).
std::unique_ptr<SubqueryExtraction> PeekNextExtraction(
    ir_sql_converter::AQPStmt *remaining_ir,
    const std::set<unsigned int> &current_executed_tables,
    unsigned int spec_temp_index,
    const std::string &spec_temp_name,
    double spec_temp_card,
    const std::vector<std::pair<unsigned int, unsigned int>> &col_mappings,
    const std::vector<std::string> &col_names);
```

Inside this method:
- Clone the remaining IR (or work on a temporary copy).
- Apply `UpdateRemainingIR` with `spec_temp_card` as the cardinality.
- Insert `spec_temp_index` into a temporary `temp_indices_` set.
- Set `table_card_[spec_temp_index] = spec_temp_card`.
- Run `PlanNext()` on the speculative IR.
- Build the sub-IR via `BuildSubIRForCluster()`.
- **Restore** all state: revert `table_card_`, `temp_indices_`, `planned_for_`, etc.
- Return the speculative extraction.

Wire into `ir_query_splitter.cpp`:
- Register the post-Prepare hook for `TOP_DOWN` (currently line 1522-1523 restricts to `NODE_BASED`).
- In the hook callback, call `TopDownSplitter::PeekNextExtraction()` instead of `NodeBasedSplitter::PeekNextSubquery()`.
- The rest of the Phase A pipeline (SQL generation, bg compile launch, SQL match check) is strategy-agnostic.

### Key Files to Modify

1. **`include/split/topdown_splitter.h`**: Add `PeekNextExtraction()` declaration.
2. **`src/split/topdown_splitter.cpp`**: Implement `PeekNextExtraction()`.
3. **`src/split/ir_query_splitter.cpp`**:
   - Line 1522-1523: Remove `NODE_BASED` restriction on hook registration (or add `TOP_DOWN` case).
   - `LaunchSpeculativeCompile()` line 2335: Remove `NODE_BASED` guard, add topdown path.
   - Need a topdown-specific branch that calls `PeekNextExtraction()` instead of `PeekNextSubquery()`.
4. **`include/split/topdown_splitter.h`** or **`src/split/fk_based_splitter.cpp`**: May need to expose `UpdateRemainingIR` in a way that supports speculative (non-destructive) calls.

### Engineering Challenges

**Building the speculative remaining IR**: `UpdateRemainingIR` is currently destructive (takes ownership via `std::move`). For speculation, you need either:
- A deep clone of the remaining IR before applying the speculative update, or
- A non-destructive `UpdateRemainingIR` variant that works on a copy.

The clone approach is simpler but has a cost proportional to IR size. For JOB queries (5-17 tables), this is small (~microseconds).

**State restoration in TopDownSplitter**: `PlanNext()` modifies `planned_for_`, `planned_splittable_`, `planned_group_`, `planned_card_`, and reads `table_card_`, `temp_indices_`, `col_distinct_hints_`, `explain_cache_`. All must be saved and restored after the speculative run.

**EXPLAIN calls in PlanNext**: For temp-involving pairs, `explain_validate()` (lambda at L1121) issues `adapter_->BatchGetEstimatedCosts()` (~0.3ms each). This contacts the engine. **`bg_mode_ = true` does NOT skip EXPLAINs — it throws `std::runtime_error("SDS bg-mode: engine EXPLAIN needed")` at L1146-1147 and L1404-1405.** Since ~75% of speculative iterations involve temps, the throw would fire frequently. For speculation, add a new `spec_mode_` flag (or similar) that makes the `explain_validate` lambda return `true` unconditionally (skip EXPLAIN, trust the DP estimate). The DP estimate alone is sufficient for group selection; the SQL-match check catches errors. Running EXPLAIN on the main thread is also possible (the adapter is available in the post-Prepare hook) but adds ~0.3-1ms latency per pair candidate.

## Cardinality Estimation Accuracy (Empirical Data)

Measured across all 455 sub-query executions (113 JOB queries, topdown split, duckdb adapter, single-run-template cache, tpde compiler):

### DP/tdom estimate (`planned_card_` = `extraction->estimated_rows`)

| Metric | Value |
|--------|-------|
| Within 1.5x of actual | 14.4% (65/451) |
| Within 2x of actual | 19.3% (87/451) |
| Within 5x of actual | 49.4% (223/451) |
| Within 10x of actual | 55.7% (251/451) |
| Median error ratio | 5.5x |
| P90 error ratio | 223x |
| Max error ratio | 319,480x |

### DuckDB Prepare estimate (from `physical_plan->Root().estimated_cardinality`)

| Metric | Value |
|--------|-------|
| Within 1.5x of actual | 11.8% (53/451) |
| Within 2x of actual | 15.7% (71/451) |
| Within 5x of actual | 35.7% (161/451) |
| Within 10x of actual | 43.7% (197/451) |
| Median error ratio | 17.5x |
| P90 error ratio | 45,969x |

### By iteration number (DP estimate, within 2x)

| Iter | Count | DP within 2x | Prepare within 2x |
|------|-------|-------------|-------------------|
| 1 | 111 | 4% | 15% |
| 2 | 101 | 21% | 12% |
| 3 | 82 | 20% | 16% |
| 4 | 60 | 20% | 18% |
| 5 | 40 | 42% | 12% |

### Key Finding

**The DuckDB Prepare/EXPLAIN estimate is WORSE than the DP/tdom estimate** (median 17.5x vs 5.5x error). DuckDB's optimizer systematically overestimates multi-table join outputs when filter selectivities compound. Since EXPLAIN uses the same optimizer, there is no benefit to running extra EXPLAINs for speculation cardinality.

### Recommendation: Use `planned_card_` (DP estimate) at zero cost

- It's already computed as part of `SplitIR(i)`.
- It's more accurate than DuckDB's Prepare estimate.
- Neither is accurate enough for the existing 2x card guard (19% vs 16% pass rate).

## Speculation Accuracy: What Actually Matters

For spec-jit, cardinality accuracy is NOT the primary concern. What matters is **group stability**: does `PlanNext(i+1)` pick the **same set of tables** with estimated vs actual cardinality?

### Two safety nets

1. **SQL-match check** (`CheckSpeculativeResult`, line 2991-2993): Compares speculative SQL with actual SQL. Catches wrong group selection at **zero wait cost** (no future.wait needed — the SQL is stored on the main thread). This is the primary safety net.

2. **Card guard** (line 3030): Checks if `assumed_card` diverges from actual by >2x. Catches cases where the right tables are picked but the physical plan (build/probe side, HT sizing) would be poor.

### Analysis of group stability (from debug build, ~280 non-first iterations)

- **21.8% of iterations** select groups with only base tables (no temps) — these are **cardinality-insensitive** and the speculation is guaranteed correct regardless of estimation accuracy.
- **75% of iterations** select groups involving the previous temp — for these, group stability depends on whether the cardinality error shifts the DP's cost ranking.
- Overall, with the DP estimate and a 2x card guard: ~43% of iterations would be "safe" (either base-only or estimate within 2x).

### Strategy for high hit rate

1. **Always speculate**: use `planned_card_` as the temp cardinality for speculative `PlanNext`.
2. **Relax the card guard**: For topdown speculation, the DP already chose a good join order. The physical plan quality is bounded. Consider a 5x or 10x threshold, or remove the card guard entirely for topdown.
3. **SQL-match as primary filter**: Wrong group selections are caught by the SQL comparison at zero cost. Only card-miss (right tables, wrong physical plan) is a concern, and it's less critical for topdown because the DP controls join ordering.
4. **Learn from misses**: The existing `g_spec_miss_history` mechanism prevents re-launching speculations that failed on previous repeats. For `single-run-template` cache (each query runs once), this only helps within a single query's iteration sequence.

## Sub-Query Counting Reference (JOB Benchmark)

For context on the scale of speculation opportunity:

| Split | Total sub-queries | Queries | Spec-eligible (total - first - final) |
|-------|-------------------|---------|--------------------------------------|
| topdown | 568 | 113 | 342 |
| node-based | 643 | 113 | 417 |

Node-based measured spec-jit results (from `count_spec_jit_rate.sh`):
- 389 speculation outcomes (hits + compensate_fast)
- 21 hits (5.4% correct rate with tpde, single-run-template)
- 368 compensate_fast
- 28 iterations skipped (cross-product, null peek)

## count_spec_jit_rate.sh Counting Explained

The script counts `total = hits + compensate_fast`. This equals the number of sub-queries where a speculation **outcome was recorded**, not the total spec-eligible count. The gap (417 - 389 = 28 for node-based) comes from:

1. **Cross-product sub-queries** (`ir_query_splitter.cpp:2396,2734`): `HasCrossProduct(spec_ir)` → no launch.
2. **`PeekNextSubquery` returns nullptr** (lines 2385, 2599): Nothing to speculate on.
3. **Final-iteration drain**: `DrainSpecs()` retires unconsumed speculation without incrementing outcome counters.

Sub-queries NOT counted: first iteration per query (113), final execution per query (113), and the ~28 skipped iterations.

## Data Collection Commands

To reproduce the cardinality estimation analysis:

```bash
# Temporarily add logging to ir_query_splitter.cpp at line ~3795
# (after cardinality is known, before the if(cardinality==0) block):
#
#   std::cerr << "[SPEC-EST] iter=" << iteration_count_
#             << " dp_est=" << extraction->estimated_rows
#             << " prepare_est=" << <adapter prepare est>
#             << " actual=" << cardinality << "\n";
#
# Then rebuild and run:

bash run_aqp.sh job duckdb topdown query none on on on all single-run-template off tpde
# stderr contains [SPEC-EST] lines

# For sub-query counting:
bash count_join_query_count.sh

# For spec-jit correct rate (node-based only, current):
bash count_spec_jit_rate.sh
```

## Raw Data Location

- `spec_est_log.txt`: 455 `[SPEC-EST]` lines with dp_est, prepare_est, actual for all JOB queries (topdown, duckdb, single-run-template, tpde). This file is ephemeral; re-run the instrumented build to regenerate.
- `spec_jit_rate_results.txt`: Results from `count_spec_jit_rate.sh` (node-based configs).
- `aqp_spec_jit_dumps/`: Per-config stderr dumps from spec-jit rate measurement.

## Repositories

- **AQPHub**: `/home/pei/Project/AQP_middleware`
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **DSB queries**: `/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_aqp/1/`
- **TPC-H queries**: `/home/pei/Project/benchmarks/tpch-postgres/dbgen/out_50/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DSB schema**: `/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql`
- **TPC-H schema**: `/home/pei/Project/benchmarks/tpch-postgres/dbgen/dss.ddl`
- **DuckDB JOB database**: `/home/pei/Project/duckdb/measure/imdb.db`
- **DuckDB DSB database**: `/home/pei/Project/duckdb/measure/dsb_50.db`
- **DuckDB TPC-H database**: `/home/pei/Project/duckdb/measure/tpch_50.db`

Build commands:
```bash
# Middleware (debug / release)
cd /home/pei/Project/AQPHub && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/AQPHub && cmake --build build_release -j$(nproc)
# DuckDB — the middleware links the PREBUILT libduckdb.so from
# ${DUCKDB_ROOT}/build/release/src (find_library + dynamic link).
cd /home/pei/Project/duckdb && cmake --build build/release -j 16
```

Build hazards:
- build_debug and build_release share `lib/*.a` outputs. If a debug build
  poisons the release archive (ASan), rebuild with
  `--target IR_SQL_Converter_C_static --clean-first`.

## Verification Workflow

### Step 1: Single query end-to-end

Take a simple JOB query that related to our changes (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_aqp.sh job). Compare result to DuckDB golden output.

### Step 2: Full JOB and DSB correctness

check measure/correctness_test_job_duckdb.sh and measure/correctness_test_dsb_duckdb.sh.

### Analysis scripts (measure/*.py)
CSV parser is: /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
---

## Implementation Status: Phase A Spec-JIT for TopDown — COMPLETE

### What was implemented

1. **`ir_utils::CloneIR()`** (`include/split/ir_utils.h`, `src/split/ir_utils.cpp`): Deep-clone of AQPStmt IR trees. Handles Projection, Join, Filter, Scan, Chunk, CrossProduct, Aggregate, OrderBy, Limit, Distinct node types.

2. **`TopDownSplitter::PeekNextExtraction()`** (`include/split/topdown_splitter.h`, `src/split/topdown_splitter.cpp`): Speculative prediction of the next sub-query. Clones the remaining IR, applies UpdateRemainingIR with estimated cardinality, runs SplitIR, then restores all state. Uses `spec_mode_` flag to skip EXPLAIN calls during speculative PlanNext.

3. **`LaunchSpeculativeCompile` TopDown branch** (`src/split/ir_query_splitter.cpp`): Strategy dispatch in the post-Prepare hook. For TopDown: uses stashed extraction metadata (`spec_stash_`) to call PeekNextExtraction, generates SQL, and launches bg compile. Supports both DuckDB and PostgreSQL backends.

4. **Hook registration for TopDown** (`src/split/ir_query_splitter.cpp`): Both DuckDB and PG post-Prepare hooks now fire for TOP_DOWN in addition to NODE_BASED.

5. **Relaxed card guard for TopDown** (`src/split/ir_query_splitter.cpp`): Card guard threshold is 5x for TopDown (vs 2x for node-based), accounting for the DP estimate's higher variance.

6. **Phase B kept as fallback**: Existing Phase B bg compile at L2793-2807 still fires when Phase A doesn't launch (e.g., learned miss).

### Results (JOB benchmark, topdown, query-jit, tpde, single-run-template cache)

| Metric | TopDown Phase A | Node-Based (reference) |
|--------|----------------|----------------------|
| Total speculations | 377 | 389 |
| Hits | 165 (43%) | 21 (5.4%) |
| SQL misses | 0 (0%) | 368 |
| Card misses (5x/2x) | 49 (12%) | — |
| Compensate fast | 163 (43%) | 368 |
| BG errors | 0 | — |

Key finding: TopDown Phase A achieves **43% hit rate** vs node-based's 5.4%. The DP consistently selects the same table group regardless of cardinality estimate accuracy (zero SQL misses). The remaining card_misses (12%) could be further reduced by relaxing the 5x threshold.

### Performance Issue: +13.8% regression (v1)

Benchmark: `measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template {off,recompile} tpde`

| Component | Baseline | Spec-JIT v1 | Delta |
|-----------|---------|------------|-------|
| Compilation | 0.568s | 0.615s | +0.047s (+8%) |
| Middleware | 0.071s | 0.255s | +0.184s (+259%) |
| Execution | 4.116s | 4.541s | +0.425s (+10%) |
| **Total** | **4.755s** | **5.411s** | **+0.656s (+14%)** |

**Root cause**: PeekNextExtraction runs on the **main thread** (inside the post-Prepare hook), adding ~0.4ms serial overhead per iteration before Execute starts. For node-based, PeekNextSubquery is trivially cheap (~µs, peeks at a pre-computed queue) so it runs on the main thread too — only the Prepare+JIT compile runs on the bg thread. But topdown's PeekNextExtraction is expensive (IR clone + UpdateRemainingIR + full DP optimizer).

**Where time is charged**: PeekNextExtraction runs during DuckDB's Prepare step, so its ~0.4ms is charged to the `jit_compile` CSV column. The +425ms execution delta is core contention from the bg Prepare+JIT thread competing with DuckDB's execution threads.

### Optimization v2: kSpecMinEstRows threshold

**Bg thread approach attempted and reverted**: Moving PeekNextExtraction to the bg thread causes data races — the save/restore pattern modifies TopDownSplitter members (`table_index_to_name_`, `table_card_`, etc.) that the main thread may access concurrently during execution. The node-based path avoids this because PeekNextSubquery (~µs) doesn't modify splitter state. A proper fix would require PeekNextExtraction to operate on a fully detached state snapshot (separate TopDownSplitter instance or static method with copied state), which is a significant refactor. Deferred.

**Applied: kSpecMinEstRows = 1000 threshold for Phase A.** Skips speculation when `extraction->estimated_rows < 1000`. This avoids overhead on the 41% of JOB sub-queries that execute in <1ms, where inline TPDE (~0.7ms) is cheaper than bg pipeline startup (~1.2ms). Applied to both DuckDB and PG paths.

**Removed: ApplyCrossSubPlanOptimizations from TOP_DOWN Phase A.** Range-pred injection has no data at post-Prepare time (temp table stats aren't ready yet). The raw SQL is used for both Phase A and Phase B SQL-match comparison. The HIT path's inline code applies range preds to the real SQL anyway.

### v3 Results: Phase B Lookahead (lightweight prediction)

Moved PeekNextExtraction from the Phase A hook (before Execute) to Phase B (after Execute). At Phase B time, predict iteration i+2 and store the SQL. The hook just uses the stored SQL — no DP on the hot path (~µs instead of ~0.4ms).

| Version | Compilation | Middleware | Execution | **Total** | Hit Rate |
|---------|------------|-----------|-----------|-----------|----------|
| Baseline | 0.568s | 0.071s | 4.116s | **4.755s** | — |
| v1 (main-thread PEEK) | 0.615s | 0.255s | 4.541s | **5.411s** | 43% |
| v2 (+ threshold) | 0.625s | 0.257s | 4.533s | **5.415s** | 68% |
| v3 (lookahead) | 0.653s | 0.254s | 4.288s | **5.195s** | 100% |

**v3 improvements over v2**: Execution -245ms (less core contention — PEEK moved off hot path). Hit rate 100% (lookahead always matches real SQL). All 180 hits are `not_ready` (bg compile hasn't finished at check time).

**Remaining overhead**: The DP work (~0.4ms/iter) shifted from the hook to Phase B — same total middleware overhead. The bg compile starts at Prepare(i+1) but only overlaps ~0.3ms before CheckSpeculativeResult runs, causing ~0.9ms spec-waits per hit (charged to compilation). Total +440ms over baseline.

**Architecture recap** (verified against code):
- `jit_compile_pool_` = **single-thread** pool (ThreadPool(1))
- Hook fires **twice** per iteration: post-Prepare (`post_execute=false`, does the real launch) and post-Execute (`post_execute=true`, returns early)
- Phase A launches bg compile at **post-Prepare(i)**, targeting iteration **i+1**
- **CheckSpeculativeResult(i+1)** runs at the start of iteration i+1, **before** Prepare(i+1)
- Overlap window = Execute(i) + inter-iteration gap (~10-100ms) — plenty for 1.2ms bg compile
- For node-based: PeekNextSubquery at Prepare(i) costs ~µs → negligible overhead
- For topdown: PeekNextExtraction at Prepare(i) costs ~0.4ms → measurable overhead

**Lookahead-2 approach** (precompute i+2's SQL during Phase B of i, use at Prepare(i+1)):
- Architecturally correct: bg compile at Prepare(i+1) overlaps Execute(i+1), checked at iter i+2
- **Failed due to SQL alias mismatch**: `GenerateSQL(ir, query_id)` uses `query_id` for table alias suffixes. The lookahead's `query_id` (captured at Phase B of iter i) differs from the real `query_id` at iteration i+2 (incremented by intermediate iterations). `NormalizedSqlEquals` doesn't normalize alias suffixes.
- Fix options: (a) normalize SQL comparison to ignore alias suffixes; (b) use a fixed query_id=0 for both lookahead and real SQL; (c) store the predicted sub-IR and regenerate SQL at Phase A time with the correct query_id (but that requires the IR to survive across iterations).

### Verification

- Build: passes (release + debug)
- Correctness: all 113 JOB queries pass for topdown+spec-jit configs
- CSV format: unchanged (no new columns)
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`

# Speculative JIT (spec-jit) for the split strategies: analysis, measurements, final state

Last revised 2026-10-13. Workload for every number: JOB (113 queries),
DuckDB adapter, query-jit, cache mode `single-run-template`, 15 repeats with
the first 5 discarded, measured with
`bash measure_breakdown_time_aqp.sh job duckdb {topdown,node-based} query none on on on all single-run-template {off,recompile} {tpde,fastisel,llvm}`
and parsed with `print_breakdown.py`. Machine: Intel E-2236, 6 cores / 12
threads; the qjit executor uses 12 workers.

## Final state and verdict

Spec-jit overlaps the compile of sub-query i+1 with the execution of
sub-query i. Under single-run semantics (no knowledge from earlier runs of
the same query) it does **not** beat the no-spec baseline on this machine
for any split strategy or backend. Last script measurement of the tree as
it is now (results byte-identical to the baseline in every run):

| configuration | off | recompile | Δ |
|---|---|---|---|
| topdown tpde | 4.654 | 4.736 | +1.8% |
| topdown fastisel | 5.863 | 5.941 | +1.3% |
| topdown llvm | 9.019 | 9.032 | +0.1% |
| node-based tpde | 5.079 | 5.190 | +2.2% |
| node-based fastisel | 6.142 | 6.294 | +2.5% |
| node-based llvm | 8.884 | 9.099 | +2.4% |

Run-to-run noise between sessions is ~2%, so topdown is neutral and
node-based is a small loss. The table predates the code-review fixes
listed under "Bugs and hazards found" (the window rule now extrapolates
from the running step only, so fewer early commits fire); results stay
byte-identical, the timing is to be re-measured. Why no design can do
much better here, all
measured below: only `min(compile(i+1), execute(i))` can be hidden, which
is 7-12% of total time before costs; the next sub-query depends on the
previous temp's actual cardinality (55% one-step predictability for topdown
with its estimate, 35-44% for node-based); a background compile shares a
physical core with an executor thread (+37% compile time, execution slowed);
and the only mechanism that reaches the oracle bound (−2.5% node-based,
−1% topdown) is memorizing the split sequence across runs of the same query.

What `--spec-jit=recompile` does now is described in "What is in the tree
now" near the end; everything in between is the investigation record.

## Investigation history (TL;DR of the 2026-10-12 analysis)

* The baseline the spec-jit work has to beat is
  `bash measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template off <tpde|fastisel|llvm>`
  (tpde 4.631 s, fastisel 5.892 s, llvm 8.901 s; a tpde re-run gave 4.636 s).
* Core idea (correct): hide compile(i+1) behind execute(i). Two bounds it
  leaves out: only `min(compile(i+1), execute(i))` can be hidden, and the
  speculated code must be the code the inline path would have produced.
* The topdown code in commit 7069036 never overlapped anything. Its Phase B
  launch runs after `Execute(i)` and is always waited on
  (`not_ready == hits`): **+7.3% (tpde), +6.6% (fastisel), +7.8% (llvm)**.
  The revert is correct.
* Topdown spec-jit is mechanically feasible. A cross-repeat oracle (exact
  next sub-SQL/IR from the previous repeat, 100% hits, identical results)
  measures the upper bound of any predictor. Best variant so far
  (oracle + metadata-only placeholder + same-IR bg compile):
  **llvm 8.802 s (−1.1%), fastisel 5.927 s (+0.6%), tpde 4.683 s (+1.1%)**;
  by wall clock all three are 0 to −1.5%.
* The bg compile is the same single-threaded work as the inline compile
  (serial bg 0.93 ms vs inline 0.96 ms, tpde) but when overlapped on this
  6-core/12-thread CPU it runs 37% slower (1.27 ms) and slows the execution
  it overlaps by ~2-3%. Hiding is not free on a fully loaded machine.
* A wrong guess is not free either: with every speculation missing
  (forced-miss run) the total is +5.0% (tpde) / +4.9% (llvm). With the
  measured 55% one-step prediction accuracy the expected result is a loss
  for all backends; break-even needs ~82% accuracy for llvm and is
  unreachable for tpde/fastisel.
* Node-based: the oracle (perfect prediction + exact cardinality) wins
  **−2.5% tpde / −2.3% llvm**, but the next sub-query depends on the actual
  temp cardinality through DuckDB's re-split (93% of sub-SQLs change with
  the 46x-error Prepare estimate), the queue-front peek is right 35-44% of
  the time, and the 2x card guard rejects ~90% of those (justified: +8.9%
  execution when bypassed). Removing the re-split for predictability costs
  +40% (tpde) / +21% (llvm). Realistic best case ≈ −0.3% tpde (noise),
  loss for llvm. Only cross-run memoization reaches the oracle numbers.

## Background: how node-based spec-jit works (verified)

Two phases per iteration i (`src/split/ir_query_splitter.cpp`):

**Phase A** — the adapter's post-Prepare hook fires inside
`ExecuteSQLandCreateTempTable` after the inline compile/Prepare of
sub-query i and *before* its execution (`duckdb_adapter.cpp`, first
`post_prepare_hook_` call). `LaunchSpeculativeCompile()` asks
`NodeBasedSplitter::PeekNextSubquery()` for sub-IR i+1, generates SQL,
registers a placeholder for temp(i) (`RegisterPlaceholderTemp`) so the bg
Prepare can bind it, and submits the bg compile to `jit_compile_pool_`
(1 thread). The compile overlaps `Execute(i)`.

**Phase B** — `PrecomputeNextExtraction()` runs after `UpdateRemainingIR`
(actual cardinality known): real `SplitIR(i+1)`, SQL comparison with the
Phase A speculation ("kept" on match, learned miss otherwise). For
TOP_DOWN it used to launch a bg compile here (removed: see TL;DR).

**Check** — `CheckSpeculativeResult()` at iteration i+1: raw-SQL match
(zero wait), 2x cardinality guard, then wait for the future
(`not_ready` counts waits). HIT → `SetQjitSpecHit`, no inline compile.
MISS → inline compile (`compensate_fast`) or interpret.

Counters printed in `[AQP-SPECJIT] summary` (stderr, only with
`--spec-jit-stats` or `--debug`; performance runs neither count nor print):
checks, hits, misses, card_misses, not_ready, bg_errors, compensate_fast,
compensate_interp, wait_ms. `checks` is the number of sub-queries that
could have used a speculation (every sub-query except the first of each
query). Every launched speculation ends in exactly one of hits / misses /
card_misses / bg_errors; `misses` includes the speculations retired before
the check (Phase B SQL mismatch, next iteration final or cross product,
never consumed). `measure/count_spec_jit_rate.sh` reports per run:
spec-check = checks, correct = hits, wrong = misses + card_misses +
bg_errors, skipped = checks − correct − wrong (no launch), not_ready
(correct but the compile was late), correct% = correct / spec-check and
ready% = (correct − not_ready) / spec-check, from `--spec-jit-stats` runs
(single repeat, otherwise the same flags as the measurement script).

## Why topdown has no cheap peek

`TopDownSplitter::PlanNext()` (IR DP, `IRJoinOptimizer`) runs on the
*updated* remaining IR and needs `table_card_[temp_i]`, the exact
cardinality of the temp produced by iteration i. `UpdateRemainingIR`
invalidates `planned_for_`, so iteration i+1's group is only known after
`Execute(i)`. Temp-involving pair candidates are additionally validated
with engine EXPLAINs (`explain_validate`, ~0.3-1 ms each); `bg_mode_`
*throws* instead of issuing them.

## Measured facts

### Baseline time budget (tpde, template cache)

| component | total | per sub-query (455) |
|---|---|---|
| jit_compile, sub-queries | 0.373 s | 0.82 ms (median 0.63) |
| jit_compile, final query | 0.152 s | 1.3 ms |
| execute, sub-queries | 2.202 s | 4.84 ms (median 2.61, p25 0.79) |
| execute, final query | 1.838 s | 16.3 ms |
| middleware | 0.066 s | |
| total | 4.631 s | |

llvm: compile 5.418 s (sub 3.922 s = 8.6 ms avg), execute 3.417 s
(sub 1.957 s = 4.3 ms avg), total 8.901 s. fastisel: 0.910 / 4.912 /
5.892 s. Note: the template cache is cleared between repeats
(`single-run-*` semantics, `aqp_middleware.cpp`), so inline compiles are
cold for the first occurrence of a shape in each pass; within a pass 61%
of inline compiles are cache hits (X6 trace).

### Ideal overlap ceiling (`measure/analyze_spec_overlap.py`)

hideable = sum_i min(jit(i+1), exe(i)) for sub-queries + min(jit_final,
exe(last sub)) for the final query, assuming the bg compile is as fast as
the inline one and costs nothing:

| backend | sub hideable | final hideable | total | of total time |
|---|---|---|---|---|
| tpde | 0.259 s (69% of sub jit) | 0.130 s | 0.388 s | 8.4% |
| fastisel | 0.401 s (61%) | 0.192 s | 0.593 s | 10.1% |
| llvm | 0.737 s (19%) | 0.341 s | 1.077 s | 12.1% |

32% of sub-queries execute in less than the bg compile's 1.35 ms (tpde),
so for those nothing is hidden and the main thread waits.

### Inline vs background compile pipeline (tpde, `AQP_QJIT_FAST_TRACE`, `AQP_SPEC_TRACE`)

| stage | inline fast path (ms, mean / median) | bg `SpeculativeQueryJitCompile` (ms, mean / median) |
|---|---|---|
| plan: `TryBuildBinaryPlanFromIR` / SQL parse+bind+Optimize | 0.10 / 0.09 | 0.51 / 0.39 |
| plan-to-IR (`ConvertDuckDBPlanToIR`) | – | 0.03 |
| `PrepareFromPlan` | 0.02 | 0.03 |
| analyze + `BuildExecutionSteps` | 0.002 | 0.01 |
| `CompileQuerySteps` (cache lookup / codegen) | 0.83 / 0.19 | 0.76 / 0.70 |
| total | 0.95 / 0.39 | 1.35 / 1.10 |
| `new Connection` + `ResetModules` | – | 0.055 |
| thread-pool queue wait | – | 0.06 |

The bg path produces a different IR (fresh binder over DuckDB's optimized
plan) and therefore a different structural cache key: in a 3-repeat run
the bg compiles hit the cache 45% of the time while the inline path hit
61%, and the bg never hits entries the inline path filled. Running the bg
compile concurrently with execution slows it from 0.95 ms (Phase B, no
overlap) to 1.49 ms (Phase A, overlapped). For llvm the bg compile median
is 12 ms versus a 4 ms average execute, so 57-60% of hits still wait.

### Split-decision stability with estimated cardinality (X4)

`--print-sql --repeat=1` with and without `--no-update-temp-card` (the DP
and the engine see `extraction->estimated_rows` instead of the actual
temp cardinality), compared with `measure/compare_sql_sequences.py`:

* iteration-2 SQL identical (pure one-step lookahead): **56/102 = 55%**
* all later sub-queries identical (errors compound): **124/417 = 30%**
* queries with a fully identical sequence: 39/113; same number of
  iterations: 49/113

So a PeekNextExtraction-style predictor seeded with the DP estimate would
mispredict roughly every other speculation; each mispredict is a wasted
bg compile plus a useless placeholder.

### Cardinality estimates (reproduced from `measure/spec_est_log.txt`, 455 sub-queries)

| metric | DP / tdom (`planned_card_`) | DuckDB Prepare estimate |
|---|---|---|
| within 1.5x | 14.4% | 12.2% |
| within 2x | 19.3% | 16.4% |
| within 5x | 49.4% | 37.2% |
| within 10x | 55.7% | 45.5% |
| median error | 5.5x | 15x |

The DP estimate is the better (free) estimate, but neither passes the 2x
card guard often, and — more importantly — the SQL itself changes (above).

### End-to-end runs (all CSVs in `measure/job_result/`)

Topdown, oracle = perfect prediction (previous repeat's sub-SQL/IR):

| configuration | compile | middleware | execute | total | vs baseline |
|---|---|---|---|---|---|
| tpde baseline | 0.525 | 0.066 | 4.040 | 4.631 | |
| tpde baseline re-run | 0.540 | 0.069 | 4.027 | 4.636 | +0.1% |
| tpde Phase-B-only spec (commit 7069036 behaviour) | 0.584 | 0.224 | 4.160 | 4.969 | +7.3% |
| tpde oracle, CDC placeholder, SQL bg path (node-based mechanism) | 0.295 | 0.195 | 5.154 | 5.644 | +21.9% |
| tpde oracle, metadata placeholder, SQL bg path | 0.310 | 0.198 | 4.274 | 4.782 | +3.3% |
| tpde oracle, metadata placeholder, same-IR bg path | 0.308 | 0.207 | 4.168 | 4.683 | +1.1% |
| tpde same, bg launched serially after execute (no overlap) | 0.454 | 0.258 | 4.109 | 4.821 | +4.1% |
| tpde same-IR, every speculation forced to MISS | 0.425 | 0.216 | 4.220 | 4.861 | +5.0% |
| tpde oracle, metadata placeholder, 11 executor threads | 0.314 | 0.196 | 4.323 | 4.832 | +3.6% (vs 4.664 at 11 threads) |
| fastisel baseline | 0.910 | 0.070 | 4.912 | 5.892 | |
| fastisel Phase-B-only | 1.031 | 0.236 | 5.010 | 6.278 | +6.6% |
| fastisel oracle, CDC placeholder | 0.549 | 0.201 | 6.032 | 6.783 | +15.1% |
| fastisel oracle, metadata placeholder | 0.601 | 0.201 | 5.198 | 5.999 | +1.8% |
| fastisel oracle, metadata placeholder, same-IR | 0.593 | 0.220 | 5.114 | 5.927 | +0.6% |
| llvm baseline | 5.418 | 0.066 | 3.417 | 8.901 | |
| llvm Phase-B-only | 5.775 | 0.240 | 3.584 | 9.599 | +7.8% |
| llvm oracle, CDC placeholder | 4.874 | 0.204 | 4.645 | 9.723 | +9.2% |
| llvm oracle, metadata placeholder | 5.181 | 0.197 | 3.658 | 9.035 | +1.5% |
| llvm oracle, metadata placeholder, same-IR | 4.947 | 0.212 | 3.642 | 8.802 | **−1.1%** |
| llvm same, serial launch (no overlap) | 5.495 | 0.269 | 3.492 | 9.256 | +4.0% |
| llvm same-IR, every speculation forced to MISS | 5.527 | 0.226 | 3.592 | 9.345 | +5.0% |

Node-based (same cache/backends, `measure_breakdown_time_aqp.sh ... node-based ...`):

| configuration | compile | middleware | execute | total | vs baseline |
|---|---|---|---|---|---|
| node-based tpde baseline | 0.414 | 0.374 | 4.339 | 5.127 | |
| node-based tpde spec recompile | 0.412 | 0.376 | 4.478 | 5.266 | +2.7% |
| node-based llvm baseline | 4.799 | 0.372 | 3.730 | 8.901 | |
| node-based llvm spec recompile | 4.860 | 0.395 | 3.910 | 9.165 | +3.0% |

Node-based spec counters per 15-repeat run: hits 315, card_misses 153,
compensate_fast 5520 (learned misses, no launch). Ideal overlap ceiling
for node-based: tpde 0.375 s (90% of its compile, 7.3% of total), llvm
1.347 s (28% of compile, 15% of total).

All oracle/forced-miss runs produce query results byte-identical to the
baseline (md5 of all result rows). Wall-clock per 15-repeat run: baseline
tpde 78 s, same-IR oracle tpde 77 s, serial 80 s, forced-miss 80 s; llvm
baseline 141 s, same-IR oracle 139 s, serial 146 s, forced-miss 147 s.

Where the same-IR oracle's time goes (tpde, per pass): compile −0.228 s;
execute +0.118 s; `extract_next_sub-IR` +0.133 s. The extract increase is
mostly an accounting shift (the DP runs inside the untimed `IsComplete()`
in the baseline and inside the timed `PrecomputeNextExtraction` in spec
mode), which is why wall clock looks ~1% better than the CSV total. The
execute increase has two parts: the SQL bg path compiles a *different*
plan than the inline IR plan constructor (build side from DuckDB's
estimates), costing +0.21 s on hits concentrated in 11d/4c/19c/12c/4a for
every backend; the same-IR path removes that and leaves +0.12 s, of which
~0.07 s is the bg thread sharing a physical core with an executor thread
(the serial variant shows +0.045 s).

With the CDC placeholder (what node-based uses), every speculated temp is
converted from the QjitTable to a `ColumnDataCollection`
(`QjitTableToCollection`, charged to execute) and `Combine()`d into the
placeholder: +1.1 s execute for tpde. Node-based spec-jit pays this today.

### Is the bg compile the same work as the inline compile?

Structurally yes once the bg task compiles the same IR through the same
plan constructor: `ResetModules` + `Connection` (0.06 ms) +
`TryBuildBinaryPlanFromIR` + `PrepareFromPlan` + analyze + `BuildExecutionSteps`
+ `CompileQuerySteps`, all single-threaded. Measured on identical sub-query
sets (same-IR oracle, 4788 compiles):

| | tpde mean / median | llvm mean / median |
|---|---|---|
| bg compile launched serially (no execution running) | 0.93 / 0.80 ms | 10.6 / 11.0 ms |
| bg compile overlapped with Execute(i) | 1.27 / 1.00 ms | 11.5 / 11.9 ms |
| inline fast path (baseline, for reference) | 0.96 / 0.39 ms | 5.9 / 0.38 ms |

The CPU is an Intel E-2236: 6 physical cores, 12 hardware threads, and
the qjit executor uses 12 workers, so the bg thread always shares a
physical core with an executor thread. Overlapped compiles are 37% slower
(tpde) and the sibling executor thread loses throughput. The llvm inline
mean is lower than the serial bg mean because 23% of sub-queries take the
non-constructor (SQL) path inline and are counted with their cheaper
fallback time.

### Cost of a wrong guess

Forced-miss run (same-IR oracle with a deliberately non-matching SQL, no
Phase B relaunch, inline compile on every iteration): tpde +5.0%, llvm
+5.0%. The wasted bg compile even warmed the template cache for the
inline compile in this experiment (identical shape), so a truly wrong
shape costs more: ≈ +7%. Per speculation (tpde, 342 per pass): a hit saves
0.67 ms compile but costs 0.35 ms execute + 0.39 ms bookkeeping; a miss
costs 0.46 ms execute + 0.41 ms bookkeeping. Expected total vs accuracy p
(linear interpolation of the oracle and forced-miss runs): tpde is slower
for every p (even p=1 is +1.1%); llvm breaks even at p≈0.82 and gains 1.1%
at p=1. The measured one-step accuracy with the DP estimate is 0.55.

## From-scratch feasibility assessment

What spec-jit can do: hide compile(i+1) behind execute(i). For topdown the
only unknown at Prepare(i) is temp(i)'s cardinality, which enters (a) the
DP's group choice for i+1 and (b) the physical plan of i+1.

1. **Ceiling.** With the template cache the inline compile is 0.8 ms per
   sub-query (tpde). Perfect, free overlap would save 8% (tpde) to 12%
   (llvm) of total time; the same-IR oracle actually realizes 0.23 s
   (tpde) and 0.51 s (llvm) of compile time, i.e. 5-6% of total, before
   its own costs.
2. **Costs that remain even with perfect prediction.** SMT contention
   (+0.07 s), bg plan parity (fixed by same-IR), placeholder data path
   (fixed by metadata placeholder), Phase A/B bookkeeping (~0.05 s real).
   Net: −1.1% (llvm) to +1.1% (tpde).
3. **Prediction.** 55% one-step accuracy with the DP estimate; 21.8% of
   iterations are base-only groups that are always predictable. A
   cardinality-independent predictor would have to emit two disjoint
   groups from one DP run (compile group B while executing group A), which
   gives up one round of cardinality feedback and changes 45% of split
   decisions; its execution-side effect is unmeasured.
4. **Verdict.** With the best achievable implementation the gain is
   bounded by ~1% (llvm) and is negative for tpde/fastisel even at 100%
   prediction; at the realistic 55% it is a 2-4% loss for all backends.
   Spec-jit for topdown should not be pursued further in this form.

What would change the verdict (not implemented):
* A free physical core for the bg compile (fewer executor workers or a
  bigger machine) removes the 37% compile slowdown and the sibling
  slowdown; `--query-jit-threads=11` alone costs +0.03 s in the baseline.
* Keeping compiled objects resident instead of `ResetModules()` per
  sub-query makes a template-cache hit a symbol lookup; this attacks the
  same 0.4 ms hit cost without speculation and without the SMT cost.
* Final-query speculation: the final IR is known at Phase B of the last
  iteration once the remaining graph has two relations; hideable 0.13 s
  (tpde) / 0.34 s (llvm) with the same placeholder/parity caveats.
* Node-based: see the dedicated section below (oracle −2.5% tpde /
  −2.3% llvm; current code +3-4%).

## Node-based: measurements and verdict

Baseline: `measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template off <mode>`.
Node-based sub-queries: 511 per pass; sub execute 6.2 ms avg vs compile
0.48 ms (tpde) / 6.8 ms (llvm); ideal overlap ceiling 0.375 s (tpde, 7.3%
of total) / 1.35 s (llvm, 15%). Note the node-based inline path compiles
from the engine plan handed over by QuerySplit (`TakePlan`), not from the
IR plan constructor, so the SQL-reparse bg path is plan-equivalent here
and the same-IR variant is not needed.

| configuration | compile | middleware | execute | total | vs baseline |
|---|---|---|---|---|---|
| tpde baseline (two runs) | 0.414 / 0.408 | 0.374 / 0.370 | 4.339 / 4.271 | 5.127 / 5.049 | |
| tpde spec recompile (current code) | 0.412 | 0.376 | 4.478 | 5.266 | +3.5% |
| tpde oracle, CDC placeholder (current mechanism) | 0.238 | 0.374 | 5.276 | 5.887 | +15.7% |
| tpde oracle, metadata placeholder (two runs) | 0.243 / 0.236 | 0.368 / 0.367 | 4.373 / 4.331 | 4.984 / 4.934 | **−2.5%** |
| tpde oracle, metadata placeholder, Prepare-estimate card, no guard | 0.244 | 0.376 | 4.922 | 5.541 | +8.9% |
| tpde oracle, every speculation forced to MISS | 0.376 | 0.372 | 4.412 | 5.159 | +1.4% |
| fastisel baseline | 0.766 | 0.366 | 4.969 | 6.102 | |
| fastisel spec recompile (current code) | 0.829 | 0.388 | 5.137 | 6.354 | +4.1% |
| fastisel oracle, metadata placeholder (two runs) | 0.468 / 0.451 | 0.382 / 0.381 | 5.366 / 5.273 | 6.216 / 6.105 | +1.0% |
| llvm baseline (two runs) | 4.799 / 4.785 | 0.372 / 0.373 | 3.730 / 3.723 | 8.901 / 8.881 | |
| llvm spec recompile (current code) | 4.860 | 0.395 | 3.910 | 9.165 | +3.1% |
| llvm oracle, metadata placeholder (two runs) | 4.504 / 4.468 | 0.373 / 0.369 | 3.822 / 3.833 | 8.699 / 8.670 | **−2.3%** |
| llvm oracle, Prepare-estimate card, no guard | 4.568 | 0.375 | 4.398 | 9.342 | +5.1% |
| llvm oracle, every speculation forced to MISS | 4.839 | 0.383 | 3.982 | 9.204 | +3.5% |

All runs: results byte-identical to the baseline. Oracle counters per
15-repeat run: 5142 hits, 416 compensate (first repeats before the record
converges), 17% (tpde) / 60% (llvm) of hits not ready in time.

Why the current node-based spec-jit hits only 5% (`measure/classify_peek_mismatch.py`
on a `--debug --repeat=1` run, 503 launches):

| Phase B outcome | count | share |
|---|---|---|
| peek SQL == real SQL ("kept") | 174 | 35% |
| of which rejected by the 2x card guard (`CARD_MISS`) | 153 | 30% |
| of which HIT | 21 | 4% |
| peek predicts a different table set (FROM differs) | 155 | 31% |
| same tables, different SELECT list | 43 | 9% |
| same tables, different SELECT and WHERE | 17 | 3% |
| no outcome (next is final / cross product / kernel) | 114 | 23% |

* FROM mismatches: DuckDB's QuerySplit re-plans the remaining sub-queries
  after each merge (`UpdateSubqueriesIndex`), so the front of the queue
  seen by `PeekNextSubquery` is not what runs next in 31% of cases (e.g.
  10a iteration 3: peek = {company_type, role_type, temp2, cast_info}, real =
  {role_type, cast_info}). Fixing this means running the same re-split
  logic speculatively on a copy of the queue.
* SELECT mismatches: the peek projects the temp's join key where the real
  sub-query projects the base table's key (10a iteration 2). Fixable in
  `PeekNextSubquery`.
* Card guard: the Prepare estimate for temp(i) is within 2x only 16% of
  the time, and the estcard runs show that compiling on it is worse than
  not speculating (+8.9% / +5.1%). Keep the guard. To turn these rejects
  into hits the bg compile has to be cardinality-independent: compile both
  join orientations (two template-cache shapes) and pick at HIT time
  when temp(i)'s actual size is known; tpde's 1.4 ms compile fits twice in
  the 6 ms average execute window, llvm's 10 ms does not.
* Node-based temp column names differ between an inline-compiled temp
  (`col0..`) and a spec-compiled temp (SQL aliases), so later sub-SQL text
  depends on earlier HIT/MISS history. Harmless for correctness, but it
  breaks any SQL-text-keyed cross-repeat learning (`g_spec_miss_history`
  and the oracle needed temp-alias re-indexing and no miss learning).

Expected total with hit rate p (interpolating oracle and forced-miss):
tpde `−2.5%·p + 1.4%·(1−p)` → break-even at p≈0.36; llvm
`−2.3%·p + 3.5%·(1−p)` → break-even at p≈0.60. Today p=0.04. Fixing the
SELECT list, the placeholder copy and keeping the guard gives
p≈(21+43·0.12)/503≈0.05 — still a loss; fixing SELECT + the card problem
(orientation variants) gives p≈0.43: tpde ≈ −0.3%, llvm ≈ +1.0%; fixing
FROM as well gives the oracle numbers.

### Node-based: is the next sub-query predictable? (verified 2026-10-12, corrects an earlier claim)

An earlier version of this section claimed the node-based split sequence
is independent of temp cardinalities. That test was invalid: node-based
never sets `extraction->estimated_rows`, so `--no-update-temp-card` fell
back to the actual count on both sides. Re-tested with a real knob
(`AQP_NB_FAKE_CARD=1`: `MergeDataChunk` gets the engine's Prepare estimate
of the just-executed sub-query, `DuckDBAdapter::GetLastPrepareEstCard()`):

* Prepare estimate quality for node-based temps: within 2x 6%, within 5x
  29%, **median error 46x** (330 sub-queries).
* With that estimate the re-split (`MergeSubquery` + `MiddleOptimize` =
  DuckDB's JoinOrderOptimizer + `Split`) produces a different next sub-SQL
  in **93%** of cases (22/305 identical; 29/113 queries fully identical,
  mostly single-sub-query ones). The node-based next sub-query is at least
  as cardinality-dependent as topdown's.
* The current queue-front peek nevertheless matches 35% of the time
  (+9% fixable SELECT-list differences): those are the iterations where the
  re-optimization happens to keep the queue order.

Making the sequence predictable by not re-splitting (`AQP_NB_NO_RESPLIT=1`:
consume the first split's queue as-is; results verified identical):

| configuration | compile | middleware | execute | total | vs baseline |
|---|---|---|---|---|---|
| tpde no-re-split, no spec | 1.148 | 0.115 | 5.914 | 7.177 | **+40%** (603 sub-queries vs 511) |
| tpde no-re-split + spec (peek now 100% exact, guard kept) | 1.060 | 0.127 | 6.097 | 7.285 | +1.5% vs no-re-split |
| llvm no-re-split, no spec | 5.360 | 0.117 | 5.251 | 10.728 | **+21%** |

So the cardinality feedback is worth 20-40% of execution time and cannot
be traded for predictability. With an exact peek the 2x card guard still
rejects 92% of matches (448 of 488 per pass), and the estcard runs show
the guard is right (+8.9% / +5.1% execution when planning on the estimate).

### Node-based verdict

Everything needed for a win was measured:

| lever | measured | consequence |
|---|---|---|
| perfect prediction + exact card (oracle) | −2.5% tpde, −2.3% llvm | the ceiling |
| realistic prediction (queue-front peek, SELECT fixed) | 44% right SQL | below llvm break-even (60%), near tpde break-even (36%) |
| card guard | rejects 88-92% of right-SQL matches; bypassing it costs +8.9% / +5.1% | the matched 44% need a cardinality-independent compile (both orientations of the temp join) to become hits |
| miss cost | +0.6-2.2% tpde, +3.4% llvm at 100% miss | with h=0.44: tpde ≈ −0.3% (noise), llvm ≈ +1% |
| CDC placeholder | +15.7% | must be replaced by the metadata-only placeholder (done) |
| no re-split for predictability | +40% / +21% | not an option |

Conclusion: with single-run semantics (no knowledge from previous
executions of the same query) node-based spec-jit cannot be made
meaningfully faster than the baseline: best case ≈ −0.3% for tpde (within
the ~1.5% run-to-run noise) and a loss for llvm. The only path to the
oracle's −2.5% / −2.3% is cross-run memoization of the split sequence and
temp cardinalities per query text (which is what the oracle does, and what
`g_spec_miss_history` already does in a negative form). That is a
plan-cache for repeated queries, not speculation, and it violates the
intent of the `single-run-*` cache modes that clear state between repeats.

If a cross-run memo is acceptable for the target workload, the
implementation is small and already prototyped: record per query
`{raw_sql, range-pred sql, actual card}` per iteration (done), launch at
Prepare(i) with the metadata placeholder (done), re-index node-based temp
aliases (done), disable miss learning (done), and add a staleness check
(fall back to the inline path on SQL mismatch, already the behaviour).

## Single-run spec-jit: every remaining lever, measured (2026-10-12)

Question: under single-run semantics (no knowledge from earlier executions
of the same query), is there any way for spec-jit to beat the no-spec
baseline for query-jit with llvm/tpde/fastisel on node-based or topdown?

### Sensitivity of the next sub-query to a bounded cardinality error

`AQP_SPEC_FAKE_CARD_RATIO=r` feeds actual×r as temp(i)'s cardinality to the
split machinery and the engine (both strategies); identical next sub-SQL at
iteration 2 (one error, no compounding) and over all later sub-queries
(error on every temp, compounding):

| error | topdown iter-2 | topdown all later | node-based iter-2 | node-based all later |
|---|---|---|---|---|
| 1.5x | 92% | 68% | 80% | 70% |
| 2x | 90% | 60% | 64% | 52% |
| 3x | 81% | 48% | 46% | 35% |
| DP estimate (median 5.5x, topdown) / Prepare estimate (46x, node-based) | 55% | 30% | — | 7% |

Topdown's DP decision is robust to a ≤2x error; node-based's re-split is
not. Base-only next groups (no temp in the group, 66 of 342 topdown
iterations) are 95% stable under 2-3x errors but only 27% under the DP
estimate's errors; their total hideable compile is 0.043 s (tpde) / 0.149 s
(llvm).

### A cardinality source nobody used: the sub-query's own progress

The qjit executor is ours: while the workers run, the main thread sits in
`QjitWorkerPool::ParallelFor`. A progress callback (`SetProgressCallback`,
fires once per parallel-for when the *completed* row fraction passes a
threshold; `QjitExecutor::EnableProgress`, trace `AQP_PROG_TRACE=<p>`)
gives an extrapolated output cardinality `rows_so_far / fraction_done`
before the sub-query finishes:

| threshold 0.5, topdown tpde, 892 compiled sub-queries (2 passes) | |
|---|---|
| snapshot taken (on the last parallel-for = result step) | 892 |
| snapshot with 0 rows so far (final > 0, unusable) | 205 (23%) |
| extrapolation error, non-zero snapshots | within 1.5x 80%, within 2x 90%, median 1.09x, p90 2.1x |
| actual completed fraction at snapshot (50 µs poll overshoot) | median 0.75 |
| remaining execution after the snapshot | median 0.30 ms; ≥1.5 ms (room for one tpde compile) in 122/892 = 14%; sum of those windows 239 ms per pass |

So a progress-triggered speculation ("early commit": run the real
UpdateRemainingIR + SplitIR(i+1) with the extrapolated cardinality on the
idle main thread, launch the same-IR bg compile, patch the exact
cardinality in afterwards) would predict topdown's next sub-query ~90%
right where it fires, but it can only fire where the remaining window
holds the DP (0.4 ms) plus the compile (1.3 ms tpde): about 14% of
iterations, worth ≤0.24 s of window per pass and realistically 0.05-0.1 s
(1-2%) of compile hidden for tpde. For llvm (8.6 ms compile) the window
never fits. A spinning poll (snapshot at 0.5 instead of 0.75) roughly
doubles the eligible iterations: ceiling ≈ 2-3% for tpde, still ~0 for
llvm. node-based: the re-split is too sensitive (64% at 2x) and the same
window limit applies.

### Verdict

| mechanism | prediction | hits needed | measured / bounded gain |
|---|---|---|---|
| node-based peek + guard (current) | 35-44% SQL, 12% after card guard | 36% tpde / 60% llvm | −3 to −4% (loss) |
| node-based, re-split removed for predictability | 100% | | +40% / +21% (loss) |
| node-based oracle (cross-run memo, not single-run) | 100% | | −2.5% tpde, −2.3% llvm |
| topdown DP-estimate speculation | 55% | 100%+ (hits are net-neutral for tpde) | loss |
| topdown same-IR oracle (cross-run) | 100% | | +1.1% tpde CSV (−1% wall), −1.1% llvm |
| topdown, speculate only base-only next groups | 27% with DP estimate | | ≤0.9% tpde, ≤1.7% llvm before misses |
| topdown, progress-triggered early commit (new) | ~90% where it fires (14-30% of iterations) | | 1-3% tpde, ~0 llvm; not implemented |
| planning on the engine estimate without a guard | | | +5 to +9% execution (loss) |

Under single-run semantics the best defensible design (progress-triggered
early commit on topdown + tpde/fastisel) has a ceiling of about 2-3% and a
realistic yield of 1-2%; nothing reaches a measurable gain for llvm,
because an 8.6 ms average compile cannot hide behind a 4.3 ms average
execute no matter how good the prediction is. Modifying DuckDB does not
change this: the binding constraints are the compile/execute ratio, the
cardinality dependence of the next split (intrinsic to both strategies),
and the SMT contention of the bg thread; none of them live in DuckDB code.

### Ideas discussed 2026-10-13 and their measured bounds

* **Work for the main thread while a spec compile is not ready / incremental
  compilation.** Before Execute(i+1) the main thread only has ~0.1 ms of work
  (SQL gen, range preds, source resolution); the executor workers do nothing
  but execute. The compile pipeline's cardinality-independent part is the
  per-shape object code, which the template cache already shares across
  queries (61% hits). The cardinality-dependent parts are (a) which group is
  next and (b) the temp join's build/probe orientation. Pre-compiling the
  DP's candidate SET does not help with (a): with the DP estimate the actual
  choice is in the estimated candidate list only 43% of the time (candidate
  sets identical 52%); at a 2x-accurate estimate 72-78% (`AQP_TD_DUMP_CANDS`).
  So "confident part first" reduces to two-orientation compiles + resident
  objects, worth ~0.1-0.3 ms per hit.
* **Commit to the predicted next sub-query (no correctness check, any valid
  decomposition is correct).** Results stay correct, planning quality does
  not: topdown planning on the DP estimate with no actual-cardinality
  feedback (`--no-update-temp-card`) runs **7.504 s vs 4.631 s (+62%)**
  for tpde (node-based analogue: +40% / +21%). The one-step variant (feed the
  actual card back for later decisions) diverges from the fed-back decision
  in 45-52% of iterations; the total hideable compile is 0.26 s per pass, so
  any planning loss above ~6% erases it. Not viable. A 6+6 thread ping-pong
  does not help compilation (single-threaded) and halves execution
  parallelism; the only "pong" work that exists is an independent base-only
  sub-query, which is a different optimization (below).
* **Other levers (not spec-jit):** intra-query parallel execution of
  disjoint base-only groups (only 4% of topdown iterations have two
  disjoint base-only candidate pairs, so little opportunity on JOB); resident compiled objects / skipping PrepareFromPlan on template
  hits (~1%); per-sub-query backend mixing via `--tune-config` (tpde for
  sub-queries, llvm where execution gains pay for compile; llvm's 5.4 s
  compile is 61% of its 8.9 s total); final-query compile overlap where the
  remaining graph has two relations (<1%).

## Implementation round (2026-10-13): what was built and measured

Plan: `/home/pei/.claude/plans/rustling-bouncing-peach.md`. Baselines re-established after the
accounting fix (step 0: the topdown DP in `has_work()` is now timed into the
extract column in both modes): topdown tpde **4.793 s**, fastisel **6.077 s**,
llvm **9.114 s** (CSV totals; wall clock unchanged).

| step | change (now default in spec mode) | tpde | fastisel | llvm |
|---|---|---|---|---|
| 1 | metadata-only placeholder for query-jit speculation, same-IR bg compile, topdown Phase B relaunch removed, source-rows window rule (off by default) | topdown 4.808 (+0.3%, no launches) ; node-based 5.163 (+0.7% vs 5.127) | — | node-based 9.182 (+3.1%, unchanged: 315 hits / 5520 misses) |
| 2 | progress-triggered early commit (topdown): at completed fraction ≥0.5 the main thread extrapolates temp(i)'s cardinality, runs the real UpdateRemainingIR + SplitIR(i+1), launches the same-IR bg compile; actual cardinality patched after execution; 3x card guard | 4.872 / 4.855 (+1.5%) | 6.112 (+0.6%) | 9.217 (+1.1%) |
| 2, window factor 2 | fire only when remaining ≥ 2·(DP+compile) | 4.852 (+1.2%) | | 9.271 (+1.7%) |
| 2, window factor 0.5 | allow partial overlap | 4.849 (+1.2%) | | |
| 2, factor 3 + threshold 0.3 | earlier, stricter | 4.910 (+2.4%) | | |

Counters per pass (tpde, default knobs): fired 90, hits 56, card misses
0.4, not-ready 19, window-skipped 278; main-thread early-commit work
(`ec_dp_ms`) 0.59 ms per fire. Why it loses: a hit saves ~0.4 ms of
compile (0.023 s / 56) while each fire costs ~1 ms of execution (0.09 s /
90): the DP + EXPLAIN validation runs on the main thread inside the
execution window and the bg compile shares a hyperthread with a worker.
Results byte-identical to the baseline in every run. Step 3 (final-query
speculation) was gated on step 2 and was not built.

Gate outcome: speculation with query-jit under single-run semantics does
not beat the no-spec baseline on any backend or split; the best variants
are within +0.3% (topdown, no launches) and +0.6% (fastisel early commit).

| step | change | tpde | llvm |
|---|---|---|---|
| 5 | resident compiled modules (`AQP_JIT_RESIDENT=1`): no per-sub-query `ResetModules` in `TryCompileQueryJit` / `ResetQueryState`; template-cache hits served as live symbols; modules dropped with the object cache at the repeat boundary | 4.758 (**−0.7%**) | 9.077 (**−0.4%**) |

Step 5 is a compile-cost reduction independent of speculation (applies to
the no-spec baseline); results byte-identical. Step 4b (two-orientation
compile for node-based) was not built: 4a (step-1 infrastructure) left
node-based at +0.7% (tpde) / +3.1% (llvm) with the predicted best case
≈ −0.3%, inside the 1.5% noise.

Current default behaviour of `--spec-jit=recompile` with query-jit after
this round: node-based = peek + metadata placeholder + 2x guard (no Phase
B relaunch); topdown = progress-triggered early commit (fires when the
remaining execution can hold DP + compile; `AQP_EC_WINDOW_FACTOR`,
`AQP_EC_MIN_PROGRESS`, `AQP_EC_DP_MS` tune it). Both are measured losses of
0.3-1.5%; without `--spec-jit` nothing changes except the step-0 timing
column (the DP time now appears in `extract_next_sub-IR` in both modes).

## Re-confirmation with the measurement script (2026-10-13, all 18 runs)

`bash measure_breakdown_time_aqp.sh job duckdb {topdown,node-based} query none on on on all single-run-template {off,recompile} {tpde,fastisel,llvm}`,
current build (steps 0-4 default; step 5 = same `off` command with
`AQP_JIT_RESIDENT=1`, CSVs renamed `..._resident_<backend>_...`), parsed with
`print_breakdown.py`. Every run's result rows are byte-identical (md5
e3c2677e). Previous canonical CSVs are in `job_result/backup_20261013/`.

| configuration | off (baseline) | recompile (spec-jit) | Δ | resident (step 5, no spec) | Δ |
|---|---|---|---|---|---|
| topdown tpde | 4.887 | 4.862 | −0.5% | 4.768 | −2.4% |
| topdown fastisel | 6.039 | 6.100 | +1.0% | 6.005 | −0.6% |
| topdown llvm | 9.153 | 9.199 | +0.5% | 9.074 | −0.9% |
| node-based tpde | 5.073 | 5.160 | +1.7% | 5.048 | −0.5% |
| node-based fastisel | 6.141 | 6.331 | +3.1% | 6.085 | −0.9% |
| node-based llvm | 8.913 | 9.171 | +2.9% | 8.837 | −0.9% |

Spec counters per 15-repeat run: topdown tpde fired 1338 / hits 832 /
card misses 7 / not ready 303; fastisel 1557 / 1067 / 7 / 515; llvm
705 / 452 / 4 / 261; node-based (all backends) hits 315 / card misses 153 /
compensate 5520.

Noise: the topdown tpde baseline measured 4.793 s (direct run, 00:25) and
4.887 s (script, 09:22) on the same binary and flags, i.e. ~2% between
sessions; differences below ~2% in the table above are not significant
without repeated runs. Within that noise: spec-jit is at best neutral
(topdown tpde −0.5%) and otherwise a 0.5-3.1% loss; step 5 is consistently
on the gain side (−0.5% to −2.4%, every configuration) but it is a
compile-cost reduction that applies with or without speculation.

## Round 2 (2026-10-13): steps 1+2+4a + spec-only resident modules + race/cancel

Changes on top of round 1, all active only with `--spec-jit`: resident
compiled modules are enabled by the splitter when spec-jit is on
(`DuckDBAdapter::SetJitResident`, so `off` runs are untouched); a confirmed
match whose bg compile is not finished races an inline-equivalent compile on
a second compile thread (`race_pool_`/`race_compiler_`, first finisher wins,
loser retired as a zombie; `WaitZombiesUsing` protects compiler slots); a
wrong speculation sets `SpeculativeCompilation::cancel`, checked by the bg
task before codegen. Measured with the script, 12 runs, results identical:

| configuration | off | recompile | Δ | race won / lost | hits / not-ready per 15 runs |
|---|---|---|---|---|---|
| topdown tpde | 4.807 | 4.809 | 0.0% | 1 / 282 | 832 / 284 |
| topdown fastisel | 6.046 | 6.081 | +0.6% | 22 / 519 | 1104 / 541 |
| topdown llvm | 9.121 | 9.215 | +1.0% | 21 / 228 | 439 / 249 |
| node-based tpde | 5.074 | 5.180 | +2.1% | 2 / 146 | 315 / 148 |
| node-based fastisel | 6.185 | 6.387 | +3.3% | 9 / 157 | 315 / 166 |
| node-based llvm | 8.952 | 9.132 | +2.0% | 20 / 165 | 315 / 185 |

The race wins 0.4-9% of the not-ready cases (the bg compile has the head
start and compiles the same shape), so it recovers almost none of the wait;
cancellation almost never fires before codegen. The spec-only resident
modules lower the spec compile column (topdown tpde 0.535 → 0.484 s) but the
execute column rises by the same amount (+0.07 s, the early-commit cost), so
the total is unchanged. Compared with round 1 every configuration moved by
less than the ~2% session noise.

**Final verdict (both rounds, 42 script runs):** with query-jit and
single-run-template, spec-jit never beats no-spec-jit on this machine:
topdown tpde is neutral (0.0% / −0.5% in two sessions), every other
configuration is 0.6-3.3% slower. The best spec-jit code path is steps
1 + 2 + 4a (+ the spec-only resident modules); it is kept in the tree as the
`--spec-jit` behaviour but it is not a win.

## Final measurement of the cleaned tree (2026-10-13, 12 script runs)

`bash measure_breakdown_time_aqp.sh job duckdb {topdown,node-based} query none on on on all single-run-template {off,recompile} {tpde,fastisel,llvm}`
on the cleaned build (metadata placeholder, topdown early commit, node-based
peek, spec-only resident modules; every experiment knob removed). Results
byte-identical in all 12 runs (md5 e3c2677e).

| configuration | off | recompile | Δ (CSV) | hits / fired / not-ready per 15 runs |
|---|---|---|---|---|
| topdown tpde | 4.654 | 4.784 | +2.8% | 835 / 1362 / 311 |
| topdown fastisel | 5.863 | 6.061 | +3.4% | 1107 / 1607 / 544 |
| topdown llvm | 9.019 | 9.139 | +1.3% | 455 / 696 / 257 |
| node-based tpde | 5.079 | 5.129 | +1.0% | 315 / – / 151 |
| node-based fastisel | 6.142 | 6.342 | +3.3% | 315 / – / 148 |
| node-based llvm | 8.884 | 9.154 | +3.0% | 315 / – / 187 |

Accounting caveat (the baseline DP-timing change was removed with the
cleanup): in the `off` runs the topdown DP runs inside the untimed
`IsComplete()` loop predicate, in `recompile` runs the same DP runs inside
the timed `PrecomputeNextExtraction`, so the spec CSV carries ~0.12 s per
pass (middleware 0.067 → 0.184 s) that the baseline also spends but does
not report. Wall-clock equivalents for topdown are therefore roughly
+0.3% (tpde), +1.4% (fastisel), 0% (llvm); node-based has no such shift.
Either way spec-jit does not beat no-spec-jit on any configuration.

## Launch guards (2026-10-13 afternoon): cache-miss guard and node-based first-iteration skip

Two guards added to the spec path (both strategies), plus timing parity per
strategy (node-based's re-split is timed in both modes because the no-spec
path times it in the extract section; topdown's DP is untimed in both
because the no-spec path runs it in the untimed loop predicate).

* **Cache-miss guard.** The next sub-query's template-cache key is
  predicted from its IR before anything is compiled: clone the IR, annotate
  build sides by estimated cardinality (`AnnotateUnannotatedJoinsByCard`),
  `BuildExecutionSteps`, `IrToLlvmCompiler::QueryStepsCacheKey` (same key
  construction as `CompileQuerySteps`, all three cache modes),
  `HasCachedObject`. If the object is already cached the launch is skipped
  (an inline hit costs ~0.4 ms). `DuckDBAdapter::PredictQjitCacheKey`.
* **Node-based first-iteration skip.** No launch at iteration 1 (the queue
  peek is right 13% of the time before the first merge; 58-74% later).
* Counters in the summary: skipped_cached, skipped_first, key_match /
  key_mismatch (predicted key vs the key the bg compile really used, on HITs).

Script measurement (recompile vs the `off` baselines of the same session):

| configuration | off | recompile | Δ | hits / launches skipped (cached, first) per pass | predicted key correct on hits |
|---|---|---|---|---|---|
| topdown tpde | 4.654 | 4.736 | +1.8% | 28 / 25, – | 71% (283/400) |
| topdown fastisel | 5.863 | 5.941 | +1.3% | 38 / 30, – | 55% |
| topdown llvm | 9.019 | 9.032 | +0.1% | 13 / 13, – | 67% |
| node-based tpde | 5.079 | 5.190 | +2.2% | 14 / 71, 113 | 29% (60/210) |
| node-based fastisel | 6.142 | 6.294 | +2.5% | 14 / 72, 113 | 29% |
| node-based llvm | 8.884 | 9.099 | +2.4% | 14 / 72, 113 | 29% |

Results byte-identical in all runs. The key prediction is reliable enough
for topdown (the bg compile uses the same plan constructor as the
prediction) but not for node-based (29%): there the bg compile re-parses
the SQL and DuckDB's optimizer produces a plan whose step structure differs
from the peek IR's, so the predicted key rarely equals the real one and the
guard mostly skips the wrong launches. Net effect of both guards: topdown
unchanged within noise, node-based hits fell from 21 to 14 per pass and
the loss stayed at +2.2-2.5%.

A note on an accounting pitfall caught in this round: making the spec
path's `PrecomputeNextExtraction` untimed for both strategies made
node-based `recompile` look 4.4% faster (middleware 0.369 → 0.072 s) by
hiding its re-split from the CSV; the per-strategy parity above restores
the honest comparison.

## What is in the tree now (after the 2026-10-13 cleanup)

All experiment code (cross-repeat oracle and its variants, fake-cardinality
knobs, candidate dumps, stage traces, the no-re-split mode, the inline/bg
compile race and cancellation, the source-rows window rule, the baseline
DP accounting change) and the analysis scripts (`analyze_spec_overlap.py`,
`parse_spec_trace.py`, `compare_sql_sequences.py`,
`classify_peek_mismatch.py`) have been removed. The numbers they produced
are recorded above. What remains, all active only with `--spec-jit`:

* Query-jit speculation registers a metadata-only placeholder for the
  speculated temp (`DuckDBAdapter::RegisterQjitPlaceholderTemp`): the real
  qjit result is stored zero-copy; the placeholder table is retired, not
  destroyed, because a discarded bg prepared statement may point at it. The
  placeholder uses the same type mapping as the real qjit store
  (`QjitDtypeForLogicalType`) and reports a temp the store would reject, in
  which case no bg compile is launched.
* Topdown: progress-triggered early commit. The qjit executor reports the
  completed fraction, rows so far and the running time of the current
  parallel step to the main thread (`QjitWorkerPool::SetProgressCallback`,
  `QjitExecutor::EnableProgress`, `DuckDBAdapter::SetProgressHook`,
  `IRQuerySplitter::OnExecuteProgress`); the pool stops polling once the
  callback has fired. If the extrapolated cardinality is usable and the
  step's remaining time (`elapsed * (1-p)/p`, a lower bound on the
  sub-query's remaining time) can hold the planning plus the running average
  of the bg compile time, the real `UpdateRemainingIR` + `SplitIR(i+1)` run
  on the (idle) main thread with that estimate and the next sub-query is
  compiled in the background through the inline plan constructor on the
  spec connection (`TryBuildBinaryPlanFromIR(ctx)` +
  `SpeculativeQueryJitCompileFromPlan`), so a HIT executes the same code the
  inline path would have compiled. After execution the actual cardinality
  replaces the estimate everywhere the non-speculative path would have had
  it: the DP's table cardinalities and the EXPLAIN cache entries that
  mention the temp (`TopDownSplitter::PatchTempCardinality`), the temp leaf
  of the remaining IR, and the temp leaf of the next sub-IR once its
  speculation is resolved. The card guard for these launches is 3x. The bg
  compile works on the next iteration's own IR, so a rejected speculation
  waits for it before the inline path annotates that IR, and an
  early-terminated query waits before destroying it. No Phase B relaunch.
* Node-based: queue peek at Prepare(i) as before, with the metadata
  placeholder; 2x card guard unchanged; no launch at the first iteration
  (the peek is right only 13% of the time before the first merge).
* Launch guard (both strategies): the next sub-query's template-cache key
  is predicted from its IR before anything is compiled
  (`DuckDBAdapter::PredictQjitCacheKey` → `IrToLlvmCompiler::QueryStepsCacheKey`
  / `HasCachedObject`) and the launch is skipped when the object is already
  cached (an inline hit costs ~0.4 ms). The prediction equals the key the
  bg compile really uses in 55-71% of topdown hits and 29% of node-based
  hits (the node-based bg compile uses DuckDB's re-parsed plan).
* CSV accounting parity per strategy: node-based's re-split is timed in
  both modes (the no-spec path times it in the extract section), topdown's
  DP is untimed in both (the no-spec path runs it in the untimed loop
  predicate). For an iteration whose early commit fires, the planning and
  launch run inside the adapter's execute timer, so the CSV's execute
  column includes them and the update-IR column holds only the patch;
  totals are unaffected.
* Spec-jit mode with an object cache keeps compiled query-jit modules
  resident (`DuckDBAdapter::SetJitResident`): template-cache hits are
  served as live symbols; modules are dropped with the object cache at the
  repeat boundary. Without a cache every compile is a miss, so residency
  stays off there. The no-spec path is unchanged.
* Summary line (`[AQP-SPECJIT] summary`, counted and printed only with
  `--spec-jit-stats` or `--debug`; performance runs skip all of it):
  hits, misses, card_misses, not_ready, bg_errors,
  compensate_fast/interp, skipped_window (early commits rejected by the
  window rule), fired (early commits taken), skipped_cached (launches
  skipped because the predicted compile is cached), skipped_first
  (node-based first-iteration skips), key_match / key_mismatch (predicted
  cache key vs the one the bg compile used, on hits), wait_ms.

CSVs from the measurement rounds are in `measure/job_result/` (step-named
files from round 1, `backup_20261013/` and `round1_20261013/` hold the
canonical files before each full re-run).

## Bugs and hazards found

* `EnsureSpecCompiler` / `EnsureCrossCompiler` called `SetCache(true)`
  (mode 1, strict key) instead of `SetCache(duck->GetJitCache())` (mode 3
  for `single-run-template`), so bg compiles computed different cache keys
  than the inline compiler. Fixed (DuckDB and PostgreSQL paths). Even
  after the fix the bg key differs because the bg IR differs (see above).
* The Phase B topdown launch compiled the *raw* SQL (no range predicates)
  while the inline path compiles the SQL after
  `ApplyCrossSubPlanOptimizations`, so a Phase B "hit" executed weaker
  code (+0.1 s execute in the Phase-B-only runs). Removed with the revert.
* `fprintf` is a no-op macro in `ir_query_splitter.cpp` (libpg_query's
  `pg_functions.hpp` does `#define fprintf(...)`; noted in
  `duckdb_adapter.cpp:54`). Use `std::cerr` in that TU.
* Phase A speculation in query-jit mode forces the speculated temp through
  `QjitTableToCollection` + `Combine` (CDC placeholder). Node-based
  spec-jit pays this today.
* `ConvertIRToSQL(ir, query_id)` does not embed `query_id` in the SQL
  (file naming only); the old "alias mismatch" concern was wrong.
* Node-based `PeekNextSubquery` predicts a different SELECT list than the
  real sub-query (e.g. 10a iter 2: `title_6.title, temp1_14.col0,
  temp1_14.col1` vs `title_6.id, title_6.title, temp1_14.col1`), so Phase B
  learns a miss for most iterations and node-based spec-jit hits only 5%.
* With the same-IR oracle a retired (MISSed) bg task can still read the
  recorded IR while the next repeat's record overwrites it; the record step
  now waits for zombie futures first (this crashed the first forced-miss run).
* Found by the final code review of the early-commit path and fixed
  (results were already identical; these affect plan fidelity, memory and
  thread safety):
  - early-terminated query destroyed the precomputed next sub-IR while the
    bg compile could still read it (use-after-free);
  - a card miss retired the speculation without waiting, so the inline
    compile annotated join build sides in the same IR the bg task was
    writing (data race);
  - the extrapolated cardinality stayed in the remaining IR's temp leaf,
    the next sub-IR's leaf and the EXPLAIN cache, so later iterations and
    the final query planned on the guess instead of the actual value;
  - resident modules were enabled even with the object cache off, where
    nothing ever released them;
  - the window rule divided the time since Run() start by the current
    step's completed fraction, overestimating the remaining time whenever
    an earlier hash-table build took long;
  - the pool kept waking every 10 µs for the rest of the step after the
    callback had fired;
  - the placeholder's type switch defaulted unknown types to VARCHAR while
    the real store rejects them; the two paths now share one helper, as do
    the constructor-plan column names (`CtorPlanNames`) and the object-cache
    key material (`QueryStepsOptTag` / `QueryStepsKeyContent`).

## Corrections to the previous version of this file

* "Phase B fallback kept — ~0.5 ms overlap, marginal benefit": wrong. It
  has zero overlap (every hit is `not_ready`) and costs +7% on all three
  backends. The revert removed the Phase A code; the Phase B launch was
  also removed because it is a net loss.
* "PeekNextExtraction SQL match rate: 100%": not reproducible. Phase B
  retires a mismatching Phase A speculation *before* `CheckSpeculativeResult`
  runs, so mismatches were counted as `compensate_fast`, never as `misses`;
  `hits/(hits+misses)` was therefore 100% by construction. Measured
  one-step stability is 55%.
* "jit_compile (sub, 568 iters) 3.894 s = 6.856 ms avg; execute (sub)
  1.956 s = 3.444 ms avg": the denominators included the 113 final
  queries. There are 455 sub-queries: 8.6 ms and 4.3 ms. Totals were
  otherwise consistent with the re-run (5.418 s compile, 3.417 s execute).
* "~29% of sub-query jit_compile is hideable (llvm)": measured 18.8%
  (0.737 s of 3.922 s); 69% for tpde.
* "bg compile is SLOWER than inline (bg: Parse+Optimize ~14 ms vs inline
  ~12 ms)": those magnitudes are LLVM cold-compile times, not the parse/
  optimize cost. Measured: parse+optimize 0.5 ms; bg total 1.35 ms vs
  inline 0.95 ms (tpde); the gap is mostly cache-key mismatch and
  contention-slowed codegen.
* "+425 ms execution contention from the bg thread": the execute penalty
  is real (+0.21 s with a metadata placeholder, +1.1 s with the CDC
  placeholder) but it is not CPU contention; it persists with a free core
  and is caused by the placeholder copy and by the bg path's different
  physical plan.
* The claim that topdown spec-jit "cannot work because sub-query i+1 is
  unknowable before Execute(i)" is too strong: it is predictable 55% of the
  time one step ahead and 100% with the oracle, but even the oracle does
  not pay off. The binding constraints are the overlap ceiling and the
  bg-path costs, not knowability.

## Repositories and build

- **AQPHub**: `/home/pei/Project/AQP_middleware` (this tree: `/home/pei/Project/JITHub`)
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

```bash
cd /home/pei/Project/JITHub && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/JITHub && cmake --build build_release -j$(nproc)
# DuckDB — the middleware links the PREBUILT libduckdb.so from
# ${DUCKDB_ROOT}/build/release/src (find_library + dynamic link).
cd /home/pei/Project/duckdb && cmake --build build/release -j 16
```

Build hazards:
- build_debug and build_release share `lib/*.a` outputs. If a debug build
  poisons the release archive (ASan), rebuild with
  `--target IR_SQL_Converter_C_static --clean-first`.
- The compile command is `ccache ccache clang++` (double wrapper via
  `/usr/lib/ccache` in PATH); `CCACHE_DISABLE=1` works for a forced rebuild.

## Verification workflow

1. Single query end-to-end: run `./build_release/aqp_middleware` with the
   flags from `measure/measure_breakdown_time_aqp.sh` on a one-query
   directory and compare the printed rows with a baseline run (the oracle
   runs above were checked this way for all 113 queries).
2. Full JOB and DSB correctness: `measure/correctness_test_job_duckdb.sh`,
   `measure/correctness_test_dsb_duckdb.sh`.
3. Breakdown parsing:
   `/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/print_breakdown.py`
   (library: `plot_middleware_jit.py`). Per-iteration view:
   `measure/analyze_spec_overlap.py`.

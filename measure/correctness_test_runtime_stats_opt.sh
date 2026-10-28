#!/usr/bin/env bash
# correctness_test_runtime_stats_opt.sh
#
# Verify that AQP_DUCK_PROBE_OPT (block skip / per-row guard / existence
# pre-check in DuckDB's hash join) produces correct results by comparing
# against query-JIT as the known-correct baseline.
#
# Steps for each benchmark:
#   1. Run query-JIT (jit_level=query) without AQP_DUCK_PROBE_OPT → baseline
#   2. Run no-JIT (jit_level=none) with AQP_DUCK_PROBE_OPT=1     → test
#   3. Extract query results (strip timing/log noise)
#   4. Diff the two result files
#   5. Report PASS if identical, FAIL if different
#
# Usage:
#   bash correctness_test_runtime_stats_opt.sh [job|dsb_50|all]
#
# Default: all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

RUN_SCRIPT="${SCRIPT_DIR}/run_aqp.sh"
COMMON_ARGS="on on on all single-run-template off tpde"

# Baseline: query-JIT (known correct)
BASELINE_ARGS="duckdb topdown query none ${COMMON_ARGS}"
# Test: no-JIT with AQP_DUCK_PROBE_OPT=1
TEST_ARGS="duckdb topdown none none ${COMMON_ARGS}"

benchmarks="${1:-all}"
if [[ "$benchmarks" == "all" ]]; then
    benchmarks="job dsb_50"
fi

# Extract query result rows from the benchmark log, stripping all timing,
# status messages, and logging noise. Only keeps query output lines.
extract_results() {
    local logfile="$1"
    grep -E "^=== Query Results|^Rows:|^[^[{(]" "$logfile" \
        | grep -vE "^(duckdb |ANALYZ|Benchmark|Total q|Passed|Failed|={5,}|Execution|Running|Loading|storage|Pre-warming)" \
        | grep -vE "[0-9]+ ms$" \
        | sed '/^$/d'
}

# Derive the log file name that run_aqp.sh produces for the given bench + args.
get_log_name() {
    local bench="$1"
    local jit_level="$2"
    local jit_simd="$3"
    local suffix
    if [[ "$bench" == "job" ]]; then
        suffix="job.txt"
    elif [[ "$bench" == dsb_* ]]; then
        suffix="dsb.txt"
    else
        suffix="${bench}.txt"
    fi
    echo "aqp_middleware_duckdb_topdown_${jit_level}_${jit_simd}_jitcache_single_run_template_tpde_${suffix}"
}

get_result_dir() {
    local bench="$1"
    if [[ "$bench" == "job" ]]; then
        echo "${SCRIPT_DIR}/job_result"
    elif [[ "$bench" == dsb_* ]]; then
        local sf="${bench#dsb_}"
        echo "${SCRIPT_DIR}/dsb_result_sf${sf}"
    else
        echo "${SCRIPT_DIR}/${bench}_result"
    fi
}

pass=0
fail=0
total=0

for bench in $benchmarks; do
    total=$((total + 1))
    bench_ok=true
    echo "========================================"
    echo " Testing: $bench"
    echo "========================================"

    result_dir=$(get_result_dir "$bench")
    baseline_log=$(get_log_name "$bench" "query" "none")
    test_log=$(get_log_name "$bench" "none" "none")

    ##################################################################
    # Step 1: Baseline — query-JIT (without AQP_DUCK_PROBE_OPT)
    ##################################################################
    echo ""
    echo "[Step 1] Baseline: query-JIT (jit_level=query, no AQP_DUCK_PROBE_OPT)"
    (
        unset AQP_DUCK_PROBE_OPT 2>/dev/null || true
        cd "$SCRIPT_DIR"
        bash "$RUN_SCRIPT" "$bench" $BASELINE_ARGS
    ) > "$TMPDIR/${bench}_run_baseline.log" 2>&1
    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "         FAILED to run (exit code $rc)"
        echo "         See: $TMPDIR/${bench}_run_baseline.log"
        fail=$((fail + 1))
        bench_ok=false
    fi

    if $bench_ok && ! cp "${result_dir}/${baseline_log}" "$TMPDIR/${bench}_baseline.txt" 2>/dev/null; then
        echo "         Log not found: ${result_dir}/${baseline_log}"
        fail=$((fail + 1))
        bench_ok=false
    fi

    if $bench_ok; then
        bl_passed=$(grep -oP 'Passed: \K[0-9]+' "$TMPDIR/${bench}_baseline.txt" | tail -1)
        bl_failed=$(grep -oP 'Failed: \K[0-9]+' "$TMPDIR/${bench}_baseline.txt" | tail -1)
        bl_total=$(grep -oP 'Total queries: \K[0-9]+' "$TMPDIR/${bench}_baseline.txt" | tail -1)
        echo "         ${bl_passed}/${bl_total} passed, ${bl_failed} failed"

        if [[ "${bl_failed:-1}" -ne 0 ]]; then
            echo "         Baseline (query-JIT) has failures — cannot verify."
            fail=$((fail + 1))
            bench_ok=false
        fi
    fi

    ##################################################################
    # Step 2: Test — no-JIT with AQP_DUCK_PROBE_OPT=1
    ##################################################################
    if $bench_ok; then
        echo ""
        echo "[Step 2] Test: no-JIT + AQP_DUCK_PROBE_OPT=1 (jit_level=none)"
        (
            export AQP_DUCK_PROBE_OPT=1
            cd "$SCRIPT_DIR"
            bash "$RUN_SCRIPT" "$bench" $TEST_ARGS
        ) > "$TMPDIR/${bench}_run_test.log" 2>&1
        rc=$?

        if [[ $rc -ne 0 ]]; then
            echo "         FAILED to run (exit code $rc)"
            echo "         See: $TMPDIR/${bench}_run_test.log"
            fail=$((fail + 1))
            bench_ok=false
        fi

        if $bench_ok && ! cp "${result_dir}/${test_log}" "$TMPDIR/${bench}_test.txt" 2>/dev/null; then
            echo "         Log not found: ${result_dir}/${test_log}"
            fail=$((fail + 1))
            bench_ok=false
        fi

        if $bench_ok; then
            t_passed=$(grep -oP 'Passed: \K[0-9]+' "$TMPDIR/${bench}_test.txt" | tail -1)
            t_failed=$(grep -oP 'Failed: \K[0-9]+' "$TMPDIR/${bench}_test.txt" | tail -1)
            t_total=$(grep -oP 'Total queries: \K[0-9]+' "$TMPDIR/${bench}_test.txt" | tail -1)
            echo "         ${t_passed}/${t_total} passed, ${t_failed} failed"

            if [[ "${t_failed:-1}" -ne 0 ]]; then
                echo "         Optimized run has query failures!"
                fail=$((fail + 1))
                bench_ok=false
            fi
        fi
    fi
unset AQP_DUCK_PROBE_OPT 2>/dev/null || true
    ##################################################################
    # Step 3: Compare query results
    ##################################################################
    if $bench_ok; then
        echo ""
        echo "[Step 3] Comparing: query-JIT vs no-JIT+AQP_DUCK_PROBE_OPT"

        extract_results "$TMPDIR/${bench}_baseline.txt" > "$TMPDIR/${bench}_baseline_results.txt"
        extract_results "$TMPDIR/${bench}_test.txt"     > "$TMPDIR/${bench}_test_results.txt"

        bl_lines=$(wc -l < "$TMPDIR/${bench}_baseline_results.txt")
        t_lines=$(wc -l < "$TMPDIR/${bench}_test_results.txt")
        echo "         Baseline (query-JIT) result lines: ${bl_lines}"
        echo "         Test (no-JIT+opt)    result lines: ${t_lines}"

        if diff -q "$TMPDIR/${bench}_baseline_results.txt" "$TMPDIR/${bench}_test_results.txt" > /dev/null 2>&1; then
            echo ""
            echo "[PASS]  $bench: results IDENTICAL"
            pass=$((pass + 1))
        else
            echo ""
            echo "[FAIL]  $bench: results DIFFER!"
            echo ""
            echo "--- First 20 lines of diff ---"
            diff "$TMPDIR/${bench}_baseline_results.txt" "$TMPDIR/${bench}_test_results.txt" | head -20
            diff_file="$TMPDIR/${bench}_diff.txt"
            diff "$TMPDIR/${bench}_baseline_results.txt" "$TMPDIR/${bench}_test_results.txt" > "$diff_file" 2>/dev/null || true
            echo "--- Full diff: $diff_file ---"
            fail=$((fail + 1))
        fi
    fi

    ##################################################################
    # Always print timing (even on failure)
    ##################################################################
    echo ""
    bl_ms=$(grep -oP 'duckdb runs: \K[0-9]+' "$TMPDIR/${bench}_run_baseline.log" 2>/dev/null | tail -1)
    t_ms=$(grep -oP 'duckdb runs: \K[0-9]+' "$TMPDIR/${bench}_run_test.log" 2>/dev/null | tail -1)
    echo "[TIME]  query-JIT=${bl_ms:-?} ms    no-JIT+opt=${t_ms:-?} ms"
    echo ""
done

echo "========================================"
echo " Summary: $pass/$total passed, $fail failed"
echo "========================================"

if [[ $fail -gt 0 ]]; then
    echo "CORRECTNESS CHECK FAILED"
    exit 1
else
    echo "ALL CORRECTNESS CHECKS PASSED"
    exit 0
fi

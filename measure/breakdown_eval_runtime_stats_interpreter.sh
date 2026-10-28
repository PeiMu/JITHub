#!/usr/bin/env bash
#
# Evaluate the effect of the runtime statistics for interpreter (e.g., DuckDB).
#
# Compares two configurations:
#   1. Baseline: Run query-JIT (jit_level=query) without AQP_DUCK_PROBE_OPT
#   2. Run no-JIT (jit_level=none) with AQP_DUCK_PROBE_OPT=1
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/env.sh"

COMMON="on on on all single-run-template off tpde"
# Baseline: query-JIT
BASELINE_ARGS="job duckdb topdown query none ${COMMON}"
# Test: no-JIT with AQP_DUCK_PROBE_OPT=1
TEST_ARGS="job duckdb topdown none none ${COMMON}"

DEST_DIR="${SCRIPT_DIR}/job_result"

# Step 1: Baseline (query-jit) JOB — reuse existing results
echo "=== Step 1: Baseline (query-jit) JOB ==="
#bash ./measure_breakdown_time_aqp.sh $BASELINE_ARGS
BASELINE_CSV="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv"
if [[ ! -f "$BASELINE_CSV" ]]; then
    echo "ERROR: Baseline CSV not found: $BASELINE_CSV"
    echo "       Run: bash ./measure_breakdown_time_aqp.sh $BASELINE_ARGS"
    exit 1
fi
cp "$BASELINE_CSV" "${DEST_DIR}/compare_runtime_stats_jit_baseline.csv"
echo "  Copied baseline from: $BASELINE_CSV"

# Step 2: no-JIT with AQP_DUCK_PROBE_OPT=1 JOB
echo ""
echo "=== Step 2: no-JIT with AQP_DUCK_PROBE_OPT=1 JOB ==="
export AQP_DUCK_PROBE_OPT=1
bash ./measure_breakdown_time_aqp.sh $TEST_ARGS "" "" "" on
unset AQP_DUCK_PROBE_OPT

TEST_CSV="${DEST_DIR}/duckdb_topdown_none_none_jitcache_single_run_template_collectstats_breakdown_time_log.csv"
if [[ ! -f "$TEST_CSV" ]]; then
    echo "ERROR: Test CSV not found: $TEST_CSV"
    exit 1
fi
mv "$TEST_CSV" "${DEST_DIR}/compare_runtime_stats_interpreter.csv"

echo ""
echo "=== Runtime Stats JIT/Interpreter Comparison Complete ==="
echo "Output:"
echo "  ${DEST_DIR}/compare_runtime_stats_jit_baseline.csv"
echo "  ${DEST_DIR}/compare_runtime_stats_interpreter.csv"

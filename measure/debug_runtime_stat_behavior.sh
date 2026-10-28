#!/usr/bin/env bash
#
# Debug runtime-stat-behavior: run JOB queries on both JIT and DuckDB interpreter
# with AQP_DEBUG_RUNTIME_STAT_BEHAVIOR=1 to count how many rows are filtered by
# range guard, block skip, and membership/existence checks.
#
# Usage:
#   bash measure/debug_runtime_stat_behavior.sh [query]
#   e.g.: bash measure/debug_runtime_stat_behavior.sh 1a
#         bash measure/debug_runtime_stat_behavior.sh        # runs all JOB queries
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

QUERY_FILTER="${1:-}"
OUT_DIR="${SCRIPT_DIR}/job_result/debug_runtime_stat"
mkdir -p "$OUT_DIR"

DB="${DUCKDB_DB_JOB}"
STORAGE_CACHE="${STORAGE_CACHE_DUCKDB_JOB}"
JOB_DIR="${JOB_PATH}/queries"
SCHEMA="${JOB_PATH}/schema.sql"
FKEYS="${JOB_PATH}/fkeys.sql"

if [[ ! -d "$JOB_DIR" ]]; then
    echo "JOB query dir not found: $JOB_DIR"
    echo "Set JOB_PATH to the JOB benchmark directory."
    exit 1
fi

export AQP_DEBUG_RUNTIME_STAT_BEHAVIOR=1  # for DuckDB interpreter side

JIT_LOG="${OUT_DIR}/jit_debug.log"
INT_LOG="${OUT_DIR}/interpreter_debug.log"
INT_STATS_LOG="${OUT_DIR}/interpreter_stats_debug.log"
> "$JIT_LOG"
> "$INT_LOG"
> "$INT_STATS_LOG"

# Collect SQL files
sql_files=()
if [[ -n "$QUERY_FILTER" ]]; then
    for f in "${JOB_DIR}/${QUERY_FILTER}"*.sql; do
        [[ -e "$f" ]] && sql_files+=("$f")
    done
else
    for f in "${JOB_DIR}"/*.sql; do
        [[ -e "$f" ]] && sql_files+=("$f")
    done
fi

if [[ ${#sql_files[@]} -eq 0 ]]; then
    echo "No queries found for filter '${QUERY_FILTER}' in $JOB_DIR"
    exit 1
fi

echo "Running ${#sql_files[@]} queries..."

for sqlfile in "${sql_files[@]}"; do
    qname=$(basename "$sqlfile" .sql)
    _tmp=$(mktemp -d)
    ln -s "$(realpath "$sqlfile")" "${_tmp}/${qname}.sql"

    # JIT run
    echo "QUERY=$qname" >> "$JIT_LOG"
    "${PROJECT}/build_release/aqp_middleware" \
        --engine=duckdb \
        --db="${DB}" \
        --schema="${SCHEMA}" \
        --fkeys="${FKEYS}" \
        --split=topdown \
        --timing \
        --no-analyze \
        --repeat=1 \
        --jit-level=query --jit-simd=none \
        --compile-mode=tpde \
        --storage-plan --storage-cache="${STORAGE_CACHE}" \
        --jit-skip-hash-cmp=all \
        --debug-runtime-stat-behavior \
        --benchmark \
        "${_tmp}" 2>> "$JIT_LOG" > /dev/null

    # Interpreter run (without collect-stats — auto resolves to off for jit_level=none)
    echo "QUERY=$qname" >> "$INT_LOG"
    AQP_DUCK_PROBE_OPT=1 \
    "${PROJECT}/build_release/aqp_middleware" \
        --engine=duckdb \
        --db="${DB}" \
        --schema="${SCHEMA}" \
        --fkeys="${FKEYS}" \
        --split=topdown \
        --timing \
        --no-analyze \
        --repeat=1 \
        --jit-level=none --jit-simd=none \
        --storage-plan --storage-cache="${STORAGE_CACHE}" \
        --benchmark \
        "${_tmp}" 2>> "$INT_LOG" > /dev/null

    # Interpreter run (with collect-stats ON — cross-sub-query stats propagation)
    echo "QUERY=$qname" >> "$INT_STATS_LOG"
    AQP_DUCK_PROBE_OPT=1 \
    "${PROJECT}/build_release/aqp_middleware" \
        --engine=duckdb \
        --db="${DB}" \
        --schema="${SCHEMA}" \
        --fkeys="${FKEYS}" \
        --split=topdown \
        --timing \
        --no-analyze \
        --repeat=1 \
        --jit-level=none --jit-simd=none --collect-stats \
        --storage-plan --storage-cache="${STORAGE_CACHE}" \
        --benchmark \
        "${_tmp}" 2>> "$INT_STATS_LOG" > /dev/null

    rm -rf "$_tmp"
    echo "  $qname done"
done

echo ""
echo "Logs written to:"
echo "  JIT:                    $JIT_LOG"
echo "  Interpreter (no stats): $INT_LOG"
echo "  Interpreter (stats ON): $INT_STATS_LOG"
echo ""
echo "Run analysis:"
echo "  python3 ${SCRIPT_DIR}/analyze_debug_runtime_stat.py ${OUT_DIR}/"

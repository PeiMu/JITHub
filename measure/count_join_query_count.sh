#!/usr/bin/env bash
#
# count_join_query_count.sh — Count join and sub-query statistics for JOB and DSB benchmarks.
#
# Reports per benchmark:
#   1. Total join count across the whole benchmark
#   2. Max join count of a single query
#   3. Number of queries
#   4. Number of generated sub-queries (from an actual middleware run)
#   5. Max number of generated sub-queries of a single query
#
# For sub-query counting, the middleware is run with --debug
# and the "Total iterations: N" output per query is parsed.
# Sub-queries per query = N iterations (split loop) + 1 (final execution).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env.sh"

BINARY="${PROJECT}/build_release/aqp_middleware"

JOB_DIR="${JOB_PATH}/queries"
JOB_SCHEMA="${JOB_PATH}/schema.sql"
JOB_FKEYS="${JOB_PATH}/fkeys.sql"

DSB_DIR="${DSB_PATH}/code/tools/1_instance_out_aqp/1/"
DSB_SCHEMA="${DSB_PATH}/scripts/create_tables.sql"
DSB_FKEYS="${DSB_PATH}/scripts/tpcds_ri_umbra.sql"

########################################
# Count joins per SQL file (static analysis)
########################################
count_joins() {
    local sql_file="$1"
    python3 -c "
import re, sys

text = open(sys.argv[1]).read()

# Remove SQL comments
text = re.sub(r'--[^\n]*', '', text)
text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

# Count tables by combining implicit joins (FROM ... WHERE) and explicit JOINs
# Handle subqueries: only count the outermost FROM clause

# Strategy: count all table references in FROM clauses
# 1. Explicit JOINs
join_count = len(re.findall(r'\bJOIN\b', text, re.IGNORECASE))

# 2. Implicit joins in FROM clause: count comma-separated table references
# Find FROM ... WHERE blocks (greedy for the outermost)
from_blocks = re.findall(r'\bFROM\s+(.*?)\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|\)|\Z)',
                          text, re.DOTALL | re.IGNORECASE)

implicit_tables = 0
for block in from_blocks:
    # Remove any JOIN ... ON ... clauses from the block
    # Split by comma, but be careful about nested parentheses
    clean = re.sub(r'\b(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL)?\s*(?:OUTER\s+)?JOIN\b.*?(?=,|\bWHERE\b|\bGROUP\b|\Z)', '', block, flags=re.DOTALL | re.IGNORECASE)
    parts = [p.strip() for p in clean.split(',') if p.strip()]
    # Filter out empty parts and parts that are just whitespace/keywords
    table_parts = [p for p in parts if p and not re.match(r'^\s*$', p)]
    implicit_tables += len(table_parts)

# Total joins = explicit_joins + (implicit_tables - 1) if implicit_tables > 0
# But we need to be careful not to double-count
total_tables = implicit_tables + join_count
total_joins = max(0, total_tables - 1) if implicit_tables > 0 else join_count

print(total_joins)
" "$sql_file" 2>/dev/null || echo "0"
}

########################################
# Count joins for a benchmark directory
########################################
count_benchmark_joins() {
    local bench_dir="$1"
    local bench_name="$2"
    local total_joins=0
    local max_joins=0
    local num_queries=0
    local max_joins_query=""

    local sql_files=()
    if [[ -d "${bench_dir}" ]]; then
        while IFS= read -r -d '' f; do
            sql_files+=("$f")
        done < <(find "$bench_dir" -name '*.sql' -print0 | sort -z)
    fi

    for sql_file in "${sql_files[@]}"; do
        local joins
        joins=$(count_joins "$sql_file")
        total_joins=$((total_joins + joins))
        if (( joins > max_joins )); then
            max_joins=$joins
            max_joins_query=$(basename "$sql_file")
        fi
        num_queries=$((num_queries + 1))
    done

    echo "${num_queries} ${total_joins} ${max_joins} ${max_joins_query}"
}

########################################
# Run middleware and count sub-queries from debug output
########################################
count_subqueries_from_run() {
    local bench="$1"
    local engine="$2"
    local split="$3"
    local jit_cache="$4"
    local spec_jit="$5"
    local compile_mode="$6"

    local bench_dir schema fkeys db_conn storage_cache helper_db_arg=""

    local real_bench="$bench"
    if [[ "$bench" == dsb_* ]]; then
        DSB_SF="${bench#dsb_}"
        real_bench="dsb"
        source "${SCRIPT_DIR}/env.sh"
    fi

    if [[ "$real_bench" == "job" ]]; then
        bench_dir="${JOB_DIR}"
        schema="${JOB_SCHEMA}"
        fkeys="${JOB_FKEYS}"
        db_conn="${DUCKDB_DB_JOB}"
        storage_cache="${STORAGE_CACHE_DUCKDB_JOB}"
    else
        bench_dir="${DSB_DIR}"
        schema="${DSB_SCHEMA}"
        fkeys="${DSB_FKEYS}"
        db_conn="${DUCKDB_DB_DSB}"
        storage_cache="${STORAGE_CACHE_DUCKDB_DSB}"
    fi

    if [[ "$split" == "node-based" || "$split" == "topdown" ]]; then
        if [[ "$engine" != "duckdb" ]]; then
            helper_db_arg="--helper-db-path=${DUCKDB_DB_JOB}"
        fi
    fi

    local compile_mode_flag=""
    if [[ "$compile_mode" != "llvm" ]]; then
        compile_mode_flag="--compile-mode=${compile_mode}"
    fi

    local jit_cache_flag=""
    if [[ "$jit_cache" != "off" ]]; then
        jit_cache_flag="--jit-cache=${jit_cache}"
    fi

    local spec_jit_flag=""
    if [[ "$spec_jit" != "off" ]]; then
        spec_jit_flag="--spec-jit=${spec_jit}"
    fi

    local tmpout
    tmpout=$(mktemp)

    "$BINARY" \
        --engine="${engine}" \
        --db="${db_conn}" \
        ${helper_db_arg} \
        --schema="${schema}" \
        --fkeys="${fkeys}" \
        --split="${split}" \
        --no-analyze --jit-level=query --jit-simd=none \
        ${jit_cache_flag} \
        ${spec_jit_flag} \
        ${compile_mode_flag} \
        --storage-plan --storage-cache="${storage_cache}" \
        --debug \
        --benchmark \
        "${bench_dir}" \
        > "$tmpout" 2>/dev/null || true

    # Parse "Total iterations: N" lines from stdout.
    # Each query produces one such line. Sub-queries = N + 1 (iterations + final).
    local total_subqueries=0
    local max_subqueries=0
    local num_queries=0
    local cur_query=""
    local max_subqueries_query=""

    while IFS= read -r line; do
        if [[ "$line" == Run\ * ]]; then
            cur_query=$(basename "${line#Run }")
        elif [[ "$line" == "Total iterations: "* ]]; then
            local iters="${line#Total iterations: }"
            local subs=$((iters + 1))
            total_subqueries=$((total_subqueries + subs))
            if (( subs > max_subqueries )); then
                max_subqueries=$subs
                max_subqueries_query="$cur_query"
            fi
            num_queries=$((num_queries + 1))
        fi
    done < "$tmpout"

    rm -f "$tmpout"
    echo "${num_queries} ${total_subqueries} ${max_subqueries} ${max_subqueries_query}"
}

########################################
# Main
########################################
main() {
    echo "=========================================================================="
    echo "              Join and Sub-Query Count Statistics"
    echo "=========================================================================="
    echo ""

    # ---- JOB ----
    echo "--- JOB Benchmark ---"
    echo "  Query dir: ${JOB_DIR}"

    read -r jq jtotal jmax jmax_q <<< "$(count_benchmark_joins "$JOB_DIR" "JOB")"
    echo "  [Static] Number of queries:       ${jq}"
    echo "  [Static] Total joins:             ${jtotal}"
    echo "  [Static] Max joins (one query):   ${jmax}  (${jmax_q})"

    echo ""
    echo "  Running middleware to count sub-queries (cache=off, no spec-jit) ..."
    read -r sq_n sq_total sq_max sq_max_q <<< "$(count_subqueries_from_run job duckdb topdown single-run-template off tpde)"
    echo "  [Run] Number of queries:              ${sq_n}"
    echo "  [Run] Total generated sub-queries:    ${sq_total}"
    echo "  [Run] Max sub-queries (one query):    ${sq_max}  (${sq_max_q})"

    echo ""
    echo "  Running middleware to count sub-queries (spec-jit=recompile) ..."
    read -r sq_n2 sq_total2 sq_max2 sq_max_q2 <<< "$(count_subqueries_from_run job duckdb topdown single-run-template recompile tpde)"
    echo "  [Run+SpecJIT] Number of queries:           ${sq_n2}"
    echo "  [Run+SpecJIT] Total generated sub-queries: ${sq_total2}"
    echo "  [Run+SpecJIT] Max sub-queries (one query): ${sq_max2}  (${sq_max_q2})"

    echo ""

    # ---- DSB ----
    echo "--- DSB Benchmark ---"
    echo "  Query dir: ${DSB_DIR}"

    read -r dq dtotal dmax dmax_q <<< "$(count_benchmark_joins "$DSB_DIR" "DSB")"
    echo "  [Static] Number of queries:       ${dq}"
    echo "  [Static] Total joins:             ${dtotal}"
    echo "  [Static] Max joins (one query):   ${dmax}  (${dmax_q})"

    echo ""
    echo "  Running middleware to count sub-queries (cache=off, no spec-jit) ..."
    read -r dsq_n dsq_total dsq_max dsq_max_q <<< "$(count_subqueries_from_run dsb_50 duckdb topdown single-run-template off tpde)"
    echo "  [Run] Number of queries:              ${dsq_n}"
    echo "  [Run] Total generated sub-queries:    ${dsq_total}"
    echo "  [Run] Max sub-queries (one query):    ${dsq_max}  (${dsq_max_q})"

    echo ""
    echo "  Running middleware to count sub-queries (spec-jit=recompile) ..."
    read -r dsq_n2 dsq_total2 dsq_max2 dsq_max_q2 <<< "$(count_subqueries_from_run dsb_50 duckdb topdown single-run-template recompile tpde)"
    echo "  [Run+SpecJIT] Number of queries:           ${dsq_n2}"
    echo "  [Run+SpecJIT] Total generated sub-queries: ${dsq_total2}"
    echo "  [Run+SpecJIT] Max sub-queries (one query): ${dsq_max2}  (${dsq_max_q2})"

    echo ""
    echo "=========================================================================="
}

main "$@"

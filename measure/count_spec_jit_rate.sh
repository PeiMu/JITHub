#!/usr/bin/env bash
#
# count_spec_jit_rate.sh — Measure the speculative JIT correct rate and ready
# rate from the [AQP-SPECJIT] summary lines. The counting itself only runs
# with --spec-jit-stats (performance runs never count or print).
#
# Per run (all queries of the benchmark, one repeat):
#   spec-check  sub-queries that could have used a speculation: every
#               sub-query except the first of each query (the final query is
#               not a sub-query and is never speculated)
#   correct     the speculation launched for it was right and its code ran
#               (summary field hits)
#   wrong       a speculation was launched but could not be used: predicted
#               sub-query differs (misses, incl. launches retired before the
#               check), cardinality guard failed (card_misses), or the bg
#               compile failed (bg_errors)
#   skipped     spec-check - correct - wrong: no speculation was launched
#               (window rule, predicted compile already cached, node-based
#               first-iteration skip, learned miss, ...)
#   not_ready   subset of correct whose bg compile was still running when the
#               next sub-query needed it (the main thread waited)
#
# correct%  = correct / spec-check
# ready%    = (correct - not_ready) / spec-check
#
# The runs use the same flags as measure_breakdown_time_aqp.sh (query-jit,
# --jit-skip-hash-cmp=all, storage plan) with a single repeat, so the rates
# describe single-run speculation without cross-repeat miss learning.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_FILE="${SCRIPT_DIR}/spec_jit_rate_results.txt"
STDERR_DIR="/tmp/aqp_spec_jit_dumps"

# bench  engine  split  jit_cache  compile_mode
CONFIGS=(
    "job       duckdb      topdown    single-run-template       tpde"
    "job       duckdb      topdown    single-run-template       fastisel"
    "job       duckdb      topdown    single-run-template       llvm"
    "job       duckdb      node-based single-run-template       tpde"
    "job       duckdb      node-based single-run-template       fastisel"
    "job       duckdb      node-based single-run-template       llvm"
    "dsb_50    duckdb      node-based single-run-template       tpde"
    "job       postgresql  node-based single-run-strict          fastisel"
    "dsb_50    postgresql  node-based single-run-parameterized   tpde"
)

########################################
# Source env.sh for paths and DB connections
########################################
source "${SCRIPT_DIR}/env.sh"

########################################
# Engine start/stop helpers
########################################
pg_running=false
pg_start() {
    if ! "$pg_running"; then
        echo "  Starting PostgreSQL..."
        ${PG_BIN}/pg_ctl start -l "${PG_LOG}" -D "${PG_DATA}"
        pg_running=true
    fi
}
pg_stop() {
    if "$pg_running"; then
        echo "  Stopping PostgreSQL..."
        ${PG_BIN}/pg_ctl stop -D "${PG_DATA}" -m smart -s
        pg_running=false
    fi
}

########################################
# ANALYZE (once per engine+bench)
########################################
declare -A analyzed=()
run_analyze() {
    local bench="$1" engine="$2"
    local key="${engine}_${bench}"
    if [[ -n "${analyzed[$key]:-}" ]]; then
        return
    fi
    echo "  ANALYZING ${engine} ${bench}..."
    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        local pg_db
        if [[ "$bench" == "job" ]]; then
            pg_db="${PG_CONN_JOB}"
        else
            pg_db="${PG_CONN_DSB}"
        fi
        ${PG_BIN}/psql -d "${pg_db}" -c "ANALYZE;"
    fi
    analyzed[$key]=1
    echo "  ANALYZE done."
}

########################################
# Resolve benchmark paths and flags
########################################
resolve_config() {
    local bench="$1" engine="$2" split="$3" jit_cache="$4" compile_mode="$5"

    local real_bench="$bench"
    if [[ "$bench" == dsb_* ]]; then
        DSB_SF="${bench#dsb_}"
        real_bench="dsb"
    fi
    unset DUCKDB_DB_DSB STORAGE_CACHE_DUCKDB_DSB STORAGE_CACHE_PG_DSB PG_CONN_DSB
    source "${SCRIPT_DIR}/env.sh"

    if [[ "$real_bench" == "job" ]]; then
        _dir="${JOB_PATH:-${PROJECT}/../benchmarks/imdb_job-postgres}/queries"
        _schema="${JOB_PATH:-${PROJECT}/../benchmarks/imdb_job-postgres}/schema.sql"
        _fkeys="${JOB_PATH:-${PROJECT}/../benchmarks/imdb_job-postgres}/fkeys.sql"
        _duckdb_db="${DUCKDB_DB_JOB}"
        _storage_cache="${STORAGE_CACHE_DUCKDB_JOB}"
        _storage_cache_pg="${STORAGE_CACHE_PG_JOB}"
        _pg_db="${PG_CONN_JOB}"
    else
        if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
            _dir="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/code/tools/1_instance_out_aqp_pg/1/"
        else
            _dir="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/code/tools/1_instance_out_aqp/1/"
        fi
        _schema="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/scripts/create_tables.sql"
        _fkeys="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/scripts/tpcds_ri_umbra.sql"
        _duckdb_db="${DUCKDB_DB_DSB}"
        _storage_cache="${STORAGE_CACHE_DUCKDB_DSB}"
        _storage_cache_pg="${STORAGE_CACHE_PG_DSB}"
        _pg_db="${PG_CONN_DSB}"
    fi

    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        _storage_flags="--storage-plan --storage-cache=${_storage_cache_pg}"
    else
        _storage_flags="--storage-plan --storage-cache=${_storage_cache}"
    fi

    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        _db_conn="${_pg_db}"
    else
        _db_conn="${_duckdb_db}"
    fi

    _helper_db_arg=""
    if [[ ("$split" == "node-based" || "$split" == "topdown") && "$engine" != "duckdb" ]]; then
        _helper_db_arg="--helper-db-path=${_duckdb_db}"
    fi

    _compile_mode_flag=""
    if [[ "$compile_mode" != "llvm" ]]; then
        _compile_mode_flag="--compile-mode=${compile_mode}"
    fi
}

########################################
# Sum one counter over all summary lines of a stderr file
########################################
sum_counter() {
    local name="$1" file="$2"
    # "(?<![a-z_])" keeps "misses=" from matching "card_misses=".
    { grep '\[AQP-SPECJIT\] summary:' "$file" || true; } \
        | { grep -oP "(?<![a-z_])${name}=\K[0-9]+" || true; } \
        | awk '{s+=$1} END{print s+0}'
}

########################################
# Run one configuration, parse spec-jit summary from stderr
########################################
run_one() {
    local bench="$1" engine="$2" split="$3" jit_cache="$4" compile_mode="$5" stderr_file="$6"

    resolve_config "$bench" "$engine" "$split" "$jit_cache" "$compile_mode"

    echo "  Running: bench=${bench} engine=${engine} split=${split} cache=${jit_cache} compile=${compile_mode}"

    # --spec-jit-stats enables the outcome counters and the per-query
    # [AQP-SPECJIT] summary line on stderr.
    "${PROJECT}/build_release/aqp_middleware" \
        --engine="${engine}" \
        --db="${_db_conn}" \
        ${_helper_db_arg} \
        --schema="${_schema}" \
        --fkeys="${_fkeys}" \
        --split="${split}" \
        --no-analyze --repeat=1 \
        --jit-level=query --jit-simd=none \
        --jit-cache="${jit_cache}" \
        --jit-skip-hash-cmp=all \
        --spec-jit=recompile --spec-jit-stats \
        ${_compile_mode_flag} \
        ${_storage_flags} \
        --benchmark \
        "${_dir}" \
        > /dev/null 2> "$stderr_file" \
        || echo "    WARNING: aqp_middleware exited with non-zero status"

    local checks hits misses card_misses bg_errors not_ready
    checks=$(sum_counter checks "$stderr_file")
    hits=$(sum_counter hits "$stderr_file")
    misses=$(sum_counter misses "$stderr_file")
    card_misses=$(sum_counter card_misses "$stderr_file")
    bg_errors=$(sum_counter bg_errors "$stderr_file")
    not_ready=$(sum_counter not_ready "$stderr_file")

    local correct=$hits
    local wrong=$((misses + card_misses + bg_errors))
    local skipped=$((checks - correct - wrong))
    local correct_rate="N/A" ready_rate="N/A"
    if (( checks > 0 )); then
        correct_rate=$(python3 -c "print(f'{100*${correct}/${checks}:.1f}%')")
        ready_rate=$(python3 -c "print(f'{100*(${correct}-${not_ready})/${checks}:.1f}%')")
    fi
    echo "    => spec-check=${checks} correct=${correct} wrong=${wrong} skipped=${skipped} not_ready=${not_ready}"
    echo "       correct%=${correct_rate} ready%=${ready_rate}"
    echo -e "${bench}\t${engine}\t${split}\t${jit_cache}\t${compile_mode}\t${checks}\t${correct}\t${wrong}\t${skipped}\t${not_ready}\t${correct_rate}\t${ready_rate}" >> "$RESULTS_FILE"
}

########################################
# Main
########################################
main() {
    mkdir -p "$STDERR_DIR"
    rm -f "$RESULTS_FILE"
    echo -e "BENCH\tENGINE\tSPLIT\tCACHE\tCOMPILE\tSPEC_CHECK\tCORRECT\tWRONG\tSKIPPED\tNOT_READY\tCORRECT_PCT\tREADY_PCT" > "$RESULTS_FILE"

    trap 'echo "Caught signal, stopping..."; pg_stop; exit 1' INT TERM

    # First pass: duckdb configs
    for config in "${CONFIGS[@]}"; do
        read -r bench engine split jit_cache compile_mode <<< "$config"
        [[ "$engine" != "duckdb" ]] && continue
        run_analyze "$bench" "$engine"
        safe_name="${bench}_${engine}_${split}_${jit_cache}_${compile_mode}"
        safe_name="${safe_name//\//_}"
        run_one "$bench" "$engine" "$split" "$jit_cache" "$compile_mode" \
            "${STDERR_DIR}/${safe_name}.stderr"
    done

    # Second pass: postgres configs
    local has_pg=false
    for config in "${CONFIGS[@]}"; do
        read -r _b engine _s _c _m <<< "$config"
        if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
            has_pg=true
            break
        fi
    done
    if "$has_pg"; then
        pg_start
        for config in "${CONFIGS[@]}"; do
            read -r bench engine split jit_cache compile_mode <<< "$config"
            [[ "$engine" != "postgres" && "$engine" != "postgresql" ]] && continue
            run_analyze "$bench" "$engine"
            safe_name="${bench}_${engine}_${split}_${jit_cache}_${compile_mode}"
            safe_name="${safe_name//\//_}"
            run_one "$bench" "$engine" "$split" "$jit_cache" "$compile_mode" \
                "${STDERR_DIR}/${safe_name}.stderr"
        done
        pg_stop
    fi

    # Print summary table
    echo ""
    echo "=============================================================================================="
    echo "                       Speculative JIT correct rate and ready rate"
    echo "=============================================================================================="
    echo "  SPEC-CHECK = sub-queries except the first of each query (could have used a speculation)"
    echo "  CORRECT    = speculation right and its code ran      WRONG = launched but unusable"
    echo "  SKIPPED    = no speculation launched                 NRDY  = correct but compile was late"
    echo "  CORRECT% = CORRECT / SPEC-CHECK      READY% = (CORRECT - NRDY) / SPEC-CHECK"
    echo "=============================================================================================="
    printf "%-7s %-11s %-11s %-27s %-9s %6s %5s %5s %5s %5s %9s %8s\n" \
        "BENCH" "ENGINE" "SPLIT" "CACHE" "COMPILE" "CHECK" "CORR" "WRONG" "SKIP" "NRDY" "CORRECT%" "READY%"
    echo "----------------------------------------------------------------------------------------------"
    tail -n +2 "$RESULTS_FILE" | while IFS=$'\t' read -r bench engine split cache compile checks correct wrong skipped nr cpct rpct; do
        printf "%-7s %-11s %-11s %-27s %-9s %6s %5s %5s %5s %5s %9s %8s\n" \
            "$bench" "$engine" "$split" "$cache" "$compile" "$checks" "$correct" "$wrong" "$skipped" "$nr" "$cpct" "$rpct"
    done
    echo "=============================================================================================="
    echo ""
    echo "Raw stderr files: ${STDERR_DIR}/"
    echo "Results TSV:      ${RESULTS_FILE}"
}

main "$@"

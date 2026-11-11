#!/usr/bin/env bash
#
# Compilation-time reduction breakdown: orthogonal evaluation.
# Target: duckdb topdown query-jit, node-based split.
#
# ============================================================
# Figure A: Compiler backend (cache=off, spec=off, lookahead=off)
# ============================================================
#
# A1: LLVM O2 (baseline)
# A2: LLVM_O0 (LLVM O0 + FastISel)
# A3: TPDE (fastest backend)
#
# ============================================================
# Figure B: JIT cache mode (spec=off, lookahead=off)
# ============================================================
#
# B1-B4: TPDE backend
#   B1: off (no caching)
#   B2: single-run-strict (exact plan match)
#   B3: single-run-parameterized (parameterized: constants from params array)
#   B4: single-run-template
# B5-B8: LLVM_O0 backend (same cache modes as B1-B4)
# B9-B12: LLVM_4OPT backend (same cache modes as B1-B4)
#
# ============================================================
# Figure C: Latency hiding (cache=template)
# ============================================================
#
# C1: baseline (spec=off, lookahead=off) tpde
# C2: +lookahead-prep (spec=off, lookahead=on) tpde
# C3: +spec-jit +lookahead-prep (spec=recompile, lookahead=on) tpde
# C4: baseline (spec=off, lookahead=off) llvm_o0
# C5: +lookahead-prep (spec=off, lookahead=on) llvm_o0
# C6: +spec-jit +lookahead-prep (spec=recompile, lookahead=on) llvm_o0
#
# ============================================================
# Total configs to measure: 19 (A3=B1 shared)
# ============================================================
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DEST_DIR="${SCRIPT_DIR}/job_result"
mkdir -p "${DEST_DIR}"

# Shared args: job duckdb topdown query none on on on all
#   positional: bench engine split jit_level jit_simd payload_prune prefetch batch_probe skip_hash_cmp
BASE="job duckdb topdown query none on on on all"

echo "============================================"
echo "Figure A: Compiler backend comparison"
echo "============================================"

# A1: LLVM O2 
echo "--- A1: LLVM O2 ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_4opt "" "" ""
A1="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_4opt_breakdown_time_log.csv"
cp "$A1" "${DEST_DIR}/figA1_llvm_4opt.csv"

# A2: LLVM_O0
echo "--- A2: LLVM_O0 ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_o0 "" "" ""
A2="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_o0_breakdown_time_log.csv"
cp "$A2" "${DEST_DIR}/figA2_llvm_o0.csv"

# A3: TPDE
echo "--- A3: TPDE ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" ""
A3="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv"
cp "$A3" "${DEST_DIR}/figA3_tpde.csv"

echo ""
echo "============================================"
echo "Figure B: JIT cache mode (TPDE, spec=off)"
echo "============================================"

# B1: off
echo "--- B1: cache=off tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off tpde "" "" ""
B1="${DEST_DIR}/duckdb_topdown_query_none_tpde_breakdown_time_log.csv"
cp "$B1" "${DEST_DIR}/figB1_cache_off_tpde.csv"

# B2: single-run-strict
echo "--- B2: cache=strict tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off tpde "" "" ""
B2="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_tpde_breakdown_time_log.csv"
cp "$B2" "${DEST_DIR}/figB2_cache_strict_tpde.csv"

# B3: single-run-parameterized
echo "--- B3: cache=parameterized tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-parameterized off tpde "" "" ""
B3="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_parameterized_tpde_breakdown_time_log.csv"
cp "$B3" "${DEST_DIR}/figB3_cache_parameterized_tpde.csv"

# B4: single-run-template - Reuse A3
echo "--- B4: cache=template tpde ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" ""
B4="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv"
cp "$B4" "${DEST_DIR}/figB4_cache_template_tpde.csv"

# --- LLVM_O0 variants (B5-B8) ---

# B5: off
echo "--- B5: cache=off llvm_4opt_o0 ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off llvm_o0 "" "" ""
B5="${DEST_DIR}/duckdb_topdown_query_none_llvm_o0_breakdown_time_log.csv"
cp "$B5" "${DEST_DIR}/figB5_cache_off_llvm_o0.csv"

# B6: single-run-strict
echo "--- B6: cache=strict llvm_4opt_o0 ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off llvm_o0 "" "" ""
B6="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_llvm_o0_breakdown_time_log.csv"
cp "$B6" "${DEST_DIR}/figB6_cache_strict_llvm_o0.csv"

# B7: single-run-parameterized
echo "--- B7: cache=parameterized llvm_4opt_o0 ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-parameterized off llvm_o0 "" "" ""
B7="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_parameterized_llvm_o0_breakdown_time_log.csv"
cp "$B7" "${DEST_DIR}/figB7_cache_parameterized_llvm_o0.csv"

# B8: single-run-template
echo "--- B8: cache=template llvm_4opt_o0 ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_o0 "" "" ""
B8="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_o0_breakdown_time_log.csv"
cp "$B8" "${DEST_DIR}/figB8_cache_template_llvm_o0.csv"

# --- LLVM variants (B9-B12) ---

# B9: off
echo "--- B9: cache=off llvm_4opt ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off llvm_4opt "" "" ""
B9="${DEST_DIR}/duckdb_topdown_query_none_llvm_4opt_breakdown_time_log.csv"
cp "$B9" "${DEST_DIR}/figB9_cache_off_llvm_4opt.csv"

# B10: single-run-strict
echo "--- B10: cache=strict llvm_4opt ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off llvm_4opt "" "" ""
B10="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_llvm_4opt_breakdown_time_log.csv"
cp "$B10" "${DEST_DIR}/figB10_cache_strict_llvm_4opt.csv"

# B11: single-run-parameterized
echo "--- B11: cache=parameterized llvm_4opt ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-parameterized off llvm_4opt "" "" ""
B11="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_parameterized_llvm_4opt_breakdown_time_log.csv"
cp "$B11" "${DEST_DIR}/figB11_cache_parameterized_llvm_4opt.csv"

# B12: single-run-template
echo "--- B12: cache=template llvm_4opt ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_4opt "" "" ""
B12="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_4opt_breakdown_time_log.csv"
cp "$B12" "${DEST_DIR}/figB12_cache_template_llvm_4opt.csv"


echo ""
echo "============================================"
echo "Figure C: Latency hiding (cache=template)"
echo "============================================"

### TPDE (topdown)
# C1: no hiding (spec=off, lookahead=off)
echo "--- C1: no hiding tpde (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" "lookahead-prep"
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_nolookaheadprep_breakdown_time_log.csv" \
   "${DEST_DIR}/figC1_no_hiding_tpde.csv"

# C2: lookahead only (spec=off, lookahead=on)
echo "--- C2: lookahead only tpde (topdown) ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" ""
cp "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv" \
   "${DEST_DIR}/figC2_cross_only_tpde.csv"

# C3: spec+lookahead (spec=recompile, lookahead=on)
echo "--- C3: spec+lookahead tpde (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template recompile tpde "" "" ""
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_specrecompile_tpde_breakdown_time_log.csv" \
   "${DEST_DIR}/figC3_spec_and_cross_tpde.csv"

### Fastisel (topdown)
# C4: no hiding (spec=off, lookahead=off)
echo "--- C4: no hiding llvm_4opt_o0 (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_o0 "" "" "lookahead-prep"
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_o0_nolookaheadprep_breakdown_time_log.csv" \
   "${DEST_DIR}/figC4_no_hiding_llvm_o0.csv"

# C5: lookahead only (spec=off, lookahead=on)
echo "--- C5: lookahead only llvm_4opt_o0 (topdown) ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_o0 "" "" ""
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_o0_breakdown_time_log.csv" \
   "${DEST_DIR}/figC5_cross_only_llvm_o0.csv"

# C6: spec+lookahead (spec=recompile, lookahead=on)
echo "--- C6: spec+lookahead llvm_4opt_o0 (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template recompile llvm_o0 "" "" ""
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_specrecompile_llvm_o0_breakdown_time_log.csv" \
   "${DEST_DIR}/figC6_spec_and_cross_llvm_o0.csv"

### LLVM (topdown)
# C7: no hiding (spec=off, lookahead=off)
echo "--- C7: no hiding llvm_4opt (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_4opt "" "" "lookahead-prep"
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_4opt_nolookaheadprep_breakdown_time_log.csv" \
   "${DEST_DIR}/figC7_no_hiding_llvm_4opt.csv"

# C8: lookahead only (spec=off, lookahead=on)
echo "--- C8: lookahead only llvm_4opt (topdown) ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm_4opt "" "" ""
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_4opt_breakdown_time_log.csv" \
   "${DEST_DIR}/figC8_cross_only_llvm_4opt.csv"

# C9: spec+lookahead (spec=recompile, lookahead=on)
echo "--- C9: spec+lookahead llvm_4opt (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template recompile llvm_4opt "" "" ""
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_specrecompile_llvm_4opt_breakdown_time_log.csv" \
   "${DEST_DIR}/figC9_spec_and_cross_llvm_4opt.csv"

#node_based_BASE="job duckdb node-based query none on on on all"
#
#### LLVM (node-based)
## C4: no hiding (spec=off, lookahead=off)
#echo "--- C4: no hiding llvm_4opt (node-based) ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template off llvm_4opt "" "" "lookahead-prep"
#mv "${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_llvm_4opt_nolookaheadprep_breakdown_time_log.csv" \
#   "${DEST_DIR}/figC4_no_hiding_llvm.csv"
#
## C5: lookahead only (spec=off, lookahead=on)
#echo "--- C5: lookahead only llvm_4opt (node-based) ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template off llvm_4opt "" "" ""
##mv "${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_llvm_4opt_breakdown_time_log.csv"  "${DEST_DIR}/figC5_cross_only_llvm.csv"
#mv "$B12" "${DEST_DIR}/figC5_cross_only_llvm.csv"
#
## C6: spec+lookahead (spec=recompile, lookahead=on)
#echo "--- C6: spec+lookahead llvm_4opt (node-based) ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template recompile llvm_4opt "" "" ""
#mv "${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_specrecompile_llvm_4opt_breakdown_time_log.csv" \
#   "${DEST_DIR}/figC6_spec_and_cross_llvm.csv"

echo ""
echo "=== Compilation-time reduction breakdown complete ==="
echo "Output files:"
echo "  Figure A: ${DEST_DIR}/figA[1-3]_*.csv"
echo "  Figure B: ${DEST_DIR}/figB[1-12]_*.csv"
echo "  Figure C: ${DEST_DIR}/figC[1-9]_*.csv"

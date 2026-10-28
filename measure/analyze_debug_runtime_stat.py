#!/usr/bin/env python3
"""
Collect and analyze debug runtime-stat-behavior counters from JIT,
interpreter (no stats), and interpreter (collect-stats ON).

Runs debug_runtime_stat_behavior.sh first to collect data, then prints analysis.

Usage:
    python3 measure/analyze_debug_runtime_stat.py [query_filter]
    e.g.: python3 measure/analyze_debug_runtime_stat.py        # all JOB queries
          python3 measure/analyze_debug_runtime_stat.py 1a     # query 1a only
"""
import sys
import re
import os
import subprocess

def parse_log(path, tag):
    """Parse debug log. Each query starts with 'QUERY=xxx' line followed
    by N debug counter lines."""
    queries = {}
    current_query = None
    if not os.path.exists(path):
        return queries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('QUERY='):
                current_query = line.split('=', 1)[1]
                if current_query not in queries:
                    queries[current_query] = []
                continue
            m = re.search(r'\[AQP-DEBUG-RUNTIME-STAT\] ' + re.escape(tag) + r': (.+)', line)
            if m and current_query:
                counters = {}
                for kv in m.group(1).split():
                    k, v = kv.split('=')
                    counters[k] = int(v)
                queries[current_query].append(counters)
    return queries

def sum_counters(steps, keys):
    return {k: sum(s.get(k, 0) for s in steps) for k in keys}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'job_result', 'debug_runtime_stat')
    query_filter = sys.argv[1] if len(sys.argv) > 1 else ''

    collect_script = os.path.join(script_dir, 'debug_runtime_stat_behavior.sh')
    print(f"Running {collect_script} {query_filter} ...")
    rc = subprocess.call(['bash', collect_script, query_filter])
    if rc != 0:
        print(f"Collection script failed with exit code {rc}")
        sys.exit(rc)
    print()

    jit_log = os.path.join(log_dir, 'jit_debug.log')
    int_log = os.path.join(log_dir, 'interpreter_debug.log')
    int_stats_log = os.path.join(log_dir, 'interpreter_stats_debug.log')

    jit_queries = parse_log(jit_log, 'qjit')
    int_queries = parse_log(int_log, 'interpreter')
    int_stats_queries = parse_log(int_stats_log, 'interpreter')

    all_queries = sorted(set(jit_queries) | set(int_queries) | set(int_stats_queries),
                         key=lambda x: (len(x), x))

    jit_keys = ['rows_total', 'range_guard_filtered', 'membership_filtered',
                'bloom_filtered', 'blocks_checked', 'blocks_skipped']
    int_keys = ['rows_total', 'rows_range_guard_filtered', 'rows_existence_filtered',
                'rows_probed', 'chunks_total', 'chunks_block_skipped']

    hdr = (f"{'Query':<8} | "
           f"{'JIT probe':>12} {'JIT guard':>10} {'JIT membr':>10} {'JIT bloom':>10} | "
           f"{'INT rows':>12} {'INT guard':>10} {'INT exist':>10} {'INT probed':>10} | "
           f"{'INT+S rows':>12} {'INT+S grd':>10} {'INT+S ext':>10} {'INT+S pbd':>10}")
    print(hdr)
    print("-" * len(hdr))

    tot_jit = {k: 0 for k in jit_keys}
    tot_int = {k: 0 for k in int_keys}
    tot_int_s = {k: 0 for k in int_keys}

    for q in all_queries:
        js = sum_counters(jit_queries.get(q, []), jit_keys)
        ints = sum_counters(int_queries.get(q, []), int_keys)
        ints_s = sum_counters(int_stats_queries.get(q, []), int_keys)

        for k in jit_keys: tot_jit[k] += js[k]
        for k in int_keys: tot_int[k] += ints[k]
        for k in int_keys: tot_int_s[k] += ints_s[k]

        print(f"{q:<8} | "
              f"{js['rows_total']:>12,} {js['range_guard_filtered']:>10,} {js['membership_filtered']:>10,} {js['bloom_filtered']:>10,} | "
              f"{ints['rows_total']:>12,} {ints['rows_range_guard_filtered']:>10,} {ints['rows_existence_filtered']:>10,} {ints['rows_probed']:>10,} | "
              f"{ints_s['rows_total']:>12,} {ints_s['rows_range_guard_filtered']:>10,} {ints_s['rows_existence_filtered']:>10,} {ints_s['rows_probed']:>10,}")

    print("-" * len(hdr))
    print(f"{'TOTAL':<8} | "
          f"{tot_jit['rows_total']:>12,} {tot_jit['range_guard_filtered']:>10,} {tot_jit['membership_filtered']:>10,} {tot_jit['bloom_filtered']:>10,} | "
          f"{tot_int['rows_total']:>12,} {tot_int['rows_range_guard_filtered']:>10,} {tot_int['rows_existence_filtered']:>10,} {tot_int['rows_probed']:>10,} | "
          f"{tot_int_s['rows_total']:>12,} {tot_int_s['rows_range_guard_filtered']:>10,} {tot_int_s['rows_existence_filtered']:>10,} {tot_int_s['rows_probed']:>10,}")

    print()
    print("Legend: JIT = query-jit (all opts), INT = interpreter (no collect-stats), INT+S = interpreter (collect-stats ON)")
    print()

    def pct(num, den):
        return f"{num/den*100:.1f}%" if den > 0 else "N/A"

    jit_filt = tot_jit['range_guard_filtered'] + tot_jit['membership_filtered'] + tot_jit['bloom_filtered']
    int_filt = tot_int['rows_range_guard_filtered'] + tot_int['rows_existence_filtered']
    int_s_filt = tot_int_s['rows_range_guard_filtered'] + tot_int_s['rows_existence_filtered']

    print(f"JIT:   {tot_jit['rows_total']:>14,} probe rows, {jit_filt:>14,} filtered ({pct(jit_filt, tot_jit['rows_total'])})")
    print(f"INT:   {tot_int['rows_total']:>14,} probe rows, {int_filt:>14,} filtered ({pct(int_filt, tot_int['rows_total'])})")
    print(f"INT+S: {tot_int_s['rows_total']:>14,} probe rows, {int_s_filt:>14,} filtered ({pct(int_s_filt, tot_int_s['rows_total'])})")
    print()
    if tot_int['rows_total'] > 0 and tot_int_s['rows_total'] > 0:
        print(f"collect-stats ON reduces interpreter probe rows by {pct(tot_int['rows_total'] - tot_int_s['rows_total'], tot_int['rows_total'])}"
              f" ({tot_int['rows_total']:,} -> {tot_int_s['rows_total']:,})")

if __name__ == '__main__':
    main()

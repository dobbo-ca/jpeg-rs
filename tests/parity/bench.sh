#!/bin/sh
# Decode and encode time of the jpg/ fixtures, Go and Rust on the same inputs, as markdown tables
# per class, in two passes: tests/parity/bench.sh [MS]. Needs `oracle` (Go) and `bench` (Rust) on PATH;
# run from the repo root. MS is the time spent per operation and side (default 40). The Rust binary
# is $BENCH (default bench). When `oracle-v4` (Go built with GOAMD64=v4) and $BENCH_V4 (default
# bench-v4, Rust built with -C target-cpu=x86-64-v4 and the false-dependency flags of docs/performance.md) are on PATH too, each table gets three more
# columns for the v4 builds, and the output bytes of the v4 and default builds must be equal.
#
#   single-threaded  Go Workers 0 (its default, serial)  against Rust max_threads 1
#   default threads  Go Workers = GOMAXPROCS             against Rust max_threads = available_parallelism
#
# Only the dec-* classes have threads on either side; rowreader, coef and enc-* are serial in both,
# so their rows repeat between the passes. Each side prints the CPU count it sees on stderr.
set -eu
export LC_ALL=C
n=${1:-40}
bench=${BENCH:-bench}
bench4=${BENCH_V4:-bench-v4}
v4=0
if command -v oracle-v4 >/dev/null 2>&1 && command -v "$bench4" >/dev/null 2>&1 && grep -q avx512bw /proc/cpuinfo; then v4=1; fi
t=$(mktemp -d)
tab=$(printf '\t')
grep -m1 'model name' /proc/cpuinfo >&2 || true
grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|bmi2|avx512(f|bw|vl|_vnni))$' | tr '\n' ' ' >&2 || true
echo >&2
cpus=$(nproc 2>/dev/null || echo '?')
echo "CPUs: nproc $cpus"
echo >&2 "nproc $cpus (the Go and Rust lines below say what each side uses)"
d=tests/data
# Best of three runs per operation and side: the machine is shared, and one run can be slow.
best() {
  awk -F'\t' '!($1 in m) || $3 < m[$1] { m[$1] = $3; b[$1] = $2 } END { for (k in m) printf "%s\t%s\t%s\n", k, b[k], m[k] }' "$@" | sort
}

# pass LABEL GO_WORKERS RUST_THREADS
pass() {
  for i in 1 2 3; do
    oracle -bench "$n" -workers "$2" $d >"$t/go$i.tsv"
    "$bench" $d "$n" "" "$3" >"$t/rs$i.tsv"
    if [ "$v4" = 1 ]; then
      oracle-v4 -bench "$n" -workers "$2" $d >"$t/gov$i.tsv"
      "$bench4" $d "$n" "" "$3" >"$t/rsv$i.tsv"
    fi
  done
  best "$t"/go?.tsv >"$t/go.tsv"
  best "$t"/rs?.tsv >"$t/rs.tsv"
  # name, Go out bytes, Go ns, Rust out bytes, Rust ns. ns per byte is ms per MB. With the v4
  # builds: Go v4 out bytes, Go v4 ns, Rust v4 out bytes, Rust v4 ns after those.
  join -t "$tab" "$t/go.tsv" "$t/rs.tsv" >"$t/joined.tsv"
  if [ "$v4" = 1 ]; then
    best "$t"/gov?.tsv >"$t/gov.tsv"
    best "$t"/rsv?.tsv >"$t/rsv.tsv"
    join -t "$tab" "$t/joined.tsv" "$t/gov.tsv" | join -t "$tab" - "$t/rsv.tsv" >"$t/joined4.tsv"
    mv "$t/joined4.tsv" "$t/joined.tsv"
  fi
  # cmp_evil_q255 is left out of the tables (it has its own rows below, and is in the per-file lists
  # on stderr): Go's 16-bit SIMD kernels give other bytes there than Rust's 32-bit ones, which equal
  # the scalar code, so the two sides do different work.
  grep -v '/cmp_evil_q255' "$t/joined.tsv" | awk -F'\t' -v label="$1" -v v4="$v4" '
function add(k) { b[k] += $2; g[k] += $3; r[k] += $5; c[k]++; if (v4) { g4[k] += $7; r4[k] += $9 } }
function row(k, name) {
  s = sprintf("| %s | %d | %.2f | %.3f | %.3f | %.3f |", name, c[k], b[k] / 1e6, g[k] / b[k], r[k] / b[k], r[k] / g[k])
  if (v4) s = s sprintf(" %.3f | %.3f | %.3f |", g4[k] / b[k], r4[k] / b[k], r4[k] / g4[k])
  print s
}
function head(first) {
  s = "| " first " | files | MB out | Go ms/MB | Rust ms/MB | Rust / Go |"
  t = "|---|---|---|---|---|---|"
  if (v4) { s = s " Go v4 ms/MB | Rust v4 ms/MB | Rust v4 / Go v4 |"; t = t "---|---|---|" }
  print s; print t
}
{
  k = substr($1, 1, index($1, "/") - 1)
  add(k); add("all")
  if (!(k in seen)) { seen[k] = 1; order[++m] = k }
}
END {
  print "#### " label
  print ""
  head("class")
  for (i = 1; i <= m; i++) row(order[i], order[i])
  row("all", "all")
}'
  # The same per output size: the class totals are set by the few big images, so the 512x512 tiles
  # (about 1 MB of output) and smaller get their own rows.
  echo
  grep -v '/cmp_evil_q255' "$t/joined.tsv" | awk -F'\t' -v v4="$v4" '
function row(k) {
  s = sprintf("| %s | %d | %.2f | %.3f | %.3f | %.3f |", k, c[k], b[k] / 1e6, g[k] / b[k], r[k] / b[k], r[k] / g[k])
  if (v4) s = s sprintf(" %.3f | %.3f | %.3f |", g4[k] / b[k], r4[k] / b[k], r4[k] / g4[k])
  print s
}
{
  k = substr($1, 1, index($1, "/") - 1)
  z = ($2 < 1500000) ? "under 1.5 MB" : "1.5 MB and over"
  key = k " / " z
  b[key] += $2; g[key] += $3; r[key] += $5; c[key]++
  if (v4) { g4[key] += $7; r4[key] += $9 }
  if (!(key in seen)) { seen[key] = 1; order[++m] = key }
}
END {
  s = "| class / output size | files | MB out | Go ms/MB | Rust ms/MB | Rust / Go |"
  t = "|---|---|---|---|---|---|"
  if (v4) { s = s " Go v4 ms/MB | Rust v4 ms/MB | Rust v4 / Go v4 |"; t = t "---|---|---|" }
  print s; print t
  for (i = 1; i <= m; i++) row(order[i])
}'
  # cmp_evil_q255 on its own: a high-quantisation fixture whose IDCT leaves Go's 16-bit SIMD range
  # (Go's kernel gives other bytes there; Rust's 32-bit kernel gives the scalar's).
  echo
  grep '/cmp_evil_q255' "$t/joined.tsv" | awk -F'\t' -v v4="$v4" '
BEGIN {
  s = "| cmp_evil_q255 (Go gives other bytes here, not in the totals) | Go ms/MB | Rust ms/MB | Rust / Go |"
  t = "|---|---|---|---|"
  if (v4) { s = s " Go v4 ms/MB | Rust v4 ms/MB | Rust v4 / Go v4 |"; t = t "---|---|---|" }
  print s; print t
}
$3 > 0 && $2 > 0 {
  s = sprintf("| %s | %.3f | %.3f | %.3f |", $1, $3 / $2, $5 / $4, $5 / $3)
  if (v4) s = s sprintf(" %.3f | %.3f | %.3f |", $7 / $6, $9 / $8, $9 / $7)
  print s
}'
  # The v4 builds do the same work as the default ones: the output bytes of each file must be equal
  # (cmp_evil_q255 included: Go's SIMD choice does not depend on GOAMD64).
  if [ "$v4" = 1 ]; then
    awk -F'\t' '$6 != $2 || $8 != $4 { print "v4 output bytes differ from the default build: " $0; bad = 1 } END { exit bad }' "$t/joined.tsv" >&2
  fi
  # The classes Go has no counterpart for: scales 2 and 4 with and without DecodeOptions::pyramid,
  # Rust only (the join above drops them), ms per MB of output.
  echo
  grep -E '^dec-rgba[24](-pyr)?/' "$t/rs.tsv" | grep -v '/cmp_evil_q255' | awk -F'\t' '
{
  k = substr($1, 1, index($1, "/") - 1)
  b[k] += $2; r[k] += $3; c[k]++
  if (!(k in seen)) { seen[k] = 1; order[++m] = k }
}
END {
  print "| Rust only | files | MB out | Rust ms/MB |"
  print "|---|---|---|---|"
  for (i = 1; i <= m; i++) {
    k = order[i]
    printf "| %s | %d | %.2f | %.3f |\n", k, c[k], b[k] / 1e6, r[k] / b[k]
  }
}'
  # Per file: the Go and Rust ns and the ratio, the worst first (stderr, for reading the log).
  echo >&2 "== $1: worst files"
  awk -F'\t' '$3 > 0 { printf "%s\t%d\t%d\t%d\t%.2f\n", $1, $2, $3, $5, $5 / $3 }' "$t/joined.tsv" | sort -t"$tab" -k5 -nr | head -60 >&2
  # Every file slower than 1.10x Go (the list above is cut at 60), in the default build and, when
  # the v4 columns exist, in the v4 build (stderr).
  echo >&2 "== $1: all files over 1.10x Go"
  awk -F'\t' -v v4="$v4" '$3 > 0 && $5 / $3 > 1.10 { printf "%s\t%.2f\n", $1, $5 / $3 } v4 && $7 > 0 && $9 / $7 > 1.10 { printf "%s\tv4\t%.2f\n", $1, $9 / $7 }' "$t/joined.tsv" | sort >&2
  # The generated inputs (gen_*), always, whatever their rank (stderr).
  echo >&2 "== $1: generated inputs"
  awk -F'\t' '$1 ~ /\/gen_/ && $3 > 0 { printf "%s\t%d\t%d\t%d\t%.2f\n", $1, $2, $3, $5, $5 / $3 }' "$t/joined.tsv" | sort >&2
  # The big inputs (500 KB or more of output) all, which set the class totals (stderr).
  echo >&2 "== $1: big inputs"
  awk -F'\t' '$2 >= 500000 && $3 > 0 { printf "%s\t%d\t%d\t%d\t%.2f\n", $1, $2, $3, $5, $5 / $3 }' "$t/joined.tsv" | sort >&2
}

pass "single-threaded (Go Workers 0, Rust max_threads 1)" 0 1
echo
pass "default threads (Go Workers = GOMAXPROCS, Rust max_threads = available_parallelism)" -1 0
rm -rf "$t"

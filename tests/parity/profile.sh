#!/bin/sh
# tests/parity/profile.sh <filter>...: the bench example's operations whose class/fixture name contains
# each filter, under callgrind, and its report (instructions per function, inclusive and exclusive).
set -eu
for f in "$@"; do
  echo "=== $f"
  valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out bench tests/data 100 "$f" >/dev/null 2>/tmp/cg.err || { cat /tmp/cg.err; exit 1; }
  grep -E "Collected|Events" /tmp/cg.err || true
  callgrind_annotate --auto=yes --threshold=96 --context=1 /tmp/cg.out 2>&1 | head -450
done

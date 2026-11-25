#!/bin/sh
# tests/parity/diff.sh <go-out> <rust-out>: succeeds when the trees are equal; otherwise shows, per
# operation, how many files differ or are missing and the first two of each, with their first bytes.
a=$1
b=$2
diff -rq "$a" "$b" >/tmp/diff.txt && exit 0
echo "$(wc -l </tmp/diff.txt) differences"
# <input>.<op>[.partial].(out|err): the operation is the last dotted part of the name.
awk '
/^Files / { f = $2; sub(/.*\//, "", f); k = "differ" }
/^Only in / { f = $0; sub(/^Only in [^:]*: /, "", f); k = "only" }
{ g = f; sub(/\.partial\.out$/, "", g); sub(/\.(out|err)$/, "", g); n = split(g, p, "."); key = k ":" p[n]
  c[key]++; if (c[key] <= 2) ex[key] = ex[key] " " f }
END { for (key in c) printf "%7d %s  e.g.%s\n", c[key], key, ex[key] }' /tmp/diff.txt | sort -rn | head -60
sed -n 's/^Files \(.*\) and \(.*\) differ$/\1 \2/p' /tmp/diff.txt | awk '{ f = $1; sub(/\.partial\.out$/, "", f); sub(/\.(out|err)$/, "", f); n = split(f, p, "."); if (++c[p[n]] <= 2) print }' | head -40 | while read -r x y; do
  echo "== $x"
  echo "-- go:"; head -c 240 "$x" | od -An -c | head -5
  echo "-- rust:"; head -c 240 "$y" | od -An -c | head -5
done
exit 1

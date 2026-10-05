#!/usr/bin/env bash
set -euo pipefail

usage(){
    echo "usage: $0 --bin PATH --data DIR --out DIR --dim D --procs N --top-k K [--max-queries Q]" >&2
    exit 2
}

limit=()
while (($#)); do
    (($# >= 2)) || usage
    case $1 in
        --bin) bin=$2 ;;
        --data) data=$2 ;;
        --out) out=$2 ;;
        --dim) dim=$2 ;;
        --procs) procs=$2 ;;
        --top-k) top_k=$2 ;;
        --max-queries) limit=(--max-queries "$2") ;;
        *) usage ;;
    esac
    shift 2
done
[[ -n ${bin:-} && -n ${data:-} && -n ${out:-} && -n ${dim:-} && -n ${procs:-} && -n ${top_k:-} ]] || usage

mkdir -p "$out"
rm -f "$out"/shard-*.run
timing=$out/timing.jsonl
exec 2> "$timing"
trap 'echo "search.sh failed; see $timing:"; tail -n 3 "$timing"' ERR

"$bin" setup --dim "$dim" --msk "$out/msk.bin"
"$bin" encrypt --msk "$out/msk.bin" --vectors "$data/docs.i8v" --out "$out/corpus.ct"
"$bin" keygen --msk "$out/msk.bin" --vectors "$data/queries.i8v" --out "$out/queries.sk" "${limit[@]}"

start=$(date +%s.%N)
pids=()
shards=()
for ((i = 0; i < procs; i++)); do
    shards+=("$out/shard-$i.run")
    "$bin" search --corpus "$out/corpus.ct" --keys "$out/queries.sk" --top-k "$top_k" --shard "$i/$procs" \
        --out "$out/shard-$i.run" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
end=$(date +%s.%N)
echo "{\"command\":\"parallel_search\",\"procs\":$procs,\"seconds\":$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e - s}')}" >&2

"$bin" merge --top-k "$top_k" --out "$out/encrypted.run" "${shards[@]}"
"$bin" plain --docs "$data/docs.i8v" --queries "$data/queries.i8v" --top-k "$top_k" --out "$out/plain.run" "${limit[@]}"
cmp "$out/plain.run" "$out/encrypted.run"
echo "runs match"

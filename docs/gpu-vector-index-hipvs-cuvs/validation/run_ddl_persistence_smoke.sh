#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER=${1:?isolated container name required}
BASE_TSV=${2:?container TSV path required}
HOST_RUN=${3:?host runtime artifact directory required}
DATABASE=${4:-p0_persistence}
[[ "$DATABASE" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
mkdir -p "$HOST_RUN/p0-evidence"
EVIDENCE=$HOST_RUN/p0-evidence
trap 'echo "FAIL line=$LINENO artifacts=$EVIDENCE" >&2' ERR
assert_absent() {
  if grep "$@"; then
    echo "unexpected match: $*" >&2
    return 1
  else
    local status=$?
    [[ "$status" -eq 1 ]]
  fi
}
sql() {
  docker exec "$CONTAINER" mysql --local-infile=1 -h127.0.0.1 -P2881 -uroot -ANB \
    -e "set ob_query_timeout=300000000; $1"
}
ready() {
  local deadline=$((SECONDS + 180))
  until sql 'select 1' > /dev/null 2>&1; do
    ((SECONDS < deadline)) || return 1
  done
}
ready
sql "create database if not exists $DATABASE; use $DATABASE;
drop table if exists items;
create table items(id bigint primary key, v vector(128));
load data local infile '$BASE_TSV' into table items
fields terminated by '\\t' lines terminated by '\\n' (id,v);"
count=$(sql "select count(*) from $DATABASE.items")
((count >= 10000)) || { echo "need at least 10000 pre-DDL rows, got $count" >&2; exit 1; }
probe=$(docker exec "$CONTAINER" head -n 1 "$BASE_TSV" | cut -f2)
offset=$(wc -c < "$HOST_RUN/log/seekdb.log")
trace_offset=$(wc -c < "$HOST_RUN/p0-trace.log")
sql "use $DATABASE; create vector index idx on items(v)
with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64);"
sql "select id from $DATABASE.items order by l2_distance(v,$probe) approximate limit 10" > "$EVIDENCE/first.ids"
sql "select id from $DATABASE.items order by l2_distance(v,$probe) limit 10" > "$EVIDENCE/exact.ids"
[[ $(wc -l < "$EVIDENCE/first.ids") -eq 10 ]]
diff -u "$EVIDENCE/exact.ids" "$EVIDENCE/first.ids"
tail -c +$((trace_offset + 1)) "$HOST_RUN/p0-trace.log" > "$EVIDENCE/first.trace"
grep -q cuvs_serve "$EVIDENCE/first.trace"
sql "use $DATABASE; drop table if exists cpu_items;
create table cpu_items(id bigint primary key, v vector(128));
load data local infile '$BASE_TSV' into table cpu_items
fields terminated by '\\t' lines terminated by '\\n' (id,v);
create vector index idx_cpu on cpu_items(v)
with (distance=l2,type=hnsw,lib=vsag,m=16,ef_construction=200,ef_search=200);"
restart_trace_offset=$(wc -c < "$HOST_RUN/p0-trace.log")
docker restart "$CONTAINER" > /dev/null
ready
sql "select id from $DATABASE.items order by l2_distance(v,$probe) approximate limit 10" > "$EVIDENCE/restart.ids"
diff -u "$EVIDENCE/exact.ids" "$EVIDENCE/restart.ids"
tail -c +$((restart_trace_offset + 1)) "$HOST_RUN/p0-trace.log" > "$EVIDENCE/restart.trace"
grep -q cuvs_serve "$EVIDENCE/restart.trace"
assert_absent -q cuvs_build "$EVIDENCE/restart.trace"
cpu_trace_offset=$(wc -c < "$HOST_RUN/p0-trace.log")
sql "select id from $DATABASE.cpu_items order by l2_distance(v,$probe) approximate limit 10" > "$EVIDENCE/cpu-restart.ids"
diff -u "$EVIDENCE/exact.ids" "$EVIDENCE/cpu-restart.ids"
tail -c +$((cpu_trace_offset + 1)) "$HOST_RUN/p0-trace.log" > "$EVIDENCE/cpu-restart.trace"
grep -q fdeserialize "$EVIDENCE/cpu-restart.trace"
assert_absent -q cuvs_ "$EVIDENCE/cpu-restart.trace"
tail -c +$((offset + 1)) "$HOST_RUN/log/seekdb.log" > "$EVIDENCE/lifecycle.log"
grep -a 'serialized GPU CAGRA snapshot' "$EVIDENCE/lifecycle.log" > "$EVIDENCE/gpu.log"
grep -aq 'built=true' "$EVIDENCE/gpu.log"
grep -aq 'has_graph=true' "$EVIDENCE/gpu.log"
assert_absent -aq 'lazily built VSAG fallback' "$EVIDENCE/lifecycle.log"
first_id=$(docker exec "$CONTAINER" head -n 1 "$BASE_TSV" | cut -f1)
[[ "$first_id" =~ ^-?[0-9]+$ ]]
sql "delete from $DATABASE.items where id=$first_id"
sql "select id from $DATABASE.items order by l2_distance(v,$probe) approximate limit 10" > "$EVIDENCE/deleted.ids"
sql "select id from $DATABASE.items order by l2_distance(v,$probe) limit 10" > "$EVIDENCE/deleted-exact.ids"
diff -u "$EVIDENCE/deleted-exact.ids" "$EVIDENCE/deleted.ids"
assert_absent -qx -- "$first_id" "$EVIDENCE/deleted.ids"
sql "insert into $DATABASE.items values(900000000000,$probe)"
sql "select id from $DATABASE.items order by l2_distance(v,$probe) approximate limit 10" > "$EVIDENCE/inserted.ids"
sql "select id from $DATABASE.items order by l2_distance(v,$probe) limit 10" > "$EVIDENCE/inserted-exact.ids"
diff -u "$EVIDENCE/inserted-exact.ids" "$EVIDENCE/inserted.ids"
grep -qx '900000000000' "$EVIDENCE/inserted.ids"
printf 'PASS pre_ddl_rows=%s first_rows=10 restart_rows=10 exact_match=1 graph_reload=1 gpu_serve=1 rebuild_after_restart=0 eager_vsag=0 cpu_reload=1 stale_ids=0\n' "$count" | tee "$EVIDENCE/result.txt"

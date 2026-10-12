#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/scratch_selector.sh"
testroot=$(mktemp -d "/dev/shm/${USER}_scratchselector_XXXXXX")
printf '%s\n' "$USER" > "$testroot/.test_owner"
cleanup_test() {
  [[ -d "$testroot" && ! -L "$testroot" && "$(cat "$testroot/.test_owner")" == "$USER" ]] || return
  chmod u+rwx "$testroot/blocked" 2>/dev/null || true
  rm -rf -- "$testroot"
}
trap cleanup_test EXIT
mkdir "$testroot/blocked" "$testroot/writable"
chmod 500 "$testroot/blocked"
export SLURM_JOB_ID="selector_test_$$" SLURM_MEM_PER_NODE=131072
scratch_free_kb() { printf '%s\n' 400000000; }
scratch_fs_type() { printf '%s\n' ext2; }
# Reproduce mkdir denial before verifying the corrected fallback.
if select_expanded_scratch "$testroot/blocked"; then echo 'FAIL nonwritable selected'; exit 1; fi
select_expanded_scratch "$testroot/blocked" "$testroot/writable"
[[ "$EXPANDED_SCRATCH" == "$testroot/writable/${USER}_nndmd_expanded_${SLURM_JOB_ID}" && "$EXPANDED_CACHE_MODE" == full101 ]]
[[ "$(cat "$EXPANDED_SCRATCH/.owner")" == "$USER:$SLURM_JOB_ID" ]]
printf 'PASS nonwritable rejection and actual mkdir/write fallback\n'
# tmpfs must stay rolling even with >350GB free, due to the128GB grant.
mkdir "$testroot/tmpfs"
scratch_fs_type() { printf '%s\n' tmpfs; }
select_expanded_scratch "$testroot/tmpfs"
[[ "$EXPANDED_CACHE_MODE" == rolling13 ]]
printf 'PASS tmpfs128GB never full101\n'
mkdir "$testroot/lowmem"
export SLURM_MEM_PER_NODE=32768
if select_expanded_scratch "$testroot/lowmem"; then echo 'FAIL insufficient RAM'; exit 1; fi
printf 'PASS insufficient tmpfs RAM rejected\n'

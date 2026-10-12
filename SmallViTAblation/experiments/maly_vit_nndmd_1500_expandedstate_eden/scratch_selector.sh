#!/bin/bash
# Single owner, no changes to other users' directories or permissions.
scratch_free_kb() { df -Pk "$1" | awk 'NR==2 {print $4}'; }
scratch_fs_type() { stat -f -c %T "$1"; }
select_expanded_scratch() {
  local candidate base free type mode dest probe memory
  memory=${SLURM_MEM_PER_NODE:-131072}
  for candidate in "$@"; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    base=$(readlink -f -- "$candidate") || continue
    case "$base" in /home*|/mnt/evafs*|/tmp) continue ;; esac
    type=$(scratch_fs_type "$base") || continue
    case "$type" in nfs*|lustre) continue ;; esac
    free=$(scratch_free_kb "$base") || continue
    [[ "$free" =~ ^[0-9]+$ ]] || continue
    if [[ "$type" == tmpfs ]]; then
      # Never allocate the 279GB full cache within a 128GB Slurm RAM grant.
      [[ "$free" -gt 50000000 && "$memory" -ge 65536 ]] || continue
      mode=rolling13
    elif [[ "$free" -gt 350000000 ]]; then mode=full101
    elif [[ "$free" -gt 50000000 ]]; then mode=rolling13
    else continue; fi
    dest="$base/${USER}_nndmd_expanded_${SLURM_JOB_ID}"
    # mkdir without -p: never claim or remove a pre-existing directory.
    if ! mkdir -- "$dest" 2>/dev/null; then
      echo "scratch skip: cannot mkdir own directory under $base" >&2; continue
    fi
    probe="$dest/.write_probe"
    if ! printf '%s\n' "$USER:$SLURM_JOB_ID" > "$probe" 2>/dev/null; then
      rmdir -- "$dest" 2>/dev/null || true; continue
    fi
    mv -- "$probe" "$dest/.owner"
    export EXPANDED_SCRATCH="$dest" EXPANDED_CACHE_MODE="$mode"
    export EXPANDED_CACHE="$dest/cache"
    mkdir -- "$EXPANDED_CACHE"
    export EXPANDED_SCRATCH_OWNER_JOB="$SLURM_JOB_ID"
    echo "scratch selected: root=$base fs=$type free_kb=$free memory_mb=$memory mode=$mode own_path=$dest"
    return 0
  done
  echo 'No writable node-local scratch with sufficient capacity' >&2
  return 2
}

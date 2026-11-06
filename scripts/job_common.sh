#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/project_env.sh"

asr_job() {
    : "${SLURM_JOB_ID:?GPU workloads require SLURM_JOB_ID on a compute node}"
    : "${ASR_DATA_ROOT:?Set ASR_DATA_ROOT to the directory containing train/ and public_test/}"
    case ${HOSTNAME:-$(hostname)} in login*) echo 'GPU execution is forbidden on login nodes' >&2; return 2;; esac
    local command=$1; shift
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    local python="$ASR_PROJECT_ROOT/.venv/bin/python"
    test -x "$python" || { echo 'Run scripts/setup_env.sh first' >&2; return 2; }
    local step_pid interrupted_wait=0 status=0
    # B: signals reach this shell; srun synchronously forwards USR1 to its Python task.
    # Do not use srun --async: it cannot forward signals or propagate task exit status.
    srun --ntasks=1 --unbuffered "$python" -m asr "$command" \
         --config "$ASR_PROJECT_ROOT/configs/official.yaml" --data-root "$ASR_DATA_ROOT" "$@" &
    step_pid=$!
    trap 'interrupted_wait=1; kill -USR1 "$step_pid" 2>/dev/null || true' USR1
    trap 'interrupted_wait=1; kill -TERM "$step_pid" 2>/dev/null || true' TERM
    trap 'interrupted_wait=1; kill -INT "$step_pid" 2>/dev/null || true' INT
    while true; do
        interrupted_wait=0
        if wait "$step_pid"; then status=0; else status=$?; fi
        if ((interrupted_wait)); then continue; fi
        break
    done
    trap - USR1 TERM INT
    return "$status"
}

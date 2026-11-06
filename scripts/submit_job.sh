#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n ${ASR_SLURM_ENV:-} ]]; then
    # An explicitly selected operator-owned shell configuration, never a dataset file.
    source "$ASR_SLURM_ENV"
fi
source "$script_dir/project_env.sh"
kind=${1:?Usage: submit_job.sh preflight|smoke|train|resume|export|infer [CLI arguments]}
shift
case "$kind" in
    preflight|smoke|export|infer) script="$script_dir/$kind.slurm"; extra=();;
    train|resume) script="$script_dir/train.slurm"; extra=("$kind");;
    *) echo "Unknown lifecycle command: $kind" >&2; exit 2;;
esac
: "${ASR_ACCOUNT:?Set ASR_ACCOUNT from verified cluster settings}"
: "${ASR_PARTITION:?Set ASR_PARTITION from verified cluster settings}"
: "${ASR_GRES:?Set ASR_GRES for one H200 MIG 2g.35gb instance}"
: "${ASR_CPUS:?Set ASR_CPUS}"
: "${ASR_MEM:?Set ASR_MEM}"
: "${ASR_TIME:?Set ASR_TIME}"
: "${ASR_SIGNAL_SECONDS:?Set ASR_SIGNAL_SECONDS after measuring window plus checkpoint time}"
[[ $ASR_CPUS =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid ASR_CPUS' >&2; exit 2; }
[[ $ASR_SIGNAL_SECONDS =~ ^[1-9][0-9]*$ ]] && ((ASR_SIGNAL_SECONDS<=65535)) || { echo 'Invalid ASR_SIGNAL_SECONDS' >&2; exit 2; }
[[ $ASR_GRES == *:1 ]] || { echo 'ASR_GRES must request exactly one GPU instance' >&2; exit 2; }
logs="$ASR_PROJECT_ROOT/runs/slurm"
mkdir -p -- "$logs"
resources=(--nodes=1 --ntasks=1 "--account=$ASR_ACCOUNT" "--partition=$ASR_PARTITION"
           "--gres=$ASR_GRES" "--cpus-per-task=$ASR_CPUS" "--mem=$ASR_MEM" "--time=$ASR_TIME"
           "--signal=B:USR1@$ASR_SIGNAL_SECONDS" "--chdir=$ASR_PROJECT_ROOT"
           "--job-name=asr-$kind" "--output=$logs/%x-%j.out" "--error=$logs/%x-%j.err" --open-mode=append --export=ALL)
if [[ -n ${ASR_QOS:-} ]]; then resources+=("--qos=$ASR_QOS"); fi
exec sbatch "${resources[@]}" "$script" "${extra[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
source "${ASR_PROJECT_ROOT:?Set ASR_PROJECT_ROOT}/scripts/job_common.sh"
asr_job infer-train "$@"

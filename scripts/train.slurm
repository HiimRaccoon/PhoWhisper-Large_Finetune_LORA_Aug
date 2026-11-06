#!/usr/bin/env bash
set -euo pipefail
source "${ASR_PROJECT_ROOT:?Set ASR_PROJECT_ROOT}/scripts/job_common.sh"
command=${1:?Expected train or resume}; shift
case "$command" in train|resume) ;; *) echo 'Expected train or resume' >&2; exit 2;; esac
asr_job "$command" "$@"

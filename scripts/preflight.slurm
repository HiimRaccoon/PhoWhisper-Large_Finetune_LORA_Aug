#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/job_common.sh"
asr_job preflight --stage gpu "$@"

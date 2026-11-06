#!/usr/bin/env bash
# Sourced by setup and allocated jobs. All managed writable paths stay in the project.
set -euo pipefail
: "${ASR_PROJECT_ROOT:?Set ASR_PROJECT_ROOT to the absolute project directory}"
cd -- "$ASR_PROJECT_ROOT"
ASR_PROJECT_ROOT=$(pwd -P)
export ASR_PROJECT_ROOT
case "/${ASR_PROJECT_ROOT,,}/" in */cs406/*) echo 'CS406 is outside the ASR scope' >&2; exit 2;; esac
test -f "$ASR_PROJECT_ROOT/configs/official.yaml" || { echo 'Missing project configuration' >&2; exit 2; }
export HF_HOME="$ASR_PROJECT_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ASR_PROJECT_ROOT/.cache"
export TORCH_HOME="$ASR_PROJECT_ROOT/.cache/torch"
export PIP_CACHE_DIR="$ASR_PROJECT_ROOT/.cache/pip"
export TMPDIR="$ASR_PROJECT_ROOT/.cache/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYTHONPYCACHEPREFIX="$ASR_PROJECT_ROOT/.cache/pycache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p -- "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"

#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/project_env.sh"
: "${ASR_TORCH_INDEX_URL:?Set the official PyTorch wheel index compatible with the target driver}"
python311=${ASR_PYTHON311:-python3.11}
"$python311" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 is required"'
"$python311" -m venv "$ASR_PROJECT_ROOT/.venv"
python="$ASR_PROJECT_ROOT/.venv/bin/python"
"$python" -m pip install --upgrade pip
"$python" -m pip install --index-url "$ASR_TORCH_INDEX_URL" 'torch==2.9.0'
"$python" -m pip install -e "$ASR_PROJECT_ROOT[train,test]"
"$python" -m pip check
mkdir -p "$ASR_PROJECT_ROOT/requirements"
"$python" -m pip freeze > "$ASR_PROJECT_ROOT/requirements/candidate-resolved-linux.txt"
echo 'Candidate environment installed. UIT GPU compatibility and uit-lock.txt remain T13 verification steps.'

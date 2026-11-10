#!/usr/bin/env bash
# Full pipeline. Requires CFBD_API_KEY in the environment or in a local .env
# (download.py reads the key; it is never printed or written to disk).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  PY=python
else
  PY="${PYTHON:-python3}"
fi

mkdir -p data/raw data/processed data/derived docs figures

$PY src/download.py      # cached: skips valid existing raw files
$PY src/clean.py
$PY src/model.py > data/processed/model_stdout.txt
$PY src/sensitivity.py > data/processed/sensitivity_stdout.txt 2>/dev/null
$PY src/report.py
$PY src/defense.py && $PY src/defense_report.py

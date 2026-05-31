#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/pillar-lense" ]]; then
  echo "PillarLense is not installed in ${VENV_DIR}. Run ./scripts/setup_venv.sh first." >&2
  exit 1
fi

exec "${VENV_DIR}/bin/pillar-lense" "$@"

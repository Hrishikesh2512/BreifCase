#!/usr/bin/env bash
# Launch Briefcase. Creates a venv and installs deps on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -e .
fi

[ -f .env ] && set -a && . ./.env && set +a

HOST="${BRIEFCASE_HOST:-127.0.0.1}"
PORT="${BRIEFCASE_PORT:-8000}"
echo "Briefcase running at http://${HOST}:${PORT}"
exec ./.venv/bin/python -m briefcase.cli --host "$HOST" --port "$PORT"

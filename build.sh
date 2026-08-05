#!/bin/bash
# Thin wrapper, kept because the README documented this entry point before CI
# needed to build the same module on Windows. `build.py` is the real script and
# takes the same arguments on every OS; this only chooses an interpreter.
set -e

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

exec "$PYTHON" "$(dirname "$0")/build.py" --clean "$@"

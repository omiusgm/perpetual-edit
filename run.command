#!/bin/zsh
set -e

cd "$(dirname "$0")"

zsh tools/ensure_venv.zsh

.venv/bin/python tools/download_models.py

echo "Starting PERPETUAL//EDIT..."
echo "If macOS asks for Camera access, allow it and restart this launcher."
echo

.venv/bin/python app.py

echo
read -k 1 "?Application closed. Press any key..."
echo

#!/bin/zsh
set -e

cd "$(dirname "$0")"

zsh tools/ensure_venv.zsh

echo "Starting local OBJECT -> MEME rehearsal..."
.venv/bin/python app.py --association-demo --mode smart

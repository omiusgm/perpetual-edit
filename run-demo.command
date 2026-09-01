#!/bin/zsh
set -e

cd "$(dirname "$0")"

zsh tools/ensure_venv.zsh

.venv/bin/python tools/download_models.py
.venv/bin/python app.py --demo --mode smart

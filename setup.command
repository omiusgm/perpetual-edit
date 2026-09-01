#!/bin/zsh
set -e

cd "$(dirname "$0")"

echo "CONFIDENCE BOOSTER // SETUP"
echo "Creating an isolated Python environment..."

zsh tools/ensure_venv.zsh
.venv/bin/python tools/download_models.py --required
.venv/bin/python -m unittest discover -s tests -v

echo
echo "SETUP COMPLETE"
echo "Now double-click run.command"
echo
read -k 1 "?Press any key to close..."
echo

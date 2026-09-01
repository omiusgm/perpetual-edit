#!/bin/zsh
set -e

cd "$(dirname "$0")"

zsh tools/ensure_venv.zsh

mkdir -p test-output
.venv/bin/python tools/download_models.py --required
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python app.py --demo --mode smart --headless --max-seconds 30 --output test-output/smart-story.mp4 --mute
.venv/bin/python app.py --association-demo --headless --max-seconds 6 --output test-output/object-meme-popup.mp4 --mute --no-save-edits
.venv/bin/python app.py --demo --headless --no-auto --script-preset crunch --script-at 20 --max-seconds 31 --output test-output/aura-morph.mp4 --mute
.venv/bin/python app.py --demo --headless --no-auto --script-preset nunca --script-at 20 --max-seconds 39 --output test-output/nunca-stutter.mp4 --mute
.venv/bin/python app.py --demo --headless --no-auto --script-preset portal --script-at 20 --max-seconds 34 --output test-output/portal-throw.mp4 --mute
.venv/bin/python app.py --demo --headless --no-auto --script-preset glass --script-at 20 --max-seconds 35 --output test-output/glass-cities.mp4 --mute

echo
echo "Self-test completed: GESTURE ARCADE, SMART STORY, OBJECT -> MEME popup and 4 reference styles rendered"
read -k 1 "?Press any key to close..."
echo

# PERPETUAL//EDIT

Local-first webcam meme camera that turns gestures, expressions and the last
20 seconds of movement into beat-synced vertical edits.

![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![macOS](https://img.shields.io/badge/platform-macOS-black)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

- Keeps a rolling 20-second camera buffer in RAM.
- Detects face geometry and sustained hand gestures locally with MediaPipe.
- Shows a passive cat reaction card on the right; it reacts only to a real,
  stable signal and never cycles on an idle timer.
- Selects several distinct moments and builds a `540×960` Reels/Shorts edit.
- Uses genuinely different montage grammars: hard cuts, velocity ramps,
  wave/surge motion, portals, boomerangs, cutouts, whip-pans and face warp.
- Pauses capture during replay, then clears the used buffer. Your reaction to
  watching an edit cannot leak into the next edit.
- Never repeats the same track twice in a row.
- Runs the normal face/gesture pipeline without cloud uploads.

The wide interface keeps the live camera on the left and the vertical result
on the right. Finished edits are saved to `captures/edits/`.

## Quick start (macOS)

Requirements: macOS, a webcam, Python 3.11 and an internet connection for the
first dependency/model download.

```bash
git clone https://github.com/omiusgm/perpetual-edit.git
cd perpetual-edit
chmod +x *.command tools/*.zsh
./setup.command
./run.command
```

You can also double-click `setup.command` and then `run.command` in Finder.
When macOS asks for camera access, allow Terminal and restart the launcher.

Demo without a physical camera:

```bash
./run-demo.command
```

## Управление

Главное управление доступно с клавиатуры:

| Key | Action |
|---|---|
| `1`–`0`, `Z X C B L J K P I S W [ E` | launch a specific edit grammar |
| `A` | automatic gesture trigger on/off |
| `T` | staged countdown on/off |
| `V` | cycle `classic → smart → kiosk` and start a clean buffer |
| `Enter` | remix the immutable previous action with another style/track |
| `R` | record the complete wide UI |
| `M` | edit music on/off |
| `N` | quiet analyzer sounds on/off (off by default) |
| `H` | HUD on/off |
| `F` | fullscreen |
| `Q` / `Esc` | quit |

The visible `ACT / FREE / CULT / RAND` switches change the director's source
pool. During tail capture, rendering and playback they are locked so the edit
cannot mutate underneath the user.

## Modes

- `smart` — default home mode; watches natural gestures and expressions and
  can trigger a short story from a stable action.
- `classic` — face + hands, manual presets and timer; the simplest path.
- `kiosk` — records nothing until a fresh opt-in gesture starts a one-shot
  session; intended for installations and events.

Object association, Apple Vision semantics and a Qwen text-only director are
kept as optional experimental paths and are disabled by default.

## Reaction cards

The repository contains a small original, non-AI starter pack drawn from
OpenCV primitives. It exists only so the app works immediately after cloning.

For actual memes, place media you are allowed to use under
`assets/local/reactions/`, copy `config.json` to the ignored
`config.local.json`, and update `gesture_playground.idle_image` and each
gesture's `images` list. Launch the local configuration with:

```bash
.venv/bin/python app.py --config config.local.json
```

Images are rendered with `contain`, so portrait and landscape memes are not
cropped. Missing reaction files are skipped safely.

## Music

Commercial tracks and audio extracted from Reels are intentionally not
included. The checked-in config points to ignored paths under
`assets/local/music/`; when a file is absent, the app generates its original
procedural soundtrack and still renders the full edit.

To use your own legally obtained audio, create `assets/local/music/` and place
files there using the configured names, or change `music_file` in your ignored
`config.local.json`. The last-track and visual-variety guards continue to work
through each preset's `track_id` and `visual_signature`.

## Privacy model

- Ordinary face, hand and gesture analysis stays on the local machine.
- The rolling buffer is held in RAM and cleared after replay or shutdown.
- New capture is frozen while an edit is playing.
- Saved camera recordings live only in ignored local directories.
- Qwen integration is off by default and receives normalized text tokens, not
  camera frames, when explicitly configured.
- `tools/download_reel.py` is optional. `--cookies-from-browser` asks yt-dlp to
  read an existing browser session and may trigger a macOS Keychain prompt; it
  is never used by the main app.

This is an entertainment prototype, not identity, emotion, ethnicity or
psychological inference. Mood labels are jokes attached to simple geometry.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -q
./self-test.command
```

The unit suite covers rolling-buffer lifecycle, replay isolation, non-repeating
tracks, gesture hold/release/freshness, async perception staleness, responsive
layout and montage grammar. Headless tests do not prove that a particular
physical camera and macOS permission setup work; validate those separately.

Project layout:

```text
app.py                    application loop and state orchestration
booster/                  perception, director, buffer, HUD and edit engine
config.json               public safe defaults and montage timing grammar
tools/                    model downloader, reel helper and Swift helpers
tests/                    deterministic unit tests
assets/memes/starter/     original procedural starter cards
captures/                 ignored local output
runtime/                  ignored caches and generated audio
```

## Media and license

Source code and procedural starter cards are MIT licensed. Downloaded Reels,
commercial music, movie clips, third-party meme packs and personal webcam
captures are deliberately excluded. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contributions are welcome. Please keep new default assets original or clearly
redistributable, keep face/gesture processing local by default, and add tests
for changes to capture/replay state.

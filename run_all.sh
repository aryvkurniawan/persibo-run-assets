#!/usr/bin/env bash
# =====================================================================
# Run the WHOLE pipeline end-to-end in ONE command and produce the
# single final video the user sees:  outputs/<video>_final_with_minimap.mp4
#
# Phases (all in this one run):
#   1 Detection  ->  2 Tracking  ->  3 Role refinement
#   -> apply your saved manual role corrections (if the file exists)
#   -> pitch homography + minimap (if calibration exists)
#   -> combined final video (team-colored boxes + roles + minimap overlay)
#
# The manual one-time files are REUSED if present; if either is missing the
# run still completes safely (auto roles / minimap skipped) — never crashes.
#
# Usage (from inside the football_ai folder):
#   ./run_all.sh                       # input_video.mp4, auto device
#   ./run_all.sh my_match.mp4          # a different video
#   ./run_all.sh my_match.mp4 cuda:0   # force GPU
# =====================================================================
set -euo pipefail

VIDEO="${1:-input_video.mp4}"
DEVICE="${2:-auto}"
STEM="$(basename "${VIDEO%.*}")"

CORRECTIONS="outputs/manual_role_corrections.json"
CALIBRATION="outputs/pitch_calibration.json"

echo "=== Running the WHOLE project on: ${VIDEO} (device=${DEVICE}) ==="
echo "  (missing calibration / corrections will be offered interactively)"

# --all runs every phase in one go and prompts for the human-in-the-loop
# steps (pitch calibration, role review) only when they're missing.
python main.py --video "${VIDEO}" --all --device "${DEVICE}"

echo ""
echo "=== DONE ==="
echo "Final video:  outputs/${STEM}_final_with_minimap.mp4"

# =====================================================================
# Run the WHOLE pipeline end-to-end in ONE command (PowerShell) and
# produce the single final video:  outputs\<video>_final_with_minimap.mp4
#
# Phases (all in this one run):
#   1 Detection -> 2 Tracking -> 3 Role refinement
#   -> apply saved manual role corrections (if the file exists)
#   -> pitch homography + minimap (if calibration exists)
#   -> combined final video (team-colored boxes + roles + minimap overlay)
#
# Missing manual files are skipped safely (auto roles / no minimap), never crash.
#
# Usage (from inside the football_ai folder):
#   .\run_all.ps1                      # input_video.mp4, auto device
#   .\run_all.ps1 my_match.mp4         # a different video
#   .\run_all.ps1 my_match.mp4 cuda:0  # force GPU
# =====================================================================
param(
    [string]$Video = "input_video.mp4",
    [string]$Device = "auto"
)

$Stem = [System.IO.Path]::GetFileNameWithoutExtension($Video)
$Corrections = "outputs/manual_role_corrections.json"
$Calibration = "outputs/pitch_calibration.json"

Write-Host "=== Running the WHOLE project on: $Video (device=$Device) ==="
Write-Host "  (missing calibration / corrections will be offered interactively)"

# --all runs every phase in one go and prompts for the human-in-the-loop
# steps (pitch calibration, role review) only when they're missing.
python main.py --video $Video --all --device $Device

Write-Host ""
Write-Host "=== DONE ==="
Write-Host "Final video:  outputs/$($Stem)_final_with_minimap.mp4"

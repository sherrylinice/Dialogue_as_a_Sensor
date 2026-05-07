#!/usr/bin/env bash
# Generate a small stuck-state dataset for the agent-system to consume.
# Mirrors run_generation.sh but always saves every trial (success OR stuck).
#
# Tunables (all optional):
#   NUM_TRIALS=20  NUM_VIDEOS=2  START_INDEX=0
#   bash run_stuck_generation.sh                       # use defaults
#   NUM_TRIALS=50 bash run_stuck_generation.sh         # override via env
#   bash run_stuck_generation.sh --save_all_phases     # forward extra args
set -euo pipefail

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/jc166/.conda/envs/convai}/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Forward any extra CLI args verbatim ("$@"). Quoting individually preserves
# args with spaces while still expanding to zero tokens when the array is
# empty - so we don't accidentally pass an empty-string arg to argparse.
python generate_stuck_dataset.py \
    --output_dir ./my_stuck_data \
    --num_trials "${NUM_TRIALS:-10}" \
    --num_videos "${NUM_VIDEOS:-1}" \
    --start_index "${START_INDEX:-0}" \
    --save_all_phases \
    "$@"

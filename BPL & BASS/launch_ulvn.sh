#!/usr/bin/env bash

set -euo pipefail

# Launch the online ULVN navigation stack in a tmux session.
#
# Usage:
#   ./launch_ulvn.sh \
#     --bass --img_dir data/landmark ... \
#     --nav  --image-topic /isaac_node/camera0/image_raw ...
#
# Environment variables:
#   ULVN_CONDA_ENV        Conda env to activate. Default: ulvn_nav

BASS_ARGS=()
NAV_ARGS=()
current=""

for arg in "$@"; do
  case "$arg" in
    --bass) current="bass"; continue ;;
    --nav) current="nav"; continue ;;
  esac

  if [[ "$current" == "bass" ]]; then
    BASS_ARGS+=("$arg")
  elif [[ "$current" == "nav" ]]; then
    NAV_ARGS+=("$arg")
  else
    echo "Arguments must be grouped after --bass or --nav: $arg" >&2
    exit 2
  fi
done

if [[ ${#BASS_ARGS[@]} -eq 0 || ${#NAV_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --bass <bass.py args> --nav <local_planner.py args>" >&2
  exit 2
fi

escape_args() {
  local out=()
  local a q
  for a in "$@"; do
    printf -v q '%q' "$a"
    out+=("$q")
  done
  printf '%s ' "${out[@]}"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$script_dir"

conda_env="${ULVN_CONDA_ENV:-ulvn_nav}"
BASS_ARGS_ESC="$(escape_args "${BASS_ARGS[@]}")"
NAV_ARGS_ESC="$(escape_args "${NAV_ARGS[@]}")"
printf -v CONDA_ENV_ESC '%q' "$conda_env"
printf -v REPO_ROOT_ESC '%q' "$repo_root"

setup_cmd="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_ESC} && export PYTHONPATH=${REPO_ROOT_ESC}:\${PYTHONPATH:-}"

session_name="ulvn_$(date +%s)"
tmux new-session -d -s "$session_name"

tmux select-pane -t 0
tmux split-window -h -p 50

tmux select-pane -t 0
tmux send-keys "${setup_cmd} && python bass.py ${BASS_ARGS_ESC}" Enter

tmux select-pane -t 1
tmux send-keys "${setup_cmd} && python local_planner.py ${NAV_ARGS_ESC}" Enter

tmux -2 attach-session -t "$session_name"

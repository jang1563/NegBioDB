#!/bin/bash
# Submit MD downstream jobs to Cayuga from the local machine via SSH.
#
# Prerequisite:
#   ssh -fN cayuga-login1
#
# Examples:
#   bash scripts_md/remote_submit_md_cayuga.sh status
#   bash scripts_md/remote_submit_md_cayuga.sh train --task m2 --split cold_both --model-type mlp
#   bash scripts_md/remote_submit_md_cayuga.sh after-ingest

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-cayuga-login1}
REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR:-${SCRATCH_DIR:-/athena/masonlab/scratch/users/jak4013}/negbiodb}
SSH_BIN=${SSH_BIN:-ssh}
CONNECT_TIMEOUT=${CONNECT_TIMEOUT:-10}
SSH_CONTROL_MASTER=${SSH_CONTROL_MASTER:-no}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-none}

if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts_md/remote_submit_md_cayuga.sh <submit_md_jobs args...>" >&2
    exit 1
fi

quoted_args=()
for arg in "$@"; do
    quoted_args+=("$(printf '%q' "$arg")")
done
remote_args="${quoted_args[*]}"

"$SSH_BIN" \
    -o ConnectTimeout="$CONNECT_TIMEOUT" \
    -o ControlMaster="$SSH_CONTROL_MASTER" \
    -o ControlPath="$SSH_CONTROL_PATH" \
    "$REMOTE_HOST" \
    "cd \"$REMOTE_PROJECT_DIR\" && bash scripts_md/submit_md_jobs.sh $remote_args"

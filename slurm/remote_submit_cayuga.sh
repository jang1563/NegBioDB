#!/bin/bash
# Submit NegBioDB jobs to Cayuga from a local machine via SSH ControlMaster.
#
# Prerequisite:
#   ssh -fN cayuga-login1
#
# Usage:
#   bash slurm/remote_submit_cayuga.sh
#   SEEDS="42 43 44" MODELS="deepdta graphdta" bash slurm/remote_submit_cayuga.sh

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-cayuga-login1}
REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR:-${SCRATCH_DIR:-/path/to/scratch}/negbiodb}
SSH_BIN=${SSH_BIN:-ssh}
CONNECT_TIMEOUT=${CONNECT_TIMEOUT:-10}

SEEDS_STR=${SEEDS:-42}
MODELS_STR=${MODELS:-"deepdta graphdta drugban"}
SPLITS_STR=${SPLITS:-"random cold_compound cold_target ddb"}
NEGATIVES_STR=${NEGATIVES:-"negbiodb uniform_random degree_matched"}
SBATCH_BIN_REMOTE=${SBATCH_BIN_REMOTE:-/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch}
SQUEUE_BIN_REMOTE=${SQUEUE_BIN_REMOTE:-/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue}

echo "=== Remote Cayuga submission ==="
echo "Host: $REMOTE_HOST"
echo "Project: $REMOTE_PROJECT_DIR"
echo "Seeds: $SEEDS_STR"
echo "Models: $MODELS_STR"
echo "Splits: $SPLITS_STR"
echo "Negatives: $NEGATIVES_STR"
echo ""

"$SSH_BIN" -o ConnectTimeout="$CONNECT_TIMEOUT" "$REMOTE_HOST" \
    "cd \"$REMOTE_PROJECT_DIR\" && \
     SEEDS='$SEEDS_STR' MODELS='$MODELS_STR' SPLITS='$SPLITS_STR' NEGATIVES='$NEGATIVES_STR' \
     SBATCH_BIN='$SBATCH_BIN_REMOTE' SQUEUE_BIN='$SQUEUE_BIN_REMOTE' \
     bash slurm/submit_all.sh"

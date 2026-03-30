#!/bin/bash
# Monitor NegBioDB Cayuga jobs/logs from a local machine via SSH ControlMaster.
#
# Prerequisite:
#   ssh -fN cayuga-login1
#
# Usage:
#   bash slurm/remote_monitor_cayuga.sh
#   LOG_GLOB="negbio_deepdta_balanced_random_negbiodb_seed42" bash slurm/remote_monitor_cayuga.sh

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-cayuga-login1}
REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR:-${SCRATCH_DIR:-/path/to/scratch}/negbiodb}
SSH_BIN=${SSH_BIN:-ssh}
CONNECT_TIMEOUT=${CONNECT_TIMEOUT:-10}
SQUEUE_BIN_REMOTE=${SQUEUE_BIN_REMOTE:-/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue}
LOG_GLOB=${LOG_GLOB:-negbio_*}

echo "=== Remote Cayuga monitor ==="
echo "Host: $REMOTE_HOST"
echo "Project: $REMOTE_PROJECT_DIR"
echo "Log pattern: $LOG_GLOB"
echo ""

"$SSH_BIN" -o ConnectTimeout="$CONNECT_TIMEOUT" "$REMOTE_HOST" \
    "echo '--- squeue ---' && \
     '$SQUEUE_BIN_REMOTE' -u $REMOTE_USER || true && \
     echo && \
     echo '--- latest logs ---' && \
     ls -lt \"$REMOTE_PROJECT_DIR/logs\"/${LOG_GLOB}_*.err 2>/dev/null | head -5 || true"

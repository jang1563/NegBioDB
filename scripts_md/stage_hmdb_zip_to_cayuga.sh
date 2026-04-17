#!/bin/bash
# Stage a browser-downloaded HMDB zip to Cayuga so the MD ingest watcher can proceed.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cayuga-login1}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/athena/masonlab/scratch/users/jak4013/negbiodb/data}"
REMOTE_LOG="${REMOTE_LOG:-/athena/masonlab/scratch/users/jak4013/negbiodb/logs/md_ingest_watch_hmdb.log}"
SOURCE_ZIP="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DATA_DIR="${SCRIPT_DIR}/../data"

if [[ -z "${SOURCE_ZIP}" ]]; then
    for candidate in \
        "${PROJECT_DATA_DIR}/hmdb_metabolites.zip" \
        "${HOME}/Downloads"/hmdb_metabolites*.zip \
        "${HOME}/Downloads"/hmdb*.zip; do
        if [[ -f "${candidate}" ]]; then
            SOURCE_ZIP="${candidate}"
            break
        fi
    done
fi

if [[ -z "${SOURCE_ZIP}" || ! -f "${SOURCE_ZIP}" ]]; then
    echo "ERROR: Could not find an HMDB zip to stage."
    echo "Pass the zip path explicitly, place hmdb_metabolites.zip in data/, or ~/Downloads."
    exit 1
fi

echo "Staging ${SOURCE_ZIP} -> ${REMOTE_HOST}:${REMOTE_DATA_DIR}/hmdb_metabolites.zip"
rsync -av --progress --partial \
    -e "ssh -o ControlMaster=no -o ControlPath=none" \
    "${SOURCE_ZIP}" \
    "${REMOTE_HOST}:${REMOTE_DATA_DIR}/hmdb_metabolites.zip"

ssh -o ControlMaster=no -o ControlPath=none "${REMOTE_HOST}" \
    "ls -lh '${REMOTE_DATA_DIR}/hmdb_metabolites.zip'; tail -n 5 '${REMOTE_LOG}' 2>/dev/null || true"

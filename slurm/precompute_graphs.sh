#!/bin/bash
# Pre-compute molecular graph cache for GraphDTA/DrugBAN.
# Submit this CPU job BEFORE submit_all.sh to avoid race conditions.
#
# Usage:
#   bash slurm/precompute_graphs.sh
#
# This builds exports/graph_cache.pt (~919K SMILES → PyG Data objects).
# Estimated time: 15-30 minutes on 8 CPU cores.

set -euo pipefail

SBATCH=${SBATCH_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch}
NEGBIODB=${SCRATCH_DIR:-/path/to/scratch}/negbiodb
LOGDIR=$NEGBIODB/logs
mkdir -p "$LOGDIR"

JOB_ID=$(
    "$SBATCH" \
        --job-name="negbio_precompute_graphs" \
        --partition=scu-gpu \
        --cpus-per-task=8 \
        --mem=32G \
        --time=1:00:00 \
        --output="$LOGDIR/precompute_graphs_%j.out" \
        --error="$LOGDIR/precompute_graphs_%j.err" \
        --wrap="
            source ${CONDA_PREFIX:-/path/to/conda}/miniconda3/etc/profile.d/conda.sh && \
            conda activate negbiodb-ml && \
            cd $NEGBIODB && \
            python -c \"
import pandas as pd
from pathlib import Path
from negbiodb.models.graphdta import smiles_to_graph
import torch, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('precompute')

data_dir = Path('exports')
m1 = pd.read_parquet(data_dir / 'negbiodb_m1_balanced.parquet', columns=['smiles'])
smiles_list = m1['smiles'].unique().tolist()
logger.info('Building cache for %d unique SMILES...', len(smiles_list))

cache = {}
CHUNK = 10000
failed = 0
for i in range(0, len(smiles_list), CHUNK):
    batch = smiles_list[i:i+CHUNK]
    for smi in batch:
        g = smiles_to_graph(smi)
        cache[smi] = g
        if g is None:
            failed += 1
    logger.info('Processed %d/%d (failed=%d)', min(i+CHUNK, len(smiles_list)), len(smiles_list), failed)

torch.save(cache, data_dir / 'graph_cache.pt')
logger.info('Saved graph_cache.pt (%d entries, %d failed)', len(cache), failed)
\"
        " | grep -oP 'batch job \K\d+'
)
echo "Submitted precompute_graphs → job $JOB_ID"
echo "Wait for completion before running submit_all.sh:"
echo "  ${SQUEUE_BIN:-/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue} -j $JOB_ID"

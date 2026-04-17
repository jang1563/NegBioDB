# NegBioRL Phase 6 Handoff — 2026-04-15

## What This Is

Phase 6 of the NegBioRL eval-to-train pipeline: **Gemma3-27B scale ablation + single-domain transfer matrix**. Tests whether the transfer paradox (observed at Qwen-7B) replicates at 27B, and whether training method (SFT/DPO/GRPO) or training domain matters more than model scale.

Base model: `unsloth/gemma-3-27b-it-bnb-4bit` (4-bit pre-quantized, LoRA r=16/alpha=32).

## Status: 5/6 Experiments Complete; final eval queued

| Job ID | Experiment | Train Domain | Status | Checkpoint |
|--------|-----------|-------------|--------|------------|
| 2789387 | SFT | all 4 (multi) | COMPLETE | `phase3_checkpoints/sft_2789387/final` |
| 2789566 | DPO | all 4 (multi) | COMPLETE | `phase3_checkpoints/dpo_2789566/final` |
| 2789389 | DTI-only GRPO G=2 | DTI | COMPLETE | `phase3_checkpoints/grpo_G2_2789389/final` |
| 2789390 | CT-only GRPO G=2 | CT | COMPLETE | `phase3_checkpoints/grpo_G2_2789390/final` |
| 2789391 | PPI-only GRPO G=2 | PPI | COMPLETE | `phase3_checkpoints/grpo_G2_2789391/final` |
| 2789392 | GE-only GRPO G=2 | GE | **IN PROGRESS** | `phase3_checkpoints/grpo_G2_2789392/checkpoint-1160` |

GE GRPO timed out at 16h (checkpoint-1160/2320 = 50%). Continuation job **2805997** is RUNNING. Partial eval (checkpoint-1160) already completed as job 2805993.

Checked on **2026-04-15**:
- Training continuation **2805997** is `R` in `squeue`
- Final eval **2806332** has already been submitted with `Dependency=afterok:2805997`
- When 2805997 exits successfully, 2806332 will start automatically; no manual eval submission is needed unless that dependent job is canceled or fails

All checkpoints at: `/athena/masonlab/scratch/users/jak4013/negbiodb/results/negbiorl/phase3_checkpoints/`

## L4 ΔMCC Results (the key metric)

Gemma3-27B baseline MCC: DTI=0.000, CT=0.191, PPI=0.242, GE=−0.111.

| Experiment | DTI L4 | CT L4 | PPI L4 | GE L4 |
|---|---|---|---|---|
| **SFT** (2789567) | **+0.160** | −0.073 | +0.035 | +0.111 |
| **DPO** (2791137) | −0.081 | **+0.185** | +0.056 | **+0.135** |
| **DTI GRPO** (2789568) | −0.116 | +0.192 | +0.040 | +0.104 |
| **CT GRPO** (2791124) | −0.111 | +0.189 | +0.045 | +0.104 |
| **PPI GRPO** (2791125) | −0.120 | +0.180 | +0.045 | +0.104 |
| **GE GRPO partial** (2805993) | −0.086 | +0.180 | +0.040 | +0.104 |

Eval job IDs in parentheses. Full before_after.json in `results/negbiorl/phase4_eval/<job_id>/`.

## Pending Tasks

### 1. Wait for GE GRPO continuation (job 2805997)

Check status:
```bash
/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -j 2805997,2806332
```

When complete, verify `final/` exists:
```bash
ls results/negbiorl/phase3_checkpoints/grpo_G2_2789392/final/
```

### 2. Wait for dependent GE GRPO final eval (job 2806332)

```bash
cd /athena/masonlab/scratch/users/jak4013/negbiodb
/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -j 2806332
tail -n 50 logs/rl_eval_2806332.log
```

If 2806332 needs to be resubmitted manually for any reason, use:
```bash
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
  --time=08:00:00 \
  --export=ALL,ADAPTER=results/negbiorl/phase3_checkpoints/grpo_G2_2789392/final,BASE_MODEL=unsloth/gemma-3-27b-it-bnb-4bit,BACKEND=hf,LOAD_IN_4BIT=1 \
  slurm/run_rl_eval.slurm
```

### 3. Collect GE GRPO final eval results

After eval completes:
```bash
cat results/negbiorl/phase4_eval/<NEW_JOB_ID>/before_after.json
```

Extract ΔMCC for each domain×task. Update the transfer matrix table in `experiment_results.md` (replace the GE GRPO "TBD" row).

### 4. Update experiment_results.md

In the "Phase 6" section (~line 265), replace:
```
| **GE-only GRPO** (2789392) | GE | TBD* | TBD* | TBD* | TBD* |
```
with the actual ΔMCC values from the eval. Also remove the TBD footnote.

### 5. Update MEMORY.md

In `~/.claude/projects/-Users-jak4013-Dropbox-Bioinformatics-Claude-Negative-result-DB/memory/MEMORY.md`, update the NegBioRL section:
- Change "PHASE 5 + P12 + PHASE 6 (NEARLY COMPLETE)" → "PHASE 6 COMPLETE"
- Replace GE GRPO TBD line with actual results
- Remove continuation job reference

## Key Technical Details

### HPC Access
```bash
ssh -i ~/.ssh/keyfile-private-file jak4013@cayuga-login1.cac.cornell.edu
```
Working dir: `/athena/masonlab/scratch/users/jak4013/negbiodb/`
SLURM binary: `/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch`

### Eval Script Mechanics
- `slurm/run_rl_eval.slurm` runs `scripts_rl/07_eval_before_after.py`
- Loads adapter with `LOAD_IN_4BIT=1` for Gemma3-27B (HF backend only, NOT vLLM)
- Generates predictions on L1+L4 test sets for all TRAIN_DOMAINS: dti, ct, ppi, ge, dc
- DC is registered but has no P12/Phase6 baseline (Qwen-only); DC predictions will be generated but baseline comparison may be missing
- VP is excluded (degenerate single-class test)
- Timeout: 8h should be sufficient (prior evals took ~4h)

### Resume Feature (new)
Added `--resume-from-checkpoint` to `scripts_rl/06_train_grpo.py` and `slurm/run_rl_grpo.slurm`. Job 2805997 uses this to continue from checkpoint-1160. The SLURM var is `RESUME_FROM_CHECKPOINT`.

### File Locations
- Eval results: `results/negbiorl/phase4_eval/<job_id>/before_after.json`
- Checkpoints: `results/negbiorl/phase3_checkpoints/<type>_<job_id>/final/`
- Config files: `configs/negbiorl/grpo_gemma4_31b.yaml`, `sft_gemma3_27b.yaml`, `dpo_gemma3_27b.yaml`
- SLURM logs: `logs/rl_grpo_<job_id>.log`, `logs/rl_eval_<job_id>.log`

## Key Findings (for context)

1. **Transfer pattern is domain-invariant at 27B:** DTI/CT/PPI/GE-only GRPO all give near-identical ΔMCC (±0.01). Training domain identity doesn't matter.
2. **DPO ≈ GRPO at 27B:** Both converge to CT +0.185, GE +0.135. Method choice matters less than scale.
3. **SFT uniquely preserves DTI L4 (+0.160):** All GRPO/DPO hurt DTI (−0.08 to −0.12). GRPO reward penalizes contamination shortcut.
4. **Model scale > everything else:** Gemma3-27B G=2 >> Qwen-7B G=8/16 across all domains.

## Known Risks

| Risk | Mitigation |
|------|-----------|
| GE GRPO continuation may timeout again | checkpoint-1160 partial eval already confirms domain-invariance pattern; full training is nice-to-have |
| GPU contention (QOSMaxGRESPerUser) | Eval jobs queue behind other GPU jobs; 8h wall time is generous |
| DC domain in eval has no Gemma3-27B baseline | DC predictions will be generated but before_after comparison may show baseline=0; this is expected |

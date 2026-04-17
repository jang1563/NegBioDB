#!/usr/bin/env python3
"""NES Step 2 (Local): Correlate domain embedding similarity with transfer ΔL4 MCC.

Hypothesis: cosine similarity between domain L1 prompt embeddings (Gemma-3-27B)
predicts cross-domain GRPO transfer gain (ΔL4 MCC from Phase 5 transfer experiments).

Analysis:
  1. Load domain embeddings from results/spinout/idea_nes/domain_embeddings.npz
  2. Compute 4×4 cosine similarity matrix
  3. Load 4×4 transfer ΔL4 matrix from results/negbiorl/phase5_transfer/{domain}/before_after.json
  4. Correlate 12 off-diagonal cells: Spearman(cos_sim_ij, delta_L4_ij)
  5. Permutation p-value (1000 shuffles of ΔL4 columns)

Gate: ρ ≥ +0.40 with perm p < 0.05

Output: results/spinout/idea_nes/gate.json + idea_nes_summary.md
"""
import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats


DOMAINS = ["dti", "ct", "ppi", "ge"]
TRANSFER_DIR = "results/negbiorl/phase5_transfer"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_transfer_matrix(transfer_dir: str, domains: list[str]) -> np.ndarray:
    """Load 4×4 ΔL4 MCC matrix from before_after.json files.

    Returns matrix where mat[i,j] = ΔL4 MCC when training on domains[i] and evaluating on domains[j].
    Diagonal cells are in-domain (training and eval domain same).
    """
    n = len(domains)
    mat = np.full((n, n), np.nan)
    for i, train_domain in enumerate(domains):
        path = Path(transfer_dir) / train_domain / "before_after.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        with open(path) as f:
            raw = f.read().replace(": NaN", ": null").replace(":NaN", ":null")
        data = json.loads(raw)
        for j, eval_domain in enumerate(domains):
            if eval_domain in data and "l4" in data[eval_domain]:
                delta = data[eval_domain]["l4"]["deltas"].get("delta_mcc")
                if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
                    mat[i, j] = float(delta)
    return mat


def permutation_spearman(x_off: np.ndarray, y_off: np.ndarray,
                         n_perms: int, rng: np.random.Generator) -> tuple[float, float]:
    """Spearman ρ with one-sided permutation p (H1: ρ > 0)."""
    obs_rho, _ = stats.spearmanr(x_off, y_off)
    count = sum(
        1 for _ in range(n_perms)
        if stats.spearmanr(x_off, rng.permutation(y_off))[0] >= obs_rho
    )
    return float(obs_rho), (count + 1) / (n_perms + 1)


def main():
    ap = argparse.ArgumentParser(description="NES embedding analysis")
    ap.add_argument("--embeddings", default="results/spinout/idea_nes/domain_embeddings.npz")
    ap.add_argument("--transfer-dir", default=TRANSFER_DIR)
    ap.add_argument("--layer", choices=["mid", "final"], default="mid",
                    help="Which embedding layer to use (default: mid)")
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/spinout/idea_nes")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load embeddings ───────────────────────────────────────────────────────
    if not Path(args.embeddings).exists():
        print(f"ERROR: embeddings file not found: {args.embeddings}")
        print("Run scripts_rl/11a_nes_extract_embeddings.py on HPC first.")
        return 1

    data = np.load(args.embeddings)
    print("Available embedding keys:", list(data.keys()))

    embeddings = {}
    for domain in DOMAINS:
        key = f"{domain}_{args.layer}"
        if key not in data:
            print(f"  WARN: key '{key}' not in embeddings file, skipping {domain}")
            continue
        embeddings[domain] = data[key]
        print(f"  {domain} ({args.layer}): shape={embeddings[domain].shape}")

    available_domains = [d for d in DOMAINS if d in embeddings]
    n = len(available_domains)
    if n < 3:
        print(f"ERROR: need ≥3 domains, got {n}")
        return 1

    # ── Cosine similarity matrix ──────────────────────────────────────────────
    print(f"\nCosine similarity matrix ({args.layer} layer):")
    cos_mat = np.zeros((n, n))
    for i, di in enumerate(available_domains):
        for j, dj in enumerate(available_domains):
            cos_mat[i, j] = cosine_similarity(embeddings[di], embeddings[dj])
    header = "\t".join(available_domains)
    print(f"\t{header}")
    for i, di in enumerate(available_domains):
        row = "\t".join(f"{cos_mat[i,j]:.4f}" for j in range(n))
        print(f"{di}\t{row}")

    # ── Load transfer ΔL4 matrix ──────────────────────────────────────────────
    print(f"\nLoading transfer ΔL4 matrix from {args.transfer_dir}...")
    delta_mat = load_transfer_matrix(args.transfer_dir, available_domains)
    print("ΔL4 MCC matrix (train→eval):")
    print(f"\t{header}")
    for i, di in enumerate(available_domains):
        row = "\t".join(
            f"{delta_mat[i,j]:+.4f}" if not np.isnan(delta_mat[i,j]) else "nan"
            for j in range(n)
        )
        print(f"{di}\t{row}")

    # ── Extract off-diagonal cells ────────────────────────────────────────────
    cos_off = []
    delta_off = []
    cell_labels = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip diagonal
            if np.isnan(delta_mat[i, j]):
                print(f"  WARN: NaN delta for {available_domains[i]}→{available_domains[j]}, skipping")
                continue
            cos_off.append(cos_mat[i, j])
            delta_off.append(delta_mat[i, j])
            cell_labels.append(f"{available_domains[i]}→{available_domains[j]}")

    cos_off = np.array(cos_off)
    delta_off = np.array(delta_off)
    n_cells = len(cos_off)
    print(f"\nOff-diagonal cells: {n_cells}")
    for label, c, d in zip(cell_labels, cos_off, delta_off):
        print(f"  {label}: cos_sim={c:.4f}, delta_L4={d:+.4f}")

    if n_cells < 6:
        print(f"ERROR: too few off-diagonal cells ({n_cells}) to compute meaningful correlation")
        return 1

    # ── Spearman ρ + permutation test ─────────────────────────────────────────
    print(f"\nSpearman ρ + {args.n_perms}-perm p-value...")
    rho, perm_p = permutation_spearman(cos_off, delta_off, args.n_perms, rng)
    scipy_rho, scipy_p = stats.spearmanr(cos_off, delta_off)
    print(f"  ρ = {rho:.4f}")
    print(f"  perm p (one-sided) = {perm_p:.4f}")
    print(f"  scipy two-sided p = {scipy_p:.4e}")

    # Also run with final layer if available
    alt_result = None
    alt_layer = "final" if args.layer == "mid" else "mid"
    alt_embeddings = {}
    for domain in available_domains:
        key = f"{domain}_{alt_layer}"
        if key in data:
            alt_embeddings[domain] = data[key]
    if len(alt_embeddings) == n:
        cos_alt = np.array([
            cosine_similarity(alt_embeddings[available_domains[i]],
                              alt_embeddings[available_domains[j]])
            for i in range(n) for j in range(n) if i != j and not np.isnan(delta_mat[i,j])
        ])
        rho_alt, _ = stats.spearmanr(cos_alt, delta_off)
        print(f"  Sensitivity ({alt_layer} layer) ρ = {rho_alt:.4f}")
        alt_result = {"layer": alt_layer, "rho": float(rho_alt)}

    # ── Gate ──────────────────────────────────────────────────────────────────
    print(f"\n=== GATE NES ===")
    c1 = n_cells >= 6
    c2 = rho >= 0.40
    c3 = perm_p < 0.05
    gate_pass = c1 and c2 and c3
    print(f"  C0: ≥6 off-diagonal cells: {n_cells}   {'PASS' if c1 else 'FAIL'}")
    print(f"  C1: ρ ≥ 0.40:              {rho:.4f}  {'PASS' if c2 else 'FAIL'}")
    print(f"  C2: perm p < 0.05:         {perm_p:.4f}  {'PASS' if c3 else 'FAIL'}")
    print(f"\n  GATE NES: {'PASS' if gate_pass else 'FAIL'}")

    result = {
        "idea": "NES",
        "layer_primary": args.layer,
        "n_domains": n,
        "available_domains": available_domains,
        "n_off_diagonal_cells": n_cells,
        "cell_labels": cell_labels,
        "cosine_similarity_off_diagonal": cos_off.tolist(),
        "delta_L4_off_diagonal": delta_off.tolist(),
        "spearman_rho": rho,
        "perm_p_one_sided": perm_p,
        "scipy_p_two_sided": float(scipy_p),
        "n_perms": args.n_perms,
        "sensitivity_alt_layer": alt_result,
        "cosine_similarity_matrix": {
            available_domains[i]: {
                available_domains[j]: float(cos_mat[i, j])
                for j in range(n)
            }
            for i in range(n)
        },
        "delta_L4_matrix": {
            available_domains[i]: {
                available_domains[j]: float(delta_mat[i, j]) if not np.isnan(delta_mat[i,j]) else None
                for j in range(n)
            }
            for i in range(n)
        },
        "gate_nes": {
            "gate_pass": bool(gate_pass),
            "c0_n_cells_ge_6": bool(c1),
            "c1_rho_ge_0_40": bool(c2),
            "c1_rho": rho,
            "c2_perm_p_lt_0_05": bool(c3),
            "c2_perm_p": perm_p,
        },
    }

    gate_path = out_dir / "gate.json"
    with open(gate_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {gate_path}")

    summary = f"""# Idea NES — Negative Embedding Similarity

**Date:** 2026-04-16
**Status:** {'GATE PASS' if gate_pass else 'GATE FAIL'} (pending HPC embeddings)

## Hypothesis
Cosine similarity between Gemma-3-27B hidden-state embeddings of domain L1 prompts
predicts cross-domain GRPO transfer gain (ΔL4 MCC). Domains whose task schemas
occupy nearby regions in the model's hidden state space should see stronger
positive transfer when trained on each other's data.

## Data
- **Embeddings:** `{args.embeddings}` (layer {args.layer} of Gemma-3-27B)
- **Transfer matrix:** `{args.transfer_dir}/{{domain}}/before_after.json`
  (Qwen-7B Phase 5 single-domain GRPO transfer, 4 training × 4 eval domains)
- **Analysis set:** {n_cells} off-diagonal cells (excludes diagonal in-domain self-eval)

## Results

### Cosine Similarity Matrix ({args.layer} layer)
| | {' | '.join(available_domains)} |
|---|{'---|' * n}
{chr(10).join(f'| {d} | ' + ' | '.join(f'{cos_mat[i,j]:.4f}' for j in range(n)) + ' |' for i, d in enumerate(available_domains))}

### Transfer ΔL4 MCC Matrix
| | {' | '.join(available_domains)} |
|---|{'---|' * n}
{chr(10).join(f'| {d} | ' + ' | '.join((f'{delta_mat[i,j]:+.4f}' if not np.isnan(delta_mat[i,j]) else 'nan') for j in range(n)) + ' |' for i, d in enumerate(available_domains))}

### Correlation
| Metric | Value |
|--------|-------|
| Spearman ρ (off-diagonal) | {rho:.4f} |
| Permutation p (one-sided, H1: ρ > 0) | {perm_p:.4f} |
| scipy two-sided p | {float(scipy_p):.3e} |
| Sensitivity ({alt_layer} layer) ρ | {f"{alt_result['rho']:.4f}" if alt_result else 'N/A'} |

## Gate NES

| Criterion | Threshold | Value | Result |
|-----------|-----------|-------|--------|
| C0: off-diagonal cells | ≥ 6 | {n_cells} | {'PASS' if c1 else 'FAIL'} |
| C1: ρ | ≥ 0.40 | {rho:.4f} | {'PASS' if c2 else 'FAIL'} |
| C2: perm p | < 0.05 | {perm_p:.4f} | {'PASS' if c3 else 'FAIL'} |

**GATE NES: {'PASS' if gate_pass else 'FAIL'}**

## Relation to Prior Art
Achille et al. 2019 (Task2Vec) used Fisher information embeddings to predict task
transfer in computer vision. NES applies an analogous idea to NLP: last-token hidden
states of causal LLMs as task representations. Unlike Task2Vec which uses the full
probe network, NES uses the LLM's own hidden states — no auxiliary probe needed.

## Outputs
- [gate.json](gate.json)
- Embeddings: `{args.embeddings}`
"""

    summary_path = out_dir / "idea_nes_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

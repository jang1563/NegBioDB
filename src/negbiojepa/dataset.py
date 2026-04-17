"""Unified multi-domain dataset for Negative-JEPA pretraining and fine-tuning.

NegJEPASample: standardized sample format across all 8 NegBioDB domains.
Domain adapters: per-domain functions that convert parquet rows → NegJEPASample.
NegJEPADataset: torch Dataset backed by parquet files (HPC) or synthetic data (smoke test).
MultiDomainSampler: temperature-scaled cross-domain sampler.
jepa_collate_fn: PyG-aware collation for multi-modal batches.

Data root on HPC: /athena/masonlab/scratch/users/jak4013/negbiodb_exports/
  dti_ml/, ppi_ml/, ge_ml/, ct_ml/, vp_ml/, dc_ml/, cp_ml/, md_ml/
Local: only exports/vp_ml/ exists. Use data_root='synthetic' for smoke tests.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data

try:
    from torch_geometric.data import Batch, Data
    from negbiodb.models.graphdta import smiles_to_graph as _smiles_to_graph_raw  # type: ignore[attr-defined]
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

# Alias for non-cached usage (smoke tests, small datasets)
smiles_to_graph = _smiles_to_graph_raw if HAS_TORCH_GEOMETRIC else None

try:
    from negbiodb.models.deepdta import smiles_to_tensor, seq_to_tensor as dta_seq_to_tensor  # noqa: F401
    from negbiodb_ppi.models.pipr import seq_to_tensor as ppi_seq_to_tensor  # noqa: F401
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# ─── Domain registry ──────────────────────────────────────────────────────────

DOMAIN_NAMES = ["dti", "ppi", "ge", "ct", "vp", "dc", "cp", "md"]
DOMAIN_ID: Dict[str, int] = {d: i for i, d in enumerate(DOMAIN_NAMES)}

# Approximate domain sizes for temperature-scaled sampling (P(d) ∝ |D_d|^0.5)
DOMAIN_APPROX_SIZES: Dict[str, int] = {
    "dti": 30_500_000,
    "ge":  28_800_000,
    "vp":   2_400_000,
    "ppi":  2_200_000,
    "dc":   1_200_000,
    "ct":     133_000,
    "cp":       9_000,
    "md":       5_000,  # TBD — updated once MD data is available
}

# Tabular feature dimensionality per domain (confirmed from design doc)
DOMAIN_TABULAR_DIMS: Dict[str, int] = {
    "dti": 50,
    "ppi": 45,
    "ge":  30,
    "ct":  80,
    "vp":  56,
    "dc":  293,
    "cp":  200,
    "md":  300,   # MD has 2068 features, pre-reduced to 300 via PCA/feature selection
}

# Parquet filenames (relative to data_root/{domain}_ml/)
DOMAIN_PARQUET: Dict[str, str] = {
    "dti": "negbiodb_dti_pairs.parquet",
    "ppi": "negbiodb_ppi_pairs.parquet",
    "ge":  "negbiodb_ge_pairs.parquet",
    "ct":  "negbiodb_ct_pairs.parquet",
    "vp":  "negbiodb_vp_pairs.parquet",
    "dc":  "negbiodb_dc_pairs.parquet",
    "cp":  "negbiodb_cp_pairs.parquet",
    "md":  "negbiodb_md_pairs.parquet",
}


# ─── Sample dataclass ─────────────────────────────────────────────────────────

@dataclass
class NegJEPASample:
    """Standardized sample format across all 8 NegBioDB domains.

    tabular_A and tabular_B are always present (zero-padded to max_features).
    Graph, sequence, and ESM2 fields are None for domains that don't use them.
    """
    tabular_A: torch.Tensor          # (max_features,) float32, zero-padded
    tabular_B: torch.Tensor          # (max_features,) float32, zero-padded
    domain_id: int                   # DOMAIN_ID[domain]
    label: int                       # 0=negative, 1=positive
    graph_A: Optional["Data"] = None # PyG Data — DTI drug, DC drug_A, CP compound
    graph_B: Optional["Data"] = None # PyG Data — DC drug_B only
    seq_A: Optional[torch.Tensor] = None  # (L,) int64 — DTI SMILES, PPI protein A
    seq_B: Optional[torch.Tensor] = None  # (L,) int64 — DTI target seq, PPI protein B
    esm2_A: Optional[torch.Tensor] = None # (1280,) float32 — VP pre-computed ESM2


# ─── Domain adapter helpers ───────────────────────────────────────────────────

def _resolve_split_col(df: pd.DataFrame, split_col: Optional[str] = None) -> str:
    """Find the correct split column in a parquet DataFrame.

    Auto-detects if split_col is None; handles _v1 suffix (GE exports).
    """
    if split_col is not None:
        if split_col in df.columns:
            return split_col
        # Handle _v1 suffix
        v1 = split_col + "_v1"
        if v1 in df.columns:
            return v1
        raise ValueError(
            f"split_col={split_col!r} not found. "
            f"Available: {[c for c in df.columns if c.startswith('split')]}"
        )
    # Auto-detect
    for candidate in ("split", "split_random", "split_random_v1"):
        if candidate in df.columns:
            return candidate
    split_cols = [c for c in df.columns if c.startswith("split_")]
    raise ValueError(
        f"Cannot auto-detect split column. Available: {split_cols}"
    )


def _is_numeric_safe(val) -> bool:
    """Check if a value is numeric (int/float/np scalar), not a string/None."""
    if val is None or isinstance(val, str):
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _pad_features(arr: np.ndarray, max_features: int) -> torch.Tensor:
    """Zero-pad a 1D feature array to max_features and convert to float32 tensor."""
    out = np.zeros(max_features, dtype=np.float32)
    n = min(len(arr), max_features)
    out[:n] = arr[:n]
    return torch.from_numpy(out)


def _get_label(row: pd.Series) -> int:
    """Extract binary label from a parquet row (0=negative, 1=positive)."""
    for col in ("label", "Y", "outcome", "y", "is_positive"):
        if col in row.index:
            return int(row[col])
    return 0


def _get_tabular(row: pd.Series, feature_cols: List[str], max_features: int) -> torch.Tensor:
    """Extract tabular feature columns and zero-pad. NaN values are filled with 0."""
    arr = row[feature_cols].values.astype(np.float32)
    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return _pad_features(arr, max_features)


# ── DTI adapter ──────────────────────────────────────────────────────────────

def _dti_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Drug-Target Interaction: drug graph + drug SMILES (seq_A) + target seq (seq_B)."""
    # Tabular features: compound descriptors + target features (columns with prefix)
    _DTI_EXCLUDE = {"pair_id", "smiles", "compound_smiles", "inchikey", "uniprot_id",
                     "target_sequence", "gene_symbol", "Y", "label",
                     "confidence_tier", "best_result_type"}
    tab_cols = [c for c in row.index if c.startswith(("feat_", "rdkit_", "desc_", "fp_"))
                or c in ("mw", "logp", "tpsa", "hba", "hbd", "rotatable_bonds")]
    if not tab_cols:
        # Fallback: numeric non-label, non-string columns
        tab_cols = [c for c in row.index
                    if _is_numeric_safe(row[c])
                    and not c.startswith("split_")
                    and c not in _DTI_EXCLUDE]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)
    tabular_B = torch.zeros(max_features)  # target features (if available)

    # Resolve SMILES column (export uses 'smiles' or 'compound_smiles')
    smiles_col = "smiles" if "smiles" in row.index else "compound_smiles"

    # Molecular graph from SMILES (use pre-computed cache if available)
    graph_A = None
    if HAS_TORCH_GEOMETRIC and smiles_col in row.index and pd.notna(row[smiles_col]):
        smi = str(row[smiles_col])
        if graph_cache and smi in graph_cache:
            graph_A = graph_cache[smi]
        else:
            graph_A = smiles_to_graph(smi)

    # Token sequences — only protein target (seq_B), NOT SMILES (seq_A).
    # SMILES vocab (76 tokens) is incompatible with SequenceEncoder (vocab_size=26).
    # The drug is already encoded by graph_A via MolGraphEncoder.
    seq_A, seq_B = None, None
    if HAS_TOKENIZERS:
        if "target_sequence" in row.index and pd.notna(row["target_sequence"]):
            seq_B = dta_seq_to_tensor([str(row["target_sequence"])])[0]

    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=tabular_B,
        domain_id=DOMAIN_ID["dti"],
        label=_get_label(row),
        graph_A=graph_A,
        seq_A=seq_A,
        seq_B=seq_B,
    )


# ── PPI adapter ──────────────────────────────────────────────────────────────

def _ppi_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Protein-Protein Interaction: seq_A (protein A) + seq_B (protein B) + tabular."""
    _PPI_EXCLUDE = {"pair_id", "uniprot_id_1", "uniprot_id_2",
                    "sequence_1", "sequence_2", "gene_symbol_1", "gene_symbol_2",
                    "subcellular_location_1", "subcellular_location_2",
                    "protein_a_sequence", "protein_b_sequence",
                    "Y", "label", "confidence_tier"}
    tab_cols = [c for c in row.index if c.startswith(("feat_a_", "feat_b_", "aa_comp_"))]
    if not tab_cols:
        # Fallback: numeric non-label, non-string columns (e.g., protein1_degree, num_sources)
        tab_cols = [c for c in row.index
                    if _is_numeric_safe(row[c])
                    and not c.startswith("split_")
                    and c not in _PPI_EXCLUDE]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)
    tabular_B = torch.zeros(max_features)

    seq_A, seq_B = None, None
    if HAS_TOKENIZERS:
        # Export uses sequence_1/sequence_2 (or protein_a_sequence/protein_b_sequence)
        seq_a_col = "sequence_1" if "sequence_1" in row.index else "protein_a_sequence"
        seq_b_col = "sequence_2" if "sequence_2" in row.index else "protein_b_sequence"
        if seq_a_col in row.index and pd.notna(row[seq_a_col]):
            seq_A = ppi_seq_to_tensor([str(row[seq_a_col])])[0]
        if seq_b_col in row.index and pd.notna(row[seq_b_col]):
            seq_B = ppi_seq_to_tensor([str(row[seq_b_col])])[0]

    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=tabular_B,
        domain_id=DOMAIN_ID["ppi"],
        label=_get_label(row),
        seq_A=seq_A,
        seq_B=seq_B,
    )


# ── GE adapter ───────────────────────────────────────────────────────────────

def _ge_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Gene Essentiality: tabular only. Label derived from DepMap essentiality flags.

    Y=1 if is_common_essential==1 (~898K rows in export).
    Y=0 if is_reference_nonessential==1 (~934K rows in export).
    Neither → label=-1 sentinel (caller must filter these out before training).
    """
    # Derive label from DepMap essentiality flags (NegBioDB GE export has no Y column)
    is_essential = int(row.get("is_common_essential", 0) or 0)
    is_nonessential = int(row.get("is_reference_nonessential", 0) or 0)
    if is_essential:
        label = 1
    elif is_nonessential:
        label = 0
    else:
        label = -1  # unlabeled — filtered by from_data_root before training

    # Exclude known string/ID and label columns; only keep numeric features
    _GE_EXCLUDE = {"gene_symbol", "gene_id", "cell_line_id", "model_id",
                   "ccle_name", "lineage", "primary_disease", "pair_id",
                   "entrez_id", "best_evidence_type",
                   "is_common_essential", "is_reference_nonessential"}
    tab_cols = [c for c in row.index
                if c.startswith(("gene_", "cell_", "feat_"))
                and c not in _GE_EXCLUDE
                and _is_numeric_safe(row[c])]
    if not tab_cols:
        # Fallback: grab any numeric column not in label/split families
        tab_cols = [c for c in row.index
                    if _is_numeric_safe(row[c])
                    and not c.startswith("split_")
                    and c not in _GE_EXCLUDE
                    and c not in ("label", "outcome", "y", "Y", "is_positive", "pair_id")]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)
    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=torch.zeros(max_features),
        domain_id=DOMAIN_ID["ge"],
        label=label,
    )


# ── CT adapter ───────────────────────────────────────────────────────────────

def _ct_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Clinical Trial: tabular only (intervention + condition features)."""
    tab_cols = [c for c in row.index if c.startswith(("feat_", "moa_", "icd_", "trial_"))]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)
    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=torch.zeros(max_features),
        domain_id=DOMAIN_ID["ct"],
        label=_get_label(row),
    )


# ── VP adapter ───────────────────────────────────────────────────────────────

def _vp_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Variant Pathogenicity: ESM2 embeddings (esm2_A) + tabular (56 features)."""
    esm2_cols = [c for c in row.index if c.startswith("esm2_")]
    esm2_A = None
    if esm2_cols:
        arr = row[esm2_cols].values.astype(np.float32)
        np.nan_to_num(arr, nan=0.0, copy=False)
        esm2_A = torch.from_numpy(arr[:1280])

    tab_cols = [c for c in row.index if c.startswith(("cadd_", "revel_", "alpha_", "phylop_", "feat_"))]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)

    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=torch.zeros(max_features),
        domain_id=DOMAIN_ID["vp"],
        label=_get_label(row),
        esm2_A=esm2_A,
    )


# ── DC adapter ───────────────────────────────────────────────────────────────

def _dc_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Drug Combination: graph_A (drug_A) + graph_B (drug_B) + tabular."""
    tab_cols = [c for c in row.index if c.startswith(("feat_a_", "feat_b_", "rdkit_a_", "rdkit_b_"))]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)

    graph_A, graph_B = None, None
    if HAS_TORCH_GEOMETRIC:
        if "drug_a_smiles" in row.index and pd.notna(row["drug_a_smiles"]):
            smi_a = str(row["drug_a_smiles"])
            graph_A = graph_cache[smi_a] if graph_cache and smi_a in graph_cache else smiles_to_graph(smi_a)
        if "drug_b_smiles" in row.index and pd.notna(row["drug_b_smiles"]):
            smi_b = str(row["drug_b_smiles"])
            graph_B = graph_cache[smi_b] if graph_cache and smi_b in graph_cache else smiles_to_graph(smi_b)

    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=torch.zeros(max_features),
        domain_id=DOMAIN_ID["dc"],
        label=_get_label(row),
        graph_A=graph_A,
        graph_B=graph_B,
    )


# ── CP adapter ───────────────────────────────────────────────────────────────

def _cp_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Cell Painting: compound graph + tabular cell line features."""
    tab_cols = [c for c in row.index if c.startswith(("feat_", "cp_", "cell_"))]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)

    graph_A = None
    if HAS_TORCH_GEOMETRIC and "smiles" in row.index and pd.notna(row["smiles"]):
        smi = str(row["smiles"])
        graph_A = graph_cache[smi] if graph_cache and smi in graph_cache else smiles_to_graph(smi)

    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=torch.zeros(max_features),
        domain_id=DOMAIN_ID["cp"],
        label=_get_label(row),
        graph_A=graph_A,
    )


# ── MD adapter ───────────────────────────────────────────────────────────────

def _md_adapter(row: pd.Series, max_features: int, graph_cache: Optional[Dict] = None) -> NegJEPASample:
    """Metabolite-Disease: tabular only (2068 → reduced to max_features via PCA on HPC)."""
    tab_cols = [c for c in row.index if c.startswith(("feat_", "ecfp_", "physico_", "disease_"))]
    tabular_A = _get_tabular(row, tab_cols[:max_features], max_features) if tab_cols else torch.zeros(max_features)
    return NegJEPASample(
        tabular_A=tabular_A,
        tabular_B=torch.zeros(max_features),
        domain_id=DOMAIN_ID["md"],
        label=_get_label(row),
    )


DOMAIN_ADAPTERS: Dict[str, Callable[[pd.Series, int], NegJEPASample]] = {
    "dti": _dti_adapter,
    "ppi": _ppi_adapter,
    "ge":  _ge_adapter,
    "ct":  _ct_adapter,
    "vp":  _vp_adapter,
    "dc":  _dc_adapter,
    "cp":  _cp_adapter,
    "md":  _md_adapter,
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NegJEPADataset(torch.utils.data.Dataset):
    """Single-domain dataset backed by a parquet file.

    Uses lazy loading: stores the DataFrame in memory (compact columnar format)
    and converts rows to NegJEPASample on-the-fly in __getitem__.
    This avoids materializing 25M+ graph objects upfront.

    For smoke tests: pass parquet_path=None and provide a pre-built samples list.
    """

    def __init__(
        self,
        parquet_path: Optional[str],
        domain: str,
        max_features: int = 300,
        split: Optional[str] = None,
        split_col: Optional[str] = None,
        max_samples: int = 0,
        samples: Optional[List[NegJEPASample]] = None,
    ) -> None:
        """
        Args:
            parquet_path: path to parquet file, or None (must provide samples).
            domain:       one of DOMAIN_NAMES.
            max_features: tabular zero-padding target.
            split:        fold name to filter, e.g. "train", "val", "test".
            split_col:    column name containing fold assignments, e.g. "split_random".
                          If None, auto-detected: tries "split", then "split_random".
            max_samples:  cap sample count (0=no cap). Subsampled randomly with seed=42.
            samples:      pre-built list of NegJEPASample (used by smoke test / tests).
        """
        if domain not in DOMAIN_ADAPTERS:
            raise ValueError(f"Unknown domain {domain!r}. Valid: {list(DOMAIN_ADAPTERS)}")
        self.domain = domain
        self.max_features = max_features
        self._adapter = DOMAIN_ADAPTERS[domain]
        self._df: Optional[pd.DataFrame] = None
        self._samples: Optional[List[NegJEPASample]] = None

        if samples is not None:
            self._samples = samples
            return

        if parquet_path is None:
            raise ValueError("Either parquet_path or samples must be provided")

        # Read parquet — for large files, subsample via row groups to cap peak memory
        if max_samples > 0 and split is None:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(parquet_path)
            total_rows = pf.metadata.num_rows
            if total_rows > max_samples:
                # Read enough row groups to get max_samples rows
                tables = []
                rows_so_far = 0
                for i in range(pf.metadata.num_row_groups):
                    rg = pf.read_row_group(i)
                    tables.append(rg)
                    rows_so_far += rg.num_rows
                    if rows_so_far >= max_samples:
                        break
                import pyarrow as pa
                table = pa.concat_tables(tables).slice(0, max_samples)
                df = table.to_pandas()
            else:
                df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)

        if split is not None:
            col = _resolve_split_col(df, split_col)
            df = df[df[col] == split].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f"No samples for split={split!r} in column {col!r} of {parquet_path}")

        # Subsample if capped
        if max_samples > 0 and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Drop non-essential columns to reduce DataFrame memory footprint.
        # Keep: (1) prefix-matched feature cols, (2) known exact cols,
        # (3) any numeric column (adapters use _is_numeric_safe fallback)
        keep_prefixes = ("feat_", "rdkit_", "desc_", "fp_", "aa_comp_", "gene_", "cell_",
                         "moa_", "icd_", "trial_", "drug_", "compound_", "target_",
                         "esm2_", "split_", "num_", "is_", "protein", "has_")
        keep_exact = {"label", "Y", "y", "outcome", "is_positive", "pair_id",
                      "is_common_essential", "is_reference_nonessential",
                      "smiles", "compound_smiles", "target_sequence",
                      "sequence_1", "sequence_2", "protein_a_sequence", "protein_b_sequence",
                      "inchikey", "uniprot_id", "uniprot_id_1", "uniprot_id_2",
                      "mw", "logp", "tpsa", "hba", "hbd", "rotatable_bonds",
                      "gene_id", "cell_line_id", "model_id",
                      "ccle_name", "lineage", "primary_disease",
                      "earliest_year", "compound_degree", "target_degree",
                      "confidence_tier", "best_confidence"}
        cols_to_keep = [c for c in df.columns
                        if c in keep_exact or any(c.startswith(p) for p in keep_prefixes)]
        if cols_to_keep:
            df = df[cols_to_keep]
        self._df = df

        # Pre-compute SMILES→graph cache for graph-capable domains.
        # This avoids redundant RDKit parsing in __getitem__ (called by each DataLoader worker).
        # DTI: ~100K unique SMILES in 2M pairs → ~5 min one-time cost vs hours per epoch.
        self._graph_cache: Dict[str, Optional["Data"]] = {}
        if HAS_TORCH_GEOMETRIC and domain in ("dti", "dc", "cp"):
            smiles_cols = []
            for col_name in ("smiles", "compound_smiles", "drug_a_smiles", "drug_b_smiles"):
                if col_name in df.columns:
                    smiles_cols.append(col_name)
            if smiles_cols:
                unique_smiles = set()
                for col_name in smiles_cols:
                    unique_smiles.update(df[col_name].dropna().unique())
                import sys
                print(f"  Pre-computing {len(unique_smiles)} unique graphs for {domain}...",
                      flush=True, file=sys.stderr)
                for smi in unique_smiles:
                    try:
                        self._graph_cache[smi] = _smiles_to_graph_raw(smi)
                    except Exception:
                        self._graph_cache[smi] = None
                print(f"  Done. {len(self._graph_cache)} graphs cached.", flush=True, file=sys.stderr)

    def __len__(self) -> int:
        if self._samples is not None:
            return len(self._samples)
        return len(self._df)

    def __getitem__(self, idx: int) -> NegJEPASample:
        if self._samples is not None:
            return self._samples[idx]
        return self._adapter(self._df.iloc[idx], self.max_features,
                             graph_cache=self._graph_cache)

    @classmethod
    def from_data_root(
        cls,
        data_root: str,
        domain: str,
        max_features: int = 300,
        split: Optional[str] = None,
        split_col: Optional[str] = None,
        max_samples: int = 0,
    ) -> "NegJEPADataset":
        """Construct from data_root/{domain}_ml/{parquet_filename}.

        For GE domain: rows where neither is_common_essential nor
        is_reference_nonessential is set are filtered out (~20.7M of 22.5M rows).
        This yields ~1.83M labeled rows for binary classification.
        """
        if data_root == "synthetic":
            return cls._make_synthetic(domain, max_features, n=100)
        parquet_path = os.path.join(data_root, f"{domain}_ml", DOMAIN_PARQUET[domain])
        ds = cls(
            parquet_path=parquet_path, domain=domain, max_features=max_features,
            split=split, split_col=split_col, max_samples=max_samples,
        )
        # Filter GE unlabeled rows (label=-1 sentinel from _ge_adapter)
        if domain == "ge" and ds._df is not None:
            df = ds._df
            if "is_common_essential" in df.columns and "is_reference_nonessential" in df.columns:
                mask = (df["is_common_essential"].fillna(0).astype(int) == 1) | \
                       (df["is_reference_nonessential"].fillna(0).astype(int) == 1)
                n_before = len(df)
                ds._df = df[mask].reset_index(drop=True)
                import sys
                print(f"  GE: filtered {n_before - len(ds._df):,} unlabeled rows "
                      f"({len(ds._df):,} labeled remain)", flush=True, file=sys.stderr)
        return ds

    @classmethod
    def _make_synthetic(
        cls,
        domain: str,
        max_features: int,
        n: int = 100,
    ) -> "NegJEPADataset":
        """Generate synthetic samples for smoke tests (no parquet file required)."""
        rng = np.random.default_rng(seed=42)
        samples = []
        for i in range(n):
            tab_A = torch.from_numpy(rng.standard_normal(max_features).astype(np.float32))
            tab_B = torch.from_numpy(rng.standard_normal(max_features).astype(np.float32))
            sample = NegJEPASample(
                tabular_A=tab_A,
                tabular_B=tab_B,
                domain_id=DOMAIN_ID[domain],
                label=int(i % 2),
            )
            # Add graph for graph-capable domains
            if domain in ("dti", "dc", "cp") and HAS_TORCH_GEOMETRIC:
                from torch_geometric.data import Data
                n_atoms = rng.integers(5, 20)
                x = torch.rand(n_atoms, 82)
                # Random connected graph
                src = torch.randint(0, n_atoms, (n_atoms,))
                dst = torch.randint(0, n_atoms, (n_atoms,))
                edge_index = torch.stack([src, dst], dim=0)
                sample.graph_A = Data(x=x, edge_index=edge_index)
            samples.append(sample)
        return cls(parquet_path=None, domain=domain, max_features=max_features, samples=samples)


# ─── MultiDomainSampler ───────────────────────────────────────────────────────

class MultiDomainSampler(torch.utils.data.Sampler):
    """Temperature-scaled sampler that yields (domain_key, local_idx) tuples.

    P(domain_d) ∝ |D_d|^temp   (temp=0.5 → sqrt-scaled; 1.0 → proportional)

    Each epoch iterates through the approximate total dataset size with resampling
    for small domains so no domain is completely ignored.
    """

    def __init__(
        self,
        datasets: Dict[str, NegJEPADataset],
        temp: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.datasets = datasets
        self.domain_keys = list(datasets.keys())
        self.temp = temp
        self.seed = seed

        # Compute sampling weights
        sizes = np.array([len(datasets[k]) for k in self.domain_keys], dtype=np.float64)
        weights = sizes ** temp
        self.probs = weights / weights.sum()

        self._total = int(sizes.sum())

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[tuple]:
        rng = np.random.default_rng(self.seed)
        # Sample domain indices according to temperature-scaled probabilities
        domain_choices = rng.choice(len(self.domain_keys), size=self._total, p=self.probs)
        for d_idx in domain_choices:
            key = self.domain_keys[d_idx]
            local_idx = int(rng.integers(0, len(self.datasets[key])))
            yield (key, local_idx)


# ─── MultiDomainDataset wrapper ───────────────────────────────────────────────

class MultiDomainDataset(torch.utils.data.Dataset):
    """Thin wrapper that resolves (domain_key, local_idx) tuples from MultiDomainSampler."""

    def __init__(self, datasets: Dict[str, NegJEPADataset]) -> None:
        self.datasets = datasets
        # Flat index mapping: list of (domain_key, local_idx) for __getitem__
        self._index: List[tuple] = []
        for key, ds in datasets.items():
            for i in range(len(ds)):
                self._index.append((key, i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> NegJEPASample:
        key, local_idx = self._index[idx]
        return self.datasets[key][local_idx]


# ─── Collate function ─────────────────────────────────────────────────────────

def jepa_collate_fn(samples: List[NegJEPASample]) -> dict:
    """PyG-aware batching for NegJEPASample lists.

    Tensors (tabular_A/B, domain_id, label, seq_A/B, esm2_A) are stacked.
    PyG graphs are batched with Batch.from_data_list when present.
    None fields are omitted from the returned dict.
    The result is the batch dict consumed by UnifiedEncoder.forward().
    """
    batch: dict = {}

    # Always-present fields
    batch["tabular_A"] = torch.stack([s.tabular_A for s in samples])
    batch["tabular_B"] = torch.stack([s.tabular_B for s in samples])
    batch["domain_id"] = torch.tensor([s.domain_id for s in samples], dtype=torch.long)
    batch["label"] = torch.tensor([s.label for s in samples], dtype=torch.long)

    # Sequence fields (only if at least one sample has them)
    if any(s.seq_A is not None for s in samples):
        # Pad sequences to max length in this batch
        seqs = [s.seq_A for s in samples if s.seq_A is not None]
        max_L = max(s.shape[0] for s in seqs)
        padded = torch.zeros(len(samples), max_L, dtype=torch.long)
        for i, s in enumerate(samples):
            if s.seq_A is not None:
                padded[i, :s.seq_A.shape[0]] = s.seq_A
        batch["seq_A"] = padded

    if any(s.seq_B is not None for s in samples):
        seqs = [s.seq_B for s in samples if s.seq_B is not None]
        max_L = max(s.shape[0] for s in seqs)
        padded = torch.zeros(len(samples), max_L, dtype=torch.long)
        for i, s in enumerate(samples):
            if s.seq_B is not None:
                padded[i, :s.seq_B.shape[0]] = s.seq_B
        batch["seq_B"] = padded

    # ESM2 embeddings (VP domain)
    if any(s.esm2_A is not None for s in samples):
        esm2 = torch.stack([
            s.esm2_A if s.esm2_A is not None else torch.zeros(1280)
            for s in samples
        ])
        batch["esm2_A"] = esm2

    # PyG molecular graphs
    if HAS_TORCH_GEOMETRIC and any(s.graph_A is not None for s in samples):
        valid_graphs = [s.graph_A for s in samples if s.graph_A is not None]
        if valid_graphs:
            batch["graph_A"] = Batch.from_data_list(valid_graphs)
            # Store mapping: for each batch sample, does it have a graph?
            batch["has_graph_A"] = torch.tensor(
                [s.graph_A is not None for s in samples], dtype=torch.bool
            )

    if HAS_TORCH_GEOMETRIC and any(s.graph_B is not None for s in samples):
        valid_graphs = [s.graph_B for s in samples if s.graph_B is not None]
        if valid_graphs:
            batch["graph_B"] = Batch.from_data_list(valid_graphs)
            batch["has_graph_B"] = torch.tensor(
                [s.graph_B is not None for s in samples], dtype=torch.bool
            )

    return batch


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    cfg: "JEPAConfig",
    split: Optional[str] = None,
) -> tuple[torch.utils.data.DataLoader, Dict[str, NegJEPADataset]]:
    """Build a multi-domain DataLoader from cfg.

    Args:
        cfg:   JEPAConfig with domains, data_root, batch_size, etc.
        split: if provided, filter each parquet to this split column value.

    Returns:
        (dataloader, domain_datasets_dict)
    """
    from negbiojepa.config import JEPAConfig  # avoid circular at module level

    datasets: Dict[str, NegJEPADataset] = {}
    for domain in cfg.domains:
        ds = NegJEPADataset.from_data_root(
            cfg.data_root, domain, cfg.tabular_max_features, split=split,
            max_samples=cfg.max_samples_per_domain,
        )
        datasets[domain] = ds

    combined = MultiDomainDataset(datasets)
    loader = torch.utils.data.DataLoader(
        combined,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=jepa_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return loader, datasets

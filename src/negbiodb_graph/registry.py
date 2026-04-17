"""Registry of graph-capable domains and their default database paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomainSpec:
    code: str
    label: str
    default_db_path: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()

DOMAIN_REGISTRY: dict[str, DomainSpec] = {
    "dti": DomainSpec("dti", "Drug-Target Interaction", PROJECT_ROOT / "data" / "negbiodb.db"),
    "ct": DomainSpec("ct", "Clinical Trial Failure", PROJECT_ROOT / "data" / "negbiodb_ct.db"),
    "ppi": DomainSpec("ppi", "Protein-Protein Interaction", PROJECT_ROOT / "data" / "negbiodb_ppi.db"),
    "ge": DomainSpec("ge", "Gene Essentiality", PROJECT_ROOT / "data" / "negbiodb_depmap.db"),
    "vp": DomainSpec("vp", "Variant Pathogenicity", PROJECT_ROOT / "data" / "negbiodb_vp.db"),
    "md": DomainSpec("md", "Metabolomics-Disease", PROJECT_ROOT / "data" / "negbiodb_md.db"),
    "dc": DomainSpec("dc", "Drug Combination Synergy", PROJECT_ROOT / "data" / "negbiodb_dc.db"),
    "cp": DomainSpec("cp", "Cell Painting", PROJECT_ROOT / "data" / "negbiodb_cp.db"),
}

GRAPH_DOMAIN_ORDER = ["dti", "ct", "ppi", "ge", "vp", "md", "dc", "cp"]

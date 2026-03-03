"""Shared RDKit compound standardization for NegBioDB.

Provides a single standardize_smiles() function used by all ETL modules
to ensure consistent InChIKey generation and descriptor computation.
"""

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# PAINS filter catalog (initialized once at module load)
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(_pains_params)


def standardize_smiles(smiles: str) -> dict | None:
    """Standardize a SMILES string using RDKit.

    Computes canonical SMILES, InChI, InChIKey, and molecular descriptors.
    Returns None if SMILES fails to parse or InChI generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)
    inchi = Chem.MolToInchi(mol)
    if inchi is None:
        return None
    inchikey = Chem.InchiToInchiKey(inchi)
    if inchikey is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    lipinski = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    pains = 1 if PAINS_CATALOG.HasMatch(mol) else 0

    return {
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "inchikey_connectivity": inchikey[:14],
        "inchi": inchi,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "qed": round(QED.qed(mol), 4),
        "pains_alert": pains,
        "lipinski_violations": lipinski,
    }

"""Build the derived NegBioGraph database from domain SQLite sources."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH, create_graph_database, get_connection
from negbiodb_graph.reference import ReferenceFeed, load_reference_manifest, manifest_summary
from negbiodb_graph.registry import DOMAIN_REGISTRY, GRAPH_DOMAIN_ORDER
from negbiodb_graph.store import GraphStore, ResolvedEntity
from negbiodb_graph.utils import (
    anchor_key,
    as_jsonable,
    context_hash,
    normalize_name_key,
    normalize_text,
    sha256_file,
    stable_json_dumps,
)


def discover_domain_paths(overrides: dict[str, str | Path] | None = None) -> dict[str, Path]:
    """Return domain DB paths, preferring explicit overrides over defaults."""
    overrides = overrides or {}
    result = {}
    for code in GRAPH_DOMAIN_ORDER:
        if code in overrides and overrides[code] is not None:
            result[code] = Path(overrides[code]).expanduser().resolve()
        else:
            result[code] = DOMAIN_REGISTRY[code].default_db_path
    return result


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


class GraphBuilder:
    """Stateful builder that ingests all supported domains into the graph DB."""

    def __init__(
        self,
        graph_db_path: str | Path = DEFAULT_GRAPH_DB_PATH,
        *,
        strict: bool = False,
        build_tag: str | None = None,
        manifest_path: str | Path | None = None,
    ):
        self.graph_db_path = Path(graph_db_path)
        self.strict = strict
        self.build_tag = build_tag
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        create_graph_database(self.graph_db_path)
        self.conn = get_connection(self.graph_db_path)
        self.store: GraphStore | None = None
        self.build_id: int | None = None

    def close(self) -> None:
        self.conn.close()

    def _wipe_graph(self) -> None:
        for table in [
            "graph_contradiction_members",
            "graph_contradiction_groups",
            "graph_claim_entities",
            "graph_evidence",
            "graph_claim_rollups",
            "graph_claims",
            "graph_bridges",
            "graph_entity_aliases",
            "graph_entities",
            "graph_build_inputs",
            "graph_builds",
        ]:
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def _start_build(self) -> int:
        self.conn.execute(
            """INSERT INTO graph_builds (build_tag, strict_mode, manifest_path, notes)
               VALUES (?, ?, ?, ?)""",
            (
                self.build_tag,
                1 if self.strict else 0,
                str(self.manifest_path) if self.manifest_path else None,
                "Derived NegBioGraph build",
            ),
        )
        return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def _finish_build(self, status: str, notes: str | None = None) -> None:
        self.conn.execute(
            """UPDATE graph_builds
               SET status = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                   notes = COALESCE(?, notes)
               WHERE build_id = ?""",
            (status, notes, self.build_id),
        )
        self.conn.commit()

    def _record_input(
        self,
        *,
        input_name: str,
        input_kind: str,
        domain_code: str | None,
        path: Path | None,
        is_required: bool,
        is_available: bool,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        checksum = sha256_file(path) if path and path.exists() else None
        self.conn.execute(
            """INSERT INTO graph_build_inputs
               (build_id, input_name, input_kind, domain_code, path, is_required,
                is_available, checksum_sha256, status, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.build_id,
                input_name,
                input_kind,
                domain_code,
                str(path) if path else None,
                1 if is_required else 0,
                1 if is_available else 0,
                checksum,
                status,
                stable_json_dumps(as_jsonable(metadata)) if metadata is not None else None,
            ),
        )

    def _assert_available(self, code: str, path: Path) -> None:
        if self.strict and not path.exists():
            raise FileNotFoundError(f"Required domain DB missing for {code}: {path}")

    def _resolved(self, entity_type: str, canonical_key: str) -> ResolvedEntity:
        assert self.store is not None
        return self.store.upsert_entity(entity_type, canonical_key)

    def resolve_small_molecule(
        self,
        *,
        domain_code: str,
        inchikey: str | None = None,
        connectivity: str | None = None,
        chembl_id: str | None = None,
        pubchem_cid: int | None = None,
        smiles: str | None = None,
        name: str | None = None,
    ) -> ResolvedEntity:
        assert self.store is not None
        inchikey = normalize_text(inchikey)
        connectivity = normalize_text(connectivity) or (inchikey[:14] if inchikey else None)
        chembl_id = normalize_text(chembl_id)
        name_key = normalize_name_key(name)
        entity = None
        if inchikey:
            entity = self.store.find_entity_by_alias("small_molecule", "inchikey", inchikey)
        if entity is None and connectivity:
            entity = self.store.find_entity_by_alias("small_molecule", "connectivity", connectivity)
        if entity is None and chembl_id:
            entity = self.store.find_entity_by_alias("small_molecule", "chembl_id", chembl_id)
        if entity is None and pubchem_cid is not None:
            entity = self.store.find_entity_by_alias("small_molecule", "pubchem_cid", str(pubchem_cid))
        if entity is None and name_key:
            entity = self.store.find_entity_by_alias("small_molecule", "normalized_name", name_key)

        if entity is None:
            if inchikey:
                canonical_key = f"inchikey:{inchikey}"
            elif connectivity:
                canonical_key = f"connectivity:{connectivity}"
            elif chembl_id:
                canonical_key = f"chembl:{chembl_id}"
            elif pubchem_cid is not None:
                canonical_key = f"pubchem:{pubchem_cid}"
            elif name_key:
                canonical_key = f"name:{name_key}"
            else:
                canonical_key = f"{domain_code}:small_molecule:unknown"
            entity = self.store.upsert_entity(
                "small_molecule",
                canonical_key,
                display_name=normalize_text(name),
                primary_domain=domain_code,
                attrs={"smiles": smiles},
            )
        self.store.add_alias(entity, "inchikey", inchikey, source_domain=domain_code)
        self.store.add_alias(entity, "connectivity", connectivity, source_domain=domain_code)
        self.store.add_alias(entity, "chembl_id", chembl_id, source_domain=domain_code)
        self.store.add_alias(entity, "pubchem_cid", str(pubchem_cid) if pubchem_cid is not None else None, source_domain=domain_code)
        self.store.add_alias(entity, "normalized_name", name_key, source_domain=domain_code)
        self.store.add_alias(entity, "smiles", normalize_text(smiles), source_domain=domain_code)
        return entity

    def resolve_gene(
        self,
        *,
        domain_code: str,
        entrez_id: int | None = None,
        hgnc_id: str | None = None,
        ensembl_id: str | None = None,
        gene_symbol: str | None = None,
        description: str | None = None,
    ) -> ResolvedEntity:
        assert self.store is not None
        gene_symbol = normalize_text(gene_symbol)
        symbol_key = normalize_name_key(gene_symbol)
        candidates = []
        if entrez_id is not None:
            candidates.append(("entrez_id", str(entrez_id), f"entrez:{entrez_id}"))
        if hgnc_id:
            candidates.append(("hgnc_id", hgnc_id, f"hgnc:{hgnc_id}"))
        if ensembl_id:
            candidates.append(("ensembl_id", ensembl_id, f"ensembl:{ensembl_id}"))
        if symbol_key:
            candidates.append(("gene_symbol", symbol_key, f"symbol:{symbol_key}"))

        entity = None
        for alias_type, alias_value, canonical_key in candidates:
            entity = self.store.find_entity("gene", canonical_key)
            if entity is None:
                entity = self.store.find_entity_by_alias("gene", alias_type, alias_value)
            if entity is not None:
                break
        if entity is None:
            alias_type, alias_value, canonical_key = candidates[0] if candidates else ("domain_local_id", f"{domain_code}:unknown_gene", f"{domain_code}:unknown_gene")
            entity = self.store.upsert_entity(
                "gene",
                canonical_key,
                display_name=gene_symbol,
                primary_domain=domain_code,
                attrs={"description": description},
            )
        self.store.add_alias(entity, "entrez_id", str(entrez_id) if entrez_id is not None else None, source_domain=domain_code)
        self.store.add_alias(entity, "hgnc_id", hgnc_id, source_domain=domain_code)
        self.store.add_alias(entity, "ensembl_id", ensembl_id, source_domain=domain_code)
        self.store.add_alias(entity, "gene_symbol", symbol_key, source_domain=domain_code)
        return entity

    def resolve_protein(
        self,
        *,
        domain_code: str,
        uniprot_accession: str | None,
        gene_symbol: str | None = None,
        sequence: str | None = None,
    ) -> tuple[ResolvedEntity, ResolvedEntity | None]:
        assert self.store is not None
        accession = normalize_text(uniprot_accession)
        if accession:
            entity = self.store.find_entity("protein", f"uniprot:{accession}")
            if entity is None:
                entity = self.store.upsert_entity(
                    "protein",
                    f"uniprot:{accession}",
                    display_name=accession,
                    primary_domain=domain_code,
                    attrs={"sequence_length": len(sequence) if sequence else None},
                )
        else:
            entity = self.store.upsert_entity(
                "protein",
                f"{domain_code}:protein:unknown",
                display_name=gene_symbol,
                primary_domain=domain_code,
            )
        self.store.add_alias(entity, "uniprot_accession", accession, source_domain=domain_code)
        gene_entity = None
        if gene_symbol:
            gene_entity = self.resolve_gene(domain_code=domain_code, gene_symbol=gene_symbol)
            self.store.add_bridge(
                entity,
                gene_entity,
                bridge_type="protein_gene_link",
                source_domain=domain_code,
                method="gene_symbol_hint",
                confidence_score=0.7,
                metadata={"gene_symbol": gene_symbol},
            )
        return entity, gene_entity

    def resolve_disease(
        self,
        *,
        domain_code: str,
        name: str | None,
        mondo_id: str | None = None,
        omim_id: str | None = None,
        medgen_cui: str | None = None,
        mesh_id: str | None = None,
        do_id: str | None = None,
        icd10_code: str | None = None,
    ) -> ResolvedEntity:
        assert self.store is not None
        name = normalize_text(name)
        name_key = normalize_name_key(name)
        candidates = []
        for alias_type, value, prefix in [
            ("mondo_id", mondo_id, "mondo"),
            ("omim_id", omim_id, "omim"),
            ("medgen_cui", medgen_cui, "medgen"),
            ("mesh_id", mesh_id, "mesh"),
            ("do_id", do_id, "do"),
            ("icd10_code", icd10_code, "icd10"),
            ("normalized_name", name_key, "name"),
        ]:
            if value:
                candidates.append((alias_type, value, f"{prefix}:{value}"))

        entity = None
        for alias_type, alias_value, canonical_key in candidates:
            entity = self.store.find_entity("disease", canonical_key)
            if entity is None:
                entity = self.store.find_entity_by_alias("disease", alias_type, alias_value)
            if entity is not None:
                break
        if entity is None:
            canonical_key = candidates[0][2] if candidates else f"{domain_code}:disease:unknown"
            entity = self.store.upsert_entity(
                "disease",
                canonical_key,
                display_name=name,
                primary_domain=domain_code,
            )
        self.store.add_alias(entity, "mondo_id", mondo_id, source_domain=domain_code)
        self.store.add_alias(entity, "omim_id", omim_id, source_domain=domain_code)
        self.store.add_alias(entity, "medgen_cui", medgen_cui, source_domain=domain_code)
        self.store.add_alias(entity, "mesh_id", mesh_id, source_domain=domain_code)
        self.store.add_alias(entity, "do_id", do_id, source_domain=domain_code)
        self.store.add_alias(entity, "icd10_code", icd10_code, source_domain=domain_code)
        self.store.add_alias(entity, "normalized_name", name_key, source_domain=domain_code)
        return entity

    def resolve_cell_line(
        self,
        *,
        domain_code: str,
        depmap_model_id: str | None = None,
        cosmic_id: int | None = None,
        name: str | None = None,
        lineage: str | None = None,
    ) -> ResolvedEntity:
        assert self.store is not None
        name = normalize_text(name)
        name_key = normalize_name_key(name)
        candidates = []
        if depmap_model_id:
            candidates.append(("depmap_model_id", depmap_model_id, f"depmap:{depmap_model_id}"))
        if cosmic_id is not None:
            candidates.append(("cosmic_id", str(cosmic_id), f"cosmic:{cosmic_id}"))
        if name_key:
            candidates.append(("normalized_name", name_key, f"name:{name_key}"))
        entity = None
        for alias_type, alias_value, canonical_key in candidates:
            entity = self.store.find_entity("cell_line", canonical_key)
            if entity is None:
                entity = self.store.find_entity_by_alias("cell_line", alias_type, alias_value)
            if entity is not None:
                break
        if entity is None:
            canonical_key = candidates[0][2] if candidates else f"{domain_code}:cell_line:unknown"
            entity = self.store.upsert_entity(
                "cell_line",
                canonical_key,
                display_name=name,
                primary_domain=domain_code,
                attrs={"lineage": lineage},
            )
        self.store.add_alias(entity, "depmap_model_id", depmap_model_id, source_domain=domain_code)
        self.store.add_alias(entity, "cosmic_id", str(cosmic_id) if cosmic_id is not None else None, source_domain=domain_code)
        self.store.add_alias(entity, "normalized_name", name_key, source_domain=domain_code)
        return entity

    def resolve_variant(
        self,
        *,
        domain_code: str,
        chromosome: str | None,
        position: int | None,
        ref: str | None,
        alt: str | None,
        display_name: str | None = None,
    ) -> ResolvedEntity:
        coords = None
        if chromosome and position is not None and ref and alt:
            coords = f"{chromosome}:{position}:{ref}>{alt}"
        canonical = f"variant:{coords}" if coords else f"{domain_code}:variant:{normalize_name_key(display_name) or 'unknown'}"
        entity = self.store.upsert_entity(
            "variant",
            canonical,
            display_name=display_name,
            primary_domain=domain_code,
        )
        self.store.add_alias(entity, "variant_display", normalize_text(display_name), source_domain=domain_code)
        return entity

    def resolve_local_entity(
        self,
        *,
        entity_type: str,
        domain_code: str,
        local_id: Any,
        display_name: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> ResolvedEntity:
        assert self.store is not None
        entity = self.store.upsert_entity(
            entity_type,
            f"{domain_code}:{entity_type}:{local_id}",
            display_name=display_name,
            primary_domain=domain_code,
            attrs=attrs,
        )
        self.store.add_alias(entity, f"{domain_code}_local_id", _safe_str(local_id), source_domain=domain_code)
        return entity

    def _claim(
        self,
        *,
        domain_code: str,
        family: str,
        label: str,
        anchor: str,
        base_anchor: str,
        context: dict[str, Any],
        text: str | None = None,
        level: str = "raw",
    ) -> int:
        assert self.store is not None
        return self.store.upsert_claim(
            domain_code=domain_code,
            claim_family=family,
            claim_label=label,
            anchor_key=anchor,
            base_anchor_key=base_anchor,
            context_hash=context_hash(context),
            context=context,
            claim_text=text,
            claim_level=level,
        )

    def _load_domain_frame(self, db_path: Path, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        conn = sqlite3.connect(str(db_path))
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def ingest_dti(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                nr.result_id, nr.assay_id, nr.result_type, nr.confidence_tier,
                nr.activity_type, nr.activity_value, nr.source_db, nr.source_record_id,
                nr.extraction_method, nr.publication_year,
                c.canonical_smiles, c.inchikey, c.inchikey_connectivity,
                c.chembl_id, c.pubchem_cid,
                t.uniprot_accession, t.gene_symbol, t.amino_acid_sequence,
                tv.variant_id AS target_variant_id, tv.variant_label
            FROM negative_results nr
            JOIN compounds c ON nr.compound_id = c.compound_id
            JOIN targets t ON nr.target_id = t.target_id
            LEFT JOIN target_variants tv ON nr.variant_id = tv.variant_id
            """,
        )
        for row in df.to_dict(orient="records"):
            mol = self.resolve_small_molecule(
                domain_code="dti",
                inchikey=row.get("inchikey"),
                connectivity=row.get("inchikey_connectivity"),
                chembl_id=row.get("chembl_id"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("canonical_smiles"),
            )
            protein, gene = self.resolve_protein(
                domain_code="dti",
                uniprot_accession=row.get("uniprot_accession"),
                gene_symbol=row.get("gene_symbol"),
                sequence=row.get("amino_acid_sequence"),
            )
            assay = None
            if row.get("assay_id") is not None:
                assay = self.resolve_local_entity(
                    entity_type="assay",
                    domain_code="dti",
                    local_id=int(row["assay_id"]),
                    display_name=f"DTI assay {int(row['assay_id'])}",
                )
            variant = None
            if row.get("target_variant_id") is not None:
                variant = self.resolve_local_entity(
                    entity_type="target_variant",
                    domain_code="dti",
                    local_id=int(row["target_variant_id"]),
                    display_name=row.get("variant_label"),
                )
                self.store.add_bridge(
                    variant,
                    protein,
                    bridge_type="target_variant_of",
                    source_domain="dti",
                    method="negative_results.variant_id",
                    confidence_score=1.0,
                )
            base = anchor_key(mol.canonical_key, protein.canonical_key)
            anchor = anchor_key(base, variant.canonical_key if variant else None)
            context = {
                "assay_id": row.get("assay_id"),
                "result_type": row.get("result_type"),
                "variant_label": row.get("variant_label"),
            }
            claim_id = self._claim(
                domain_code="dti",
                family="binding",
                label="inactive_against",
                anchor=anchor,
                base_anchor=base,
                context=context,
                text=f"{mol.canonical_key} inactive against {protein.canonical_key}",
            )
            self.store.add_claim_entity(claim_id, mol, role="subject")
            self.store.add_claim_entity(claim_id, protein, role="object")
            if gene is not None:
                self.store.add_claim_entity(claim_id, gene, role="object_gene")
            if assay is not None:
                self.store.add_claim_entity(claim_id, assay, role="context_assay")
            if variant is not None:
                self.store.add_claim_entity(claim_id, variant, role="context_variant")
            self.store.add_evidence(
                claim_id,
                source_domain="dti",
                source_table="negative_results",
                source_record_id=_safe_str(row.get("source_record_id") or row.get("result_id")),
                source_db=row.get("source_db"),
                extraction_method=row.get("extraction_method"),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("result_type"),
                publication_year=int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                payload={"result_id": row.get("result_id"), "activity_type": row.get("activity_type"), "activity_value": row.get("activity_value")},
                provenance={"domain_row_id": row.get("result_id")},
            )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                ctp.pair_id, ctp.num_assays, ctp.num_sources, ctp.best_confidence,
                ctp.best_result_type, ctp.earliest_year, ctp.compound_degree, ctp.target_degree,
                c.inchikey, c.inchikey_connectivity, c.chembl_id, c.pubchem_cid, c.canonical_smiles,
                t.uniprot_accession, t.gene_symbol, t.amino_acid_sequence
            FROM compound_target_pairs ctp
            JOIN compounds c ON ctp.compound_id = c.compound_id
            JOIN targets t ON ctp.target_id = t.target_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            mol = self.resolve_small_molecule(
                domain_code="dti",
                inchikey=row.get("inchikey"),
                connectivity=row.get("inchikey_connectivity"),
                chembl_id=row.get("chembl_id"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("canonical_smiles"),
            )
            protein, _ = self.resolve_protein(
                domain_code="dti",
                uniprot_accession=row.get("uniprot_accession"),
                gene_symbol=row.get("gene_symbol"),
                sequence=row.get("amino_acid_sequence"),
            )
            self.store.add_rollup(
                domain_code="dti",
                source_table="compound_target_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="binding",
                anchor_key=anchor_key(mol.canonical_key, protein.canonical_key),
                rollup_type="pair_aggregation",
                summary={
                    "num_assays": row.get("num_assays"),
                    "num_sources": row.get("num_sources"),
                    "best_confidence": row.get("best_confidence"),
                    "best_result_type": row.get("best_result_type"),
                    "earliest_year": row.get("earliest_year"),
                    "compound_degree": row.get("compound_degree"),
                    "target_degree": row.get("target_degree"),
                },
            )

    def ingest_ct(self, db_path: Path) -> None:
        assert self.store is not None
        target_rows = self._load_domain_frame(
            db_path,
            """
            SELECT intervention_id, uniprot_accession, gene_symbol, source
            FROM intervention_targets
            """,
        )
        targets_by_intervention: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in target_rows.to_dict(orient="records"):
            targets_by_intervention[int(row["intervention_id"])].append(row)

        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                tfr.result_id, tfr.intervention_id, tfr.condition_id, tfr.trial_id,
                tfr.failure_category, tfr.confidence_tier, tfr.source_db, tfr.source_record_id,
                tfr.extraction_method, tfr.publication_year, tfr.highest_phase_reached,
                i.intervention_name, i.canonical_name, i.chembl_id, i.pubchem_cid,
                i.inchikey, i.inchikey_connectivity, i.canonical_smiles,
                c.condition_name, c.canonical_name AS condition_canonical_name,
                c.mesh_id, c.do_id, c.icd10_code,
                ct.source_trial_id, ct.trial_phase, ct.primary_completion_date
            FROM trial_failure_results tfr
            JOIN interventions i ON tfr.intervention_id = i.intervention_id
            JOIN conditions c ON tfr.condition_id = c.condition_id
            LEFT JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
            """,
        )
        for row in df.to_dict(orient="records"):
            intervention = self.resolve_local_entity(
                entity_type="intervention",
                domain_code="ct",
                local_id=int(row["intervention_id"]),
                display_name=row.get("canonical_name") or row.get("intervention_name"),
                attrs={"intervention_name": row.get("intervention_name")},
            )
            self.store.add_alias(intervention, "chembl_id", row.get("chembl_id"), source_domain="ct")
            self.store.add_alias(intervention, "pubchem_cid", _safe_str(row.get("pubchem_cid")) if row.get("pubchem_cid") is not None else None, source_domain="ct")
            chemical = None
            if row.get("inchikey") or row.get("inchikey_connectivity") or row.get("chembl_id") or row.get("pubchem_cid") is not None:
                chemical = self.resolve_small_molecule(
                    domain_code="ct",
                    inchikey=row.get("inchikey"),
                    connectivity=row.get("inchikey_connectivity"),
                    chembl_id=row.get("chembl_id"),
                    pubchem_cid=row.get("pubchem_cid"),
                    smiles=row.get("canonical_smiles"),
                    name=row.get("canonical_name") or row.get("intervention_name"),
                )
                self.store.add_bridge(
                    intervention,
                    chemical,
                    bridge_type="chemical_equivalence",
                    source_domain="ct",
                    method="interventions.inchikey_connectivity",
                    confidence_score=1.0,
                )
            disease = self.resolve_disease(
                domain_code="ct",
                name=row.get("condition_canonical_name") or row.get("condition_name"),
                mesh_id=row.get("mesh_id"),
                do_id=row.get("do_id"),
                icd10_code=row.get("icd10_code"),
            )
            trial = None
            if row.get("trial_id") is not None:
                trial = self.resolve_local_entity(
                    entity_type="trial",
                    domain_code="ct",
                    local_id=int(row["trial_id"]),
                    display_name=row.get("source_trial_id"),
                    attrs={"phase": row.get("trial_phase")},
                )
            base = anchor_key(intervention.canonical_key, disease.canonical_key)
            anchor = anchor_key(base, f"trial:{row['trial_id']}" if row.get("trial_id") is not None else None)
            context = {
                "trial_id": row.get("trial_id"),
                "failure_category": row.get("failure_category"),
                "highest_phase_reached": row.get("highest_phase_reached"),
            }
            claim_id = self._claim(
                domain_code="ct",
                family="trial_outcome",
                label="failed_for",
                anchor=anchor,
                base_anchor=base,
                context=context,
                text=f"{intervention.canonical_key} failed for {disease.canonical_key}",
            )
            self.store.add_claim_entity(claim_id, intervention, role="subject")
            self.store.add_claim_entity(claim_id, disease, role="object")
            if chemical is not None:
                self.store.add_claim_entity(claim_id, chemical, role="subject_chemical")
            if trial is not None:
                self.store.add_claim_entity(claim_id, trial, role="context_trial")
            for target in targets_by_intervention.get(int(row["intervention_id"]), []):
                protein, gene = self.resolve_protein(
                    domain_code="ct",
                    uniprot_accession=target.get("uniprot_accession"),
                    gene_symbol=target.get("gene_symbol"),
                )
                self.store.add_claim_entity(claim_id, protein, role="mediator_protein")
                self.store.add_bridge(
                    intervention,
                    protein,
                    bridge_type="intervention_targets",
                    source_domain="ct",
                    method="intervention_targets",
                    confidence_score=1.0,
                )
                if gene is not None:
                    self.store.add_claim_entity(claim_id, gene, role="mediator_gene")
            self.store.add_evidence(
                claim_id,
                source_domain="ct",
                source_table="trial_failure_results",
                source_record_id=_safe_str(row.get("source_record_id") or row.get("result_id")),
                source_db=row.get("source_db"),
                extraction_method=row.get("extraction_method"),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("failure_category"),
                publication_year=int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                event_date=row.get("primary_completion_date"),
                payload={"result_id": row.get("result_id"), "trial_id": row.get("trial_id")},
                provenance={"domain_row_id": row.get("result_id")},
            )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                icp.pair_id, icp.num_trials, icp.num_sources, icp.best_confidence,
                icp.primary_failure_category, icp.earliest_year, icp.highest_phase_reached,
                i.intervention_id, i.intervention_name, i.canonical_name, i.chembl_id,
                i.pubchem_cid, i.inchikey, i.inchikey_connectivity, i.canonical_smiles,
                c.condition_name, c.canonical_name AS condition_canonical_name,
                c.mesh_id, c.do_id, c.icd10_code
            FROM intervention_condition_pairs icp
            JOIN interventions i ON icp.intervention_id = i.intervention_id
            JOIN conditions c ON icp.condition_id = c.condition_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            intervention = self.resolve_local_entity(
                entity_type="intervention",
                domain_code="ct",
                local_id=int(row["intervention_id"]),
                display_name=row.get("canonical_name") or row.get("intervention_name"),
            )
            disease = self.resolve_disease(
                domain_code="ct",
                name=row.get("condition_canonical_name") or row.get("condition_name"),
                mesh_id=row.get("mesh_id"),
                do_id=row.get("do_id"),
                icd10_code=row.get("icd10_code"),
            )
            self.store.add_rollup(
                domain_code="ct",
                source_table="intervention_condition_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="trial_outcome",
                anchor_key=anchor_key(intervention.canonical_key, disease.canonical_key),
                rollup_type="pair_aggregation",
                summary={
                    "num_trials": row.get("num_trials"),
                    "num_sources": row.get("num_sources"),
                    "best_confidence": row.get("best_confidence"),
                    "primary_failure_category": row.get("primary_failure_category"),
                    "earliest_year": row.get("earliest_year"),
                    "highest_phase_reached": row.get("highest_phase_reached"),
                },
            )

    def ingest_ppi(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                nr.result_id, nr.experiment_id, nr.evidence_type, nr.confidence_tier,
                nr.source_db, nr.source_record_id, nr.extraction_method, nr.publication_year,
                p1.uniprot_accession AS uniprot_id_1, p1.gene_symbol AS gene_symbol_1,
                p1.amino_acid_sequence AS seq_1,
                p2.uniprot_accession AS uniprot_id_2, p2.gene_symbol AS gene_symbol_2,
                p2.amino_acid_sequence AS seq_2
            FROM ppi_negative_results nr
            JOIN proteins p1 ON nr.protein1_id = p1.protein_id
            JOIN proteins p2 ON nr.protein2_id = p2.protein_id
            """,
        )
        for row in df.to_dict(orient="records"):
            p1, g1 = self.resolve_protein(domain_code="ppi", uniprot_accession=row.get("uniprot_id_1"), gene_symbol=row.get("gene_symbol_1"), sequence=row.get("seq_1"))
            p2, g2 = self.resolve_protein(domain_code="ppi", uniprot_accession=row.get("uniprot_id_2"), gene_symbol=row.get("gene_symbol_2"), sequence=row.get("seq_2"))
            ordered = sorted([p1.canonical_key, p2.canonical_key])
            base = anchor_key(*ordered)
            context = {"experiment_id": row.get("experiment_id"), "evidence_type": row.get("evidence_type")}
            claim_id = self._claim(
                domain_code="ppi",
                family="interaction",
                label="non_interacting_with",
                anchor=base,
                base_anchor=base,
                context=context,
                text=f"{ordered[0]} non-interacting with {ordered[1]}",
            )
            self.store.add_claim_entity(claim_id, p1, role="subject")
            self.store.add_claim_entity(claim_id, p2, role="object")
            if g1 is not None:
                self.store.add_claim_entity(claim_id, g1, role="subject_gene")
            if g2 is not None:
                self.store.add_claim_entity(claim_id, g2, role="object_gene")
            self.store.add_evidence(
                claim_id,
                source_domain="ppi",
                source_table="ppi_negative_results",
                source_record_id=_safe_str(row.get("source_record_id") or row.get("result_id")),
                source_db=row.get("source_db"),
                extraction_method=row.get("extraction_method"),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("evidence_type"),
                publication_year=int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                payload={"result_id": row.get("result_id")},
                provenance={"domain_row_id": row.get("result_id")},
            )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                ppp.pair_id, ppp.num_experiments, ppp.num_sources, ppp.best_confidence,
                ppp.best_evidence_type, ppp.earliest_year,
                p1.uniprot_accession AS uniprot_id_1, p1.gene_symbol AS gene_symbol_1,
                p1.amino_acid_sequence AS seq_1,
                p2.uniprot_accession AS uniprot_id_2, p2.gene_symbol AS gene_symbol_2,
                p2.amino_acid_sequence AS seq_2
            FROM protein_protein_pairs ppp
            JOIN proteins p1 ON ppp.protein1_id = p1.protein_id
            JOIN proteins p2 ON ppp.protein2_id = p2.protein_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            p1, _ = self.resolve_protein(domain_code="ppi", uniprot_accession=row.get("uniprot_id_1"), gene_symbol=row.get("gene_symbol_1"), sequence=row.get("seq_1"))
            p2, _ = self.resolve_protein(domain_code="ppi", uniprot_accession=row.get("uniprot_id_2"), gene_symbol=row.get("gene_symbol_2"), sequence=row.get("seq_2"))
            self.store.add_rollup(
                domain_code="ppi",
                source_table="protein_protein_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="interaction",
                anchor_key=anchor_key(*sorted([p1.canonical_key, p2.canonical_key])),
                rollup_type="pair_aggregation",
                summary={
                    "num_experiments": row.get("num_experiments"),
                    "num_sources": row.get("num_sources"),
                    "best_confidence": row.get("best_confidence"),
                    "best_evidence_type": row.get("best_evidence_type"),
                    "earliest_year": row.get("earliest_year"),
                },
            )

    def ingest_ge(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                nr.result_id, nr.screen_id, nr.gene_effect_score, nr.dependency_probability,
                nr.evidence_type, nr.confidence_tier, nr.source_db, nr.source_record_id,
                nr.extraction_method,
                g.entrez_id, g.gene_symbol, g.ensembl_id,
                cl.model_id, cl.ccle_name, cl.lineage
            FROM ge_negative_results nr
            JOIN genes g ON nr.gene_id = g.gene_id
            JOIN cell_lines cl ON nr.cell_line_id = cl.cell_line_id
            """,
        )
        for row in df.to_dict(orient="records"):
            gene = self.resolve_gene(
                domain_code="ge",
                entrez_id=int(row["entrez_id"]) if pd.notna(row.get("entrez_id")) else None,
                ensembl_id=row.get("ensembl_id"),
                gene_symbol=row.get("gene_symbol"),
            )
            cell_line = self.resolve_cell_line(
                domain_code="ge",
                depmap_model_id=row.get("model_id"),
                name=row.get("ccle_name"),
                lineage=row.get("lineage"),
            )
            screen = None
            if row.get("screen_id") is not None:
                screen = self.resolve_local_entity(
                    entity_type="screen",
                    domain_code="ge",
                    local_id=int(row["screen_id"]),
                    display_name=f"GE screen {int(row['screen_id'])}",
                )
            base = anchor_key(gene.canonical_key, cell_line.canonical_key)
            context = {"screen_id": row.get("screen_id"), "evidence_type": row.get("evidence_type")}
            claim_id = self._claim(
                domain_code="ge",
                family="essentiality",
                label="nonessential_in",
                anchor=base,
                base_anchor=base,
                context=context,
                text=f"{gene.canonical_key} non-essential in {cell_line.canonical_key}",
            )
            self.store.add_claim_entity(claim_id, gene, role="subject")
            self.store.add_claim_entity(claim_id, cell_line, role="object")
            if screen is not None:
                self.store.add_claim_entity(claim_id, screen, role="context_screen")
            self.store.add_evidence(
                claim_id,
                source_domain="ge",
                source_table="ge_negative_results",
                source_record_id=_safe_str(row.get("source_record_id") or row.get("result_id")),
                source_db=row.get("source_db"),
                extraction_method=row.get("extraction_method"),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("evidence_type"),
                payload={"result_id": row.get("result_id"), "gene_effect_score": row.get("gene_effect_score"), "dependency_probability": row.get("dependency_probability")},
                provenance={"domain_row_id": row.get("result_id")},
            )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                gcp.pair_id, gcp.num_screens, gcp.num_sources, gcp.best_confidence,
                gcp.best_evidence_type, gcp.min_gene_effect, gcp.max_gene_effect, gcp.mean_gene_effect,
                g.entrez_id, g.gene_symbol, g.ensembl_id,
                cl.model_id, cl.ccle_name, cl.lineage
            FROM gene_cell_pairs gcp
            JOIN genes g ON gcp.gene_id = g.gene_id
            JOIN cell_lines cl ON gcp.cell_line_id = cl.cell_line_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            gene = self.resolve_gene(
                domain_code="ge",
                entrez_id=int(row["entrez_id"]) if pd.notna(row.get("entrez_id")) else None,
                ensembl_id=row.get("ensembl_id"),
                gene_symbol=row.get("gene_symbol"),
            )
            cell_line = self.resolve_cell_line(
                domain_code="ge",
                depmap_model_id=row.get("model_id"),
                name=row.get("ccle_name"),
                lineage=row.get("lineage"),
            )
            self.store.add_rollup(
                domain_code="ge",
                source_table="gene_cell_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="essentiality",
                anchor_key=anchor_key(gene.canonical_key, cell_line.canonical_key),
                rollup_type="pair_aggregation",
                summary={
                    "num_screens": row.get("num_screens"),
                    "num_sources": row.get("num_sources"),
                    "best_confidence": row.get("best_confidence"),
                    "best_evidence_type": row.get("best_evidence_type"),
                    "min_gene_effect": row.get("min_gene_effect"),
                    "max_gene_effect": row.get("max_gene_effect"),
                    "mean_gene_effect": row.get("mean_gene_effect"),
                },
            )

        prism = self._load_domain_frame(
            db_path,
            """
            SELECT
                ps.sensitivity_id, ps.auc, ps.ic50, ps.ec50,
                pc.name, pc.inchikey, pc.chembl_id, pc.pubchem_cid, pc.smiles,
                cl.model_id, cl.ccle_name, cl.lineage
            FROM prism_sensitivity ps
            JOIN prism_compounds pc ON ps.compound_id = pc.compound_id
            JOIN cell_lines cl ON ps.cell_line_id = cl.cell_line_id
            """,
        )
        for row in prism.to_dict(orient="records"):
            mol = self.resolve_small_molecule(
                domain_code="ge",
                inchikey=row.get("inchikey"),
                chembl_id=row.get("chembl_id"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("smiles"),
                name=row.get("name"),
            )
            cell_line = self.resolve_cell_line(
                domain_code="ge",
                depmap_model_id=row.get("model_id"),
                name=row.get("ccle_name"),
                lineage=row.get("lineage"),
            )
            self.store.add_bridge(
                mol,
                cell_line,
                bridge_type="prism_tested_in",
                source_domain="ge",
                method="prism_sensitivity",
                confidence_score=0.8,
                metadata={"auc": row.get("auc"), "ic50": row.get("ic50"), "ec50": row.get("ec50")},
            )

    def ingest_vp(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                nr.result_id, nr.classification, nr.evidence_type, nr.confidence_tier,
                nr.source_db, nr.source_record_id, nr.extraction_method, nr.submission_year,
                nr.has_conflict, nr.num_benign_criteria,
                v.chromosome, v.position, v.ref_allele, v.alt_allele,
                v.hgvs_protein, v.hgvs_coding,
                g.entrez_id, g.gene_symbol, g.hgnc_id, g.ensembl_id,
                d.canonical_name, d.mondo_id, d.omim_id, d.medgen_cui
            FROM vp_negative_results nr
            JOIN variants v ON nr.variant_id = v.variant_id
            JOIN diseases d ON nr.disease_id = d.disease_id
            LEFT JOIN genes g ON v.gene_id = g.gene_id
            """,
        )
        for row in df.to_dict(orient="records"):
            variant = self.resolve_variant(
                domain_code="vp",
                chromosome=row.get("chromosome"),
                position=int(row["position"]) if pd.notna(row.get("position")) else None,
                ref=row.get("ref_allele"),
                alt=row.get("alt_allele"),
                display_name=row.get("hgvs_protein") or row.get("hgvs_coding"),
            )
            gene = None
            if row.get("gene_symbol") or row.get("entrez_id") is not None:
                gene = self.resolve_gene(
                    domain_code="vp",
                    entrez_id=int(row["entrez_id"]) if pd.notna(row.get("entrez_id")) else None,
                    hgnc_id=row.get("hgnc_id"),
                    ensembl_id=row.get("ensembl_id"),
                    gene_symbol=row.get("gene_symbol"),
                )
                self.store.add_bridge(
                    variant,
                    gene,
                    bridge_type="variant_in_gene",
                    source_domain="vp",
                    method="variants.gene_id",
                    confidence_score=1.0,
                )
            disease = self.resolve_disease(
                domain_code="vp",
                name=row.get("canonical_name"),
                mondo_id=row.get("mondo_id"),
                omim_id=row.get("omim_id"),
                medgen_cui=row.get("medgen_cui"),
            )
            base = anchor_key(variant.canonical_key, disease.canonical_key)
            context = {"submission_year": row.get("submission_year"), "evidence_type": row.get("evidence_type")}
            claim_id = self._claim(
                domain_code="vp",
                family="classification",
                label="benign_for",
                anchor=base,
                base_anchor=base,
                context=context,
                text=f"{variant.canonical_key} benign for {disease.canonical_key}",
            )
            self.store.add_claim_entity(claim_id, variant, role="subject")
            self.store.add_claim_entity(claim_id, disease, role="object")
            if gene is not None:
                self.store.add_claim_entity(claim_id, gene, role="mediator_gene")
            self.store.add_evidence(
                claim_id,
                source_domain="vp",
                source_table="vp_negative_results",
                source_record_id=_safe_str(row.get("source_record_id") or row.get("result_id")),
                source_db=row.get("source_db"),
                extraction_method=row.get("extraction_method"),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("evidence_type"),
                publication_year=int(row["submission_year"]) if pd.notna(row.get("submission_year")) else None,
                payload={"result_id": row.get("result_id"), "has_conflict": row.get("has_conflict"), "num_benign_criteria": row.get("num_benign_criteria")},
                provenance={"domain_row_id": row.get("result_id")},
            )

        bridges = self._load_domain_frame(
            db_path,
            """
            SELECT vg.bridge_id, vg.domain, vg.external_id,
                   g.entrez_id, g.gene_symbol, g.hgnc_id, g.ensembl_id
            FROM vp_cross_domain_genes vg
            JOIN genes g ON vg.gene_id = g.gene_id
            """,
        )
        for row in bridges.to_dict(orient="records"):
            gene = self.resolve_gene(
                domain_code="vp",
                entrez_id=int(row["entrez_id"]) if pd.notna(row.get("entrez_id")) else None,
                hgnc_id=row.get("hgnc_id"),
                ensembl_id=row.get("ensembl_id"),
                gene_symbol=row.get("gene_symbol"),
            )
            domain = row.get("domain")
            external_id = row.get("external_id")
            if domain == "ppi":
                protein, _ = self.resolve_protein(domain_code="vp", uniprot_accession=external_id, gene_symbol=row.get("gene_symbol"))
                self.store.add_bridge(
                    gene,
                    protein,
                    bridge_type="vp_cross_domain_gene",
                    source_domain="vp",
                    method="vp_cross_domain_genes",
                    confidence_score=1.0,
                    metadata={"external_domain": domain},
                )
            else:
                ref = self.resolve_local_entity(
                    entity_type="external_reference",
                    domain_code="vp",
                    local_id=f"{domain}:{external_id}",
                    display_name=f"{domain}:{external_id}",
                )
                self.store.add_bridge(
                    gene,
                    ref,
                    bridge_type="vp_cross_domain_gene",
                    source_domain="vp",
                    method="vp_cross_domain_genes",
                    confidence_score=0.8,
                    metadata={"external_domain": domain},
                )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                vdp.pair_id, vdp.num_submissions, vdp.num_submitters, vdp.best_confidence,
                vdp.best_evidence_type, vdp.earliest_year, vdp.has_conflict, vdp.max_population_af,
                v.chromosome, v.position, v.ref_allele, v.alt_allele, v.hgvs_protein, v.hgvs_coding,
                d.canonical_name, d.mondo_id, d.omim_id, d.medgen_cui
            FROM variant_disease_pairs vdp
            JOIN variants v ON vdp.variant_id = v.variant_id
            JOIN diseases d ON vdp.disease_id = d.disease_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            variant = self.resolve_variant(
                domain_code="vp",
                chromosome=row.get("chromosome"),
                position=int(row["position"]) if pd.notna(row.get("position")) else None,
                ref=row.get("ref_allele"),
                alt=row.get("alt_allele"),
                display_name=row.get("hgvs_protein") or row.get("hgvs_coding"),
            )
            disease = self.resolve_disease(
                domain_code="vp",
                name=row.get("canonical_name"),
                mondo_id=row.get("mondo_id"),
                omim_id=row.get("omim_id"),
                medgen_cui=row.get("medgen_cui"),
            )
            self.store.add_rollup(
                domain_code="vp",
                source_table="variant_disease_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="classification",
                anchor_key=anchor_key(variant.canonical_key, disease.canonical_key),
                rollup_type="pair_aggregation",
                summary={
                    "num_submissions": row.get("num_submissions"),
                    "num_submitters": row.get("num_submitters"),
                    "best_confidence": row.get("best_confidence"),
                    "best_evidence_type": row.get("best_evidence_type"),
                    "earliest_year": row.get("earliest_year"),
                    "has_conflict": row.get("has_conflict"),
                    "max_population_af": row.get("max_population_af"),
                },
            )

    def ingest_md(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                r.result_id, r.fold_change, r.log2_fc, r.p_value, r.fdr,
                r.is_significant, r.tier,
                m.name, m.pubchem_cid, m.inchikey, m.canonical_smiles,
                d.name AS disease_name, d.mondo_id, d.mesh_id, d.doid,
                s.study_id, s.source, s.external_id, s.platform, s.biofluid, s.pmid
            FROM md_biomarker_results r
            JOIN md_metabolites m ON r.metabolite_id = m.metabolite_id
            JOIN md_diseases d ON r.disease_id = d.disease_id
            JOIN md_studies s ON r.study_id = s.study_id
            """,
        )
        for row in df.to_dict(orient="records"):
            mol = self.resolve_small_molecule(
                domain_code="md",
                inchikey=row.get("inchikey"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("canonical_smiles"),
                name=row.get("name"),
            )
            disease = self.resolve_disease(
                domain_code="md",
                name=row.get("disease_name"),
                mondo_id=row.get("mondo_id"),
                mesh_id=row.get("mesh_id"),
                do_id=row.get("doid"),
            )
            study = self.resolve_local_entity(
                entity_type="study",
                domain_code="md",
                local_id=int(row["study_id"]),
                display_name=row.get("external_id"),
                attrs={"platform": row.get("platform"), "biofluid": row.get("biofluid")},
            )
            label = "differential_in" if int(row.get("is_significant", 0)) == 1 else "not_differential_in"
            base = anchor_key(mol.canonical_key, disease.canonical_key)
            context = {"study_id": row.get("study_id"), "platform": row.get("platform"), "biofluid": row.get("biofluid")}
            claim_id = self._claim(
                domain_code="md",
                family="biomarker",
                label=label,
                anchor=base,
                base_anchor=base,
                context=context,
                text=f"{mol.canonical_key} {label} {disease.canonical_key}",
            )
            self.store.add_claim_entity(claim_id, mol, role="subject")
            self.store.add_claim_entity(claim_id, disease, role="object")
            self.store.add_claim_entity(claim_id, study, role="context_study")
            self.store.add_evidence(
                claim_id,
                source_domain="md",
                source_table="md_biomarker_results",
                source_record_id=_safe_str(row.get("result_id")),
                source_db=row.get("source"),
                confidence_tier=row.get("tier"),
                evidence_type="significant" if label == "differential_in" else "not_significant",
                publication_year=None,
                payload={
                    "result_id": row.get("result_id"),
                    "fold_change": row.get("fold_change"),
                    "log2_fc": row.get("log2_fc"),
                    "p_value": row.get("p_value"),
                    "fdr": row.get("fdr"),
                },
                provenance={"study_external_id": row.get("external_id")},
            )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                p.pair_id, p.n_studies_total, p.n_studies_negative, p.n_studies_positive,
                p.consensus, p.best_tier,
                m.name, m.pubchem_cid, m.inchikey, m.canonical_smiles,
                d.name AS disease_name, d.mondo_id, d.mesh_id, d.doid
            FROM md_metabolite_disease_pairs p
            JOIN md_metabolites m ON p.metabolite_id = m.metabolite_id
            JOIN md_diseases d ON p.disease_id = d.disease_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            mol = self.resolve_small_molecule(
                domain_code="md",
                inchikey=row.get("inchikey"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("canonical_smiles"),
                name=row.get("name"),
            )
            disease = self.resolve_disease(
                domain_code="md",
                name=row.get("disease_name"),
                mondo_id=row.get("mondo_id"),
                mesh_id=row.get("mesh_id"),
                do_id=row.get("doid"),
            )
            self.store.add_rollup(
                domain_code="md",
                source_table="md_metabolite_disease_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="biomarker",
                anchor_key=anchor_key(mol.canonical_key, disease.canonical_key),
                rollup_type="pair_aggregation",
                summary={
                    "n_studies_total": row.get("n_studies_total"),
                    "n_studies_negative": row.get("n_studies_negative"),
                    "n_studies_positive": row.get("n_studies_positive"),
                    "consensus": row.get("consensus"),
                    "best_tier": row.get("best_tier"),
                },
            )

    def ingest_dc(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                sr.result_id, sr.synergy_class, sr.confidence_tier, sr.evidence_type,
                sr.source_db, sr.source_study_id,
                ca.drug_name AS name_a, ca.inchikey AS inchikey_a, ca.canonical_smiles AS smiles_a,
                ca.chembl_id AS chembl_id_a, ca.pubchem_cid AS pubchem_cid_a,
                cb.drug_name AS name_b, cb.inchikey AS inchikey_b, cb.canonical_smiles AS smiles_b,
                cb.chembl_id AS chembl_id_b, cb.pubchem_cid AS pubchem_cid_b,
                cl.depmap_model_id, cl.cosmic_id, cl.cell_line_name, cl.lineage
            FROM dc_synergy_results sr
            JOIN compounds ca ON sr.compound_a_id = ca.compound_id
            JOIN compounds cb ON sr.compound_b_id = cb.compound_id
            JOIN cell_lines cl ON sr.cell_line_id = cl.cell_line_id
            """,
        )
        for row in df.to_dict(orient="records"):
            label = None
            if row.get("synergy_class") in {"synergistic", "strongly_synergistic"}:
                label = "synergistic_in"
            elif row.get("synergy_class") in {"antagonistic", "strongly_antagonistic"}:
                label = "antagonistic_in"
            elif row.get("synergy_class") == "additive":
                label = "additive_in"
            if label is None:
                continue
            mol_a = self.resolve_small_molecule(
                domain_code="dc",
                inchikey=row.get("inchikey_a"),
                chembl_id=row.get("chembl_id_a"),
                pubchem_cid=row.get("pubchem_cid_a"),
                smiles=row.get("smiles_a"),
                name=row.get("name_a"),
            )
            mol_b = self.resolve_small_molecule(
                domain_code="dc",
                inchikey=row.get("inchikey_b"),
                chembl_id=row.get("chembl_id_b"),
                pubchem_cid=row.get("pubchem_cid_b"),
                smiles=row.get("smiles_b"),
                name=row.get("name_b"),
            )
            cell_line = self.resolve_cell_line(
                domain_code="dc",
                depmap_model_id=row.get("depmap_model_id"),
                cosmic_id=int(row["cosmic_id"]) if pd.notna(row.get("cosmic_id")) else None,
                name=row.get("cell_line_name"),
                lineage=row.get("lineage"),
            )
            ordered = sorted([mol_a.canonical_key, mol_b.canonical_key])
            base = anchor_key(*ordered)
            anchor = anchor_key(base, cell_line.canonical_key)
            context = {"cell_line": row.get("cell_line_name"), "source_db": row.get("source_db")}
            claim_id = self._claim(
                domain_code="dc",
                family="combination",
                label=label,
                anchor=anchor,
                base_anchor=base,
                context=context,
                text=f"{ordered[0]} / {ordered[1]} {label} in {cell_line.canonical_key}",
            )
            self.store.add_claim_entity(claim_id, mol_a, role="subject")
            self.store.add_claim_entity(claim_id, mol_b, role="object")
            self.store.add_claim_entity(claim_id, cell_line, role="context_cell_line")
            self.store.add_evidence(
                claim_id,
                source_domain="dc",
                source_table="dc_synergy_results",
                source_record_id=_safe_str(row.get("result_id")),
                source_db=row.get("source_db"),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("evidence_type"),
                payload={"result_id": row.get("result_id"), "source_study_id": row.get("source_study_id"), "synergy_class": row.get("synergy_class")},
                provenance={"domain_row_id": row.get("result_id")},
            )

        rollups = self._load_domain_frame(
            db_path,
            """
            SELECT
                ddp.pair_id, ddp.consensus_class, ddp.best_confidence, ddp.num_cell_lines,
                ddp.num_sources, ddp.antagonism_fraction, ddp.synergy_fraction,
                ca.drug_name AS name_a, ca.inchikey AS inchikey_a, ca.canonical_smiles AS smiles_a,
                ca.chembl_id AS chembl_id_a, ca.pubchem_cid AS pubchem_cid_a,
                cb.drug_name AS name_b, cb.inchikey AS inchikey_b, cb.canonical_smiles AS smiles_b,
                cb.chembl_id AS chembl_id_b, cb.pubchem_cid AS pubchem_cid_b
            FROM drug_drug_pairs ddp
            JOIN compounds ca ON ddp.compound_a_id = ca.compound_id
            JOIN compounds cb ON ddp.compound_b_id = cb.compound_id
            """,
        )
        for row in rollups.to_dict(orient="records"):
            mol_a = self.resolve_small_molecule(
                domain_code="dc",
                inchikey=row.get("inchikey_a"),
                chembl_id=row.get("chembl_id_a"),
                pubchem_cid=row.get("pubchem_cid_a"),
                smiles=row.get("smiles_a"),
                name=row.get("name_a"),
            )
            mol_b = self.resolve_small_molecule(
                domain_code="dc",
                inchikey=row.get("inchikey_b"),
                chembl_id=row.get("chembl_id_b"),
                pubchem_cid=row.get("pubchem_cid_b"),
                smiles=row.get("smiles_b"),
                name=row.get("name_b"),
            )
            self.store.add_rollup(
                domain_code="dc",
                source_table="drug_drug_pairs",
                source_row_id=_safe_str(row.get("pair_id")),
                claim_family="combination",
                anchor_key=anchor_key(*sorted([mol_a.canonical_key, mol_b.canonical_key])),
                rollup_type="pair_aggregation",
                summary={
                    "consensus_class": row.get("consensus_class"),
                    "best_confidence": row.get("best_confidence"),
                    "num_cell_lines": row.get("num_cell_lines"),
                    "num_sources": row.get("num_sources"),
                    "antagonism_fraction": row.get("antagonism_fraction"),
                    "synergy_fraction": row.get("synergy_fraction"),
                },
            )

        compound_bridges = self._load_domain_frame(
            db_path,
            """
            SELECT dcc.domain, dcc.external_id,
                   c.drug_name, c.inchikey, c.canonical_smiles, c.chembl_id, c.pubchem_cid
            FROM dc_cross_domain_compounds dcc
            JOIN compounds c ON dcc.compound_id = c.compound_id
            """,
        )
        for row in compound_bridges.to_dict(orient="records"):
            mol = self.resolve_small_molecule(
                domain_code="dc",
                inchikey=row.get("inchikey"),
                chembl_id=row.get("chembl_id"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("canonical_smiles"),
                name=row.get("drug_name"),
            )
            ref = self.resolve_local_entity(
                entity_type="external_reference",
                domain_code="dc",
                local_id=f"{row.get('domain')}:{row.get('external_id')}",
                display_name=f"{row.get('domain')}:{row.get('external_id')}",
            )
            self.store.add_bridge(
                mol,
                ref,
                bridge_type="dc_cross_domain_compound",
                source_domain="dc",
                method="dc_cross_domain_compounds",
                confidence_score=0.8,
            )

        cell_bridges = self._load_domain_frame(
            db_path,
            """
            SELECT dcl.domain, dcl.external_id, cl.depmap_model_id, cl.cosmic_id, cl.cell_line_name, cl.lineage
            FROM dc_cross_domain_cell_lines dcl
            JOIN cell_lines cl ON dcl.cell_line_id = cl.cell_line_id
            """,
        )
        for row in cell_bridges.to_dict(orient="records"):
            cell_line = self.resolve_cell_line(
                domain_code="dc",
                depmap_model_id=row.get("depmap_model_id"),
                cosmic_id=int(row["cosmic_id"]) if pd.notna(row.get("cosmic_id")) else None,
                name=row.get("cell_line_name"),
                lineage=row.get("lineage"),
            )
            if row.get("external_id") == row.get("depmap_model_id"):
                peer = self.resolve_cell_line(
                    domain_code="ge",
                    depmap_model_id=row.get("external_id"),
                    name=row.get("cell_line_name"),
                    lineage=row.get("lineage"),
                )
            else:
                peer = self.resolve_local_entity(
                    entity_type="external_reference",
                    domain_code="dc",
                    local_id=f"{row.get('domain')}:{row.get('external_id')}",
                    display_name=f"{row.get('domain')}:{row.get('external_id')}",
                )
            self.store.add_bridge(
                cell_line,
                peer,
                bridge_type="dc_cross_domain_cell_line",
                source_domain="dc",
                method="dc_cross_domain_cell_lines",
                confidence_score=0.8,
            )

    def ingest_cp(self, db_path: Path) -> None:
        assert self.store is not None
        df = self._load_domain_frame(
            db_path,
            """
            SELECT
                r.cp_result_id, r.outcome_label, r.confidence_tier, r.has_orthogonal_evidence,
                r.dose, r.dose_unit, r.timepoint_h,
                c.compound_name, c.canonical_smiles, c.inchikey, c.inchikey_connectivity,
                c.chembl_id, c.pubchem_cid,
                cl.cell_line_name, cl.tissue, cl.disease,
                a.assay_context_id, a.assay_name, a.cell_painting_version,
                b.batch_id, b.batch_name
            FROM cp_perturbation_results r
            JOIN compounds c ON r.compound_id = c.compound_id
            JOIN cp_cell_lines cl ON r.cell_line_id = cl.cell_line_id
            JOIN cp_assay_contexts a ON r.assay_context_id = a.assay_context_id
            JOIN cp_batches b ON r.batch_id = b.batch_id
            """,
        )
        cp_result_to_claim: dict[int, tuple[int, ResolvedEntity]] = {}
        label_map = {
            "inactive": "inactive_in",
            "weak_phenotype": "weak_phenotype_in",
            "strong_phenotype": "strong_phenotype_in",
            "toxic_or_artifact": "artifact_or_toxic_in",
        }
        for row in df.to_dict(orient="records"):
            label = label_map.get(row.get("outcome_label"))
            if label is None:
                continue
            mol = self.resolve_small_molecule(
                domain_code="cp",
                inchikey=row.get("inchikey"),
                connectivity=row.get("inchikey_connectivity"),
                chembl_id=row.get("chembl_id"),
                pubchem_cid=row.get("pubchem_cid"),
                smiles=row.get("canonical_smiles"),
                name=row.get("compound_name"),
            )
            cell_line = self.resolve_cell_line(
                domain_code="cp",
                name=row.get("cell_line_name"),
                lineage=row.get("tissue"),
            )
            assay_context = self.resolve_local_entity(
                entity_type="assay_context",
                domain_code="cp",
                local_id=int(row["assay_context_id"]),
                display_name=row.get("assay_name"),
                attrs={"version": row.get("cell_painting_version")},
            )
            batch = self.resolve_local_entity(
                entity_type="batch",
                domain_code="cp",
                local_id=int(row["batch_id"]),
                display_name=row.get("batch_name"),
            )
            base = anchor_key(mol.canonical_key, cell_line.canonical_key, assay_context.canonical_key)
            anchor = anchor_key(
                base,
                f"dose:{row.get('dose')}{row.get('dose_unit')}",
                f"time:{row.get('timepoint_h')}",
            )
            context = {
                "dose": row.get("dose"),
                "dose_unit": row.get("dose_unit"),
                "timepoint_h": row.get("timepoint_h"),
                "batch_name": row.get("batch_name"),
            }
            claim_id = self._claim(
                domain_code="cp",
                family="phenotype",
                label=label,
                anchor=anchor,
                base_anchor=base,
                context=context,
                text=f"{mol.canonical_key} {label} in {cell_line.canonical_key}",
            )
            cp_result_to_claim[int(row["cp_result_id"])] = (claim_id, mol)
            self.store.add_claim_entity(claim_id, mol, role="subject")
            self.store.add_claim_entity(claim_id, cell_line, role="object")
            self.store.add_claim_entity(claim_id, assay_context, role="context_assay")
            self.store.add_claim_entity(claim_id, batch, role="context_batch")
            self.store.add_evidence(
                claim_id,
                source_domain="cp",
                source_table="cp_perturbation_results",
                source_record_id=_safe_str(row.get("cp_result_id")),
                confidence_tier=row.get("confidence_tier"),
                evidence_type=row.get("outcome_label"),
                payload={"cp_result_id": row.get("cp_result_id"), "has_orthogonal_evidence": row.get("has_orthogonal_evidence")},
                provenance={"domain_row_id": row.get("cp_result_id")},
            )

        orth = self._load_domain_frame(
            db_path,
            """
            SELECT evidence_id, cp_result_id, evidence_domain, evidence_label,
                   source_name, source_record_id, match_key, notes
            FROM cp_orthogonal_evidence
            """,
        )
        for row in orth.to_dict(orient="records"):
            mapped = cp_result_to_claim.get(int(row["cp_result_id"]))
            if mapped is None:
                continue
            claim_id, mol = mapped
            ref = self.resolve_local_entity(
                entity_type="external_reference",
                domain_code="cp",
                local_id=f"{row.get('evidence_domain')}:{row.get('source_name')}:{row.get('source_record_id') or row.get('match_key')}",
                display_name=row.get("source_name"),
                attrs={"match_key": row.get("match_key"), "evidence_label": row.get("evidence_label")},
            )
            self.store.add_bridge(
                mol,
                ref,
                bridge_type="cp_orthogonal_evidence",
                source_domain="cp",
                method="cp_orthogonal_evidence",
                confidence_score=0.7,
                metadata={"evidence_domain": row.get("evidence_domain"), "match_key": row.get("match_key")},
            )
            self.store.add_evidence(
                claim_id,
                source_domain="cp",
                source_table="cp_orthogonal_evidence",
                source_record_id=_safe_str(row.get("evidence_id")),
                reference_name=row.get("source_name"),
                evidence_type=row.get("evidence_label"),
                payload={
                    "cp_result_id": row.get("cp_result_id"),
                    "evidence_domain": row.get("evidence_domain"),
                    "match_key": row.get("match_key"),
                    "notes": row.get("notes"),
                },
                provenance={"source_record_id": row.get("source_record_id")},
            )

    def ingest_reference_feed(self, feed: ReferenceFeed, domain_paths: dict[str, Path]) -> None:
        assert self.store is not None
        if not feed.path.exists():
            if self.strict and feed.required:
                raise FileNotFoundError(f"Required reference feed missing: {feed.path}")
            return
        df = pd.read_parquet(feed.path)

        if feed.kind == "dti_positive_pairs":
            for idx, row in enumerate(df.to_dict(orient="records")):
                mol = self.resolve_small_molecule(
                    domain_code="dti",
                    inchikey=row.get("inchikey"),
                    connectivity=row.get("inchikey_connectivity"),
                    chembl_id=row.get("chembl_id"),
                    pubchem_cid=row.get("pubchem_cid"),
                    smiles=row.get("smiles"),
                    name=row.get("compound_name"),
                )
                protein, gene = self.resolve_protein(
                    domain_code="dti",
                    uniprot_accession=row.get("uniprot_id"),
                    gene_symbol=row.get("gene_symbol"),
                    sequence=row.get("target_sequence"),
                )
                base = anchor_key(mol.canonical_key, protein.canonical_key)
                claim_id = self._claim(
                    domain_code="dti",
                    family="binding",
                    label="active_against",
                    anchor=base,
                    base_anchor=base,
                    context={"reference_name": feed.name},
                    text=f"{mol.canonical_key} active against {protein.canonical_key}",
                    level="reference",
                )
                self.store.add_claim_entity(claim_id, mol, role="subject")
                self.store.add_claim_entity(claim_id, protein, role="object")
                if gene is not None:
                    self.store.add_claim_entity(claim_id, gene, role="object_gene")
                self.store.add_evidence(
                    claim_id,
                    source_domain="dti",
                    source_table="reference_feed",
                    source_record_id=_safe_str(row.get("source_record_id") or idx),
                    reference_name=feed.name,
                    publication_year=int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                    payload=row,
                    provenance={"feed_path": str(feed.path)},
                )

        elif feed.kind == "ppi_positive_pairs":
            for idx, row in enumerate(df.to_dict(orient="records")):
                p1, g1 = self.resolve_protein(domain_code="ppi", uniprot_accession=row.get("uniprot_id_1"), gene_symbol=row.get("gene_symbol_1"), sequence=row.get("sequence_1"))
                p2, g2 = self.resolve_protein(domain_code="ppi", uniprot_accession=row.get("uniprot_id_2"), gene_symbol=row.get("gene_symbol_2"), sequence=row.get("sequence_2"))
                base = anchor_key(*sorted([p1.canonical_key, p2.canonical_key]))
                claim_id = self._claim(
                    domain_code="ppi",
                    family="interaction",
                    label="interacts_with",
                    anchor=base,
                    base_anchor=base,
                    context={"reference_name": feed.name},
                    text=f"{base} interacts",
                    level="reference",
                )
                self.store.add_claim_entity(claim_id, p1, role="subject")
                self.store.add_claim_entity(claim_id, p2, role="object")
                if g1 is not None:
                    self.store.add_claim_entity(claim_id, g1, role="subject_gene")
                if g2 is not None:
                    self.store.add_claim_entity(claim_id, g2, role="object_gene")
                self.store.add_evidence(
                    claim_id,
                    source_domain="ppi",
                    source_table="reference_feed",
                    source_record_id=_safe_str(row.get("source_record_id") or idx),
                    reference_name=feed.name,
                    payload=row,
                    provenance={"feed_path": str(feed.path)},
                )

        elif feed.kind == "ge_essential_pairs":
            for idx, row in enumerate(df.to_dict(orient="records")):
                gene = self.resolve_gene(
                    domain_code="ge",
                    entrez_id=int(row["entrez_id"]) if pd.notna(row.get("entrez_id")) else None,
                    gene_symbol=row.get("gene_symbol"),
                )
                cell_line = self.resolve_cell_line(
                    domain_code="ge",
                    depmap_model_id=row.get("model_id"),
                    name=row.get("ccle_name"),
                    lineage=row.get("lineage"),
                )
                base = anchor_key(gene.canonical_key, cell_line.canonical_key)
                claim_id = self._claim(
                    domain_code="ge",
                    family="essentiality",
                    label="essential_in",
                    anchor=base,
                    base_anchor=base,
                    context={"reference_name": feed.name},
                    text=f"{gene.canonical_key} essential in {cell_line.canonical_key}",
                    level="reference",
                )
                self.store.add_claim_entity(claim_id, gene, role="subject")
                self.store.add_claim_entity(claim_id, cell_line, role="object")
                self.store.add_evidence(
                    claim_id,
                    source_domain="ge",
                    source_table="reference_feed",
                    source_record_id=_safe_str(row.get("source_record_id") or idx),
                    reference_name=feed.name,
                    payload=row,
                    provenance={"feed_path": str(feed.path)},
                )

        elif feed.kind == "ct_success_pairs":
            for idx, row in enumerate(df.to_dict(orient="records")):
                intervention = self.resolve_local_entity(
                    entity_type="intervention",
                    domain_code="ct",
                    local_id=row.get("intervention_id") or idx,
                    display_name=row.get("intervention_name"),
                )
                disease = self.resolve_disease(
                    domain_code="ct",
                    name=row.get("condition_name"),
                    mesh_id=row.get("mesh_id"),
                    do_id=row.get("do_id"),
                    icd10_code=row.get("icd10_code"),
                )
                if row.get("inchikey") or row.get("inchikey_connectivity") or row.get("chembl_id"):
                    mol = self.resolve_small_molecule(
                        domain_code="ct",
                        inchikey=row.get("inchikey"),
                        connectivity=row.get("inchikey_connectivity"),
                        chembl_id=row.get("chembl_id"),
                        pubchem_cid=row.get("pubchem_cid"),
                        smiles=row.get("smiles"),
                        name=row.get("intervention_name"),
                    )
                    self.store.add_bridge(
                        intervention,
                        mol,
                        bridge_type="chemical_equivalence",
                        source_domain="ct",
                        method="reference_feed",
                        confidence_score=1.0,
                    )
                base = anchor_key(intervention.canonical_key, disease.canonical_key)
                claim_id = self._claim(
                    domain_code="ct",
                    family="trial_outcome",
                    label="successful_for",
                    anchor=base,
                    base_anchor=base,
                    context={"reference_name": feed.name},
                    text=f"{intervention.canonical_key} successful for {disease.canonical_key}",
                    level="reference",
                )
                self.store.add_claim_entity(claim_id, intervention, role="subject")
                self.store.add_claim_entity(claim_id, disease, role="object")
                self.store.add_evidence(
                    claim_id,
                    source_domain="ct",
                    source_table="reference_feed",
                    source_record_id=_safe_str(row.get("source_record_id") or idx),
                    reference_name=feed.name,
                    payload=row,
                    provenance={"feed_path": str(feed.path)},
                )

        else:
            if self.strict:
                raise ValueError(f"Unsupported reference feed kind: {feed.kind}")

    def build(
        self,
        *,
        domain_paths: dict[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        self._wipe_graph()
        self.build_id = self._start_build()
        self.store = GraphStore(self.conn, self.build_id)
        manifest_feeds = load_reference_manifest(self.manifest_path)

        try:
            resolved_paths = discover_domain_paths(domain_paths)
            for code in GRAPH_DOMAIN_ORDER:
                path = resolved_paths[code]
                available = path.exists()
                self._record_input(
                    input_name=f"{code}_db",
                    input_kind="domain_db",
                    domain_code=code,
                    path=path,
                    is_required=True,
                    is_available=available,
                    status="available" if available else "missing",
                    metadata={"label": DOMAIN_REGISTRY[code].label},
                )
                self._assert_available(code, path)
            for feed in manifest_feeds:
                self._record_input(
                    input_name=feed.name,
                    input_kind="reference_feed",
                    domain_code=feed.domain_code,
                    path=feed.path,
                    is_required=feed.required,
                    is_available=feed.path.exists(),
                    status="available" if feed.path.exists() else "missing",
                    metadata={"kind": feed.kind, "description": feed.description},
                )
                if self.strict and feed.required and not feed.path.exists():
                    raise FileNotFoundError(f"Required reference feed missing: {feed.path}")

            ingest_map = {
                "dti": self.ingest_dti,
                "ct": self.ingest_ct,
                "ppi": self.ingest_ppi,
                "ge": self.ingest_ge,
                "vp": self.ingest_vp,
                "md": self.ingest_md,
                "dc": self.ingest_dc,
                "cp": self.ingest_cp,
            }
            for code, ingester in ingest_map.items():
                path = resolved_paths[code]
                if not path.exists():
                    continue
                ingester(path)
                self.conn.commit()

            for feed in manifest_feeds:
                if feed.path.exists():
                    self.ingest_reference_feed(feed, resolved_paths)
                    self.conn.commit()

            self._finish_build(
                "complete",
                notes=stable_json_dumps({"manifest": manifest_summary(manifest_feeds)}),
            )
            counts = {}
            for table in [
                "graph_entities",
                "graph_entity_aliases",
                "graph_bridges",
                "graph_claims",
                "graph_evidence",
                "graph_claim_rollups",
            ]:
                counts[table] = int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            counts["build_id"] = self.build_id
            counts["graph_db_path"] = str(self.graph_db_path)
            return counts
        except Exception as exc:
            self._finish_build("failed", notes=str(exc))
            raise


def build_graph(
    graph_db_path: str | Path = DEFAULT_GRAPH_DB_PATH,
    *,
    domain_paths: dict[str, str | Path] | None = None,
    manifest_path: str | Path | None = None,
    strict: bool = False,
    build_tag: str | None = None,
) -> dict[str, Any]:
    """Build NegBioGraph from domain databases and optional reference feeds."""
    builder = GraphBuilder(
        graph_db_path,
        strict=strict,
        build_tag=build_tag,
        manifest_path=manifest_path,
    )
    try:
        return builder.build(domain_paths=domain_paths)
    finally:
        builder.close()

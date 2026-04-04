"""ClinVar ETL for NegBioDB VP domain.

Parses variant_summary.txt.gz and submission_summary.txt.gz to extract
benign/likely benign variant-disease pairs as negative results.

Key decisions:
  - GRCh38 assembly only
  - Phase 1 scope: SNVs + small indels (no structural variants)
  - Confidence tiers: gold/silver/bronze based on ClinVar review status
  - Conflict detection: has_conflict=1 if any P/LP submission for same variant-disease
  - ACMG criteria: best-effort regex extraction from submission text
"""

import csv
import gzip
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Phase 1 variant types
_ALLOWED_VARIANT_TYPES = frozenset({
    "single nucleotide variant",
    "Deletion",
    "Insertion",
    "Indel",
    "Duplication",
})

# Benign classifications (for negative results)
_BENIGN_CLASSIFICATIONS = frozenset({
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
})

# Pathogenic classifications (for conflict detection and M1 positives)
_PATHOGENIC_CLASSIFICATIONS = frozenset({
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
})

# ClinVar review status → confidence tier mapping
_GOLD_REVIEW_STATUSES = frozenset({
    "practice guideline",
    "reviewed by expert panel",
})

_SILVER_REVIEW_STATUS = "criteria provided, multiple submitters, no conflicts"

_BRONZE_REVIEW_STATUSES = frozenset({
    "criteria provided, single submitter",
    "criteria provided, conflicting classifications",
})

# Map ClinVar variant types to our consequence categories
_CONSEQUENCE_MAP = {
    "missense_variant": "missense",
    "missense variant": "missense",
    "nonsense": "nonsense",
    "stop gained": "nonsense",
    "synonymous_variant": "synonymous",
    "synonymous variant": "synonymous",
    "frameshift_variant": "frameshift",
    "frameshift variant": "frameshift",
    "splice_site_variant": "splice",
    "splice site variant": "splice",
    "splice_donor_variant": "splice",
    "splice_acceptor_variant": "splice",
    "inframe_deletion": "inframe_indel",
    "inframe_insertion": "inframe_indel",
    "inframe deletion": "inframe_indel",
    "inframe insertion": "inframe_indel",
    "intron_variant": "intronic",
    "intron variant": "intronic",
}

# ACMG benign criteria regex
_ACMG_CRITERIA_RE = re.compile(
    r"\b(BA1|BS[1-4]|BP[1-7]|PP3|PM[1-6]|PS[1-4]|PVS1)\b"
)
_MEDGEN_CUI_RE = re.compile(r"\b(CN?\d{6,8})\b")


def _normalize_classification(raw: str) -> str | None:
    """Normalize ClinVar classification to our schema values."""
    lower = raw.strip().lower()
    if lower in ("benign",):
        return "benign"
    elif lower in ("likely benign",):
        return "likely_benign"
    elif lower in ("benign/likely benign",):
        return "benign/likely_benign"
    elif lower in ("pathogenic",):
        return "pathogenic"
    elif lower in ("likely pathogenic",):
        return "likely_pathogenic"
    elif lower in ("pathogenic/likely pathogenic",):
        return "pathogenic/likely_pathogenic"
    elif lower in ("uncertain significance",):
        return "uncertain_significance"
    return None


def _classify_tier(review_status: str) -> str:
    """Map ClinVar review status to confidence tier."""
    rs_lower = review_status.strip().lower()
    if rs_lower in _GOLD_REVIEW_STATUSES:
        return "gold"
    if rs_lower == _SILVER_REVIEW_STATUS:
        return "silver"
    if rs_lower in _BRONZE_REVIEW_STATUSES:
        return "bronze"
    return "bronze"  # fallback for other review statuses


def _classify_evidence_type(review_status: str, num_submitters: int) -> str:
    """Map review status to evidence type."""
    rs_lower = review_status.strip().lower()
    if rs_lower in _GOLD_REVIEW_STATUSES:
        return "expert_reviewed"
    if rs_lower == _SILVER_REVIEW_STATUS or num_submitters >= 3:
        return "multi_submitter_concordant"
    return "single_submitter"


def _parse_consequence(name: str) -> str | None:
    """Extract consequence type from ClinVar variant Name field."""
    name_lower = name.lower() if name else ""
    for key, val in _CONSEQUENCE_MAP.items():
        if key in name_lower:
            return val
    if "missense" in name_lower or "(p." in name_lower:
        return "missense"
    return "other"


def _extract_acmg_criteria(text: str | None) -> list[str]:
    """Extract ACMG criteria codes from free text. Best-effort."""
    if not text:
        return []
    return sorted(set(_ACMG_CRITERIA_RE.findall(text)))


def _parse_hgvs_from_name(name: str) -> tuple[str | None, str | None]:
    """Extract HGVS coding and protein notations from ClinVar Name field.

    Examples:
      'NM_000059.4(BRCA2):c.5123C>A (p.Ala1708Asp)' → ('NM_000059.4:c.5123C>A', 'p.Ala1708Asp')
      'NM_000059.4:c.5123C>A' → ('NM_000059.4:c.5123C>A', None)
    """
    hgvs_c = None
    hgvs_p = None

    if not name:
        return hgvs_c, hgvs_p

    # Extract protein notation
    p_match = re.search(r"\((p\.\w+)\)", name)
    if p_match:
        hgvs_p = p_match.group(1)

    # Extract coding notation: strip gene symbol in parentheses
    c_match = re.search(r"(N[MR]_[\d.]+)(?:\([^)]+\))?(:[cgrnomp]\.\S+)", name)
    if c_match:
        hgvs_c = c_match.group(1) + c_match.group(2)
        # Remove trailing protein annotation
        hgvs_c = re.sub(r"\s*\(p\..*\)\s*$", "", hgvs_c)

    return hgvs_c, hgvs_p


def _map_variant_type(clinvar_type: str) -> str:
    """Map ClinVar Type to our variant_type CHECK constraint values."""
    mapping = {
        "single nucleotide variant": "single nucleotide variant",
        "Deletion": "Deletion",
        "Insertion": "Insertion",
        "Indel": "Indel",
        "Duplication": "Duplication",
    }
    return mapping.get(clinvar_type, "other")


def parse_variant_summary(
    variant_summary_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Parse variant_summary.txt.gz for benign and pathogenic variants.

    Returns:
        (benign_records, pathogenic_records) — lists of dicts with parsed fields.
    """
    benign_records = []
    pathogenic_records = []

    opener = gzip.open if str(variant_summary_path).endswith(".gz") else open

    with opener(variant_summary_path, "rt", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_num, row in enumerate(reader, 1):
            # Filter GRCh38 only
            if row.get("Assembly") != "GRCh38":
                continue

            # Filter variant types (Phase 1: SNVs + small indels)
            vtype = row.get("Type", "")
            if vtype not in _ALLOWED_VARIANT_TYPES:
                continue

            clin_sig = row.get("ClinicalSignificance", "")
            variation_id = row.get("VariationID", "")
            if not variation_id:
                continue

            # Parse common fields — prefer VCF-normalised columns for locus
            chromosome = row.get("Chromosome", "")
            ref_allele_vcf = row.get("ReferenceAlleleVCF", "")
            alt_allele_vcf = row.get("AlternateAlleleVCF", "")
            pos_vcf = row.get("PositionVCF", "")

            # Fallback to non-VCF columns only if VCF columns are missing/na
            _na = {"", "na", "-", "-1"}
            if ref_allele_vcf.lower() not in _na and alt_allele_vcf.lower() not in _na:
                ref_allele = ref_allele_vcf
                alt_allele = alt_allele_vcf
                start = pos_vcf
            else:
                ref_allele = row.get("ReferenceAllele", "")
                alt_allele = row.get("AlternateAllele", "")
                start = row.get("Start", "")
                if ref_allele.lower() in _na or alt_allele.lower() in _na:
                    continue

            if not (chromosome and start and ref_allele and alt_allele):
                continue

            # Skip if ref or alt is too long (likely structural)
            if len(ref_allele) > 200 or len(alt_allele) > 200:
                continue

            try:
                position = int(start)
            except (ValueError, TypeError):
                continue

            gene_symbol = row.get("GeneSymbol", "")
            gene_id_raw = row.get("GeneID", "")
            hgnc_id = row.get("HGNC_ID", "")
            name = row.get("Name", "")
            review_status = row.get("ReviewStatus", "")
            rs_raw = row.get("RS# (dbSNP)", "")
            phenotype_ids = row.get("PhenotypeIDS", "")
            phenotype_list = row.get("PhenotypeList", "")
            num_submitters_raw = row.get("NumberSubmitters", "0")

            try:
                gene_id = int(gene_id_raw) if gene_id_raw and gene_id_raw != "-1" else None
            except (ValueError, TypeError):
                gene_id = None

            try:
                rs_id = int(rs_raw) if rs_raw and rs_raw != "-1" else None
            except (ValueError, TypeError):
                rs_id = None

            try:
                num_submitters = int(num_submitters_raw)
            except (ValueError, TypeError):
                num_submitters = 1

            hgvs_c, hgvs_p = _parse_hgvs_from_name(name)
            consequence = _parse_consequence(name)
            mapped_type = _map_variant_type(vtype)

            record = {
                "variation_id": int(variation_id),
                "chromosome": chromosome,
                "position": position,
                "ref_allele": ref_allele,
                "alt_allele": alt_allele,
                "variant_type": mapped_type,
                "gene_symbol": gene_symbol,
                "entrez_id": gene_id,
                "hgnc_id": hgnc_id if hgnc_id else None,
                "rs_id": rs_id,
                "hgvs_coding": hgvs_c,
                "hgvs_protein": hgvs_p,
                "consequence_type": consequence,
                "review_status": review_status,
                "num_submitters": num_submitters,
                "phenotype_ids": phenotype_ids,
                "phenotype_list": phenotype_list,
                "name": name,
            }

            # Classify
            is_benign = any(b in clin_sig for b in ("Benign", "Likely benign"))
            is_pathogenic = any(
                p in clin_sig for p in ("Pathogenic", "Likely pathogenic")
            )

            if is_benign:
                record["classification"] = _normalize_classification(clin_sig)
                record["confidence_tier"] = _classify_tier(review_status)
                record["evidence_type"] = _classify_evidence_type(
                    review_status, num_submitters
                )
                benign_records.append(record)

            if is_pathogenic:
                record_copy = dict(record)
                record_copy["classification"] = _normalize_classification(clin_sig)
                pathogenic_records.append(record_copy)

            if row_num % 500_000 == 0:
                logger.info(
                    "Parsed %d rows: %d benign, %d pathogenic",
                    row_num,
                    len(benign_records),
                    len(pathogenic_records),
                )

    logger.info(
        "variant_summary parsed: %d benign, %d pathogenic",
        len(benign_records),
        len(pathogenic_records),
    )
    return benign_records, pathogenic_records


def parse_submission_summary(
    submission_summary_path: Path,
    variation_ids: set[int] | None = None,
) -> dict[int, list[dict]]:
    """Parse submission_summary.txt.gz for per-submission detail.

    Args:
        submission_summary_path: Path to submission_summary.txt.gz
        variation_ids: If provided, only parse submissions for these VariationIDs

    Returns:
        Dict mapping VariationID → list of submission dicts
    """
    submissions: dict[int, list[dict]] = {}

    opener = gzip.open if str(submission_summary_path).endswith(".gz") else open

    with opener(submission_summary_path, "rt", errors="replace") as f:
        # Skip ## comment lines but keep the #VariationID header line
        def _filter_comments(fh):
            for line in fh:
                if line.startswith("##"):
                    continue
                # The header line has tabs: #VariationID\tClinicalSignificance\t...
                # Description lines have colons: #VariationID:   the identifier...
                if line.startswith("#VariationID\t"):
                    yield line.lstrip("#")
                elif line.startswith("#"):
                    continue  # comment/description lines
                else:
                    yield line

        reader = csv.DictReader(
            _filter_comments(f),
            delimiter="\t",
        )
        for row_num, row in enumerate(reader, 1):
            vid_raw = row.get("VariationID", "")
            if not vid_raw:
                continue
            try:
                vid = int(vid_raw)
            except (ValueError, TypeError):
                continue

            if variation_ids is not None and vid not in variation_ids:
                continue

            # Column is "SCV" in current ClinVar format
            scv = row.get("SCV", row.get("ClinVarAccession", ""))
            # Remove version: SCV000001.3 → SCV000001
            if scv and "." in scv:
                scv = scv.split(".")[0]

            submitter = row.get("Submitter", "")
            classification = row.get("ClinicalSignificance", "")
            review_status = row.get("ReviewStatus", "")
            date_eval = row.get("DateLastEvaluated", "")
            reported_pheno = row.get("ReportedPhenotypeInfo", "")
            description = row.get("Description", "")
            interpretation = row.get("ExplanationOfInterpretation", "")

            # Combine text fields for ACMG criteria extraction
            combined_text = " ".join(
                filter(None, [description, interpretation, classification])
            )
            acmg_criteria = _extract_acmg_criteria(combined_text)

            # Extract submission year from various date formats
            # ClinVar uses "Jun 29, 2010" or "2024-03-14" or similar
            year = None
            if date_eval:
                try:
                    # Try ISO format first (2024-03-14)
                    year = int(date_eval.split("-")[0])
                    if year < 1990 or year > 2030:
                        # Try "Mon DD, YYYY" format
                        parts = date_eval.replace(",", "").split()
                        for part in parts:
                            if part.isdigit() and len(part) == 4:
                                year = int(part)
                                break
                        else:
                            year = None
                except (ValueError, IndexError):
                    pass

            sub = {
                "scv_accession": scv if scv else None,
                "submitter_name": submitter if submitter else None,
                "classification": _normalize_classification(classification),
                "review_status": review_status,
                "date_last_evaluated": date_eval if date_eval else None,
                "submission_year": year,
                "reported_phenotype": reported_pheno if reported_pheno else None,
                "acmg_criteria": acmg_criteria,
            }

            if vid not in submissions:
                submissions[vid] = []
            submissions[vid].append(sub)

            if row_num % 500_000 == 0:
                logger.info(
                    "Parsed %d submission rows, %d unique VariationIDs",
                    row_num,
                    len(submissions),
                )

    logger.info(
        "submission_summary parsed: %d submissions for %d VariationIDs",
        sum(len(v) for v in submissions.values()),
        len(submissions),
    )
    return submissions


def _parse_phenotype_ids(phenotype_ids: str) -> list[dict]:
    """Parse ClinVar PhenotypeIDS field into disease records.

    Format: 'MedGen:C0006142,OMIM:114480|MedGen:C0677776'
    Returns list of dicts with medgen_cui, omim_id, orphanet_id, mondo_id.
    """
    diseases = []
    if not phenotype_ids or phenotype_ids == "na":
        return diseases

    for group in phenotype_ids.split("|"):
        disease = {"medgen_cui": None, "omim_id": None, "orphanet_id": None, "mondo_id": None}
        for item in group.split(","):
            item = item.strip()
            if item.startswith("MedGen:"):
                disease["medgen_cui"] = item.split(":", 1)[1]
            elif item.startswith("OMIM:"):
                disease["omim_id"] = item.split(":", 1)[1]
            elif item.startswith("Orphanet:"):
                disease["orphanet_id"] = item.split(":", 1)[1]
            elif item.startswith("MONDO:"):
                disease["mondo_id"] = item.split(":", 1)[1]

        if disease["medgen_cui"] or disease["omim_id"]:
            diseases.append(disease)

    return diseases


def _parse_phenotype_names(phenotype_list: str) -> list[str]:
    """Parse ClinVar PhenotypeList field into disease names.

    Format: 'Breast cancer|Hereditary cancer-predisposing syndrome'
    """
    if not phenotype_list or phenotype_list == "na" or phenotype_list == "not provided":
        return []
    return [name.strip() for name in phenotype_list.split("|") if name.strip()]


def _extract_reported_phenotype_cuis(reported_phenotype: str | None) -> set[str]:
    """Extract MedGen CUIs from submission_summary ReportedPhenotypeInfo."""
    if not reported_phenotype:
        return set()
    return set(_MEDGEN_CUI_RE.findall(reported_phenotype))


def load_clinvar_data(
    conn,
    benign_records: list[dict],
    pathogenic_records: list[dict],
    submissions_by_vid: dict[int, list[dict]],
    batch_size: int = 5000,
) -> dict:
    """Load parsed ClinVar data into the VP database.

    Inserts genes, variants, diseases, submissions, and negative results.
    Detects conflicts (pathogenic submissions for same variant-disease).

    Returns stats dict.
    """
    stats = {
        "genes_inserted": 0,
        "variants_inserted": 0,
        "diseases_inserted": 0,
        "submissions_inserted": 0,
        "negative_results_inserted": 0,
        "conflicts_detected": 0,
        "skipped_no_disease": 0,
        "skipped_no_classification": 0,
    }

    # ── Build lookup caches ──────────────────────────────────────────

    # Gene cache: entrez_id → gene_id
    gene_cache: dict[int, int] = {}
    # Also symbol → gene_id for genes without entrez
    gene_symbol_cache: dict[str, int] = {}

    # Variant cache: (chr, pos, ref, alt) → variant_id
    variant_cache: dict[tuple, int] = {}

    # Disease cache: medgen_cui → disease_id
    disease_cache: dict[str, int] = {}

    # Pathogenic variant-disease pairs for conflict detection
    pathogenic_pairs: set[tuple[int, str]] = set()  # (variation_id, medgen_cui)
    for rec in pathogenic_records:
        pheno_ids = _parse_phenotype_ids(rec.get("phenotype_ids", ""))
        for d in pheno_ids:
            if d["medgen_cui"]:
                pathogenic_pairs.add((rec["variation_id"], d["medgen_cui"]))

    # Some benign aggregate variants still have P/LP submissions for a disease.
    for vid, subs in submissions_by_vid.items():
        for sub in subs:
            if sub.get("classification") not in {
                "pathogenic",
                "likely_pathogenic",
                "pathogenic/likely_pathogenic",
            }:
                continue
            for cui in _extract_reported_phenotype_cuis(sub.get("reported_phenotype")):
                pathogenic_pairs.add((vid, cui))

    # SCV cache to avoid duplicate inserts
    scv_cache: set[str] = set()

    # ── Insert genes ─────────────────────────────────────────────────

    all_records = benign_records + pathogenic_records
    for rec in all_records:
        entrez_id = rec.get("entrez_id")
        symbol = rec.get("gene_symbol", "")

        if not symbol:
            continue

        if entrez_id and entrez_id in gene_cache:
            continue
        if not entrez_id and symbol in gene_symbol_cache:
            continue

        conn.execute(
            """INSERT OR IGNORE INTO genes (entrez_id, gene_symbol, hgnc_id)
            VALUES (?, ?, ?)""",
            (entrez_id, symbol, rec.get("hgnc_id")),
        )

        # Retrieve gene_id
        if entrez_id:
            row = conn.execute(
                "SELECT gene_id FROM genes WHERE entrez_id = ?", (entrez_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT gene_id FROM genes WHERE gene_symbol = ?", (symbol,)
            ).fetchone()

        if row:
            gid = row[0]
            if entrez_id:
                gene_cache[entrez_id] = gid
            gene_symbol_cache[symbol] = gid
            stats["genes_inserted"] += 1

    conn.commit()
    logger.info("Inserted %d genes", stats["genes_inserted"])

    # ── Insert variants ──────────────────────────────────────────────

    for i, rec in enumerate(all_records):
        locus = (rec["chromosome"], rec["position"], rec["ref_allele"], rec["alt_allele"])
        if locus in variant_cache:
            continue

        entrez_id = rec.get("entrez_id")
        gene_id = None
        if entrez_id and entrez_id in gene_cache:
            gene_id = gene_cache[entrez_id]
        elif rec.get("gene_symbol") in gene_symbol_cache:
            gene_id = gene_symbol_cache[rec["gene_symbol"]]

        conn.execute(
            """INSERT OR IGNORE INTO variants
            (clinvar_variation_id, chromosome, position, ref_allele, alt_allele,
             variant_type, gene_id, rs_id, hgvs_coding, hgvs_protein, consequence_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec["variation_id"],
                rec["chromosome"],
                rec["position"],
                rec["ref_allele"],
                rec["alt_allele"],
                rec["variant_type"],
                gene_id,
                rec.get("rs_id"),
                rec.get("hgvs_coding"),
                rec.get("hgvs_protein"),
                rec.get("consequence_type"),
            ),
        )

        row = conn.execute(
            """SELECT variant_id FROM variants
            WHERE chromosome = ? AND position = ? AND ref_allele = ? AND alt_allele = ?""",
            locus,
        ).fetchone()

        if row:
            variant_cache[locus] = row[0]
            stats["variants_inserted"] += 1

        if (i + 1) % batch_size == 0:
            conn.commit()

    conn.commit()
    logger.info("Inserted %d variants", stats["variants_inserted"])

    # ── Insert diseases ──────────────────────────────────────────────

    for rec in all_records:
        pheno_ids = _parse_phenotype_ids(rec.get("phenotype_ids", ""))
        pheno_names = _parse_phenotype_names(rec.get("phenotype_list", ""))

        for j, disease_info in enumerate(pheno_ids):
            cui = disease_info.get("medgen_cui")
            if not cui or cui in disease_cache:
                continue

            name = pheno_names[j] if j < len(pheno_names) else "not provided"

            conn.execute(
                """INSERT OR IGNORE INTO diseases
                (medgen_cui, omim_id, orphanet_id, mondo_id, canonical_name)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    cui,
                    disease_info.get("omim_id"),
                    disease_info.get("orphanet_id"),
                    disease_info.get("mondo_id"),
                    name,
                ),
            )

            row = conn.execute(
                "SELECT disease_id FROM diseases WHERE medgen_cui = ?", (cui,)
            ).fetchone()
            if row:
                disease_cache[cui] = row[0]
                stats["diseases_inserted"] += 1

    conn.commit()
    logger.info("Inserted %d diseases", stats["diseases_inserted"])

    # ── Insert submissions ───────────────────────────────────────────

    submission_id_cache: dict[str, int] = {}  # scv → submission_id

    for rec in benign_records:
        vid = rec["variation_id"]
        locus = (rec["chromosome"], rec["position"], rec["ref_allele"], rec["alt_allele"])
        variant_id = variant_cache.get(locus)
        if not variant_id:
            continue

        subs = submissions_by_vid.get(vid, [])
        for sub in subs:
            scv = sub.get("scv_accession")
            if not scv or scv in scv_cache:
                continue

            acmg_json = None
            if sub["acmg_criteria"]:
                import json
                acmg_json = json.dumps(sub["acmg_criteria"])

            conn.execute(
                """INSERT OR IGNORE INTO vp_submissions
                (scv_accession, variant_id, submitter_name,
                 classification, review_status, date_last_evaluated,
                 submission_year, reported_phenotype, acmg_criteria)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scv,
                    variant_id,
                    sub.get("submitter_name"),
                    sub.get("classification"),
                    sub.get("review_status", ""),
                    sub.get("date_last_evaluated"),
                    sub.get("submission_year"),
                    sub.get("reported_phenotype"),
                    acmg_json,
                ),
            )

            scv_cache.add(scv)

            row = conn.execute(
                "SELECT submission_id FROM vp_submissions WHERE scv_accession = ?",
                (scv,),
            ).fetchone()
            if row:
                submission_id_cache[scv] = row[0]
                stats["submissions_inserted"] += 1

        if stats["submissions_inserted"] % batch_size == 0:
            conn.commit()

    conn.commit()
    logger.info("Inserted %d submissions", stats["submissions_inserted"])

    # ── Insert negative results ──────────────────────────────────────

    insert_count = 0
    for rec in benign_records:
        classification = rec.get("classification")
        if not classification:
            stats["skipped_no_classification"] += 1
            continue

        locus = (rec["chromosome"], rec["position"], rec["ref_allele"], rec["alt_allele"])
        variant_id = variant_cache.get(locus)
        if not variant_id:
            continue

        pheno_ids = _parse_phenotype_ids(rec.get("phenotype_ids", ""))
        if not pheno_ids:
            stats["skipped_no_disease"] += 1
            continue

        vid = rec["variation_id"]
        subs = submissions_by_vid.get(vid, [])

        for disease_info in pheno_ids:
            cui = disease_info.get("medgen_cui")
            if not cui:
                continue
            disease_id = disease_cache.get(cui)
            if not disease_id:
                continue

            # Conflict detection
            has_conflict = 1 if (vid, cui) in pathogenic_pairs else 0
            if has_conflict:
                stats["conflicts_detected"] += 1

            # Find best submission for this variant
            best_sub_id = None
            best_year = None
            best_criteria_count = 0
            for sub in subs:
                scv = sub.get("scv_accession")
                if scv and scv in submission_id_cache:
                    best_sub_id = submission_id_cache[scv]
                    if sub.get("submission_year"):
                        if best_year is None or sub["submission_year"] < best_year:
                            best_year = sub["submission_year"]
                    best_criteria_count = max(
                        best_criteria_count, len(sub.get("acmg_criteria", []))
                    )

            # Use variant-level year if no submission year
            if best_year is None:
                for sub in subs:
                    if sub.get("submission_year"):
                        best_year = sub["submission_year"]
                        break

            # Determine extraction method
            tier = rec["confidence_tier"]
            if tier == "gold":
                extraction_method = "review_status"
            elif tier == "silver":
                extraction_method = "submitter_concordance"
            else:
                extraction_method = "single_submission"

            conn.execute(
                """INSERT OR IGNORE INTO vp_negative_results
                (variant_id, disease_id, submission_id, classification,
                 evidence_type, confidence_tier, source_db, source_record_id,
                 extraction_method, submission_year, has_conflict, num_benign_criteria)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    variant_id,
                    disease_id,
                    best_sub_id,
                    classification,
                    rec["evidence_type"],
                    tier,
                    "clinvar",
                    str(vid),
                    extraction_method,
                    best_year,
                    has_conflict,
                    best_criteria_count,
                ),
            )
            insert_count += 1
            stats["negative_results_inserted"] += 1

        if insert_count % batch_size == 0:
            conn.commit()

    conn.commit()
    logger.info("Inserted %d negative results", stats["negative_results_inserted"])
    logger.info("Conflicts detected: %d", stats["conflicts_detected"])

    return stats


def run_clinvar_etl(
    db_path,
    data_dir: Path,
    batch_size: int = 5000,
) -> dict:
    """Run the full ClinVar ETL pipeline.

    Args:
        db_path: Path to VP SQLite database
        data_dir: Directory containing ClinVar downloads
        batch_size: Commit every N inserts

    Returns:
        Stats dict with counts
    """
    from negbiodb_vp.vp_db import get_connection

    variant_summary = data_dir / "variant_summary.txt.gz"
    submission_summary = data_dir / "submission_summary.txt.gz"

    if not variant_summary.exists():
        # Try uncompressed
        variant_summary = data_dir / "variant_summary.txt"
    if not submission_summary.exists():
        submission_summary = data_dir / "submission_summary.txt"

    logger.info("Parsing variant_summary: %s", variant_summary)
    benign_records, pathogenic_records = parse_variant_summary(variant_summary)

    # Collect all variation IDs for filtering submissions
    all_vids = {r["variation_id"] for r in benign_records}
    all_vids.update(r["variation_id"] for r in pathogenic_records)

    logger.info("Parsing submission_summary: %s", submission_summary)
    submissions = parse_submission_summary(submission_summary, variation_ids=all_vids)

    logger.info("Loading into database: %s", db_path)
    conn = get_connection(db_path)
    try:
        stats = load_clinvar_data(
            conn, benign_records, pathogenic_records, submissions, batch_size
        )

        # Record dataset version
        conn.execute(
            """INSERT OR IGNORE INTO dataset_versions
            (name, version, source_url, notes)
            VALUES ('ClinVar', 'latest', 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/', ?)""",
            (f"benign={len(benign_records)}, pathogenic={len(pathogenic_records)}",),
        )
        conn.commit()

        return stats
    finally:
        conn.close()

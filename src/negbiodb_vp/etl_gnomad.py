"""gnomAD ETL for NegBioDB VP domain.

Handles two data sources:
1. Gene constraint metrics (pLI, LOEUF, missense_z) → updates genes table
2. Population allele frequencies → updates variants table (gnomAD AFs)
3. Copper tier generation → inserts new variants + negative results for
   common variants (AF > 0.01) not in ClinVar
4. Sites VCF extraction → exports TSVs for local frequency/copper loading

Key decisions:
  - Constraint file: small (~15 MB), downloaded locally
  - Sites VCF: large (~15-20 GB), downloaded on HPC via gsutil
  - Copper tier threshold: global AF > 0.01 (BA1 standalone benign)
"""

import csv
import gzip
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# BA1 standalone benign threshold
COPPER_AF_THRESHOLD = 0.01

_VCF_AF_FIELD_CANDIDATES = {
    "af_global": ("AF", "AF_global", "gnomad_AF"),
    "af_afr": ("AF_afr", "AF_AFR", "AF-afr"),
    "af_amr": ("AF_amr", "AF_AMR", "AF-amr"),
    "af_asj": ("AF_asj", "AF_ASJ", "AF-asj"),
    "af_eas": ("AF_eas", "AF_EAS", "AF-eas"),
    "af_fin": ("AF_fin", "AF_FIN", "AF-fin"),
    "af_nfe": ("AF_nfe", "AF_NFE", "AF-nfe"),
    "af_sas": ("AF_sas", "AF_SAS", "AF-sas"),
    "af_oth": (
        "AF_oth",
        "AF_OTH",
        "AF_remaining",
        "AF_REMAINING",
        "AF-remaining",
    ),
}

_GNOMAD_CONSEQUENCE_MAP = {
    "missense_variant": "missense",
    "stop_gained": "nonsense",
    "stop_lost": "nonsense",
    "synonymous_variant": "synonymous",
    "frameshift_variant": "frameshift",
    "splice_donor_variant": "splice",
    "splice_acceptor_variant": "splice",
    "splice_region_variant": "splice",
    "inframe_deletion": "inframe_indel",
    "inframe_insertion": "inframe_indel",
    "intron_variant": "intronic",
}


def load_gene_constraints(conn, constraint_file: Path) -> dict:
    """Load gnomAD gene constraint metrics into genes table.

    Updates pli_score, loeuf_score, missense_z for existing genes.

    Args:
        conn: SQLite connection to VP database
        constraint_file: Path to gnomad.v4.1.constraint_metrics.tsv

    Returns:
        Stats dict
    """
    stats = {"genes_updated": 0, "genes_not_found": 0, "rows_parsed": 0}

    # Build gene_symbol → gene_id lookup
    gene_lookup = {}
    for row in conn.execute("SELECT gene_id, gene_symbol FROM genes"):
        gene_lookup[row[1].upper()] = row[0]

    best_by_gene: dict[int, tuple[tuple[int, int, int, int, int], float | None, float | None, float | None]] = {}

    with open(constraint_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["rows_parsed"] += 1
            symbol = row.get("gene", "").strip().upper()
            if not symbol:
                continue

            gene_id = gene_lookup.get(symbol)
            if gene_id is None:
                stats["genes_not_found"] += 1
                continue

            # Support both legacy headers and gnomAD v4.1 transcript-level exports.
            pli = _safe_float(_first_present(row, "lof.pLI", "lof_hc_lc.pLI", "pLI"))
            loeuf = _safe_float(
                _first_present(
                    row,
                    "lof.oe_ci.upper",
                    "lof_hc_lc.oe_ci.upper",
                    "oe_lof_upper",
                )
            )
            mis_z = _safe_float(_first_present(row, "mis.z_score", "mis_z"))

            rank = _constraint_row_rank(row, pli, loeuf, mis_z)
            current = best_by_gene.get(gene_id)
            if current is None or rank > current[0]:
                best_by_gene[gene_id] = (rank, pli, loeuf, mis_z)

    for gene_id, (_, pli, loeuf, mis_z) in best_by_gene.items():
        if pli is None and loeuf is None and mis_z is None:
            continue

        conn.execute(
            """UPDATE genes SET
                pli_score = COALESCE(?, pli_score),
                loeuf_score = COALESCE(?, loeuf_score),
                missense_z = COALESCE(?, missense_z),
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE gene_id = ?""",
            (pli, loeuf, mis_z, gene_id),
        )
        stats["genes_updated"] += 1

    conn.commit()
    logger.info(
        "Gene constraints: %d updated, %d not found in DB",
        stats["genes_updated"],
        stats["genes_not_found"],
    )
    return stats


def annotate_variant_frequencies(
    conn,
    frequency_tsv: Path,
    batch_size: int = 5000,
) -> dict:
    """Annotate existing variants with gnomAD allele frequencies.

    Reads a TSV file with columns: chromosome, position, ref, alt,
    af_global, af_afr, af_amr, af_asj, af_eas, af_fin, af_nfe, af_sas, af_oth

    This TSV is pre-extracted from gnomAD sites VCF on HPC.

    Returns stats dict.
    """
    stats = {"variants_annotated": 0, "variants_not_found": 0, "rows_parsed": 0}

    # Build variant locus lookup: (chr, pos, ref, alt) → variant_id
    variant_lookup = {}
    for row in conn.execute(
        "SELECT variant_id, chromosome, position, ref_allele, alt_allele FROM variants"
    ):
        key = (row[1], row[2], row[3], row[4])
        variant_lookup[key] = row[0]

    with open(frequency_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["rows_parsed"] += 1
            chrom = row.get("chromosome", "").replace("chr", "")
            try:
                pos = int(row.get("position", 0))
            except (ValueError, TypeError):
                continue
            ref = row.get("ref", "")
            alt = row.get("alt", "")

            key = (chrom, pos, ref, alt)
            variant_id = variant_lookup.get(key)
            if variant_id is None:
                stats["variants_not_found"] += 1
                continue

            conn.execute(
                """UPDATE variants SET
                    gnomad_af_global = ?,
                    gnomad_af_afr = ?, gnomad_af_amr = ?,
                    gnomad_af_asj = ?, gnomad_af_eas = ?,
                    gnomad_af_fin = ?, gnomad_af_nfe = ?,
                    gnomad_af_sas = ?, gnomad_af_oth = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE variant_id = ?""",
                (
                    _safe_float(row.get("af_global")),
                    _safe_float(row.get("af_afr")),
                    _safe_float(row.get("af_amr")),
                    _safe_float(row.get("af_asj")),
                    _safe_float(row.get("af_eas")),
                    _safe_float(row.get("af_fin")),
                    _safe_float(row.get("af_nfe")),
                    _safe_float(row.get("af_sas")),
                    _safe_float(row.get("af_oth")),
                    variant_id,
                ),
            )
            stats["variants_annotated"] += 1

            if stats["variants_annotated"] % batch_size == 0:
                conn.commit()

    conn.commit()
    logger.info(
        "Annotated %d variants with gnomAD AFs (%d not found)",
        stats["variants_annotated"],
        stats["variants_not_found"],
    )
    return stats


def generate_copper_tier(
    conn,
    copper_tsv: Path,
    af_threshold: float = COPPER_AF_THRESHOLD,
    batch_size: int = 5000,
) -> dict:
    """Generate copper-tier entries from gnomAD common variants.

    Reads a TSV of common variants (AF > threshold) pre-filtered on HPC.
    Inserts new variants and negative results for those not already in ClinVar.

    Columns: chromosome, position, ref, alt, af_global, consequence, gene_symbol

    Returns stats dict.
    """
    stats = {
        "variants_inserted": 0,
        "results_inserted": 0,
        "skipped_already_in_db": 0,
        "rows_parsed": 0,
    }

    # Build existing variant locus set
    existing_loci = set()
    for row in conn.execute(
        "SELECT chromosome, position, ref_allele, alt_allele FROM variants"
    ):
        existing_loci.add((row[0], row[1], row[2], row[3]))

    # Get or create "not provided" disease for copper tier
    conn.execute(
        """INSERT OR IGNORE INTO diseases (medgen_cui, canonical_name)
        VALUES ('CN169374', 'not provided')"""
    )
    disease_row = conn.execute(
        "SELECT disease_id FROM diseases WHERE medgen_cui = 'CN169374'"
    ).fetchone()
    disease_id = disease_row[0]

    # Gene symbol → gene_id lookup
    gene_lookup = {}
    for row in conn.execute("SELECT gene_id, gene_symbol FROM genes"):
        gene_lookup[row[1].upper()] = row[0]

    insert_count = 0
    with open(copper_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["rows_parsed"] += 1
            chrom = row.get("chromosome", "").replace("chr", "")
            try:
                pos = int(row.get("position", 0))
            except (ValueError, TypeError):
                continue
            ref = row.get("ref", "")
            alt = row.get("alt", "")

            locus = (chrom, pos, ref, alt)
            if locus in existing_loci:
                stats["skipped_already_in_db"] += 1
                continue

            af_global = _safe_float(row.get("af_global"))
            if af_global is None or af_global <= af_threshold:
                continue

            consequence = row.get("consequence", "other")
            gene_symbol = row.get("gene_symbol", "").strip().upper()
            gene_id = gene_lookup.get(gene_symbol)

            # Determine variant type
            if len(ref) == 1 and len(alt) == 1:
                variant_type = "single nucleotide variant"
            elif len(ref) > len(alt):
                variant_type = "Deletion"
            elif len(ref) < len(alt):
                variant_type = "Insertion"
            else:
                variant_type = "Indel"

            # Insert variant
            conn.execute(
                """INSERT OR IGNORE INTO variants
                (chromosome, position, ref_allele, alt_allele, variant_type,
                 gene_id, gnomad_af_global, consequence_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (chrom, pos, ref, alt, variant_type, gene_id, af_global, consequence),
            )

            var_row = conn.execute(
                """SELECT variant_id FROM variants
                WHERE chromosome = ? AND position = ? AND ref_allele = ? AND alt_allele = ?""",
                locus,
            ).fetchone()
            if not var_row:
                continue

            variant_id = var_row[0]
            existing_loci.add(locus)
            stats["variants_inserted"] += 1

            # Insert copper-tier negative result
            source_record = f"chr{chrom}:{pos}:{ref}:{alt}"
            conn.execute(
                """INSERT OR IGNORE INTO vp_negative_results
                (variant_id, disease_id, classification, evidence_type,
                 confidence_tier, source_db, source_record_id, extraction_method,
                 has_conflict, num_benign_criteria)
                VALUES (?, ?, 'benign', 'population_frequency', 'copper',
                        'gnomad', ?, 'population_af_threshold', 0, 0)""",
                (variant_id, disease_id, source_record),
            )
            stats["results_inserted"] += 1

            insert_count += 1
            if insert_count % batch_size == 0:
                conn.commit()

    conn.commit()
    logger.info(
        "Copper tier: %d variants, %d results inserted (%d skipped existing)",
        stats["variants_inserted"],
        stats["results_inserted"],
        stats["skipped_already_in_db"],
    )
    return stats


def export_gnomad_site_tsvs(
    conn,
    vcf_paths: list[Path],
    frequencies_out: Path | None = None,
    copper_out: Path | None = None,
    af_threshold: float = COPPER_AF_THRESHOLD,
) -> dict:
    """Stream gnomAD sites VCFs into TSVs consumable by load_gnomad.py.

    The exported TSVs are meant to be generated on HPC from the large gnomAD
    sites VCF shards, then transferred to local storage and loaded with:

      - annotate_variant_frequencies(..., variant_frequencies.tsv)
      - generate_copper_tier(..., copper_variants.tsv)
    """
    if frequencies_out is None and copper_out is None:
        raise ValueError("At least one of frequencies_out or copper_out must be provided")

    stats = {
        "vcf_files_processed": 0,
        "vcf_records_read": 0,
        "alt_alleles_seen": 0,
        "frequency_rows_written": 0,
        "copper_rows_written": 0,
        "skipped_symbolic": 0,
        "skipped_missing_af": 0,
    }

    existing_loci = {
        (row[0], row[1], row[2], row[3])
        for row in conn.execute(
            "SELECT chromosome, position, ref_allele, alt_allele FROM variants"
        )
    }

    seen_freq = set()
    seen_copper = set()

    freq_fh = None
    copper_fh = None
    freq_writer = None
    copper_writer = None

    try:
        if frequencies_out is not None:
            frequencies_out.parent.mkdir(parents=True, exist_ok=True)
            freq_fh = open(frequencies_out, "w", newline="")
            freq_writer = csv.writer(freq_fh, delimiter="\t")
            freq_writer.writerow(
                [
                    "chromosome",
                    "position",
                    "ref",
                    "alt",
                    "af_global",
                    "af_afr",
                    "af_amr",
                    "af_asj",
                    "af_eas",
                    "af_fin",
                    "af_nfe",
                    "af_sas",
                    "af_oth",
                ]
            )

        if copper_out is not None:
            copper_out.parent.mkdir(parents=True, exist_ok=True)
            copper_fh = open(copper_out, "w", newline="")
            copper_writer = csv.writer(copper_fh, delimiter="\t")
            copper_writer.writerow(
                [
                    "chromosome",
                    "position",
                    "ref",
                    "alt",
                    "af_global",
                    "consequence",
                    "gene_symbol",
                ]
            )

        for vcf_path in vcf_paths:
            stats["vcf_files_processed"] += 1
            _export_single_gnomad_vcf(
                vcf_path,
                existing_loci,
                freq_writer,
                copper_writer,
                seen_freq,
                seen_copper,
                stats,
                af_threshold,
            )
    finally:
        if freq_fh is not None:
            freq_fh.close()
        if copper_fh is not None:
            copper_fh.close()

    logger.info(
        "gnomAD sites extraction: %d VCFs, %d AF rows, %d copper rows",
        stats["vcf_files_processed"],
        stats["frequency_rows_written"],
        stats["copper_rows_written"],
    )
    return stats


def merge_gnomad_tsv_shards(
    shard_paths: list[Path],
    output_path: Path,
    key_columns: tuple[str, ...] = ("chromosome", "position", "ref", "alt"),
) -> dict:
    """Merge per-chromosome TSV shards into a single deduplicated TSV."""
    if not shard_paths:
        raise ValueError("No shard paths provided")

    shard_paths = sorted(Path(path) for path in shard_paths)
    stats = {
        "shards_read": 0,
        "rows_read": 0,
        "rows_written": 0,
        "duplicate_rows_skipped": 0,
    }

    header = None
    seen_keys = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as out_fh:
        writer = None
        for shard_path in shard_paths:
            stats["shards_read"] += 1
            with open(shard_path) as in_fh:
                reader = csv.DictReader(in_fh, delimiter="\t")
                if reader.fieldnames is None:
                    continue

                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(
                        out_fh,
                        fieldnames=header,
                        delimiter="\t",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise ValueError(
                        f"Header mismatch in {shard_path}: expected {header}, got {reader.fieldnames}"
                    )

                for row in reader:
                    stats["rows_read"] += 1
                    key = tuple(row.get(column, "") for column in key_columns)
                    if key in seen_keys:
                        stats["duplicate_rows_skipped"] += 1
                        continue
                    seen_keys.add(key)
                    writer.writerow(row)
                    stats["rows_written"] += 1

    if header is None:
        raise ValueError("No readable TSV shards found")

    logger.info(
        "Merged %d gnomAD TSV shards into %s (%d rows, %d duplicates skipped)",
        stats["shards_read"],
        output_path,
        stats["rows_written"],
        stats["duplicate_rows_skipped"],
    )
    return stats


def _safe_float(value) -> float | None:
    """Safely convert to float, returning None for invalid values."""
    if value is None or value == "" or value == "NA" or value == "nan":
        return None
    try:
        v = float(value)
        if v != v:  # NaN check
            return None
        return v
    except (ValueError, TypeError):
        return None


def _first_present(row: dict, *columns: str) -> str | None:
    """Return the first present non-empty column value from a row."""
    for column in columns:
        if column in row:
            value = row.get(column)
            if value not in (None, ""):
                return value
    return None


def _constraint_row_rank(
    row: dict,
    pli: float | None,
    loeuf: float | None,
    mis_z: float | None,
) -> tuple[int, int, int, int, int]:
    """Rank transcript rows so one representative row is chosen per gene."""
    metric_count = sum(value is not None for value in (pli, loeuf, mis_z))
    transcript = row.get("transcript", "")
    return (
        1 if metric_count else 0,
        metric_count,
        1 if str(row.get("mane_select", "")).lower() == "true" else 0,
        1 if str(row.get("canonical", "")).lower() == "true" else 0,
        1 if transcript.startswith("NM_") else 0,
    )


def _export_single_gnomad_vcf(
    vcf_path: Path,
    existing_loci: set[tuple[str, int, str, str]],
    freq_writer,
    copper_writer,
    seen_freq: set[tuple[str, int, str, str]],
    seen_copper: set[tuple[str, int, str, str]],
    stats: dict,
    af_threshold: float,
) -> None:
    """Process one gnomAD sites VCF shard and write TSV rows."""
    vep_spec = None
    opener = gzip.open if str(vcf_path).endswith((".gz", ".bgz")) else open

    with opener(vcf_path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("##INFO="):
                parsed = _parse_vep_spec(line)
                if parsed is not None:
                    vep_spec = parsed
                continue
            if line.startswith("#"):
                continue

            stats["vcf_records_read"] += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom = fields[0].removeprefix("chr")
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            ref = fields[3]
            alt_values = fields[4].split(",")
            info = _parse_vcf_info(fields[7])

            for alt_index, alt in enumerate(alt_values):
                stats["alt_alleles_seen"] += 1
                if _is_symbolic_allele(ref, alt):
                    stats["skipped_symbolic"] += 1
                    continue

                locus = (chrom, pos, ref, alt)
                af_values = {
                    key: _extract_number_a_value(info, candidates, alt_index)
                    for key, candidates in _VCF_AF_FIELD_CANDIDATES.items()
                }
                af_global = _safe_float(af_values["af_global"])
                if af_global is None:
                    stats["skipped_missing_af"] += 1

                if freq_writer is not None and locus in existing_loci and locus not in seen_freq:
                    freq_writer.writerow(
                        [
                            chrom,
                            pos,
                            ref,
                            alt,
                            af_values["af_global"] or "",
                            af_values["af_afr"] or "",
                            af_values["af_amr"] or "",
                            af_values["af_asj"] or "",
                            af_values["af_eas"] or "",
                            af_values["af_fin"] or "",
                            af_values["af_nfe"] or "",
                            af_values["af_sas"] or "",
                            af_values["af_oth"] or "",
                        ]
                    )
                    seen_freq.add(locus)
                    stats["frequency_rows_written"] += 1

                if (
                    copper_writer is not None
                    and locus not in existing_loci
                    and locus not in seen_copper
                    and af_global is not None
                    and af_global > af_threshold
                ):
                    consequence, gene_symbol = _extract_annotation_fields(info, alt, vep_spec)
                    copper_writer.writerow(
                        [
                            chrom,
                            pos,
                            ref,
                            alt,
                            af_values["af_global"] or "",
                            consequence,
                            gene_symbol,
                        ]
                    )
                    seen_copper.add(locus)
                    stats["copper_rows_written"] += 1


def _parse_vcf_info(info_field: str) -> dict[str, str | bool]:
    """Parse a VCF INFO field into a key/value dict."""
    info: dict[str, str | bool] = {}
    for item in info_field.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = True
    return info


def _extract_number_a_value(
    info: dict[str, str | bool],
    field_candidates: tuple[str, ...],
    alt_index: int,
) -> str | None:
    """Extract an allele-specific INFO Number=A value."""
    for field in field_candidates:
        raw = info.get(field)
        if raw in (None, "", True):
            continue
        values = str(raw).split(",")
        if alt_index < len(values):
            return values[alt_index]
    return None


def _parse_vep_spec(header_line: str) -> tuple[str, list[str]] | None:
    """Parse INFO header metadata for CSQ/VEP-style annotations."""
    m = re.search(r"ID=([^,>]+)", header_line)
    fmt = re.search(r"Format: ([^\"]+)", header_line)
    if not m or not fmt:
        return None
    info_id = m.group(1)
    if info_id not in {"vep", "CSQ", "vep_annotation"}:
        return None
    fields = [field.strip() for field in fmt.group(1).split("|")]
    return info_id, fields


def _extract_annotation_fields(
    info: dict[str, str | bool],
    alt: str,
    vep_spec: tuple[str, list[str]] | None,
) -> tuple[str, str]:
    """Extract normalized consequence and gene symbol for one alt allele."""
    consequence = _normalize_consequence(
        _first_present(
            info,
            "consequence",
            "Consequence",
            "most_severe_consequence",
            "most_severe_csq",
        )
    )
    gene_symbol = str(
        _first_present(info, "gene_symbol", "SYMBOL", "symbol", "gene", "Gene") or ""
    ).strip().upper()

    if vep_spec is None:
        return consequence, gene_symbol

    info_id, fields = vep_spec
    raw = info.get(info_id)
    if raw in (None, "", True):
        return consequence, gene_symbol

    field_index = {name: idx for idx, name in enumerate(fields)}
    allele_idx = field_index.get("Allele")
    consequence_idx = field_index.get("Consequence")
    symbol_idx = field_index.get("SYMBOL")

    for annotation in str(raw).split(","):
        parts = annotation.split("|")
        if allele_idx is not None and allele_idx < len(parts):
            ann_alt = parts[allele_idx]
            if ann_alt and ann_alt != alt:
                continue
        if consequence_idx is not None and consequence_idx < len(parts):
            consequence = _normalize_consequence(parts[consequence_idx])
        if symbol_idx is not None and symbol_idx < len(parts) and parts[symbol_idx]:
            gene_symbol = parts[symbol_idx].upper()
        break

    return consequence, gene_symbol


def _normalize_consequence(raw: str | bool | None) -> str:
    """Map raw gnomAD/VEP consequence labels to DB-safe consequence_type values."""
    if raw in (None, "", True):
        return "other"

    text = str(raw).strip()
    if not text:
        return "other"

    for token in text.split("&"):
        mapped = _GNOMAD_CONSEQUENCE_MAP.get(token)
        if mapped:
            return mapped
    return "other"


def _is_symbolic_allele(ref: str, alt: str) -> bool:
    """Return True for symbolic or unsupported VCF alleles."""
    return (
        not ref
        or not alt
        or alt == "*"
        or alt.startswith("<")
        or "[" in alt
        or "]" in alt
    )

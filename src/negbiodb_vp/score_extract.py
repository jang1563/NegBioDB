"""HPC-oriented score extraction utilities for VP variants.

This module exports the variant loci we care about, then streams large
third-party score files (CADD, REVEL, AlphaMissense) and emits one merged TSV
that `scripts_vp/load_scores.py` can load locally.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

SCORE_OUTPUT_COLUMNS = [
    "chromosome",
    "position",
    "ref",
    "alt",
    "cadd_phred",
    "revel_score",
    "alphamissense_score",
    "alphamissense_class",
    "phylop_score",
    "gerp_score",
    "sift_score",
    "polyphen2_score",
]


def export_score_targets(conn, output_tsv: Path) -> dict:
    """Export variant loci needed for HPC score extraction."""
    stats = {"targets_exported": 0}
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "variant_id",
                "chromosome",
                "position",
                "ref",
                "alt",
                "consequence_type",
            ]
        )
        for row in conn.execute(
            """SELECT variant_id, chromosome, position, ref_allele, alt_allele, consequence_type
               FROM variants
               ORDER BY variant_id"""
        ):
            writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5] or ""])
            stats["targets_exported"] += 1

    logger.info("Exported %d VP score targets to %s", stats["targets_exported"], output_tsv)
    return stats


def extract_scores_for_targets(
    targets_tsv: Path,
    output_tsv: Path,
    cadd_tsv: Path | None = None,
    revel_tsv: Path | None = None,
    alphamissense_tsv: Path | None = None,
) -> dict:
    """Stream large score files and emit one merged TSV for local loading."""
    targets = _load_score_targets(targets_tsv)
    stats = {
        "targets_loaded": len(targets),
        "cadd_rows_read": 0,
        "revel_rows_read": 0,
        "alphamissense_rows_read": 0,
        "cadd_matches": 0,
        "revel_matches": 0,
        "alphamissense_matches": 0,
        "rows_written": 0,
    }

    if cadd_tsv is not None:
        stats["cadd_rows_read"], stats["cadd_matches"] = _apply_cadd_scores(targets, cadd_tsv)
    if revel_tsv is not None:
        stats["revel_rows_read"], stats["revel_matches"] = _apply_revel_scores(targets, revel_tsv)
    if alphamissense_tsv is not None:
        (
            stats["alphamissense_rows_read"],
            stats["alphamissense_matches"],
        ) = _apply_alphamissense_scores(targets, alphamissense_tsv)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tsv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SCORE_OUTPUT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for key in sorted(targets, key=lambda x: (x[0], x[1], x[2], x[3])):
            rec = targets[key]
            if not _has_any_score(rec):
                continue
            writer.writerow({col: rec.get(col, "") for col in SCORE_OUTPUT_COLUMNS})
            stats["rows_written"] += 1

    logger.info(
        "Extracted merged VP scores for %d target variants to %s",
        stats["rows_written"],
        output_tsv,
    )
    return stats


def _load_score_targets(targets_tsv: Path) -> dict[tuple[str, int, str, str], dict]:
    targets = {}
    with open(targets_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chrom = (row.get("chromosome") or "").replace("chr", "")
            try:
                pos = int(row.get("position", 0))
            except (ValueError, TypeError):
                continue
            ref = row.get("ref", "")
            alt = row.get("alt", "")
            if not (chrom and pos and ref and alt):
                continue
            key = (chrom, pos, ref, alt)
            targets[key] = {
                "chromosome": chrom,
                "position": pos,
                "ref": ref,
                "alt": alt,
                "consequence_type": row.get("consequence_type", "") or "",
                "cadd_phred": "",
                "revel_score": "",
                "alphamissense_score": "",
                "alphamissense_class": "",
                "phylop_score": "",
                "gerp_score": "",
                "sift_score": "",
                "polyphen2_score": "",
            }
    return targets


def _apply_cadd_scores(targets: dict, cadd_tsv: Path) -> tuple[int, int]:
    rows_read = 0
    matches = 0
    with _open_table_reader(cadd_tsv) as reader:
        for row in reader:
            rows_read += 1
            key = _score_key(
                row,
                chrom_cols=("#Chrom", "Chrom", "chr", "chromosome", "CHROM"),
                pos_cols=("Pos", "POS", "position", "Position"),
                ref_cols=("Ref", "REF", "ref"),
                alt_cols=("Alt", "ALT", "alt"),
            )
            if key is None or key not in targets:
                continue
            rec = targets[key]
            rec["cadd_phred"] = _pick(row, "PHRED", "CADD_PHRED", "cadd_phred") or rec["cadd_phred"]
            rec["phylop_score"] = _pick(
                row,
                "PhyloP100way_vertebrate",
                "phyloP100way_vertebrate",
                "phylop_score",
            ) or rec["phylop_score"]
            rec["gerp_score"] = _pick(row, "GerpRS", "gerp_score", "GERP_RS") or rec["gerp_score"]
            rec["sift_score"] = _pick(row, "SIFTval", "SIFT", "sift_score") or rec["sift_score"]
            rec["polyphen2_score"] = _pick(
                row,
                "PolyPhenVal",
                "Polyphen2_HVAR",
                "polyphen2_score",
            ) or rec["polyphen2_score"]
            matches += 1
    return rows_read, matches


def _apply_revel_scores(targets: dict, revel_tsv: Path) -> tuple[int, int]:
    rows_read = 0
    matches = 0
    with _open_table_reader(revel_tsv) as reader:
        for row in reader:
            rows_read += 1
            key = _score_key(
                row,
                chrom_cols=("chr", "chromosome", "Chromosome", "CHROM"),
                pos_cols=("grch38_pos", "hg38_pos", "position", "Position", "pos"),
                ref_cols=("ref", "REF"),
                alt_cols=("alt", "ALT"),
            )
            if key is None or key not in targets:
                continue
            rec = targets[key]
            rec["revel_score"] = _pick(row, "REVEL", "revel_score", "score", "revel") or rec["revel_score"]
            matches += 1
    return rows_read, matches


def _apply_alphamissense_scores(targets: dict, alphamissense_tsv: Path) -> tuple[int, int]:
    rows_read = 0
    matches = 0
    with _open_table_reader(alphamissense_tsv) as reader:
        for row in reader:
            rows_read += 1
            genome = (row.get("genome") or "").strip().lower()
            if genome and genome not in {"hg38", "grch38"}:
                continue
            key = _score_key(
                row,
                chrom_cols=("CHROM", "chromosome", "chr", "Chrom"),
                pos_cols=("POS", "position", "Position", "pos"),
                ref_cols=("REF", "ref"),
                alt_cols=("ALT", "alt"),
            )
            if key is None or key not in targets:
                continue
            rec = targets[key]
            rec["alphamissense_score"] = _pick(
                row,
                "am_pathogenicity",
                "alphamissense_score",
                "score",
            ) or rec["alphamissense_score"]
            rec["alphamissense_class"] = _pick(
                row,
                "am_class",
                "alphamissense_class",
                "class",
            ) or rec["alphamissense_class"]
            matches += 1
    return rows_read, matches


class _ReaderContext:
    def __init__(self, fh, reader):
        self._fh = fh
        self._reader = reader

    def __enter__(self):
        return self._reader

    def __exit__(self, exc_type, exc, tb):
        self._fh.close()
        return False


class _ZipReaderContext:
    def __init__(self, archive: zipfile.ZipFile, binary_fh):
        self.archive = archive
        self.binary_fh = binary_fh
        self.text_fh = None
        self.reader = None

    def __enter__(self):
        self.text_fh = io.TextIOWrapper(self.binary_fh, encoding="utf-8", errors="replace")
        self.reader = _make_reader(self.text_fh)
        return self.reader

    def __exit__(self, exc_type, exc, tb):
        if self.text_fh is not None:
            self.text_fh.close()
        self.archive.close()
        return False


def _open_table_reader(path: Path):
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            archive.close()
            raise ValueError(f"No files found inside zip archive: {path}")
        return _ZipReaderContext(archive, archive.open(members[0], "r"))

    if path.suffix == ".gz":
        fh = gzip.open(path, "rt", errors="replace")
        return _ReaderContext(fh, _make_reader(fh))

    fh = open(path, "r", errors="replace")
    return _ReaderContext(fh, _make_reader(fh))


def _make_reader(fh):
    header = fh.readline()
    if not header:
        raise ValueError("Empty score table")
    delimiter = "\t" if header.count("\t") >= header.count(",") else ","
    fh.seek(0)
    return csv.DictReader(fh, delimiter=delimiter)


def _score_key(
    row: dict,
    chrom_cols: tuple[str, ...],
    pos_cols: tuple[str, ...],
    ref_cols: tuple[str, ...],
    alt_cols: tuple[str, ...],
) -> tuple[str, int, str, str] | None:
    chrom = (_pick(row, *chrom_cols) or "").replace("chr", "")
    pos_raw = _pick(row, *pos_cols) or ""
    ref = _pick(row, *ref_cols) or ""
    alt = _pick(row, *alt_cols) or ""
    try:
        pos = int(pos_raw)
    except (ValueError, TypeError):
        return None
    if not (chrom and ref and alt):
        return None
    return chrom, pos, ref, alt


def _pick(row: dict, *columns: str) -> str | None:
    for col in columns:
        if col in row:
            value = row.get(col)
            if value not in (None, ""):
                return str(value).strip()
    return None


def _has_any_score(rec: dict) -> bool:
    return any(
        rec.get(col) not in (None, "")
        for col in SCORE_OUTPUT_COLUMNS
        if col not in {"chromosome", "position", "ref", "alt"}
    )

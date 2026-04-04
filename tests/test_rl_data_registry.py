"""Tests for negbiorl.data_registry."""

import json
import tempfile
from pathlib import Path

import pytest

from negbiorl.data_registry import (
    ALL_DOMAINS,
    DOMAIN_REGISTRY,
    TRAIN_DOMAINS,
    TRANSFER_TEST_DOMAIN,
    TRAINING_MODELS,
    get_domain,
    get_export_path,
    get_gold_answer_field,
    get_gold_class_field,
    get_l1_parser,
    get_prefixed_task,
    load_jsonl,
    parse_l4_unified,
)


# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

class TestDomainRegistry:
    """Verify registry is complete and consistent."""

    def test_all_five_domains_registered(self):
        assert set(ALL_DOMAINS) == {"dti", "ct", "ppi", "ge", "vp"}

    def test_train_domains_exclude_vp(self):
        assert set(TRAIN_DOMAINS) == {"dti", "ct", "ppi", "ge"}
        assert TRANSFER_TEST_DOMAIN == "vp"

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_required_fields_present(self, domain):
        reg = get_domain(domain)
        required = [
            "label", "eval_module", "prompt_module", "exports_dir",
            "results_dir", "gold_answer_field", "gold_class_field",
            "l1_choices", "l1_file", "l4_file", "parse_l1", "parse_l4",
            "l4_returns_tuple", "evidence_keywords", "l3_judge_dims",
        ]
        for field in required:
            assert field in reg, f"Missing field '{field}' in domain '{domain}'"

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_l1_choices_valid(self, domain):
        reg = get_domain(domain)
        assert reg["l1_choices"] in (4, 5)

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_evidence_keywords_nonempty(self, domain):
        reg = get_domain(domain)
        assert len(reg["evidence_keywords"]) >= 5

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_l3_judge_has_4_dims(self, domain):
        reg = get_domain(domain)
        assert len(reg["l3_judge_dims"]) == 4

    def test_unknown_domain_raises(self):
        with pytest.raises(KeyError, match="Unknown domain"):
            get_domain("unknown")


# ---------------------------------------------------------------------------
# Field name handling
# ---------------------------------------------------------------------------

class TestFieldNames:
    def test_dti_uses_correct_answer(self):
        assert get_gold_answer_field("dti") == "correct_answer"

    def test_ct_uses_gold_answer(self):
        assert get_gold_answer_field("ct") == "gold_answer"

    def test_ppi_uses_gold_answer(self):
        assert get_gold_answer_field("ppi") == "gold_answer"

    def test_ge_uses_gold_answer(self):
        assert get_gold_answer_field("ge") == "gold_answer"

    def test_vp_uses_gold_answer(self):
        assert get_gold_answer_field("vp") == "gold_answer"

    def test_dti_class_field(self):
        assert get_gold_class_field("dti") == "class"

    def test_others_use_gold_category(self):
        for domain in ["ct", "ppi", "ge", "vp"]:
            assert get_gold_class_field(domain) == "gold_category"


# ---------------------------------------------------------------------------
# Export paths
# ---------------------------------------------------------------------------

class TestExportPaths:
    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_l1_export_path_valid(self, domain):
        path = get_export_path(domain, "l1")
        assert path.suffix == ".jsonl"
        # Verify the path contains the expected exports directory
        assert "exports" in str(path)

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_l4_export_path_valid(self, domain):
        path = get_export_path(domain, "l4")
        assert path.suffix == ".jsonl"

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="No export file"):
            get_export_path("dti", "l99")


# ---------------------------------------------------------------------------
# L4 return type tracking
# ---------------------------------------------------------------------------

class TestL4ReturnTypes:
    def test_ge_is_non_tuple(self):
        assert get_domain("ge")["l4_returns_tuple"] is False

    def test_others_are_tuple(self):
        for domain in ["dti", "ct", "ppi", "vp"]:
            assert get_domain(domain)["l4_returns_tuple"] is True


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_load_valid_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        path.write_text("\n".join(json.dumps(r) for r in records))

        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["text"] == "world"

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_jsonl(path) == []

    def test_load_with_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n')
        loaded = load_jsonl(path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Training models
# ---------------------------------------------------------------------------

class TestTrainingModels:
    def test_primary_model_exists(self):
        primary = [k for k, v in TRAINING_MODELS.items() if v["role"] == "primary"]
        assert len(primary) == 1
        assert "qwen3" in primary[0]

    def test_historical_baselines_exist(self):
        baselines = [k for k, v in TRAINING_MODELS.items() if v["role"] == "historical_baseline"]
        assert len(baselines) >= 2

    def test_all_have_hf_id(self):
        for name, info in TRAINING_MODELS.items():
            assert "hf_id" in info
            assert "/" in info["hf_id"]


# ---------------------------------------------------------------------------
# Prefixed task
# ---------------------------------------------------------------------------

class TestPrefixedTask:
    def test_dti_bare(self):
        assert get_prefixed_task("dti", "l1") == "l1"
        assert get_prefixed_task("dti", "l4") == "l4"

    def test_ct_prefixed(self):
        assert get_prefixed_task("ct", "l1") == "ct-l1"
        assert get_prefixed_task("ct", "l4") == "ct-l4"

    def test_ppi_prefixed(self):
        assert get_prefixed_task("ppi", "l3") == "ppi-l3"

    def test_ge_prefixed(self):
        assert get_prefixed_task("ge", "l4") == "ge-l4"

    def test_vp_prefixed(self):
        assert get_prefixed_task("vp", "l1") == "vp-l1"


# ---------------------------------------------------------------------------
# L1 parser
# ---------------------------------------------------------------------------

class TestL1Parser:
    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_parser_loads(self, domain):
        parser = get_l1_parser(domain)
        assert callable(parser)

    def test_dti_parser_extracts_letter(self):
        parser = get_l1_parser("dti")
        assert parser("B") == "B"
        assert parser("The answer is C.") is not None

    def test_ct_parser_extracts_letter(self):
        parser = get_l1_parser("ct")
        assert parser("A") == "A"


# ---------------------------------------------------------------------------
# parse_l4_unified
# ---------------------------------------------------------------------------

class TestParseL4Unified:
    def test_returns_tuple_all_domains(self):
        for domain in ALL_DOMAINS:
            result = parse_l4_unified("tested", domain)
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_ge_non_tuple_wrapped(self):
        answer, evidence = parse_l4_unified("tested", "ge")
        # GE parser wraps to (answer, None)
        assert evidence is None

    def test_dti_tuple_returned(self):
        answer, evidence = parse_l4_unified("tested\nEvidence: ChEMBL binding", "dti")
        assert isinstance(answer, str) or answer is None

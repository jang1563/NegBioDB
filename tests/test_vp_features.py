"""Tests for VP feature engineering."""

import math

import numpy as np
import pytest

from negbiodb_vp.vp_features import (
    FEATURE_NAMES,
    TABULAR_DIM,
    compute_aa_features,
    compute_features,
)


class TestFeatureDimension:
    def test_tabular_dim_is_56(self):
        assert TABULAR_DIM == 56

    def test_feature_names_match_dim(self):
        assert len(FEATURE_NAMES) == TABULAR_DIM

    def test_feature_names_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


class TestComputeAaFeatures:
    def test_missense_three_letter(self):
        feats = compute_aa_features("p.Ala1708Asp")
        assert len(feats) == 5
        # Blosum62(A,D) should be negative
        assert feats[0] < 0  # blosum62
        assert feats[1] > 0  # grantham distance

    def test_missense_one_letter(self):
        feats = compute_aa_features("p.A1708D")
        assert len(feats) == 5
        assert feats[0] < 0  # A→D is unfavorable

    def test_synonymous_zeros(self):
        feats = compute_aa_features("p.Ala100Ala")
        assert feats == [4.0, 0.0, 0.0, 0.0, 0.0]  # Self-match

    def test_none_zeros(self):
        feats = compute_aa_features(None)
        assert feats == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_empty_zeros(self):
        feats = compute_aa_features("")
        assert feats == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_stop_codon_zeros(self):
        feats = compute_aa_features("p.Arg100*")
        # * not in _AA_CODES → all zeros
        assert feats == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestComputeFeatures:
    def _make_row(self, **overrides):
        """Create a complete test row with default values."""
        row = {
            "cadd_phred": 25.0,
            "revel_score": 0.8,
            "alphamissense_score": 0.7,
            "phylop_score": 4.5,
            "gerp_score": 3.2,
            "sift_score": 0.01,
            "polyphen2_score": 0.95,
            "variant_type": "single nucleotide variant",
            "consequence_type": "missense",
            "gnomad_af_global": 0.001,
            "gnomad_af_afr": 0.002,
            "gnomad_af_amr": 0.001,
            "gnomad_af_asj": 0.0005,
            "gnomad_af_eas": 0.001,
            "gnomad_af_fin": 0.001,
            "gnomad_af_nfe": 0.001,
            "gnomad_af_sas": 0.001,
            "gnomad_af_oth": 0.001,
            "pli_score": 0.99,
            "loeuf_score": 0.12,
            "missense_z": 3.5,
            "gene_moi": "AD",
            "hgvs_protein": "p.Ala100Asp",
            "variant_degree": 3,
            "disease_degree": 50,
            "num_submissions": 5,
            "num_submitters": 3,
            "max_population_af": 0.001,
            "is_in_known_domain": True,
            "gene_disease_count": 10,
            "clingen_validity": "Definitive",
            "exon_flag": True,
            "has_functional_evidence": False,
            "num_benign_criteria": 2,
        }
        row.update(overrides)
        return row

    def test_output_shape(self):
        row = self._make_row()
        features = compute_features(row)
        assert features.shape == (TABULAR_DIM,)
        assert features.dtype == np.float32

    def test_variant_scores_populated(self):
        row = self._make_row()
        features = compute_features(row)
        assert features[0] == pytest.approx(25.0)  # cadd_phred
        assert features[1] == pytest.approx(0.8)   # revel
        assert features[6] == pytest.approx(0.95)  # polyphen2

    def test_variant_type_onehot(self):
        row = self._make_row(variant_type="single nucleotide variant")
        features = compute_features(row)
        # SNV is index 0 in VARIANT_TYPES, starts at feature index 7
        assert features[7] == 1.0   # snv
        assert features[8] == 0.0   # insertion

    def test_consequence_onehot(self):
        row = self._make_row(consequence_type="missense")
        features = compute_features(row)
        # missense is index 0 in CONSEQUENCE_TYPES, starts at index 13
        assert features[13] == 1.0  # missense
        assert features[14] == 0.0  # nonsense

    def test_population_af_log_transform(self):
        row = self._make_row(gnomad_af_global=0.001)
        features = compute_features(row)
        # log10(0.001 + 1e-8) ≈ -3.0
        assert features[21] == pytest.approx(math.log10(0.001 + 1e-8), abs=0.01)

    def test_nan_for_missing_xgboost(self):
        row = self._make_row(cadd_phred=None, revel_score=None)
        features = compute_features(row, use_sentinel=False)
        assert np.isnan(features[0])  # cadd_phred → NaN
        assert np.isnan(features[1])  # revel_score → NaN

    def test_sentinel_for_missing_mlp(self):
        row = self._make_row(cadd_phred=None, revel_score=None)
        features = compute_features(row, use_sentinel=True)
        assert features[0] == pytest.approx(-1.0)  # cadd_phred → -1
        assert features[1] == pytest.approx(-1.0)  # revel_score → -1

    def test_ba1_met(self):
        # AF > 0.01 → BA1 met
        row = self._make_row(gnomad_af_global=0.05)
        features = compute_features(row)
        assert features[53] == 1.0  # BA1_met

    def test_ba1_not_met(self):
        row = self._make_row(gnomad_af_global=0.001)
        features = compute_features(row)
        assert features[53] == 0.0  # BA1_met

    def test_clingen_ordinal(self):
        row = self._make_row(clingen_validity="Definitive")
        features = compute_features(row)
        assert features[50] == 5.0  # clingen ordinal

    def test_clingen_none(self):
        row = self._make_row(clingen_validity=None)
        features = compute_features(row)
        assert features[50] == 0.0  # clingen ordinal default

    def test_inheritance_onehot(self):
        row = self._make_row(gene_moi="AR")
        features = compute_features(row)
        # AR is index 1 in INHERITANCE_MODES, starts at index 33
        assert features[33] == 0.0  # AD
        assert features[34] == 1.0  # AR

    def test_minimal_row(self):
        """Should not crash with minimal/empty row."""
        features = compute_features({})
        assert features.shape == (TABULAR_DIM,)

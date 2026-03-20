"""Tests for CT feature encoding (ct_features.py).

5 test classes:
  TestEncodeDrugFeatures: 6 tests
  TestEncodeConditionFeatures: 2 tests
  TestEncodeTrialFeatures: 5 tests
  TestBuildFeatureMatrix: 4 tests
  TestGNNTabFeatures: 2 tests
"""

import numpy as np
import pandas as pd
import pytest

from negbiodb_ct.ct_features import (
    BLINDING_KEYWORDS,
    CONDITION_DIM,
    DRUG_FP_DIM,
    DRUG_TAB_DIM,
    M2_TRIAL_DIM,
    MOLECULAR_TYPES,
    PHASE_ORDER,
    SPONSOR_TYPES,
    TOTAL_M1_DIM,
    TOTAL_M2_DIM,
    TRIAL_PHASES,
    _encode_blinding,
    _one_hot,
    build_feature_matrix,
    build_gnn_tab_features,
    build_mlp_features,
    build_xgboost_features,
    encode_condition_features,
    encode_drug_features,
    encode_trial_features,
)

# Real SMILES for testing
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def _make_drug_df(
    n: int = 3,
    *,
    with_smiles: bool = True,
    with_all_cols: bool = True,
) -> pd.DataFrame:
    """Build a minimal drug DataFrame for testing."""
    data: dict = {
        "smiles": [ASPIRIN_SMILES, CAFFEINE_SMILES, None][:n] if with_smiles else [None] * n,
        "molecular_type": ["small_molecule", "small_molecule", "monoclonal_antibody"][:n],
    }
    if with_all_cols:
        data["target_count"] = [2, 0, 5][:n]
        data["intervention_degree"] = [10, 1, 50][:n]
        data["highest_phase_reached"] = ["phase_3", "phase_1", None][:n]
        data["condition_degree"] = [5, 1, 20][:n]
    return pd.DataFrame(data)


def _make_m2_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal M2 DataFrame with trial features."""
    df = _make_drug_df(n, with_smiles=True, with_all_cols=True)
    df["trial_phase"] = ["phase_2", "phase_3", "not_applicable"][:n]
    df["blinding"] = [
        "Double (Participant, Investigator)",
        "None (Open Label)",
        None,
    ][:n]
    df["sponsor_type"] = ["industry", "academic", None][:n]
    df["randomized"] = [1, 0, None][:n]
    df["enrollment_actual"] = [500, 30, None][:n]
    df["failure_category"] = ["efficacy", "safety", "enrollment"][:n]
    df["failure_category_int"] = [0, 4, 1][:n]
    return df


# ============================================================================
# TestEncodeDrugFeatures
# ============================================================================


class TestEncodeDrugFeatures:
    """Test encode_drug_features with various inputs."""

    def test_dims_with_fp(self):
        """With FP: output should be (N, DRUG_FP_DIM + DRUG_TAB_DIM)."""
        df = _make_drug_df(3)
        result = encode_drug_features(df, include_fp=True)
        assert result.shape == (3, DRUG_FP_DIM + DRUG_TAB_DIM)
        # 1030 + 13 = 1043
        assert result.shape[1] == 1043

    def test_dims_without_fp(self):
        """Without FP: output should be (N, DRUG_TAB_DIM)."""
        df = _make_drug_df(3)
        result = encode_drug_features(df, include_fp=False)
        assert result.shape == (3, DRUG_TAB_DIM)
        assert result.shape[1] == 13

    def test_nan_for_missing_smiles(self):
        """Row 2 has None SMILES → FP columns should be NaN."""
        df = _make_drug_df(3)
        result = encode_drug_features(df, include_fp=True)
        # Row 2 (no SMILES) should have NaN in FP region
        assert np.all(np.isnan(result[2, :DRUG_FP_DIM]))
        # Row 0 (aspirin) should NOT have NaN in FP region
        assert not np.any(np.isnan(result[0, :DRUG_FP_DIM]))

    def test_fp_nonzero_for_valid_smiles(self):
        """Valid SMILES should produce nonzero FP bits."""
        df = _make_drug_df(1)
        result = encode_drug_features(df, include_fp=True)
        # Aspirin FP should have at least some bits set
        fp_region = result[0, :1024]
        assert np.sum(fp_region) > 0

    def test_molecular_type_one_hot(self):
        """Molecular type should use 10-dim one-hot (9 types + unknown)."""
        df = _make_drug_df(3, with_smiles=False)
        result = encode_drug_features(df, include_fp=False)
        # First 10 columns are mol_type one-hot
        mol_type_region = result[:, :len(MOLECULAR_TYPES) + 1]
        assert mol_type_region.shape[1] == 10
        # Each row should sum to 1 (one-hot)
        for i in range(3):
            assert mol_type_region[i].sum() == 1.0
        # Row 0: small_molecule → index 0
        assert mol_type_region[0, 0] == 1.0
        # Row 2: monoclonal_antibody → index 1
        assert mol_type_region[2, 1] == 1.0

    def test_coalesce_missing_columns(self):
        """Missing target_count/degree/phase columns → defaults, no error."""
        df = _make_drug_df(2, with_all_cols=False)
        result = encode_drug_features(df, include_fp=False)
        assert result.shape == (2, DRUG_TAB_DIM)
        # Should not contain NaN (all defaults applied)
        assert not np.any(np.isnan(result))

    def test_deterministic(self):
        """Same input → same output."""
        df = _make_drug_df(3)
        r1 = encode_drug_features(df, include_fp=True)
        r2 = encode_drug_features(df, include_fp=True)
        np.testing.assert_array_equal(r1, r2)


# ============================================================================
# TestEncodeConditionFeatures
# ============================================================================


class TestEncodeConditionFeatures:
    """Test encode_condition_features."""

    def test_dims(self):
        """Output should be (N, 1)."""
        df = pd.DataFrame({"condition_degree": [5, 1, 20]})
        result = encode_condition_features(df)
        assert result.shape == (3, CONDITION_DIM)
        assert result.shape[1] == 1

    def test_log_transform(self):
        """Values should be log1p-transformed."""
        df = pd.DataFrame({"condition_degree": [0, 1, 100]})
        result = encode_condition_features(df)
        np.testing.assert_allclose(result[0, 0], np.log1p(0), rtol=1e-5)
        np.testing.assert_allclose(result[1, 0], np.log1p(1), rtol=1e-5)
        np.testing.assert_allclose(result[2, 0], np.log1p(100), rtol=1e-5)

    def test_missing_column_default(self):
        """Missing condition_degree column → default to 1."""
        df = pd.DataFrame({"other_col": [1, 2]})
        result = encode_condition_features(df)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result[0, 0], np.log1p(1), rtol=1e-5)


# ============================================================================
# TestEncodeTrialFeatures
# ============================================================================


class TestEncodeTrialFeatures:
    """Test encode_trial_features for M2."""

    def test_dims(self):
        """Output should be (N, 22)."""
        df = _make_m2_df(3)
        result = encode_trial_features(df)
        assert result.shape == (3, M2_TRIAL_DIM)
        assert result.shape[1] == 22

    def test_trial_phase_one_hot(self):
        """Trial phase should be 9-dim one-hot."""
        df = _make_m2_df(3)
        result = encode_trial_features(df)
        # First 9 columns are trial_phase one-hot
        tp_region = result[:, :len(TRIAL_PHASES) + 1]
        assert tp_region.shape[1] == 9
        # Each row sums to 1
        for i in range(3):
            assert tp_region[i].sum() == 1.0
        # Row 0: phase_2 → index 3
        assert tp_region[0, TRIAL_PHASES.index("phase_2")] == 1.0
        # Row 1: phase_3 → index 5
        assert tp_region[1, TRIAL_PHASES.index("phase_3")] == 1.0

    def test_blinding_case_insensitive(self):
        """Blinding should match case-insensitively via substring."""
        df = _make_m2_df(3)
        result = encode_trial_features(df)
        # Blinding is at columns 9:15 (after trial_phase 9-dim)
        bl_start = len(TRIAL_PHASES) + 1  # 9
        bl_end = bl_start + len(BLINDING_KEYWORDS) + 1  # 9 + 6 = 15
        bl_region = result[:, bl_start:bl_end]
        # Row 0: "Double (Participant, Investigator)" → "double" match → index 2
        assert bl_region[0, BLINDING_KEYWORDS.index("double")] == 1.0
        # Row 1: "None (Open Label)" → "none" match → index 0
        assert bl_region[1, BLINDING_KEYWORDS.index("none")] == 1.0
        # Row 2: None → unknown bucket (index 5)
        assert bl_region[2, -1] == 1.0

    def test_sponsor_one_hot(self):
        """Sponsor type should use 5-dim one-hot (4 types + unknown)."""
        df = _make_m2_df(3)
        result = encode_trial_features(df)
        sp_start = (len(TRIAL_PHASES) + 1) + (len(BLINDING_KEYWORDS) + 1)  # 9 + 6 = 15
        sp_end = sp_start + len(SPONSOR_TYPES) + 1  # 15 + 5 = 20
        sp_region = result[:, sp_start:sp_end]
        # Row 0: industry → index 0
        assert sp_region[0, SPONSOR_TYPES.index("industry")] == 1.0
        # Row 1: academic → index 1
        assert sp_region[1, SPONSOR_TYPES.index("academic")] == 1.0
        # Row 2: None → unknown bucket
        assert sp_region[2, -1] == 1.0

    def test_enrollment_log1p(self):
        """Enrollment should be log1p-transformed."""
        df = _make_m2_df(3)
        result = encode_trial_features(df)
        # enrollment is the last column (index 21)
        np.testing.assert_allclose(result[0, -1], np.log1p(500), rtol=1e-5)
        np.testing.assert_allclose(result[1, -1], np.log1p(30), rtol=1e-5)
        # Row 2: None → log1p(0) = 0
        np.testing.assert_allclose(result[2, -1], 0.0, atol=1e-6)


# ============================================================================
# TestBuildFeatureMatrix
# ============================================================================


class TestBuildFeatureMatrix:
    """Test composite build functions."""

    def test_m1_dims(self):
        """M1 with FP: (N, 1044)."""
        df = _make_drug_df(3)
        result = build_feature_matrix(df, task="m1", include_fp=True)
        assert result.shape == (3, TOTAL_M1_DIM)
        assert result.shape[1] == 1044

    def test_m2_dims(self):
        """M2 with FP: (N, 1066)."""
        df = _make_m2_df(3)
        result = build_feature_matrix(df, task="m2", include_fp=True)
        assert result.shape == (3, TOTAL_M2_DIM)
        assert result.shape[1] == 1066

    def test_xgboost_nan_preserved(self):
        """XGBoost features should preserve NaN for missing SMILES."""
        df = _make_drug_df(3)
        result = build_xgboost_features(df, task="m1")
        # Row 2 has no SMILES → NaN in FP region
        assert np.any(np.isnan(result[2, :DRUG_FP_DIM]))

    def test_mlp_zero_padded(self):
        """MLP features should replace NaN with 0.0."""
        df = _make_drug_df(3)
        result = build_mlp_features(df, task="m1")
        assert not np.any(np.isnan(result))
        # Row 2 FP region should be all zeros (zero-padded)
        assert np.all(result[2, :DRUG_FP_DIM] == 0.0)


# ============================================================================
# TestGNNTabFeatures
# ============================================================================


class TestGNNTabFeatures:
    """Test GNN tabular feature builder (no FP)."""

    def test_m1_dims(self):
        """GNN tab M1: (N, DRUG_TAB_DIM + CONDITION_DIM) = (N, 14)."""
        df = _make_drug_df(3)
        result = build_gnn_tab_features(df, task="m1")
        expected_dim = DRUG_TAB_DIM + CONDITION_DIM
        assert result.shape == (3, expected_dim)
        assert result.shape[1] == 14

    def test_m2_dims(self):
        """GNN tab M2: (N, DRUG_TAB_DIM + CONDITION_DIM + M2_TRIAL_DIM) = (N, 36)."""
        df = _make_m2_df(3)
        result = build_gnn_tab_features(df, task="m2")
        expected_dim = DRUG_TAB_DIM + CONDITION_DIM + M2_TRIAL_DIM
        assert result.shape == (3, expected_dim)
        assert result.shape[1] == 36

    def test_no_nan_in_output(self):
        """GNN tab features should have no NaN (zero-padded)."""
        df = _make_drug_df(3)
        result = build_gnn_tab_features(df, task="m1")
        assert not np.any(np.isnan(result))


# ============================================================================
# Additional edge-case tests
# ============================================================================


class TestHelpers:
    """Test helper functions directly."""

    def test_one_hot_known_value(self):
        """Known value → correct index set."""
        result = _one_hot("phase_2", TRIAL_PHASES)
        assert len(result) == len(TRIAL_PHASES) + 1
        assert result[TRIAL_PHASES.index("phase_2")] == 1
        assert sum(result) == 1

    def test_one_hot_unknown_value(self):
        """Unknown value → last bucket."""
        result = _one_hot("bogus", TRIAL_PHASES)
        assert result[-1] == 1
        assert sum(result) == 1

    def test_one_hot_none(self):
        """None → unknown bucket."""
        result = _one_hot(None, TRIAL_PHASES)
        assert result[-1] == 1

    def test_encode_blinding_mixed_case(self):
        """Case variations should all match."""
        assert _encode_blinding("DOUBLE")[BLINDING_KEYWORDS.index("double")] == 1
        assert _encode_blinding("Double (Participant)")[BLINDING_KEYWORDS.index("double")] == 1
        assert _encode_blinding("single blind")[BLINDING_KEYWORDS.index("single")] == 1

    def test_encode_blinding_none(self):
        """None blinding → unknown bucket."""
        result = _encode_blinding(None)
        assert result[-1] == 1
        assert sum(result) == 1

    def test_dimension_constants_consistent(self):
        """Verify dimension constants add up correctly."""
        assert DRUG_FP_DIM == 1024 + 6  # FP_NBITS + N_MOL_PROPS
        assert DRUG_TAB_DIM == 10 + 1 + 1 + 1  # mol_type + tc + deg + phase
        assert CONDITION_DIM == 1
        assert M2_TRIAL_DIM == 9 + 6 + 5 + 1 + 1  # tp + bl + sp + rand + enr
        assert TOTAL_M1_DIM == DRUG_FP_DIM + DRUG_TAB_DIM + CONDITION_DIM
        assert TOTAL_M2_DIM == TOTAL_M1_DIM + M2_TRIAL_DIM

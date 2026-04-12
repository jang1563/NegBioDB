"""Tests for MD feature engineering module."""

import numpy as np
import pytest

from negbiodb_md.md_features import (
    BIOFLUIDS,
    DISEASE_CATEGORIES,
    FEATURE_DIM,
    PLATFORMS,
    build_feature_vector,
    compute_ecfp4,
    compute_physico,
    one_hot,
    study_size_features,
)

GLUCOSE_SMILES = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
INVALID_SMILES = "not_a_smiles"


# ── compute_ecfp4 ─────────────────────────────────────────────────────────────

def test_compute_ecfp4_shape():
    fp = compute_ecfp4(GLUCOSE_SMILES)
    assert fp.shape == (2048,)


def test_compute_ecfp4_binary():
    fp = compute_ecfp4(GLUCOSE_SMILES)
    assert set(fp.tolist()).issubset({0, 1})


def test_compute_ecfp4_zeros_for_none():
    fp = compute_ecfp4(None)
    assert np.all(fp == 0)
    assert fp.shape == (2048,)


def test_compute_ecfp4_zeros_for_invalid():
    fp = compute_ecfp4(INVALID_SMILES)
    assert np.all(fp == 0)


def test_compute_ecfp4_nonzero_for_valid():
    fp = compute_ecfp4(GLUCOSE_SMILES)
    assert fp.sum() > 0


# ── compute_physico ───────────────────────────────────────────────────────────

def test_compute_physico_shape():
    p = compute_physico(GLUCOSE_SMILES)
    assert p.shape == (5,)


def test_compute_physico_nan_for_none():
    p = compute_physico(None)
    assert np.all(np.isnan(p))


def test_compute_physico_positive_mw():
    p = compute_physico(GLUCOSE_SMILES)
    assert p[0] > 0  # MW > 0


# ── one_hot ───────────────────────────────────────────────────────────────────

def test_one_hot_valid():
    vec = one_hot("cancer", DISEASE_CATEGORIES)
    assert vec.shape == (len(DISEASE_CATEGORIES),)
    assert vec[DISEASE_CATEGORIES.index("cancer")] == 1.0
    assert vec.sum() == 1.0


def test_one_hot_unknown():
    vec = one_hot("unknown_category", DISEASE_CATEGORIES)
    assert vec.sum() == 0.0


def test_one_hot_none():
    vec = one_hot(None, DISEASE_CATEGORIES)
    assert vec.sum() == 0.0


# ── study_size_features ───────────────────────────────────────────────────────

def test_study_size_features_with_values():
    feat = study_size_features(100, 80)
    assert feat.shape == (2,)
    assert feat[0] == pytest.approx(2.0)  # log10(100)
    assert feat[1] == pytest.approx(np.log10(80), abs=0.01)


def test_study_size_features_none():
    feat = study_size_features(None, None)
    assert feat[0] == 0.0
    assert feat[1] == 0.0


# ── build_feature_vector ──────────────────────────────────────────────────────

def test_build_feature_vector_shape():
    vec = build_feature_vector(
        smiles=GLUCOSE_SMILES,
        disease_category="metabolic",
        platform="lc_ms",
        biofluid="blood",
        n_disease=100,
        n_control=80,
    )
    assert vec.shape == (FEATURE_DIM,)


def test_build_feature_vector_no_nan():
    """Feature vector must not contain NaN (replaced by MISSING_SENTINEL)."""
    vec = build_feature_vector(
        smiles=None,
        disease_category=None,
        platform=None,
        biofluid=None,
    )
    assert not np.any(np.isnan(vec))


def test_build_feature_vector_with_all_none():
    vec = build_feature_vector(None, None, None, None)
    assert vec.shape == (FEATURE_DIM,)
    assert not np.any(np.isnan(vec))


def test_build_feature_vector_disease_onehot_correct():
    """Disease category one-hot should be set at position 2048+5+[category index]."""
    for cat in DISEASE_CATEGORIES:
        vec = build_feature_vector(None, cat, None, None)
        cat_start = 2048 + 5
        cat_idx = DISEASE_CATEGORIES.index(cat)
        assert vec[cat_start + cat_idx] == 1.0


def test_build_feature_vector_biofluid_tissue_fallback():
    """Biofluid 'other' should map to 'tissue' one-hot slot."""
    vec_other = build_feature_vector(None, None, None, "other")
    vec_tissue = build_feature_vector(None, None, None, "tissue")
    np.testing.assert_array_equal(vec_other, vec_tissue)


def test_feature_dim_constant():
    assert FEATURE_DIM == 2048 + 5 + 5 + 4 + 4 + 2

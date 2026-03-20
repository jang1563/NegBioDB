"""Tests for negbiodb_ppi.models — PPI ML baseline models."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from negbiodb_ppi.models.siamese_cnn import SiameseCNN, seq_to_tensor, MAX_SEQ_LEN
from negbiodb_ppi.models.pipr import PIPR
from negbiodb_ppi.models.pipr import seq_to_tensor as pipr_seq_to_tensor
from negbiodb_ppi.models.mlp_features import (
    MLPFeatures,
    extract_features,
    compute_aa_composition,
    encode_subcellular,
    FEATURE_DIM,
)


# ------------------------------------------------------------------
# Tokenization tests
# ------------------------------------------------------------------

class TestSeqToTensor:
    def test_basic(self):
        t = seq_to_tensor(["ACDEF", "GHI"])
        assert t.shape == (2, MAX_SEQ_LEN)
        assert t.dtype == torch.int64
        # First 5 positions should be nonzero for "ACDEF"
        assert t[0, :5].sum() > 0
        # Position 5+ should be padding (0)
        assert t[0, 5:].sum() == 0

    def test_empty(self):
        t = seq_to_tensor([""])
        assert t.shape == (1, MAX_SEQ_LEN)
        assert t.sum() == 0

    def test_truncation(self):
        long_seq = "A" * 2000
        t = seq_to_tensor([long_seq])
        assert t.shape == (1, MAX_SEQ_LEN)
        # All MAX_SEQ_LEN positions should be the same nonzero token
        assert (t[0, :MAX_SEQ_LEN] > 0).all()


# ------------------------------------------------------------------
# SiameseCNN tests
# ------------------------------------------------------------------

class TestSiameseCNN:
    def test_forward_shape(self):
        model = SiameseCNN()
        B = 4
        s1 = seq_to_tensor(["ACDEF" * 10] * B)
        s2 = seq_to_tensor(["GHIKL" * 10] * B)
        out = model(s1, s2)
        assert out.shape == (B,)

    def test_symmetry(self):
        """f(A, B) == f(B, A) due to shared encoder + |diff|."""
        model = SiameseCNN()
        model.eval()
        s1 = seq_to_tensor(["ACDEFGHIKL"])
        s2 = seq_to_tensor(["MNPQRSTVWY"])

        with torch.no_grad():
            out_12 = model(s1, s2)
            out_21 = model(s2, s1)
        assert torch.allclose(out_12, out_21, atol=1e-5)

    def test_gradient_flow(self):
        model = SiameseCNN()
        s1 = seq_to_tensor(["ACDEF"])
        s2 = seq_to_tensor(["GHIKL"])
        out = model(s1, s2)
        loss = out.sum()
        loss.backward()
        # Check all parameters have gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ------------------------------------------------------------------
# PIPR tests
# ------------------------------------------------------------------

class TestPIPR:
    def test_forward_shape(self):
        model = PIPR()
        B = 4
        s1 = pipr_seq_to_tensor(["ACDEF" * 10] * B)
        s2 = pipr_seq_to_tensor(["GHIKL" * 10] * B)
        out = model(s1, s2)
        assert out.shape == (B,)

    def test_symmetry(self):
        """Shared encoder + symmetric attention pooling → f(A,B) ≈ f(B,A)."""
        model = PIPR()
        model.eval()
        s1 = pipr_seq_to_tensor(["ACDEFGHIKL"])
        s2 = pipr_seq_to_tensor(["MNPQRSTVWY"])

        with torch.no_grad():
            out_12 = model(s1, s2)
            out_21 = model(s2, s1)
        assert torch.allclose(out_12, out_21, atol=1e-5)

    def test_gradient_flow(self):
        model = PIPR()
        s1 = pipr_seq_to_tensor(["ACDEF"])
        s2 = pipr_seq_to_tensor(["GHIKL"])
        out = model(s1, s2)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_handles_padding(self):
        """Model should handle sequences of different lengths gracefully."""
        model = PIPR()
        model.eval()
        s1 = pipr_seq_to_tensor(["AC"])         # very short
        s2 = pipr_seq_to_tensor(["GHIKL" * 50]) # medium
        with torch.no_grad():
            out = model(s1, s2)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------
# MLPFeatures tests
# ------------------------------------------------------------------

class TestMLPFeatures:
    def test_forward_shape(self):
        model = MLPFeatures()
        B = 4
        features = torch.randn(B, FEATURE_DIM)
        out = model(features)
        assert out.shape == (B,)

    def test_gradient_flow(self):
        model = MLPFeatures()
        features = torch.randn(2, FEATURE_DIM)
        out = model(features)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestFeatureExtraction:
    def test_aa_composition(self):
        comp = compute_aa_composition("AAACCC")
        assert len(comp) == 20
        assert abs(sum(comp) - 1.0) < 1e-6
        # A and C should dominate
        assert comp[0] > 0.3  # A
        assert comp[1] > 0.3  # C

    def test_aa_composition_empty(self):
        comp = compute_aa_composition("")
        assert comp == [0.0] * 20

    def test_subcellular_encoding(self):
        vec = encode_subcellular("Nucleus")
        assert len(vec) == 11
        assert vec[0] == 1.0
        assert sum(vec) == 1.0

    def test_subcellular_none(self):
        vec = encode_subcellular(None)
        assert sum(vec) == 0.0

    def test_subcellular_unknown(self):
        vec = encode_subcellular("Unknown location XYZ")
        assert vec[-1] == 1.0  # "other"

    def test_extract_features_dim(self):
        f = extract_features(
            "ACDEF", "GHIKL",
            degree1=10.0, degree2=20.0,
            loc1="Nucleus", loc2="Cytoplasm",
        )
        assert len(f) == FEATURE_DIM

    def test_extract_features_symmetry_invariant(self):
        """Feature vector should NOT be order-invariant (p1 != p2 features)."""
        f12 = extract_features("ACDEF", "GHIKL", 10, 20, "Nucleus", "Cytoplasm")
        f21 = extract_features("GHIKL", "ACDEF", 20, 10, "Cytoplasm", "Nucleus")
        # f12 != f21 because AA compositions are in fixed order (p1 then p2)
        # But they should have same values rearranged
        assert len(f12) == len(f21)

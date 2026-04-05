"""Tests for DC synergy computation module.

Tests the pure-Python Bliss fallback (R/rpy2 not expected in test environment),
the auto-dispatch logic, and edge cases.
"""

import pytest

from negbiodb_dc.synergy_compute import (
    SynergyScores,
    compute_synergy,
    compute_synergy_bliss_python,
    is_r_available,
)


# ── SynergyScores dataclass ──────────────────────────────────────────


class TestSynergyScores:
    def test_defaults_none(self):
        s = SynergyScores()
        assert s.zip_score is None
        assert s.bliss_score is None
        assert s.loewe_score is None
        assert s.hsa_score is None

    def test_custom_values(self):
        s = SynergyScores(zip_score=5.0, bliss_score=-3.0)
        assert s.zip_score == 5.0
        assert s.bliss_score == -3.0
        assert s.loewe_score is None

    def test_all_values(self):
        s = SynergyScores(
            zip_score=10.0, bliss_score=8.0,
            loewe_score=-2.0, hsa_score=1.0,
        )
        assert s.zip_score == 10.0
        assert s.hsa_score == 1.0


# ── is_r_available ───────────────────────────────────────────────────


class TestIsRAvailable:
    def test_returns_bool(self):
        result = is_r_available()
        assert isinstance(result, bool)


# ── Bliss independence (Python fallback) ─────────────────────────────


class TestBlissPython:
    def test_additive_response(self):
        """When combination equals Bliss prediction → score ~0 (additive)."""
        conc_row = [0, 1, 10]
        conc_col = [0, 1, 10]
        # Mono: drug A alone = [0, 30, 60], drug B alone = [0, 20, 50]
        # Bliss prediction for (1,1): 30+20 - 30*20/100 = 44
        # If observed matches prediction → synergy ≈ 0
        # Build a response matrix where combinations match Bliss prediction
        mono_a = [0, 30, 60]  # row drug alone
        mono_b = [0, 20, 50]  # col drug alone
        response = []
        for i in range(3):
            row = []
            for j in range(3):
                ea = mono_a[i] / 100.0
                eb = mono_b[j] / 100.0
                predicted = (ea + eb - ea * eb) * 100.0
                row.append(predicted)
            response.append(row)

        scores = compute_synergy_bliss_python(conc_row, conc_col, response)
        assert scores.bliss_score is not None
        assert abs(scores.bliss_score) < 1.0  # Should be near zero

    def test_synergistic_response(self):
        """When observed > predicted → positive Bliss score (synergistic)."""
        conc_row = [0, 1, 10]
        conc_col = [0, 1, 10]
        # Mono: A = [0, 20, 40], B = [0, 15, 35]
        # Bliss prediction for (1,1): 20+15 - 20*15/100 = 32
        # If observed = 50 (much higher) → synergistic
        response = [
            [0, 15, 35],     # Drug A = 0 (only drug B)
            [20, 50, 70],    # Drug A = 1: observed >> predicted
            [40, 65, 85],    # Drug A = 10: observed >> predicted
        ]

        scores = compute_synergy_bliss_python(conc_row, conc_col, response)
        assert scores.bliss_score is not None
        assert scores.bliss_score > 0  # Synergistic

    def test_antagonistic_response(self):
        """When observed < predicted → negative Bliss score (antagonistic)."""
        conc_row = [0, 1, 10]
        conc_col = [0, 1, 10]
        # Mono: A = [0, 40, 70], B = [0, 30, 60]
        # Bliss prediction for (1,1): 40+30 - 40*30/100 = 58
        # If observed = 20 (much lower) → antagonistic
        response = [
            [0, 30, 60],     # Drug A = 0
            [40, 20, 30],    # Observed << predicted
            [70, 40, 50],    # Observed << predicted
        ]

        scores = compute_synergy_bliss_python(conc_row, conc_col, response)
        assert scores.bliss_score is not None
        assert scores.bliss_score < 0  # Antagonistic

    def test_only_bliss_returned(self):
        """Python fallback only produces Bliss score, not ZIP/Loewe/HSA."""
        response = [[0, 10, 20], [15, 30, 40], [25, 45, 60]]
        scores = compute_synergy_bliss_python([0, 1, 10], [0, 1, 10], response)
        assert scores.bliss_score is not None
        assert scores.zip_score is None
        assert scores.loewe_score is None
        assert scores.hsa_score is None

    def test_single_concentration_pair(self):
        """Minimal 2x2 matrix still computes a score."""
        response = [[0, 10], [20, 40]]
        scores = compute_synergy_bliss_python([0, 1], [0, 1], response)
        assert scores.bliss_score is not None

    def test_zero_response_matrix(self):
        """All zeros → Bliss prediction = 0, observed = 0, synergy = 0."""
        response = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        scores = compute_synergy_bliss_python([0, 1, 10], [0, 1, 10], response)
        assert scores.bliss_score is not None
        assert abs(scores.bliss_score) < 0.01

    def test_high_inhibition(self):
        """Matrix with 100% inhibition values."""
        response = [[0, 100, 100], [100, 100, 100], [100, 100, 100]]
        scores = compute_synergy_bliss_python([0, 1, 10], [0, 1, 10], response)
        # Bliss prediction for (1,1): 100+100 - 100*100/100 = 100
        # Observed = 100, so synergy = 0
        assert scores.bliss_score is not None


# ── compute_synergy dispatcher ───────────────────────────────────────


class TestComputeSynergy:
    def test_force_python(self):
        """Forcing use_r=False uses Python fallback."""
        response = [[0, 10, 20], [15, 30, 40], [25, 45, 60]]
        scores = compute_synergy(
            "DrugA", "DrugB", [0, 1, 10], [0, 1, 10],
            response, use_r=False,
        )
        assert scores.bliss_score is not None
        assert scores.zip_score is None  # Python only computes Bliss

    def test_auto_detect(self):
        """Auto-detection returns valid SynergyScores regardless of R availability."""
        response = [[0, 10, 20], [15, 30, 40], [25, 45, 60]]
        scores = compute_synergy(
            "DrugA", "DrugB", [0, 1, 10], [0, 1, 10],
            response, use_r=None,
        )
        assert isinstance(scores, SynergyScores)
        # At minimum, Bliss should be computed (either path)
        assert scores.bliss_score is not None

    def test_force_r_without_rpy2_raises(self):
        """Forcing use_r=True when R isn't available raises RuntimeError."""
        if is_r_available():
            pytest.skip("R is available; cannot test fallback")
        response = [[0, 10], [20, 40]]
        with pytest.raises(RuntimeError, match="rpy2"):
            compute_synergy(
                "DrugA", "DrugB", [0, 1], [0, 1],
                response, use_r=True,
            )

"""Regression tests for scripts_dc/build_dc_l1_dataset.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "scripts_dc" / "build_dc_l1_dataset.py"
    spec = importlib.util.spec_from_file_location("dc_build_l1_dataset_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestClassifyByZip:
    def test_nan_returns_none(self):
        module = _load_module()
        assert module.classify_by_zip(np.nan) is None

    def test_missing_returns_none(self):
        module = _load_module()
        assert module.classify_by_zip(None) is None

    def test_bliss_fallback_values_still_classify(self):
        module = _load_module()
        assert module.classify_by_zip(6.0) == "B"
        assert module.classify_by_zip(-11.0) == "D"

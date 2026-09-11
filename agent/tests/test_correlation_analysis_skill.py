"""Regression tests for the correlation-analysis skill examples."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SKILL_MD = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "skills"
    / "correlation-analysis"
    / "SKILL.md"
)


def _load_engle_granger():
    text = SKILL_MD.read_text(encoding="utf-8")
    section = text.split("### Engle-Granger Two-Step Method", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"pd": pd}
    exec(code, namespace)
    return namespace["engle_granger_coint"]


@pytest.mark.parametrize("name", [None, "price_x"])
def test_engle_granger_handles_series_name(name):
    rng = np.random.default_rng(42)
    x = pd.Series(np.cumsum(rng.normal(size=250)), name=name)
    y = 1.7 * x + pd.Series(rng.normal(scale=0.2, size=250))

    result = _load_engle_granger()(y, x)

    assert result["hedge_ratio"] == pytest.approx(1.7, abs=0.02)

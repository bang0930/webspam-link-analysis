"""figures.py 강민수 그림 (최종 비교표, PR 곡선) 테스트."""
import numpy as np
import pandas as pd

from src import figures as F
from src.evaluate import EvalTarget


def _target(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return EvalTarget(np.arange(n), (rng.random(n) < 0.08).astype(int),
                      np.array([f"d{i % 120}" for i in range(n)]))


def test_fig_final_table(tmp_path):
    table = pd.DataFrame({
        "key": ["lsh", "ppr", "random", "tr_set1_trust"],
        "family": ["lsh", "ppr", "baseline", "trustrank"],
        "pr_auc": [0.10, 0.08, 0.06, 0.12],
        "pr_auc_ci95": ["[0.080, 0.130]", "[0.060, 0.100]", "[0.050, 0.070]", "[0.090, 0.150]"],
    })
    p = F.fig_final_table(table, 0.063, tmp_path)
    assert p.exists() and p.stat().st_size > 10_000


def test_fig_pr_curves(tmp_path):
    t = _target()
    rng = np.random.default_rng(1)
    scores = {k: rng.random(t.n) + t.y * w for k, w in
              (("tr_set1_trust", 0.6), ("ppr", 0.2), ("lsh", 0.4), ("pagerank", 0.0), ("in_degree", 0.1))}
    p = F.fig_pr_curves(scores, t, families={"tr_set1_trust": "trustrank", "in_degree": "baseline"},
                        out_dir=tmp_path)
    assert p.exists() and p.stat().st_size > 10_000


def test_parse_ci():
    assert F._parse_ci("[0.081, 0.142]") == (0.081, 0.142)
    assert F._parse_ci((0.1, 0.2)) == (0.1, 0.2)

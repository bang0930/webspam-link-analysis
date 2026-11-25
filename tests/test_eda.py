"""eda.py, figures.py 테스트 (합성 데이터). 실행: python -m pytest tests -q"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import eda as D
from src import figures as F


def _graph(n, edges):
    src, dst = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (src, dst)), shape=(n, n))


def _labels(rows):
    return pd.DataFrame(rows, columns=["hostid", "label", "spamicity", "assessments"])


def test_parse_and_agreement():
    assert D.parse_assessments("j1:N,j2:S,j3:U") == ["N", "S", "U"]
    assert D.parse_assessments("") == [] and D.parse_assessments(float("nan")) == []
    lab = _labels([
        (0, "nonspam", 0.0, "j1:N,j2:N"),          # 쌍 1개, 일치
        (1, "spam", 1.0, "j1:S,j2:S,j3:N"),        # 쌍 3개 중 1개 일치
        (2, "undecided", np.nan, "j1:U,j2:N"),     # 유효 판정 1개 → 제외
    ])
    r = D.assessor_agreement(lab)
    assert r["hosts_with_2plus_valid"] == 2
    assert r["unanimous_rate"] == 0.5
    # host 0: 쌍 1개 중 1개 일치 = 1.0, host 1: 쌍 3개 중 1개 일치 = 1/3 → host 평균 2/3
    assert r["pairwise_agreement"] == pytest.approx((1 + 1 / 3) / 2, abs=1e-3)
    # 판정 N 3개, S 2개 → Pe = 0.6² + 0.4² = 0.52
    assert r["chance_agreement"] == pytest.approx(0.52, abs=1e-3)
    po = (1 + 1 / 3) / 2
    assert r["kappa"] == pytest.approx((po - 0.52) / (1 - 0.52), abs=1e-3)


def test_link_assumption():
    # 라벨: 0,1 nonspam / 2,3 spam / 4 라벨 없음
    A = _graph(5, [(0, 1), (0, 2), (1, 0), (2, 3), (3, 2), (2, 4)])
    train = _labels([(0, "nonspam", 0, ""), (1, "nonspam", 0, ""), (2, "spam", 1, ""), (3, "spam", 1, "")])
    t = D.link_assumption(A, train).set_index("source")
    assert t.loc["nonspam", "links_to_labeled"] == 3 and t.loc["nonspam", "to_spam"] == 1
    assert t.loc["spam", "links_to_labeled"] == 2 and t.loc["spam", "to_spam"] == 2   # 2→4는 라벨 없어서 제외
    assert t.loc["spam", "base_spam_rate"] == 0.5 and t.loc["spam", "lift"] == 2.0


def test_stats_and_degree():
    A = _graph(5, [(0, 1), (1, 2), (2, 0)])
    train = _labels([(0, "nonspam", 0, ""), (1, "spam", 1, ""), (3, "undecided", np.nan, "")])
    test = _labels([(2, "spam", 1, ""), (4, "nonspam", 0, "")])
    t1 = D.data_stats(A, ["a"] * 5, train, test, np.array([2, 4]), np.zeros(5, bool)).set_index("항목")["값"]
    assert t1["host 수"] == 5 and t1["dangling host (나가는 링크 0)"] == 2
    assert t1["SET1 undecided"] == 1 and t1["채점 대상 eval_set 스팸"] == 1
    deg = D.degree_by_label(A, train)
    assert set(deg["label"]) == {"nonspam", "spam"} and (deg["n_hosts"] == 1).all()


def test_figures_smoke(tmp_path):
    A = _graph(6, [(0, 1), (0, 2), (1, 2), (3, 2), (4, 0), (5, 0)])
    train = _labels([(0, "nonspam", 0, ""), (1, "nonspam", 0, ""), (2, "spam", 1, ""), (5, "spam", 1, "")])
    p = F.fig_degree_distribution(A, train, out_dir=tmp_path)
    assert p.exists() and p.stat().st_size > 0
    curve = pd.DataFrame({"jaccard_lo": [0.0, 0.5], "jaccard_hi": [0.5, 1.0], "candidate_rate": [0.1, 0.9],
                          "bands": [16, 16], "rows": [8, 8]})
    p2 = F.fig_lsh_scurve(curve, out_dir=tmp_path)
    assert p2.exists() and p2.stat().st_size > 0

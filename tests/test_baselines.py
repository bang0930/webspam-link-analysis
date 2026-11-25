"""baselines.py 테스트 (합성 데이터). 실행: python -m pytest tests -q"""

import numpy as np
import pytest
import scipy.sparse as sp

from src import baselines as B


def _graph():
    # 0→2, 1→2, 3→2, 2→4, 4→0  (5는 고립)
    edges = [(0, 2), (1, 2), (3, 2), (2, 4), (4, 0)]
    src, dst = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (src, dst)), shape=(6, 6))


def test_random_and_degree():
    A = _graph()
    r = B.random_scores(6, seed=42)
    assert r.shape == (6,) and np.array_equal(r, B.random_scores(6, seed=42))
    assert not np.array_equal(r, B.random_scores(6, seed=43))
    assert B.in_degree_scores(A).tolist() == [1, 0, 3, 0, 1, 0]
    assert B.in_degree_scores(A, sign=-1).tolist() == [-1, 0, -3, 0, -1, 0]
    assert B.out_degree_scores(A).tolist() == [1, 1, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        B.in_degree_scores(A, sign=2)


def test_one_hop_in_and_out():
    A = _graph()
    spam, nonspam = [0, 1], [3, 4]            # p = 0.5
    p, a = 0.5, 3
    s_in, cov_in = B.one_hop_scores(A, spam, nonspam, "in", alpha=a)
    # host 2에게 링크한 0(spam), 1(spam), 3(nonspam) → n=3, s=2
    assert s_in[2] == pytest.approx((2 + a * p) / (3 + a))
    # host 0에게 링크한 4(nonspam) → n=1, s=0
    assert s_in[0] == pytest.approx((0 + a * p) / (1 + a))
    # host 1, 3, 5는 링크받지 않음 → p, 커버 아님
    assert s_in[[1, 3, 5]].tolist() == [p, p, p]
    assert cov_in.tolist() == [True, False, True, False, False, False]

    s_out, cov_out = B.one_hop_scores(A, spam, nonspam, "out", alpha=a)
    # host 4가 링크한 0(spam) → n=1, s=1
    assert s_out[4] == pytest.approx((1 + a * p) / (1 + a))
    # host 0, 1, 3이 링크한 2(라벨 없음) → p
    assert s_out[[0, 1, 3]].tolist() == [p, p, p]
    assert cov_out.tolist() == [False, False, True, False, True, False]


def test_errors_and_all():
    A = _graph()
    with pytest.raises(ValueError):
        B.one_hop_scores(A, [0], [0])                # 같은 host가 양쪽
    with pytest.raises(ValueError):
        B.one_hop_scores(A, [], [])                  # 라벨 없음
    with pytest.raises(ValueError):
        B.one_hop_scores(A, [0], [3], direction="both")
    scores = B.all_baselines(A, [0, 1], [3, 4], in_sign=-1)
    assert set(scores) == {"random", "in_degree", "out_degree", "one_hop_in", "one_hop_out"}
    assert all(v.shape == (6,) for v in scores.values())
    assert scores["in_degree"][2] == -3

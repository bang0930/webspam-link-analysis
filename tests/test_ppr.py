"""ppr.py 테스트: 손계산, 선형방정식의 닫힌 해, 뒤집은 그래프 예제."""
import numpy as np
import pytest
import scipy.sparse as sp

from src import ppr as P

B = 0.85


def _graph(edges, n):
    r, c = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))


def _closed_form(A, v, d, beta=B):
    """r = β Pᵀ r + (1-β) v 의 정확한 해. P = 행 정규화, dangling 행은 d로 채움."""
    A = A.toarray()
    n = len(A)
    Pm = np.zeros((n, n))
    for i in range(n):
        Pm[i] = A[i] / A[i].sum() if A[i].sum() else d
    r = np.linalg.solve(np.eye(n) - beta * Pm.T, (1 - beta) * v)
    return r / r.sum()


# 6개 host: 0,1 = 스팸 seed. 0→2, 1→2, 2→3, 3→2, 4→5, 5 dangling, 1 → 4
EDGES = [(0, 2), (1, 2), (2, 3), (3, 2), (1, 4), (4, 5)]
A6 = _graph(EDGES, 6)
SEEDS = [0, 1]


def test_hand_computed_two_nodes():
    # seed 0 → 1, 1은 dangling (dangling → seed)
    # r0 = 0.15 + 0.85·r1,  r1 = 0.85·r0  →  r0 = 0.15 / (1 - 0.85²)
    A = _graph([(0, 1)], 2)
    r = P.ppr_score(A, [0], B, "forward", "raw").scores
    r0 = 0.15 / (1 - B ** 2)
    assert r == pytest.approx([r0, B * r0], abs=1e-7)


def test_forward_raw_matches_closed_form():
    v = P.seed_vector(6, SEEDS)
    res = P.ppr_score(A6, SEEDS, B, "forward", "raw")
    assert res.scores == pytest.approx(_closed_form(A6, v, v), abs=1e-7)
    assert res.scores.sum() == pytest.approx(1.0)
    assert res.info["n_spreading_seeds"] == 2


def test_reverse_uses_flipped_graph():
    v = P.seed_vector(6, SEEDS)
    res = P.ppr_score(A6, SEEDS, B, "reverse", "raw")
    assert res.scores == pytest.approx(_closed_form(A6.T.tocsr(), v, v), abs=1e-7)
    # 뒤집은 그래프에서 seed 0·1은 들어오는 링크가 없었으므로 나가는 링크 0 → 퍼뜨리지 못함
    assert res.info["n_spreading_seeds"] == 0
    # dangling 집합도 뒤집은 그래프 기준으로 다시 계산됨 (원래 들어오는 링크 0인 host: 0, 1)
    assert res.info["n_dangling"] == 2


def test_ratio_uses_uniform_on_same_graph():
    n = 6
    v, u = P.seed_vector(n, SEEDS), np.full(n, 1 / n)
    for direction, G in (("forward", A6), ("reverse", A6.T.tocsr())):
        res = P.ppr_score(A6, SEEDS, B, direction, "ratio")
        want = _closed_form(G, v, u) / _closed_form(G, u, u)
        assert res.scores == pytest.approx(want, rel=1e-6)


def test_suspicion_flows_from_seeds():
    r = P.ppr_score(A6, SEEDS, B, "forward", "raw").scores
    # 스팸이 직접 링크한 2가 스팸과 무관한 경로의 5보다 의심 점수가 높음
    assert r[2] > r[5]


def test_dangling_seed_cannot_spread():
    # seed 5는 나가는 링크가 없음 → 링크 방향에선 의심이 다른 host로 퍼지지 않음
    res = P.ppr_score(A6, [5], B, "forward", "raw")
    assert res.info["n_spreading_seeds"] == 0
    assert res.scores[5] == pytest.approx(1.0) and res.scores[:5] == pytest.approx(0.0, abs=1e-12)
    # 역방향에선 5에 링크한 4로 퍼짐
    assert P.ppr_score(A6, [5], B, "reverse", "raw").scores[4] > 0


def test_variants_and_names():
    vs = P.all_variants(A6, SEEDS)
    assert set(vs) == {"ppr_forward_raw", "ppr_forward_ratio", "ppr_reverse_raw", "ppr_reverse_ratio"}
    assert P.setting_name("forward", "raw") == "ppr_dir-forward_score-raw_d0.85_seeds-set1"


def test_bad_inputs():
    with pytest.raises(ValueError):
        P.ppr_score(A6, [], B)
    with pytest.raises(ValueError):
        P.ppr_score(A6, [99], B)
    with pytest.raises(ValueError):
        P.ppr_score(A6, SEEDS, B, direction="sideways")

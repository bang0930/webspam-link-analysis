"""engine.py, pagerank.py 테스트. 실행: python -m pytest tests -q

담당: 박병언 (팀 작성 코드)

- 강의 11강 예제: p16, p18 흐름 방정식, p33 spider trap, p46 Topic-Specific PageRank, p28, p29 dead end
- 일반 검증: 닫힌 해, networkx, 뒤집은 그래프, 시작값, 선형성, 잘못된 입력 (확정안 7절)
노드 번호는 0부터. 강의 예제의 y = 0, a = 1, m = 2.
"""

import warnings

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from src import config
from src import pagerank as PR
from src.engine import power_iteration
from src.load_graph import dangling_mask


def _graph(edges, n):
    r, c = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))


def _uniform(n):
    return np.full(n, 1.0 / n)


def _iterate(A, v, beta, k, init="uniform"):
    """k회만 반복한 값 (수렴 경고는 무시)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r, it = power_iteration(A, v, damping=beta, max_iter=k, init=init)
    assert it == k
    return r


def _closed_form(A, v, beta, dangling):
    """(I − βPᵀ) r = (1 − β) v 의 해. P = 행 정규화, dangling 행은 규칙의 분배 벡터 d로 채움."""
    A = np.asarray(A.toarray(), dtype=float)
    n = len(A)
    v = np.asarray(v, float) / np.sum(v)
    d = v if dangling == "teleport" else np.full(n, 1.0 / n)
    P = np.zeros((n, n))
    for i in range(n):
        s = A[i].sum()
        P[i] = A[i] / s if s else d
    return np.linalg.solve(np.eye(n) - beta * P.T, (1 - beta) * v)


def _random_graph(n=40, p=0.08, seed=config.SEED):
    """dangling host와 자기 자신 링크가 섞인 무작위 그래프."""
    rng = np.random.default_rng(seed)
    M = (rng.random((n, n)) < p).astype(float)
    M[rng.choice(n, size=n // 4, replace=False)] = 0.0   # 일부 행을 비워 dangling 만들기
    M[0, 0] = 1.0                                         # 자기 자신 링크
    A = sp.csr_matrix(M)
    assert dangling_mask(A).sum() >= n // 4
    return A


# 11강 예제 그래프
FLOW = _graph([(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)], 3)       # p16, p18
TRAP = _graph([(0, 0), (0, 1), (1, 0), (1, 2), (2, 2)], 3)       # p33 (m -> m spider trap)
DEAD = _graph([(0, 0), (0, 1), (1, 0), (1, 2)], 3)               # p28, p29 (m은 dead end)
TOPIC = _graph([(0, 1), (0, 2), (1, 0), (2, 3), (3, 2)], 4)      # p46 (노드 1 ~ 4를 0 ~ 3으로)


# ---------------------------------------------------------------------------
# 강의 예제 1: p16, p18 흐름 방정식 (β = 1, teleport 없음)
# ---------------------------------------------------------------------------
def test_lecture_p18_flow_iterations():
    v = _uniform(3)
    assert _iterate(FLOW, v, 1.0, 1) == pytest.approx([1 / 3, 1 / 2, 1 / 6], abs=1e-12)
    assert _iterate(FLOW, v, 1.0, 2) == pytest.approx([5 / 12, 1 / 3, 1 / 4], abs=1e-12)


def test_lecture_p16_flow_converged():
    r, it = power_iteration(FLOW, _uniform(3), damping=1.0, init="uniform")
    assert r == pytest.approx([2 / 5, 2 / 5, 1 / 5], abs=1e-8)
    assert 0 < it < config.MAX_ITER


# ---------------------------------------------------------------------------
# 강의 예제 2: p33 spider trap (β = 0.8). 슬라이드 반복 표는 오타가 있어 수렴값만 본다.
# ---------------------------------------------------------------------------
def test_lecture_p33_spider_trap():
    r, _ = power_iteration(TRAP, _uniform(3), damping=0.8)
    assert r == pytest.approx([7 / 33, 5 / 33, 21 / 33], abs=1e-8)


# ---------------------------------------------------------------------------
# 강의 예제 3: p46 Topic-Specific PageRank
# ---------------------------------------------------------------------------
def _teleport_set(S, n=4):
    v = np.zeros(n)
    v[list(S)] = 1.0
    return v


def test_lecture_p46_iterations_s0():
    v = _teleport_set({0})
    assert _iterate(TOPIC, v, 0.8, 1) == pytest.approx([0.4, 0.1, 0.3, 0.2], abs=1e-12)
    assert _iterate(TOPIC, v, 0.8, 2) == pytest.approx([0.28, 0.16, 0.32, 0.24], abs=1e-12)


def test_lecture_p46_converged_s0():
    r, _ = power_iteration(TOPIC, _teleport_set({0}), damping=0.8)
    assert r == pytest.approx([0.294, 0.118, 0.327, 0.261], abs=5e-4)


# 슬라이드 오른쪽 표: 소수 둘째 자리에서 버림한 값. "표기 값 이상, 표기 값 + 0.01 미만"으로 검사.
P46_TABLE = [
    ({0, 1, 2, 3}, 0.8, [0.13, 0.10, 0.39, 0.36]),
    ({0, 1, 2}, 0.8, [0.17, 0.13, 0.38, 0.30]),
    ({0, 1}, 0.8, [0.26, 0.20, 0.29, 0.23]),
    ({0}, 0.9, [0.16, 0.07, 0.39, 0.35]),
    ({0}, 0.7, [0.39, 0.13, 0.27, 0.19]),
]


@pytest.mark.parametrize("S, beta, shown", P46_TABLE, ids=[f"S{sorted(s)}_b{b}" for s, b, _ in P46_TABLE])
def test_lecture_p46_table(S, beta, shown):
    r, _ = power_iteration(TOPIC, _teleport_set(S), damping=beta)
    shown = np.array(shown)
    assert (r >= shown - 1e-12).all() and (r < shown + 0.01).all(), f"실제 값 {np.round(r, 4)}"


# ---------------------------------------------------------------------------
# 강의 예제 4: p28, p29 dead end (β = 0.8). 슬라이드에 수치가 없어 닫힌 해와 비교.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["teleport", "uniform"])
def test_lecture_p28_dead_end(rule):
    v = _uniform(3)
    r, _ = power_iteration(DEAD, v, damping=0.8, dangling=rule)
    assert r == pytest.approx(_closed_form(DEAD, v, 0.8, rule), abs=1e-8)
    assert r.sum() == pytest.approx(1.0, abs=1e-12)   # p28처럼 질량이 새지 않음
    assert r[2] > 0


# ---------------------------------------------------------------------------
# 일반 검증: 닫힌 해, networkx
# ---------------------------------------------------------------------------
def _teleports(n, seed=config.SEED):
    rng = np.random.default_rng(seed)
    seeds = np.zeros(n)
    seeds[rng.choice(n, size=5, replace=False)] = 1.0
    return {"uniform": _uniform(n), "seeds": seeds, "random": rng.random(n)}


@pytest.mark.parametrize("rule", ["teleport", "uniform"])
@pytest.mark.parametrize("beta", [0.7, 0.85, 0.95])
def test_closed_form_random_graph(rule, beta):
    A = _random_graph()
    for name, v in _teleports(A.shape[0]).items():
        r, _ = power_iteration(A, v, damping=beta, dangling=rule)
        assert r == pytest.approx(_closed_form(A, v, beta, rule), abs=1e-8), name


@pytest.mark.parametrize("rule", ["teleport", "uniform"])
def test_matches_networkx(rule):
    A = _random_graph()
    n = A.shape[0]
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    for name, v in _teleports(n).items():
        r, _ = power_iteration(A, v, damping=0.85, dangling=rule)
        pers = {i: float(v[i]) for i in range(n)}
        # networkx는 "L1 오차 < N x tol" 기준이라 기본 tol(1e-6)로는 자기 오차가 1e-6을 넘는다. 충분히 수렴시킨다.
        if rule == "teleport":
            want = nx.pagerank(G, alpha=0.85, personalization=pers, tol=1e-12)   # dangling은 personalization을 따름
        else:
            want = nx.pagerank(G, alpha=0.85, personalization=pers, tol=1e-12,
                               dangling={i: 1.0 for i in range(n)})
        assert r == pytest.approx([want[i] for i in range(n)], abs=1e-6), name


# ---------------------------------------------------------------------------
# 일반 검증: 뒤집은 그래프, 출력 형태
# ---------------------------------------------------------------------------
def test_reversed_graph_recomputes_dangling():
    # 0 -> 1 -> 2. 뒤집으면 2 -> 1 -> 0. dangling이 2에서 0으로 바뀐다.
    A = _graph([(0, 1), (1, 2)], 3)
    AT = A.T.tocsr()
    assert dangling_mask(A).tolist() == [False, False, True]
    assert dangling_mask(AT).tolist() == [True, False, False]
    v = np.array([1.0, 0.0, 0.0])
    fwd, _ = power_iteration(A, v, damping=0.85)
    rev, _ = power_iteration(AT, v, damping=0.85)
    # 원래 방향: seed 0이 1, 2로 퍼뜨림. 뒤집으면 seed 0이 dangling이라 아무 데도 못 감.
    assert (fwd > 0).all()
    assert rev.tolist() == [1.0, 0.0, 0.0]
    for rule in ("teleport", "uniform"):
        r, _ = power_iteration(AT, v, damping=0.85, dangling=rule)
        assert r == pytest.approx(_closed_form(AT, v, 0.85, rule), abs=1e-8)


@pytest.mark.parametrize("rule", ["teleport", "uniform"])
def test_output_sum_nonnegative_and_n_iter(rule):
    A = _random_graph()
    for v in _teleports(A.shape[0]).values():
        r, it = power_iteration(A, v, dangling=rule)
        assert r.shape == (A.shape[0],)
        assert r.sum() == pytest.approx(1.0, abs=1e-12)
        assert (r >= 0).all()
        assert isinstance(it, int) and 1 <= it < config.MAX_ITER


def test_teleport_is_normalized_inside():
    A = _random_graph()
    v = _teleports(A.shape[0])["random"]
    r1, _ = power_iteration(A, v)
    r2, _ = power_iteration(A, 37.0 * v)
    assert r1 == pytest.approx(r2, abs=1e-9)


def test_not_converged_warns():
    with pytest.warns(RuntimeWarning, match="수렴"):
        r, it = power_iteration(FLOW, _uniform(3), damping=1.0, max_iter=3)
    assert it == 3 and r.sum() == pytest.approx(1.0)


def test_dense_input_and_self_loops():
    r_sparse, _ = power_iteration(TRAP, _uniform(3), damping=0.8)
    r_dense, _ = power_iteration(TRAP.toarray(), _uniform(3), damping=0.8)
    assert r_sparse == pytest.approx(r_dense, abs=1e-15)


# ---------------------------------------------------------------------------
# 일반 검증: 시작값
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["teleport", "uniform"])
def test_init_does_not_change_limit(rule):
    A = _random_graph()
    for name, v in _teleports(A.shape[0]).items():
        r_v, _ = power_iteration(A, v, dangling=rule, init=None)
        r_u, _ = power_iteration(A, v, dangling=rule, init="uniform")
        assert r_v == pytest.approx(r_u, abs=1e-8), name


def test_dangling_seed_only_stays_exactly_on_seed():
    # seed 2는 나가는 링크가 없음. "teleport" 규칙이면 정확한 해는 seed에만 질량 1.
    # 0 -> 1 -> 0 순환이 있어서 1/N에서 시작하면 0, 1의 질량이 조금씩만 줄어든다.
    A = _graph([(0, 1), (1, 0), (1, 2), (3, 2)], 4)
    v = np.array([0.0, 0.0, 1.0, 0.0])
    r, _ = power_iteration(A, v, dangling="teleport", init=None)
    assert r[2] == 1.0
    assert (r[[0, 1, 3]] == 0.0).all()
    # 1/N에서 시작하면 작은 값이 남는다 (그래서 기본 시작값이 v)
    r_u, _ = power_iteration(A, v, dangling="teleport", init="uniform")
    assert r_u[[0, 1, 3]].max() > 0
    assert r_u == pytest.approx(r, abs=1e-7)


# ---------------------------------------------------------------------------
# 일반 검증: 선형성 (dangling = "uniform", Spam Mass 근거, 11강 p80)
# ---------------------------------------------------------------------------
def test_uniform_rule_is_linear_in_teleport():
    A = _random_graph()
    t = _teleports(A.shape[0])
    v1 = t["seeds"] / t["seeds"].sum()
    v2 = t["random"] / t["random"].sum()
    r1, _ = power_iteration(A, v1, dangling="uniform")
    r2, _ = power_iteration(A, v2, dangling="uniform")
    r12, _ = power_iteration(A, (v1 + v2) / 2, dangling="uniform")
    assert r12 == pytest.approx((r1 + r2) / 2, abs=1e-8)


def test_teleport_rule_is_not_linear_in_teleport():
    # 같은 조합을 "teleport" 규칙으로 하면 일반적으로 성립하지 않는다 (d = v가 v를 따라 바뀜)
    A = _random_graph()
    t = _teleports(A.shape[0])
    v1 = t["seeds"] / t["seeds"].sum()
    v2 = t["random"] / t["random"].sum()
    r1, _ = power_iteration(A, v1, dangling="teleport")
    r2, _ = power_iteration(A, v2, dangling="teleport")
    r12, _ = power_iteration(A, (v1 + v2) / 2, dangling="teleport")
    assert np.abs(r12 - (r1 + r2) / 2).max() > 1e-4


# ---------------------------------------------------------------------------
# 일반 검증: 잘못된 입력
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"A": sp.csr_matrix((3, 4))},                       # 정사각이 아님
    {"teleport": np.ones(4)},                           # 길이 불일치
    {"teleport": np.ones((3, 1))},                      # 모양 불일치
    {"teleport": np.array([1.0, -0.5, 0.5])},           # 음수
    {"teleport": np.zeros(3)},                          # 합 0
    {"teleport": np.array([1.0, np.nan, 0.0])},         # NaN
    {"A": sp.csr_matrix(-FLOW.toarray())},              # A에 음수
    {"dangling": "uniformm"},                           # 오타
    {"init": "v"},                                      # 오타
    {"damping": 0.0},
    {"damping": -0.1},
    {"damping": 1.5},
    {"damping": float("nan")},
    {"tol": 0.0},
    {"max_iter": 0},
], ids=lambda kw: next(iter(kw)))
def test_bad_inputs(kwargs):
    args = {"A": FLOW, "teleport": _uniform(3)}
    args.update(kwargs)
    with pytest.raises(ValueError):
        power_iteration(args.pop("A"), args.pop("teleport"), **args)


# ---------------------------------------------------------------------------
# pagerank.py
# ---------------------------------------------------------------------------
def test_pagerank_matches_lecture_and_info():
    res = PR.pagerank(TRAP, damping=0.8)
    assert res.scores == pytest.approx([7 / 33, 5 / 33, 21 / 33], abs=1e-8)
    for key in ("method", "damping", "dangling", "n_iter", "seconds", "converged"):
        assert key in res.info
    assert res.info["method"] == "pagerank" and res.info["dangling"] == "teleport"
    assert res.info["n_iter"] >= 1 and res.info["converged"]


def test_pagerank_uniform_teleport_rules_agree():
    A = _random_graph()
    a = PR.pagerank(A, dangling="teleport").scores
    b = PR.pagerank(A, dangling="uniform").scores
    assert a == pytest.approx(b, abs=1e-12)
    assert a == pytest.approx(_closed_form(A, _uniform(A.shape[0]), config.DAMPING, "uniform"), abs=1e-8)
    assert PR.pagerank(A).info["n_dangling"] == int(dangling_mask(A).sum())


def test_pagerank_detection_sign():
    A = _random_graph()
    r = PR.pagerank(A).scores
    plus = PR.pagerank_detection_score(A, config.DAMPING, sign=+1)
    minus = PR.pagerank_detection_score(A, config.DAMPING, sign=-1)
    assert plus.scores == pytest.approx(r) and minus.scores == pytest.approx(-r)
    assert plus.info["sign"] == 1 and minus.info["sign"] == -1
    with pytest.raises(ValueError):
        PR.pagerank_detection_score(A, config.DAMPING, sign=0)

"""seeds.py, trustrank.py 테스트. 실행: python -m pytest tests -q

담당: 박병언 (팀 작성 코드)

- 강의 11강 p46 예제로 TrustRank = topic-specific PageRank 확인 (p73, p75)
- 신뢰 감쇠 (p76), Spam Mass와 link farm (p68, p79, p80), 상수배 불변
- seed 선택 (p77, p78), make_score_fn, final_rows (확정안 8.2절)
"""

import numpy as np
import pytest
import scipy.sparse as sp

from src import config
from src import seeds as S
from src import trustrank as T
from src.engine import power_iteration
from src.pagerank import pagerank


def _graph(edges, n):
    r, c = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))


def _ranks(x):
    """동률은 같은 순위 (작은 값부터 0, 1, 2, ...)."""
    return np.unique(x, return_inverse=True)[1]


TOPIC = _graph([(0, 1), (0, 2), (1, 0), (2, 3), (3, 2)], 4)   # 11강 p46


# link farm 그래프 (11강 p68)
#   정상 seed 0, 1, 2: 0 -> 1 -> 2 -> 0 순환, 0 -> 3, 1 -> 3 (3은 정상 seed에서만 링크를 받음)
#   3 -> 4 -> 5 -> 0 (seed에서 멀어지는 정상 host). 4 -> 6 (정상 페이지가 블로그 댓글란 같은 곳에 링크)
#   스패머 접근 가능 페이지 6 -> t, 목표 t = 7, farm 페이지 8 ~ 12: 각각 -> t, t -> 각각
#   정상 seed는 farm에 링크하지 않는다. 13 <-> 14, 14 -> 15 (seed에서 닿지 않음, 15는 dangling)
GOOD = [0, 1, 2]
TARGET = 7
FARM = [8, 9, 10, 11, 12]
FARM_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (3, 4), (4, 5), (5, 0), (4, 6), (6, TARGET),
              (13, 14), (14, 13), (14, 15)]
FARM_EDGES += [(f, TARGET) for f in FARM] + [(TARGET, f) for f in FARM]
N_FARM = 16
AF = _graph(FARM_EDGES, N_FARM)
NAMES = ["www.ox.ac.uk", "dept.cam.ac.uk", "nhs.nhs.uk", "a.co.uk", "b.co.uk", "c.co.uk",
         "blog.co.uk", "cheap-pills.co.uk", "f1.co.uk", "f2.co.uk", "f3.co.uk", "f4.co.uk",
         "spam.gov.uk", "x.co.uk", "y.co.uk", "z.co.uk"]          # 12는 신뢰 도메인 이름이지만 스팸


# ---------------------------------------------------------------------------
# TrustRank = topic-specific PageRank (p46, p73, p75)
# ---------------------------------------------------------------------------
def test_trustrank_equals_topic_specific_pagerank_p46():
    res = T.trustrank(TOPIC, [0], damping=0.8)
    assert res.scores == pytest.approx([0.294, 0.118, 0.327, 0.261], abs=5e-4)
    assert res.scores.sum() == pytest.approx(1.0)
    assert res.info["n_seeds"] == 1 and res.info["dangling"] == "teleport" and res.info["n_iter"] >= 1


def test_trust_decays_with_distance_p76():
    # 0 = seed. 0 -> 1 -> 2 -> 3 -> 4 사슬. 5, 6은 서로만 링크하고 5 -> 2 (seed에서 닿지 않음)
    A = _graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 5), (5, 2)], 7)
    r = T.trustrank(A, [0]).scores
    assert r[0] > r[1] > r[2] > r[3] > r[4]          # 거리에 따라 약해짐
    assert r[0] == r.max()
    assert r[5] == r[6] == r.min() == 0.0            # seed에서 닿지 않으면 신뢰 0


def test_trust_splits_by_out_degree_p76():
    # seed 0이 1에만 링크, seed 2는 3, 4, 5에 나눠 링크 -> 1이 3보다 신뢰를 많이 받는다
    A = _graph([(0, 1), (2, 3), (2, 4), (2, 5)], 6)
    r = T.trustrank(A, [0, 2]).scores
    assert r[1] > r[3] == pytest.approx(r[4]) == pytest.approx(r[5])


# ---------------------------------------------------------------------------
# Spam Mass (p68, p79, p80)
# ---------------------------------------------------------------------------
def test_spam_mass_link_farm():
    sm = T.spam_mass_score(AF, GOOD).scores
    assert np.isfinite(sm).all()
    assert sm[3] < np.median(sm)                     # 정상 seed에서만 링크를 받으면 낮음
    assert sm[3] == sm[[3, 4, 5]].min()
    assert sm[TARGET] > sm[[3, 4, 5]].max()          # farm 목표 t는 정상 non-seed 누구보다 높음
    assert sm[TARGET] > sm[:7].max() and sm[TARGET] > 0.5   # 정상 host 0 ~ 6 누구보다 높음
    assert (sm[[13, 14, 15]] == 1.0).all()           # 신뢰가 전혀 안 닿으면 PageRank 전부가 스팸 몫
    # 목표 t는 PageRank도 높다 (link farm이 PageRank를 끌어올림, p68 ~ p70)
    pr = pagerank(AF).scores
    assert pr[TARGET] == pr.max()


def test_spam_mass_uses_uniform_rule_and_formula():
    pr = pagerank(AF, dangling="uniform")
    tr, _ = power_iteration(AF, T.seed_vector(N_FARM, GOOD), dangling="uniform")
    res = T.spam_mass_score(AF, GOOD, pr=pr)
    assert res.scores == pytest.approx(1 - tr / pr.scores, abs=1e-12)
    assert res.info["dangling"] == "uniform" and res.info["score"] == "spam_mass"
    with pytest.raises(ValueError):   # "teleport" 규칙 PageRank를 넘기면 거부
        T.spam_mass_score(AF, GOOD, pr=pagerank(AF, dangling="teleport"))


def test_lecture_r_plus_is_scaled_trustrank_p75_p80():
    # 강의 p80의 r⁺ (teleport 질량을 신뢰 페이지에만 1/N씩)는 (|S|/N) x TR 과 같다 ("within a scaling factor")
    beta, n = config.DAMPING, N_FARM
    A = AF.toarray()
    P = np.zeros((n, n))
    for i in range(n):
        P[i] = A[i] / A[i].sum() if A[i].sum() else 1.0 / n   # dangling "uniform"
    ind = np.zeros(n)
    ind[GOOD] = 1.0 / n
    r_plus = np.linalg.solve(np.eye(n) - beta * P.T, (1 - beta) * ind)
    tr = T.trustrank(AF, GOOD, dangling="uniform").scores
    assert r_plus == pytest.approx(len(GOOD) / n * tr, abs=1e-8)


@pytest.mark.parametrize("c", [0.5, len(GOOD) / N_FARM, 7.0])
def test_spam_mass_rank_invariant_to_tr_scale(c):
    pr = pagerank(AF, dangling="uniform").scores
    tr = T.trustrank(AF, GOOD, dangling="uniform").scores
    base = 1 - tr / pr
    scaled = 1 - (c * tr) / pr
    assert (_ranks(base) == _ranks(scaled)).all()


def test_trustrank_score_is_negated_trust():
    r = T.trustrank(AF, GOOD).scores
    s = T.trustrank_score(AF, GOOD)
    assert s.scores == pytest.approx(-r)
    assert s.info["score"] == "trust" and s.info["dangling"] == "teleport"
    assert s.scores[FARM[0]] > s.scores[[3, 4, 5]].max()    # 높을수록 스팸: farm 페이지가 정상 non-seed보다 높음


def test_farm_amplifies_leaked_trust_but_spam_mass_catches_target():
    # 4 -> 6 -> t 로 새어 들어간 신뢰가 t -> farm -> t 순환에서 불어나, t의 신뢰가 정상 host 4보다 높아진다.
    # 신뢰 점수 하나로는 t를 못 잡는 경우. PageRank 대비 비율(Spam Mass)로 보면 t가 4보다 훨씬 높다 (p79, p80).
    trust = T.trustrank(AF, GOOD).scores
    sm = T.spam_mass_score(AF, GOOD).scores
    assert trust[TARGET] > trust[4]
    assert sm[TARGET] > sm[4]


# ---------------------------------------------------------------------------
# seeds.py (p77, p78)
# ---------------------------------------------------------------------------
NON = np.array([3, 5, 8, 13, 21, 34, 55, 89])


def test_random_seeds_reproducible_and_inside_candidates():
    a = S.random_seeds(NON, 4, seed=42)
    b = S.random_seeds(list(reversed(NON)), 4, seed=42)    # 입력 순서와 무관
    assert (a == b).all() and len(a) == 4 and len(set(a)) == 4
    assert set(a) <= set(NON)
    results = {tuple(S.random_seeds(NON, 4, seed=s)) for s in config.REPEAT_SEEDS}
    assert len(results) > 1                                # seed가 다르면 보통 다른 결과
    assert (S.random_seeds(NON, None) == np.sort(NON)).all()
    assert (S.random_seeds(NON, len(NON)) == np.sort(NON)).all()


def test_pagerank_top_seeds_order_and_ties():
    pr = np.zeros(100)
    pr[[3, 5, 8, 13]] = [0.1, 0.3, 0.3, 0.2]
    pr[50] = 0.9                                           # 정상 목록 밖은 상위여도 안 나옴
    top = S.pagerank_top_seeds(NON, pr, 3)
    assert top.tolist() == [5, 8, 13]                      # 0.3 동률은 hostid 작은 순
    assert 50 not in S.pagerank_top_seeds(NON, pr, None)
    assert set(S.pagerank_top_seeds(NON, pr, None)) == set(NON)


def test_trusted_domain_suffix_rule():
    yes = ["www.ox.ac.uk", "ac.uk", "WWW.Cabinet.GOV.UK.", "a.b.nhs.uk:8080", "x.sch.uk",
           "y.police.uk", "z.mod.uk"]
    no = ["fakeac.uk.example.com", "xac.uk", "gov.uk.spam.com", "ac.uk.co", "www.bbc.co.uk", "uk"]
    assert all(S.is_trusted_host(h) for h in yes)
    assert not any(S.is_trusted_host(h) for h in no)


def test_trusted_domain_rule_matches_splits():
    # 채점 대상에서 빼는 규칙(splits.is_trust_domain)과 seed 규칙이 같아야 한다
    from src.splits import is_trust_domain
    names = ["www.ox.ac.uk", "ac.uk", "fakeac.uk.example.com", "xac.uk", "WWW.Cabinet.GOV.UK.",
             "a.nhs.uk:80", "www.bbc.co.uk"] + NAMES
    assert [S.is_trusted_host(h) for h in names] == [is_trust_domain(h) for h in names]


def test_trusted_domain_seeds_excludes_given_spam():
    all_trusted = S.trusted_domain_ids(NAMES)
    assert all_trusted.tolist() == [0, 1, 2, 12]
    got = S.trusted_domain_seeds(NAMES, exclude_spam_ids=[12, 7])
    assert got.tolist() == [0, 1, 2]
    two = S.trusted_domain_seeds(NAMES, exclude_spam_ids=[12], k=2, seed=43)
    assert len(two) == 2 and set(two) <= {0, 1, 2}
    assert (two == S.trusted_domain_seeds(NAMES, exclude_spam_ids=[12], k=2, seed=43)).all()


@pytest.mark.parametrize("call", [
    lambda: S.random_seeds(NON, len(NON) + 1),
    lambda: S.random_seeds(NON, 0),
    lambda: S.pagerank_top_seeds(NON, np.ones(100), 9),
    lambda: S.trusted_domain_seeds(NAMES, exclude_spam_ids=[12], k=4),
    lambda: S.random_seeds([], 1),
])
def test_seeds_bad_k(call):
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------------------
# make_score_fn, final_rows
# ---------------------------------------------------------------------------
SEED_SPAM = np.array([TARGET, 12])      # 그 fold의 seed용 스팸
SEED_NON = np.array([0, 1, 2, 3])       # 그 fold의 seed용 정상


@pytest.mark.parametrize("method, k", [("set1", None), ("set1", 2), ("pagerank_top", 2), ("trusted_domain", None)])
@pytest.mark.parametrize("score", ["trust", "spam_mass"])
def test_make_score_fn_output(method, k, score):
    fn = T.make_score_fn(AF, NAMES, method, score, k=k)
    s = fn(SEED_SPAM, SEED_NON)
    assert s.shape == (N_FARM,) and np.isfinite(s).all()
    assert s[FARM[0]] > s[[3, 4, 5]].max()        # 높을수록 스팸: farm 페이지가 정상 non-seed보다 높음
    if score == "spam_mass":
        assert s[TARGET] > s[[3, 4, 5]].max()     # Spam Mass는 farm 목표 t도 정상 non-seed보다 높음
    info = fn.infos[-1]
    assert info["seed_method"] == method and info["score"] == score and info["n_iter"] >= 1
    if method == "trusted_domain":
        assert info["n_seeds"] == 3                # 12는 seed_spam이라 빠짐


def test_make_score_fn_pagerank_once(monkeypatch):
    calls = []
    real = T.pagerank
    monkeypatch.setattr(T, "pagerank", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    fn = T.make_score_fn(AF, NAMES, "pagerank_top", "spam_mass", k=2)
    for _ in range(3):
        fn(SEED_SPAM, SEED_NON)
    assert len(calls) == 1
    fn2 = T.make_score_fn(AF, NAMES, "set1", "trust")
    fn2(SEED_SPAM, SEED_NON)
    assert len(calls) == 1                         # trust + set1이면 PageRank가 필요 없음


def test_make_score_fn_bad_args():
    with pytest.raises(ValueError):
        T.make_score_fn(AF, NAMES, "random", "trust")
    with pytest.raises(ValueError):
        T.make_score_fn(AF, NAMES, "set1", "mass")
    with pytest.raises(ValueError):
        T.make_score_fn(AF, NAMES[:-1], "set1", "trust")


def test_final_rows_four_lines():
    rows = T.final_rows(AF, NAMES, set1_spam=SEED_SPAM, set1_nonspam=SEED_NON)
    assert set(rows) == {
        "trustrank_seed-set1_kall_score-trust_d0.85",
        "trustrank_seed-set1_kall_score-spam_mass_d0.85",
        "trustrank_seed-trusted_domain_kall_score-trust_d0.85",
        "trustrank_seed-trusted_domain_kall_score-spam_mass_d0.85",
    }
    assert rows["trustrank_seed-set1_kall_score-trust_d0.85"].info["n_seeds"] == 4
    assert rows["trustrank_seed-trusted_domain_kall_score-trust_d0.85"].info["n_seeds"] == 3   # SET1 스팸 12 제외
    for res in rows.values():
        assert res.scores.shape == (N_FARM,) and np.isfinite(res.scores).all()
    # make_score_fn(전부)과 같은 값
    fn = T.make_score_fn(AF, NAMES, "set1", "spam_mass")
    assert fn(SEED_SPAM, SEED_NON) == pytest.approx(rows["trustrank_seed-set1_kall_score-spam_mass_d0.85"].scores)


def test_setting_name():
    assert T.setting_name("set1", "trust") == "trustrank_seed-set1_kall_score-trust_d0.85"
    assert T.setting_name("set1", "trust", k=50, seed=42) == "trustrank_seed-set1_k50_score-trust_d0.85_s42"
    assert T.setting_name("pagerank_top", "trust", k=50, seed=42) == "trustrank_seed-pagerank_top_k50_score-trust_d0.85"

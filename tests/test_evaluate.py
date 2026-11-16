"""evaluate.py 테스트 (합성 데이터)."""
import numpy as np
import pytest

from src import evaluate as E


def _target(n=400, n_dom=120, seed=1):
    rng = np.random.default_rng(seed)
    dom = np.array([f"d{rng.integers(n_dom)}" for _ in range(n)])
    y = (rng.random(n) < 0.08).astype(int)
    return E.EvalTarget(np.arange(n), y, dom)


def test_precision_at_k_ties():
    s = np.array([5, 4, 4, 4, 4, 1.0])
    y = np.array([1, 1, 0, 0, 0, 1])
    # 상위 3: 5 하나 확정 + 동률 4개 중 2자리, 그 묶음 스팸 1/4 → 1 + 2*(1/4) = 1.5 → 0.5
    assert E.precision_at_k(s, y, 3) == pytest.approx(0.5)
    assert E.precision_at_k(np.arange(6.0), np.array([0, 0, 0, 0, 1, 1]), 2) == 1.0


def test_perfect_and_random_scores():
    t = _target()
    perfect = t.y.astype(float)
    r = E.evaluate_detection(perfect, t, boot=E.bootstrap_indices(t, 50))
    assert r["pr_auc"] == pytest.approx(1.0) and r["roc_auc"] == pytest.approx(1.0)
    const = np.zeros(t.n)
    r = E.evaluate_detection(const, t, boot=E.bootstrap_indices(t, 50))
    assert r["roc_auc"] == pytest.approx(0.5)
    assert r["p@50"] == pytest.approx(t.y.mean())  # 전부 동률 → 기댓값 = 스팸 비율


def test_bootstrap_reproducible_and_domain_level():
    t = _target()
    b1, b2 = E.bootstrap_indices(t, 20), E.bootstrap_indices(t, 20)
    assert all((x == y).all() for x, y in zip(b1, b2))
    # 도메인 단위: 한 표본에서 어떤 도메인이 뽑히면 그 도메인 host가 전부 들어감
    idx = b1[0]
    dom_counts = {d: (t.domain == d).sum() for d in np.unique(t.domain)}
    for d in np.unique(t.domain[idx]):
        assert (t.domain[idx] == d).sum() % dom_counts[d] == 0


def test_compare_identical_is_no_difference():
    t = _target()
    s = np.random.default_rng(3).random(t.n) + t.y * 0.5
    c = E.compare(s, s, t, boot=E.bootstrap_indices(t, 100))
    assert c["diff"] == 0 and c["verdict"] == "차이 없음"
    good, bad = t.y + np.random.default_rng(4).random(t.n) * 0.1, np.random.default_rng(5).random(t.n)
    c = E.compare(good, bad, t, boot=E.bootstrap_indices(t, 100))
    assert c["verdict"] == "A가 높음"


def test_top_k_tiebreak_deterministic():
    t = _target()
    s = np.zeros(t.n)
    a, b = E.top_k_hosts(s, t, 100), E.top_k_hosts(s, t, 100)
    assert (a == b).all() and len(set(a)) == 100


def test_ranking_buckets():
    t = _target()
    s = t.y.astype(float)  # 스팸이 전부 상위
    res = E.evaluate_ranking(s, t, n_buckets=10, boot=E.bootstrap_indices(t, 30))
    assert len(res) == 10 and res[0]["spam_rate"] >= res[-1]["spam_rate"]
    assert sum(r["n_hosts"] for r in res) == t.n


def test_errors():
    t = _target()
    with pytest.raises(ValueError):
        E.evaluate_detection(np.full(t.n, np.nan), t, boot=[])
    with pytest.raises(ValueError):
        E.EvalTarget.build([0, 1], {0: "spam", 1: "undecided"}, lambda i: "d")

"""담당: 박병언 (팀 작성 코드).

동률 진단 검증, 확정안 6절, 8.1절. 강의 11강 p73-78.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from experiments.tie_check import tie_statistics, reachable_from_seeds, rank_bucket_members
from src.evaluate import EvalTarget, evaluate_ranking, precision_at_k
from src.trustrank import trustrank


def test_exact_ties_and_empty():
    """근접한 값은 동률로 합치지 않고 빈 부분집합도 기록한다."""
    r = tie_statistics([0, -0., 1, 1, np.nextafter(1., 2)])
    assert r['zero_fraction'] == .4
    assert r['modal_fraction'] == .4
    assert r['n_unique'] == 3
    assert tie_statistics([])['n_hosts'] == 0
    with pytest.raises(ValueError):
        tie_statistics([np.nan])


def test_reachability_direction_and_zero():
    """역방향 링크와 고립 host는 seed에서 도달할 수 없다."""
    a = csr_matrix(([1, 1, 1], ([0, 1, 3], [1, 2, 0])), shape=(5, 5))
    reachable = reachable_from_seeds(a, [0])
    assert reachable.tolist() == [True, True, True, False, False]
    assert np.array_equal(trustrank(a, [0]).scores == 0, ~reachable)


def test_bucket_members_match_evaluator():
    """구간별 비율과 구체적인 host 목록이 기존 동률 규칙과 일치한다."""
    s = np.array([1., 1., 1., 1., 0., 0.])
    t = EvalTarget(np.arange(6), np.array([1, 1, 0, 0, 1, 0]), np.arange(6))
    parts = rank_bucket_members(s, t, 3)
    r = evaluate_ranking(s, t, 3, [np.arange(6)])
    assert [t.y[p].mean() for p in parts] == [b['spam_rate'] for b in r]
    assert len(set(np.concatenate(parts))) == 6
    assert precision_at_k(s, t.y, 1) == .5


def test_bucket_figure_and_dirty_guard(tmp_path, monkeypatch):
    """구간 그림을 저장하며 dirty 최종 분석은 데이터 로드 전에 거부한다."""
    import pandas as pd
    from src.figures import fig_rank_buckets
    from experiments import tie_check
    t = EvalTarget(np.arange(20), np.arange(20) % 2, np.arange(20))
    rows = []
    for key in ('pagerank', 'trustrank_set1'):
        rows.extend(dict(ranking=key, **b) for b in evaluate_ranking(np.arange(20), t, boot=[np.arange(20)]))
    assert fig_rank_buckets(pd.DataFrame(rows), .5, tmp_path).exists()
    monkeypatch.setattr(tie_check, 'src_dirty', lambda: True)
    assert tie_check.main(['--full']) == 1

"""담당: 박병언 (팀 작성 코드).

순위 평균 검증, 확정안 6절, 9절. 강의 11강 p73-80, 4강 p24-58.
"""
import numpy as np
import pytest
from src.combine import rank_average


def test_average_ties_and_spam_direction():
    """동률은 평균 순위이고 높은 평균 순위가 스팸 방향이다."""
    np.testing.assert_allclose(rank_average([[0, 0, 10], [30, 10, 10]]), [2.25, 1.5, 2.25])


def test_scale_and_method_order_invariance():
    """방법의 단위와 입력 순서가 결합 가중치를 바꾸지 않는다."""
    a = [1, 3, 2]
    b = [3, 2, 1]
    np.testing.assert_equal(rank_average([a, b]), rank_average([b, np.exp(a)]))


@pytest.mark.parametrize('scores', [[], [[]], [[np.nan]], [[1, 2], [1]], [1, 2]])
def test_invalid(scores):
    """잘못된 모양과 결측 점수는 거부한다."""
    with pytest.raises(ValueError):
        rank_average(scores)


def test_fold_combination_uses_only_seed_labels(tmp_path, monkeypatch):
    """결합 채점의 모든 단독 방법이 같은 seed와 같은 bootstrap을 사용한다."""
    from experiments import run_combine as runner
    from experiments.run_main import Context, Method
    from src.evaluate import EvalTarget, bootstrap_indices
    from scipy.sparse import csr_matrix
    target = EvalTarget(np.array([2, 3, 4, 5]), np.array([1, 0, 1, 0]), np.array(['a', 'b', 'c', 'd']))
    ctx = Context('check', csr_matrix((6, 6)), ['x'] * 6, np.array([0]), np.array([1]),
                  target, np.ones(6, dtype=bool), tmp_path, {'fold': 0})
    calls = []
    def compute(c, seed):
        calls.append((c.seed_spam.tolist(), c.seed_non.tolist(), seed))
        return np.arange(6, dtype=float), {}
    monkeypatch.setattr(runner, 'methods', lambda: [Method(k, k, k, True, (42,), compute) for k in runner.KEYS])
    monkeypatch.setattr(runner, 'analysis_cached', lambda name, compute, mode, force: compute())
    scores = runner.component_scores(ctx)
    assert calls == [([0], [1], 42)] * 3
    boot = bootstrap_indices(target, n_boot=20)
    monkeypatch.setattr(runner, 'bootstrap_indices', lambda t: boot)
    rows = runner.comparison_rows(ctx, scores, 'validation')
    assert len(rows) == 4
    for row in rows[:3]:
        assert row['combination_minus_single'] == 0
        assert row['diff_ci95_lo'] == row['diff_ci95_hi'] == 0
        assert row['verdict'] == '차이 없음'


def test_cv_mean_difference_paired_bootstrap(tmp_path):
    """동일 점수들의 검증 평균 차이와 차이 구간은 정확히 0이다."""
    from experiments.run_combine import validation_differences, KEYS
    from experiments.run_main import Context
    from src.evaluate import EvalTarget
    from scipy.sparse import csr_matrix
    contexts, scores = [], {}
    for f in range(2):
        t = EvalTarget(np.arange(8), np.arange(8) % 2, np.arange(8))
        contexts.append(Context('check', csr_matrix((8, 8)), ['x'] * 8, np.array([], int),
                                np.array([], int), t, np.ones(8, bool), tmp_path, {'fold': f}))
        scores[f] = {k: np.arange(8, dtype=float) for k in (*KEYS, 'rank_average')}
    for result in validation_differences(contexts, scores, n_boot=20).values():
        assert result['combination_minus_single'] == 0
        assert result['diff_ci95_lo'] == result['diff_ci95_hi'] == 0


def test_full_dirty_guard(monkeypatch):
    """dirty 최종 결합은 데이터 로드 전에 거부한다."""
    from experiments import run_combine
    monkeypatch.setattr(run_combine, 'src_dirty', lambda: True)
    assert run_combine.main(['--full']) == 1

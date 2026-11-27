"""담당: 박병언 (팀 작성 코드).

민감도 계산 검증, 확정안 4절, 6절, 8.2절, 9절. 강의 11강 p77-78.
"""
import numpy as np
import pytest
from src.evaluate import EvalTarget
from src.sensitivity import choose_seeds, score_point, summarize_repeats, validation_contexts
import pandas as pd
from scipy.sparse import csr_matrix


def test_all_seeds_and_domain_exclusion():
    """전부 선택은 반복 seed와 무관하며 도메인 방식은 알려진 스팸만 뺀다."""
    names = ['a.gov.uk', 'b.gov.uk', 'c.co.uk', 'd.gov.uk']
    for repeat in range(42, 47):
        np.testing.assert_equal(choose_seeds(names, [1], [0, 2], 'set1', None, repeat), [0, 2])
        np.testing.assert_equal(choose_seeds(names, [1], [0, 2], 'trusted_domain', None, repeat), [0, 3])
    np.testing.assert_equal(choose_seeds(names, [], [0, 2], 'pagerank_top', 1, 42, np.array([1, 0, 3, 0])), [2])


def test_seed_excluded_from_metrics():
    """seed 라벨은 평가 지표에 포함되지 않는다."""
    t = EvalTarget(np.arange(4), np.array([0, 1, 0, 1]), np.arange(4))
    r = score_point([100, 3, 1, 2], t, [0])
    assert r['n_hosts'] == 3 and r['pr_auc'] == 1.


def test_repeat_sd_not_fold_sd():
    """전부의 반복 표준편차는 fold 간 지표가 달라도 0이다."""
    rows = [dict(analysis='seed_curve', split='validation', method='trustrank', seed_method='set1',
                 requested_k='all', damping=.85, fold=f, repeat=r, n_seeds=30 + f,
                 n_hosts=10, n_spam=2, pr_auc=.2 + f * .2, roc_auc=.5 + f * .1)
            for f in range(2) for r in range(42, 47)]
    s = summarize_repeats(rows).iloc[0]
    assert s.pr_auc_std == 0
    assert s.pr_auc_fold_std > 0
    assert s.n_seeds == 30.5
    assert s.pr_auc_mean == .3 or np.isclose(s.pr_auc_mean, .3)


def test_fold_labels_never_enter_seeds():
    """검증 fold 라벨은 seed 배열에 들어가지 않는다."""
    names = [f'a{i}.co.uk' for i in range(6)]
    train = pd.DataFrame({'hostid': range(6), 'label': ['spam', 'nonspam'] * 3})
    folds = pd.DataFrame({'hostid': range(6), 'fold': [0, 0, 1, 1, 2, 2]})
    contexts = validation_contexts(csr_matrix((6, 6)), names, train, folds, True)
    assert len(contexts) == 1
    c = contexts[0]
    assert not set(c.target.hostid) & set(np.r_[c.seed_spam, c.seed_non])
    assert set(c.target.hostid) == {0, 1}


def test_check_points_and_figures(tmp_path, monkeypatch):
    """합성 그래프로 축소 설정의 모든 점과 그림 세 개의 저장을 확인한다."""
    from experiments import run_sensitivity as runner
    from experiments.run_main import Context
    from src import figures
    n = 350
    a = csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n))
    names = [f'h{i}.gov.uk' if i < 80 else f'h{i}.co.uk' for i in range(n)]
    t = EvalTarget(np.arange(300, 350), np.arange(50) % 2, np.arange(50))
    ctx = Context('check', a, names, np.arange(80, 90), np.arange(90, 300), t,
                  np.zeros(n, dtype=bool), tmp_path, {'fold': 0})
    monkeypatch.setattr(runner, 'analysis_cached', lambda name, compute, mode, force: compute())
    rows = runner.run_context(ctx, 'validation', True, False)
    assert len(rows) == 25
    assert {r['split'] for r in rows} == {'validation'}
    summary = summarize_repeats(rows)
    all_point = summary[(summary.analysis == 'seed_curve') & (summary.requested_k == 'all')].iloc[0]
    assert all_point.n_seeds == 210 and all_point.pr_auc_std == 0
    for analysis in ('seed_curve', 'seed_methods', 'damping_curve'):
        path = getattr(figures, f'fig_{analysis}')(summary[summary.analysis == analysis], tmp_path, check=True)
        assert path.exists()


def test_final_dirty_refuses_before_loading(monkeypatch):
    """최종 진입점은 데이터 로드 전에 dirty 실행을 거부한다."""
    from experiments import run_sensitivity
    monkeypatch.setattr(run_sensitivity, 'src_dirty', lambda: True)
    assert run_sensitivity.main(['--final']) == 1


def test_validation_repeat_sd_uses_fold_means():
    """fold 안의 반복 변동이 상쇄되면 검증 평균의 반복 표준편차는 0이다."""
    rows = [dict(analysis='seed_curve', split='validation', method='trustrank', seed_method='set1',
                 requested_k='10', damping=.85, fold=f, repeat=r, n_seeds=10,
                 n_hosts=10, n_spam=2, pr_auc=v, roc_auc=v)
            for f, r, v in [(0, 42, .2), (0, 43, .4), (1, 42, .4), (1, 43, .2)]]
    s = summarize_repeats(rows).iloc[0]
    assert s.pr_auc_std == 0


@pytest.mark.parametrize('argv', [[], ['--force'], ['--final']])
def test_main_set2_access_requires_final(tmp_path, monkeypatch, argv):
    """기본과 force 실행은 SET1만 읽고 최종 실행만 검증 후 SET2 줄을 만든다."""
    from types import SimpleNamespace
    from experiments import run_sensitivity as runner
    from src import load_graph as loader

    raw = tmp_path / 'raw'
    train_path = raw / loader.LABEL_FILES['train']
    train_path.parent.mkdir(parents=True)
    train_path.write_text('0 spam 1\n1 nonspam 0\n', encoding='utf-8')
    assert not (raw / loader.LABEL_FILES['test']).exists()
    monkeypatch.setattr(runner.config, 'RAW_DIR', raw)
    monkeypatch.setattr(runner.config, 'TABLES_DIR', tmp_path / 'tables')
    monkeypatch.setattr(runner.config, 'FIGURES_DIR', tmp_path / 'figures')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'src_dirty', lambda: False)
    monkeypatch.setattr(runner, 'load_graph', lambda: (csr_matrix((2, 2)), ['a.co.uk', 'b.co.uk']))
    monkeypatch.setattr(runner, 'load_splits', lambda path: None)
    reads, runs, plots = [], [], []

    def read_train(path):
        """SET1 외의 라벨 읽기를 거부한다."""
        assert path == train_path
        reads.append(path)
        return loader.read_labels(path)

    def forbidden_labels(*args, **kwargs):
        """SET2도 읽는 공통 로더 호출을 거부한다."""
        pytest.fail('SET2를 읽는 load_labels 호출 금지')

    def contexts(A, names, train, folds, check):
        """check는 fold 0, 최종은 5개 fold로 분리한다."""
        assert train.hostid.tolist() == [0, 1]
        return [SimpleNamespace(extra={'fold': f}) for f in ([0] if check else range(5))]

    def final_context(mode):
        """명시적인 최종 요청과 검증 완료 후에만 SET2를 허용한다."""
        assert argv == ['--final']
        assert mode == 'final'
        assert runs == [('validation', f) for f in range(5)]
        return SimpleNamespace(extra={'fold': -1})

    def run(ctx, split, check, force):
        """실제 SET2 채점 없이 모드 분기와 표, 그림 입력을 검증한다."""
        assert check == ('--final' not in argv)
        assert force == ('--force' in argv)
        fold = ctx.extra['fold']
        runs.append((split, fold))
        return [dict(analysis=analysis, split=split, fold=fold, method='trustrank',
                     seed_method='set1', requested_k='all', n_seeds=1, damping=.85,
                     repeat=42, pr_auc=.5, roc_auc=.5, n_hosts=2, n_spam=1)
                for analysis in ('seed_curve', 'seed_methods', 'damping_curve')]

    def plot(table, out_dir, check):
        """SET2 줄과 check 표시가 요청 인자에 맞는지 확인한다."""
        assert check == ('--final' not in argv)
        plots.append(set(table.split))

    monkeypatch.setattr(runner, 'read_labels', read_train)
    monkeypatch.setattr(loader, 'load_labels', forbidden_labels)
    monkeypatch.setattr(runner, 'validation_contexts', contexts)
    monkeypatch.setattr(runner, 'build_context', final_context)
    monkeypatch.setattr(runner, 'run_context', run)
    for analysis in ('seed_curve', 'seed_methods', 'damping_curve'):
        monkeypatch.setattr(runner.figures, f'fig_{analysis}', plot)

    assert runner.main(argv) == 0
    assert reads == [train_path]
    expected = {'validation', 'SET2'} if '--final' in argv else {'validation'}
    assert plots == [expected] * 3
    out = tmp_path / 'tables' if '--final' in argv else tmp_path / 'results/check'
    for path in out.glob('sensitivity_*.csv'):
        assert set(pd.read_csv(path).split) == expected
    assert len(list(out.glob('sensitivity_*.csv'))) == 6


def test_full_alias_rejected():
    """--final 이외의 이전 SET2 실행 인자는 거부한다."""
    from experiments import run_sensitivity
    with pytest.raises(SystemExit) as error:
        run_sensitivity.main(['--full'])
    assert error.value.code == 2

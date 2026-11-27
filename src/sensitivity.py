"""담당: 박병언 (팀 작성 코드).

민감도 계산, 확정안 4절, 6절, 8.2절, 9절, 10절. 강의 11강 p17-18, p73-80.
SET2로 설정을 선택하지 않는다. 반복 표준편차와 fold 표준편차를 분리한다.
"""
from __future__ import annotations

import warnings
import json
import numpy as np
import pandas as pd
from src import config, seeds
from src.evaluate import EvalTarget, metrics
from src.results_io import cached, src_dirty
from src.splits import domain_of, fold_hosts, validation_eval_ids


def validation_contexts(A, hostnames, train, folds, check=False):
    """seed용 fold 라벨과 검증 대상을 분리한 Context 목록을 만든다."""
    from experiments.run_main import Context
    from src.load_graph import dangling_mask
    labels = dict(zip(train.hostid, train.label))
    out = []
    for fold in ([0] if check else sorted(folds.fold.unique())):
        ids, _ = fold_hosts(folds, fold)
        spam = np.array([i for i in ids if labels[int(i)] == 'spam'], dtype=int)
        non = np.array([i for i in ids if labels[int(i)] == 'nonspam'], dtype=int)
        target = EvalTarget.build(validation_eval_ids(folds, fold, hostnames), labels,
                                  lambda i: domain_of(hostnames[i]))
        ctx = Context('check', A, hostnames, spam, non, target, dangling_mask(A), config.ROOT / 'results/check')
        ctx.extra['fold'] = int(fold)
        out.append(ctx)
    return out


def analysis_cached(name, compute, mode, force=False):
    """기존 점수 캐시를 재사용하고 dirty 상태를 명시적으로 경고한다."""
    meta = config.SCORES_DIR / mode / f'{name}.json'
    stored_dirty = meta.exists() and json.loads(meta.read_text()).get('dirty', False)
    if src_dirty() or stored_dirty:
        warnings.warn('src/ dirty 상태: 분석 캐시와 현재 코드가 다를 수 있습니다', stacklevel=2)
    return cached(name, compute, mode, force)


def choose_seeds(hostnames, spam, non, method, k, repeat, pr=None):
    """허용된 seed용 라벨만 사용해 seed를 고른다."""
    if method == 'set1':
        return seeds.random_seeds(non, k, repeat)
    if method == 'trusted_domain':
        return seeds.trusted_domain_seeds(hostnames, spam, k, repeat)
    if method == 'pagerank_top':
        return seeds.pagerank_top_seeds(non, pr, k)
    raise ValueError(method)


def without_seeds(target, good):
    """seed host를 채점 대상에서 제외한다."""
    return target.subset(~np.isin(target.hostid, good))


def score_point(scores, target, good):
    """seed를 제외한 PR-AUC와 ROC-AUC, 대상 수를 반환한다."""
    t = without_seeds(target, good)
    return {**metrics(np.asarray(scores)[t.hostid], t.y), 'n_hosts': t.n, 'n_spam': t.n_spam}


def summarize_repeats(rows):
    """반복별 fold 평균의 평균과 표준편차를 낸다. fold 변동은 따로 기록한다."""
    df = pd.DataFrame(rows)
    keys = ['analysis', 'split', 'method', 'seed_method', 'requested_k', 'damping']
    out = []
    for group, d in df.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, group))
        row['n_seeds'] = d.n_seeds.mean()
        row['n_folds'] = d.fold.nunique()
        row['n_repeats'] = d.repeat.nunique()
        row['n_hosts'] = d.groupby('fold').n_hosts.mean().mean()
        row['n_spam'] = d.groupby('fold').n_spam.mean().mean()
        for metric in ('pr_auc', 'roc_auc'):
            per = d.groupby('fold')[metric]
            means = per.mean()
            row[metric + '_mean'] = means.mean()
            repeat_means = d.groupby('repeat')[metric].mean()
            row[metric + '_std'] = repeat_means.std(ddof=1) if len(repeat_means) > 1 else 0.
            row[metric + '_fold_std'] = means.std(ddof=1) if len(means) > 1 else 0.
        out.append(row)
    return pd.DataFrame(out)

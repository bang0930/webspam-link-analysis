"""담당: 박병언 (팀 작성 코드).

동률과 도달 가능성 진단, 확정안 6절, 8.1절, 9절, 10절. 강의 11강 p17-18, p73-80.
기본 check는 SET1 fold 0이다. --final은 저장된 최종 점수로 SET2를 분석한다.
구체적인 목록은 evaluate_ranking과 동일한 seed 42 규칙으로 만든다.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src import config, figures
from src.evaluate import bootstrap_indices
from src.results_io import src_dirty
from experiments.run_main import build_context, methods, ranking, FINAL


def analysis_cached(name, compute, mode, force=False):
    """공통 캐시를 사용하며 현재 또는 저장 당시 dirty 상태를 경고한다."""
    import json
    import warnings
    from src.results_io import cached
    meta = config.SCORES_DIR / mode / f'{name}.json'
    stored_dirty = meta.exists() and json.loads(meta.read_text()).get('dirty', False)
    if src_dirty() or stored_dirty:
        warnings.warn('src/ dirty 상태: 분석 캐시와 현재 코드가 다를 수 있습니다', stacklevel=2)
    return cached(name, compute, mode, force)


def tie_statistics(scores):
    """정확한 부동소수점 동등성으로 0 비율, 최빈값 비율, 고유값 수를 계산한다."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError('유한한 1차원 점수 필요')
    values, counts = np.unique(s, return_counts=True)
    return dict(n_hosts=len(s), zero_count=int((s == 0).sum()),
                zero_fraction=float(np.mean(s == 0)) if len(s) else np.nan,
                modal_value=float(values[counts.argmax()]) if len(s) else np.nan,
                modal_count=int(counts.max()) if len(s) else 0,
                modal_fraction=float(counts.max() / len(s)) if len(s) else np.nan,
                n_unique=len(values), minimum=float(s.min()) if len(s) else np.nan,
                maximum=float(s.max()) if len(s) else np.nan)


def reachable_from_seeds(A, seed_ids):
    """i -> j 링크를 따라 seed에서 도달 가능한 host를 직접 탐색한다."""
    G = csr_matrix(A, copy=True)
    G.eliminate_zeros()
    seen = np.zeros(G.shape[0], dtype=bool)
    stack = list(np.unique(seed_ids))
    seen[stack] = True
    while stack:
        i = stack.pop()
        for j in G.indices[G.indptr[i]:G.indptr[i + 1]]:
            if not seen[j]:
                seen[j] = True
                stack.append(int(j))
    return seen


def rank_bucket_members(scores, target, n_buckets=10, seed=42):
    """evaluate_ranking과 동일하게 동률을 무작위로 끊어 구간 목록을 만든다."""
    s = np.asarray(scores)[target.hostid]
    order = np.lexsort((np.random.default_rng(seed).permutation(len(s)), -s))
    return [target.hostid[p] for p in np.array_split(order, n_buckets)]


def diagnose(ctx, scores):
    """방법별 부분집합 통계, 도달 불가 비교, 순위 구간 겹침을 한 표에 담는다."""
    t = ctx.target
    ids = t.hostid
    subsets = {'graph': np.arange(ctx.A.shape[0]), 'eval': ids,
               'eval_spam': ids[t.y == 1], 'eval_nonspam': ids[t.y == 0],
               'eval_dangling': ids[ctx.dangling[ids]],
               'eval_non_dangling': ids[~ctx.dangling[ids]]}
    rows = []
    for key, s in scores.items():
        for subset, members in subsets.items():
            rows.append(dict(record='statistics', method=key, subset=subset,
                             **tie_statistics(s[members])))
    zero = scores['tr_set1_trust'] == 0
    unreachable = ~reachable_from_seeds(ctx.A, ctx.seed_non)
    for subset in ('graph', 'eval'):
        members = subsets[subset]
        rows.append(dict(record='reachability', method='tr_set1_trust', subset=subset,
                         n_hosts=len(members), zero_count=int(zero[members].sum()),
                         unreachable_count=int(unreachable[members].sum()),
                         zero_unreachable_overlap=int((zero[members] & unreachable[members]).sum()),
                         mismatch_count=int((zero[members] != unreachable[members]).sum())))
    pr = scores['pagerank'] * FINAL['pagerank_sign']
    tr = -scores['tr_set1_trust']
    pb, tb = rank_bucket_members(pr, t), rank_bucket_members(tr, t)
    for b, (p, q) in enumerate(zip(pb, tb), 1):
        rows.append(dict(record='bucket_overlap', bucket=b, n_hosts=len(q),
                         overlap_count=len(set(p) & set(q)), identical_hosts=set(p) == set(q)))
        for key, s, members in (('pagerank', pr, p), ('tr_set1_trust', tr, q)):
            mask = np.isin(ids, members)
            rows.append(dict(record='bucket', method=key, bucket=b,
                             n_spam=int(t.y[mask].sum()), spam_rate=float(t.y[mask].mean()),
                             **tie_statistics(s[members])))
    for repeat in range(42, 47):
        parts = rank_bucket_members(tr, t, seed=repeat)
        for b, members in enumerate(parts, 1):
            mask = np.isin(ids, members)
            rows.append(dict(record='tie_break_repeat', method='tr_set1_trust', repeat=repeat,
                             bucket=b, n_hosts=len(members), n_spam=int(t.y[mask].sum()),
                             spam_rate=float(t.y[mask].mean())))
    for key, s in (('pagerank', pr), ('tr_set1_trust', tr)):
        values, counts = np.unique(s[ids], return_counts=True)
        modal = values[counts.argmax()]
        parts = pb if key == 'pagerank' else tb
        for b, members in enumerate(parts, 1):
            rows.append(dict(record='modal_bucket', method=key, bucket=b, modal_value=modal,
                             modal_count=int((s[members] == modal).sum())))
    middle = tb[2:8]
    rates = [float(t.y[np.isin(ids, members)].mean()) for members in middle]
    tied = sum(len(np.unique(tr[members])) == 1 for members in middle)
    text = (f'구간 3~8 비율 동일={len(set(rates)) == 1}, 단일 점수 구간={tied}/6. '
            'evaluate_ranking은 seed 42 무작위 동률 분할이며 비율을 같게 만드는 기댓값 계산이 아니다. '
            '비율이 같은 구간은 이 분할에서 스팸 수가 우연히 같은 결과다. '
            '동률의 영향은 modal_bucket과 tie_break_repeat 행으로 확인한다. '
            f'구간 9 host 겹침={len(set(pb[8]) & set(tb[8]))}/{len(tb[8])}.')
    rows.append(dict(record='interpretation', explanation=text))
    return pd.DataFrame(rows)


def main(argv=None):
    """SET2를 사용하지 않는 check 또는 최종 분석을 실행한다."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--final', '--full', dest='final', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    if args.final and src_dirty():
        print('[err] src/ dirty: 최종 분석 거부')
        return 1
    ctx = build_context('final' if args.final else 'check')
    ms = [m for m in methods() if m.key != 'random']
    scores = {}
    for m in ms:
        name = m.key if len(m.seeds) == 1 else f'{m.key}_s42'
        scores[m.key], _ = analysis_cached(name, lambda m=m: m.fn(ctx, 42), ctx.mode, args.force)
    table = diagnose(ctx, scores)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    table.insert(0, 'mode', ctx.mode)
    table.to_csv(ctx.out_dir / 'tie_check.csv', index=False)
    r = ranking(ctx, {k: {42: s} for k, s in scores.items()}, bootstrap_indices(ctx.target))
    figures.fig_rank_buckets(r, float(ctx.target.y.mean()),
                             config.FIGURES_DIR if args.final else ctx.out_dir / 'figures',
                             title=None if args.final else 'CHECK MODE - not report numbers')
    print(table[table.record.isin(['reachability', 'bucket_overlap', 'bucket', 'interpretation'])].to_string(index=False))
    print('evaluate_ranking: seed 42 무작위 동률 분할, P@K: 기댓값 방식. 규칙 변경 없음.')
    print(f'저장: {ctx.out_dir}/tie_check.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())

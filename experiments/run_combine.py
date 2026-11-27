"""담당: 박병언 (팀 작성 코드).

단순 순위 결합, 확정안 4절, 6절, 9절, 10절. 강의 11강 p73-80, 4강 p24-58.
기본 check는 fold 0만, --full 또는 --final은 먼저 5-fold 후 SET2를 한 번 채점한다.
가중치 탐색 없이 FINAL 설정을 고정한다. 최종표에는 넣지 않는다.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import config, lsh
from src.combine import rank_average
from src.evaluate import bootstrap_indices, evaluate_detection, compare, cross_validate, metrics
from src.load_graph import load_graph, load_labels
from src.splits import load_splits
from src.results_io import src_dirty
from experiments.run_main import FINAL, methods, build_context, Context

KEYS = ('tr_set1_trust', 'ppr', 'lsh')


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


def validation_contexts(A, hostnames, train, folds, check=False):
    """결합 검증의 seed 라벨과 평가 fold를 분리한다. 확정안 4절, 6절."""
    from src.splits import fold_hosts, validation_eval_ids, domain_of
    from src.evaluate import EvalTarget
    from src.load_graph import dangling_mask
    labels = dict(zip(train.hostid, train.label))
    out = []
    for fold in ([0] if check else sorted(folds.fold.unique())):
        ids, _ = fold_hosts(folds, fold)
        spam = np.array([i for i in ids if labels[int(i)] == 'spam'], dtype=int)
        non = np.array([i for i in ids if labels[int(i)] == 'nonspam'], dtype=int)
        target = EvalTarget.build(validation_eval_ids(folds, fold, hostnames), labels,
                                  lambda i: domain_of(hostnames[i]))
        out.append(Context('check', A, hostnames, spam, non, target, dangling_mask(A),
                           ROOT / 'results/check', {'fold': int(fold)}))
    return out


def component_scores(ctx, force=False, lsh_fn=None):
    """FINAL 설정으로 seed용 라벨만 넘겨 세 방법을 계산한다."""
    out = {}
    for m in methods():
        if m.key not in KEYS:
            continue
        name = f'combination_fold{ctx.extra.get("fold", -1)}_{m.key}_{FINAL["ppr_direction"]}_{FINAL["ppr_score"]}_min{FINAL["lsh_min_set_size"]}_b{FINAL["lsh_bands_rows"][0]}r{FINAL["lsh_bands_rows"][1]}_d{FINAL["damping"]:g}_s42'
        def compute(m=m):
            if m.key == 'lsh' and lsh_fn is not None:
                s = lsh_fn(ctx.seed_spam, ctx.seed_non)
                return s, dict(lsh_fn.structure.stats, fold=ctx.extra.get('fold'), **FINAL)
            return m.fn(ctx, 42)
        out[m.key], _ = analysis_cached(name, compute, ctx.mode, force)
    out['rank_average'] = rank_average([out[k] for k in KEYS])
    return out


def comparison_rows(ctx, scores, split):
    """같은 도메인 bootstrap 표본으로 단독 방법과 결합 차이를 비교한다."""
    boot = bootstrap_indices(ctx.target)
    rows = []
    for key, score in scores.items():
        r = evaluate_detection(score, ctx.target, boot)
        row = dict(split=split, fold=ctx.extra.get('fold', -1), method=key,
                   **r, **FINAL)
        if key != 'rank_average':
            c = compare(scores['rank_average'], score, ctx.target, boot)
            row.update(combination_minus_single=c['diff'], diff_ci95_lo=c['ci95'][0],
                       diff_ci95_hi=c['ci95'][1], verdict=c['verdict'])
        rows.append(row)
    return rows


def validation_differences(contexts, scores_by_fold, n_boot=config.N_BOOT):
    """fold 평균 PR-AUC 차이의 paired 도메인 bootstrap 구간을 계산한다."""
    samples = []
    point = {k: [] for k in KEYS}
    for ctx in contexts:
        scores = scores_by_fold[ctx.extra['fold']]
        t = ctx.target
        per = {k: s[t.hostid] for k, s in scores.items()}
        combo = metrics(per['rank_average'], t.y)['pr_auc']
        for key in KEYS:
            point[key].append(combo - metrics(per[key], t.y)['pr_auc'])
        boot = bootstrap_indices(t, n_boot=n_boot, seed=config.SEED + ctx.extra['fold'])
        fold_draws = {k: [] for k in KEYS}
        for idx in boot:
            ca = metrics(per['rank_average'][idx], t.y[idx])['pr_auc']
            for key in KEYS:
                fold_draws[key].append(ca - metrics(per[key][idx], t.y[idx])['pr_auc'])
        samples.append(fold_draws)
    out = {}
    for key in KEYS:
        d = np.asarray([sample[key] for sample in samples])
        valid = np.isfinite(d).all(axis=0)
        lo, hi = np.quantile(d[:, valid].mean(axis=0), [.025, .975]) if valid.any() else (np.nan, np.nan)
        out[key] = dict(combination_minus_single=float(np.mean(point[key])),
                        diff_ci95_lo=float(lo), diff_ci95_hi=float(hi),
                        verdict='차이 없음' if lo <= 0 <= hi else
                        'A가 높음' if lo > 0 else 'B가 높음' if hi < 0 else '판단 불가',
                        n_boot_valid=int(valid.sum()))
    return out


def main(argv=None):
    """검증을 먼저 끝낸 뒤에만 SET2를 채점한다."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full', '--final', dest='full', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    if args.full and src_dirty():
        print('[err] src/ dirty: full 분석 거부')
        return 1
    A, names = load_graph()
    train, _ = load_labels()
    folds = load_splits(config.SPLITS_PATH)
    contexts = validation_contexts(A, names, train, folds, not args.full)
    structure_fn = None
    def lazy_lsh(spam, non):
        nonlocal structure_fn
        if structure_fn is None:
            structure_fn = lsh.make_score_fn(A, min_set_size=FINAL['lsh_min_set_size'],
                                            bands_rows=FINAL['lsh_bands_rows'], seed=42)
        lazy_lsh.structure = structure_fn.structure
        return structure_fn(spam, non)
    rows, cv_scores = [], {}
    for ctx in contexts:
        scores = component_scores(ctx, args.force, lazy_lsh)
        cv_scores[ctx.extra['fold']] = scores
        rows.extend(comparison_rows(ctx, scores, 'validation'))
    if args.full:
        # 기존 cross_validate로 동일 fold 지표를 요약한다. 계산 점수는 재사용한다.
        diffs = validation_differences(contexts, cv_scores)
        for key in (*KEYS, 'rank_average'):
            def score_fn(spam, non, key=key):
                for ctx in contexts:
                    if np.array_equal(np.sort(spam), np.sort(ctx.seed_spam)) and np.array_equal(np.sort(non), np.sort(ctx.seed_non)):
                        return cv_scores[ctx.extra['fold']][key]
                raise ValueError('캐시와 fold seed 라벨 불일치')
            cv = cross_validate(score_fn, folds, train, names)
            rows.append(dict(split='validation_mean', fold=-1, method=key,
                             **{k: v for k, v in cv.items() if k != 'per_fold'},
                             **diffs.get(key, {}), **FINAL))
        ctx = build_context('final')
        rows.extend(comparison_rows(ctx, component_scores(ctx, args.force, lazy_lsh), 'SET2'))
    out = config.TABLES_DIR if args.full else ROOT / 'results/check'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'combination.csv', index=False)
    print(f'저장: {out}/combination.csv (CHECK, 보고서 숫자 아님)' if not args.full else f'저장: {out}/combination.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())

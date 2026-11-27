"""담당: 박병언 (팀 작성 코드).

민감도 실행, 확정안 4절, 6절, 8.2절, 9절, 10절. 강의 11강 p17-18, p73-80.
기본 check는 SET1만 읽고 fold 0, 축소 설정으로 실행한다. --final만 5-fold 후 SET2를 분석한다.
PPR damping은 미포함. SET2로 어떤 값도 선택하지 않는다.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src import config, figures
from src.load_graph import LABEL_FILES, load_graph, read_labels
from src.splits import load_splits
from src.pagerank import pagerank, pagerank_detection_score
from src.trustrank import trustrank_score
from src.sensitivity import (validation_contexts, analysis_cached, choose_seeds,
                             score_point, summarize_repeats)
from src.results_io import src_dirty
from experiments.run_main import FINAL, build_context


def run_context(ctx, split, check, force):
    """한 fold 또는 SET2에 대해 분석 설정을 계산한다."""
    rows = []
    fold = ctx.extra.get('fold', -1)
    scope = f'{split}_fold{fold}'
    def calculate(analysis, method, seed_method, k, damping, repeat):
        pr = None
        if seed_method == 'pagerank_top':
            pr, _ = analysis_cached(f'sensitivity_pr_seed_d{damping:g}',
                                    lambda: _result(pagerank(ctx.A, damping)), ctx.mode, force)
        good = choose_seeds(ctx.hostnames, ctx.seed_spam, ctx.seed_non, seed_method,
                            k, repeat, pr) if method == 'trustrank' else []
        name = f'sensitivity_{scope}_{method}_{seed_method}_k{k}_d{damping:g}_s{repeat}'
        def compute():
            if method == 'trustrank':
                r = trustrank_score(ctx.A, good, damping, seed_method)
            else:
                r = pagerank_detection_score(ctx.A, damping, FINAL['pagerank_sign'])
            return r.scores, dict(r.info, analysis=analysis, repeat=repeat, fold=fold)
        scores, _ = analysis_cached(name, compute, ctx.mode, force)
        rows.append(dict(analysis=analysis, split=split, fold=fold, method=method,
                         seed_method=seed_method, requested_k='all' if k is None else str(k),
                         n_seeds=len(good), damping=damping, repeat=repeat,
                         **score_point(scores, ctx.target, good)))
    for k in ((10, None) if check else (*config.SEED_COUNTS, None)):
        for repeat in config.REPEAT_SEEDS:
            calculate('seed_curve', 'trustrank', 'set1', k, config.DAMPING, repeat)
    for k in ((50,) if check else config.SEED_METHOD_COUNTS):
        for method in ('set1', 'trusted_domain', 'pagerank_top'):
            for repeat in ((42,) if method == 'pagerank_top' else config.REPEAT_SEEDS):
                calculate('seed_methods', 'trustrank', method, k, config.DAMPING, repeat)
    for damping in ((.7, .85) if check else config.DAMPING_GRID):
        for method in ('trustrank', 'pagerank'):
            calculate('damping_curve', method, 'set1' if method == 'trustrank' else 'none',
                      None, damping, 42)
    return rows


def _result(r):
    """기존 결과 객체를 캐시 콜백 형태로 바꾼다."""
    return r.scores, r.info


def main(argv=None):
    """SET1 check 또는 --final로 요청한 SET2 분석을 실행한다."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--final', action='store_true', help='5-fold 검증 후 SET2 분석')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    if args.final and src_dirty():
        print('[err] src/ dirty: 커밋 후 --final을 실행하세요.')
        return 1
    A, names = load_graph()
    train = read_labels(config.RAW_DIR / LABEL_FILES['train'])
    contexts = validation_contexts(A, names, train, load_splits(config.SPLITS_PATH), not args.final)
    rows = []
    for ctx in contexts:
        rows.extend(run_context(ctx, 'validation', not args.final, args.force))
    if args.final:
        rows.extend(run_context(build_context('final'), 'SET2', False, args.force))
    out = config.TABLES_DIR if args.final else ROOT / 'results/check'
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = summarize_repeats(rows)
    for analysis in ('seed_curve', 'seed_methods', 'damping_curve'):
        raw[raw.analysis == analysis].to_csv(out / f'sensitivity_{analysis}_repeats.csv', index=False)
        d = summary[summary.analysis == analysis]
        d.to_csv(out / f'sensitivity_{analysis}.csv', index=False)
        getattr(figures, f'fig_{analysis}')(d, config.FIGURES_DIR if args.final else out / 'figures',
                                         check=not args.final)
    print(f'저장: {out}/sensitivity_*.csv, PPR damping 미포함')
    return 0


if __name__ == '__main__':
    sys.exit(main())

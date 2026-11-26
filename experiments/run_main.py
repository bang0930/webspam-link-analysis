"""전체 실행: 최종 설정으로 모든 방법 점수 → 채점 → 표 (→ 그림).

공동 파일 (확정안 13절). 틀: 김현중. 수정 전에 단톡방에 알린다.

모드
    python experiments/run_main.py            check 모드 (기본). SET2를 건드리지 않는다.
        seed 라벨 = SET1 fold 1~4, 채점 대상 = SET1 fold 0 − 신뢰 도메인.
        파이프라인 전체를 시험하는 용도. 결과는 results/check/ (깃에 안 올림).
    python experiments/run_main.py --final    최종 모드. 설정이 모두 확정된 뒤 딱 한 번.
        seed 라벨 = SET1 전체, 채점 대상 = eval_set.json (SET2 − undecided − 신뢰 도메인).
        src/에 커밋 안 한 수정이 있으면 실행하지 않는다. 결과는 results/tables/final_*.csv.
    --force: 저장된 점수를 무시하고 전부 다시 계산 (최종 채점과 12/4 재현 테스트에서는 꼭 쓴다).
    --analysis: SET1만 쓰는 분석(EDA, LSH 후보 쌍·검증·S-curve, PPR 검증, 곡선)도 먼저 실행한다.
        보고서의 모든 표·그림을 한 번에 재현할 때: python experiments/run_main.py --final --force --analysis

만드는 표 (확정안 6절)
    *_table.csv        최종 비교표 (방법별 PR-AUC·ROC-AUC·P@50·P@100, 95% 신뢰구간, 5회 반복 표준편차)
    *_comparisons.csv  핵심 비교 4개 + 참고 1개 (같은 bootstrap 표본으로 PR-AUC 차이의 95% 구간)
    *_ranking.csv      PageRank vs TrustRank 순위 구간 10개별 스팸 비율 (Q1)
    *_caught.csv       상위 100개로 "잡은" 스팸: 방법별 수, 그 방법만 잡은 수, 두 방법이 같이 잡은 수 (H3)
    *_subsets.csv      하위 집합 채점: dangling / 비dangling / LSH 커버 (사이트 수·스팸 수·신뢰구간)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                                                  # noqa: E402
from src import baselines as B                                          # noqa: E402
from src import lsh as LSH                                              # noqa: E402
from src import ppr as PPR                                              # noqa: E402
from src import trustrank as TR                                         # noqa: E402
from src.evaluate import (EvalTarget, bootstrap_indices, caught_spam,  # noqa: E402
                          compare, evaluate_detection, evaluate_ranking, metrics)
from src.pagerank import pagerank_detection_score                       # noqa: E402
from src.results_io import cached, git_state, src_dirty                 # noqa: E402

# ---------------------------------------------------------------------------
# 최종 설정 (검증 5-fold 평균 PR-AUC + evaluate.select_setting 규칙으로 고른 값)
# 값을 바꾸면 근거도 같이 고친다. 근거 숫자는 모두 SET1 검증 fold 결과 (SET2 아님).
# ---------------------------------------------------------------------------
FINAL = {
    "pagerank_sign": -1,      # +1: 0.058±0.008, −1: 0.088±0.015 → 차이 ≥ 표준편차라 −1 (2026-11-25)
    "in_degree_sign": -1,     # +1: 0.054±0.005, −1: 0.089±0.011 → −1 (2026-11-22)
    "out_degree_sign": 1,     # +1: 0.072±0.010, −1: 0.073±0.007 → 차이 < 표준편차라 기본값 +1 유지
    "ppr_direction": "forward",   # 링크 방향 raw 0.078 vs 최고 역방향 raw 0.102, 차이 0.024 < 표준편차 0.024
    "ppr_score": "raw",           #   → 기본값 유지 (2026-11-25)
    "lsh_min_set_size": 1,        # lsh_validation.csv: min1 (32,4) 0.103±0.017 vs 기본값 (16,8) 0.070
    "lsh_bands_rows": (32, 4),    #   → 차이 ≥ 표준편차라 (32,4)
    "damping": config.DAMPING,
}
# TrustRank는 최종표 4줄이 미리 정해져 있어 고르는 값이 없다 (seed 2종 × 점수 2종, 확정안 6절).


# ---------------------------------------------------------------------------
# 데이터·라벨 준비
# ---------------------------------------------------------------------------
@dataclass
class Context:
    mode: str                      # "check" 또는 "final"
    A: object                      # 0·1 희소 그래프
    hostnames: list
    seed_spam: np.ndarray          # 점수 계산에 쓰는 라벨
    seed_non: np.ndarray
    target: EvalTarget             # 채점 대상
    dangling: np.ndarray           # 길이 N
    out_dir: Path
    extra: dict = field(default_factory=dict)


def build_context(mode: str) -> Context:
    from src.load_graph import dangling_mask, load_graph, load_labels
    from src.splits import domain_of, load_eval_set, load_splits, validation_eval_ids

    A, hostnames = load_graph()
    train, test = load_labels()
    set1 = train[train["label"].isin(["spam", "nonspam"])]

    if mode == "final":
        seed = set1
        target_ids = load_eval_set(config.EVAL_SET_PATH)
        label_of = dict(zip(test["hostid"], test["label"]))
        out_dir = config.TABLES_DIR
    elif mode == "check":
        folds = load_splits(config.SPLITS_PATH)
        val = set(folds.loc[folds["fold"] == 0, "hostid"])
        seed = set1[~set1["hostid"].isin(val)]
        target_ids = validation_eval_ids(folds, 0, hostnames)
        label_of = dict(zip(train["hostid"], train["label"]))
        out_dir = ROOT / "results" / "check"
    else:
        raise ValueError(mode)

    target = EvalTarget.build(target_ids, label_of, lambda i: domain_of(hostnames[i]))
    return Context(mode, A, hostnames,
                   seed.loc[seed["label"] == "spam", "hostid"].to_numpy(),
                   seed.loc[seed["label"] == "nonspam", "hostid"].to_numpy(),
                   target, dangling_mask(A), out_dir)


# ---------------------------------------------------------------------------
# 방법 목록: (키, 표에 쓸 이름, 묶음, 라벨 사용 여부, 반복 seed, 계산 함수)
# 계산 함수: (ctx, seed) -> (점수 배열, 설정 dict). 새 방법은 여기에 한 줄 추가한다.
# ---------------------------------------------------------------------------
@dataclass
class Method:
    key: str
    label: str
    family: str
    uses_labels: bool
    seeds: tuple
    fn: object


def _pagerank(ctx, seed):
    r = pagerank_detection_score(ctx.A, damping=FINAL["damping"], sign=FINAL["pagerank_sign"])
    return r.scores, r.info


def _trustrank_rows(ctx):
    """최종표 TrustRank 4줄은 PageRank를 공유하므로 한 번에 계산해 ctx.extra에 둔다."""
    if "trustrank" not in ctx.extra:
        ctx.extra["trustrank"] = TR.final_rows(ctx.A, ctx.hostnames, ctx.seed_spam, ctx.seed_non,
                                               damping=FINAL["damping"])
    return ctx.extra["trustrank"]


def _trustrank(name):
    def fn(ctx, seed):
        r = _trustrank_rows(ctx)[name]
        return r.scores, r.info
    return fn


def _ppr(ctx, seed):
    r = PPR.ppr_score(ctx.A, ctx.seed_spam, FINAL["damping"], FINAL["ppr_direction"], FINAL["ppr_score"])
    return r.scores, r.info


def _lsh(ctx, seed):
    r = LSH.run_lsh(ctx.A, ctx.seed_spam, ctx.seed_non, min_set_size=FINAL["lsh_min_set_size"],
                    bands_rows=FINAL["lsh_bands_rows"], seed=seed)
    ctx.extra.setdefault("lsh_covered", {})[seed] = r.covered
    return r.scores, r.stats


def _baseline(kind):
    def fn(ctx, seed):
        if kind == "random":
            return B.random_scores(ctx.A.shape[0], seed), {"seed": seed}
        if kind == "in_degree":
            return B.in_degree_scores(ctx.A, FINAL["in_degree_sign"]), {"sign": FINAL["in_degree_sign"]}
        if kind == "out_degree":
            return B.out_degree_scores(ctx.A, FINAL["out_degree_sign"]), {"sign": FINAL["out_degree_sign"]}
        s, cov = B.one_hop_scores(ctx.A, ctx.seed_spam, ctx.seed_non, kind.split("_")[-1])
        return s, {"direction": kind.split("_")[-1], "covered": int(cov.sum())}
    return fn


def methods() -> list[Method]:
    one = (config.SEED,)
    rep = config.REPEAT_SEEDS
    tr = TR.setting_name
    b, r = FINAL["lsh_bands_rows"]
    return [
        Method("pagerank", "PageRank (참고)", "pagerank", False, one, _pagerank),
        Method("tr_set1_trust", "TrustRank · SET1 seed · 신뢰 부호 반전", "trustrank", True, one,
               _trustrank(tr("set1", "trust"))),
        Method("tr_set1_mass", "TrustRank · SET1 seed · Spam Mass", "trustrank", True, one,
               _trustrank(tr("set1", "spam_mass"))),
        Method("tr_domain_trust", "TrustRank · 신뢰 도메인 seed · 신뢰 부호 반전", "trustrank", False, one,
               _trustrank(tr("trusted_domain", "trust"))),
        Method("tr_domain_mass", "TrustRank · 신뢰 도메인 seed · Spam Mass", "trustrank", False, one,
               _trustrank(tr("trusted_domain", "spam_mass"))),
        Method("ppr", f"PPR · {FINAL['ppr_direction']} · {FINAL['ppr_score']}", "ppr", True, one, _ppr),
        Method("lsh", f"MinHash/LSH · min{FINAL['lsh_min_set_size']} · b{b} r{r}", "lsh", True, rep, _lsh),
        Method("random", "Baseline · 무작위", "baseline", False, rep, _baseline("random")),
        Method("in_degree", "Baseline · in-degree", "baseline", False, one, _baseline("in_degree")),
        Method("out_degree", "Baseline · out-degree", "baseline", False, one, _baseline("out_degree")),
        Method("one_hop_in", "Baseline · 1단계 이웃 (나에게 링크한)", "baseline", True, one, _baseline("one_hop_in")),
        Method("one_hop_out", "Baseline · 1단계 이웃 (내가 링크한)", "baseline", True, one, _baseline("one_hop_out")),
    ]


def compute_scores(ctx: Context, ms: list[Method], force: bool = False,
                   root: Path | None = None) -> dict[str, dict[int, np.ndarray]]:
    """{방법 키: {seed: 점수}}. 결과는 results/scores/<모드>/에 저장·재사용한다."""
    out = {}
    for m in ms:
        out[m.key] = {}
        for seed in m.seeds:
            name = m.key if len(m.seeds) == 1 else f"{m.key}_s{seed}"
            t0 = time.perf_counter()
            scores, _ = cached(name, lambda m=m, seed=seed: m.fn(ctx, seed), ctx.mode, force, root)
            out[m.key][seed] = scores
            print(f"  [score] {name:<22s} {time.perf_counter() - t0:6.1f}s", flush=True)
        if m.key == "lsh":   # 커버 집합 (seed 42)도 저장해 두고 하위 집합 채점에 쓴다
            fresh = config.SEED in ctx.extra.get("lsh_covered", {})   # 이번에 LSH를 새로 계산했는지

            def covered():
                if config.SEED not in ctx.extra.get("lsh_covered", {}):
                    _lsh(ctx, config.SEED)                           # 점수만 캐시에 있으면 다시 계산
                return ctx.extra["lsh_covered"][config.SEED].astype(float), {"seed": config.SEED}

            cov, _ = cached("lsh_covered", covered, ctx.mode, force or fresh, root)
            ctx.extra["lsh_covered_mask"] = cov.astype(bool)
    return out


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def _fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def score_table(ctx, ms, scores, boot) -> pd.DataFrame:
    """최종 비교표. 반복 방법은 seed 42로 신뢰구간, 5회로 평균 ± 표준편차."""
    rows = []
    y = ctx.target.y
    for m in ms:
        main = evaluate_detection(scores[m.key][config.SEED], ctx.target, boot)
        row = {"method": m.label, "key": m.key, "family": m.family, "uses_labels": m.uses_labels,
               "pr_auc": main["pr_auc"], "pr_auc_ci95": _fmt_ci(main["pr_auc_ci95"]),
               "roc_auc": main["roc_auc"], "roc_auc_ci95": _fmt_ci(main["roc_auc_ci95"]),
               "p@50": main["p@50"], "p@100": main["p@100"],
               "n_hosts": main["n_hosts"], "n_spam": main["n_spam"], "n_seeds": len(m.seeds)}
        if len(m.seeds) > 1:
            per = [metrics(scores[m.key][s][ctx.target.hostid], y) for s in m.seeds]
            for k in ("pr_auc", "roc_auc", "p@100"):
                v = np.array([p[k] for p in per])
                row[f"{k}_mean{len(m.seeds)}"] = float(v.mean())
                row[f"{k}_std{len(m.seeds)}"] = float(v.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(ctx, scores, boot) -> pd.DataFrame:
    """핵심 비교 4개 (확정안 6절). TrustRank는 ① 부호 반전 · SET1 seed로 고정."""
    one_hop = "one_hop_in" if FINAL["ppr_direction"] == "forward" else "one_hop_out"
    pairs = [
        ("H2", "tr_set1_trust", "ppr"),
        ("H3", "tr_set1_trust", "lsh"),
        ("H6", "ppr", one_hop),
        ("H5", "tr_set1_trust", "tr_domain_trust"),
        ("참고", "pagerank", "in_degree"),
    ]
    rows = []
    for hyp, a, b in pairs:
        c = compare(scores[a][config.SEED], scores[b][config.SEED], ctx.target, boot)
        rows.append({"hypothesis": hyp, "A": a, "B": b, "pr_auc_diff": c["diff"],
                     "ci95": _fmt_ci(c["ci95"]), "verdict": c["verdict"]})
    return pd.DataFrame(rows)


def ranking(ctx, scores, boot) -> pd.DataFrame:
    """Q1: PageRank 순위와 TrustRank 신뢰 순위의 구간별 스팸 비율 (높을수록 상위로 줄 세움)."""
    pr_rank = scores["pagerank"][config.SEED] * FINAL["pagerank_sign"]      # 원래 PageRank로 되돌림
    tr_rank = -scores["tr_set1_trust"][config.SEED]                         # 원래 신뢰 점수로 되돌림
    rows = []
    for name, s in (("pagerank", pr_rank), ("trustrank_set1", tr_rank)):
        for b in evaluate_ranking(s, ctx.target, config.N_BUCKETS, boot):
            rows.append({"ranking": name, **b, "ci95": _fmt_ci(b["ci95"])})
    return pd.DataFrame(rows)


def caught(ctx, scores) -> pd.DataFrame:
    """H3: 상위 100개로 잡은 스팸. 경계 동률은 seed 42 무작위 순서로 끊는다 (확정안 6절)."""
    keys = ["tr_set1_trust", "tr_domain_mass", "ppr", "lsh", "in_degree"]
    sets = {k: caught_spam(scores[k][config.SEED], ctx.target, config.CAUGHT_K) for k in keys}
    rows = []
    for k in keys:
        others = set().union(*(sets[o] for o in keys if o != k))
        rows.append({"method": k, "with": "", "caught": len(sets[k]), "only_this": len(sets[k] - others)})
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rows.append({"method": a, "with": b, "caught": len(sets[a] & sets[b]), "only_this": None})
    rows.append({"method": "합집합", "with": "", "caught": len(set().union(*sets.values())), "only_this": None})
    return pd.DataFrame(rows)


def subsets(ctx, ms, scores) -> pd.DataFrame:
    """하위 집합 채점: dangling / 비dangling / LSH 커버 (seed 42 점수, 하위 집합별 bootstrap)."""
    t = ctx.target
    masks = {"dangling": ctx.dangling[t.hostid], "non_dangling": ~ctx.dangling[t.hostid]}
    if "lsh_covered_mask" in ctx.extra:
        masks["lsh_covered"] = ctx.extra["lsh_covered_mask"][t.hostid]
    rows = []
    for sub, mask in masks.items():
        st = t.subset(mask)
        if st.n_spam == 0 or st.n_spam == st.n:
            continue
        sboot = bootstrap_indices(st)
        for m in ms:
            r = evaluate_detection(scores[m.key][config.SEED], st, sboot)
            rows.append({"subset": sub, "method": m.key, "n_hosts": st.n, "n_spam": st.n_spam,
                         "pr_auc": r["pr_auc"], "pr_auc_ci95": _fmt_ci(r["pr_auc_ci95"]),
                         "base_rate": st.n_spam / st.n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 그림: 결과를 만든 사람이 src/figures.py에 함수를 추가하고 여기서 부른다 (확정안 13절)
#   김현중: 잡은 스팸 겹침 (caught 표, H3)
#   박병언: 구간별 스팸 비율(ranking 표), seed·damping 곡선(run_sensitivity.py)
#   강민수: 최종 비교표, PR 곡선 (scores와 ctx.target 사용)
# check 모드 그림은 results/check/에, final 그림은 results/figures/에 저장한다.
# ---------------------------------------------------------------------------
def make_figures(ctx, scores, tables) -> list[Path]:
    from src import figures as F

    out_dir = config.FIGURES_DIR if ctx.mode == "final" else ctx.out_dir / "figures"
    prefix = "" if ctx.mode == "final" else "check_"
    fams = {m.key: m.family for m in methods()}
    seed_scores = {k: v[config.SEED] for k, v in scores.items()}
    title = None if ctx.mode == "final" else "CHECK MODE (SET1 fold 0) - not report numbers"
    return [
        F.fig_final_table(tables["table"], float(ctx.target.y.mean()), out_dir,
                          name=f"{prefix}fig_final_pr_auc", title=title),
        F.fig_pr_curves(seed_scores, ctx.target, families=fams, out_dir=out_dir,
                        name=f"{prefix}fig_pr_curves", title=title, y_max=0.5),
        F.fig_caught_overlap(tables["caught"], out_dir, name=f"{prefix}fig_caught_overlap", title=title),
    ]


# ---------------------------------------------------------------------------
# 분석 단계 (--analysis): SET1만 쓰는 표·그림. 모드와 무관하게 results/tables, results/figures에 저장.
# 새 분석 스크립트가 생기면 여기에 한 줄 추가한다.
# ---------------------------------------------------------------------------
def _script_main(path: Path):
    """experiments/*.py의 main(argv)를 불러온다. 파일이 아직 없으면 None."""
    import importlib.util

    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod.main


def analysis_steps() -> list[tuple[str, object]]:
    from src import eda

    steps = [
        ("EDA: 데이터 통계·degree·판정자 일치도·가정 확인", lambda: eda.main([])),
        ("LSH: 설정별 후보 쌍 수", lambda: LSH.main(["counts"])),
        ("LSH: 검증 5-fold로 설정 선택", lambda: LSH.main(["validate"])),
        ("LSH: S-curve와 recall", lambda: LSH.main(["scurve"])),
    ]
    for title, script in (("PPR: 검증 5-fold로 설정 선택", "cv_ppr.py"),
                          ("곡선: damping·seed 개수", "run_sensitivity.py")):
        fn = _script_main(ROOT / "experiments" / script)
        if fn is not None:
            steps.append((title, lambda fn=fn: fn([])))
        else:
            print(f"[skip] {title} ({script} 아직 없음)")
    return steps


def run_analysis(steps=None) -> None:
    for title, fn in (steps if steps is not None else analysis_steps()):
        t0 = time.perf_counter()
        print(f"\n===== {title} =====", flush=True)
        code = fn()
        if code not in (0, None):
            raise SystemExit(f"[err] '{title}' 단계가 실패했습니다 (종료 코드 {code})")
        print(f"----- {title}: {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="최종 설정으로 전체 실행")
    ap.add_argument("--final", action="store_true", help="SET2로 최종 채점 (설정 확정 후 한 번만)")
    ap.add_argument("--force", action="store_true", help="저장된 점수를 무시하고 다시 계산")
    ap.add_argument("--analysis", action="store_true", help="SET1 분석(EDA, LSH, PPR 검증, 곡선)도 먼저 실행")
    args = ap.parse_args(argv)
    mode = "final" if args.final else "check"

    if mode == "final" and src_dirty():
        print("[err] src/에 커밋 안 한 수정이 있어 최종 채점을 하지 않습니다. 커밋 후 다시 실행하세요.")
        return 1

    t0 = time.perf_counter()
    if args.analysis:
        run_analysis()
        print("\n===== 최종 설정으로 점수 → 채점 → 표 =====", flush=True)
    ctx = build_context(mode)
    print(f"[{mode}] seed 라벨: 스팸 {len(ctx.seed_spam)} / 정상 {len(ctx.seed_non)}, "
          f"채점 대상: {ctx.target.n}개 (스팸 {ctx.target.n_spam})")
    ms = methods()
    scores = compute_scores(ctx, ms, force=args.force)

    boot = bootstrap_indices(ctx.target)
    tables = {
        "table": score_table(ctx, ms, scores, boot),
        "comparisons": comparisons(ctx, scores, boot),
        "ranking": ranking(ctx, scores, boot),
        "caught": caught(ctx, scores),
        "subsets": subsets(ctx, ms, scores),
    }
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "final_" if mode == "final" else "check_"
    for name, df in tables.items():
        df.to_csv(ctx.out_dir / f"{prefix}{name}.csv", index=False)
    figs = make_figures(ctx, scores, tables)

    pd.set_option("display.width", 200)
    show = tables["table"][["method", "pr_auc", "pr_auc_ci95", "roc_auc", "p@100", "n_seeds"]]
    print(f"\n## 비교표 ({mode})\n{show.round(3).to_string(index=False)}")
    print(f"\n## 핵심 비교\n{tables['comparisons'].round(3).to_string(index=False)}")
    state = git_state()
    print(f"\n저장: {ctx.out_dir.relative_to(ROOT)}/{prefix}*.csv, 그림 {len(figs)}개 | "
          f"commit {str(state['commit'])[:7]} dirty={state['dirty']} | {time.perf_counter() - t0:.0f}s")
    if mode == "check":
        print("※ check 모드: SET1 fold 0으로 채점한 시험 결과입니다. 보고서 숫자가 아닙니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

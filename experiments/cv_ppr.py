"""PPR 5-fold 검증: 방향 2 × 점수 2 중 최종표에 넣을 설정 1개를 고른다 (확정안 4·5·6절).

담당: 강민수 (팀 작성 코드)

- SET1 5-fold (splits.json). 점수 계산에는 seed용 4개 fold 스팸만 쓰고, 검증 fold는 채점에만 쓴다.
- 검증 fold 채점 대상 = 검증 fold - 신뢰 도메인. SET2는 쓰지 않는다.
- 선택 규칙: 최고 설정과 기본값(링크 방향, 점수 ① 그대로)의 평균 PR-AUC 차이가
  fold 간 표준편차보다 작으면 기본값 유지.
- 참고용으로 같은 방향의 1단계 이웃 baseline도 같은 fold로 채점한다 (H6 미리보기, 선택에는 안 씀).

실행 (저장소 루트에서): python3 experiments/cv_ppr.py
결과: results/tables/cv_ppr.csv, results/tables/cv_ppr_selection.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.baselines import one_hop_scores  # noqa: E402
from src.evaluate import cross_validate, select_setting  # noqa: E402
from src.load_graph import load_graph, load_labels  # noqa: E402
from src.ppr import DIRECTIONS, SCORE_TYPES, ppr_score  # noqa: E402
from src.splits import load_splits  # noqa: E402

DEFAULT = "ppr_forward_raw"           # 확정안 5절 기본값: 링크 방향, 점수 ① 그대로
ONE_HOP_PAIR = {"forward": "in", "reverse": "out"}   # 같은 방향의 1단계 이웃 (확정안 8.5절)


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"src_hash": run("rev-parse", "HEAD:src"), "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "src"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=None, help="원본 데이터 폴더 (기본: data/raw)")
    ap.add_argument("--splits", type=Path, default=config.SPLITS_PATH)
    ap.add_argument("--cache-dir", type=Path, default=None, help="그래프 캐시 폴더 (기본: data/processed)")
    ap.add_argument("--out-dir", type=Path, default=config.TABLES_DIR)
    ap.add_argument("--damping", type=float, default=config.DAMPING)
    args = ap.parse_args(argv)

    A, hostnames = load_graph(raw_dir=args.raw_dir, cache_dir=args.cache_dir)
    train, _ = load_labels(raw_dir=args.raw_dir)          # SET2는 읽기만 하고 쓰지 않음
    folds = load_splits(args.splits)
    info = git_info()
    if info["dirty"]:
        print("[warn] src/에 커밋 안 한 수정이 있음 (dirty). 결과 공유 전에 커밋할 것")

    rows, cvs = [], {}
    for d in DIRECTIONS:
        for s in SCORE_TYPES:
            name = f"ppr_{d}_{s}"
            extra = []

            def score_fn(spam, non, d=d, s=s, extra=extra):
                res = ppr_score(A, spam, args.damping, d, s)
                extra.append(res.info)
                return res.scores

            t0 = time.time()
            cv = cross_validate(score_fn, folds, train, hostnames)
            cvs[name] = cv
            rows.append(_row(name, cv, "후보", time.time() - t0, extra))
            print(f"[ok ] {name}: PR-AUC {cv['pr_auc_mean']:.4f} ± {cv['pr_auc_std']:.4f}")

    for d, nb in ONE_HOP_PAIR.items():
        name = f"one_hop_{nb}"
        t0 = time.time()
        cv = cross_validate(lambda spam, non, nb=nb: one_hop_scores(A, spam, non, nb)[0],
                            folds, train, hostnames)
        rows.append(_row(name, cv, f"참고 ({d} PPR과 짝)", time.time() - t0, []))
        print(f"[ref] {name}: PR-AUC {cv['pr_auc_mean']:.4f} ± {cv['pr_auc_std']:.4f}")

    sel = select_setting(cvs, default=DEFAULT)
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "cv_ppr.csv", index=False, float_format="%.4f")
    sel.update(damping=args.damping, created=datetime.now().isoformat(timespec="seconds"), **info)
    (args.out_dir / "cv_ppr_selection.json").write_text(
        json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(table.drop(columns=["seconds"]).to_string(index=False))
    print(f"\n선택: {sel['chosen']}  ({sel['reason']}; 최고 {sel['best']}, "
          f"차이 {sel['gap']:+.4f}, 최고 설정 표준편차 {sel['best_std']:.4f})")
    print(f"저장: {args.out_dir / 'cv_ppr.csv'}, {args.out_dir / 'cv_ppr_selection.json'}")
    return 0


def _row(name: str, cv: dict, role: str, seconds: float, infos: list[dict]) -> dict:
    row = {"setting": name, "role": role}
    for key in ("pr_auc", "roc_auc", "p@50", "p@100"):
        row[f"{key}_mean"] = cv[f"{key}_mean"]
        row[f"{key}_std"] = cv[f"{key}_std"]
    if infos:
        row["spreading_seeds_mean"] = sum(i["n_spreading_seeds"] for i in infos) / len(infos)
        row["seeds_mean"] = sum(i["n_seeds"] for i in infos) / len(infos)
        row["n_iter_max"] = max((i.get("n_iter") or 0) for i in infos)
    row["seconds"] = round(seconds, 1)
    return row


if __name__ == "__main__":
    sys.exit(main())

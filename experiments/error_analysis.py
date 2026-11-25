"""Error analysis 실행: 놓친 스팸·오탐 정상의 특징 표와 사례 표.

담당: 강민수 (팀 작성 코드). run_main.py와 같은 채점 대상·같은 점수(results/scores/ 캐시)를 쓴다.

    python3 experiments/error_analysis.py            check 모드 (SET1 fold 0, SET2 안 건드림)
    python3 experiments/error_analysis.py --final    최종 모드 (run_main.py --final을 먼저 돌린 뒤)

결과 (check → results/check/, final → results/tables/)
    <모드>_error_summary.csv    그룹별 특징 (놓친 스팸 / 잡은 스팸 / 오탐 정상 / 나머지 정상)
    <모드>_error_per_method.csv 방법별 잡은 스팸·오탐 정상·그 방법만 잡은 스팸
    <모드>_error_suffix.csv     그룹별 주소 끝(co.uk, org.uk 등) 비율
    <모드>_error_cases.csv      사례 표 (보고서 T4)
    <모드>_error_hosts.csv      host별 전체 표 (check 모드만. 최종 모드는 SET2 hostname 목록이라 저장 안 함)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                                   # noqa: E402
from src import error_analysis as EA                     # noqa: E402
from src.load_graph import in_degree, load_labels, out_degree  # noqa: E402
from src.results_io import src_dirty                     # noqa: E402


def _run_main():
    spec = importlib.util.spec_from_file_location("run_main", ROOT / "experiments" / "run_main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_main"] = mod          # dataclass가 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="error analysis")
    ap.add_argument("--final", action="store_true", help="최종 모드 (SET2). run_main.py --final 뒤에 실행")
    ap.add_argument("--k", type=int, default=config.CAUGHT_K)
    args = ap.parse_args(argv)
    mode = "final" if args.final else "check"
    if mode == "final" and src_dirty():
        print("[err] src/에 커밋 안 한 수정이 있어 최종 분석을 하지 않습니다.")
        return 1

    rm = _run_main()
    ctx = rm.build_context(mode)
    ms = [m for m in rm.methods() if m.key in EA.ALGO_KEYS]
    scores = {k: v[config.SEED] for k, v in rm.compute_scores(ctx, ms).items()}   # 캐시가 있으면 불러옴

    train, test = load_labels()
    lab = test if mode == "final" else train
    spamicity = dict(zip(lab["hostid"], lab["spamicity"]))
    df = EA.host_table(scores, ctx.target, ctx.hostnames, in_degree(ctx.A), out_degree(ctx.A),
                       k=args.k, covered=ctx.extra.get("lsh_covered_mask"), spamicity=spamicity)

    tables = {"error_summary": EA.summarize(df), "error_per_method": EA.per_method(df),
              "error_suffix": EA.suffix_table(df), "error_cases": EA.cases(df)}
    if mode == "check":
        tables["error_hosts"] = df
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(ctx.out_dir / f"{mode}_{name}.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print(f"[{mode}] 채점 대상 {ctx.target.n}개 (스팸 {ctx.target.n_spam}), 상위 {args.k}개 기준, "
          f"방법 {len(df.attrs['keys'])}개, 오탐 기준 {df.attrs['false_alarm_min']}개 이상")
    print("\n## 그룹별 특징\n" + tables["error_summary"].round(3).to_string(index=False))
    print("\n## 방법별\n" + tables["error_per_method"].to_string(index=False))
    print("\n## 사례\n" + tables["error_cases"].round(3).to_string(index=False))
    print(f"\n저장: {ctx.out_dir.relative_to(ROOT)}/{mode}_error_*.csv")
    if mode == "check":
        print("※ check 모드: SET1 fold 0 결과입니다. 보고서 숫자가 아닙니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

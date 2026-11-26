"""Error analysis: 모든 방법이 놓친 스팸, 여러 방법이 잘못 잡은 정상 사이트의 특징 (확정안 9절).

담당: 강민수 (팀 작성 코드). 외부 라이브러리: numpy, pandas

정의 (확정안 6절 "잡았다"와 같음)
- 잡았다(flag) = 채점 대상에서 그 방법의 상위 K(=100)개 안에 듦. 경계 동률은 seed 42 무작위 순서로 끊는다.
- 놓친 스팸   = 스팸인데 어느 방법의 상위 K에도 안 든 사이트
- 잡은 스팸   = 스팸이고 한 방법 이상이 잡은 사이트
- 오탐 정상   = 정상인데 여러 계열이 잡은 사이트 (기본: 계열 3개 중 2개 이상)
  계열 = PageRank 계열(PageRank·TrustRank 4줄) / PPR / LSH. PageRank·TrustRank는 상위 100개가 거의 같아서
  방법 수로 세면 한 계열이 혼자 과반을 채워 오탐이 부풀려진다.
- 나머지 정상 = 그 밖의 정상 사이트
분석 대상 방법은 수업 알고리즘만 (PageRank, TrustRank 4줄, PPR, LSH). baseline은 넣지 않는다.

이 파일은 점수와 채점 대상만 받는 순수 함수다. 실행은 experiments/error_analysis.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src import config
from src.evaluate import EvalTarget, top_k_hosts

ALGO_KEYS = ("pagerank", "tr_set1_trust", "tr_set1_mass", "tr_domain_trust", "tr_domain_mass", "ppr", "lsh")
FAMILIES = {
    "pr_family": ("pagerank", "tr_set1_trust", "tr_set1_mass", "tr_domain_trust", "tr_domain_mass"),
    "ppr": ("ppr",),
    "lsh": ("lsh",),
}
GROUPS = ("missed_spam", "caught_spam", "false_alarm", "other_nonspam")


def suffix_of(host: str) -> str:
    """주소의 마지막 두 칸 (예: www.example.co.uk -> co.uk, x.ac.uk -> ac.uk)."""
    return ".".join(host.strip().lower().rstrip(".").split(".")[-2:])


def rank_percentile(scores: np.ndarray, target: EvalTarget, seed: int = config.SEED) -> np.ndarray:
    """채점 대상 안에서의 순위 백분위 (0 = 가장 의심, 1 = 가장 덜 의심). 동률은 seed 순서로 끊는다."""
    order = top_k_hosts(scores, target, k=target.n, seed=seed)       # 의심 높은 순 hostid
    pos = {int(h): i for i, h in enumerate(order)}
    return np.array([pos[int(h)] / max(target.n - 1, 1) for h in target.hostid])


def host_table(scores: dict, target: EvalTarget, hostnames: Sequence[str], in_deg: np.ndarray,
               out_deg: np.ndarray, keys: Sequence[str] = ALGO_KEYS, k: int = config.CAUGHT_K,
               covered: np.ndarray | None = None, spamicity: dict | None = None,
               false_alarm_min: int | None = None, families: dict | None = None) -> pd.DataFrame:
    """채점 대상 host마다: 특징, 방법별 순위 백분위, 잡은 방법 수·계열 수, 그룹.

    families: {계열 이름: 방법 키들}. 기본 FAMILIES. 오탐은 잡은 계열 수로 정한다.
    """
    keys = [kk for kk in keys if kk in scores]
    if not keys:
        raise ValueError("분석할 방법 점수가 없음")
    families = FAMILIES if families is None else families
    need = false_alarm_min or 2                                      # 계열 3개 중 2개 이상
    ids = target.hostid
    df = pd.DataFrame({
        "hostid": ids,
        "hostname": [hostnames[i] for i in ids],
        "suffix": [suffix_of(hostnames[i]) for i in ids],
        "spam": target.y.astype(int),
        "in_degree": in_deg[ids],
        "out_degree": out_deg[ids],
        "dangling": out_deg[ids] == 0,
    })
    if covered is not None:
        df["lsh_covered"] = np.asarray(covered, dtype=bool)[ids]
    if spamicity is not None:
        df["spamicity"] = [spamicity.get(int(i), np.nan) for i in ids]
    caught = np.zeros(len(ids), dtype=int)
    for key in keys:
        top = set(top_k_hosts(scores[key], target, k).tolist())
        flag = np.array([int(h) in top for h in ids])
        df[f"top{k}_{key}"] = flag
        df[f"pct_{key}"] = rank_percentile(scores[key], target)
        caught += flag
    df["n_methods_flag"] = caught                                   # 참고용
    fam_flag = {}
    for fam, ks in families.items():
        cols = [f"top{k}_{kk}" for kk in ks if kk in keys]
        if cols:
            fam_flag[fam] = df[cols].any(axis=1)
            df[f"fam_{fam}"] = fam_flag[fam]
    if not fam_flag:
        raise ValueError("계열에 들어가는 방법이 없음")
    df["n_families_flag"] = sum(f.astype(int) for f in fam_flag.values())
    df["best_pct"] = df[[f"pct_{key}" for key in keys]].min(axis=1)
    df["group"] = np.select(
        [(df.spam == 1) & (caught == 0), (df.spam == 1), (df.n_families_flag >= need)],
        ["missed_spam", "caught_spam", "false_alarm"], default="other_nonspam")
    df.attrs.update(keys=keys, k=k, false_alarm_min=need, families=list(fam_flag))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """그룹별 특징 요약 (보고서 Error analysis 표)."""
    rows = []
    for g in GROUPS:
        d = df[df.group == g]
        if d.empty:
            rows.append({"group": g, "n_hosts": 0})
            continue
        row = {
            "group": g,
            "n_hosts": len(d),
            "in_degree_median": float(d.in_degree.median()),
            "out_degree_median": float(d.out_degree.median()),
            "dangling_rate": float(d.dangling.mean()),
            "no_inlink_rate": float((d.in_degree == 0).mean()),
            "best_pct_median": float(d.best_pct.median()),
            "n_methods_flag_mean": float(d.n_methods_flag.mean()),
            "n_families_flag_mean": float(d.n_families_flag.mean()),
        }
        if "lsh_covered" in d:
            row["lsh_covered_rate"] = float(d.lsh_covered.mean())
        if "spamicity" in d:
            row["spamicity_mean"] = float(d.spamicity.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def per_method(df: pd.DataFrame) -> pd.DataFrame:
    """방법별로 잡은 스팸 수와 잘못 잡은 정상 수 (상위 K 기준), 그 방법만 잡은 스팸 수."""
    keys, k = df.attrs["keys"], df.attrs["k"]
    spam = df[df.spam == 1]
    rows = []
    for key in keys:
        col = f"top{k}_{key}"
        others = [f"top{k}_{o}" for o in keys if o != key]
        only = spam[spam[col] & ~spam[others].any(axis=1)]
        rows.append({"method": key, "spam_caught": int(spam[col].sum()),
                     "nonspam_flagged": int(df[(df.spam == 0) & df[col]].shape[0]),
                     "spam_only_this": len(only)})
    return pd.DataFrame(rows)


def suffix_table(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """그룹별 주소 끝 두 칸(co.uk, org.uk 등) 비율. 그룹마다 많은 순 top개."""
    rows = []
    for g in GROUPS:
        d = df[df.group == g]
        if d.empty:
            continue
        vc = d.suffix.value_counts()
        for suf, c in vc.head(top).items():
            rows.append({"group": g, "suffix": suf, "n": int(c), "share": c / len(d)})
    return pd.DataFrame(rows)


def cases(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """보고서 사례 표 (T4): 가장 덜 의심받은 놓친 스팸 n개, 가장 많은 방법이 잡은 오탐 정상 n개."""
    cols = ["group", "hostname", "in_degree", "out_degree", "dangling", "n_methods_flag", "n_families_flag",
            "best_pct"]
    cols += [c for c in ("lsh_covered", "spamicity") if c in df]
    missed = df[df.group == "missed_spam"].sort_values("best_pct", ascending=False).head(n)
    fa = df[df.group == "false_alarm"].sort_values(["n_families_flag", "n_methods_flag", "best_pct"],
                                                 ascending=[False, False, True]).head(n)
    return pd.concat([missed, fa])[cols].reset_index(drop=True)

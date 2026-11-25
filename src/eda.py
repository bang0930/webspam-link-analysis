"""데이터 탐색 (EDA): 보고서 2. Data 섹션의 표와 그림.

담당: 김현중 (팀 작성 코드). 외부 라이브러리: numpy, pandas, scipy

만드는 것 (확정안 8절 T1, F1, 2절 데이터 한계, 9절 가정 확인)
- data_stats        : 데이터 통계표 T1 (host·edge·dangling·라벨 수·채점 대상 수)
- degree_by_label   : 스팸 vs 정상의 in/out-degree 요약 (SET1만)
- assessor_agreement: 판정자 간 일치도 (SET1, SET2 따로. 라벨 품질 기술용)
- link_assumption   : TrustRank·PPR 가정 확인 — 정상→스팸, 스팸→스팸 링크 비율 (SET1만)

라벨 사용 규칙
- 방법 설계에 영향을 줄 수 있는 분석(degree, 링크 가정)은 SET1만 쓴다.
- SET2는 개수와 판정자 일치도처럼 데이터 자체를 기술하는 숫자에만 쓴다 (점수·설정 선택에 쓰지 않음).

실행: python -m src.eda   -> results/tables/eda_*.csv, results/figures/fig_degree_distribution.png
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config
from src.load_graph import dangling_mask, in_degree, out_degree

VALID_JUDGMENTS = ("N", "S", "B")      # nonspam, spam, borderline (U = unknown은 제외)


# ---------------------------------------------------------------------------
# T1 데이터 통계표
# ---------------------------------------------------------------------------
def data_stats(A: sp.spmatrix, hostnames, train: pd.DataFrame, test: pd.DataFrame,
               eval_ids: np.ndarray, trust_mask: np.ndarray) -> pd.DataFrame:
    """보고서 T1. 반환: (항목, 값) 표."""
    n = A.shape[0]
    ind, outd = in_degree(A), out_degree(A)
    rows = [
        ("host 수", n),
        ("방향 있는 edge 수 (0·1, 자기 자신 링크 제외)", int(A.nnz)),
        ("평균 out-degree", round(float(outd.mean()), 2)),
        ("dangling host (나가는 링크 0)", int(dangling_mask(A).sum())),
        ("들어오는 링크 0", int((ind == 0).sum())),
        ("완전히 고립된 host", int(((ind == 0) & (outd == 0)).sum())),
        ("신뢰 도메인 host", int(trust_mask.sum())),
    ]
    for name, df in (("SET1", train), ("SET2", test)):
        vc = df["label"].value_counts()
        for lab in ("nonspam", "spam", "undecided"):
            rows.append((f"{name} {lab}", int(vc.get(lab, 0))))
    lab2 = dict(zip(test["hostid"], test["label"]))
    rows += [
        ("채점 대상 eval_set host", int(len(eval_ids))),
        ("채점 대상 eval_set 스팸", int(sum(lab2[int(i)] == "spam" for i in eval_ids))),
    ]
    return pd.DataFrame(rows, columns=["항목", "값"], dtype=object)   # 정수는 정수로 보이게


# ---------------------------------------------------------------------------
# degree: 스팸 vs 정상 (SET1만)
# ---------------------------------------------------------------------------
def degree_by_label(A: sp.spmatrix, train: pd.DataFrame) -> pd.DataFrame:
    """SET1 스팸·정상의 in/out-degree 요약 (평균, 중앙값, 0인 비율, 상위 10%)."""
    ind, outd = in_degree(A), out_degree(A)
    rows = []
    for lab in ("nonspam", "spam"):
        ids = train.loc[train["label"] == lab, "hostid"].to_numpy()
        for kind, deg in (("in", ind), ("out", outd)):
            d = deg[ids]
            rows.append({"label": lab, "degree": kind, "n_hosts": len(ids),
                         "mean": round(float(d.mean()), 2), "median": float(np.median(d)),
                         "p90": float(np.percentile(d, 90)), "zero_rate": round(float((d == 0).mean()), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 판정자 간 일치도
# ---------------------------------------------------------------------------
def parse_assessments(text: str) -> list[str]:
    """'j1:N,j2:S,j3:U' -> ['N', 'S', 'U']"""
    if not isinstance(text, str) or not text.strip():
        return []
    return [tok.split(":", 1)[1] for tok in text.split(",") if ":" in tok]


def assessor_agreement(labels: pd.DataFrame) -> dict:
    """판정자 간 일치도.

    유효 판정 = N, S, B (U = unknown 제외). 유효 판정이 2개 이상인 host만 본다.
    - 만장일치 비율: 유효 판정이 모두 같은 host 비율
    - 일치율 Po: host마다 "판정 쌍 중 같은 쌍의 비율"을 구해 host 평균 (Fleiss kappa 방식).
      쌍을 전부 합쳐 세면 판정자가 많은 host(대개 의견이 갈린 host)가 과하게 반영되므로 host 단위로 평균한다.
    - 우연 일치율 Pe = Σ_c (전체 판정 중 c의 비율)²
    - kappa = (Po − Pe) / (1 − Pe)  (판정자 수가 host마다 다른 경우의 Fleiss kappa)
    """
    per_host = [[j for j in parse_assessments(t) if j in VALID_JUDGMENTS] for t in labels["assessments"]]
    multi = [js for js in per_host if len(js) >= 2]
    counts = {c: 0 for c in VALID_JUDGMENTS}
    host_agree = []
    for js in multi:
        pairs = list(combinations(js, 2))
        host_agree.append(sum(a == b for a, b in pairs) / len(pairs))
        for j in js:
            counts[j] += 1
    n_j = sum(counts.values())
    total = len(host_agree)
    po = float(np.mean(host_agree)) if host_agree else float("nan")
    pe = sum((c / n_j) ** 2 for c in counts.values()) if n_j else float("nan")
    return {
        "hosts": len(per_host),
        "hosts_with_2plus_valid": len(multi),
        "mean_valid_judgments": round(float(np.mean([len(j) for j in per_host])), 2) if per_host else 0.0,
        "unanimous_rate": round(float(np.mean([len(set(js)) == 1 for js in multi])), 3) if multi else float("nan"),
        "pairwise_agreement": round(po, 3),
        "chance_agreement": round(pe, 3),
        "kappa": round((po - pe) / (1 - pe), 3) if total and pe < 1 else float("nan"),
        "share_N": round(counts["N"] / n_j, 3) if n_j else float("nan"),
        "share_S": round(counts["S"] / n_j, 3) if n_j else float("nan"),
        "share_B": round(counts["B"] / n_j, 3) if n_j else float("nan"),
    }


# ---------------------------------------------------------------------------
# 가정 확인: 정상은 스팸에 잘 링크하지 않는다 (TrustRank, 11강 p73) / 스팸끼리 링크한다 (PPR)
# ---------------------------------------------------------------------------
def link_assumption(A: sp.spmatrix, train: pd.DataFrame) -> pd.DataFrame:
    """SET1 라벨 host 사이의 링크만 세어, 출발 라벨별로 도착이 스팸인 비율을 본다.

    비교 기준: SET1 라벨 host 중 스팸 비율 (출발과 무관하게 무작위로 링크한다면 기대되는 값).
    """
    lab = train.loc[train["label"].isin(["spam", "nonspam"])]
    n = A.shape[0]
    is_spam = np.zeros(n, dtype=bool)
    is_lab = np.zeros(n, dtype=bool)
    is_lab[lab["hostid"]] = True
    is_spam[lab.loc[lab["label"] == "spam", "hostid"]] = True
    base = is_spam[is_lab].mean()

    C = A.tocoo()
    m = is_lab[C.row] & is_lab[C.col]
    src_spam, dst_spam = is_spam[C.row[m]], is_spam[C.col[m]]
    rows = []
    for name, sel in (("nonspam", ~src_spam), ("spam", src_spam)):
        k = int(sel.sum())
        to_spam = int(dst_spam[sel].sum())
        rows.append({"source": name, "links_to_labeled": k, "to_spam": to_spam, "to_nonspam": k - to_spam,
                     "to_spam_rate": round(to_spam / k, 4) if k else float("nan"),
                     "base_spam_rate": round(float(base), 4),
                     "lift": round((to_spam / k) / base, 2) if k else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 실행: python -m src.eda
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    from src import figures
    from src.load_graph import load_graph, load_labels
    from src.splits import is_trust_domain, load_eval_set

    A, hostnames = load_graph()
    train, test = load_labels()
    eval_ids = load_eval_set(config.EVAL_SET_PATH)
    trust = np.array([is_trust_domain(h) for h in hostnames])

    out = config.TABLES_DIR
    out.mkdir(parents=True, exist_ok=True)
    t1 = data_stats(A, hostnames, train, test, eval_ids, trust)
    deg = degree_by_label(A, train)
    agr = pd.DataFrame([{"set": "SET1", **assessor_agreement(train)},
                        {"set": "SET2", **assessor_agreement(test)}])
    asm = link_assumption(A, train)
    t1.to_csv(out / "eda_data_stats.csv", index=False)
    deg.to_csv(out / "eda_degree_by_label.csv", index=False)
    agr.to_csv(out / "eda_assessor_agreement.csv", index=False)
    asm.to_csv(out / "eda_link_assumption.csv", index=False)
    fig = figures.fig_degree_distribution(A, train)

    pd.set_option("display.width", 160)
    for title, df in (("T1 데이터 통계", t1), ("degree (SET1)", deg),
                      ("판정자 간 일치도", agr), ("링크 가정 확인 (SET1)", asm)):
        print(f"\n## {title}\n{df.to_string(index=False)}")
    print(f"\n저장: results/tables/eda_*.csv, {fig.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

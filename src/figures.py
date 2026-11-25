"""보고서 그림 (results/figures/*.png). 보고서 그림은 전부 이 파일에서 만든다 (노트북에 의존하지 않음).

파일 관리: 김현중. 그림 함수는 그 결과를 만든 사람이 작성한다 (확정안 13절).
    김현중: EDA 그림, LSH S-curve
    박병언: seed·damping 곡선, PageRank vs TrustRank 구간별 그림
    강민수: 최종 비교표, PR 곡선

규칙
- 함수 이름은 fig_<내용>, 반환값은 저장한 파일 경로.
- 글자는 영어로 (한글 글꼴이 컴퓨터마다 달라 깨질 수 있음). 보고서 캡션은 한국어로 따로 쓴다.
- out_dir를 주지 않으면 results/figures/에 저장한다 (테스트는 임시 폴더를 넘긴다).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # 화면 없이 파일로만 그림
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src import config                   # noqa: E402
from src.load_graph import in_degree, out_degree  # noqa: E402

COLORS = {"nonspam": "#2b6cb0", "spam": "#c53030", "theory": "#4a5568"}
DPI = 200


def savefig(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 김현중
# ---------------------------------------------------------------------------
def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """degree ≥ 1인 값의 CCDF: x 이상인 비율 (0인 host는 범례에 비율로 따로 적는다)."""
    v = np.sort(values[values > 0])
    if len(v) == 0:
        return np.array([1.0]), np.array([0.0])
    x, idx = np.unique(v, return_index=True)
    return x, 1.0 - idx / len(v)


def fig_degree_distribution(A, train, out_dir: Path | None = None) -> Path:
    """F1: SET1 스팸 vs 정상의 in/out-degree CCDF (로그-로그)."""
    ind, outd = in_degree(A), out_degree(A)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (kind, deg) in zip(axes, (("in-degree", ind), ("out-degree", outd))):
        for lab in ("nonspam", "spam"):
            d = deg[train.loc[train["label"] == lab, "hostid"].to_numpy()]
            x, y = _ccdf(d)
            ax.step(x, y, where="post", color=COLORS[lab],
                    label=f"{lab} (n={len(d)}, zero={np.mean(d == 0):.0%})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{kind} (≥ 1)")
        ax.set_title(kind)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction of hosts with degree ≥ x")
    return savefig(fig, "fig_degree_distribution", out_dir)


def fig_lsh_scurve(curve_df, out_dir: Path | None = None) -> Path:
    """F6: (b, r)별 S-curve — 실제 후보 비율(점) vs 이론 1 − (1 − t^r)^b (선). 강의 4강 p55-57."""
    from src.lsh import jaccard_threshold, theoretical_curve

    settings = curve_df[["bands", "rows"]].drop_duplicates().itertuples(index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    t = np.linspace(0, 1, 201)
    for (b, r), color in zip(settings, ("#2b6cb0", "#c05621", "#2f855a")):
        d = curve_df[(curve_df.bands == b) & (curve_df.rows == r)]
        mid = (d.jaccard_lo + d.jaccard_hi) / 2
        ax.plot(t, theoretical_curve(t, b, r), color=color, lw=1.2, alpha=0.8)
        ax.scatter(mid, d.candidate_rate, color=color, s=14, label=f"b={b}, r={r} (thr {jaccard_threshold(b, r):.2f})")
    ax.set_xlabel("Jaccard similarity of in-link sets")
    ax.set_ylabel("P(candidate pair)")
    ax.set_title("LSH S-curve: observed (dots) vs theory (lines)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return savefig(fig, "fig_lsh_scurve", out_dir)


# ---------------------------------------------------------------------------
# 박병언: seed·damping 곡선, PageRank vs TrustRank 구간별 그림 (작성 예정)
# 강민수: 최종 비교표, PR 곡선 (작성 예정)
# ---------------------------------------------------------------------------

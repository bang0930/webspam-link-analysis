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


CAUGHT_LABELS = {
    "tr_set1_trust": "TrustRank\n(SET1 seeds)",
    "tr_domain_mass": "TrustRank\n(trusted domains,\nSpam Mass)",
    "ppr": "PPR\n(spam seeds)",
    "lsh": "MinHash/LSH",
    "in_degree": "in-degree\n(baseline)",
}


def fig_caught_overlap(caught_df, out_dir: Path | None = None,
                       name: str = "fig_caught_overlap", title: str | None = None) -> Path:
    """H3: 방법별로 상위 100개 안에 잡은 스팸 수와, 그중 그 방법만 잡은 수 (run_main.caught 표)."""
    d = caught_df[caught_df["with"].fillna("") == ""]
    d = d[d["method"].isin(CAUGHT_LABELS)]
    union = caught_df.loc[caught_df["method"] == "합집합", "caught"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x, d["caught"], color="#a0aec0", label="caught in top 100")
    ax.bar(x, d["only_this"].astype(float), color="#2b6cb0", label="caught only by this method")
    for xi, (c, o) in enumerate(zip(d["caught"], d["only_this"].astype(int))):
        ax.text(xi, c + 0.3, f"{c} ({o})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [CAUGHT_LABELS[m] for m in d["method"]], fontsize=8)
    ax.set_ylabel("spam hosts")
    extra_title = title
    title = "Spam caught in each method's top 100"
    if len(union):
        title += f"  (union = {int(union.iloc[0])})"
    if extra_title:
        title = f"{extra_title}\n{title}"
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(d["caught"].max() * 1.45, 1))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    return savefig(fig, name, out_dir)


# ---------------------------------------------------------------------------
# 박병언: seed 개수, damping 곡선, PageRank vs TrustRank 구간별 그림
# ---------------------------------------------------------------------------


def fig_rank_buckets(table, base_rate, out_dir=None, name="fig_rank_buckets", title=None):
    """담당: 박병언 (팀 작성 코드). 확정안 6절, 8.1절, 강의 11강 p73-78.

    두 순위의 구간별 스팸 비율과 도메인 bootstrap 구간을 나란히 그린다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, label in zip(axes, ("pagerank", "trustrank_set1"), ("PageRank", "TrustRank (SET1 seeds)")):
        d = table[table.ranking == key].sort_values("bucket")
        cis = np.array([_parse_ci(v) for v in d.ci95])
        ax.vlines(d.bucket, cis[:, 0], cis[:, 1], color="#2b6cb0")
        ax.plot(d.bucket, d.spam_rate, "o-", color="#2b6cb0")
        ax.axhline(base_rate, ls="--", color="#555555", label=f"Eval spam share = {base_rate:.3f}")
        ax.set(title=label, xlabel="Rank bucket (1 = highest)", xticks=range(1, 11))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Spam fraction, 95% domain-bootstrap CI")
    if title:
        fig.suptitle(title)
    return savefig(fig, name, out_dir)


def _fig_sensitivity(table, xcolumn, name, out_dir, check=False, seed_methods=False):
    """민감도 요약 표를 그린다. 선택에 사용하지 않는 분석임을 그림에 표시한다."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (split, method, seed_method), d in table.groupby(["split", "method", "seed_method"]):
        d = d.sort_values(xcolumn)
        x = d[xcolumn].to_numpy(dtype=float)
        y = d.pr_auc_mean.to_numpy(dtype=float)
        stdcol = "pr_auc_fold_std" if name == "fig_damping_curve" else "pr_auc_std"
        sd = d[stdcol].to_numpy(dtype=float)
        label = f"{split}, {method}"
        if seed_methods:
            label += f", {seed_method}"
        ax.plot(x, y, "o-" if split == "validation" else "s--", label=label)
        ax.fill_between(x, y - sd, y + sd, alpha=.12)
    if xcolumn == "n_seeds":
        ax.set_xscale("log")
        ax.set_xlabel("Actual seed count (log scale)")
    else:
        ax.set_xlabel("Damping")
    ax.set_ylabel("PR-AUC (average precision)")
    ax.set_title(("CHECK MODE - not report numbers\n" if check else "") + "Analysis only: no selection from SET2")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
    note = ("All seeds: same set in every repeat, repeat SD = 0. Validation x = mean fold seed count."
            if name == "fig_seed_curve" else
            "Bands: fold SD. PPR damping not included." if name == "fig_damping_curve" else
            "Bands: repeat SD; PageRank-top uses one run. No method selection.")
    fig.text(.5, .01, note, ha="center", fontsize=7)
    fig.subplots_adjust(bottom=.18)
    return savefig(fig, name, out_dir)


def fig_seed_curve(table, out_dir=None, check=False):
    """담당: 박병언 (팀 작성 코드). seed 개수 곡선, 확정안 6절, 8.2절, 강의 11강 p77-78."""
    return _fig_sensitivity(table, "n_seeds", "fig_seed_curve", out_dir, check)


def fig_seed_methods(table, out_dir=None, check=False):
    """담당: 박병언 (팀 작성 코드). seed 방식 비교, 확정안 8.2절, 9절, 강의 11강 p77-78."""
    return _fig_sensitivity(table, "n_seeds", "fig_seed_methods", out_dir, check, True)


def fig_damping_curve(table, out_dir=None, check=False):
    """담당: 박병언 (팀 작성 코드). damping 곡선, 확정안 6절, 9절, 강의 11강 p17-18, p76."""
    return _fig_sensitivity(table, "damping", "fig_damping_curve", out_dir, check)


# ---------------------------------------------------------------------------
# 강민수: 최종 비교표, PR 곡선
# ---------------------------------------------------------------------------
# 방법 묶음별 색 (분류용 색 순서 고정, 무작위 순환 안 함). baseline은 회색으로 물러나게.
FAMILY_COLORS = {"trustrank": "#257de0", "ppr": "#f47230", "lsh": "#00b894",
                 "pagerank": "#f2a600", "baseline": "#718096"}
FAMILY_ORDER = ("trustrank", "ppr", "lsh", "pagerank", "baseline")

# 그림 글자는 영어 (run_main 방법 키 -> 그림용 이름)
METHOD_NAMES = {
    "pagerank": "PageRank (reference)",
    "tr_set1_trust": "TrustRank · SET1 seeds · trust",
    "tr_set1_mass": "TrustRank · SET1 seeds · spam mass",
    "tr_domain_trust": "TrustRank · trusted-domain seeds · trust",
    "tr_domain_mass": "TrustRank · trusted-domain seeds · spam mass",
    "ppr": "PPR · spam seeds",
    "lsh": "MinHash/LSH",
    "random": "Baseline · random",
    "in_degree": "Baseline · in-degree",
    "out_degree": "Baseline · out-degree",
    "one_hop_in": "Baseline · 1-hop (in-links)",
    "one_hop_out": "Baseline · 1-hop (out-links)",
}


def _parse_ci(ci) -> tuple[float, float]:
    """run_main 표의 "[0.081, 0.142]" 문자열 또는 (lo, hi)."""
    if isinstance(ci, str):
        lo, hi = ci.strip("[] ").split(",")
        return float(lo), float(hi)
    return float(ci[0]), float(ci[1])


def fig_final_table(table, base_rate: float, out_dir: Path | None = None,
                    name: str = "fig_final_pr_auc", title: str | None = None) -> Path:
    """최종 비교표 그림: 방법별 PR-AUC (점) + 95% 신뢰구간 (가로 막대), 무작위 기준선 (점선).

    table: run_main score_table() 결과 (key, family, pr_auc, pr_auc_ci95 열).
    base_rate: 채점 대상의 스팸 비율 = 무작위로 줄 세웠을 때 PR-AUC 기댓값.
    위에서부터 PR-AUC 높은 순. 색은 방법 묶음, 범례로 묶음 이름을 같이 적는다.
    """
    df = table.sort_values("pr_auc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(df) + 1.3))
    for i, row in df.iterrows():
        lo, hi = _parse_ci(row["pr_auc_ci95"])
        color = FAMILY_COLORS.get(row["family"], FAMILY_COLORS["baseline"])
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.scatter(row["pr_auc"], i, s=40, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(hi + 0.004, i, f"{row['pr_auc']:.3f}", va="center", fontsize=7.5, color="#2d3748")
    ax.axvline(base_rate, color="#4a5568", ls="--", lw=1)
    ax.text(base_rate, len(df) - 0.4, f" spam share = {base_rate:.3f}", fontsize=7.5, color="#4a5568", va="bottom")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([METHOD_NAMES.get(k, k) for k in df["key"]], fontsize=8)
    ax.set_xlabel("PR-AUC (average precision), 95% domain-bootstrap CI")
    ax.set_ylim(-0.6, len(df) - 0.1)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=FAMILY_COLORS[f], marker="o", lw=2, label=f)
               for f in FAMILY_ORDER if f in set(df["family"])]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False)
    if title:
        ax.set_title(title, fontsize=9.5)
    return savefig(fig, name, out_dir)


PR_CURVE_KEYS = ("tr_set1_trust", "ppr", "lsh", "pagerank", "in_degree")


def fig_pr_curves(scores: dict, target, keys=PR_CURVE_KEYS, families: dict | None = None,
                  out_dir: Path | None = None, name: str = "fig_pr_curves",
                  title: str | None = None, y_max: float | None = None) -> Path:
    """PR 곡선: 대표 방법들의 precision-recall 곡선 + 무작위 기준선.

    scores: {방법 키: 길이 N 점수} (높을수록 스팸). target: evaluate.EvalTarget (채점 대상).
    families: {방법 키: 묶음} (색). 선 모양도 방법마다 달리해 색만으로 구분하지 않는다.
    y_max: 세로축 위쪽 끝. 신호가 약해 곡선이 바닥에 붙을 때 확대용 (재현율 0 근처 튀는 값은 잘림).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    families = families or {}
    styles = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for k, ls in zip(keys, styles):
        s = np.asarray(scores[k], dtype=float)[target.hostid]
        p, r, _ = precision_recall_curve(target.y, s)
        ap = average_precision_score(target.y, s)
        fam = families.get(k, k if k in FAMILY_COLORS else "baseline")
        ax.step(r, p, where="post", color=FAMILY_COLORS.get(fam, FAMILY_COLORS["baseline"]), ls=ls, lw=1.6,
                label=f"{METHOD_NAMES.get(k, k)} (AP {ap:.3f})")
    base = float(target.y.mean())
    ax.axhline(base, color="#4a5568", ls="--", lw=1, label=f"spam share = random ({base:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max if y_max else 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    if title:
        ax.set_title(title, fontsize=9.5)
    return savefig(fig, name, out_dir)

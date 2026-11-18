"""채점 코드: 모든 방법을 같은 규칙으로 채점한다.

담당: 강민수 (팀 작성 코드). 외부 라이브러리: scikit-learn (average_precision_score, roc_auc_score)

확정안 6절 규칙
- 점수는 "높을수록 스팸". 길이 = 전체 host 수(114,529)인 numpy 배열.
- 메인 지표 PR-AUC = average precision, 보조 ROC-AUC, Precision@50·100.
- 동률: AUC는 라이브러리 기본, Precision@K는 기댓값 방식.
- 신뢰구간: bootstrap 1,000회, 도메인 단위 재표집.
- 방법끼리 비교: 같은 bootstrap 표본으로 PR-AUC 차이의 95% 구간 (0 포함이면 "차이 없음").
- "잡았다" = eval_set에서 상위 100개 안에 든 스팸. 100번째 경계 동률은 seed 42 무작위 순서로 끊는다.
- PageRank는 탐지 점수가 아니라 순위 → evaluate_ranking (구간 10개별 스팸 비율).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
N_BOOT = 1000
KS = (50, 100)


# ---------------------------------------------------------------------------
# 채점 대상 준비
# ---------------------------------------------------------------------------
@dataclass
class EvalTarget:
    """채점 대상 host와 정답. 모든 방법이 같은 EvalTarget으로 채점된다."""

    hostid: np.ndarray   # 채점할 host id
    y: np.ndarray        # 1 = spam, 0 = nonspam
    domain: np.ndarray   # bootstrap 재표집 단위

    @classmethod
    def build(cls, hostids, label_of: dict, domain_of: Callable[[int], str]) -> "EvalTarget":
        """label_of: hostid -> "spam"/"nonspam". undecided나 라벨 없는 host가 섞이면 오류."""
        ids = np.asarray(hostids, dtype=int)
        labels = [label_of.get(int(i)) for i in ids]
        bad = [int(i) for i, l in zip(ids, labels) if l not in ("spam", "nonspam")]
        if bad:
            raise ValueError(f"채점 대상에 spam/nonspam이 아닌 host {len(bad)}개 (예: {bad[:5]})")
        y = np.array([l == "spam" for l in labels], dtype=int)
        dom = np.array([domain_of(int(i)) for i in ids])
        return cls(ids, y, dom)

    def subset(self, mask: np.ndarray) -> "EvalTarget":
        """하위 집합 채점용 (dangling 따로, LSH 커버 사이트만 등)."""
        m = np.asarray(mask, dtype=bool)
        return EvalTarget(self.hostid[m], self.y[m], self.domain[m])

    @property
    def n(self) -> int:
        return len(self.hostid)

    @property
    def n_spam(self) -> int:
        return int(self.y.sum())


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def precision_at_k(s: np.ndarray, y: np.ndarray, k: int) -> float:
    """상위 k개 중 스팸 비율. k번째 경계 동률은 기댓값으로 처리한다.

    경계에 걸린 동률 묶음에서 m자리를 채워야 하면,
    기댓값 스팸 수 = (그 묶음의 스팸 수) * m / (묶음 크기).
    """
    n = len(s)
    if n == 0:
        return float("nan")
    k = min(k, n)
    order = np.sort(s)[::-1]
    thr = order[k - 1]
    above = s > thr
    tie = s == thr
    m = k - int(above.sum())
    exp_pos = y[above].sum() + y[tie].sum() * m / tie.sum()
    return float(exp_pos / k)


def metrics(s: np.ndarray, y: np.ndarray, ks: Sequence[int] = KS) -> dict:
    out = {"pr_auc": float("nan"), "roc_auc": float("nan")}
    if 0 < y.sum() < len(y):
        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))
    for k in ks:
        out[f"p@{k}"] = precision_at_k(s, y, k)
    return out


def _check_scores(scores: np.ndarray, target: EvalTarget) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("점수는 1차원 배열이어야 함")
    if target.hostid.max(initial=-1) >= len(scores):
        raise ValueError("점수 배열이 채점 대상 hostid보다 짧음")
    s = scores[target.hostid]
    if not np.isfinite(s).all():
        raise ValueError("채점 대상 점수에 NaN/inf가 있음")
    return s


# ---------------------------------------------------------------------------
# 도메인 단위 bootstrap
# ---------------------------------------------------------------------------
def bootstrap_indices(target: EvalTarget, n_boot: int = N_BOOT, seed: int = SEED) -> list[np.ndarray]:
    """도메인을 복원추출해서 host 인덱스 목록을 만든다. 같은 seed면 항상 같은 표본.

    여러 방법을 비교할 때는 이 목록을 한 번 만들어 모든 방법에 같이 쓴다 (paired).
    """
    rng = np.random.default_rng(seed)
    doms, inv = np.unique(target.domain, return_inverse=True)
    members = [np.flatnonzero(inv == d) for d in range(len(doms))]
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(doms), size=len(doms))
        samples.append(np.concatenate([members[d] for d in pick]))
    return samples


def _ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))


def evaluate_detection(scores: np.ndarray, target: EvalTarget, boot: list[np.ndarray] | None = None,
                       ks: Sequence[int] = KS) -> dict:
    """탐지 점수 채점. 반환: 지표 + 각 지표의 95% 신뢰구간 + 대상 수."""
    s = _check_scores(scores, target)
    out = metrics(s, target.y, ks)
    out["n_hosts"], out["n_spam"] = target.n, target.n_spam
    if boot is None:
        boot = bootstrap_indices(target)
    vals = {key: [] for key in out if key not in ("n_hosts", "n_spam")}
    for idx in boot:
        m = metrics(s[idx], target.y[idx], ks)
        for key in vals:
            vals[key].append(m[key])
    for key, v in vals.items():
        out[f"{key}_ci95"] = _ci(np.array(v))
    out["n_boot_valid"] = int(np.isfinite(np.array(vals["pr_auc"])).sum())
    return out


def compare(scores_a: np.ndarray, scores_b: np.ndarray, target: EvalTarget,
            boot: list[np.ndarray] | None = None, metric: str = "pr_auc") -> dict:
    """두 방법의 지표 차이 (A - B)와 같은 bootstrap 표본으로 구한 95% 구간."""
    sa, sb = _check_scores(scores_a, target), _check_scores(scores_b, target)
    if boot is None:
        boot = bootstrap_indices(target)
    diff = metrics(sa, target.y)[metric] - metrics(sb, target.y)[metric]
    d = np.array([metrics(sa[i], target.y[i])[metric] - metrics(sb[i], target.y[i])[metric] for i in boot])
    lo, hi = _ci(d)
    return {
        "metric": metric,
        "diff": float(diff),
        "ci95": (lo, hi),
        "verdict": "차이 없음" if lo <= 0 <= hi else ("A가 높음" if lo > 0 else "B가 높음"),
    }


# ---------------------------------------------------------------------------
# "잡았다" host 목록 (H3: 한 방법만 잡은 스팸 수)
# ---------------------------------------------------------------------------
def top_k_hosts(scores: np.ndarray, target: EvalTarget, k: int = 100, seed: int = SEED) -> np.ndarray:
    """eval_set 상위 k개 hostid. 경계 동률은 seed 무작위 순서로 끊는다."""
    s = _check_scores(scores, target)
    rng = np.random.default_rng(seed)
    tiebreak = rng.permutation(len(s))
    order = np.lexsort((tiebreak, -s))  # 점수 내림차순, 같으면 무작위 순서
    return target.hostid[order[:k]]


def caught_spam(scores: np.ndarray, target: EvalTarget, k: int = 100, seed: int = SEED) -> set[int]:
    top = set(top_k_hosts(scores, target, k, seed).tolist())
    spam = set(target.hostid[target.y == 1].tolist())
    return top & spam


# ---------------------------------------------------------------------------
# PageRank 순위 평가 (Q1): 순위 구간별 스팸 비율
# ---------------------------------------------------------------------------
def evaluate_ranking(rank_scores: np.ndarray, target: EvalTarget, n_buckets: int = 10,
                     boot: list[np.ndarray] | None = None, seed: int = SEED) -> list[dict]:
    """eval_set host만 점수 내림차순으로 줄 세워 n_buckets개 구간으로 나누고, 구간별 스팸 비율.

    rank_scores: 높을수록 상위 (PageRank, TrustRank 신뢰 점수 그대로)
    구간 신뢰구간은 bootstrap 표본마다 같은 방식으로 다시 구간을 나눠 계산한다.
    """
    s = _check_scores(rank_scores, target)

    def bucket_rates(sv, yv):
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.permutation(len(sv)), -sv))
        parts = np.array_split(order, n_buckets)
        return np.array([yv[p].mean() if len(p) else np.nan for p in parts])

    base = bucket_rates(s, target.y)
    if boot is None:
        boot = bootstrap_indices(target)
    bs = np.array([bucket_rates(s[i], target.y[i]) for i in boot])
    sizes = [len(p) for p in np.array_split(np.arange(target.n), n_buckets)]
    return [
        {"bucket": b + 1, "n_hosts": sizes[b], "spam_rate": float(base[b]), "ci95": _ci(bs[:, b])}
        for b in range(n_buckets)
    ]

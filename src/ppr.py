"""Personalized PageRank (PPR): SET1 스팸 seed에서 의심을 퍼뜨린다.

담당: 강민수 (팀 작성 코드). 공통 엔진 src/engine.py(박병언)의 power_iteration을 쓴다.

확정안 8.3절 규칙
- teleport: SET1 스팸 seed (검증 땐 seed용 fold 스팸만, 최종 땐 SET1 스팸 전부. 호출하는 쪽이 넘긴다)
- 방향
  · "forward" (링크 방향): seed가 링크한 곳으로 의심이 퍼진다. 나가는 링크 없는 seed는 퍼뜨리지 못한다.
  · "reverse" (역방향 = Anti-TrustRank 방식): seed에 링크한 곳으로 퍼진다. 그래프를 뒤집어 같은 엔진에 넣는다.
- 점수 (높을수록 스팸)
  ① "raw"  : PPR 그대로. dangling → seed ("teleport" 규칙)
  ② "ratio": PPR / PR. 분자·분모 모두 같은 그래프, dangling "uniform".
             역방향이면 분모도 뒤집은 그래프의 PageRank. 상수배는 순위에 무관하므로 값 자체는 해석하지 않는다.
- 실제로 퍼뜨린 seed 수(넣은 그래프에서 나가는 링크가 있는 seed 수)를 기록한다.
- TrustRank·PageRank와 같은 수식 계열이다 (teleport만 다름). 보고서 Methods에 명시.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from src import config
from src.load_graph import dangling_mask

DIRECTIONS = ("forward", "reverse")
SCORE_TYPES = ("raw", "ratio")


@dataclass
class PPRResult:
    scores: np.ndarray                      # 길이 N, 높을수록 스팸
    info: dict = field(default_factory=dict)  # 설정, 반복 횟수, 퍼뜨린 seed 수 (.json에 같이 저장)


def _engine():
    # 지연 import: engine.py가 아직 없어도 이 파일의 다른 함수와 테스트는 쓸 수 있게
    from src.engine import power_iteration
    return power_iteration


def _run(A: sp.csr_matrix, teleport: np.ndarray, damping: float, dangling: str) -> tuple[np.ndarray, int | None]:
    """엔진 호출. 엔진이 r만 돌려주든 (r, 반복 횟수)를 돌려주든 받는다."""
    out = _engine()(A, teleport, damping=damping, dangling=dangling,
                    tol=config.TOL, max_iter=config.MAX_ITER)
    if isinstance(out, tuple):
        r, n_iter = out[0], out[1]
    else:
        r, n_iter = out, None
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (A.shape[0],) or not np.isfinite(r).all():
        raise ValueError("엔진 출력 모양 또는 값이 이상함")
    return r, (int(n_iter) if n_iter is not None else None)


def oriented(A: sp.spmatrix, direction: str) -> sp.csr_matrix:
    """forward면 그대로, reverse면 뒤집은 그래프 (A.T). 0·1 그래프를 가정."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction은 {DIRECTIONS} 중 하나")
    return (A.T if direction == "reverse" else A).tocsr()


def seed_vector(n: int, spam_seeds) -> np.ndarray:
    """스팸 seed에 균등한 teleport 벡터 (합 1)."""
    ids = np.unique(np.asarray(list(spam_seeds), dtype=np.int64))
    if len(ids) == 0:
        raise ValueError("스팸 seed가 없음")
    if ids.min() < 0 or ids.max() >= n:
        raise ValueError("seed hostid가 범위를 벗어남")
    v = np.zeros(n)
    v[ids] = 1.0 / len(ids)
    return v


def n_spreading_seeds(G: sp.spmatrix, spam_seeds) -> int:
    """넣은 그래프 G에서 나가는 링크가 있는 seed 수 (= 실제로 의심을 퍼뜨리는 seed)."""
    ids = np.unique(np.asarray(list(spam_seeds), dtype=np.int64))
    return int((~dangling_mask(G)[ids]).sum())


def ppr_score(A: sp.spmatrix, spam_seeds, damping: float = config.DAMPING,
              direction: str = "forward", score: str = "raw") -> PPRResult:
    """PPR 탐지 점수 하나를 계산한다.

    A: load_graph()의 0·1 그래프 (A[i, j] = 1 이면 i -> j)
    spam_seeds: seed로 쓸 스팸 hostid (SET2는 절대 넘기지 않는다)
    """
    if score not in SCORE_TYPES:
        raise ValueError(f"score는 {SCORE_TYPES} 중 하나")
    G = oriented(A, direction)
    n = G.shape[0]
    v = seed_vector(n, spam_seeds)
    info = {
        "method": "ppr",
        "direction": direction,
        "score": score,
        "damping": damping,
        "n_seeds": int(np.count_nonzero(v)),
        "n_spreading_seeds": n_spreading_seeds(G, spam_seeds),
        "n_dangling": int(dangling_mask(G).sum()),
    }
    if score == "raw":
        r, it = _run(G, v, damping, "teleport")
        info["n_iter"] = it
        return PPRResult(r, info)

    # ratio: 분자·분모 모두 같은 그래프 G, dangling "uniform"
    num, it_num = _run(G, v, damping, "uniform")
    den, it_den = _run(G, np.full(n, 1.0 / n), damping, "uniform")
    if (den <= 0).any():
        raise ValueError("PageRank에 0 이하 값이 있음 (teleport가 있으면 생길 수 없음)")
    info["n_iter"] = it_num
    info["n_iter_pagerank"] = it_den
    return PPRResult(num / den, info)


def all_variants(A: sp.spmatrix, spam_seeds, damping: float = config.DAMPING) -> dict[str, PPRResult]:
    """검증 후보 4개 (방향 2 × 점수 2). 키 예: "ppr_forward_raw"."""
    return {
        f"ppr_{d}_{s}": ppr_score(A, spam_seeds, damping, d, s)
        for d in DIRECTIONS for s in SCORE_TYPES
    }


def setting_name(direction: str, score: str, damping: float = config.DAMPING, seed_set: str = "set1") -> str:
    """점수 파일 이름 규칙 (확정안 10절): 방법_설정. 예: ppr_dir-forward_score-raw_d0.85_seeds-set1"""
    return f"ppr_dir-{direction}_score-{score}_d{damping:g}_seeds-{seed_set}"

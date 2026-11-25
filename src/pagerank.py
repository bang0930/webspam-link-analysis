"""PageRank: 전체 균등 teleport. 순위 오염 기준선 (확정안 8.1절, 연구 질문 Q1).

담당: 박병언 (팀 작성 코드). 공통 엔진 src/engine.py의 power_iteration을 쓴다.

확정안 규칙
- teleport: 전체 균등 v = 1/N (11강 p8 ~ p40).
- 균등 teleport에서는 dangling 규칙 "teleport"와 "uniform"이 같은 계산이다 (d = v = 1/N).
  인자는 다른 방법과 맞추려고 남겨 둔다. Spam Mass와 PPR/PR의 분모로 쓸 때는 "uniform"으로 부른다.
- pagerank(): 순위 점수. "높을수록 상위"라서 evaluate.evaluate_ranking에 그대로 넣는다
  (구간 10개별 스팸 비율 그림, TrustRank 신뢰 점수 순위와 나란히, 확정안 8.1절).
- pagerank_detection_score(): 최종표 참고용 1줄. 방향(sign)은 검증으로 정한다
  (확정안 5절 기본값 +1 = 높을수록 스팸, 후보 +1 / -1). 6절 규칙대로 "높을수록 스팸" 배열을 돌려준다.
- info는 ppr.py의 PPRResult처럼 설정, 반복 횟수, 실행 시간을 담는다 (점수 .json에 같이 저장, 확정안 10절).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from src import config
from src.engine import power_iteration
from src.load_graph import dangling_mask

SIGNS = (1, -1)


@dataclass
class PageRankResult:
    scores: np.ndarray                        # 길이 N
    info: dict = field(default_factory=dict)  # 설정, 반복 횟수, 실행 시간 (.json에 같이 저장)


def pagerank(A: sp.spmatrix, damping: float = config.DAMPING, dangling: str = "teleport") -> PageRankResult:
    """균등 teleport PageRank. scores는 합 1, 높을수록 상위.

    A: load_graph()의 0, 1 그래프 (A[i, j] = 1 이면 i -> j)
    """
    n = A.shape[0]
    t0 = time.perf_counter()
    r, n_iter = power_iteration(A, np.full(n, 1.0 / n), damping=damping, dangling=dangling,
                                tol=config.TOL, max_iter=config.MAX_ITER)
    seconds = time.perf_counter() - t0
    info = {
        "method": "pagerank",
        "score": "rank",
        "damping": float(damping),
        "dangling": dangling,
        "tol": config.TOL,
        "max_iter": config.MAX_ITER,
        "n_iter": int(n_iter),
        "converged": bool(n_iter < config.MAX_ITER),
        "seconds": round(seconds, 4),
        "n_hosts": int(n),
        "n_dangling": int(dangling_mask(A).sum()),
    }
    return PageRankResult(r, info)


def pagerank_detection_score(A: sp.spmatrix, damping: float = config.DAMPING, sign: int = +1) -> PageRankResult:
    """최종표 참고용 탐지 점수 = sign × PageRank (높을수록 스팸).

    sign = +1: PageRank가 높을수록 스팸 (확정안 5절 기본값)
    sign = -1: PageRank가 낮을수록 스팸
    """
    if sign not in SIGNS or isinstance(sign, bool):
        raise ValueError(f"sign은 {SIGNS} 중 하나여야 함 (받은 값 {sign!r})")
    res = pagerank(A, damping=damping, dangling="teleport")
    info = dict(res.info, score="detection", sign=int(sign))
    return PageRankResult(sign * res.scores, info)

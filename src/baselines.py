"""Baseline 5개: 무작위, in-degree, out-degree, 1단계 이웃(두 방향).

담당: 김현중 (팀 작성 코드). 외부 라이브러리: numpy, scipy

확정안 8.5절 규칙
- 모든 함수는 길이 N의 점수 배열을 돌려준다. 높을수록 스팸.
- degree 방향(많을수록 스팸 / 적을수록 스팸)은 검증으로 정한다 → `sign` 인자.
  기본값은 "많을수록 스팸"(sign=+1, 확정안 5절).
- 1단계 이웃은 LSH와 같은 보정식: 점수 = (s + α·p) / (n + α)
  n = 이웃 중 라벨 수, s = 그중 스팸 수, p = 쓸 수 있는 라벨의 스팸 비율. 이웃 라벨이 없으면 p.
  · "in"  : 나에게 링크한 host 중 스팸 비율 (링크 방향 PPR을 한 걸음에서 끊은 것)
  · "out" : 내가 링크한 host 중 스팸 비율 (역방향 PPR을 한 걸음에서 끊은 것)
- 라벨은 호출하는 쪽이 넘긴다: 검증 땐 seed용 fold 라벨만, 최종 땐 SET1 전체. SET2는 넘기지 않는다.
  그래프에 자기 자신 링크가 없으므로 자기 라벨은 자동으로 빠진다.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src import config
from src.load_graph import in_degree, out_degree


def random_scores(n: int, seed: int = config.SEED) -> np.ndarray:
    """무작위 점수. 같은 seed면 항상 같은 값."""
    return np.random.default_rng(seed).random(n)


def _signed(x: np.ndarray, sign: int) -> np.ndarray:
    if sign not in (1, -1):
        raise ValueError("sign은 +1(많을수록 스팸) 또는 -1(적을수록 스팸)")
    return sign * x.astype(np.float64)


def in_degree_scores(A: sp.spmatrix, sign: int = 1) -> np.ndarray:
    """들어오는 링크 수. sign=-1이면 적을수록 스팸."""
    return _signed(in_degree(A), sign)


def out_degree_scores(A: sp.spmatrix, sign: int = 1) -> np.ndarray:
    """나가는 링크 수. sign=-1이면 적을수록 스팸."""
    return _signed(out_degree(A), sign)


def label_vectors(n: int, spam_ids, nonspam_ids) -> tuple[np.ndarray, np.ndarray, float]:
    """(라벨 있음 0·1, 스팸 0·1, p). 같은 host가 양쪽에 있으면 오류."""
    spam_ids = np.unique(np.asarray(list(spam_ids), dtype=np.int64))
    nonspam_ids = np.unique(np.asarray(list(nonspam_ids), dtype=np.int64))
    if np.intersect1d(spam_ids, nonspam_ids).size:
        raise ValueError("같은 host가 spam과 nonspam에 모두 있습니다")
    n_lab = len(spam_ids) + len(nonspam_ids)
    if n_lab == 0:
        raise ValueError("라벨이 하나도 없습니다")
    is_lab = np.zeros(n)
    is_spam = np.zeros(n)
    is_lab[spam_ids] = 1
    is_lab[nonspam_ids] = 1
    is_spam[spam_ids] = 1
    return is_lab, is_spam, len(spam_ids) / n_lab


def one_hop_scores(A: sp.spmatrix, spam_ids, nonspam_ids, direction: str = "in",
                   alpha: float = config.LSH_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """1단계 이웃 스팸 비율 (보정). 반환: (점수, 커버 = 라벨된 이웃이 1개 이상).

    A는 0·1 그래프여야 한다 (load_graph 결과). A[i, j] = 1 이면 i -> j.
    """
    A = A.tocsr()
    n = A.shape[0]
    is_lab, is_spam, p = label_vectors(n, spam_ids, nonspam_ids)
    if direction == "in":        # 나에게 링크한 i: A[i, me] = 1 → A.T @ x
        M = A.T
    elif direction == "out":     # 내가 링크한 j: A[me, j] = 1 → A @ x
        M = A
    else:
        raise ValueError('direction은 "in" 또는 "out"')
    n_nb = np.asarray(M @ is_lab).ravel()
    s_nb = np.asarray(M @ is_spam).ravel()
    return (s_nb + alpha * p) / (n_nb + alpha), n_nb >= 1


def all_baselines(A: sp.spmatrix, spam_ids, nonspam_ids, *, in_sign: int = 1, out_sign: int = 1,
                  seed: int = config.SEED) -> dict[str, np.ndarray]:
    """최종표용 baseline 5개를 한 번에. degree 방향은 검증으로 고른 값을 넘긴다."""
    n = A.shape[0]
    return {
        "random": random_scores(n, seed),
        "in_degree": in_degree_scores(A, in_sign),
        "out_degree": out_degree_scores(A, out_sign),
        "one_hop_in": one_hop_scores(A, spam_ids, nonspam_ids, "in")[0],
        "one_hop_out": one_hop_scores(A, spam_ids, nonspam_ids, "out")[0],
    }

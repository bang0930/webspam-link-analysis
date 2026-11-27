"""공통 power iteration 엔진. PageRank, TrustRank, PPR이 모두 이 함수 하나를 쓴다.

담당: 박병언 (팀 작성 코드). 외부 라이브러리: numpy, scipy

확정안 7절 규격
    power_iteration(A, teleport, damping=0.85, dangling="teleport", tol=1e-8, max_iter=500, init=None)
        -> (r, n_iter)

- A: N x N 희소행렬, A[i, j] = 1 이면 i -> j 링크. 행 정규화로 전이 행렬 P를 만든다.
  자기 자신 링크(A[i, i])가 있어도 그대로 한 칸으로 센다 (11강 p16, p33 예제에 y -> y가 있음).
- teleport: 길이 N, 음수 없음, 합 > 0. 내부에서 합 1로 정규화한 벡터를 v라 한다.
- dangling(나가는 링크가 없는 host) 목록은 넣은 A에서 매번 새로 계산한다.
  뒤집은 그래프(A.T)를 넣으면 dangling 집합이 달라진다 (확정안 8.3절 역방향 PPR).
- dangling 규칙 (dangling host가 가진 질량을 어디로 보낼지, 분배 벡터 d)
    "teleport": d = v. 탐지 점수용 기본 (PageRank는 균등, TrustRank는 정상 seed, PPR은 스팸 seed)
    "uniform" : d = 1/N. 나누기 점수(Spam Mass, PPR/PR)용 (확정안 8.2절, 8.3절)
- 한 번 반복:
    r_new = β (Pᵀ r + d × (dangling host의 r 합)) + (1 − β) v
  r의 합이 1이면 r_new의 합도 1이다 (합 보존). 그래서 매 반복 renormalize하지 않는다.
  부동소수점 오차만 없애려고 출력 직전에 한 번만 합 1로 맞춘다.
- 수렴: L1 오차 |r_new − r|_1 < tol 이면 멈춘다 (11강 p17). 최대 max_iter회.
  max_iter까지 수렴하지 못하면 RuntimeWarning을 내고 마지막 r과 max_iter를 돌려준다.
- damping β는 0 초과 1 이하. β = 1 은 teleport가 없는 강의 p16, p18 예제용이다.

시작값 (init)
- 기본 init=None: r(0) = v.
- init="uniform": 강의 11강 p17처럼 r(0) = 1/N. 강의 반복 표와 값을 맞출 때 쓴다.
- 기본을 v로 둔 이유: 나가는 링크가 없는 seed만 teleport로 줄 때 ("teleport" 규칙),
  정확한 해는 seed에만 질량이 있고 다른 host는 정확히 0이다. r(0) = 1/N에서 시작하면
  다른 host의 값이 β^k 꼴로 줄어들 뿐 0이 되지 않아 tol에서 멈춘 뒤에도 작은 값이 남는다.
  r(0) = v면 처음부터 0이라 계속 0으로 남는다 (PPR 테스트 test_dangling_seed_cannot_spread는
  다른 host가 1e-12 이내로 0인지 본다). 수렴값 자체는 시작값과 무관하다
  (β < 1이면 반복이 축소 사상이라 해가 하나뿐이다).

강의 근거 (11강)
- p17: power iteration. r(0) = 1/N에서 시작해 r(t+1) = M r(t)를 L1 오차 < ε까지 반복.
- p22 ~ p34: spider trap과 dead end. spider trap은 teleport(1 − β)로 빠져나오고 (p29 ~ p33),
  dead end는 질량이 새므로 (p28) dangling 질량을 다시 분배해 막는다 (p34).
- p36 ~ p39: 희소행렬 계산. 꽉 찬 행렬 A = βM + (1 − β)[1/N]을 만들지 않고
  희소한 M만 저장해 r = βMr + (1 − β)/N 으로 계산한다. 여기서도 Mᵀ에 해당하는
  행렬(Pᵀ) 하나만 희소행렬로 저장한다.
- p40: 완전한 알고리즘. r' = βMr을 계산한 뒤 새어 나간 질량 (1 − S)를 다시 넣는다
  (S = r'의 합). 새어 나가는 질량은 teleport 몫 (1 − β)와 dangling host 몫
  β × (dangling host의 r 합) 두 가지다. teleport 벡터가 v일 때 이 (1 − S)를 전부 v로 넣으면
    r = βMr + (1 − S) v = β (Mr + v × dangling 합) + (1 − β) v
  가 되어 dangling="teleport" 규칙과 같은 계산이다. p40은 v가 균등(1/N)인 경우이고,
  p45는 이를 teleport 집합 S(여기서는 v)로 일반화한다.
- p45, p56: Topic-Specific PageRank, Personalized PageRank, Random Walk with Restart는
  teleport 벡터 v만 다른 같은 계산이다. TrustRank(p73, p75)도 신뢰 seed로 teleport하는
  topic-specific PageRank라서 같은 엔진을 쓴다 (확정안 3절).

선형성 (Spam Mass가 기대는 성질, 11강 p80)
- dangling="uniform"에서 u = 1/N, q = dangling 표시 벡터라고 두면 Q = Pᵀ + u qᵀ이다.
  P는 dangling 행이 0인 행 정규화 행렬이고, u qᵀ가 dangling 질량을 균등하게 재분배한다.
  β < 1에서 해는 r(v) = (1 − β)(I − βQ)⁻¹ v 이므로 정규화된 v에 대해 선형이다.
  예: 합 1인 v1, v2에 대해 r((v1 + v2) / 2) = (r(v1) + r(v2)) / 2.
  그래서 TR과 PR을 둘 다 "uniform"으로 계산해야 Spam Mass = 1 − TR/PR 의 뜻이 맞는다.
- dangling="teleport"면 d = v라서 전이 행렬 자체가 v에 따라 바뀌고, 일반적으로 선형이 아니다.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from src import config
from src.load_graph import dangling_mask

DANGLING_RULES = ("teleport", "uniform")
INIT_RULES = (None, "uniform")


def _check_graph(A) -> sp.csr_matrix:
    """A를 float64 csr로 바꾸고 모양과 값을 확인한다. 명시적으로 저장된 0은 지운다."""
    if not sp.issparse(A):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"A는 2차원 행렬이어야 함 (받은 차원 {A.ndim})")
    A = sp.csr_matrix(A, dtype=np.float64, copy=True)
    if A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError(f"A는 N x N (N > 0) 이어야 함 (받은 모양 {A.shape})")
    if not np.isfinite(A.data).all() or (A.data < 0).any():
        raise ValueError("A에 음수 또는 NaN/inf가 있음")
    A.eliminate_zeros()
    return A


def _check_teleport(teleport, n: int) -> np.ndarray:
    """teleport를 확인하고 합 1로 정규화한 v를 돌려준다."""
    t = np.asarray(teleport, dtype=np.float64)
    if t.shape != (n,):
        raise ValueError(f"teleport 길이가 N = {n}이어야 함 (받은 모양 {t.shape})")
    if not np.isfinite(t).all():
        raise ValueError("teleport에 NaN/inf가 있음")
    if (t < 0).any():
        raise ValueError("teleport에 음수가 있음")
    s = t.sum()
    if s <= 0:
        raise ValueError("teleport 합이 0임")
    return t / s


def _transition_T(A: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """(Pᵀ, dangling 여부). P = 행 정규화한 A. dangling 행은 0으로 두고 반복에서 따로 분배한다.

    강의 11강 p36 ~ p39처럼 희소행렬 하나만 저장한다.
    """
    dang = dangling_mask(A)
    out_w = np.asarray(A.sum(axis=1)).ravel()   # 0, 1 그래프면 out-degree와 같음
    inv = np.zeros(A.shape[0])
    inv[~dang] = 1.0 / out_w[~dang]
    PT = (sp.diags(inv) @ A).T.tocsr()
    return PT, dang


def power_iteration(A, teleport, damping: float = config.DAMPING, dangling: str = "teleport",
                    tol: float = config.TOL, max_iter: int = config.MAX_ITER,
                    init: str | None = None) -> tuple[np.ndarray, int]:
    """r = β(Pᵀr + d × dangling 질량 합) + (1 − β)v 의 고정점을 반복으로 구한다.

    A: N x N 그래프 (A[i, j] = 1 이면 i -> j)
    teleport: 길이 N, 음수 없음, 합 > 0 (내부에서 합 1로 정규화)
    damping: β, 0 < β <= 1
    dangling: "teleport" (d = v) 또는 "uniform" (d = 1/N)
    tol: L1 오차 기준, max_iter: 최대 반복 횟수
    init: None이면 r(0) = v, "uniform"이면 r(0) = 1/N (강의 11강 p17)

    반환: (r, n_iter). r은 길이 N, 음수 없음, 합 1. n_iter는 실제 반복 횟수
          (수렴하지 못하면 경고와 함께 max_iter).
    """
    A = _check_graph(A)
    n = A.shape[0]
    v = _check_teleport(teleport, n)

    if isinstance(damping, bool) or not isinstance(damping, (int, float, np.integer, np.floating)):
        raise ValueError(f"damping은 숫자여야 함 (받은 값 {damping!r})")
    if not (0.0 < float(damping) <= 1.0):
        raise ValueError(f"damping은 0 초과 1 이하여야 함 (받은 값 {damping!r})")
    beta = float(damping)
    if dangling not in DANGLING_RULES:
        raise ValueError(f"dangling은 {DANGLING_RULES} 중 하나여야 함 (받은 값 {dangling!r})")
    if init not in INIT_RULES:
        raise ValueError(f"init은 {INIT_RULES} 중 하나여야 함 (받은 값 {init!r})")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol은 양수여야 함 (받은 값 {tol!r})")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter는 1 이상의 정수여야 함 (받은 값 {max_iter!r})")
    max_iter = int(max_iter)

    PT, dang = _transition_T(A)
    d = v if dangling == "teleport" else np.full(n, 1.0 / n)
    restart = (1.0 - beta) * v

    r = v.copy() if init is None else np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        new = beta * (PT @ r + r[dang].sum() * d) + restart
        err = np.abs(new - r).sum()
        r = new
        if err < tol:
            return r / r.sum(), it

    warnings.warn(
        f"power_iteration이 max_iter = {max_iter}회 안에 수렴하지 못함 "
        f"(마지막 L1 오차 {err:.3e}, tol {tol:g})",
        RuntimeWarning,
        stacklevel=2,
    )
    return r / r.sum(), max_iter

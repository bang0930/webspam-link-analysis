"""TrustRank와 Spam Mass: 정상 seed에서 신뢰를 퍼뜨린다 (확정안 8.2절, 연구 질문 Q1, Q2, H5).

담당: 박병언 (팀 작성 코드). 공통 엔진 src/engine.py의 power_iteration을 쓴다.

강의 근거 (11강)
- p73: TrustRank는 topic-sensitive PageRank이고, topic은 믿을 만한 페이지 집합이다.
  정상 페이지가 스팸에 링크하는 일은 드물다는 가정 (approximate isolation).
- p75: 신뢰 전파. 스케일 차이를 빼면 (within a scaling factor) TrustRank는 신뢰 페이지를
  teleport 집합으로 쓴 PageRank다. 그래서 엔진에 teleport = seed 균등만 넣으면 된다.
- p76: 신뢰 감쇠 (seed에서 멀어질수록 β배씩 약해짐)와 신뢰 분할 (나가는 링크 수로 나눔)은
  엔진의 β Pᵀ r 한 줄에 둘 다 들어 있다.
- p77, p78: seed 고르는 법은 src/seeds.py.
- p79, p80: Spam Mass. r = PageRank, r⁺ = 신뢰 페이지로만 teleport한 PageRank,
  r⁻ = r − r⁺, Spam Mass = r⁻ / r = 1 − r⁺ / r.

점수 (모두 길이 N)
- trustrank(): 신뢰 점수 r (합 1, 높을수록 믿을 만함). dangling 기본 "teleport" (dangling 질량이 seed로).
  순위 그림(evaluate.evaluate_ranking, 확정안 8.1절)에는 이 r을 그대로 (높을수록 상위) 넘긴다.
- 탐지 점수 ① trustrank_score(): 신뢰 부호를 뒤집은 −r (높을수록 스팸). dangling "teleport".
- 탐지 점수 ② spam_mass_score(): 1 − TR/PR. TR과 PR 모두 dangling "uniform", 같은 그래프 A.
  상수배는 순위에 무관하므로 스케일 보정은 하지 않고, 값 자체(예: 50%)는 해석하지 않는다.
  근거:
    dangling = "uniform"에서 u = 1/N, q = dangling 표시 벡터, Q = Pᵀ + u qᵀ라고 두면
    엔진 해는 r(v) = (1 − β)(I − βQ)⁻¹ v 이다. P는 dangling 행이 0인 행 정규화 행렬이다.
    Q가 v와 무관하므로 정규화된 teleport v에 대해 선형이다 (src/engine.py).
    균등 teleport u = 1/N을 seed 집합 S 몫과 나머지로 나누면
    u = (|S|/N) v_S + (나머지)이므로, 강의 p80의 r⁺ (teleport 질량을 S에만 1/N씩 둔 PageRank)는
    r⁺ = (|S|/N) × TR 이다 (TR = v_S로 teleport한 합 1 TrustRank). p75의 "within a scaling factor"가
    바로 이 |S|/N이다. 따라서 강의의 Spam Mass = 1 − (|S|/N) TR/PR 와 여기 값 1 − TR/PR 는
    TR/PR에 양의 상수를 곱한 차이뿐이라 순위가 같다. 반대로 dangling "teleport"에서는 엔진이
    선형이 아니라서 이 분해가 성립하지 않으므로 Spam Mass에는 "uniform"을 쓴다.

검증과 최종 (확정안 4절, 6절)
- make_score_fn(): evaluate.cross_validate에 넣는 score_fn(seed_spam, seed_non)을 만든다.
  PageRank는 fold와 무관하니 한 번만 계산해 재사용한다.
- final_rows(): 최종표 TrustRank 4줄 (seed 2종 x 점수 2종). SET1 라벨만 넘긴다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from src import config
from src import seeds as S
from src.engine import power_iteration
from src.load_graph import dangling_mask
from src.pagerank import pagerank

SEED_METHODS = ("set1", "pagerank_top", "trusted_domain")
SCORE_TYPES = ("trust", "spam_mass")


@dataclass
class TrustRankResult:
    scores: np.ndarray                        # 길이 N (trust면 신뢰 점수 또는 −신뢰, spam_mass면 1 − TR/PR)
    info: dict = field(default_factory=dict)  # 설정, seed 수, 반복 횟수, 실행 시간 (.json에 같이 저장)


def seed_vector(n: int, good_seeds: Iterable[int]) -> np.ndarray:
    """정상 seed에 균등한 teleport 벡터 (합 1)."""
    ids = np.unique(np.asarray(list(good_seeds), dtype=np.int64))
    if len(ids) == 0:
        raise ValueError("정상 seed가 없음")
    if ids.min() < 0 or ids.max() >= n:
        raise ValueError("seed hostid가 범위를 벗어남")
    v = np.zeros(n)
    v[ids] = 1.0 / len(ids)
    return v


def _seed_info(G: sp.csr_matrix, ids: np.ndarray, seed_method: str | None) -> dict:
    return {
        "method": "trustrank",
        "seed_method": seed_method,
        "n_seeds": int(len(ids)),
        "n_spreading_seeds": int((~dangling_mask(G)[ids]).sum()),   # 나가는 링크가 있는 seed
        "n_dangling": int(dangling_mask(G).sum()),
    }


def trustrank(A: sp.spmatrix, good_seeds: Iterable[int], damping: float = config.DAMPING,
              dangling: str = "teleport", seed_method: str | None = None) -> TrustRankResult:
    """TrustRank 신뢰 점수 r (합 1, 높을수록 믿을 만함). teleport = seed 균등 (강의 p75).

    A: load_graph()의 0, 1 그래프 (A[i, j] = 1 이면 i -> j)
    good_seeds: 정상 seed hostid (seeds.py에서 고른 것)
    seed_method: info 기록용 이름 ("set1", "pagerank_top", "trusted_domain" 등)
    """
    G = sp.csr_matrix(A)
    v = seed_vector(G.shape[0], good_seeds)
    ids = np.flatnonzero(v)
    t0 = time.perf_counter()
    r, n_iter = power_iteration(G, v, damping=damping, dangling=dangling,
                                tol=config.TOL, max_iter=config.MAX_ITER)
    info = _seed_info(G, ids, seed_method)
    info.update(score="rank", damping=float(damping), dangling=dangling,
                n_iter=int(n_iter), seconds=round(time.perf_counter() - t0, 4))
    return TrustRankResult(r, info)


def trustrank_score(A: sp.spmatrix, good_seeds: Iterable[int], damping: float = config.DAMPING,
                    seed_method: str | None = None) -> TrustRankResult:
    """탐지 점수 ①: −TrustRank (높을수록 스팸). dangling "teleport" (dangling 질량이 seed로)."""
    res = trustrank(A, good_seeds, damping, dangling="teleport", seed_method=seed_method)
    res.info["score"] = "trust"
    return TrustRankResult(-res.scores, res.info)


def _pagerank_uniform(A: sp.spmatrix, damping: float):
    """Spam Mass 분모용 PageRank (dangling "uniform"). 균등 teleport라 "teleport" 규칙과 같은 값."""
    return pagerank(A, damping=damping, dangling="uniform")


def spam_mass_score(A: sp.spmatrix, good_seeds: Iterable[int], damping: float = config.DAMPING,
                    seed_method: str | None = None, pr=None) -> TrustRankResult:
    """탐지 점수 ②: Spam Mass = 1 − TR/PR (높을수록 스팸, 강의 p79, p80).

    TR, PR 모두 같은 그래프 A, dangling "uniform". 상수배는 순위에 무관하므로 스케일 보정은 하지 않고,
    값 자체(예: 50%)는 해석하지 않는다 (근거는 파일 맨 위 docstring).
    pr: 미리 계산한 PageRank (pagerank.PageRankResult, 같은 A와 damping, dangling "uniform"). 없으면 계산.
    """
    if pr is None:
        pr = _pagerank_uniform(A, damping)
    if pr.info.get("dangling") != "uniform" or pr.info.get("damping") != float(damping):
        raise ValueError("pr은 같은 damping, dangling \"uniform\"으로 계산한 PageRank여야 함")
    den = np.asarray(pr.scores, dtype=np.float64)
    if den.shape != (A.shape[0],) or (den <= 0).any():
        raise ValueError("PageRank 모양이 다르거나 0 이하 값이 있음 (teleport가 있으면 생길 수 없음)")
    tr = trustrank(A, good_seeds, damping, dangling="uniform", seed_method=seed_method)
    info = dict(tr.info, score="spam_mass", n_iter_pagerank=pr.info.get("n_iter"))
    return TrustRankResult(1.0 - tr.scores / den, info)


# ---------------------------------------------------------------------------
# 검증 (cross_validate)과 최종표
# ---------------------------------------------------------------------------
def setting_name(method: str, score: str, k: int | None = None, damping: float = config.DAMPING,
                 seed: int | None = None) -> str:
    """점수 파일 이름 규칙 (확정안 10절): 방법_설정.

    예: trustrank_seed-set1_kall_score-trust_d0.85, trustrank_seed-set1_k50_score-trust_d0.85_s42
    무작위가 들어간 설정(set1 또는 trusted_domain에서 k개)만 _s{seed}를 붙인다.
    """
    name = f"trustrank_seed-{method}_k{'all' if k is None else k}_score-{score}_d{damping:g}"
    if k is not None and method in ("set1", "trusted_domain") and seed is not None:
        name += f"_s{seed}"
    return name


def make_score_fn(A: sp.spmatrix, hostnames: Sequence[str], method: str, score: str,
                  k: int | None = None, seed: int = config.SEED,
                  damping: float = config.DAMPING) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """evaluate.cross_validate에 넣을 score_fn(seed_spam, seed_non) -> 길이 N 점수 (높을수록 스팸).

    method
      "set1"          : seed_non 전부 (k = None) 또는 무작위 k개
      "pagerank_top"  : seed_non 중 PageRank 상위 k개 (k = None 이면 전부)
      "trusted_domain": 신뢰 도메인 host에서 seed_spam을 뺀 것 전부 또는 무작위 k개
    score: "trust" (−TrustRank) 또는 "spam_mass" (1 − TR/PR)

    PageRank는 fold와 무관하므로 처음 한 번만 계산해 재사용한다 (pagerank_top seed, Spam Mass 분모).
    호출할 때마다의 info는 score_fn.infos 목록에 쌓인다.
    """
    if method not in SEED_METHODS:
        raise ValueError(f"method는 {SEED_METHODS} 중 하나 (받은 값 {method!r})")
    if score not in SCORE_TYPES:
        raise ValueError(f"score는 {SCORE_TYPES} 중 하나 (받은 값 {score!r})")
    G = sp.csr_matrix(A)
    if len(hostnames) != G.shape[0]:
        raise ValueError("hostnames 길이가 그래프 host 수와 다름")
    pr = _pagerank_uniform(G, damping) if (method == "pagerank_top" or score == "spam_mass") else None

    def score_fn(seed_spam: np.ndarray, seed_non: np.ndarray) -> np.ndarray:
        if method == "set1":
            good = S.random_seeds(seed_non, k, seed)
        elif method == "pagerank_top":
            good = S.pagerank_top_seeds(seed_non, pr.scores, k)
        else:
            good = S.trusted_domain_seeds(hostnames, exclude_spam_ids=seed_spam, k=k, seed=seed)
        if score == "trust":
            res = trustrank_score(G, good, damping, seed_method=method)
        else:
            res = spam_mass_score(G, good, damping, seed_method=method, pr=pr)
        res.info.update(k=k, seed=seed if k is not None else None,
                        name=setting_name(method, score, k, damping, seed))
        score_fn.infos.append(res.info)
        return res.scores

    score_fn.infos = []
    score_fn.pagerank = pr
    return score_fn


def final_rows(A: sp.spmatrix, hostnames: Sequence[str], set1_spam: Iterable[int],
               set1_nonspam: Iterable[int], damping: float = config.DAMPING) -> dict[str, TrustRankResult]:
    """최종표 TrustRank 4줄 = seed 2종 (SET1 정상 전부, 신뢰 도메인 전부) x 점수 2종 (trust, spam_mass).

    set1_spam, set1_nonspam: SET1 라벨만 넘긴다 (확정안 4절 최종).
    신뢰 도메인 "전부"에서도 SET1 스팸은 뺀다. SET2 스팸은 빼지 않는다 (확정안 8.2절).
    PageRank는 한 번만 계산해 Spam Mass 두 줄이 같이 쓴다.
    키는 setting_name() (예: trustrank_seed-set1_kall_score-trust_d0.85).
    """
    G = sp.csr_matrix(A)
    if len(hostnames) != G.shape[0]:
        raise ValueError("hostnames 길이가 그래프 host 수와 다름")
    pr = _pagerank_uniform(G, damping)
    seed_sets = {
        "set1": S.random_seeds(set1_nonspam, None),
        "trusted_domain": S.trusted_domain_seeds(hostnames, exclude_spam_ids=set1_spam, k=None),
    }
    out: dict[str, TrustRankResult] = {}
    for method, good in seed_sets.items():
        trust = trustrank_score(G, good, damping, seed_method=method)
        mass = spam_mass_score(G, good, damping, seed_method=method, pr=pr)
        for score, res in (("trust", trust), ("spam_mass", mass)):
            name = setting_name(method, score, None, damping)
            res.info.update(k=None, name=name)
            out[name] = res
    return out

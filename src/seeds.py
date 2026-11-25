"""TrustRank seed 선택 (확정안 8.2절, 강의 11강 p77, p78).

담당: 박병언 (팀 작성 코드). 외부 라이브러리: numpy

확정안 규칙
- 라벨은 호출하는 쪽이 넘긴다. 검증 땐 seed용 fold 라벨만, 최종 땐 SET1 전체 (확정안 4절).
  이 파일은 라벨 파일을 읽지 않는다 (SET2 라벨은 절대 쓰지 않음).
- seed 방식 3가지
  random_seeds        : 정상 host에서 무작위 k개 (seed 42 ~ 46, config.REPEAT_SEEDS로 5회)
  pagerank_top_seeds  : 정상 host 중 PageRank 상위 k개 (1회). 강의 p78 (1)
  trusted_domain_seeds: 가입이 통제된 신뢰 도메인 host (config.TRUST_SUFFIXES). 강의 p78 (2)
                        넘겨받은 알려진 스팸은 뺀다 (검증 땐 seed용 fold 스팸, 최종 땐 SET1 스팸).
- k = None 이면 후보 전부. k가 후보 수보다 크면 ValueError.
- 같은 seed 값이면 항상 같은 결과 (numpy default_rng). 반환은 hostid 배열.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src import config


def _ids(hostids: Iterable[int]) -> np.ndarray:
    """중복 없는 hostid 오름차순 배열."""
    ids = np.unique(np.asarray(list(hostids), dtype=np.int64))
    if len(ids) and ids.min() < 0:
        raise ValueError("hostid에 음수가 있음")
    return ids


def _check_k(k: int | None, n_candidates: int) -> int:
    """k = None 이면 후보 전부. 1 이상, 후보 수 이하만 허용."""
    if k is None:
        return n_candidates
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError(f"k는 1 이상의 정수 또는 None (받은 값 {k!r})")
    if k > n_candidates:
        raise ValueError(f"k = {k}가 후보 수 {n_candidates}보다 큼")
    return int(k)


def _sample(ids: np.ndarray, k: int | None, seed: int) -> np.ndarray:
    k = _check_k(k, len(ids))
    if k == len(ids):
        return ids.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=k, replace=False))


def random_seeds(nonspam_ids: Iterable[int], k: int | None = None, seed: int = config.SEED) -> np.ndarray:
    """정상 host에서 무작위 k개 (오름차순). k = None 이면 전부."""
    return _sample(_ids(nonspam_ids), k, seed)


def pagerank_top_seeds(nonspam_ids: Iterable[int], pr: np.ndarray, k: int | None = None) -> np.ndarray:
    """정상 host 중 PageRank 상위 k개 (강의 p78 (1)). 동률은 hostid 작은 순.

    pr: 길이 N PageRank (pagerank.pagerank(A).scores)
    반환 순서: PageRank 내림차순 (동률이면 hostid 오름차순).
    """
    ids = _ids(nonspam_ids)
    pr = np.asarray(pr, dtype=np.float64)
    if pr.ndim != 1 or (len(ids) and ids.max() >= len(pr)):
        raise ValueError("pr은 길이 N 배열이어야 하고 hostid가 그 범위 안이어야 함")
    k = _check_k(k, len(ids))
    order = np.lexsort((ids, -pr[ids]))   # 마지막 키가 우선: PageRank 내림차순, 그다음 hostid
    return ids[order[:k]]


def is_trusted_host(name: str, suffixes: Sequence[str] = config.TRUST_SUFFIXES) -> bool:
    """이름이 신뢰 도메인 접미사로 끝나는지. splits.is_trust_domain과 같은 규칙.

    소문자로 바꾸고 앞뒤 공백, 끝의 점, 포트를 뗀 뒤 비교한다.
    "www.ox.ac.uk" 는 True, "fakeac.uk.example.com" 처럼 중간에 들어간 것은 False.
    접미사 앞에 점이 있어야 하므로 "xac.uk" 같은 이름도 False. "ac.uk" 자체는 True.
    """
    h = name.strip().lower().rstrip(".")
    if ":" in h:
        h = h.split(":", 1)[0]
    return any(h.endswith(s) or h == s[1:] for s in suffixes)


def trusted_domain_ids(hostnames: Sequence[str]) -> np.ndarray:
    """신뢰 도메인 host 전부 (스팸 제외 전, 오름차순). 확정안 2절 실측 8,877개."""
    return np.array([i for i, h in enumerate(hostnames) if is_trusted_host(h)], dtype=np.int64)


def trusted_domain_seeds(hostnames: Sequence[str], exclude_spam_ids: Iterable[int] = (),
                         k: int | None = None, seed: int = config.SEED) -> np.ndarray:
    """신뢰 도메인 host에서 넘겨받은 알려진 스팸을 뺀 seed (강의 p78 (2), 확정안 8.2절).

    exclude_spam_ids: 검증 땐 seed용 fold 스팸, 최종 땐 SET1 스팸. SET2 스팸은 넘기지 않는다.
    k = None 이면 전부, 아니면 무작위 k개 (오름차순).
    """
    cand = trusted_domain_ids(hostnames)
    cand = np.setdiff1d(cand, _ids(exclude_spam_ids), assume_unique=False)
    return _sample(cand, k, seed)

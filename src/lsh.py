"""MinHash + LSH: 들어오는 링크 집합이 비슷한 host 묶음으로 스팸 점수 매기기.

담당: 김현중 (팀 작성 코드). 외부 라이브러리: numpy, scipy
강의 근거: 4강 p24-38 (MinHash), p39-58 (LSH banding, S-curve)

확정안 8.4절 규칙
- 집합: host마다 "나에게 링크한 host들" (스팸 중 들어오는 링크 있음 92%, 나가는 링크 있음 45%).
- 순서: 최소 집합 크기 필터 -> 집합이 완전히 같은 host끼리 묶음 -> 묶음 대표끼리만 MinHash + LSH.
  (같은 집합 공유 host 27,268개, 쌍으로 만들면 약 403만 쌍이라 먼저 묶는다)
- 버킷 크기(묶음 대표 수 기준)가 LSH_BUCKET_CAP 이하일 때만 쌍을 만들고,
  실제 Jaccard >= (1/b)^(1/r) 인 쌍만 남긴다. 상한을 넘는 버킷은 건너뛰고 수를 기록한다.
- 이웃 = 자기 묶음의 다른 host + LSH로 연결된 다른 묶음의 host 전부 (모든 band의 합집합, 중복 없음).
  이웃의 이웃으로 이어 붙이지 않는다.
- 점수 = (s + α·p) / (n + α). n = 이웃 중 라벨 수, s = 그중 스팸 수, p = 쓸 수 있는 라벨의 스팸 비율.
  이웃 라벨이 없으면 점수 = p. 자기 자신의 라벨은 쓰지 않는다.
- 커버 = 라벨된 이웃이 1개 이상인 host.
- 라벨은 호출하는 쪽이 넘긴다: 검증 땐 seed용 fold 라벨만, 최종 땐 SET1 전체. SET2는 절대 넘기지 않는다.

실행: python -m src.lsh   (설정별 후보 쌍 수·시간·커버리지를 출력, SET1 라벨만 사용)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from src import config

MERSENNE_P = (1 << 31) - 1      # MinHash 해시 h(x) = (a·x + b) mod P
HASH_CHUNK = 16                 # 해시 함수를 몇 개씩 나눠 계산할지 (메모리 절약)
PAIR_CHUNK = 500_000            # Jaccard 재확인을 몇 쌍씩 나눠 계산할지


# ---------------------------------------------------------------------------
# 1. 집합 만들기와 같은 집합 묶기
# ---------------------------------------------------------------------------
def in_link_sets(A: sp.spmatrix) -> sp.csr_matrix:
    """행 j = host j에게 링크한 host들 (0·1). A의 전치."""
    S = A.T.tocsr()
    S.sort_indices()
    return S


def group_identical(S: sp.csr_matrix, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """대상 host 중 집합이 완전히 같은 것끼리 묶는다.

    반환: (group, reps)
        group[h] = h가 속한 묶음 번호 (대상이 아니면 -1)
        reps[g]  = 묶음 g의 대표 host (가장 작은 hostid)
    """
    group = np.full(S.shape[0], -1, dtype=np.int64)
    first: dict[bytes, int] = {}
    reps: list[int] = []
    indptr, indices = S.indptr, S.indices
    for h in np.flatnonzero(eligible):
        key = indices[indptr[h]:indptr[h + 1]].tobytes()
        g = first.get(key)
        if g is None:
            g = len(reps)
            first[key] = g
            reps.append(int(h))
        group[h] = g
    return group, np.asarray(reps, dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. MinHash
# ---------------------------------------------------------------------------
def minhash_signatures(S_rep: sp.csr_matrix, num_perm: int = config.NUM_PERM,
                       seed: int = config.SEED) -> np.ndarray:
    """묶음 대표 집합의 MinHash signature. 반환 shape = (num_perm, 대표 수).

    강의 4강 p38 "row hashing": 무작위 순열 대신 h(x) = (a·x + b) mod P 를 num_perm개 쓴다.
    같은 seed면 항상 같은 signature.
    """
    n_rep = S_rep.shape[0]
    if n_rep == 0:
        return np.zeros((num_perm, 0), dtype=np.int64)
    lengths = np.diff(S_rep.indptr)
    if (lengths == 0).any():
        raise ValueError("빈 집합이 있습니다 (최소 집합 크기는 1 이상이어야 함)")

    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE_P, size=num_perm, dtype=np.int64)
    b = rng.integers(0, MERSENNE_P, size=num_perm, dtype=np.int64)
    x = S_rep.indices.astype(np.int64)
    starts = S_rep.indptr[:-1]

    sig = np.empty((num_perm, n_rep), dtype=np.int64)
    for lo in range(0, num_perm, HASH_CHUNK):
        hi = min(lo + HASH_CHUNK, num_perm)
        hv = (a[lo:hi, None] * x[None, :] + b[lo:hi, None]) % MERSENNE_P
        sig[lo:hi] = np.minimum.reduceat(hv, starts, axis=1)
    return sig


# ---------------------------------------------------------------------------
# 3. LSH banding -> 후보 쌍
# ---------------------------------------------------------------------------
@dataclass
class BandingResult:
    pairs: np.ndarray              # (k, 2) 대표 인덱스 쌍, i < j, 중복 없음
    skipped_buckets: int           # 상한을 넘어 건너뛴 버킷 수 (모든 band 합)
    skipped_reps: np.ndarray       # 건너뛴 버킷에 한 번이라도 들어간 대표 인덱스


def lsh_candidate_pairs(sig: np.ndarray, bands: int, rows: int,
                        bucket_cap: int = config.LSH_BUCKET_CAP) -> BandingResult:
    """signature를 b개 band × r줄로 나눠, 한 band라도 같으면 후보 쌍 (강의 4강 p44-46)."""
    if bands * rows != sig.shape[0]:
        raise ValueError(f"b·r = {bands * rows} 가 signature 길이 {sig.shape[0]} 와 다릅니다")
    n_rep = sig.shape[1]
    codes: list[np.ndarray] = []
    skipped = 0
    skipped_reps: list[np.ndarray] = []

    for band in range(bands):
        block = np.ascontiguousarray(sig[band * rows:(band + 1) * rows].T)
        _, inv, counts = np.unique(block, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(counts)])
        for bucket in np.flatnonzero(counts >= 2):
            members = order[bounds[bucket]:bounds[bucket + 1]]
            if len(members) > bucket_cap:
                skipped += 1
                skipped_reps.append(members)
                continue
            i, j = np.triu_indices(len(members), k=1)
            lo = np.minimum(members[i], members[j])
            hi = np.maximum(members[i], members[j])
            codes.append(lo * n_rep + hi)

    if codes:
        uniq = np.unique(np.concatenate(codes))
        pairs = np.stack([uniq // n_rep, uniq % n_rep], axis=1)
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
    skipped_set = np.unique(np.concatenate(skipped_reps)) if skipped_reps else np.zeros(0, dtype=np.int64)
    return BandingResult(pairs, skipped, skipped_set)


def jaccard_threshold(bands: int, rows: int) -> float:
    """S-curve 문턱값 (1/b)^(1/r) (강의 4강 p55)."""
    return (1.0 / bands) ** (1.0 / rows)


def exact_jaccard(S_rep: sp.csr_matrix, pairs: np.ndarray) -> np.ndarray:
    """후보 쌍의 실제 Jaccard = |A∩B| / |A∪B|."""
    if len(pairs) == 0:
        return np.zeros(0)
    size = np.diff(S_rep.indptr).astype(np.float64)
    out = np.empty(len(pairs))
    for lo in range(0, len(pairs), PAIR_CHUNK):
        p = pairs[lo:lo + PAIR_CHUNK]
        inter = np.asarray(S_rep[p[:, 0]].multiply(S_rep[p[:, 1]]).sum(axis=1)).ravel()
        out[lo:lo + PAIR_CHUNK] = inter / (size[p[:, 0]] + size[p[:, 1]] - inter)
    return out


# ---------------------------------------------------------------------------
# 4. 전체 실행과 점수
# ---------------------------------------------------------------------------
@dataclass
class LSHResult:
    scores: np.ndarray             # 길이 N, 높을수록 스팸
    covered: np.ndarray            # 라벨된 이웃이 1개 이상인 host
    group: np.ndarray              # host -> 묶음 번호 (-1 = 대상 아님)
    reps: np.ndarray               # 묶음 대표 host
    pairs: np.ndarray              # Jaccard 재확인을 통과한 대표 인덱스 쌍
    stats: dict = field(default_factory=dict)


def _label_vectors(n: int, spam_ids, nonspam_ids) -> tuple[np.ndarray, np.ndarray, float]:
    spam_ids = np.asarray(list(spam_ids), dtype=np.int64)
    nonspam_ids = np.asarray(list(nonspam_ids), dtype=np.int64)
    if np.intersect1d(spam_ids, nonspam_ids).size:
        raise ValueError("같은 host가 spam과 nonspam에 모두 있습니다")
    is_lab = np.zeros(n, dtype=np.float64)
    is_spam = np.zeros(n, dtype=np.float64)
    is_lab[spam_ids] = 1
    is_lab[nonspam_ids] = 1
    is_spam[spam_ids] = 1
    n_lab = len(spam_ids) + len(nonspam_ids)
    if n_lab == 0:
        raise ValueError("라벨이 하나도 없습니다")
    return is_lab, is_spam, len(spam_ids) / n_lab


def run_lsh(A: sp.spmatrix, spam_ids, nonspam_ids, *,
            min_set_size: int = config.MIN_SET_SIZE_DEFAULT,
            bands_rows: tuple[int, int] = config.LSH_BANDS_DEFAULT,
            num_perm: int = config.NUM_PERM,
            alpha: float = config.LSH_ALPHA,
            bucket_cap: int = config.LSH_BUCKET_CAP,
            seed: int = config.SEED) -> LSHResult:
    """LSH 점수 전체 계산.

    spam_ids, nonspam_ids: 점수 계산에 써도 되는 라벨 (검증 땐 seed용 fold, 최종 땐 SET1).
    """
    if min_set_size < 1:
        raise ValueError("min_set_size는 1 이상")
    bands, rows = bands_rows
    if bands * rows != num_perm:
        raise ValueError(f"b·r({bands * rows})이 num_perm({num_perm})과 달라요")

    t0 = time.perf_counter()
    n = A.shape[0]
    S = in_link_sets(A)
    eligible = np.diff(S.indptr) >= min_set_size

    group, reps = group_identical(S, eligible)
    S_rep = S[reps]
    sig = minhash_signatures(S_rep, num_perm, seed)
    band = lsh_candidate_pairs(sig, bands, rows, bucket_cap)
    jac = exact_jaccard(S_rep, band.pairs)
    thr = jaccard_threshold(bands, rows)
    kept = band.pairs[jac >= thr]

    # 묶음 사이 연결 (대칭, 0·1)
    n_rep = len(reps)
    G = sp.coo_matrix((np.ones(len(kept)), (kept[:, 0], kept[:, 1])), shape=(n_rep, n_rep)).tocsr()
    G = ((G + G.T) > 0).astype(np.float64)

    # 묶음별 라벨 수 → 이웃 라벨 수 (자기 묶음 + 연결된 묶음, 자기 자신 제외)
    is_lab, is_spam, p = _label_vectors(n, spam_ids, nonspam_ids)
    members = np.flatnonzero(eligible)
    g_of = group[members]
    L = np.bincount(g_of, weights=is_lab[members], minlength=n_rep)
    Sp = np.bincount(g_of, weights=is_spam[members], minlength=n_rep)
    L_nb, S_nb = G @ L, G @ Sp

    n_lab_nb = np.zeros(n)
    n_spam_nb = np.zeros(n)
    n_lab_nb[members] = L[g_of] - is_lab[members] + L_nb[g_of]
    n_spam_nb[members] = Sp[g_of] - is_spam[members] + S_nb[g_of]

    scores = (n_spam_nb + alpha * p) / (n_lab_nb + alpha)
    covered = n_lab_nb >= 1

    group_size = np.bincount(g_of, minlength=n_rep)
    skipped_hosts = int(group_size[band.skipped_reps].sum()) if len(band.skipped_reps) else 0
    stats = {
        "min_set_size": min_set_size, "bands": bands, "rows": rows, "num_perm": num_perm,
        "alpha": alpha, "bucket_cap": bucket_cap, "seed": seed,
        "jaccard_threshold": thr, "p": p,
        "eligible_hosts": int(eligible.sum()),
        "groups": n_rep,
        "hosts_in_shared_groups": int(group_size[group_size > 1].sum()),
        "candidate_pairs": int(len(band.pairs)),
        "verified_pairs": int(len(kept)),
        "skipped_buckets": band.skipped_buckets,
        "skipped_groups": int(len(band.skipped_reps)),
        "skipped_hosts": skipped_hosts,
        "covered_hosts": int(covered.sum()),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return LSHResult(scores, covered, group, reps, kept, stats)


# ---------------------------------------------------------------------------
# 실행: python -m src.lsh  — 설정별 후보 쌍 수 먼저 세기 (확정안 9절 일정 11/20)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    from src.load_graph import load_graph, load_labels

    A, _ = load_graph()
    train, _ = load_labels()                       # SET1만 사용 (SET2는 절대 쓰지 않음)
    spam = train.loc[train.label == "spam", "hostid"]
    nonspam = train.loc[train.label == "nonspam", "hostid"]
    labeled = np.zeros(A.shape[0], dtype=bool)
    labeled[np.concatenate([spam, nonspam])] = True

    import pandas as pd

    cols = ["min", "b,r", "thr", "eligible", "groups", "cand_pairs", "verified",
            "skip_bkt", "skip_hosts", "covered", "cov_SET1", "sec"]
    print(" ".join(f"{c:>10s}" for c in cols))
    records = []
    for m in config.MIN_SET_SIZE_GRID:
        for b, r in config.LSH_BANDS_GRID:
            res = run_lsh(A, spam, nonspam, min_set_size=m, bands_rows=(b, r))
            s = res.stats
            cov = (res.covered & labeled).sum() / labeled.sum()
            row = [m, f"{b},{r}", f"{s['jaccard_threshold']:.2f}", s["eligible_hosts"], s["groups"],
                   s["candidate_pairs"], s["verified_pairs"], s["skipped_buckets"], s["skipped_hosts"],
                   s["covered_hosts"], f"{cov:.1%}", s["seconds"]]
            print(" ".join(f"{str(v):>10s}" for v in row), flush=True)
            records.append({**s, "coverage_set1": round(float(cov), 4)})

    out = config.TABLES_DIR / "lsh_candidate_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\n저장: {out.relative_to(config.ROOT)}  (SET1 라벨만 사용, 실행 시간은 컴퓨터마다 다름)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

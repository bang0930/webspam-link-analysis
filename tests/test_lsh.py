"""lsh.py 테스트 (합성 데이터). 실행: python -m pytest tests -q"""

import numpy as np
import pytest
import scipy.sparse as sp

from src import lsh as H


def _graph(n, edges):
    src, dst = zip(*edges)
    return sp.csr_matrix((np.ones(len(edges)), (src, dst)), shape=(n, n))


def _in_from(sources, targets):
    """sources 모두가 targets 각각에 링크 → targets의 들어오는 링크 집합 = sources."""
    return [(s, t) for t in targets for s in sources]


def test_group_identical_and_filter():
    # 10, 11, 12는 {0,1,2}에서 링크받음 (같은 집합), 13은 {0}에서만, 14는 링크 없음
    A = _graph(20, _in_from([0, 1, 2], [10, 11, 12]) + [(0, 13)])
    S = H.in_link_sets(A)
    group, reps = H.group_identical(S, np.diff(S.indptr) >= 1)
    assert group[10] == group[11] == group[12] != group[13]
    assert group[14] == -1
    assert reps[group[10]] == 10                     # 대표 = 가장 작은 hostid
    group2, _ = H.group_identical(S, np.diff(S.indptr) >= 2)
    assert group2[13] == -1                          # 최소 집합 크기 2 → 13 제외


def test_minhash_estimates_jaccard():
    # 집합 X = 0..99, Y = 50..149 → Jaccard = 50/150 = 1/3
    rows = [list(range(0, 100)), list(range(50, 150)), list(range(0, 100))]
    S = sp.csr_matrix((np.ones(300), (np.repeat([0, 1, 2], 100), np.concatenate(rows))), shape=(3, 200))
    sig = H.minhash_signatures(S, num_perm=128, seed=42)
    assert sig.shape == (128, 3)
    assert (sig[:, 0] == sig[:, 2]).all()            # 같은 집합 → 같은 signature
    est = (sig[:, 0] == sig[:, 1]).mean()
    assert abs(est - 1 / 3) < 0.13                   # 표준편차 약 0.04
    assert (H.minhash_signatures(S, 128, 42) == sig).all()   # seed 고정 → 재현
    with pytest.raises(ValueError):
        H.minhash_signatures(sp.csr_matrix((1, 5)), 8)       # 빈 집합 금지


def test_banding_and_bucket_cap():
    sig = np.array([
        [1, 1, 2, 3, 3, 3],    # band 0: (0,1) 같음, (3,4,5) 같음
        [7, 7, 8, 9, 9, 9],
        [5, 6, 5, 4, 4, 4],    # band 1: (0,2) 같음
        [2, 3, 2, 1, 1, 1],
    ])
    res = H.lsh_candidate_pairs(sig, bands=2, rows=2, bucket_cap=10)
    got = {tuple(p) for p in res.pairs.tolist()}
    assert got == {(0, 1), (0, 2), (3, 4), (3, 5), (4, 5)}   # 두 band에서 나와도 중복 없음
    capped = H.lsh_candidate_pairs(sig, bands=2, rows=2, bucket_cap=2)
    assert {tuple(p) for p in capped.pairs.tolist()} == {(0, 1), (0, 2)}
    assert capped.skipped_buckets == 2 and set(capped.skipped_reps) == {3, 4, 5}
    with pytest.raises(ValueError):
        H.lsh_candidate_pairs(sig, bands=3, rows=2)


def test_exact_jaccard_and_threshold():
    S = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=float))
    jac = H.exact_jaccard(S, np.array([[0, 1], [0, 2]]))
    assert jac.tolist() == pytest.approx([1 / 3, 1.0])
    assert H.jaccard_threshold(32, 4) == pytest.approx(0.420, abs=1e-3)
    assert H.jaccard_threshold(16, 8) == pytest.approx(0.707, abs=1e-3)
    assert H.jaccard_threshold(8, 16) == pytest.approx(0.878, abs=1e-3)


def _scenario():
    # 묶음 X: 10~13은 {0..9}에서 링크받음 (같은 집합)
    # 14는 {0..8}에서 링크받음 → X와 Jaccard 0.9 (LSH로 연결될 것)
    # 15는 {30..39}에서 링크받음 → 아무와도 안 비슷함
    # 16은 들어오는 링크 없음
    edges = _in_from(range(10), [10, 11, 12, 13]) + _in_from(range(9), [14]) + _in_from(range(30, 40), [15])
    return _graph(50, edges)


def test_run_lsh_scores():
    A = _scenario()
    spam, nonspam = [10, 11, 15], [12, 40, 41]       # p = 3/6 = 0.5
    res = H.run_lsh(A, spam, nonspam, bands_rows=(32, 4), alpha=3)
    p = 0.5
    s = res.stats
    assert s["p"] == p and s["verified_pairs"] >= 1
    # 13: 이웃 = 10,11,12(같은 묶음) + 14(연결, 라벨 없음) → n=3, s=2
    assert res.scores[13] == pytest.approx((2 + 3 * p) / (3 + 3))
    # 10: 자기 자신 제외 → 이웃 11(spam), 12(nonspam) → n=2, s=1
    assert res.scores[10] == pytest.approx((1 + 3 * p) / (2 + 3))
    # 14: 연결된 묶음 X의 라벨 → n=3, s=2
    assert res.scores[14] == pytest.approx((2 + 3 * p) / (3 + 3))
    # 15: 혼자 (자기 라벨은 안 씀) → p, 16: 들어오는 링크 없음 → p
    assert res.scores[15] == p and res.scores[16] == p
    assert res.covered[[10, 13, 14]].all() and not res.covered[[15, 16]].any()


def test_run_lsh_filter_and_determinism():
    A = _scenario()
    a = H.run_lsh(A, [10], [12], bands_rows=(32, 4))
    b = H.run_lsh(A, [10], [12], bands_rows=(32, 4))
    assert np.array_equal(a.scores, b.scores)                    # seed 고정 → 같은 결과
    big = H.run_lsh(A, [10], [12], bands_rows=(32, 4), min_set_size=10)
    assert big.group[14] == -1 and big.scores[14] == big.stats["p"]   # 집합 크기 9 < 10 → 제외
    with pytest.raises(ValueError):
        H.run_lsh(A, [10], [12], bands_rows=(16, 4))            # b·r ≠ num_perm
    with pytest.raises(ValueError):
        H.run_lsh(A, [10], [10], bands_rows=(32, 4))            # spam·nonspam 겹침


# ---------------------------------------------------------------------------
# 구조/점수 분리, 검증용 score_fn, S-curve
# ---------------------------------------------------------------------------
def test_build_score_split_matches_run_lsh():
    A = _scenario()
    whole = H.run_lsh(A, [10, 11, 15], [12, 40, 41], bands_rows=(32, 4))
    st = H.build_lsh(A, bands_rows=(32, 4))
    scores, covered, p = H.score_lsh(st, [10, 11, 15], [12, 40, 41])
    assert np.array_equal(scores, whole.scores) and np.array_equal(covered, whole.covered)
    # 같은 구조로 라벨만 바꿔 다시 매길 수 있음 (검증 fold마다 구조를 다시 안 만듦)
    fn = H.make_score_fn(A, bands_rows=(32, 4))
    assert np.array_equal(fn([10, 11, 15], [12, 40, 41]), whole.scores)
    assert not np.array_equal(fn([12], [10]), whole.scores)
    assert H.setting_name(1, (16, 8)) == "lsh_min1_b16r8"


def test_theory_and_band_check():
    assert H.theoretical_curve(0.0, 16, 8) == 0.0
    assert H.theoretical_curve(1.0, 16, 8) == 1.0
    # 강의 4강 p56 표: b=20, r=5, t=0.8 → 0.9996
    assert H.theoretical_curve(0.8, 20, 5) == pytest.approx(0.9996, abs=1e-4)
    sig = np.array([[1, 1, 2], [3, 3, 4], [5, 6, 5], [7, 8, 7]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    assert H.shares_band(sig, pairs, bands=2, rows=2).tolist() == [True, True, False]


def test_sample_pairs_and_scurve():
    A = _scenario()
    st = H.build_lsh(A, bands_rows=(32, 4))
    pairs = H.sample_pairs(st.S_rep, 50, seed=1)
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert (H.exact_jaccard(st.S_rep, pairs) > 0).all()        # 원소를 공유하는 쌍만
    df, summ = H.scurve(st, n_pairs=50, n_bins=10)
    assert {"jaccard_lo", "candidate_rate", "theory_mid", "n_pairs"} <= set(df.columns)
    assert 0 <= summ["recall_final"] <= 1 and summ["n_above_threshold"] >= 1

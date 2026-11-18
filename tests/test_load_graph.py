"""load_graph.py 테스트. 실행: python -m pytest tests -q

- 합성 데이터: 파일 형식 해석, 0·1 변환, 자기 자신 링크 제거, 오류 처리, 캐시
- 실제 데이터: data/raw가 있을 때만 알려진 통계와 대조 (없으면 건너뜀)
"""

import gzip
import os

import numpy as np
import pytest

from src import config
from src import load_graph as L


def _write_raw(d, hostnames, adj_lines, n=None, set1="", set2=""):
    with gzip.open(d / L.HOSTNAMES_FILE, "wt") as f:
        f.writelines(f"{i} {h}\n" for i, h in enumerate(hostnames))
    with gzip.open(d / L.GRAPH_FILE, "wt") as f:
        f.write(f"{len(hostnames) if n is None else n}\n")
        f.writelines(line + "\n" for line in adj_lines)
    for key, text in (("train", set1), ("test", set2)):
        p = d / L.LABEL_FILES[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    # host 2는 자기 자신 링크(2:4), host 0은 같은 목적지 중복, host 3은 dangling
    _write_raw(
        d,
        ["a.example.co.uk", "b.example.co.uk", "c.test.ac.uk", "d.spam.co.uk"],
        ["1:3 2:1 1:1", "0:1", "2:4 3:2", ""],
        set1="0 nonspam 0.000000 j1:N,j2:N\n3 spam 1.000000 j1:S\n",
        set2="1 undecided - j3:U\n2 nonspam 0.333333 j1:N,j2:S,j3:N\n",
    )
    return d


def test_graph_binary_no_selfloop(raw):
    A = L.read_graph(raw)
    assert A.shape == (4, 4)
    assert A.dtype == np.float64
    assert set(A.data) == {1.0}                   # 링크 수는 버리고 0·1
    assert A.diagonal().sum() == 0                # 자기 자신 링크 제거
    assert A.nnz == 4                             # 0→1, 0→2, 1→0, 2→3
    assert A[0, 1] == 1 and A[2, 3] == 1 and A[2, 2] == 0
    assert L.dangling_mask(A).tolist() == [False, False, False, True]
    # 역방향 그래프는 dangling 집합이 다르다
    assert L.dangling_mask(A.T.tocsr()).tolist() == [False, False, False, False]


def test_labels(raw):
    train, test = L.load_labels(raw)
    assert list(train.columns) == ["hostid", "label", "spamicity", "assessments"]
    assert train["label"].tolist() == ["nonspam", "spam"]
    assert np.isnan(test.loc[test.hostid == 1, "spamicity"]).all()   # '-' → NaN
    assert test.loc[test.hostid == 2, "spamicity"].item() == pytest.approx(0.333333)


def test_errors(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    _write_raw(d, ["a.co.uk", "b.co.uk"], ["5:1", ""])           # 범위 밖 목적지
    with pytest.raises(ValueError):
        L.read_graph(d)
    _write_raw(d, ["a.co.uk", "b.co.uk"], ["1:1"])               # 줄 수 부족
    with pytest.raises(ValueError):
        L.read_graph(d)
    _write_raw(d, ["a.co.uk"], [""], set1="0 maybe 0.5 j1:N\n")  # 알 수 없는 라벨
    with pytest.raises(ValueError):
        L.load_labels(d)
    _write_raw(d, ["a.co.uk", "b.co.uk"], ["", ""],
               set1="0 spam 1 j1:S\n", set2="0 spam 1 j1:S\n")    # SET1·SET2 겹침
    with pytest.raises(ValueError):
        L.load_labels(d)
    with pytest.raises(FileNotFoundError):
        L.load_graph(tmp_path / "없음", cache_dir=tmp_path / "cache")


def test_cache(raw, tmp_path):
    cache = tmp_path / "cache"
    A1, names = L.load_graph(raw, cache)
    assert (cache / "graph.npz").exists()
    A2, _ = L.load_graph(raw, cache)                             # 캐시에서 읽음
    assert (A1 != A2).nnz == 0 and names[0] == "a.example.co.uk"
    # 원본이 바뀌면 캐시를 다시 만든다
    _write_raw(raw, names, ["1:1", "", "", ""],
               set1="0 nonspam 0 j1:N\n", set2="1 spam 1 j1:S\n")
    st = (raw / L.GRAPH_FILE).stat()
    os.utime(raw / L.GRAPH_FILE, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    A3, _ = L.load_graph(raw, cache)
    assert A3.nnz == 1


# ---------------------------------------------------------------------------
# 실제 데이터 (2026-11-16 실측값과 대조)
# ---------------------------------------------------------------------------
real = pytest.mark.skipif(not (config.RAW_DIR / L.GRAPH_FILE).exists(),
                          reason="data/raw 없음 (scripts/download_data.py 먼저 실행)")


@real
def test_real_data(tmp_path):
    A, hostnames = L.load_graph(cache_dir=tmp_path / "cache")
    train, test = L.load_labels()
    got = L.summarize(A, train, test)
    for key, want in L.EXPECTED.items():
        assert got[key] == want, key
    assert len(hostnames) == config.N_HOSTS
    assert hostnames[0] == "0800.loan-line.co.uk"
    assert set(A.data) == {1.0} and A.diagonal().sum() == 0

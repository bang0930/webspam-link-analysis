"""그래프와 라벨 불러오기.

담당: 김현중 (팀 작성 코드). 외부 라이브러리: numpy, scipy, pandas

확정안 규칙 (docs/STRUCTURE.md 2·3·10절)
- 그래프는 링크 있음/없음(0·1)으로 쓴다. 링크 수 가중치는 범위 밖이라 버린다.
- 자기 자신으로 가는 링크는 제거한다 (원본에는 0개지만 규칙으로 보장).
- 라벨은 undecided까지 그대로 돌려준다. 거르는 건 splits.py / evaluate.py가 한다.
- 처음 한 번만 원본을 읽고 data/processed/에 캐시한다. 원본 파일이 바뀌면 캐시를 다시 만든다.

인터페이스
    load_graph()  -> (A, hostnames)
        A: N x N scipy.sparse.csr_matrix (float64), A[i, j] = 1 이면 i -> j 링크
        hostnames: hostid 순서의 사이트 이름 리스트 (길이 N)
    load_labels() -> (train, test)
        pandas 표: hostid(int), label("spam"/"nonspam"/"undecided"), spamicity(float, 없으면 NaN), assessments(str)

실행: python -m src.load_graph   (데이터 통계를 출력하고 알려진 값과 대조한다)
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config

GRAPH_FILE = "uk-2007-05.hostgraph_weighted.graph-txt.gz"
HOSTNAMES_FILE = "uk-2007-05.hostnames.txt.gz"
LABEL_FILES = {
    "train": Path("webspam-uk2007-set1-1.0") / "WEBSPAM-UK2007-SET1-labels.txt",
    "test": Path("webspam-uk2007-set2-1.0") / "WEBSPAM-UK2007-SET2-labels.txt",
}
CACHE_VERSION = 1  # 캐시 형식을 바꾸면 올린다

DOWNLOAD_HINT = "먼저 `python scripts/download_data.py`로 데이터를 받으세요."


# ---------------------------------------------------------------------------
# 원본 읽기
# ---------------------------------------------------------------------------
def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. {DOWNLOAD_HINT}")
    return path


def read_hostnames(raw_dir: Path | None = None) -> list[str]:
    """hostid 순서의 사이트 이름. hostid가 0부터 빈틈없이 이어지는지 확인한다."""
    path = _require(Path(raw_dir or config.RAW_DIR) / HOSTNAMES_FILE)
    names: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for expected, line in enumerate(f):
            parts = line.split()
            if len(parts) != 2 or int(parts[0]) != expected:
                raise ValueError(f"{path.name} {expected + 1}번째 줄 형식 오류: {line!r}")
            names.append(parts[1])
    return names


def read_graph(raw_dir: Path | None = None) -> sp.csr_matrix:
    """host 그래프를 0·1 희소행렬로 읽는다.

    파일 형식: 첫 줄 = host 수 N, 이후 i번째 줄 = host i의 out-link 목록 "목적지:링크수".
    링크 수는 버리고, 중복 링크는 하나로, 자기 자신 링크는 제거한다.
    """
    path = _require(Path(raw_dir or config.RAW_DIR) / GRAPH_FILE)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        n = int(f.readline())
        indptr = [0]
        indices: list[int] = []
        n_lines = 0
        for src, line in enumerate(f):
            if src >= n:
                if line.strip():
                    raise ValueError(f"{path.name}: host 수({n})보다 줄이 많습니다")
                continue
            for tok in line.split():
                indices.append(int(tok.split(":", 1)[0]))
            indptr.append(len(indices))
            n_lines += 1
    if n_lines != n:
        raise ValueError(f"{path.name}: host 수는 {n}인데 인접 목록은 {n_lines}줄입니다")

    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"{path.name}: 범위를 벗어난 목적지 host가 있습니다")

    A = sp.csr_matrix(
        (np.ones(len(idx), dtype=np.float64), idx, np.asarray(indptr, dtype=np.int64)),
        shape=(n, n),
    )
    A.sum_duplicates()
    A.setdiag(0)                  # 자기 자신 링크 제거
    A.eliminate_zeros()
    A.data[:] = 1.0               # 0·1 (중복이 합쳐져 2 이상이 된 것도 1로)
    A.sort_indices()
    return A


def read_labels(path: Path) -> pd.DataFrame:
    """라벨 파일: `hostid label spamicity assessments`. spamicity가 '-'면 NaN."""
    rows = []
    with open(_require(Path(path)), encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"{Path(path).name} {lineno}번째 줄 형식 오류: {line!r}")
            hostid, label, spamicity = parts[0], parts[1], parts[2]
            assessments = parts[3] if len(parts) > 3 else ""
            if label not in config.LABEL_SETS:
                raise ValueError(f"{Path(path).name} {lineno}번째 줄: 알 수 없는 라벨 {label!r}")
            rows.append((int(hostid), label, np.nan if spamicity == "-" else float(spamicity), assessments))
    df = pd.DataFrame(rows, columns=["hostid", "label", "spamicity", "assessments"])
    if df["hostid"].duplicated().any():
        raise ValueError(f"{Path(path).name}: 중복된 hostid가 있습니다")
    return df.sort_values("hostid", ignore_index=True)


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------
def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"name": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "graph.npz", cache_dir / "graph_meta.json"


def _cache_valid(meta_path: Path, npz_path: Path, source: dict) -> bool:
    if not (meta_path.exists() and npz_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("cache_version") == CACHE_VERSION and meta.get("source") == source


# ---------------------------------------------------------------------------
# 인터페이스
# ---------------------------------------------------------------------------
def load_graph(raw_dir: Path | None = None, cache_dir: Path | None = None,
               force: bool = False) -> tuple[sp.csr_matrix, list[str]]:
    """(A, hostnames). 캐시가 있고 원본이 그대로면 캐시를 읽는다. force=True면 다시 만든다."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    cache_dir = Path(cache_dir or config.PROCESSED_DIR)
    hostnames = read_hostnames(raw_dir)

    npz_path, meta_path = _cache_paths(cache_dir)
    source = _fingerprint(_require(raw_dir / GRAPH_FILE))
    if not force and _cache_valid(meta_path, npz_path, source):
        A = sp.load_npz(npz_path).tocsr()
    else:
        A = read_graph(raw_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(npz_path, A)
        meta = {"cache_version": CACHE_VERSION, "source": source,
                "n_hosts": A.shape[0], "n_edges": int(A.nnz)}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    if A.shape[0] != len(hostnames):
        raise ValueError(f"그래프 host 수 {A.shape[0]}와 hostname 수 {len(hostnames)}가 다릅니다")
    return A, hostnames


def load_labels(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) = (SET1, SET2). undecided 포함, hostid 오름차순."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    train = read_labels(raw_dir / LABEL_FILES["train"])
    test = read_labels(raw_dir / LABEL_FILES["test"])
    both = set(train["hostid"]) & set(test["hostid"])
    if both:
        raise ValueError(f"SET1과 SET2에 같이 있는 hostid {len(both)}개 (예: {sorted(both)[:5]})")
    return train, test


# ---------------------------------------------------------------------------
# 그래프 기본 통계 (EDA와 다른 모듈에서 같이 씀)
# ---------------------------------------------------------------------------
def out_degree(A: sp.spmatrix) -> np.ndarray:
    return np.diff(A.tocsr().indptr)


def in_degree(A: sp.spmatrix) -> np.ndarray:
    return np.diff(A.tocsc().indptr)


def dangling_mask(A: sp.spmatrix) -> np.ndarray:
    """나가는 링크가 없는 host. 넘겨받은 그래프에서 매번 계산한다 (역방향 그래프면 결과가 달라짐)."""
    return out_degree(A) == 0


# ---------------------------------------------------------------------------
# 실행: python -m src.load_graph
# ---------------------------------------------------------------------------
# 2026-11-16 실측값 (확정안 2절). 원본이 바뀌지 않았다면 항상 같아야 한다.
EXPECTED = {
    "n_hosts": 114_529,
    "n_edges": 1_836_441,
    "n_dangling": 49_379,
    "train": {"nonspam": 3776, "spam": 222, "undecided": 277},
    "test": {"nonspam": 1933, "spam": 122, "undecided": 149},
}


def summarize(A: sp.spmatrix, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "n_hosts": A.shape[0],
        "n_edges": int(A.nnz),
        "n_dangling": int(dangling_mask(A).sum()),
        "train": train["label"].value_counts().to_dict(),
        "test": test["label"].value_counts().to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    force = "--force" in (argv if argv is not None else sys.argv[1:])
    A, hostnames = load_graph(force=force)
    train, test = load_labels()
    got = summarize(A, train, test)

    ok = True
    for key, want in EXPECTED.items():
        have = got[key]
        mark = "ok " if have == want else "err"
        ok &= have == want
        print(f"[{mark}] {key}: {have}" + ("" if have == want else f"  (기대값 {want})"))
    print(f"[info] 들어오는 링크 0: {int((in_degree(A) == 0).sum())}, hostname 예: {hostnames[0]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

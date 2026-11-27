"""WEBSPAM-UK2007 원본 데이터 다운로드 스크립트. 담당: 김현중 (팀 작성 코드).

공식 배포처(https://chato.cl/webspam/datasets/uk2007/)에서 host 그래프, host 이름,
spam/nonspam 라벨(set1, set2)을 받아 data/raw/ 에 저장하고 라벨 압축을 푼다.
라벨 라이선스는 CC BY-NC-SA 3.0이며, 원본 파일은 저장소에 올리지 않고 이 스크립트로 받는다.

사용법 (저장소 루트에서):
    python scripts/download_data.py
"""

import hashlib
import shutil
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

try:  # python.org macOS 설치판은 시스템 인증서를 못 찾는 경우가 있어 certifi를 우선 사용
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

BASE_URL = "https://chato.cl/webspam/datasets/uk2007/"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# (원격 경로, 기대 크기 bytes) — 크기는 2026-11-16 접근 기준
FILES = [
    ("links/uk-2007-05.hostgraph_weighted.graph-txt.gz", 5394352),
    ("links/uk-2007-05.hostnames.txt.gz", 971436),
    ("webspam-uk2007-set1-1.0.tar.gz", 1170426),
    ("webspam-uk2007-set2-1.0.tar.gz", 1074409),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(remote: str, expected_size: int) -> Path:
    dest = RAW_DIR / Path(remote).name
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} (이미 있음)")
        return dest
    print(f"[get ] {remote}")
    # 서버가 Python 기본 User-Agent를 406으로 거부하므로 일반 클라이언트 이름을 지정
    req = urllib.request.Request(BASE_URL + remote, headers={"User-Agent": "Mozilla/5.0 (webspam-link-analysis downloader)"})
    with urllib.request.urlopen(req, context=SSL_CONTEXT) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    size = dest.stat().st_size
    if size != expected_size:
        sys.exit(f"[err ] {dest.name}: 크기 {size} != 기대값 {expected_size}")
    return dest


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for remote, size in FILES:
        path = download(remote, size)
        print(f"       sha256 {sha256(path)}")
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path) as tar:
                # filter 인자는 Python 3.12+ (일부 3.8~3.11 패치 버전)에서만 지원
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(RAW_DIR, filter="data")
                else:
                    tar.extractall(RAW_DIR)
            print(f"       압축 해제 완료")
    print(f"\n완료: {RAW_DIR}")


if __name__ == "__main__":
    main()

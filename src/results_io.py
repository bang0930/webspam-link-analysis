"""점수 저장·불러오기 (확정안 10절).

담당: 김현중 (팀 작성 코드). 외부 라이브러리: numpy

규칙
- 점수: results/scores/<모드>/<이름>.npy, 설정: 같은 이름의 .json
  모드는 "check"(SET1 fold 0으로 시험) / "final"(SET2 최종) — 쓰는 라벨이 달라서 캐시를 섞지 않는다.
- .json에는 설정값, 날짜, 실행 시간, src/ 폴더 해시(`git rev-parse HEAD:src`), dirty 여부를 넣는다.
- 불러올 때 src/ 해시가 지금과 다르면 경고하고 다시 계산한다 (코드를 고친 뒤 예전 점수를 쓰는 것 방지).
- 저장할 때 `git status --porcelain src` 출력이 있으면(커밋 안 한 수정) dirty=true로 기록하고 경고한다.
- 최종 채점과 12/4 재현 테스트는 dirty가 아닐 때만 돌린다 (run_main.py --final이 확인).
"""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from src import config

_WARNED = {"dirty": False}


# ---------------------------------------------------------------------------
# git 상태
# ---------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def src_hash() -> str | None:
    """마지막 커밋의 src/ 폴더 해시. git이 없으면 None."""
    return _git("rev-parse", "HEAD:src")


def src_dirty() -> bool | None:
    """src/에 커밋 안 한 수정이 있으면 True. git이 없으면 None."""
    out = _git("status", "--porcelain", "src")
    return None if out is None else bool(out)


def git_state() -> dict:
    return {"commit": _git("rev-parse", "HEAD"), "src_hash": src_hash(), "dirty": src_dirty()}


# ---------------------------------------------------------------------------
# 저장·불러오기
# ---------------------------------------------------------------------------
def _paths(name: str, mode: str, root: Path | None = None) -> tuple[Path, Path]:
    d = Path(root or config.SCORES_DIR) / mode
    return d / f"{name}.npy", d / f"{name}.json"


def save_scores(name: str, scores: np.ndarray, info: dict | None = None, mode: str = "check",
                root: Path | None = None, seconds: float | None = None) -> Path:
    """점수와 설정을 저장한다. 반환: .npy 경로."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError(f"{name}: 점수는 NaN/inf 없는 1차원 배열이어야 함")
    npy, meta = _paths(name, mode, root)
    npy.parent.mkdir(parents=True, exist_ok=True)
    state = git_state()
    if state["dirty"] and not _WARNED["dirty"]:
        warnings.warn("src/에 커밋 안 한 수정이 있는 상태로 점수를 저장합니다 (dirty=true로 기록)", stacklevel=2)
        _WARNED["dirty"] = True        # 실행당 한 번만 경고 (기록은 점수마다 남김)
    np.save(npy, scores)
    record = {"name": name, "mode": mode, "n": int(len(scores)),
              "saved_at": datetime.now().isoformat(timespec="seconds"),
              "seconds": None if seconds is None else round(seconds, 3),
              **state, "info": _jsonable(info or {})}
    meta.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return npy


def load_scores(name: str, mode: str = "check", root: Path | None = None) -> tuple[np.ndarray, dict] | None:
    """저장된 점수. 없거나 src/ 해시가 지금과 다르면 None (다르면 경고)."""
    npy, meta = _paths(name, mode, root)
    if not (npy.exists() and meta.exists()):
        return None
    record = json.loads(meta.read_text(encoding="utf-8"))
    now = src_hash()
    if now is not None and record.get("src_hash") != now:
        warnings.warn(f"{name}: 저장 당시와 src/ 코드가 달라서 다시 계산합니다", stacklevel=2)
        return None
    return np.load(npy), record


def cached(name: str, compute: Callable[[], tuple[np.ndarray, dict]], mode: str = "check",
           force: bool = False, root: Path | None = None) -> tuple[np.ndarray, dict]:
    """저장된 점수가 쓸 수 있으면 불러오고, 아니면 compute()로 계산해 저장한다.

    compute: () -> (점수, 설정 dict)
    """
    if not force:
        got = load_scores(name, mode, root)
        if got is not None:
            return got[0], got[1]["info"]
    t0 = time.perf_counter()
    scores, info = compute()
    save_scores(name, scores, info, mode, root, seconds=time.perf_counter() - t0)
    return np.asarray(scores, dtype=np.float64), _jsonable(info)


def _jsonable(x):
    """numpy 값을 json으로 쓸 수 있게 바꾼다."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

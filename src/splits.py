"""데이터 나누기: 도메인 함수, SET1 도메인 단위 5-fold, 최종 채점 대상(eval_set).

담당: 강민수 (팀 작성 코드)

확정안 4절·6절 규칙
- 도메인 = 3단계 도메인 (www1.example.co.uk -> example.co.uk). 원본 SET1/SET2 분할과 같은 기준.
- SET1 -> 도메인 단위 5-fold. 4개 fold는 seed용, 1개 fold는 검증용.
- 채점 대상(eval_set) = SET2 라벨 - undecided - 신뢰 도메인 사이트.
- splits.json, eval_set.json 에는 hostid와 fold 번호만 저장한다 (라벨 재배포 방지).

이 파일의 함수는 pandas 표와 hostname 목록만 받는 순수 함수다.
실제 데이터 읽기는 load_graph.py(김현중)가 맡고, 맨 아래 main()에서 연결한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5

# 신뢰 도메인 (확정안 2절). 강의 11강 p78 "신뢰 도메인을 seed로 쓰는 방법"
TRUST_SUFFIXES = (".gov.uk", ".ac.uk", ".nhs.uk", ".sch.uk", ".police.uk", ".mod.uk")

LABELS = ("nonspam", "spam", "undecided")


# ---------------------------------------------------------------------------
# 도메인
# ---------------------------------------------------------------------------
def normalize_host(host: str) -> str:
    """소문자, 앞뒤 공백·끝의 점·포트 제거."""
    h = host.strip().lower().rstrip(".")
    if ":" in h:
        h = h.split(":", 1)[0]
    return h


def domain_of(host: str) -> str:
    """3단계 도메인 (마지막 라벨 3개). 라벨이 3개 이하면 그대로 돌려준다.

    www1.example.co.uk -> example.co.uk
    news.bbc.co.uk     -> bbc.co.uk
    a.b.cabinet.gov.uk -> cabinet.gov.uk
    """
    parts = normalize_host(host).split(".")
    return ".".join(parts[-3:])


def is_trust_domain(host: str) -> bool:
    h = normalize_host(host)
    return any(h.endswith(s) or h == s[1:] for s in TRUST_SUFFIXES)


def overlapping_domains(train_hosts: Iterable[str], test_hosts: Iterable[str]) -> list[str]:
    """SET1과 SET2에 같이 나오는 도메인 목록. 확정안: 0개여야 한다."""
    d1 = {domain_of(h) for h in train_hosts}
    d2 = {domain_of(h) for h in test_hosts}
    return sorted(d1 & d2)


# ---------------------------------------------------------------------------
# SET1 5-fold
# ---------------------------------------------------------------------------
def make_folds(
    train: pd.DataFrame,
    hostnames: Sequence[str],
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """SET1 라벨 host(undecided 제외)를 도메인 단위로 n_folds개로 나눈다.

    같은 도메인은 항상 같은 fold. 스팸 비율이 fold마다 비슷하도록 층화한다
    (StratifiedGroupKFold, shuffle, random_state=seed).

    train: hostid, label 열
    hostnames: hostid -> hostname (길이 = 전체 host 수)
    반환: hostid, fold 열 (hostid 오름차순)
    """
    df = train.loc[train["label"].isin(["spam", "nonspam"]), ["hostid", "label"]].copy()
    df = df.sort_values("hostid").reset_index(drop=True)
    groups = np.array([domain_of(hostnames[i]) for i in df["hostid"]])
    y = (df["label"] == "spam").astype(int).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        fold[val_idx] = k
    assert (fold >= 0).all()
    return pd.DataFrame({"hostid": df["hostid"].to_numpy(), "fold": fold})


def fold_hosts(folds: pd.DataFrame, val_fold: int) -> tuple[np.ndarray, np.ndarray]:
    """(seed용 hostid, 검증용 hostid). 검증용은 아직 신뢰 도메인을 빼기 전이다."""
    is_val = folds["fold"].to_numpy() == val_fold
    return folds["hostid"].to_numpy()[~is_val], folds["hostid"].to_numpy()[is_val]


def drop_trust(hostids: Iterable[int], hostnames: Sequence[str]) -> np.ndarray:
    """신뢰 도메인 host를 뺀다 (검증 채점·최종 채점 공통 규칙)."""
    ids = np.asarray(list(hostids), dtype=int)
    keep = np.array([not is_trust_domain(hostnames[i]) for i in ids], dtype=bool)
    return ids[keep] if len(ids) else ids


def validation_eval_ids(folds: pd.DataFrame, val_fold: int, hostnames: Sequence[str]) -> np.ndarray:
    """검증 fold의 채점 대상 = 검증 fold host - 신뢰 도메인."""
    _, val = fold_hosts(folds, val_fold)
    return drop_trust(val, hostnames)


# ---------------------------------------------------------------------------
# 최종 채점 대상
# ---------------------------------------------------------------------------
def make_eval_set(test: pd.DataFrame, hostnames: Sequence[str]) -> np.ndarray:
    """eval_set = SET2 라벨 - undecided - 신뢰 도메인. hostid 오름차순."""
    ids = test.loc[test["label"].isin(["spam", "nonspam"]), "hostid"].to_numpy()
    return np.sort(drop_trust(ids, hostnames))


# ---------------------------------------------------------------------------
# 저장 / 불러오기 (hostid와 fold 번호만, 라벨 없음)
# ---------------------------------------------------------------------------
def save_splits(folds: pd.DataFrame, path: Path, n_folds: int = N_FOLDS, seed: int = SEED) -> None:
    obj = {
        "description": "SET1 도메인 단위 fold. 라벨은 넣지 않음 (원본에서 읽을 것).",
        "domain_rule": "3단계 도메인 (마지막 라벨 3개)",
        "n_folds": n_folds,
        "seed": seed,
        "hostid": folds["hostid"].astype(int).tolist(),
        "fold": folds["fold"].astype(int).tolist(),
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_splits(path: Path) -> pd.DataFrame:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame({"hostid": obj["hostid"], "fold": obj["fold"]})


def save_eval_set(ids: np.ndarray, path: Path) -> None:
    obj = {
        "description": "최종 채점 대상 = SET2 라벨 - undecided - 신뢰 도메인. 라벨은 넣지 않음.",
        "trust_suffixes": list(TRUST_SUFFIXES),
        "n_hosts": int(len(ids)),
        "hostid": [int(i) for i in ids],
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_eval_set(path: Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["hostid"], dtype=int)


# ---------------------------------------------------------------------------
# 요약 통계 (데이터 통계표 T1 용, 화면 출력만)
# ---------------------------------------------------------------------------
def summarize(folds: pd.DataFrame, train: pd.DataFrame, eval_ids: np.ndarray,
              test: pd.DataFrame, hostnames: Sequence[str]) -> pd.DataFrame:
    lab = pd.concat([train, test]).set_index("hostid")["label"]
    rows = []
    for k in sorted(folds["fold"].unique()):
        ids = folds.loc[folds["fold"] == k, "hostid"].to_numpy()
        ev = validation_eval_ids(folds, k, hostnames)
        rows.append({
            "set": f"SET1 fold {k}",
            "hosts": len(ids),
            "spam": int((lab.loc[ids] == "spam").sum()),
            "domains": len({domain_of(hostnames[i]) for i in ids}),
            "eval_hosts(신뢰도메인 제외)": len(ev),
            "eval_spam": int((lab.loc[ev] == "spam").sum()),
        })
    rows.append({
        "set": "SET2 eval_set",
        "hosts": len(eval_ids),
        "spam": int((lab.loc[eval_ids] == "spam").sum()),
        "domains": len({domain_of(hostnames[i]) for i in eval_ids}),
        "eval_hosts(신뢰도메인 제외)": len(eval_ids),
        "eval_spam": int((lab.loc[eval_ids] == "spam").sum()),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 실행: python -m src.splits
# ---------------------------------------------------------------------------
def main() -> None:
    # load_graph.py(김현중) 인터페이스: load_graph() -> (A, hostnames), load_labels() -> (train, test)
    from src.load_graph import load_graph, load_labels  # noqa: WPS433

    _, hostnames = load_graph()
    train, test = load_labels()

    over = overlapping_domains(
        [hostnames[i] for i in train["hostid"]], [hostnames[i] for i in test["hostid"]]
    )
    if over:
        raise SystemExit(
            f"[err] SET1/SET2 겹치는 도메인 {len(over)}개 (예: {over[:10]}). domain_of()를 확인할 것."
        )
    print("[ok ] SET1/SET2 겹치는 도메인 0개")

    folds = make_folds(train, hostnames)
    eval_ids = make_eval_set(test, hostnames)

    root = Path(__file__).resolve().parent.parent
    save_splits(folds, root / "splits.json")
    save_eval_set(eval_ids, root / "eval_set.json")
    print(summarize(folds, train, eval_ids, test, hostnames).to_string(index=False))
    print(f"\n저장: {root / 'splits.json'}, {root / 'eval_set.json'}")


if __name__ == "__main__":
    main()

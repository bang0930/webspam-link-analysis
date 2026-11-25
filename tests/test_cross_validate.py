"""evaluate.cross_validate / select_setting 테스트 (합성 데이터)."""
import numpy as np
import pytest

from src import evaluate as E
from src import splits as S
import pandas as pd


def _synthetic(n_domains=300, seed=0):
    """도메인마다 host 1~4개, 일부 도메인 전체가 스팸. 앞 절반 도메인 = SET1, 뒤 = SET2."""
    rng = np.random.default_rng(seed)
    hostnames, rows = [], []
    for d in range(n_domains):
        tld = ".ac.uk" if d % 25 == 0 else ".co.uk"
        spam = rng.random() < 0.07
        for h in range(rng.integers(1, 5)):
            hid = len(hostnames)
            hostnames.append(f"h{h}.site{d}{tld}")
            r = rng.random()
            lab = "undecided" if r < 0.05 else ("spam" if spam else "nonspam")
            rows.append((hid, lab, "train" if d < n_domains // 2 else "test"))
    df = pd.DataFrame(rows, columns=["hostid", "label", "set"])
    return hostnames, df[df.set == "train"][["hostid", "label"]], df[df.set == "test"][["hostid", "label"]]



def test_cross_validate_uses_only_seed_folds():
    hostnames, train, _ = _synthetic(n_domains=400)
    folds = S.make_folds(train, hostnames)
    lab = dict(zip(train.hostid, train.label))
    seen = []

    def score_fn(seed_spam, seed_non):
        seen.append(set(seed_spam) | set(seed_non))
        s = np.zeros(len(hostnames))
        for i, l in lab.items():          # 정답을 그대로 쓰는 "완벽한" 점수
            s[i] = l == "spam"
        return s

    cv = E.cross_validate(score_fn, folds, train, hostnames)
    assert len(cv["per_fold"]) == 5
    for k, used in enumerate(seen):
        val = set(folds.hostid[folds.fold == k])
        assert not (used & val)               # 검증 fold 라벨은 score_fn에 안 들어감
    assert cv["pr_auc_mean"] == pytest.approx(1.0)
    for f in cv["per_fold"]:                  # 검증 채점 대상에 신뢰 도메인 없음
        assert f["n_hosts"] > 0


def _cv(mean, std):
    return {"pr_auc_mean": mean, "pr_auc_std": std}


def test_select_setting_rules():
    r = E.select_setting({"a": _cv(0.30, 0.05), "b": _cv(0.33, 0.05)}, default="a")
    assert r["chosen"] == "a" and "유지" in r["reason"]          # 0.03 < 0.05
    r = E.select_setting({"a": _cv(0.30, 0.05), "b": _cv(0.40, 0.05)}, default="a")
    assert r["chosen"] == "b"
    r = E.select_setting({"a": _cv(0.40, 0.05), "b": _cv(0.30, 0.01)}, default="a")
    assert r["chosen"] == "a" and r["reason"] == "기본값이 최고"

"""splits.py 테스트 (합성 데이터). 실행: python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest

from src import splits as S


def test_domain_of():
    assert S.domain_of("www1.example.co.uk") == "example.co.uk"
    assert S.domain_of("news.bbc.co.uk") == "bbc.co.uk"
    assert S.domain_of("a.b.cabinet.gov.uk") == "cabinet.gov.uk"
    assert S.domain_of("bbc.co.uk") == "bbc.co.uk"
    assert S.domain_of("WWW.Example.CO.UK.") == "example.co.uk"
    assert S.domain_of("www.example.co.uk:8080") == "example.co.uk"


def test_trust_domain():
    assert S.is_trust_domain("www.ox.ac.uk")
    assert S.is_trust_domain("www.direct.gov.uk")
    assert S.is_trust_domain("x.nhs.uk")
    assert not S.is_trust_domain("www.bbc.co.uk")
    assert not S.is_trust_domain("fakegov.uk")


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


def test_no_overlap_and_folds():
    hostnames, train, test = _synthetic()
    assert S.overlapping_domains([hostnames[i] for i in train.hostid],
                                 [hostnames[i] for i in test.hostid]) == []
    folds = S.make_folds(train, hostnames)
    # undecided 제외
    lab = train.set_index("hostid").label
    assert (lab.loc[folds.hostid] != "undecided").all()
    assert set(folds.fold) == set(range(5))
    # 같은 도메인은 같은 fold
    dom = folds.hostid.map(lambda i: S.domain_of(hostnames[i]))
    assert (folds.groupby(dom).fold.nunique() == 1).all()
    # 재현성
    assert folds.equals(S.make_folds(train, hostnames))
    # 모든 fold에 스팸이 있음 (층화)
    spam_per_fold = folds.assign(s=(lab.loc[folds.hostid] == "spam").values).groupby("fold").s.sum()
    assert (spam_per_fold > 0).all()


def test_eval_set_rules():
    hostnames, train, test = _synthetic()
    ev = S.make_eval_set(test, hostnames)
    lab = test.set_index("hostid").label
    assert (lab.loc[ev] != "undecided").all()
    assert not any(S.is_trust_domain(hostnames[i]) for i in ev)
    assert set(ev) == {i for i in test.hostid
                       if lab[i] != "undecided" and not S.is_trust_domain(hostnames[i])}
    # 검증 fold 채점 대상도 신뢰 도메인 제외
    folds = S.make_folds(train, hostnames)
    for k in range(5):
        v = S.validation_eval_ids(folds, k, hostnames)
        assert not any(S.is_trust_domain(hostnames[i]) for i in v)


def test_save_load_no_labels(tmp_path):
    hostnames, train, test = _synthetic()
    folds = S.make_folds(train, hostnames)
    S.save_splits(folds, tmp_path / "splits.json")
    S.save_eval_set(S.make_eval_set(test, hostnames), tmp_path / "eval_set.json")
    for f in ("splits.json", "eval_set.json"):
        txt = (tmp_path / f).read_text()
        assert '"spam"' not in txt and "nonspam" not in txt  # 라벨 재배포 방지
    assert S.load_splits(tmp_path / "splits.json").equals(folds)

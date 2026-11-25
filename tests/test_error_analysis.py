"""error_analysis.py 테스트 (합성 데이터)."""
import numpy as np

from src import error_analysis as EA
from src.evaluate import EvalTarget

KEYS = ("a", "b", "c")


def _setup():
    # host 0~9 채점 대상, 0·1·2 스팸. 방법 a·b·c 모두 스팸 0을 최상위로, 스팸 2는 최하위로.
    n = 12
    t = EvalTarget(np.arange(10), np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), np.array([f"d{i}" for i in range(10)]))
    base = -np.arange(n, dtype=float)
    scores = {k: base.copy() for k in KEYS}
    for k in KEYS:
        scores[k][2] = -100           # 스팸 2: 모든 방법에서 꼴찌 → 놓친 스팸
        scores[k][3] = 10             # 정상 3: 모든 방법에서 1등 → 오탐
    scores["c"][1] = -50              # 스팸 1: c만 놓침
    names = [f"h{i}.site{i}.co.uk" if i % 2 else f"h{i}.org{i}.org.uk" for i in range(n)]
    deg_in, deg_out = np.arange(n), np.where(np.arange(n) % 3 == 0, 0, 5)
    return scores, t, names, deg_in, deg_out


def test_groups_and_counts():
    scores, t, names, din, dout = _setup()
    df = EA.host_table(scores, t, names, din, dout, keys=KEYS, k=3)
    g = dict(zip(df.hostid, df.group))
    assert g[2] == "missed_spam" and g[0] == "caught_spam" and g[1] == "caught_spam"
    assert g[3] == "false_alarm"                     # 3개 중 2개 이상 (과반)
    assert df.attrs["false_alarm_min"] == 2
    assert (df.loc[df.hostid == 3, "n_methods_flag"] == 3).all()
    pm = EA.per_method(df).set_index("method")
    assert pm.loc["a", "spam_caught"] == 2 and pm.loc["c", "spam_caught"] == 1
    assert pm.loc["a", "spam_only_this"] == 0


def test_percentile_and_summary():
    scores, t, names, din, dout = _setup()
    df = EA.host_table(scores, t, names, din, dout, keys=KEYS, k=3)
    assert df.loc[df.hostid == 3, "best_pct"].iloc[0] == 0.0       # 가장 의심
    assert df.loc[df.hostid == 2, "best_pct"].iloc[0] == 1.0       # 가장 덜 의심
    s = EA.summarize(df).set_index("group")
    assert s.loc["missed_spam", "n_hosts"] == 1 and s.loc["false_alarm", "n_hosts"] == 1
    assert len(EA.cases(df)) == 2
    assert set(EA.suffix_table(df).suffix) <= {"co.uk", "org.uk"}


def test_suffix_of():
    assert EA.suffix_of("www.Example.CO.UK.") == "co.uk"
    assert EA.suffix_of("x.ac.uk") == "ac.uk"

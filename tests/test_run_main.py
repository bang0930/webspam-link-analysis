"""results_io.py와 experiments/run_main.py 테스트 (합성 데이터). 실행: python -m pytest tests -q"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src import results_io as R
from src.evaluate import EvalTarget
from src.load_graph import dangling_mask

ROOT = Path(__file__).resolve().parent.parent


def _load_run_main():
    spec = importlib.util.spec_from_file_location("run_main", ROOT / "experiments" / "run_main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_main"] = mod          # dataclass가 모듈을 찾을 수 있게 등록
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# results_io
# ---------------------------------------------------------------------------
def test_save_load_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "src_hash", lambda: "abc")
    monkeypatch.setattr(R, "git_state", lambda: {"commit": "c0", "src_hash": "abc", "dirty": False})
    s = np.array([0.1, 0.5, 0.2])
    R.save_scores("m", s, {"k": np.int64(3), "arr": np.array([1, 2])}, mode="check", root=tmp_path)
    got, rec = R.load_scores("m", "check", tmp_path)
    assert np.array_equal(got, s) and rec["info"] == {"k": 3, "arr": [1, 2]} and rec["src_hash"] == "abc"
    meta = json.loads((tmp_path / "check" / "m.json").read_text())
    assert meta["dirty"] is False and meta["n"] == 3
    assert R.load_scores("m", "final", tmp_path) is None          # 모드가 다르면 다른 캐시

    calls = []
    def compute():
        calls.append(1)
        return np.array([1.0, 2.0]), {"x": 1}
    R.cached("c", compute, "check", root=tmp_path)
    R.cached("c", compute, "check", root=tmp_path)
    assert len(calls) == 1                                        # 두 번째는 저장된 것 사용
    R.cached("c", compute, "check", force=True, root=tmp_path)
    assert len(calls) == 2
    monkeypatch.setattr(R, "src_hash", lambda: "changed")         # 코드가 바뀌면 다시 계산
    with pytest.warns(UserWarning):
        R.cached("c", compute, "check", root=tmp_path)
    assert len(calls) == 3
    with pytest.raises(ValueError):
        R.save_scores("bad", np.array([np.nan]), root=tmp_path)


def test_dirty_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "git_state", lambda: {"commit": "c0", "src_hash": "abc", "dirty": True})
    monkeypatch.setitem(R._WARNED, "dirty", False)
    with pytest.warns(UserWarning):
        R.save_scores("d", np.array([1.0]), root=tmp_path)
    assert json.loads((tmp_path / "check" / "d.json").read_text())["dirty"] is True


# ---------------------------------------------------------------------------
# run_main: 합성 그래프로 점수 → 표까지 전체 흐름
# ---------------------------------------------------------------------------
def _synthetic_ctx(rm, tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    edges = set()
    for _ in range(1500):                                         # 정상끼리 무작위 링크
        a, b = rng.integers(30, n, size=2)
        if a != b:
            edges.add((a, b))
    for a in range(30):                                           # 스팸 0~29는 서로 촘촘히 링크
        for b in rng.choice(30, size=6, replace=False):
            if a != b:
                edges.add((a, int(b)))
        edges.add((int(rng.integers(30, n)), a))                  # 정상에서 스팸으로 약간
    src, dst = zip(*edges)
    A = sp.csr_matrix((np.ones(len(edges)), (src, dst)), shape=(n, n))
    hostnames = [f"h{i}.{'school.ac.uk' if 100 <= i < 120 else f'site{i // 3}.co.uk'}" for i in range(n)]
    label_of = {i: "spam" for i in range(30)} | {i: "nonspam" for i in range(30, n)}
    target_ids = np.r_[np.arange(20, 30), np.arange(200, 300)]
    target = EvalTarget.build(target_ids, label_of, lambda i: hostnames[i].split(".", 1)[1])
    return rm.Context("check", A, hostnames, np.arange(0, 20), np.arange(30, 200), target,
                      dangling_mask(A), tmp_path / "out")


def test_run_main_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "git_state", lambda: {"commit": "c0", "src_hash": "abc", "dirty": False})
    monkeypatch.setattr(R, "src_hash", lambda: "abc")
    rm = _load_run_main()
    ctx = _synthetic_ctx(rm, tmp_path)
    ms = rm.methods()
    scores = rm.compute_scores(ctx, ms, root=tmp_path / "scores")
    assert set(scores) == {m.key for m in ms}
    assert all(len(scores[m.key]) == len(m.seeds) for m in ms)   # 반복 방법은 seed 5개
    assert all(s.shape == (300,) for d in scores.values() for s in d.values())

    boot = rm.bootstrap_indices(ctx.target, n_boot=50)
    table = rm.score_table(ctx, ms, scores, boot)
    assert len(table) == len(ms) and table["pr_auc"].notna().all()
    assert {"pr_auc_mean5", "pr_auc_std5"} <= set(table.columns)
    comp = rm.comparisons(ctx, scores, boot)
    assert list(comp["hypothesis"]) == ["H2", "H3", "H6", "H5", "참고"]
    rank = rm.ranking(ctx, scores, boot)
    assert set(rank["ranking"]) == {"pagerank", "trustrank_set1"} and len(rank) == 20
    cau = rm.caught(ctx, scores)
    assert (cau.loc[cau["with"] == "", "caught"] <= ctx.target.n_spam).all()
    sub = rm.subsets(ctx, ms, scores)
    assert set(sub["subset"]) <= {"dangling", "non_dangling", "lsh_covered"}

    # 두 번째 실행은 저장된 점수를 그대로 쓴다 (LSH 커버 집합 포함)
    ctx2 = _synthetic_ctx(rm, tmp_path)
    scores2 = rm.compute_scores(ctx2, ms, root=tmp_path / "scores")
    assert np.array_equal(scores2["lsh"][42], scores["lsh"][42])
    assert np.array_equal(ctx2.extra["lsh_covered_mask"], ctx.extra["lsh_covered_mask"])


def test_final_refuses_dirty(monkeypatch, capsys):
    rm = _load_run_main()
    monkeypatch.setattr(rm, "src_dirty", lambda: True)
    assert rm.main(["--final"]) == 1
    assert "커밋 안 한 수정" in capsys.readouterr().out


def test_figures_and_analysis_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "git_state", lambda: {"commit": "c0", "src_hash": "abc", "dirty": False})
    monkeypatch.setattr(R, "src_hash", lambda: "abc")
    rm = _load_run_main()
    ctx = _synthetic_ctx(rm, tmp_path)
    ms = rm.methods()
    scores = rm.compute_scores(ctx, ms, root=tmp_path / "scores")
    boot = rm.bootstrap_indices(ctx.target)
    tables = {"table": rm.score_table(ctx, ms, scores, boot), "caught": rm.caught(ctx, scores)}
    paths = rm.make_figures(ctx, scores, tables)            # check 모드 → ctx.out_dir/figures에 check_ 이름으로
    assert len(paths) == 3
    assert all(p.exists() and p.parent == ctx.out_dir / "figures" and p.name.startswith("check_") for p in paths)

    calls = []
    rm.run_analysis([("a", lambda: calls.append("a") or 0), ("b", lambda: calls.append("b"))])
    assert calls == ["a", "b"]
    with pytest.raises(SystemExit):                          # 실패한 단계가 있으면 멈춤
        rm.run_analysis([("bad", lambda: 1)])
    titles = [t for t, _ in rm.analysis_steps()]
    assert titles[0].startswith("EDA") and any("PPR" in t for t in titles)

    seen = []                                                # 곡선 스크립트에는 최종 모드에서만 --final
    monkeypatch.setattr(rm, "_script_main", lambda path: (lambda argv: seen.append((path.name, argv))))
    for mode in ("check", "final"):
        dict(rm.analysis_steps(mode))["곡선: damping·seed 개수"]()
        rm.error_analysis_step(mode)()
    assert seen == [("run_sensitivity.py", []), ("error_analysis.py", []),
                    ("run_sensitivity.py", ["--final"]), ("error_analysis.py", ["--final"])]

import importlib.util
import sys
from pathlib import Path

# src/engine.py(박병언)가 아직 없으면 테스트에서만 참조 엔진을 끼워 넣는다
if importlib.util.find_spec("src.engine") is None:
    sys.path.insert(0, str(Path(__file__).parent))
    import _ref_engine  # noqa: E402

    sys.modules["src.engine"] = _ref_engine

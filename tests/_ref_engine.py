"""테스트 전용 참조 엔진 (확정안 7절 규격). src/engine.py가 아직 없을 때만 쓴다.

실제 엔진이 생기면 conftest가 실제 엔진을 쓰므로, 아래 PPR 테스트가 엔진 검증도 겸한다.
"""
import numpy as np
import scipy.sparse as sp


def power_iteration(A, teleport, damping=0.85, dangling="teleport", tol=1e-8, max_iter=500):
    A = sp.csr_matrix(A)
    n = A.shape[0]
    out = np.diff(A.indptr)
    dang = out == 0
    inv = np.zeros(n)
    inv[~dang] = 1.0 / out[~dang]
    MT = (sp.diags(inv) @ A).T.tocsr()
    v = np.asarray(teleport, float) / np.sum(teleport)
    d = v if dangling == "teleport" else np.full(n, 1.0 / n)
    r = v.copy()
    for it in range(1, max_iter + 1):
        new = damping * (MT @ r + r[dang].sum() * d) + (1 - damping) * v
        new /= new.sum()
        if np.abs(new - r).sum() < tol:
            return new, it
        r = new
    return r, max_iter

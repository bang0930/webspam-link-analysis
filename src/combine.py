"""담당: 박병언 (팀 작성 코드).

단순 순위 평균, 확정안 6절, 9절. 강의 11강 p73-80, 4강 p24-58.
"""
import numpy as np
from scipy.stats import rankdata


def rank_average(scores):
    """높을수록 스팸인 점수들을 평균 동률 순위로 바꿔 같은 가중치로 평균한다.

    입력은 방법별 1차원 배열의 목록이다. 반환도 높을수록 스팸이다.
    """
    a = np.asarray(scores, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError('비어 있지 않고 유한한 동일 길이 점수 배열들이 필요합니다')
    return np.mean([rankdata(s, method='average') for s in a], axis=0)

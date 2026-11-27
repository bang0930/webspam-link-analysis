# 링크 기반 웹 스팸 탐지 (webspam-link-analysis)

SCE331 Data Mining Team Project

> 학번, 최종 보고서와 영상, 제출 commit hash는 제출 전에 확정합니다.

## Overview

웹사이트 사이의 **링크 구조만으로** 스팸 사이트를 찾고, 어떤 방법이 가장 잘 찾는지 비교합니다.
검색엔진의 PageRank는 "링크를 많이 받는 사이트가 중요하다"는 가정 때문에, 스패머가 링크 농장(link farm)으로 순위를 조작할 수 있습니다.
이 프로젝트는 WEBSPAM-UK2007 host 그래프(사이트 114,529개)에 수업에서 배운 링크 분석·유사 항목 탐색 알고리즘을 적용해 이 문제를 다룹니다.

**연구 질문**

1. PageRank 순위는 스팸에 얼마나 오염되는가? TrustRank는 이를 줄이는가?
2. 정상 사이트에서 신뢰를 퍼뜨리기(TrustRank) vs 스팸에서 의심을 퍼뜨리기(PPR), 무엇이 더 잘 찾는가?
3. 링크를 따라가는 방법 vs 링크 구조가 비슷한 사이트를 찾는 방법(MinHash/LSH), 무엇이 낫고 서로 다른 스팸을 찾는가?
4. 사람이 seed를 몇 개 검수해야 충분한가?

자세한 설계는 [프로젝트 확정안](docs/STRUCTURE.md)에 있습니다.

## Team

| 이름 | 학번 | 담당 |
|---|---|---|
| 김현중 | 🚧 | 데이터 불러오기, MinHash/LSH, baseline, EDA, README |
| 박병언 | 🚧 | 공통 엔진, PageRank, TrustRank, Spam Mass, seed 실험 |
| 강민수 | 🚧 | 데이터 분할, 채점, PPR, 최종 비교표, error analysis |

팀 번호: G06

## Repository layout

```
webspam-link-analysis/
├── README.md
├── requirements.txt           # 라이브러리 버전 고정
├── splits.json                # SET1 도메인 단위 5-fold (hostid와 fold 번호만)
├── eval_set.json              # 최종 채점 대상 hostid
├── docs/
│   ├── STRUCTURE.md           # 프로젝트 확정안
│   ├── AI_USAGE.md            # AI 사용 기록
│   └── REFERENCES.md          # IEEE 형식 참고문헌 초안
├── scripts/
│   └── download_data.py       # 원본 데이터 다운로드
├── src/                       # 팀 작성 알고리즘과 분석 함수
│   ├── config.py              # 경로, 랜덤 seed, 기본 parameter
│   ├── load_graph.py          # 그래프(0·1 희소행렬)·라벨 불러오기, 캐시
│   ├── splits.py              # 도메인 단위 분할, 채점 대상 고정
│   ├── seeds.py               # TrustRank seed 선택
│   ├── engine.py              # 공통 power iteration 엔진
│   ├── pagerank.py
│   ├── trustrank.py           # TrustRank, Spam Mass
│   ├── ppr.py
│   ├── lsh.py                 # MinHash, LSH
│   ├── baselines.py
│   ├── evaluate.py            # 지표, bootstrap 신뢰구간, 검증 5-fold
│   ├── eda.py                 # 데이터 통계표, degree, 판정자 일치도, 링크 가정 확인
│   ├── results_io.py          # 점수 저장과 불러오기, src 해시, dirty 기록
│   ├── sensitivity.py         # seed와 damping 분석 계산
│   ├── combine.py             # 평균 동률 순위의 단순 평균
│   └── figures.py             # 보고서 그림 (전부 여기서 생성)
├── experiments/
│   ├── run_main.py            # 고정 설정의 점수, 채점, 표, 그림
│   ├── run_sensitivity.py     # seed와 damping 분석 곡선
│   ├── tie_check.py           # 동률, seed 도달 가능성, 순위 구간 진단
│   ├── run_combine.py         # TrustRank, PPR, LSH 순위 평균 결합
│   ├── record_environment.py  # 실행 환경과 점수 메타데이터 기록
│   ├── cv_ppr.py              # PPR 설정의 SET1 검증
│   └── error_analysis.py      # 놓친 스팸과 오탐 정상 분석
├── tests/                     # pytest (합성 데이터 + 실제 데이터 통계 대조)
├── notebooks/                 # EDA·확인용 (보고서 그림은 만들지 않음)
├── results/
│   ├── tables/                # 결과 표 (.csv)
│   └── figures/               # 결과 그림 (.png)
└── data/                      # 저장소에 포함하지 않음 (다운로드 스크립트로 생성)
```

## Environment / Setup

- Python 3.12 이상 (`requirements.txt`의 numpy·scipy·networkx 버전이 3.12+를 요구)
- 라이브러리 설치 (가상환경 권장)

```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

- 아래 명령은 저장소 루트에서 실행합니다.
- 테스트: `python -m pytest tests -q` (데이터를 받은 뒤 실행하면 실제 데이터 통계까지 확인)

### 기록된 실험 환경

2026-11-27 기록 기준으로 macOS 26.7.3, Apple M4, 물리 코어 10개, 논리 코어 10개, 메모리 24 GiB, Python 3.13.3을 사용했습니다.
numpy 2.5.5, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1, networkx 3.7, matplotlib 3.11.2는 requirements.txt의 고정 버전과 일치합니다.
자세한 기록은 `results/tables/environment.csv`에 있습니다.

환경 기록은 다음 명령으로 갱신합니다.

```bash
python experiments/record_environment.py
```

`seconds`는 점수 저장용 캐시 콜백에 걸린 시간입니다. 공유 계산이 다른 행에 포함되면 0에 가까울 수 있습니다.
`reported_algorithm_seconds`는 점수 JSON의 `info.seconds`이며, TrustRank에서는 신뢰 전파 반복만 포함합니다.
`seconds_scope`, `reported_algorithm_seconds_scope`, `timing_note`를 함께 확인합니다.
이 값들을 별도 측정 없이 알고리즘 단독 실행 시간의 공정한 비교로 해석하지 않습니다.

## Data

WEBSPAM-UK2007 (Yahoo! Research Barcelona, Web Spam Challenge 2008)
- 배포처: https://chato.cl/webspam/datasets/uk2007/ (접근일 2026-11-16)
- 라벨 라이선스: CC BY-NC-SA 3.0 — 원본 데이터는 저장소에 포함하지 않음

### 받는 방법

```bash
python scripts/download_data.py
```

표준 라이브러리만 사용합니다 (Python 3.10 이상이면 실행 가능). `data/raw/` 아래에 저장되며, 이미 받은 파일은 건너뜁니다.

받은 데이터가 정상인지 확인:

```bash
python -m src.load_graph
```

host 수, edge 수, dangling 수, 라벨 수를 알려진 값과 대조해 `[ok ]`/`[err]`로 출력합니다. 처음 실행 때 `data/processed/`에 그래프 캐시를 만들고, 원본이 바뀌면 캐시를 다시 만듭니다 (`--force`로 강제 재생성).

### 받는 파일

| 파일 | 내용 |
|---|---|
| `uk-2007-05.hostgraph_weighted.graph-txt.gz` | host 단위 링크 그래프. 첫 줄은 host 수(114,529), 이후 i번째 줄은 host i의 out-link 목록 `목적지:링크수` |
| `uk-2007-05.hostnames.txt.gz` | `hostid hostname` 대응표 (0–114,528) |
| `webspam-uk2007-set1-1.0/` | 학습용 라벨 (SET1) |
| `webspam-uk2007-set2-1.0/` | 테스트용 라벨 (SET2) |

라벨 파일 형식: `hostid label spamicity assessments` (label ∈ {nonspam, spam, undecided})

| | nonspam | spam | undecided |
|---|---|---|---|
| SET1 (train) | 3,776 | 222 | 277 |
| SET2 (test) | 1,933 | 122 | 149 |

SET1/SET2는 3단계 도메인 기준으로 무작위 분할되어 있으며 겹치는 host는 없습니다.

### 데이터 통계 (2026-11-16 확인)

| 항목 | 값 |
|---|---|
| host / 방향 있는 edge | 114,529 / 1,836,441 |
| 나가는 링크가 없는 host (dangling) | 49,379 (43.1%) |
| 라벨된 스팸 중 나가는 링크가 없는 host | 189 / 344 (55%) |

### 데이터 탐색 (EDA)

```bash
python -m src.eda
```

데이터 통계표, 스팸·정상의 degree 요약(SET1), 판정자 간 일치도(SET1·SET2), TrustRank·PPR 가정 확인(SET1 라벨 host 사이의 링크)을 `results/tables/eda_*.csv`에, degree 분포 그림을 `results/figures/fig_degree_distribution.png`에 저장합니다. 방법 설계에 영향을 줄 수 있는 분석은 SET1만 사용합니다.

## Preprocessing

- undecided 라벨 제외
- 링크는 있음/없음(0·1)으로 사용 (링크 수 가중치는 사용하지 않음)
- 도메인(3단계 도메인) 단위로 SET1을 5-fold로 나눔 → `splits.json`
- 채점 대상 = SET2 라벨 − undecided − 신뢰 도메인 host → `eval_set.json`
- 두 파일에는 hostid와 fold 번호만 저장하고, 라벨은 실행할 때 원본에서 읽음 (라벨 재배포 방지)
- 최종 채점 대상: 1,921개 host (스팸 121개). SET2 스팸 중 1개는 신뢰 도메인이라 빠짐

## Algorithms

| 알고리즘 | 강의 근거 | 역할 | 구현 |
|---|---|---|---|
| PageRank | 11강 p8–40 | 순위 오염 기준선 | 직접 구현 (power iteration, 희소행렬), networkx로 검증 |
| TrustRank | 11강 p73–78 | 정상 seed에서 신뢰 전파 | 공통 엔진 사용 |
| Personalized PageRank | 11강 p43–47, p56 | 스팸 seed에서 의심 전파 (링크 방향, 역방향 = Anti-TrustRank 방식) | 공통 엔진 사용 |
| MinHash + LSH | 4강 p24–58 | 들어오는 링크 집합이 비슷한 host 묶음 탐지 | 직접 구현 |
| Spam Mass (분석 지표) | 11강 p79–80 | 스팸에서 온 PageRank 비율 | PageRank·TrustRank 결과로 계산 |

Baseline: 무작위, in-degree, out-degree, 1단계 이웃 스팸 비율(두 방향)

LSH 설정 검증과 점수 계산에는 SET1 라벨만 사용합니다. 공통 `load_labels()`는 SET2 파일도 읽지만, 반환된 SET2 라벨은 이 단계의 계산과 설정 선택에 사용하지 않습니다.

| 명령 | 내용 | 결과 | 시간·메모리 (참고) |
|---|---|---|---|
| `python -m src.lsh` | 설정 15개의 후보 쌍 수·커버리지 | `results/tables/lsh_candidate_counts.csv` | 약 1.5분, 최대 약 3.3GB |
| `python -m src.lsh validate` | 검증 5-fold PR-AUC로 (b, r)·최소 집합 크기 선택 | `results/tables/lsh_validation.csv` | 약 1.5분 |
| `python -m src.lsh scurve` | S-curve(이론 vs 실제)와 recall | `results/tables/lsh_scurve*.csv`, `results/figures/fig_lsh_scurve.png` | 약 30초, 최대 약 4.4GB |

PageRank, TrustRank, PPR은 teleport가 다른 같은 PageRank 수식 계열이며 하나의 공통 엔진을 공유합니다.
PPR의 스팸 seed 전파는 주제 기반 seed 대신 SET1 스팸 seed를 사용하도록 구성한 프로젝트 변형입니다.
전체 host 그래프를 사용하는 transductive 실험이며, seed와 설정 선택에는 검증 대상이나 SET2의 라벨을 사용하지 않습니다.

강의 11강 46쪽 예제는 원문 표기값과 계산 기대값을 구분합니다. 엔진의 계산값은 독립적인 닫힌 해로 검증하고,
소수 표기가 일관된 반올림 또는 버림 규칙을 따르지 않는 원문 표와는 성분별 오차 0.01 미만으로 대조합니다.

## Running experiments

### Check 모드

check 결과는 시험용이며 보고서 숫자가 아닙니다. 기본 채점 대상은 SET1 fold 0에서 신뢰 도메인 host를 제외한 집합입니다.

```bash
python experiments/run_main.py
python experiments/run_sensitivity.py
python experiments/tie_check.py
python experiments/run_combine.py
python -m pytest -q
```

결과 표와 그림은 `results/check/`에 저장합니다. 민감도 기본 실행은 작은 설정으로 fold 0만 사용하고 SET1 라벨 파일만 읽습니다.
`run_sensitivity.py --force`도 SET1만 사용하며, 이 스크립트는 `--final`을 받은 경우에만 SET2를 분석합니다. `--full` 별칭은 허용하지 않습니다.
다른 check 스크립트는 공통 라벨 로더를 통해 SET2 파일을 함께 읽을 수 있으나, SET2 탐지 채점이나 설정 선택에는 사용하지 않습니다.

### 최종 결과 재현

코드를 커밋하고 `src/`에 미커밋 변경이 없는 상태에서 실행합니다. 설정은 SET1 검증으로 확정하며, SET2 결과로 다시 선택하지 않습니다.

```bash
python scripts/download_data.py
python experiments/run_main.py --final --force --analysis
python experiments/tie_check.py --final --force
python experiments/run_combine.py --final --force
python experiments/record_environment.py
python -m pytest -q
```

첫 번째 실험 명령은 EDA, LSH 후보 수와 검증 및 S-curve, PPR 검증, 민감도 분석을 수행한 뒤 기본 최종표와 그림, error analysis를 생성합니다.
`--force`는 민감도 분석에도 전달됩니다. 동률 진단, 단순 결합, 환경 기록까지 재현하려면 위의 나머지 명령도 실행해야 합니다.
EDA의 SET2 통계와 판정자 일치도는 데이터 기술용이며 설정 선택에 사용하지 않습니다.

민감도 분석만 최종 모드로 실행하는 명령은 다음과 같습니다.

```bash
python experiments/run_sensitivity.py --final --force
```

`FINAL` 설정은 `experiments/run_main.py`에 있습니다. 현재 damping은 0.85, PPR은 forward와 raw, LSH는 최소 집합 크기 1과 (b, r)=(32, 4)입니다.
PageRank의 참고용 탐지 방향은 부호 반전이며, TrustRank는 SET1 정상 seed 전부와 신뢰 부호 반전을 핵심 비교에 사용합니다.

점수는 `results/scores/<모드>/`의 `.npy`와 `.json`으로 저장합니다. JSON에는 설정, 시각, 실행 시간, 반복 횟수, commit, src 해시, dirty 상태를 기록합니다.
캐시가 있으면 재사용하며 `--force`로 다시 계산합니다. 기존 캐시 로더는 src 해시 불일치 시 경고 후 재계산합니다.
최종 모드에서는 `src/`가 dirty이면 실행을 거부합니다. 원본 데이터와 점수 캐시는 Git에 올리지 않습니다.
무작위 반복 seed는 42부터 46까지입니다.

## Evaluation

- 최종 평가 집합은 SET2 라벨 host에서 undecided와 신뢰 도메인 host를 제외한 1,921개이며 스팸은 121개입니다.
- SET1 도메인 단위 5-fold에서 설정을 선택합니다. 모든 라벨 기반 계산은 seed용 fold 라벨만 사용합니다.
- 모든 탐지 점수는 높을수록 스팸입니다. PR-AUC는 average precision이며 ROC-AUC, Precision@50과 Precision@100을 함께 기록합니다.
- Precision@K는 경계 동률의 기댓값으로 계산합니다. 구체적인 상위 K 목록과 순위 구간은 seed 42 무작위 순서로 동률을 끊습니다.
- 신뢰구간은 도메인 bootstrap 1,000회로 계산합니다. 방법 차이는 같은 표본에서 구한 PR-AUC 차이의 95% 구간으로 판단하며 0을 포함하면 차이 없음으로 기록합니다.
- 반복 방법의 차이 비교는 seed 42 점수를 사용하고, 5회 반복 평균과 표준편차는 별도로 기록합니다.
- seed 개수, seed 방식, damping 곡선은 분석용입니다. SET2 곡선으로 설정을 선택하거나 최종표 설정을 바꾸지 않습니다.
- 단순 결합은 가중치 없는 평균 동률 순위이며, 최종표나 알고리즘 개수에 포함하지 않습니다.

## Reproducing tables and figures

| 결과 | 파일 | 실행 |
|---|---|---|
| 데이터 통계, degree, 판정자 일치도, 링크 가정 확인 | `results/tables/eda_*.csv`, `results/figures/fig_degree_distribution.png` | `python -m src.eda` |
| LSH 후보 수 | `results/tables/lsh_candidate_counts.csv` | `python -m src.lsh counts` |
| LSH 검증 | `results/tables/lsh_validation.csv` | `python -m src.lsh validate` |
| LSH S-curve와 recall | `results/tables/lsh_scurve*.csv`, `results/figures/fig_lsh_scurve.png` | `python -m src.lsh scurve` |
| PPR 검증 | `results/tables/cv_ppr.csv`, `results/tables/cv_ppr_selection.json` | `python experiments/cv_ppr.py` |
| 기본 최종표, 비교, 구간, 잡은 스팸, 하위 집합 | `results/tables/final_{table,comparisons,ranking,caught,subsets}.csv` | `python experiments/run_main.py --final --force` |
| 최종 비교표, PR 곡선, 잡은 스팸 겹침, 순위 구간 그림 | `results/figures/fig_final_pr_auc.png`, `fig_pr_curves.png`, `fig_caught_overlap.png`, `fig_rank_buckets.png` | 위와 같은 명령 |
| error analysis | `results/tables/final_error_*.csv` | 위 명령에 포함, 개별 실행은 `python experiments/error_analysis.py --final` |
| seed 개수, seed 방식, damping 분석 | `results/tables/sensitivity_*.csv`, `results/figures/fig_seed_curve.png`, `fig_seed_methods.png`, `fig_damping_curve.png` | `python experiments/run_sensitivity.py --final --force` |
| 동률과 도달 가능성 진단 | `results/tables/tie_check.csv`, `results/figures/fig_rank_buckets.png` | `python experiments/tie_check.py --final --force` |
| 순위 평균 결합 | `results/tables/combination.csv` | `python experiments/run_combine.py --final --force` |
| 실행 환경 | `results/tables/environment.csv` | `python experiments/record_environment.py` |

최종 error analysis의 사례와 host 목록은 hostname을 포함하므로 Git에 올리지 않습니다.
LSH 허브 제거 변형과 전체 상위 K host 집합의 방법별 겹침은 현재 구현 및 결과에 포함되지 않습니다.

## Results

아래는 2026-11-27에 저장한 고정 설정의 SET2 결과입니다. LSH와의 차이 비교는 seed 42를 사용하므로 아래 표에도 seed 42 값을 적었습니다.
LSH 5회 평균과 표준편차는 `final_table.csv`에 따로 기록되어 있습니다.

| 방법 | PR-AUC |
|---|---:|
| PageRank, 참고용 탐지 점수 | 0.074533 |
| TrustRank, SET1 정상 seed와 신뢰 부호 반전 | 0.076070 |
| PPR, forward와 raw | 0.086243 |
| LSH, 최소 집합 크기 1과 (32, 4), seed 42 | 0.113362 |
| 단순 순위 평균 결합, 추가 분석 | 0.159424 |

TrustRank와 PPR의 차이 구간은 0을 포함합니다. LSH는 TrustRank보다 높고, 신뢰 도메인 seed TrustRank는 SET1 seed보다 높았습니다.
초기 가설과 반대되는 결과도 그대로 보고하며, seed 수가 많아지면 항상 개선된다고 해석하지 않습니다.

결합과 TrustRank, PPR, LSH의 SET2 PR-AUC 차이 95% 구간은 각각 [0.037560, 0.153180], [0.022632, 0.139795], [0.013961, 0.087027]입니다.
검증 평균에서도 세 차이 구간이 모두 0보다 큽니다. 이 결과는 현재 데이터와 고정된 설정에 대한 결과입니다.

TrustRank 0점 host와 SET1 정상 seed에서 도달 불가능한 host는 전체 4,831개, 평가 집합 93개로 정확히 일치했습니다.
순위 구간 경계에는 동률이 없었습니다. TrustRank 3번부터 8번 구간의 스팸 수는 11, 12, 11, 11, 11, 11개이며,
같은 비율의 반복은 동률 분할 때문이 아닙니다. 9번 구간은 PageRank와 TrustRank 모두 스팸 22개지만 host 겹침은 149/192개입니다.

## References and external code

IEEE 형식 초안과 인용 위치는 [docs/REFERENCES.md](docs/REFERENCES.md)에 있습니다. 강의 11 표지의 강의자는 Sael Lee, 이슬로 확인되며, 참고문헌의 미확인 표기는 제출 전에 정리합니다.

- WEBSPAM-UK2007 dataset, Yahoo! Research Barcelona (Web Spam Challenge 2008)
- PageRank (Page, Brin, Motwani, Winograd, 1999)
- Topic-sensitive PageRank (Haveliwala, 2002)
- TrustRank (Gyöngyi, Garcia-Molina, Pedersen, 2004)
- Spam Mass (Gyöngyi, Berkhin, Garcia-Molina, Pedersen, 2006)
- Anti-TrustRank (Krishnan, Raj, 2006)
- MinHash (Broder, 1997)
- J. Leskovec, A. Rajaraman, J. Ullman, *Mining of Massive Datasets*
- SCE331 강의 자료 4강(LSH), 11강(Link Analysis)

외부 라이브러리: numpy, scipy, pandas, scikit-learn(분할·평가 지표), networkx(PageRank 검증), matplotlib, pytest — 버전은 `requirements.txt`에 고정

외부 코드를 가져오거나 수정해 쓴 경우 해당 파일 머리말에 출처와 라이선스를 적습니다. 현재 별도 외부 소스 코드를 가져온 항목은 기록되어 있지 않습니다.

## AI-use statement

이 프로젝트는 일부 작업에 생성형 AI 도구를 사용했습니다. 날짜, 도구·모델, 도움받은 작업, 영향받은 파일, 검증 방법은 [docs/AI_USAGE.md](docs/AI_USAGE.md)에 기록합니다. 제출한 작업에 대한 책임은 팀에 있습니다.

## 협업 규칙

### 기본

- 작업 시작 전에 항상 `git pull`
- 각자 **자기 파일만** 수정
  - 다른 사람 파일을 고쳐야 하면 담당자에게 먼저 말하기
  - 같이 쓰는 파일(`README.md`, `src/config.py`, `experiments/run_main.py`)은 고치기 전에 단톡방에 말하기
- 커밋 메시지는 **한글 한 줄**로, 무엇을 했는지
- `data/`(원본 데이터)는 절대 올리지 않음
- AI를 쓰면 `docs/AI_USAGE.md`에 한 줄 추가

### 브랜치와 PR

- 기능마다 브랜치 하나 → PR (예: `feat/engine`, `feat/lsh`, `fix/splits`)
- PR은 작게 (파일 1~3개, 하루치 작업)
- **1명 승인**이면 merge
- 24시간 안에 리뷰가 없으면 단톡방에 알리고 본인이 merge
- 문서 수정(README 오타, `AI_USAGE.md` 추가 등)은 PR 없이 바로 push 가능

### 리뷰

그 코드를 가져다 쓰는 사람이 리뷰합니다.

| 코드 | 작성 | 리뷰 |
|---|---|---|
| 공통 엔진, PageRank, TrustRank | 박병언 | 강민수 |
| 분할, 채점, PPR | 강민수 | 김현중 |
| load_graph, LSH, baseline | 김현중 | 박병언 |

리뷰에서 확인하는 것

- ✅ [확정안](docs/STRUCTURE.md)과 맞는지
- ✅ 강의 예제 검증을 통과했는지
- ❌ 코드 스타일은 보지 않음

### PR 설명 (3줄)

```text
무엇을: 공통 엔진 power_iteration 추가
확인한 것: 강의 11강 p16, p33 예제와 값 일치 / networkx와 차이 1e-10 이하
확정안과 다른 점: 없음 (있으면 적기)
```

### 이슈

- **바로 못 고치는 문제**(버그, 이상한 숫자) → 이슈로 남기고 PR에 `closes #번호`
- **11/24 세부 값 결정 기록** → 보고서 실험 설계의 근거로 사용
- 진행 체크리스트 이슈 하나를 함께 체크
- 할 일마다 이슈를 만들지는 않음

### 결과 파일

- 올리는 것: 코드, `splits.json`, `eval_set.json`(hostid와 fold 번호만), 표(`.csv`), 그림(`.png`)
- 올리지 않는 것: `data/`, `results/scores/` (다시 만들 수 있음)
- 최종 채점과 12/4 재현 테스트는 **커밋을 모두 마친 상태**(`dirty`가 아닐 때)에서 `--force`로 실행

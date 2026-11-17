# 링크 기반 웹 스팸 탐지 (webspam-link-analysis)

SCE331 Data Mining Team Project

> 🚧 표시는 프로젝트 진행에 따라 채울 예정인 항목입니다.

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

팀 번호: 🚧

## Repository layout

```
webspam-link-analysis/
├── README.md
├── requirements.txt           # 라이브러리 버전 고정
├── splits.json                # SET1 도메인 단위 5-fold (hostid와 fold 번호만)
├── eval_set.json              # 최종 채점 대상 hostid
├── docs/
│   ├── STRUCTURE.md           # 프로젝트 확정안
│   └── AI_USAGE.md            # AI 사용 기록
├── scripts/
│   └── download_data.py       # 원본 데이터 다운로드
├── src/                       # 🚧 일부 작성 중
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
│   ├── evaluate.py            # 지표, bootstrap 신뢰구간
│   └── figures.py             # 보고서 그림
├── experiments/               # 🚧
│   ├── run_main.py            # 점수 → 채점 → 표 → 그림
│   └── run_sensitivity.py     # 분석 곡선
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

- 모든 명령은 저장소 루트에서 `python -m ...` 형태로 실행합니다.
- 테스트: `python -m pytest tests -q` (데이터를 받은 뒤 실행하면 실제 데이터 통계까지 확인)

- 🚧 실험 환경 (보고서 기재용): OS, CPU, 메모리, Python 버전, 주요 라이브러리 버전

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

## Preprocessing

- undecided 라벨 제외
- 링크는 있음/없음(0·1)으로 사용 (링크 수 가중치는 사용하지 않음)
- 도메인(3단계 도메인) 단위로 SET1을 5-fold로 나눔 → `splits.json`
- 채점 대상 = SET2 라벨 − undecided − 신뢰 도메인 host → `eval_set.json`
- 두 파일에는 hostid와 fold 번호만 저장하고, 라벨은 실행할 때 원본에서 읽음 (라벨 재배포 방지)
- 🚧 최종 채점 대상 host 수·스팸 수

## Algorithms

| 알고리즘 | 강의 근거 | 역할 | 구현 |
|---|---|---|---|
| PageRank | 11강 p8–40 | 순위 오염 기준선 | 직접 구현 (power iteration, 희소행렬), networkx로 검증 |
| TrustRank | 11강 p73–78 | 정상 seed에서 신뢰 전파 | 공통 엔진 사용 |
| Personalized PageRank | 11강 p43–47, p56 | 스팸 seed에서 의심 전파 (링크 방향, 역방향 = Anti-TrustRank 방식) | 공통 엔진 사용 |
| MinHash + LSH | 4강 p24–58 | 들어오는 링크 집합이 비슷한 host 묶음 탐지 | 직접 구현 |
| Spam Mass (분석 지표) | 11강 p79–80 | 스팸에서 온 PageRank 비율 | PageRank·TrustRank 결과로 계산 |

Baseline: 무작위, in-degree, out-degree, 1단계 이웃 스팸 비율(두 방향)

PageRank·TrustRank·PPR은 같은 수식 계열(teleport만 다른 PageRank)이며, 하나의 공통 엔진을 공유합니다.

## Running experiments

🚧 코드 완성 후 확정

```bash
python scripts/download_data.py          # 1. 데이터 받기
python experiments/run_main.py           # 2. 점수 → 채점 → 표 → 그림
python experiments/run_sensitivity.py    # 3. 분석 곡선 (damping, seed 개수)
```

- 랜덤 seed: 42 (반복 실험은 42–46)
- 계산 결과는 `results/scores/`에 캐시되며, `--force`로 다시 계산합니다.
- 최종 결과는 커밋을 모두 마친 상태에서 `--force`로 실행한 결과입니다.

## Evaluation

- 채점 대상: `eval_set.json` (SET2 라벨 host − undecided − 신뢰 도메인 host)
- 세부 parameter는 SET1 검증 fold로만 고르고, SET2는 최종 채점에 한 번만 사용
- 지표: PR-AUC(average precision, 메인), ROC-AUC, Precision@50·100
- 신뢰구간: 도메인 단위 bootstrap 1,000회, 방법 간 비교는 PR-AUC 차이의 95% 구간
- PageRank는 탐지 점수가 아니라 순위 구간별 스팸 비율로 평가

## Reproducing tables and figures

🚧 결과 생성 후 작성

| 결과 | 파일 | 만드는 명령 |
|---|---|---|
| 데이터 통계표 | `results/tables/` | `run_main.py` |
| 최종 비교표 | `results/tables/` | `run_main.py` |
| PageRank vs TrustRank 구간별 스팸 비율 | `results/figures/` | `run_main.py` |
| PR 곡선 | `results/figures/` | `run_main.py` |
| seed 개수 곡선 | `results/figures/` | `run_sensitivity.py` |
| damping 곡선 | `results/figures/` | `run_sensitivity.py` |

## Results

🚧 최종 채점 후 작성

## References and external code

🚧 서지 정보는 원문 확인 후 IEEE 형식으로 정리

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

외부 코드를 가져오거나 수정해 쓴 경우 해당 파일 머리말에 출처와 라이선스를 적습니다. 🚧 현재 없음

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

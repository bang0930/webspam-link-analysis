# 링크 기반 웹 스팸 탐지 — 프로젝트 확정안

> SCE331 데이터마이닝 팀 프로젝트 · 최종 초안 (2026-11-16)
> 팀원: 김현중 · 박병언 · 강민수
> 저장소: https://github.com/bang0930/webspam-link-analysis

---

## 0. 한 줄 요약

**링크 구조만으로 스팸 사이트를 찾고, 어떤 방법이 가장 잘 찾는지 비교한다.**

- PageRank · TrustRank · PPR은 Topic Ideas **Link Analysis C** 구성을 바탕으로 한다.
- PPR은 주제 seed 대신 **SET1 스팸 seed**로 의심을 퍼뜨리도록 변형했다. 링크 방향과 역방향을 둘 다 실험하며, 역방향은 Anti-TrustRank와 같은 방식이다.
- 여기에 **MinHash/LSH**를 추가했다. 일부 알고리즘이 같은 것으로 쳐질 경우를 대비한 보험이자 차별화 요소다.
- 4개 모두 정식 알고리즘이고, 전부 최종 비교표에 들어간다.

---

## 1. 연구 질문과 가설

**연구 질문**

| # | 연구 질문 | 가설 |
|---|---|---|
| Q1 | PageRank 순위는 스팸에 얼마나 오염되는가? TrustRank는 이를 줄이는가? | H1. TrustRank 상위권의 스팸 비율이 PageRank보다 낮다 |
| Q2 | 정상에서 신뢰를 퍼뜨리기(TrustRank) vs 스팸에서 의심을 퍼뜨리기(PPR), 무엇이 더 잘 찾는가? | H2. TrustRank가 PPR보다 잘 찾는다 (링크 방향 PPR의 경우, SET1 스팸 seed의 56%가 나가는 링크가 없어 의심 전파가 약하므로) |
| Q3 | 링크를 따라가는 방법 vs 링크 구조가 비슷한 것을 찾는 방법(LSH), 무엇이 낫고 서로 다른 스팸을 찾는가? | H3. LSH는 전파 방법이 놓친 스팸(링크 농장) 일부를 잡는다 |
| Q4 | 사람이 seed를 몇 개 검수해야 충분한가? | H4. 일정 개수 이후 성능이 포화된다 |

**보조 가설** (핵심 비교와 연결)

| # | 가설 | 확인 방법 |
|---|---|---|
| H5 | 라벨을 쓰는 방법이 라벨을 안 쓰는 방법보다 잘 찾는다 | TrustRank(SET1 정상 seed 전부 3,776개) vs TrustRank(신뢰 도메인 seed 전부 8,877개). seed 수가 달라서, **같은 개수(50개, 200개)에서의 비교를 보조 근거로 함께 보고**한다. 신뢰 도메인 seed도 알려진 스팸 1개를 SET1 라벨로 빼므로 "라벨을 전혀 안 씀"은 엄밀히 아님을 보고서에 적는다 |
| H6 | 여러 단계 전파가 1단계 이웃보다 잘 찾는다 | PPR vs 같은 방향의 1단계 이웃 baseline |

**동기**: 강의 11강 p67–70의 link farm 분석에 따르면, damping 0.85에서 스패머는 자기 PageRank를 약 3.6배 증폭할 수 있다.

**범위**: host 단위 링크 그래프만 쓴다. 페이지 본문과 링크 수 가중치는 범위 밖이다.

결과가 가설과 반대로 나와도 그대로 보고한다.

---

## 2. 데이터 요약 (2026-11-16 실측)

| 항목 | 값 |
|---|---|
| 출처 | WEBSPAM-UK2007, https://chato.cl/webspam/datasets/uk2007/ (접근일 2026-11-16) |
| 그래프 | host 114,529개, 방향 있는 edge 1,836,441개, 자기 자신 링크 0개 |
| dangling (나가는 링크 0) | 49,379개 (43.1%) |
| 들어오는 링크 0 / 완전 고립 | 4,766개 / 3,380개 |
| SET1 (학습) | 정상 3,776 / 스팸 222 / undecided 277 |
| SET2 (테스트) | 정상 1,933 / 스팸 122 / undecided 149 |
| 라벨 라이선스 | CC BY-NC-SA 3.0 (원본은 저장소에 올리지 않음) |

**설계에 영향을 준 사실**

| 사실 | 영향 |
|---|---|
| 라벨된 스팸 344개 중 189개(55%)가 나가는 링크 없음 | PPR 두 방향 모두 실험, LSH는 들어오는 링크 집합 사용 |
| 스팸 중 들어오는 링크 있음 92%, 나가는 링크 있음 45% | LSH 집합 = 들어오는 링크로 확정 |
| 신뢰 도메인(.gov.uk, .ac.uk, .nhs.uk, .sch.uk, .police.uk, .mod.uk) 8,877개, 라벨된 것 중 스팸 1개 | 라벨 없는 TrustRank seed로 사용 가능 (강의 p78) |
| 나가는 링크 상위 1% 사이트의 스팸 비율 8.6% (전체 5.7%) | LSH 허브 제거는 본 실험에서 하지 않음 |
| 들어오는 링크 집합이 완전히 같은 사이트 27,268개. 모든 묶음의 쌍 합계 약 403만 쌍 (최대 묶음 1,884개 하나가 약 177만 쌍) | LSH 전에 같은 집합끼리 먼저 묶음 |
| seed로 쓰는 SET1 스팸 222개 중 125개(56.3%)가 나가는 링크 없음 | 링크 방향 PPR에서 실제로 퍼뜨리는 seed는 약 97개 |

**데이터 한계** (보고서에 기록)

- 2007년 데이터라 오래됐다.
- 라벨된 host는 전체의 약 6%이고, 무작위 표본이라는 보장이 없다.
- 판정자끼리 의견이 갈린 경우가 있다 (판정자 간 일치도 보고).
- host 단위로 합쳐서 페이지 단위 정보는 잃는다.

---

## 3. 알고리즘과 수업 근거

인정 기준: ① `SCE331_DM_Algorithms_by_Topic.pdf` 목록에 있음, ② 수업 슬라이드 5장 이상

| 알고리즘 | 목록 | 강의 슬라이드 | 역할 |
|---|---|---|---|
| PageRank | 6. Link Analysis | 11강 p8–40 (power iteration p17–18, dead end p22–34, 희소행렬 p36–40) | 순위 오염 기준선 |
| TrustRank | 6. Link Analysis | 11강 p73–78 (seed 선택 p77–78) | 정상 seed에서 신뢰 전파 |
| Personalized PageRank | 6. Link Analysis | 11강 p43–47, p56 | 스팸 seed에서 의심 전파 |
| MinHash + LSH | 2. Similar-Item Search | 4강 p24–58 (MinHash p24–38, LSH p39–58) | 들어오는 링크가 비슷한 host 묶음 찾기 |
| Spam Mass (분석 지표) | TrustRank 부속 | 11강 p79–80 | 스팸에서 온 PageRank 비율 |

- 강의(11강 p73, p75)에서 TrustRank는 "신뢰 페이지로 teleport하는 topic-specific PageRank"다. 즉 PageRank·TrustRank·PPR은 **같은 수식 계열**이고 teleport만 다르다. 보고서에 이 동등성을 명시한다.
- Shingling은 별도 알고리즘으로 내세우지 않는다 (k-gram을 쓰지 않으므로).
- 직접 구현: PageRank 계열 엔진, MinHash, LSH banding. 검증용: networkx.

---

## 4. 데이터 나누기

- **도메인** = 3단계 도메인 (예: `www1.example.co.uk` → `example.co.uk`). 원본 SET1/SET2 분할과 같은 기준이다.
  - 이 함수로 SET1과 SET2에 **겹치는 도메인이 0개**인지 테스트한다.
  - 겹치면 해당 사례를 보고 함수를 고친 뒤 다시 확인한다. 11/18 전에 끝낸다.
- SET1 → 도메인 단위 5-fold (`splits.json`으로 고정). 4개 fold는 seed용, 1개 fold는 검증용이다.

**검증할 때**

- 모든 방법이 **seed용 fold 라벨만** 쓴다. TrustRank·PPR seed, LSH 점수, 1단계 이웃, p 전부 해당된다.
- 검증 fold 채점에서도 신뢰 도메인 사이트를 뺀다.
- 세부 값은 **5-fold 평균 PR-AUC**로 고른다. 차이가 fold 간 표준편차보다 작으면 5절의 기본값을 유지한다.
- 11/24 중간 점검도 검증 결과로만 본다.

**최종**

- SET1 전체를 seed와 라벨로 쓰고, SET2로 **한 번만** 채점한다.
- seed "전부"는 검증 때 SET1의 약 80%, 최종 때 100%다. 보고서에 기록한다.

**p (스팸 비율)** = 사용 가능한 라벨 중 스팸 / (스팸 + 정상), undecided 제외. 검증 때는 seed용 fold에서, 최종 때는 SET1에서 계산한다.

---

## 5. 기본값 표

검증에서 후보 간 차이가 작으면 이 값을 쓴다.

| parameter | 기본값 | 검증 후보 |
|---|---|---|
| damping | 0.85 | (곡선으로만 분석) |
| TrustRank seed 개수 | 전부 (이때 무작위·PageRank 상위 방식은 상관없음) | – (고르지 않음, 곡선으로만 분석) |
| TrustRank seed 방식 비교 | – | – (고르지 않음, 개수 50·200에서 분석만) |
| PPR 방향 | 링크 방향 | 링크 방향 / 역방향 |
| PPR 점수 | ① 그대로 | ① 그대로 / ② PPR/PR |
| LSH 최소 집합 크기 | 1 | 1, 2, 3, 5, 10 |
| LSH (b, r) | (16, 8) | (32, 4), (16, 8), (8, 16) |
| degree 방향 | 많을수록 스팸 | 많을수록 / 적을수록 |
| PageRank 방향 (참고용) | 높을수록 스팸 | 높을수록 / 낮을수록 |

---

## 6. 채점 규칙

**채점 대상** = SET2 라벨 − undecided − 신뢰 도메인 사이트. 알고리즘을 돌리기 전에 `eval_set.json`으로 고정한다. 최종 사이트 수와 스팸 수를 데이터 통계표에 적는다.

| 항목 | 규칙 |
|---|---|
| 메인 지표 | PR-AUC = average precision (`sklearn.metrics.average_precision_score`) |
| 보조 지표 | ROC-AUC, Precision@50, Precision@100 |
| 신뢰구간 | bootstrap 1,000회, **도메인 단위**로 재표집 |
| 방법끼리 비교 | 같은 bootstrap 표본으로 두 방법의 PR-AUC **차이**를 계산. 95% 구간이 0을 포함하면 "차이 없음". 5회 반복하는 방법(LSH, 무작위 baseline)은 **seed 42 결과**로 계산하고 5회 표준편차는 따로 적는다 |
| "잡았다"의 기준 | eval_set에서 각 방법의 **상위 100개** 안에 든 스팸 (Precision@100과 같은 K). "한 방법만 잡은 스팸 수"(H3)에 쓴다. 100번째 경계에 동률이 걸리면 **seed 42 무작위 순서로 끊어** 구체적인 host 목록을 만든다 |
| 동률 | AUC는 라이브러리 기본, Precision@K는 기댓값 방식 |
| 하위 집합 채점 | dangling 따로, LSH 커버 사이트만 → 사이트 수·스팸 수·신뢰구간 함께 표기 |
| 점수 방향 | 높을수록 스팸 (PageRank는 방향을 검증으로 정해 참고용) |

**핵심 비교 4개** (차이 신뢰구간으로 판단)

여기서 "TrustRank"는 **① 부호 반전, SET1 정상 seed 전부**로 미리 고정한다.

| 비교 | 가설 |
|---|---|
| 1. TrustRank vs PPR | H2 |
| 2. TrustRank vs LSH (+ 한 방법만 잡은 스팸 수) | H3 |
| 3. PPR vs 1단계 이웃 (**PPR에서 고른 방향과 같은 방향**) | H6 |
| 4. TrustRank(SET1 정상 seed) vs TrustRank(신뢰 도메인 seed) | H5 |

PageRank vs in-degree는 참고용으로 표에만 싣는다.

**최종표 구성** (미리 고정)

| 방법 | 들어가는 줄 |
|---|---|
| PageRank | 1줄 (참고용) |
| TrustRank | 4줄 = seed 2종(SET1 정상 전부 / 신뢰 도메인 전부 8,877개) × 점수 2종(① 부호 반전 / ② Spam Mass) |
| PPR | 방향 2 × 점수 2 중 검증으로 고른 1줄 |
| LSH | 검증으로 고른 설정 1줄 |
| Baseline | 5줄 (무작위, in-degree, out-degree, 1단계 이웃 두 방향) |

무작위가 들어간 것은 5회 평균 ± 표준편차로 적는다.

**분석 그림** (damping 곡선, seed 개수 곡선 등)

- 검증 결과와 SET2 결과를 **두 줄**로 그린다.
- seed 개수와 damping은 고르지 않는 값이므로(5절), 검증 곡선은 **SET2 곡선과 모양을 비교하는 용도**다.
- **SET2 곡선이 결론의 근거**다. SET2 곡선으로는 아무것도 고르지 않는다.
- 두 곡선 모양이 비슷하면 결과가 특정 데이터에 치우치지 않았다는 근거가 된다.
- 검증 결과로 고르는 것은 5절 표에서 "검증 후보"가 있는 값뿐이다.

---

## 7. 공통 엔진

```python
power_iteration(A, teleport, damping=0.85, dangling="teleport", tol=1e-8, max_iter=500) -> r
```

- 오차 1e-8, 최대 500회, 실제 반복 횟수 기록, 출력 합을 1로 정규화
- dangling 목록은 **입력 그래프에서 매번 새로 계산**한다 (역방향 그래프를 넣으면 dangling 집합이 달라짐)
- dangling 규칙 2가지
  - `"teleport"`: 탐지 점수용 기본 (PageRank는 균등, TrustRank는 정상 seed, PPR은 스팸 seed)
  - `"uniform"`: 나누기 점수(Spam Mass, PPR/PR)용
- 강의 예제(11강 p16, p33, p46)와 뒤집은 그래프 예제로 먼저 검증

---

## 8. 알고리즘별 규격

### 8.1 PageRank

- teleport: 전체 균등
- 최종표: 탐지 점수 1줄 (방향은 검증으로, 참고용)
- 그림: `eval_set` 사이트만 순위를 매겨 **구간 10개별 스팸 비율** + 구간마다 신뢰구간. TrustRank **① 신뢰 점수 순위**(SET1 정상 seed 전부)와 나란히 그린다 (Q1).

### 8.2 TrustRank

- teleport: 정상 seed
- seed 방식 (개수 50, 200에서 비교)
  - 무작위: seed용 fold 정상 사이트에서 무작위, 5회 (seed 42–46)
  - PageRank 상위: seed용 fold 정상 사이트 중 PageRank 상위, 1회
  - 신뢰 도메인: 8,877개에서 무작위, 5회. "전부"면 8,877개 전부 (SET1 라벨과 무관해서 별도 비교)
- 신뢰 도메인 seed에서 알려진 스팸은 뺀다. 검증 때는 seed용 fold 라벨로, 최종 때는 SET1 라벨로만 뺀다. SET2에 있는 스팸은 빼지 않는다.
- seed 개수 곡선: 10 / 50 / 100 / 200 / 전부, **무작위 방식 5회 평균 ± 표준편차**. x축은 **실제 seed 개수**(로그 스케일). 검증의 마지막 점은 5개 fold의 평균 seed 수(약 3,000개), 최종은 3,776개.
- 점수
  - ① 부호 반전 (dangling → seed)
  - ② Spam Mass = 1 − TR/PR. TR과 PR 모두 dangling `"uniform"`, 같은 그래프에서 계산한다. 상수배는 순위에 무관하므로 스케일 보정은 하지 않고, 값 자체(예: 50%)는 해석하지 않는다.

### 8.3 PPR

- teleport: SET1 스팸 seed
- 방향: 링크 방향 + 역방향 (역방향 = Anti-TrustRank 방식)
  - 링크 방향: 나가는 링크 없는 스팸 seed(SET1 스팸 222개 중 125개, 56.3%)는 퍼뜨리지 못한다
  - 역방향: 나가는 링크 없는 스팸은 점수를 받지 못한다
- 점수
  - ① PPR 그대로 (dangling → seed)
  - ② PPR / PR: 분자·분모 모두 **같은 그래프**, dangling `"uniform"`. 역방향이면 분모도 **뒤집은 그래프의 PageRank**.
- 실제로 퍼뜨린 seed 수를 기록한다.

### 8.4 MinHash / LSH

- 집합: 사이트마다 "나에게 링크한 사이트들"
- 순서
  1. 최소 집합 크기 필터 (1, 2, 3, 5, 10)
  2. 집합이 **완전히 같은** 사이트끼리 묶음. Jaccard = 1이라 재확인 없이 묶음 단위로 센다. 최대 묶음 1,884개도 여기서 처리된다.
  3. **묶음 대표끼리만** MinHash(signature 128) + LSH banding
     - 버킷 크기(대표 수 기준) 1,000 이하: 쌍을 만들고 실제 Jaccard ≥ (1/b)^(1/r)만 남김
     - 1,000 초과: 건너뛰고, 건너뛴 버킷 수와 영향받은 사이트 수를 기록 (보고서에 숫자로)
- 재확인 기준: (32,4) ≈ 0.42, (16,8) ≈ 0.71, (8,16) ≈ 0.88 → (b, r) 선택이 곧 유사도 기준 선택이다 (보고서에 기록).
- **이웃** = 자기 묶음의 다른 사이트 + LSH로 연결된 다른 묶음의 사이트 전부. 모든 band에서 나온 이웃의 합집합(중복 제거). 이웃의 이웃으로 이어 붙이지 않는다.
- **점수** = (s + 3·p) / (n + 3). n = 이웃 중 라벨 수, s = 그중 스팸 수. 이웃이 없으면 p. 3은 "라벨 3개 정도는 모여야 전체 비율에서 벗어나게" 하려는 고정값이다.
- 허브 제거는 본 실험에서 하지 않는다. 빼면 신호를 잃을 수 있어 추가 실험(나가는 링크 상위 1% 제거)으로 확인한다.
- 11/20까지 실제 데이터로 **후보 쌍 수부터** 센다.
- **해시 seed**: MinHash 해시 함수는 seed 42로 고정한다. 최종 설정에서만 해시 seed 42–46으로 5회 돌려 평균 ± 표준편차를 최종표에 적는다.
- **커버 = 라벨된 이웃이 1개 이상인 사이트.** 커버리지(%)와 하위 집합 재채점 모두 이 정의를 쓴다. 하위 집합 재채점은 **해시 seed 42 기준** 커버 집합으로 고정한다.
- 평가: 전체 채점 + 커버리지(%) + 커버 사이트만 모아 전 방법 재채점. 근사 품질은 실제 Jaccard 대비 recall과 강의 S-curve(4강 p51–57) 비교.

### 8.5 Baseline (5개)

- 무작위
- in-degree, out-degree (방향은 검증으로)
- 1단계 이웃 (LSH와 같은 보정식)
  - 나에게 링크한 사이트 중 스팸 비율 (링크 방향 PPR과 짝)
  - 내가 링크한 사이트 중 스팸 비율 (역방향 PPR과 짝)

---

## 9. 추가 분석 (최종표 아님)

- 가정 확인 EDA: 정상→스팸 링크 비율, 스팸→스팸 링크 비율 (TrustRank·PPR 가정 확인)
- damping 곡선 0.7 – 0.95
- seed 개수 곡선
- 방법끼리 비교: 한 방법만 잡은 스팸 수, 상위 K 겹침
- 단순 결합: TrustRank·PPR·LSH 순위 평균 (검증으로 먼저 확인, 알고리즘 개수엔 넣지 않음)
- dangling / 비dangling 따로 채점
- LSH 허브 제거 변형
- error analysis: 모든 방법이 놓친 스팸, 잘못 잡은 정상

---

## 10. 파이프라인과 파일

```
[데이터 받기]   scripts/download_data.py ✅        → data/raw/
[전처리]       src/load_graph.py                  → data/processed/ (그래프, 라벨)
[분할]         src/splits.py                      → splits.json, eval_set.json (깃에 올림)
[seed 선택]    src/seeds.py
[알고리즘]     src/engine.py  pagerank.py  trustrank.py  ppr.py  lsh.py  baselines.py
                                                   → results/scores/방법_설정.npy + .json
[채점]         src/evaluate.py                    → results/tables/*.csv (깃에 올림)
[그림]         src/figures.py                     → results/figures/*.png (깃에 올림)
[전체 실행]    experiments/run_main.py  (점수 → 채점 → 표 → 그림까지 한 번에)
               experiments/run_sensitivity.py (분석 곡선)
[탐색]         notebooks/  (EDA·확인용, 보고서 그림은 만들지 않음)
```

- 각 단계는 앞 단계 파일을 읽고 결과를 파일로 저장한다. 파일이 있으면 다시 계산하지 않는다 (`--force`로 재계산).
  - 불러올 때 `src/` 폴더의 해시(`git rev-parse HEAD:src`)가 저장 당시와 다르면 경고한다. README나 문서만 고쳤을 때는 경고하지 않는다.
  - 저장할 때 `git status --porcelain src`도 확인한다. 출력이 있으면(커밋 안 한 수정이 있으면) `.json`에 `"dirty": true`를 기록하고 경고한다.
  - 최종 채점과 12/4 재현 테스트는 **dirty가 아닐 때만** 돌린다.
  - 최종 채점 전과 12/4 재현 테스트에서는 반드시 `--force`로 전부 다시 계산한다.
- `splits.json`, `eval_set.json`에는 **hostid와 fold 번호만** 저장한다. 라벨(spam/nonspam)은 넣지 않고 실행할 때 원본에서 읽는다 (라벨 재배포 방지).
- 점수 파일 이름에 설정을 넣는다 (예: `trustrank_seed-random_k50_d0.85_s42.npy`).
- 설정 `.json`에는 설정값, 날짜, 실행 시간, 반복 횟수, **git commit hash**를 넣는다.
- 랜덤 seed 42. 반복 실험은 42–46.
- 깃에 올리는 것: 코드, `splits.json`, `eval_set.json`, 표, 그림. 원본 데이터와 중간 점수 파일은 올리지 않는다.

---

## 11. 보고서 계획 (8–10쪽)

| 섹션 | 분량 | 담당 | 내용 |
|---|---|---|---|
| 제목·팀 정보 | – | 같이 | 팀 번호, 이름, 학번, GitHub URL, 전체 commit hash |
| Abstract | 0.3쪽 | 김현중 | 그림·인용 없이 정량 결과 포함 |
| 1. Introduction | 1쪽 | 김현중 | 웹 스팸, link farm 동기, 연구 질문, 선행연구 |
| 2. Data | 1.5쪽 | 김현중 | 2절 내용, 통계표, degree 분포, 한계 |
| 3. Methods | 2쪽 | 박병언 | 알고리즘별 수식·가정·설정·강의 근거, 동등성, 구현 |
| 4. Experimental design | 1쪽 | 박병언 | 분할, 채점 규칙, 기본값, 실행 환경 |
| 5. Results | 2.5쪽 | 강민수 | 최종표, 그림, 해석 |
| 6. Conclusion | 0.5쪽 | 같이 | 질문별 답, 한계, 후속 과제, 팀원별 기여 |
| References | 0.5쪽 | 강민수 | IEEE 번호 형식 |
| AI-use disclosure | 0.3쪽 | 같이 | `docs/AI_USAGE.md` 정리 |

**본문 그림 우선순위** (Results 2.5쪽)

1. 최종 비교표
2. PageRank vs TrustRank 구간별 스팸 비율
3. PR 곡선
4. seed 개수 곡선 (검증·SET2 두 줄)

여유가 있으면 방법끼리 겹침, dangling 따로 채점. 나머지(damping 곡선, LSH 변형, 결합, error analysis 사례)는 부록이나 GitHub에 둔다.

**보고서 한계에 쓸 것**: 데이터 한계(2절), 링크 수 가중치 제외, 핵심 비교 4개에 대한 다중 비교 보정 없음.

**형식**: 10pt, 줄간격 1.5, 영문 Times New Roman, 한글 글꼴 통일, IEEE 번호 인용

---

## 12. GitHub 제출물 (10점)

- [ ] README: 개요, 팀원·학번, 저장소 구조, 환경, 데이터 받는 법, 전처리, 알고리즘, 실행 명령, 평가, 표·그림 재현, 외부 코드 출처, AI 사용
- [ ] `requirements.txt` 버전 고정, Python 3.10 이상
- [ ] `docs/AI_USAGE.md`: AI를 쓸 때마다 각자 한 줄
- [ ] `download_data.py` → `run_main.py` → `run_sensitivity.py`로 모든 표·그림 재현 (보고서 그림은 전부 `src/figures.py`가 생성, 노트북에 의존하지 않음)
- [ ] 팀 작성 코드와 외부 코드 구분 표시
- [ ] 보고서·Blackboard의 URL과 commit hash 일치

---

## 13. 역할

| | 김현중 | 박병언 (공통 엔진) | 강민수 |
|---|---|---|---|
| 코드 | `load_graph.py`, `lsh.py`, `baselines.py`, `figures.py` | `engine.py`, `pagerank.py`, `trustrank.py`, `seeds.py` | 도메인 함수·`splits.py`, `eval_set.json`, `evaluate.py`, `ppr.py` |
| 실험·분석 | 기본 EDA(통계표, degree 분포, 판정자 간 일치도), 가정 확인 EDA, LSH 후보 쌍 수, 방법끼리 비교 | Spam Mass, seed 방식·개수 곡선, damping 곡선, 단순 결합, dangling 따로 채점, 실행 환경 기록 | 채점(동률, 도메인 bootstrap, 차이 신뢰구간), 최종 비교표, error analysis |
| 보고서 | Abstract, Introduction, Data | Methods, Experimental design | Results, References |
| 발표 | 문제, 데이터 | 방법 | 결과, 한계 |
| 그 외 | README 관리, `figures.py` 파일 관리 | | |

`figures.py`의 그림 함수는 **그 결과를 만든 사람이 작성**한다 (최종표·PR 곡선은 강민수, seed·damping 곡선과 구간별 그림은 박병언, EDA 그림은 김현중). 김현중은 파일 구조와 `run_main.py` 연결만 맡는다.

같이: Conclusion, AI 사용 공개, 팀원별 기여, `config.py`, `run_main.py`, 발표 영상 편집

---

## 14. 일정

| 날짜 | 김현중 | 박병언 | 강민수 |
|---|---|---|---|
| ~11/18 | `load_graph` | 공통 엔진 | 도메인 함수(겹침 0 테스트), `splits`, `eval_set`, 채점 뼈대 |
| ~11/20 | LSH 구현, 강의 예제 검증, 후보 쌍 수 세기, 기본 EDA | PageRank·TrustRank 강의 예제 검증 | PPR 강의 예제 검증, baseline을 채점 코드에 연결 (`baselines.py` 수정이 필요하면 김현중에게 먼저 공유) |
| **11/24 (화)** | **중간 점검: 검증 결과로 세부 값 확정** | | |
| ~11/30 | 방법끼리 비교, 가정 확인 EDA | seed·damping 곡선, Spam Mass, 결합 | SET2 최종 채점, 최종표, error analysis |
| ~12/3 | 보고서 Intro·Data, README | 보고서 Methods·실험 설계 | 보고서 Results·References |
| 12/4 (금) | 새 폴더에 clone해서 `--force`로 처음부터 재현 테스트, 교차 검토, 발표 녹화 (5분 이하·30MB 이하·재생 확인) | | |
| 12/5 (토) | commit hash 확정 → 보고서·발표·URL 제출 (23:55) | | |

---

## 15. 지금 할 일

- [ ] 분반(발표 언어), 팀 번호 확인
- [ ] 다운로드 스크립트 수정본 push (Python 3.10 이상에서 동작하도록 대체 처리 추가, README 표기도 "3.10 이상"으로 수정)
- [ ] 교수님께 알고리즘 구성 확인 메일 (아래)
- [ ] 저장소를 private으로 바꾸면 `leesael` collaborator 추가

**교수님 메일 초안**

> 제목: [SCE331 팀 프로젝트] 알고리즘 구성 인정 여부 문의 (팀 번호 __)
>
> 교수님 안녕하세요, SCE331 데이터마이닝 __팀 김현중입니다.
> 팀 프로젝트 알고리즘 구성이 "수업 알고리즘 3개 이상" 요건을 충족하는지 여쭙고자 연락드립니다.
>
> 주제: WEBSPAM-UK2007 host 그래프를 이용한 링크 기반 웹 스팸 탐지
>
> 1. PageRank (11강)
> 2. TrustRank: SET1 정상 host 또는 신뢰 도메인을 seed로 사용 (11강)
> 3. Personalized PageRank: Topic Ideas Link Analysis C 구성을 바탕으로 하되, 주제 seed 대신 SET1 스팸 host를 seed로 사용해 의심도를 전파하도록 변형. 링크 방향과 역방향 모두 실험하며 역방향은 Anti-TrustRank 방식 (11강)
> 4. MinHash + LSH: 들어오는 링크 집합이 유사한 host 묶음 탐지 (4강, Algorithms by Topic 목록 2. Similar-Item Search)
>
> TrustRank와 Personalized PageRank가 같은 수식 계열이라 별도 알고리즘으로 인정되는지, 그리고 위 구성 전체가 요건을 충족하는지 확인 부탁드립니다.
> 만약 TrustRank와 PPR을 하나로 보신다면, PageRank·TrustRank·MinHash/LSH 3개로 요건이 충족되는지도 함께 여쭙습니다.
>
> 감사합니다. 김현중 드림

---

## 16. 깃 규칙

- 각자 **자기 파일만** 수정한다. 같이 쓰는 파일(README, `config.py`, `run_main.py`)은 수정 전에 단톡방에 말한다.
- 다른 사람 파일을 고쳐야 하면 담당자에게 먼저 말하고 고친다.
- 작업 시작 전에 항상 `git pull`
- 커밋 메시지는 한글로, 무엇을 했는지 한 줄로
- AI를 쓰면 `docs/AI_USAGE.md`에 한 줄 추가

---

## 17. 제출 체크리스트 (마감 12/5 23:55)

- [ ] `Team#_Report.pdf` → Blackboard (8–10쪽, 필수 섹션, commit hash)
- [ ] GitHub URL → Blackboard와 보고서
- [ ] `Team#_Presentation.mp4` → 지정 Google Drive (5분 이하, 30MB 이하)
- [ ] 동료평가: 각자 배정된 발표 7–10개, 12/24 23:55까지
- [ ] 지각: 하루 10% 감점, 12/8 이후 0점. 보고서나 발표 중 하나라도 미제출 시 C 미만

---

## 18. 인용 후보 (서지 정보는 원문에서 확인 후 사용)

- WEBSPAM-UK2007 데이터셋 (Yahoo! Research Barcelona, Web Spam Challenge 2008), Castillo 등의 web spam 참조 컬렉션 논문
- Boldi & Vigna, WebGraph framework (그래프 수집·압축)
- Page, Brin, Motwani, Winograd (1999), PageRank
- Haveliwala (2002), Topic-sensitive PageRank
- Gyöngyi, Garcia-Molina, Pedersen (2004), TrustRank
- Gyöngyi, Berkhin, Garcia-Molina, Pedersen (2006), Spam Mass
- Krishnan & Raj (2006), Anti-TrustRank
- Broder (1997), MinHash
- Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets* (교재)
- SCE331 강의 자료 4강(LSH), 11강(Link Analysis)
- 소프트웨어: numpy, scipy, pandas, scikit-learn, networkx, matplotlib (버전 명시)

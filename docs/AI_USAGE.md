# AI 사용 기록

보고서 "AI-use disclosure"와 README의 AI 사용 공개에 그대로 쓰기 위한 기록입니다.
AI 도움을 받을 때마다 아래 표에 한 줄씩 추가해 주세요.

- 모델 이름은 도구 화면에서 확인한 것만 적고, 확인할 수 없으면 "확인 불가"로 적습니다.
- 검증 방법은 구체적으로 적습니다. (예: 직접 실행해 결과 확인, 원 논문과 대조, 손으로 계산한 예제와 비교)

| 날짜 | 사람 | 도구 / 모델 | 도움받은 작업 | 영향받은 파일·섹션 | 검증·수정 방법 |
|---|---|---|---|---|---|
| 2026-11-09 ~ 11-16 | 김현중 | Claude Code (Anthropic) / Claude Sonnet 5.5, Claude Opus 5.5 | 과제 안내문 정리, 주제 후보 비교, 프로젝트 구조 초안 작성 | `docs/STRUCTURE.md` | 과제 안내 PDF와 강의 슬라이드(4강, 11강)를 직접 대조, 팀 회의에서 검토 예정 |
| 2026-11-16 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | 데이터 다운로드 스크립트, `.gitignore`, README 데이터 섹션 작성, 데이터 기본 통계 확인 | `scripts/download_data.py`, `.gitignore`, `README.md` | 스크립트를 직접 실행해 파일 크기와 라벨 개수가 배포처 README의 통계와 일치하는지 확인 |
| 2026-11-16 | 강민수 | Claude (Anthropic) / Claude Opus 5.5 | `splits.py`, `evaluate.py` 및 테스트 초안 작성 | `src/splits.py`, `src/evaluate.py`, `tests/` | 합성 데이터 pytest 12개 통과 확인, 실제 데이터 겹침 0개 확인은 load_graph 완성 후 |
| 2026-11-17 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | `load_graph.py`, `config.py`, 테스트, `requirements.txt` 작성, PR #1·#2 리뷰 보조 | `src/load_graph.py`, `src/config.py`, `tests/test_load_graph.py`, `requirements.txt`, `README.md` | pytest 5개(합성 4 + 실제 데이터 1) 통과, 실제 데이터 통계가 11-16 실측값과 일치, PR #1·#2 코드와 합쳐 테스트 17개 통과 및 `python -m src.splits` 실행 확인 |
| 2026-11-19 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | PR #1·#2 머지, PR #3 브랜치의 `AI_USAGE.md` 병합 충돌 해결 (두 줄 모두 유지) | `docs/AI_USAGE.md` | 병합 후 전체 pytest 실행으로 확인 |
| 2026-11-19 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | MinHash/LSH 구현 (같은 집합 묶기, banding, Jaccard 재확인, 보정 점수), 테스트, 설정별 후보 쌍 수 측정 | `src/lsh.py`, `tests/test_lsh.py`, `results/tables/lsh_candidate_counts.csv`, `README.md` | 합성 데이터 pytest 6개(손계산 점수·Jaccard·문턱값 대조) 통과, 실제 데이터 묶음 수·공유 host 수가 11-16 실측값과 일치 |
| 2026-11-22 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | baseline 5개 구현(무작위, degree 두 방향, 1단계 이웃 두 방향), 테스트, SET1 검증 fold 미리 채점 | `src/baselines.py`, `tests/test_baselines.py` | 손계산 예제로 pytest 3개 통과, 무작위 점수를 seed 20개로 돌려 ROC-AUC 평균 0.510 확인 (채점 코드 이상 없음) |
| 2026-11-24 | 강민수 | Claude (Anthropic) / Claude Opus 5.5 | `ppr.py`, 검증 채점 함수(`cross_validate`, `select_setting`)와 테스트 초안 작성 | `src/ppr.py`, `src/evaluate.py`, `tests/test_ppr.py`, `tests/test_cross_validate.py`, `tests/conftest.py`, `tests/_ref_engine.py` | 2노드 손계산, 선형방정식 닫힌 해와 비교(오차 1e-7 이하), 뒤집은 그래프 예제로 확인 |
| 2026-11-25 | 박병언 | Claude Code (Anthropic) / Claude Opus 5.5 | 공통 엔진, PageRank, 엔진 테스트 초안 작성 | `src/engine.py`, `src/pagerank.py`, `tests/test_engine.py` | 11강 p16, p18, p33, p46, p28 예제 값과 대조, 닫힌 해와 networkx 비교, feat/ppr 테스트 8개를 새 엔진으로 실행해 직접 실행 결과와 비교 검증 후 통과 확인 |
| 2026-11-16 ~ 11-25 | 박병언 | Claude (Anthropic) / Claude Opus 5.5 | 확정안 검토와 수정 사항 제안, 공통 엔진과 TrustRank 설계 검토, 11강 예제 기대값 계산, Claude Code 작업 지시문 작성 | `docs/STRUCTURE.md`, `src/engine.py`, `src/pagerank.py`, `src/seeds.py`, `src/trustrank.py`, 테스트 | 예제 기대값은 엔진 테스트(닫힌 해, networkx)로 교차 확인, 슬라이드 표기 차이는 슬라이드 원문과 대조 |
| 2026-11-25 | 박병언 | Claude Code (Anthropic) / Claude Opus 5.5 | seed 선택, TrustRank, Spam Mass, 테스트 초안 작성 | src/seeds.py, src/trustrank.py, tests/test_trustrank.py | 11강 p46 예제 값과 대조, link farm 합성 그래프로 Spam Mass 확인, 강의 r^+와 닫힌 해 비교, cross-validate 브랜치 evaluate.py와 합성 데이터로 연결 확인, 전체 pytest 통과 |
| 2026-11-25 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | EDA(`eda.py`), 보고서 그림 틀(`figures.py`), LSH 구조·점수 분리와 검증 설정 선택·S-curve 분석, 테스트 | `src/eda.py`, `src/figures.py`, `src/lsh.py`, `tests/test_eda.py`, `tests/test_lsh.py`, `results/`, `README.md` | 손계산 예제로 pytest 7개 추가(전체 43개 통과). 판정자 일치도는 쌍 합산 방식이 판정자 많은 host를 과대 반영해 host 평균(Fleiss) 방식으로 수정. S-curve는 LSH가 찾은 쌍을 표본에 섞어 비율이 부풀려진 것을 발견해 해시와 무관한 무작위 쌍만 쓰도록 수정, 수정 후 이론값과 평균 차이 0.01 이하 확인 |
| 2026-11-25 | 강민수 | Claude (Anthropic) / Claude Opus 5.5 | PPR 5-fold 검증 스크립트 작성 | `experiments/cv_ppr.py` | 실제 데이터 형식과 같은 합성 데이터로 처음부터 끝까지 실행 확인, 실제 데이터 결과는 맥에서 직접 실행 |
| 2026-11-25 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | `run_main.py` 틀(check / --final 모드, 최종 설정, 비교표·핵심 비교·구간별·잡은 스팸·하위 집합 표), 점수 저장 `results_io.py`, PageRank 방향 검증, 테스트 | `experiments/run_main.py`, `src/results_io.py`, `tests/test_run_main.py`, `README.md`, `.gitignore` | 합성 그래프로 전체 흐름 pytest 4개 통과(전체 129개), 실제 데이터 check 모드(SET1 fold 0, SET2 미사용) 3분 실행 확인, out-degree 방향은 PR-AUC 차이가 표준편차보다 작아 규칙대로 기본값 +1 유지 |
| 2026-11-25 | 강민수 | Claude (Anthropic) / Claude Opus 5.5 | 최종 비교표·PR 곡선 그림 함수와 run_main 연결 | `src/figures.py`, `experiments/run_main.py`, `tests/test_figures_results.py` | 합성 점수로 그림을 그려 직접 확인 (라벨 겹침 수정), 실제 데이터 check 모드로 확인 |
| 2026-11-26 | 강민수 | Claude (Anthropic) / Claude Opus 5.5 | error analysis 함수·실행 스크립트와 테스트 작성 | `src/error_analysis.py`, `experiments/error_analysis.py`, `tests/test_error_analysis.py` | 손으로 만든 예제로 그룹 분류 확인, 실제 데이터 check 모드로 실행해 표 확인 |
| 2026-11-25 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | `run_main.py --analysis`(SET1 분석 단계 연결)와 잡은 스팸 겹침 그림(H3, #15 그림과 같은 방식으로 저장), 테스트, README 재현 표 | `experiments/run_main.py`, `src/figures.py`, `tests/test_run_main.py`, `README.md` | pytest 1개 추가(전체 137개 통과), 실제 데이터로 --analysis 전체 실행(약 7분) 후 다시 만든 분석 표가 커밋된 표와 실행 시간 칸 외에 완전히 같은 것 확인 |

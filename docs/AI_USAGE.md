# AI 사용 기록

보고서 "AI-use disclosure"와 README의 AI 사용 공개에 그대로 쓰기 위한 기록입니다.
AI 도움을 받을 때마다 아래 표에 한 줄씩 추가해 주세요.

- 모델 이름은 도구 화면에서 확인한 것만 적고, 확인할 수 없으면 "확인 불가"로 적습니다.
- 검증 방법은 구체적으로 적습니다. (예: 직접 실행해 결과 확인, 원 논문과 대조, 손으로 계산한 예제와 비교)

| 날짜 | 사람 | 도구 / 모델 | 도움받은 작업 | 영향받은 파일·섹션 | 검증·수정 방법 |
|---|---|---|---|---|---|
| 2026-11-09 ~ 11-16 | 김현중 | Claude Code (Anthropic) / Claude Sonnet 5.5, Claude Opus 5.5 | 과제 안내문 정리, 주제 후보 비교, 프로젝트 구조 초안 작성 | `docs/STRUCTURE.md` | 과제 안내 PDF와 강의 슬라이드(4강, 11강)를 직접 대조, 팀 회의에서 검토 예정 |
| 2026-11-16 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | 데이터 다운로드 스크립트, `.gitignore`, README 데이터 섹션 작성, 데이터 기본 통계 확인 | `scripts/download_data.py`, `.gitignore`, `README.md` | 스크립트를 직접 실행해 파일 크기와 라벨 개수가 배포처 README의 통계와 일치하는지 확인 |
| 2026-11-17 | 김현중 | Claude Code (Anthropic) / Claude Opus 5.5 | `load_graph.py`, `config.py`, 테스트, `requirements.txt` 작성, PR #1·#2 리뷰 보조 | `src/load_graph.py`, `src/config.py`, `tests/test_load_graph.py`, `requirements.txt`, `README.md` | pytest 5개(합성 4 + 실제 데이터 1) 통과, 실제 데이터 통계가 11-16 실측값과 일치, PR #1·#2 코드와 합쳐 테스트 17개 통과 및 `python -m src.splits` 실행 확인 |

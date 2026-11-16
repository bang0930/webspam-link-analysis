# webspam-link-analysis
SCE331 Data Mining Team Project

## Data

WEBSPAM-UK2007 (Yahoo! Research Barcelona, Web Spam Challenge 2008)
- 배포처: https://chato.cl/webspam/datasets/uk2007/ (접근일 2026-11-16)
- 라벨 라이선스: CC BY-NC-SA 3.0 — 원본 데이터는 저장소에 포함하지 않음

### 받는 방법

```bash
python scripts/download_data.py
```

Python 3.8+ 표준 라이브러리만 사용합니다. `data/raw/` 아래에 저장되며, 이미 받은 파일은 건너뜁니다.

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

# 고정 10개 도구 단일 선택(`decision-export-v2` / `hermes-ten-tools-v1`) 독립 감사 보고서

## 1. 감사 대상 및 범위

루트 워크스페이스(`/Volumes/Dove-Nest-SSD/projects/Jery`)에 생성된 고정 10개 도구 진단 과제 계약·생성기·데이터셋·파생 export를 읽기 전용으로 독립 검증했다.

- **계약 및 설정**: `docs/contracts/decision-export-v2.md`, `configs/training/laya-ten-tools-catalog-20261020.json` (`sha256: b4648cc19a90b301fcca32b55ae5028361b8c04461e76d155f0f542593267c59`, `seed=3407`)
- **생성기**: `scripts/build_laya_ten_tools.py` (`sha256: 65d855cf10d920232fa87ff2f54289f5ab6e7b31a5308ab50ce317cf6e25ba2d`)
- **파생 데이터셋 메타데이터**: `datasets/hermes-ten-tools-v1/{tools.json,selection.json,manifest.json}` (`catalog_version=hermes-ten-tools-b4648cc19a90`)
- **파생 학습·평가 export (Git 무시 경로)**: `outputs/laya-ten-tools-20261020/{memorization,full}` (`export_version=decision-export-v2`, `export_scope=fixed_ten_tools_select_only`, `evaluation_task=ten_tool_choice`)
- **로컬 무시 증빙 JSON**: `outputs/laya-ten-tools-20261020-audit/audit_summary.json`

## 2. 원본 보존 및 SHA-256 무결성 검증

원본 `datasets/mcp-routing-v2`는 전혀 수정되지 않았으며, 재계산한 해시가 원본 `manifest.json` 및 모든 파생 `manifest.json`의 `source_sha256`과 정확히 일치한다.

| 파일 | 재계산 SHA-256 | 일치 여부 |
|---|---|---|
| `datasets/mcp-routing-v2/tools.json` | `d8a97f14b63fe9321f65942c6947d0c7d2370542b7cc9cb22f0683a42ca9a48b` | PASS |
| `datasets/mcp-routing-v2/train.jsonl` | `5eb09bc5d05668f82fe669c36b44ec2bd4d758e46cdec9a4b8c73edd5ddb9bd9` | PASS |
| `datasets/mcp-routing-v2/validation.jsonl` | `6840efb974df1ddded18b6b27b765c1cfa43d182238eca2b6fae5c0c63d5ddf6` | PASS |
| `datasets/mcp-routing-v2/test.jsonl` | `7a1fc5600a24b265f6864914a567e03ba0bd18f5b53f896f692465c09df832c4` | PASS |
| `datasets/hermes-ten-tools-v1/tools.json` | `8522bfa1826d1d8a73020d26cf9428d956d458e28b4969d5eb177827e09d5aed` | PASS |
| `datasets/hermes-ten-tools-v1/selection.json` | `bdc19c7082969461f1faf0f3ab62e6d582d7eb84cc376b9bc86eeb95dcbfee4e` | PASS |
| `datasets/hermes-ten-tools-v1/manifest.json` | `fe1be4908f0648c45798b0c5c14e672b364aed51b1fc4d9dfe0c865decdb3f51` | PASS |

임시 디렉터리에서 `build_laya_ten_tools.build()`를 독립 재실행한 결과, 위 3개 파일과 두 profile의 모든 JSONL·manifest 해시가 바이트 단위로 100% 재현됨을 확인했다.

## 3. 고정 10개 후보 카탈로그 및 한국어 선택지 설명 검증

- **10개 고정 도구 구성**: Obsidian(`obsidian_vault`) 5종(`vault`, `edit`, `graph`, `dataview`, `bases`)과 Swallow(`swallow_trader`) 5종(`swallow_get_stock_price`, `swallow_get_price_history`, `swallow_get_portfolio`, `swallow_order`, `swallow_backtest`) 총 10개가 `mcp-routing-v2/tools.json` 원본 명세와 바이트 단위로 동일하게 `datasets/hermes-ten-tools-v1/tools.json`에 보존되었다.
- **한국어 선택지 설명(`criteria`)**: 기존 `mcp-routing-v2/tools.json`에서는 Obsidian 도구가 영문 설명, Swallow 도구가 국문 설명이어서 앞부분 절단 시 서버 간 한·영 언어 단서(shortcut)가 존재했으나, `laya-ten-tools-catalog-20261020.json`은 10개 도구 모두에 고유한 한국어 기능 설명을 부여해 설명 언어를 통일했다. 이것만으로 모델의 편향이 해소됐다고 단정할 수는 없다.

## 4. 입력 중복 제거, 클래스 균형, 시나리오 분리 및 선택지 위치 분포

- **원본 global/subset 변형 중복 제거 검증**: 원본 `mcp-routing-v2`는 각 시나리오 표현마다 전체 후보(`-g`, 96개)와 부분 후보(`-s`) 쌍이 동일한 `(request, context)`를 공유한다. 고정 10개 후보 과제에서는 두 행의 모델 입력(`state.request`, `state.context`)이 완전히 같아지므로, 생성기가 ID 정렬 후 `(request, context)` 기준 중복을 제거해 적격 사례가 `train` 138→69행(도구당 6~13행), `validation` 46→23행(도구당 2~4행), `test` 46→23행(도구당 2~4행)으로 축소된다. 분할 내부 및 분할 간 동일 `(request, context)`의 정답 도구 충돌은 `0`건이며, 사전순 정렬(`-g` < `-s`)에 의해 선택된 모든 원본 행은 `-g`(`len(source_candidate_tools) == 96`) 계보를 갖는다.
- **Profile별 분할 크기·클래스 균형·시나리오 그룹 분리**:

| Profile / Split | 행 수 | 도구당 행 수 (10개 균등) | 고유 시나리오 그룹 수 | 분할 간 시나리오·입력 교집합 | 다중 원본 대안 행 수 | 파일 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `memorization / train` | 20 | 2 | 20 | 0 | 2 | `31563cf1c8d8fa75fe342e0f0c83484260395090b5d2572af0e2c1aec7d10ef5` |
| `memorization / validation` | 20 | 2 | 12 | 0 | 1 | `9d70f66d63e25cbf963dcec2f225e18d179609348558eafcc3c325b0d38411c9` |
| `memorization / test` | 20 | 2 | 12 | 0 | 1 | `158a0baa6125758bb7fbddf7a98efa35e579e76edb42012ac3b1142a79092e7f` |
| `full / train` | 60 | 6 | 35 | 0 | 2 | `8dbe90fc2541b572bcd6f428f5d19533764c9216ce20c56efb62b8fa6aa288c9` |
| `full / validation` | 20 | 2 | 12 | 0 | 1 | `9d70f66d63e25cbf963dcec2f225e18d179609348558eafcc3c325b0d38411c9` |
| `full / test` | 20 | 2 | 12 | 0 | 1 | `158a0baa6125758bb7fbddf7a98efa35e579e76edb42012ac3b1142a79092e7f` |

- **Profile 간 일관성**: `memorization`과 `full`의 `validation.jsonl`, `test.jsonl`은 바이트 단위로 완전히 동일하며, `memorization/train` 20행은 `full/train` 60행의 엄밀한 부분집합(동일 `source_id`에 대해 선택지 배치와 정답 alias까지 완전히 일치)이다.
- **선택지 위치(`T0`~`T9`) 정답 비참조 셔플 및 one-hot 타깃**: 모든 행은 `window_index=0`, `window_count=1` 단일 창이며 `no_tool`, `clarify`, `window_none` 없이 `T0`~`T9` 10개 선택지만 포함한다. 후보 순서는 정답을 참조하지 않고 `random.Random(f"{seed}:{row['id']}")`로 섞이며, `gold.route_in_window`는 `{label: 1.0}` one-hot 분포다. 정답 위치 분포는 `full/train`(`T0`:6, `T1`:5, `T2`:8, `T3`:8, `T4`:6, `T5`:8, `T6`:7, `T7`:6, `T8`:3, `T9`:3), `validation`(`T0`:2, `T1`:2, `T2`:2, `T3`:3, `T4`:0, `T5`:1, `T6`:3, `T7`:2, `T8`:3, `T9`:2), `test`(`T0`:2, `T1`:1, `T2`:0, `T3`:1, `T4`:4, `T5`:2, `T6`:2, `T7`:3, `T8`:2, `T9`:3)다. 이 분포는 셔플 결과를 기록한 것이며 통계적 무편향의 증명은 아니다.

## 5. 입력-메타데이터 경계 및 계보 보존 소견

- **입력 경계 및 누수 차단**: 모든 행의 `state`는 `request`와 `context` 두 키만 가지며 원본 `mcp-routing-v2`의 문자열과 정확히 일치한다. `state` 및 `questions` 내부에 도구 ID(`mcp__*`), `source_id`, `scenario_id`, `source_split`, `review_status`, `execution_verified` 등 메타데이터·정답 문자열 누수는 `0`건이다.
- **원본 다중 허용 대안 계보 보존 확인**: `mcp__swallow_trader__swallow_get_stock_price`가 정답인 일부 사례(`train` 2행, `validation` 1행, `test` 1행)는 원본 `source_acceptable_choices`에 10개 카탈로그 밖 도구인 `mcp__swallow_trader__swallow_check_stock`이 대안으로 함께 존재한다. 10개 고정 후보와의 교집합은 유일하게 `swallow_get_stock_price` 하나이므로 계약 조건을 충족하며, `metadata.acceptable_choices`에는 해당 도구의 action 대안만, `metadata.source_acceptable_choices`에는 원본 대안 전체가 손실 없이 보존되어 있다.
- **합성 미검토 상태 보존**: 전 분할 모든 행에서 `review_status="needs_human_review"`, `execution_verified=false`가 그대로 유지되었고, 숨겨진 test 튜닝이나 후처리 흔적은 없다.
